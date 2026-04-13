from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from data_gen.canonicalize import structural_string
from data_gen.scoped_actions import apply_scoped_action_unchecked, parse_scoped_action_id


@dataclass(frozen=True)
class CandidateFrontierLabels:
    action_id: str
    candidate_valid: bool
    next_expr: str | None
    reaches_target_site_within_horizon: bool
    reaches_target_action_within_horizon: bool
    reaches_goal_within_horizon: bool
    reduces_distance_to_goal: bool
    nodes_expanded: int


def parse_action_order(action_order: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in action_order.split("->") if part.strip())


def infer_hidden_cancel_action(action_order: str) -> str:
    actions = parse_action_order(action_order)
    cancel_actions = []
    for action_id in actions:
        site_id, op_name = parse_scoped_action_id(action_id)
        if op_name == "cancel" and site_id.startswith("expr@") and site_id != "expr@root":
            cancel_actions.append(action_id)
    if not cancel_actions:
        raise ValueError(f"No non-root expr cancel action in guided order: {action_order}")
    return cancel_actions[-1]


def apply_action_id(expression: str, action_id: str) -> str | None:
    site_id, op_name = parse_scoped_action_id(action_id)
    next_expr = apply_scoped_action_unchecked(expression, site_id, op_name)
    if next_expr is None:
        return None
    next_expr_str = str(next_expr)
    if structural_string(next_expr_str) == structural_string(expression):
        return None
    return next_expr_str


def valid_candidate_actions(expression: str, action_vocab: list[str]) -> list[tuple[str, str]]:
    valid: list[tuple[str, str]] = []
    for action_id in action_vocab:
        next_expr = apply_action_id(expression, action_id)
        if next_expr is not None:
            valid.append((action_id, next_expr))
    return valid


def _target_site_is_accessible(expression: str, action_vocab: list[str], target_site_id: str) -> bool:
    for action_id in action_vocab:
        site_id, _ = parse_scoped_action_id(action_id)
        if site_id == target_site_id and apply_action_id(expression, action_id) is not None:
            return True
    return False


def label_candidate_action(
    expression: str,
    goal: str,
    action_id: str,
    action_vocab: list[str],
    target_action_id: str,
    target_next_expr: str | None = None,
    target_distance: int | None = None,
    horizon: int = 3,
    current_distance: int | None = None,
    max_nodes: int = 4_000,
) -> CandidateFrontierLabels:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    next_expr = apply_action_id(expression, action_id)
    if next_expr is None:
        return CandidateFrontierLabels(
            action_id=action_id,
            candidate_valid=False,
            next_expr=None,
            reaches_target_site_within_horizon=False,
            reaches_target_action_within_horizon=False,
            reaches_goal_within_horizon=False,
            reduces_distance_to_goal=False,
            nodes_expanded=0,
        )

    target_site_id, _ = parse_scoped_action_id(target_action_id)
    target_next_struct = structural_string(target_next_expr) if target_next_expr is not None else None
    target_is_still_ahead = not (
        current_distance is not None and target_distance is not None and current_distance < target_distance
    )
    goal_struct = structural_string(goal)
    reaches_goal = structural_string(next_expr) == goal_struct
    reaches_target_action = (
        target_is_still_ahead and action_id == target_action_id and _matches_target_next(next_expr, target_next_struct)
    )
    reaches_target_site = target_is_still_ahead and parse_scoped_action_id(action_id)[0] == target_site_id
    reduces_distance = False
    nodes_expanded = 0

    queue: deque[tuple[str, int]] = deque([(next_expr, 1)])
    visited = {structural_string(next_expr)}

    while queue and nodes_expanded < max_nodes:
        current, depth = queue.popleft()
        nodes_expanded += 1
        if structural_string(current) == goal_struct:
            reaches_goal = True
        if target_is_still_ahead and depth < horizon:
            target_child = apply_action_id(current, target_action_id)
            if target_child is not None and _matches_target_next(target_child, target_next_struct):
                reaches_target_action = True
                reaches_target_site = True
            elif _target_site_is_accessible(current, action_vocab, target_site_id):
                reaches_target_site = True
        if depth >= horizon:
            continue
        for next_action_id, child in valid_candidate_actions(current, action_vocab):
            child_depth = depth + 1
            child_struct = structural_string(child)
            if (
                target_is_still_ahead
                and next_action_id == target_action_id
                and child_depth <= horizon
                and _matches_target_next(child, target_next_struct)
            ):
                reaches_target_action = True
                reaches_target_site = True
            elif target_is_still_ahead and parse_scoped_action_id(next_action_id)[0] == target_site_id and child_depth <= horizon:
                reaches_target_site = True
            if child_struct == goal_struct and child_depth <= horizon:
                reaches_goal = True
            if child_struct in visited:
                continue
            visited.add(child_struct)
            queue.append((child, child_depth))

    if current_distance is not None and current_distance > 0:
        reduces_distance = _can_reach_goal_within(
            next_expr,
            goal,
            max_steps=current_distance - 1,
            action_vocab=action_vocab,
            max_nodes=max_nodes,
        )

    return CandidateFrontierLabels(
        action_id=action_id,
        candidate_valid=True,
        next_expr=next_expr,
        reaches_target_site_within_horizon=reaches_target_site,
        reaches_target_action_within_horizon=reaches_target_action,
        reaches_goal_within_horizon=reaches_goal,
        reduces_distance_to_goal=reduces_distance,
        nodes_expanded=nodes_expanded,
    )


def _matches_target_next(candidate_expr: str, target_next_struct: str | None) -> bool:
    return target_next_struct is None or structural_string(candidate_expr) == target_next_struct


def _can_reach_goal_within(start_expr: str, goal: str, max_steps: int, action_vocab: list[str], max_nodes: int) -> bool:
    goal_struct = structural_string(goal)
    if structural_string(start_expr) == goal_struct:
        return True
    if max_steps <= 0:
        return False

    queue: deque[tuple[str, int]] = deque([(start_expr, 0)])
    visited = {structural_string(start_expr)}
    nodes_expanded = 0
    while queue:
        current, depth = queue.popleft()
        nodes_expanded += 1
        if nodes_expanded > max_nodes:
            return False
        if depth >= max_steps:
            continue
        for _, child in valid_candidate_actions(current, action_vocab):
            child_depth = depth + 1
            child_struct = structural_string(child)
            if child_struct == goal_struct:
                return True
            if child_struct in visited:
                continue
            visited.add(child_struct)
            queue.append((child, child_depth))
    return False


def build_frontier_label_frame(
    frame: pd.DataFrame,
    action_vocab: list[str],
    horizon: int = 3,
    max_nodes: int = 4_000,
    include_distance_label: bool = True,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    target_next_by_trajectory: dict[str, str] = {}
    target_distance_by_trajectory: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        if int(row.distance_to_goal) <= 0:
            continue
        target_action_id = infer_hidden_cancel_action(str(row.action_order))
        if str(row.guided_action_id) == target_action_id:
            target_next_expr = apply_action_id(str(row.state_str), target_action_id)
            if target_next_expr is not None:
                target_next_by_trajectory[str(row.trajectory_id)] = target_next_expr
                target_distance_by_trajectory[str(row.trajectory_id)] = int(row.distance_to_goal)

    for row in frame.itertuples(index=False):
        if int(row.distance_to_goal) <= 0:
            continue
        target_action_id = infer_hidden_cancel_action(str(row.action_order))
        guided_action_id = str(row.guided_action_id)
        target_next_expr = target_next_by_trajectory.get(str(row.trajectory_id))
        target_distance = target_distance_by_trajectory.get(str(row.trajectory_id))
        for action_id, _ in valid_candidate_actions(str(row.state_str), action_vocab):
            labels = label_candidate_action(
                expression=str(row.state_str),
                goal=str(row.goal_str),
                action_id=action_id,
                action_vocab=action_vocab,
                target_action_id=target_action_id,
                target_next_expr=target_next_expr,
                target_distance=target_distance,
                horizon=horizon,
                current_distance=int(row.distance_to_goal) if include_distance_label else None,
                max_nodes=max_nodes,
            )
            record = asdict(labels)
            record.update(
                {
                    "state_str": str(row.state_str),
                    "goal_str": str(row.goal_str),
                    "trajectory_id": str(row.trajectory_id),
                    "expr_family": str(row.expr_family),
                    "distance_to_goal": int(row.distance_to_goal),
                    "guided_action_id": guided_action_id,
                    "is_guided_action": action_id == guided_action_id,
                    "target_action_id": target_action_id,
                    "target_distance": target_distance,
                    "horizon": int(horizon),
                }
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def write_frontier_label_frame(
    dataset_path: str | Path,
    action_vocab: list[str],
    output_path: str | Path,
    horizon: int = 3,
    max_nodes: int = 4_000,
    include_distance_label: bool = True,
) -> pd.DataFrame:
    frame = pd.read_parquet(dataset_path)
    label_frame = build_frontier_label_frame(
        frame,
        action_vocab=action_vocab,
        horizon=horizon,
        max_nodes=max_nodes,
        include_distance_label=include_distance_label,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    label_frame.to_parquet(output, index=False)
    return label_frame
