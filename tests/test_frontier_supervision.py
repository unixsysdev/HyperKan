from __future__ import annotations

from data_gen.frontier_supervision import infer_hidden_cancel_action, label_candidate_action
from scripts.build_scoped_structural_dataset import (
    FAMILY_ACTIONS,
    build_action_vocab,
    build_mixed_trig_hidden_apart_start,
    replay_guided_path,
)


def _depth7_fixture():
    params = ((1, 2, 1, 3), (1, 2, 6), (1, 3, 1, 2))
    start, _ = build_mixed_trig_hidden_apart_start(params)
    actions = FAMILY_ACTIONS["mixed_trig_hidden_apart"]
    states, goal = replay_guided_path(start, actions)
    vocab = [action_id for action_id, _ in sorted(build_action_vocab(("mixed_trig_hidden_apart",)).items(), key=lambda item: item[1])]
    return actions, states, goal, vocab


def test_infers_depth7_hidden_cancel_action_from_guided_order() -> None:
    action_order = " -> ".join(FAMILY_ACTIONS["mixed_trig_hidden_apart"])

    assert infer_hidden_cancel_action(action_order) == "expr@3::cancel"


def test_guided_candidate_reaches_hidden_cancel_within_three_from_middle_prefix() -> None:
    actions, states, goal, vocab = _depth7_fixture()
    state = str(states[2])
    target_next_expr = str(states[5])

    labels = label_candidate_action(
        expression=state,
        goal=str(goal),
        action_id=actions[2],
        action_vocab=vocab,
        target_action_id="expr@3::cancel",
        target_next_expr=target_next_expr,
        target_distance=3,
        horizon=3,
        current_distance=5,
    )

    assert labels.candidate_valid
    assert labels.reaches_target_site_within_horizon
    assert labels.reaches_target_action_within_horizon
    assert labels.reduces_distance_to_goal


def test_guided_start_candidate_does_not_reach_hidden_cancel_within_three() -> None:
    actions, states, goal, vocab = _depth7_fixture()
    state = str(states[0])
    target_next_expr = str(states[5])

    labels = label_candidate_action(
        expression=state,
        goal=str(goal),
        action_id=actions[0],
        action_vocab=vocab,
        target_action_id="expr@3::cancel",
        target_next_expr=target_next_expr,
        target_distance=3,
        horizon=3,
        current_distance=7,
    )

    assert labels.candidate_valid
    assert not labels.reaches_target_action_within_horizon
    assert labels.reduces_distance_to_goal


def test_one_step_apart_shortcut_is_goal_label_not_hidden_cancel_label() -> None:
    actions, states, goal, vocab = _depth7_fixture()
    state = str(states[6])
    target_next_expr = str(states[5])

    labels = label_candidate_action(
        expression=state,
        goal=str(goal),
        action_id=actions[6],
        action_vocab=vocab,
        target_action_id="expr@3::cancel",
        target_next_expr=target_next_expr,
        target_distance=3,
        horizon=3,
        current_distance=1,
    )

    assert labels.candidate_valid
    assert labels.reaches_goal_within_horizon
    assert labels.reduces_distance_to_goal
    assert not labels.reaches_target_action_within_horizon
