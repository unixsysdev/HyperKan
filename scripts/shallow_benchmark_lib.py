from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.factory import create_model
from search.beam_search import run_beam_search
from tokenizer.srepr_tokenizer import SReprTokenizer


DEFAULT_MODELS = {
    "mlp": "artifacts/checkpoints/mlp/best.pt",
    "static_kan": "artifacts/checkpoints/static_kan/best.pt",
    "hyperkan": "artifacts/checkpoints/hyperkan/best.pt",
}


def load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, SReprTokenizer, dict, torch.device]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload["config"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id
    model = create_model(payload["model_type"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, config, device


def summarize_outcomes(outcomes: list[dict[str, object]]) -> dict[str, float]:
    attempts = len(outcomes)
    solved_rows = [row for row in outcomes if row["solved"]]
    solved = len(solved_rows)
    solve_rate = solved / attempts if attempts else 0.0
    mean_steps = (sum(int(row["steps"]) for row in solved_rows if row["steps"] is not None) / solved) if solved else 0.0
    return {"attempts": attempts, "solved": solved, "solve_rate": solve_rate, "mean_solved_steps": mean_steps}


def summarize_by_key(outcomes: list[dict[str, object]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in outcomes:
        grouped[str(row[key])].append(row)
    return {group: summarize_outcomes(rows) for group, rows in sorted(grouped.items())}


def get_search_config(config: dict, mode: str) -> dict[str, float | int]:
    search_cfg = config.get("search", {})
    return {
        "beam_width": 1 if mode == "greedy" else int(search_cfg.get("beam_width", 4)),
        "max_steps": int(search_cfg.get("max_steps", 8)),
        "max_length": int(config["data"].get("max_length", 256)),
        "value_weight": float(search_cfg.get("value_weight", 0.5)),
        "revisit_penalty": float(search_cfg.get("revisit_penalty", 1.5)),
    }


def collect_row_outcomes(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    frame: pd.DataFrame,
    beam_width: int,
    max_steps: int,
    max_length: int,
    value_weight: float,
    revisit_penalty: float,
    device: torch.device,
    label: str,
    progress_every: int,
    status_path: Path | None,
    model_name: str,
    mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    non_terminal = frame[frame["distance_to_goal"] > 0]
    total = len(non_terminal)
    for index, row in enumerate(non_terminal.itertuples(index=False), start=1):
        outcome = run_beam_search(
            model=model,
            tokenizer=tokenizer,
            start_expr=row.state_str,
            goal_expr=row.goal_str,
            beam_width=beam_width,
            max_steps=max_steps,
            max_length=max_length,
            value_weight=value_weight,
            revisit_penalty=revisit_penalty,
            device=device,
        )
        rows.append(
            {
                "state_str": row.state_str,
                "goal_str": row.goal_str,
                "expr_family": row.expr_family,
                "depth": int(row.distance_to_goal),
                "solved": bool(outcome["success"]),
                "steps": len(outcome["node"].steps) if outcome["success"] and outcome["node"] is not None else None,
            }
        )
        if index == 1 or index % progress_every == 0 or index == total:
            elapsed = time.perf_counter() - start_time
            rate = index / elapsed if elapsed > 0 else 0.0
            progress_payload = {
                "event": "stage_progress",
                "model": model_name,
                "mode": mode,
                "stage": label,
                "processed": index,
                "total": total,
                "elapsed_sec": round(elapsed, 2),
                "rows_per_sec": round(rate, 2),
                "solved_so_far": sum(1 for item in rows if item["solved"]),
            }
            print(json.dumps(progress_payload), flush=True)
            if status_path is not None:
                status_path.write_text(json.dumps(progress_payload, indent=2), encoding="utf-8")
    return rows


def build_mode_summary(
    model_name: str,
    checkpoint_path: Path,
    history_path: Path,
    dataset_path: Path,
    output_dir: Path,
    progress_every: int,
    mode: str,
    write_rows: bool,
) -> dict[str, object]:
    mode_start = time.perf_counter()
    status_path = output_dir / f"{model_name}_{mode}_status.json"
    start_payload = {
        "event": "worker_start",
        "model": model_name,
        "mode": mode,
        "checkpoint": str(checkpoint_path),
    }
    print(json.dumps(start_payload), flush=True)
    status_path.write_text(json.dumps(start_payload, indent=2), encoding="utf-8")

    torch.set_num_threads(1)
    model, tokenizer, config, device = load_model(checkpoint_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best_val_loss = min(record["val"]["total_loss"] for record in history)
    frame = pd.read_parquet(dataset_path)
    search = get_search_config(config, mode=mode)

    outcomes = collect_row_outcomes(
        model=model,
        tokenizer=tokenizer,
        frame=frame,
        beam_width=int(search["beam_width"]),
        max_steps=int(search["max_steps"]),
        max_length=int(search["max_length"]),
        value_weight=float(search["value_weight"]),
        revisit_penalty=float(search["revisit_penalty"]),
        device=device,
        label=f"{model_name}:{mode}",
        progress_every=progress_every,
        status_path=status_path,
        model_name=model_name,
        mode=mode,
    )

    summary = {
        "model": model_name,
        "mode": mode,
        "checkpoint": str(checkpoint_path),
        "best_val_total_loss": best_val_loss,
        "elapsed_sec": round(time.perf_counter() - mode_start, 2),
        "search": search,
        "metrics": summarize_outcomes(outcomes),
        "per_depth": summarize_by_key(outcomes, "depth"),
        "per_family": summarize_by_key(outcomes, "expr_family"),
    }
    (output_dir / f"{model_name}_{mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if write_rows:
        (output_dir / f"{model_name}_{mode}_rows.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    print(json.dumps({"event": "worker_done", "model": model_name, "mode": mode, "elapsed_sec": summary["elapsed_sec"]}), flush=True)
    return summary

