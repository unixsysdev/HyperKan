from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shallow_benchmark_lib import DEFAULT_MODELS, build_mode_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shallow depth-2/3 benchmark across saved checkpoints")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/generated/test.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/shallow_benchmark"))
    parser.add_argument("--models", nargs="+", choices=sorted(DEFAULT_MODELS.keys()), default=sorted(DEFAULT_MODELS.keys()))
    parser.add_argument("--write-rows", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"dataset": str(args.dataset), "models": {}}

    for model_name in args.models:
        checkpoint = DEFAULT_MODELS[model_name]
        checkpoint_path = Path(checkpoint)
        history_path = checkpoint_path.parent / "history.json"
        greedy_summary = build_mode_summary(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            progress_every=args.progress_every,
            mode="greedy",
            write_rows=args.write_rows,
        )
        beam_summary = build_mode_summary(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            progress_every=args.progress_every,
            mode="beam",
            write_rows=args.write_rows,
        )
        model_summary = {
            "model": model_name,
            "checkpoint": greedy_summary["checkpoint"],
            "best_val_total_loss": greedy_summary["best_val_total_loss"],
            "elapsed_sec": round(greedy_summary["elapsed_sec"] + beam_summary["elapsed_sec"], 2),
            "search": beam_summary["search"],
            "greedy": greedy_summary["metrics"],
            "beam": beam_summary["metrics"],
            "rescue": {
                "count": beam_summary["metrics"]["solved"] - greedy_summary["metrics"]["solved"],
                "rate": (
                    (beam_summary["metrics"]["solved"] - greedy_summary["metrics"]["solved"]) / greedy_summary["metrics"]["attempts"]
                    if greedy_summary["metrics"]["attempts"]
                    else 0.0
                ),
            },
            "per_depth_beam": beam_summary["per_depth"],
            "per_family_beam": beam_summary["per_family"],
        }
        report["models"][model_name] = model_summary
        (args.output_dir / f"{model_name}_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
