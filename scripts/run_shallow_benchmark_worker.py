from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shallow_benchmark_lib import DEFAULT_MODELS, build_mode_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one official benchmark worker pass")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/generated/test.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(DEFAULT_MODELS.keys()), required=True)
    parser.add_argument("--mode", choices=("greedy", "beam"), required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--write-rows", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(DEFAULT_MODELS[args.model])
    history_path = checkpoint_path.parent / "history.json"
    build_mode_summary(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
        mode=args.mode,
        write_rows=args.write_rows,
    )


if __name__ == "__main__":
    main()

