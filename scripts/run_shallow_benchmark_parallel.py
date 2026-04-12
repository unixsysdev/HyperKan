from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shallow_benchmark_lib import DEFAULT_MODELS


MODES = ("greedy", "beam")


def build_run_status(output_dir: Path, jobs: list[dict[str, object]]) -> dict[str, object]:
    status = {"event": "run_status", "updated_at": round(time.time(), 2), "jobs": []}
    for job in jobs:
        model = str(job["model"])
        mode = str(job["mode"])
        status_path = output_dir / f"{model}_{mode}_status.json"
        summary_path = output_dir / f"{model}_{mode}_summary.json"
        entry = {
            "model": model,
            "mode": mode,
            "pid": job["process"].pid,
            "returncode": job["process"].poll(),
            "status_file": str(status_path),
            "summary_file": str(summary_path),
            "status": None,
        }
        if status_path.exists():
            entry["status"] = json.loads(status_path.read_text(encoding="utf-8"))
        if summary_path.exists():
            entry["summary_ready"] = True
        status["jobs"].append(entry)
    return status


def aggregate_model_summary(output_dir: Path, model: str) -> dict[str, object]:
    greedy = json.loads((output_dir / f"{model}_greedy_summary.json").read_text(encoding="utf-8"))
    beam = json.loads((output_dir / f"{model}_beam_summary.json").read_text(encoding="utf-8"))
    greedy_metrics = greedy["metrics"]
    beam_metrics = beam["metrics"]
    rescue_count = beam_metrics["solved"] - greedy_metrics["solved"]
    rescue_rate = rescue_count / greedy_metrics["attempts"] if greedy_metrics["attempts"] else 0.0
    return {
        "model": model,
        "checkpoint": greedy["checkpoint"],
        "best_val_total_loss": greedy["best_val_total_loss"],
        "elapsed_sec": round(greedy["elapsed_sec"] + beam["elapsed_sec"], 2),
        "greedy_elapsed_sec": greedy["elapsed_sec"],
        "beam_elapsed_sec": beam["elapsed_sec"],
        "search": beam["search"],
        "greedy": greedy_metrics,
        "beam": beam_metrics,
        "rescue": {"count": rescue_count, "rate": rescue_rate},
        "per_depth_beam": beam["per_depth"],
        "per_family_beam": beam["per_family"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official shallow benchmark in parallel")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/generated/test.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/shallow_benchmark_parallel"))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--write-rows", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, object]] = []
    for model in DEFAULT_MODELS:
        for mode in MODES:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_shallow_benchmark_worker.py"),
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(args.output_dir),
                "--model",
                model,
                "--mode",
                mode,
                "--progress-every",
                str(args.progress_every),
            ]
            if args.write_rows:
                cmd.append("--write-rows")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            process = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
            jobs.append({"model": model, "mode": mode, "process": process})
            print(json.dumps({"event": "spawned", "model": model, "mode": mode, "pid": process.pid}), flush=True)

    failures: list[dict[str, object]] = []
    while True:
        run_status = build_run_status(args.output_dir, jobs)
        (args.output_dir / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")
        alive = False
        for job in jobs:
            rc = job["process"].poll()
            if rc is None:
                alive = True
            elif rc != 0 and not any(f["pid"] == job["process"].pid for f in failures):
                failures.append({"model": job["model"], "mode": job["mode"], "pid": job["process"].pid, "returncode": rc})
        if not alive:
            break
        time.sleep(args.poll_seconds)

    summary = {
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "failures": failures,
        "models": {},
    }
    if not failures:
        for model in DEFAULT_MODELS:
            summary["models"][model] = aggregate_model_summary(args.output_dir, model)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        print(json.dumps({"event": "parallel_failed", "failures": failures}, indent=2), flush=True)
        raise SystemExit(1)
    print(json.dumps({"event": "parallel_done", "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
