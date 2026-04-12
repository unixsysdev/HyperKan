from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VARIANTS: list[dict[str, object]] = [
    {
        "name": "last_layer_only",
        "config": "configs/hyperkan_recovery/hyperkan_last_layer_only.yaml",
        "train": True,
    },
    {
        "name": "small_hyper",
        "config": "configs/hyperkan_recovery/hyperkan_small_hyper.yaml",
        "train": True,
    },
    {
        "name": "templates_2",
        "config": "configs/hyperkan_recovery/hyperkan_templates_2.yaml",
        "train": True,
    },
    {
        "name": "soft_routing",
        "config": "configs/hyperkan_recovery/hyperkan_soft_routing.yaml",
        "train": True,
    },
    # Eval-only calibration on the existing HyperKAN checkpoint.
    {
        "name": "search_temp",
        "config": None,
        "train": False,
        "policy_temperature": 1.5,
    },
]


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(json.dumps({"event": "exec", "cmd": cmd}), flush=True)
    subprocess.check_call(cmd, cwd=REPO_ROOT, env=env)


def main() -> None:
    dataset = Path(os.environ.get("MATHY_TEST_DATASET", "artifacts/generated/test.parquet"))
    checkpoints_root = Path(os.environ.get("MATHY_RECOVERY_CKPT_ROOT", "artifacts/checkpoints_recovery"))
    outputs_root = Path(os.environ.get("MATHY_RECOVERY_OUT_ROOT", "artifacts/hyperkan_recovery"))
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    started = time.time()
    for variant in VARIANTS:
        name = str(variant["name"])
        variant_out = outputs_root / name
        variant_out.mkdir(parents=True, exist_ok=True)

        if bool(variant["train"]):
            ckpt_out = checkpoints_root / name
            ckpt_out.mkdir(parents=True, exist_ok=True)
            run(
                [
                    sys.executable,
                    "-m",
                    "train.run_experiment",
                    "--config",
                    str(REPO_ROOT / str(variant["config"])),
                    "--model-type",
                    "hyperkan",
                    "--output-dir",
                    str(ckpt_out),
                ],
                env=env,
            )
            checkpoint = ckpt_out / "hyperkan" / "best.pt"
        else:
            checkpoint = REPO_ROOT / "artifacts" / "checkpoints" / "hyperkan" / "best.pt"

        policy_temp = variant.get("policy_temperature")
        for mode in ("greedy", "beam"):
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_shallow_benchmark_worker.py"),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(variant_out),
                "--model",
                "hyperkan",
                "--mode",
                mode,
                "--checkpoint",
                str(checkpoint),
                "--progress-every",
                "25",
            ]
            if policy_temp is not None:
                cmd += ["--policy-temperature", str(policy_temp)]
            run(cmd, env=env)

        print(json.dumps({"event": "variant_done", "variant": name}), flush=True)

    print(json.dumps({"event": "sweep_done", "elapsed_sec": round(time.time() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()

