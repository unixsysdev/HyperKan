#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/toolbox_env.sh"

python3 -m data_gen.generate_backward --samples 5000 --output-dir artifacts/generated
python3 - <<'PY'
from pathlib import Path
from train.train_one_epoch import build_tokenizer

build_tokenizer("artifacts/generated/train.parquet", Path("artifacts/generated/tokenizer.json"))
print("tokenizer written to artifacts/generated/tokenizer.json")
PY

python3 -m train.run_experiment --config configs/local_poc.yaml --model-type mlp
python3 -m train.run_experiment --config configs/local_poc.yaml --model-type static_kan
python3 -m train.run_experiment --config configs/local_poc.yaml --model-type hyperkan
