from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.factory import create_model
from search.beam_search import run_beam_search
from tokenizer.srepr_tokenizer import SReprTokenizer
from train.run_experiment import set_seed
from train.train_one_epoch import build_dataloader, run_epoch


def beam_eval_dataset(
    model: torch.nn.Module,
    tokenizer: SReprTokenizer,
    dataset_path: Path,
    beam_width: int,
    max_steps: int,
    max_length: int,
    value_weight: float,
    revisit_penalty: float,
    policy_temperature: float,
    device: torch.device,
    progress_every: int = 25,
) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    non_terminal = frame[frame["distance_to_goal"] > 0]
    total = len(non_terminal)
    solved = 0
    solved_steps = 0
    start = time.perf_counter()
    for idx, row in enumerate(non_terminal.itertuples(index=False), start=1):
        out = run_beam_search(
            model=model,
            tokenizer=tokenizer,
            start_expr=row.state_str,
            goal_expr=row.goal_str,
            beam_width=beam_width,
            max_steps=max_steps,
            max_length=max_length,
            value_weight=value_weight,
            revisit_penalty=revisit_penalty,
            policy_temperature=policy_temperature,
            device=device,
        )
        if out["success"]:
            solved += 1
            solved_steps += len(out["node"].steps) if out["node"] is not None else 0
        if idx == 1 or idx % progress_every == 0 or idx == total:
            elapsed = time.perf_counter() - start
            payload = {
                "event": "beam_eval_progress",
                "processed": idx,
                "total": total,
                "solved_so_far": solved,
                "rows_per_sec": round(idx / elapsed, 3) if elapsed > 0 else 0.0,
                "elapsed_sec": round(elapsed, 2),
            }
            print(json.dumps(payload), flush=True)
    rate = solved / total if total else 0.0
    mean_steps = solved_steps / solved if solved else 0.0
    return {
        "attempts": total,
        "solved": solved,
        "solve_rate": rate,
        "mean_solved_steps": mean_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue training from a checkpoint and run periodic beam evals.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-epochs", type=int, default=30)
    parser.add_argument("--eval-dataset", type=Path, required=True, help="Parquet subset for fast beam eval")
    parser.add_argument("--eval-epochs", type=str, default="35,50", help="Comma-separated epochs to run beam eval at")
    parser.add_argument("--save-every", type=int, default=5, help="Save a checkpoint every N epochs")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    model_type = payload["model_type"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id

    set_seed(int(config.get("seed", 17)))
    model = create_model(model_type, config)
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    train_loader = build_dataloader(
        config["data"]["train_path"],
        tokenizer,
        max_length=int(config["data"]["max_length"]),
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
    )
    val_loader = build_dataloader(
        config["data"]["val_path"],
        tokenizer,
        max_length=int(config["data"]["max_length"]),
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
    )

    search_cfg = config.get("search", {})
    beam_width = int(search_cfg.get("beam_width", 4))
    max_steps = int(search_cfg.get("max_steps", 8))
    max_length = int(config["data"].get("max_length", 256))
    value_weight = float(search_cfg.get("value_weight", 0.5))
    revisit_penalty = float(search_cfg.get("revisit_penalty", 1.5))
    policy_temperature = float(search_cfg.get("policy_temperature", 1.0))

    eval_epochs = {int(x.strip()) for x in str(args.eval_epochs).split(",") if x.strip()}

    # We don't know the true epoch number of the checkpoint; assume it corresponds to the
    # end of the original run's max_epochs (typically 20) and continue numbering from there.
    start_epoch = int(config["train"].get("max_epochs", 20))
    end_epoch = start_epoch + int(args.extra_epochs)

    run_log = args.output_dir / "train_and_beam_log.jsonl"
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            value_weight=float(config["train"]["value_loss_weight"]),
            entropy_weight=float(config["train"]["entropy_weight"]),
            mixture_entropy_weight=float(config["train"].get("mixture_entropy_weight", 0.0)),
            grad_clip_norm=float(config["train"]["grad_clip_norm"]),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            value_weight=float(config["train"]["value_loss_weight"]),
            entropy_weight=float(config["train"]["entropy_weight"]),
            mixture_entropy_weight=float(config["train"].get("mixture_entropy_weight", 0.0)),
            grad_clip_norm=float(config["train"]["grad_clip_norm"]),
        )
        record: dict[str, object] = {"epoch": epoch, "train": train_metrics, "val": val_metrics}

        if epoch % int(args.save_every) == 0 or epoch == end_epoch:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "model_type": model_type,
                    "config": config,
                    "state_dict": model.state_dict(),
                    "tokenizer_path": payload["tokenizer_path"],
                    "source_checkpoint": str(args.checkpoint),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            record["checkpoint"] = str(ckpt_path)

        if epoch in eval_epochs:
            beam_metrics = beam_eval_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset_path=args.eval_dataset,
                beam_width=beam_width,
                max_steps=max_steps,
                max_length=max_length,
                value_weight=value_weight,
                revisit_penalty=revisit_penalty,
                policy_temperature=policy_temperature,
                device=device,
                progress_every=int(args.progress_every),
            )
            record["beam_eval"] = {
                "dataset": str(args.eval_dataset),
                "beam_width": beam_width,
                "max_steps": max_steps,
                "value_weight": value_weight,
                "revisit_penalty": revisit_penalty,
                "policy_temperature": policy_temperature,
                "metrics": beam_metrics,
            }

        run_log.write_text("", encoding="utf-8") if not run_log.exists() else None
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({"event": "epoch_done", "epoch": epoch, "val_total_loss": val_metrics.get("total_loss")}), flush=True)

    (args.output_dir / "done.json").write_text(
        json.dumps({"event": "done", "elapsed_sec": round(time.time() - started, 2), "end_epoch": end_epoch}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

