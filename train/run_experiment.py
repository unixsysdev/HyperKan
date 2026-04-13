from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from models.factory import create_model
from search.scoped_beam_search import load_scoped_action_vocab
from train.train_one_epoch import build_dataloader, build_tokenizer, run_epoch
from tokenizer.srepr_tokenizer import SReprTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_tokenizer(config: dict, output_dir: Path) -> Path:
    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return tokenizer_path
    build_tokenizer(config["data"]["train_path"], tokenizer_path)
    return tokenizer_path


def save_checkpoint(
    path: Path,
    model_type: str,
    config: dict,
    model: torch.nn.Module,
    tokenizer_path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_type": model_type,
            "config": config,
            "state_dict": model.state_dict(),
            "tokenizer_path": str(tokenizer_path),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def build_expr_root_action_mask(action_vocab_path: str | Path) -> torch.Tensor:
    action_vocab = load_scoped_action_vocab(action_vocab_path)
    return torch.tensor(
        [action_id.startswith("expr@root::") for action_id in action_vocab],
        dtype=torch.bool,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model variant for verified algebraic rewriting")
    parser.add_argument("--config", type=Path, default=Path("configs/local_poc.yaml"))
    parser.add_argument("--model-type", choices=("mlp", "static_kan", "hyperkan"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints"))
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])

    tokenizer_path = ensure_tokenizer(config, output_dir)
    tokenizer = SReprTokenizer.load(tokenizer_path)
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id
    expr_root_action_mask = build_expr_root_action_mask(config["data"]["action_vocab_path"])

    train_loader = build_dataloader(
        config["data"]["train_path"],
        tokenizer,
        max_length=config["data"]["max_length"],
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        config["data"]["val_path"],
        tokenizer,
        max_length=config["data"]["max_length"],
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    model = create_model(args.model_type, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    history = []
    best_val = float("inf")
    best_checkpoint = output_dir / "best.pt"
    save_every_epoch = bool(config["train"].get("save_every_epoch", True))

    for epoch in range(1, config["train"]["max_epochs"] + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            value_weight=config["train"]["value_loss_weight"],
            entropy_weight=config["train"]["entropy_weight"],
            mixture_entropy_weight=float(config["train"].get("mixture_entropy_weight", 0.0)),
            grad_clip_norm=config["train"]["grad_clip_norm"],
            expr_root_action_mask=expr_root_action_mask,
            expr_root_avoidance_weight=float(config["train"].get("expr_root_avoidance_weight", 0.0)),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            value_weight=config["train"]["value_loss_weight"],
            entropy_weight=config["train"]["entropy_weight"],
            mixture_entropy_weight=float(config["train"].get("mixture_entropy_weight", 0.0)),
            grad_clip_norm=config["train"]["grad_clip_norm"],
            expr_root_action_mask=expr_root_action_mask,
            expr_root_avoidance_weight=float(config["train"].get("expr_root_avoidance_weight", 0.0)),
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record))

        if save_every_epoch:
            save_checkpoint(
                output_dir / f"epoch_{epoch}.pt",
                args.model_type,
                config,
                model,
                tokenizer_path,
                epoch,
                train_metrics,
                val_metrics,
            )

        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            save_checkpoint(
                best_checkpoint,
                args.model_type,
                config,
                model,
                tokenizer_path,
                epoch,
                train_metrics,
                val_metrics,
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"best_checkpoint": str(best_checkpoint), "device": str(device)}))


if __name__ == "__main__":
    main()
