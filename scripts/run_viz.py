"""Generate visualizations from trained checkpoints.

Produces:
  - trajectory_*.png: beam search exploration graphs for solved examples
  - spline_mixtures.png: HyperKAN spline mixture weights
  - spline_by_action.png: spline geometry grouped by optimal first action
"""
from __future__ import annotations

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data_gen.actions import ACTION_NAMES
from models.factory import create_model
from tokenizer.srepr_tokenizer import SReprTokenizer
from search.beam_search import run_beam_search
from viz.plot_splines import plot_goal_conditioned_splines
from viz.plot_trajectories import plot_trajectory_graph


def load_model(checkpoint_path: str):
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
    return model, tokenizer, device


def encode_batch(tokenizer, pairs, max_length, device):
    s_ids, g_ids, s_lens, g_lens = [], [], [], []
    for s, g in pairs:
        enc = tokenizer.encode_pair(s, g, max_length=max_length)
        s_ids.append(enc.state_ids)
        g_ids.append(enc.goal_ids)
        s_lens.append(enc.state_length)
        g_lens.append(enc.goal_length)
    return (
        torch.tensor(s_ids, device=device),
        torch.tensor(s_lens, device=device),
        torch.tensor(g_ids, device=device),
        torch.tensor(g_lens, device=device),
    )


def main():
    out_dir = Path("artifacts/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model("artifacts/checkpoints/hyperkan/best.pt")

    test = pd.read_parquet("artifacts/generated/test.parquet")
    non_terminal = test[test["distance_to_goal"] > 0].head(20)

    # 1. Trajectory graphs for first 3 solved examples
    solved = []
    for row in non_terminal.itertuples(index=False):
        outcome = run_beam_search(
            model=model, tokenizer=tokenizer,
            start_expr=row.state_str, goal_expr=row.goal_str,
            beam_width=4, max_steps=8, max_length=256, device=device,
        )
        if outcome["success"] and outcome["explored"]:
            solved.append((row, outcome))
            if len(solved) >= 3:
                break

    for i, (row, outcome) in enumerate(solved):
        plot_trajectory_graph(
            outcome["explored"],
            out_dir / f"trajectory_{i}.png",
            title=f"{row.expr_family} (depth {row.distance_to_goal})\n"
                  f"{row.state_str[:60]} -> {row.goal_str[:60]}",
        )
        print(f"trajectory_{i}.png: {row.expr_family}, depth={row.distance_to_goal}, "
              f"explored={len(outcome['explored'])} edges")

    # 2. Spline mixtures for solved examples
    if solved:
        pairs = [(ex[0].state_str, ex[0].goal_str) for ex in solved]
        batch = encode_batch(tokenizer, pairs, 256, device)
        with torch.no_grad():
            out = model(*batch)
        if "spline_states" in out:
            plot_goal_conditioned_splines(
                out["spline_states"],
                out_dir / "spline_mixtures.png",
                title="HyperKAN Goal-Conditioned Spline Mixtures",
            )
            print("spline_mixtures.png: done")

    # 3. Per-action spline geometry
    action_groups: dict[str, list[tuple[str, str]]] = {}
    for row in non_terminal.itertuples(index=False):
        actions = row.valid_shortest_actions
        for idx, v in enumerate(actions):
            if v:
                name = ACTION_NAMES[idx]
                if name not in action_groups:
                    action_groups[name] = []
                if len(action_groups[name]) < 8:
                    action_groups[name].append((row.state_str, row.goal_str))
                break

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    for ax_idx, action in enumerate(ACTION_NAMES):
        ax = axes_flat[ax_idx]
        if action not in action_groups or not action_groups[action]:
            ax.set_title(f"{action} (no examples)")
            ax.axis("off")
            continue
        pairs = action_groups[action]
        batch = encode_batch(tokenizer, pairs, 256, device)
        with torch.no_grad():
            o = model(*batch)
        if "spline_states" in o:
            weights = o["spline_states"][0].mixture_weights.detach().cpu().numpy()
            ax.imshow(weights, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{action} ({len(pairs)} examples)")
            ax.set_xlabel("Template")
            ax.set_ylabel("Example")
        else:
            ax.set_title(action)
            ax.axis("off")

    fig.suptitle("KAN Layer 1 Spline Mixtures by Optimal Action", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "spline_by_action.png", dpi=150)
    plt.close(fig)
    print("spline_by_action.png: done")

    print("all viz done")


if __name__ == "__main__":
    main()
