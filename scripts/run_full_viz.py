"""Generate all publication-quality visualizations from trained checkpoints.

Produces:
  - model_comparison.png: headline bar chart (solve rate, mean steps, val loss)
  - solve_rate_by_depth.png: depth-2 vs depth-3 breakdown per model
  - trajectory_solved_all.png: example solved by all models
  - trajectory_static_kan_only.png: example solved only by static_kan
  - trajectory_failed_all.png: example failed by all models
  - action_confusion.png: predicted vs optimal first action per model
  - spline_comparison.png: static_kan vs hyperkan spline geometry
"""
from __future__ import annotations

import json
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

from data_gen.actions import ACTION_NAMES, action_mask, apply_action
from data_gen.canonicalize import canonicalize
from models.factory import create_model
from tokenizer.srepr_tokenizer import SReprTokenizer
from search.beam_search import run_beam_search
from viz.plot_trajectories import plot_trajectory_graph


OUT = Path("artifacts/logs")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "mlp": "artifacts/checkpoints/mlp/best.pt",
    "static_kan": "artifacts/checkpoints/static_kan/best.pt",
    "hyperkan": "artifacts/checkpoints/hyperkan/best.pt",
}
COLORS = {"mlp": "#6366f1", "static_kan": "#f59e0b", "hyperkan": "#10b981"}


def load_model(checkpoint_path: str):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    tokenizer = SReprTokenizer.load(payload["tokenizer_path"])
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_id"] = tokenizer.pad_id
    model = create_model(payload["model_type"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device, config


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


def get_predicted_action(model, tokenizer, state_str, goal_str, max_length, device):
    batch = encode_batch(tokenizer, [(state_str, goal_str)], max_length, device)
    with torch.no_grad():
        out = model(*batch)
    logits = out["logits"][0]
    probs = torch.softmax(logits, dim=-1)
    mask = action_mask(state_str)
    masked_probs = probs.cpu().numpy() * np.array(mask, dtype=float)
    if masked_probs.sum() == 0:
        return None, probs.cpu().numpy()
    return ACTION_NAMES[int(np.argmax(masked_probs))], probs.cpu().numpy()


print("Loading models...")
models = {}
for name, path in MODELS.items():
    models[name] = load_model(path)
    print(f"  {name}: loaded")

test = pd.read_parquet("artifacts/generated/test.parquet")
non_terminal = test[test["distance_to_goal"] > 0].copy()
print(f"Test set: {len(non_terminal)} non-terminal rows")

# ============================================================
# 1. Per-row beam search for all models (on subset for speed)
# ============================================================
print("\nRunning beam search for all models...")
subset = non_terminal.head(15)  # small subset for fast iteration

results = {name: [] for name in MODELS}
for idx, row in enumerate(subset.itertuples(index=False)):
    if idx % 10 == 0:
        print(f"  row {idx}/{len(subset)}...")
    for name in MODELS:
        model, tokenizer, device, config = models[name]
        outcome = run_beam_search(
            model=model, tokenizer=tokenizer,
            start_expr=row.state_str, goal_expr=row.goal_str,
            beam_width=4, max_steps=8, max_length=256, device=device,
        )
        results[name].append({
            "state_str": row.state_str,
            "goal_str": row.goal_str,
            "family": row.expr_family,
            "depth": row.distance_to_goal,
            "solved": outcome["success"],
            "steps": len(outcome["node"].steps) if outcome["success"] and outcome["node"] else 0,
            "explored": outcome.get("explored", []),
        })

# ============================================================
# 2. Model comparison bar chart
# ============================================================
print("\nPlotting model comparison...")
val_losses = {
    "mlp": 0.0221,
    "static_kan": 0.0222,
    "hyperkan": 0.0122,
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Solve rate
solve_rates = {name: sum(1 for r in res if r["solved"]) / len(res) for name, res in results.items()}
ax = axes[0]
bars = ax.bar(solve_rates.keys(), solve_rates.values(), color=[COLORS[n] for n in solve_rates])
ax.set_ylabel("Solve Rate")
ax.set_title("Verified Solve Rate")
ax.set_ylim(0, 1)
for bar, val in zip(bars, solve_rates.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.1%}", ha="center", fontsize=11)

# Mean steps
mean_steps = {}
for name, res in results.items():
    solved = [r["steps"] for r in res if r["solved"]]
    mean_steps[name] = np.mean(solved) if solved else 0
ax = axes[1]
bars = ax.bar(mean_steps.keys(), mean_steps.values(), color=[COLORS[n] for n in mean_steps])
ax.set_ylabel("Mean Steps to Solve")
ax.set_title("Efficiency (lower = better)")
for bar, val in zip(bars, mean_steps.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", fontsize=11)

# Val action loss
ax = axes[2]
bars = ax.bar(val_losses.keys(), val_losses.values(), color=[COLORS[n] for n in val_losses])
ax.set_ylabel("Val Action Loss (BCE)")
ax.set_title("Supervised Loss (lower = better)")
for bar, val in zip(bars, val_losses.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, f"{val:.4f}", ha="center", fontsize=11)

fig.suptitle("Model Comparison: MLP vs Static KAN vs HyperKAN", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  model_comparison.png")

# ============================================================
# 3. Solve rate by depth
# ============================================================
print("\nPlotting solve rate by depth...")
depths = sorted(set(r["depth"] for r in results["mlp"]))
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(depths))
width = 0.25
for i, name in enumerate(MODELS):
    rates = []
    for d in depths:
        rows_at_d = [r for r in results[name] if r["depth"] == d]
        rate = sum(1 for r in rows_at_d if r["solved"]) / len(rows_at_d) if rows_at_d else 0
        rates.append(rate)
    bars = ax.bar(x + i * width, rates, width, label=name, color=COLORS[name])
    for bar, val in zip(bars, rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.0%}", ha="center", fontsize=9)

ax.set_xlabel("Distance to Goal (depth)")
ax.set_ylabel("Solve Rate")
ax.set_title("Verified Solve Rate by Problem Depth")
ax.set_xticks(x + width)
ax.set_xticklabels([f"depth {d}" for d in depths])
ax.legend()
ax.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig(OUT / "solve_rate_by_depth.png", dpi=150)
plt.close(fig)
print("  solve_rate_by_depth.png")

# ============================================================
# 4. Trajectory examples
# ============================================================
print("\nFinding trajectory examples...")

# Solved by all
solved_all = None
for i in range(len(subset)):
    if all(results[name][i]["solved"] for name in MODELS):
        solved_all = i
        break

# Solved only by static_kan
static_only = None
for i in range(len(subset)):
    if results["static_kan"][i]["solved"] and not results["mlp"][i]["solved"] and not results["hyperkan"][i]["solved"]:
        static_only = i
        break

# Failed by all
failed_all = None
for i in range(len(subset)):
    if not any(results[name][i]["solved"] for name in MODELS):
        failed_all = i
        break

if solved_all is not None:
    r = results["static_kan"][solved_all]
    plot_trajectory_graph(
        r["explored"], OUT / "trajectory_solved_all.png",
        title=f"Solved by all models ({r['family']}, depth {r['depth']})\n{r['state_str'][:70]} -> {r['goal_str'][:70]}",
    )
    print(f"  trajectory_solved_all.png (row {solved_all})")
else:
    print("  no example solved by all models found")

if static_only is not None:
    r = results["static_kan"][static_only]
    plot_trajectory_graph(
        r["explored"], OUT / "trajectory_static_kan_only.png",
        title=f"Solved ONLY by static_kan ({r['family']}, depth {r['depth']})\n{r['state_str'][:70]} -> {r['goal_str'][:70]}",
    )
    print(f"  trajectory_static_kan_only.png (row {static_only})")
else:
    print("  no example solved only by static_kan found")

if failed_all is not None:
    r = results["mlp"][failed_all]
    plot_trajectory_graph(
        r["explored"], OUT / "trajectory_failed_all.png",
        title=f"Failed by all models ({r['family']}, depth {r['depth']})\n{r['state_str'][:70]} -> {r['goal_str'][:70]}",
    )
    print(f"  trajectory_failed_all.png (row {failed_all})")
else:
    print("  no example failed by all models found")

# ============================================================
# 5. Action confusion: predicted vs optimal first action
# ============================================================
print("\nPlotting action confusion...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
analysis_rows = non_terminal.head(15)

for ax_idx, name in enumerate(MODELS):
    model, tokenizer, device, config = models[name]
    confusion = np.zeros((len(ACTION_NAMES), len(ACTION_NAMES)))
    for row in analysis_rows.itertuples(index=False):
        pred_action, _ = get_predicted_action(model, tokenizer, row.state_str, row.goal_str, 256, device)
        if pred_action is None:
            continue
        pred_idx = ACTION_NAMES.index(pred_action)
        for opt_idx, v in enumerate(row.valid_shortest_actions):
            if v:
                confusion[opt_idx, pred_idx] += 1

    ax = axes[ax_idx]
    im = ax.imshow(confusion, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_yticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ACTION_NAMES, fontsize=8)
    ax.set_xlabel("Predicted Action")
    ax.set_ylabel("Optimal Action")
    ax.set_title(name)
    for i in range(len(ACTION_NAMES)):
        for j in range(len(ACTION_NAMES)):
            val = int(confusion[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8,
                        color="white" if val > confusion.max()/2 else "black")

fig.suptitle("Action Confusion: Predicted vs Optimal First Action", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "action_confusion.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  action_confusion.png")

# ============================================================
# 6. Spline comparison: static_kan vs hyperkan
# ============================================================
print("\nPlotting spline comparison...")
# Use a few examples with different goals but same state-family
spline_rows = non_terminal.head(8)
pairs = [(r.state_str, r.goal_str) for r in spline_rows.itertuples(index=False)]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col, name in enumerate(["static_kan", "hyperkan"]):
    model, tokenizer, device, config = models[name]
    batch = encode_batch(tokenizer, pairs, 256, device)
    with torch.no_grad():
        out = model(*batch)

    if "spline_states" not in out:
        continue

    for layer_idx in range(2):
        ax = axes[layer_idx, col]
        weights = out["spline_states"][layer_idx].mixture_weights.detach().cpu().numpy()
        im = ax.imshow(weights, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{name} — Layer {layer_idx + 1}")
        ax.set_xlabel("Spline Template")
        ax.set_ylabel("Example")
        plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Spline Mixture Weights: Static KAN vs HyperKAN\n(same 12 test examples, different goal expressions)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "spline_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  spline_comparison.png")

# ============================================================
# 7. Solve rate by family
# ============================================================
print("\nPlotting solve rate by family...")
families = sorted(set(r["family"] for r in results["mlp"]))
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(families))
width = 0.25
for i, name in enumerate(MODELS):
    rates = []
    for f in families:
        rows_f = [r for r in results[name] if r["family"] == f]
        rate = sum(1 for r in rows_f if r["solved"]) / len(rows_f) if rows_f else 0
        rates.append(rate)
    ax.bar(x + i * width, rates, width, label=name, color=COLORS[name])

ax.set_xlabel("Expression Family")
ax.set_ylabel("Solve Rate")
ax.set_title("Verified Solve Rate by Motif Family")
ax.set_xticks(x + width)
ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7)
ax.legend()
ax.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig(OUT / "solve_rate_by_family.png", dpi=150)
plt.close(fig)
print("  solve_rate_by_family.png")

print("\nAll visualizations done!")
