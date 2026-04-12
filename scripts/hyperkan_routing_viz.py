"""HyperKAN conditional routing visualizations.

1. same_state_diff_goals.png  — fixed state, varying goal → mixture weights change
2. family_routing.png         — same family, multiple instances → clustering
3. family_avg_routing.png     — averaged routing per motif family
4. trajectory_routing.png     — routing at each step of one solved example
"""
from __future__ import annotations

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data_gen.actions import ACTION_NAMES, apply_action
from data_gen.canonicalize import canonicalize
from models.factory import create_model
from tokenizer.srepr_tokenizer import SReprTokenizer
from search.beam_search import run_beam_search

OUT = Path("docs")
OUT.mkdir(parents=True, exist_ok=True)

def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    tok = SReprTokenizer.load(payload["tokenizer_path"])
    config["model"]["vocab_size"] = tok.vocab_size
    config["model"]["pad_id"] = tok.pad_id
    model = create_model(payload["model_type"], config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tok, device

def encode(tok, pairs, max_length, device):
    s_ids, g_ids, s_lens, g_lens = [], [], [], []
    for s, g in pairs:
        enc = tok.encode_pair(s, g, max_length=max_length)
        s_ids.append(enc.state_ids); g_ids.append(enc.goal_ids)
        s_lens.append(enc.state_length); g_lens.append(enc.goal_length)
    return (torch.tensor(s_ids, device=device), torch.tensor(s_lens, device=device),
            torch.tensor(g_ids, device=device), torch.tensor(g_lens, device=device))

def get_routing(model, tok, pairs, device):
    batch = encode(tok, pairs, 256, device)
    with torch.no_grad():
        out = model(*batch)
    return [s.mixture_weights.detach().cpu().numpy() for s in out["spline_states"]]

print("Loading HyperKAN...")
model, tok, device = load_model("artifacts/checkpoints/hyperkan/best.pt")
test = pd.read_parquet("artifacts/generated/test.parquet")
non_terminal = test[test["distance_to_goal"] > 0].reset_index(drop=True)

# ============================================================
# 1. Same state, different goals
# ============================================================
print("1. Same state, different goals...")

# Pick one state and pair it with goals from different families
anchor_row = non_terminal.iloc[0]
anchor_state = anchor_row.state_str

# Collect 6 distinct goals from different rows
diff_goal_rows = non_terminal.drop_duplicates("expr_family").head(6)
pairs = [(anchor_state, row.goal_str) for row in diff_goal_rows.itertuples(index=False)]
labels = [f"{row.expr_family.replace('rat_','').replace('poly_','').replace('_trig','')[:20]}\n{row.goal_str[:25]}…"
          for row in diff_goal_rows.itertuples(index=False)]

routing = get_routing(model, tok, pairs, device)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for layer_idx, ax in enumerate(axes):
    w = routing[layer_idx]
    im = ax.imshow(w, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(w.shape[1]))
    ax.set_xticklabels([f"T{i}" for i in range(w.shape[1])], fontsize=9)
    ax.set_xlabel("Spline Template")
    ax.set_title(f"Layer {layer_idx + 1}")
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("HyperKAN: Same State, Different Goals\n"
             "Each row = same current expression, different goal → different routing",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "hyperkan_same_state_diff_goals.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  done")

# ============================================================
# 2. Same family, multiple instances — does family cluster?
# ============================================================
print("2. Same family, multiple instances...")

families = non_terminal["expr_family"].unique()
n_families = min(4, len(families))
chosen_families = list(families[:n_families])

fig, axes = plt.subplots(n_families, 2, figsize=(12, 3 * n_families))
if n_families == 1:
    axes = [axes]

for fam_idx, family in enumerate(chosen_families):
    rows = non_terminal[non_terminal["expr_family"] == family].head(6)
    pairs = [(r.state_str, r.goal_str) for r in rows.itertuples(index=False)]
    routing = get_routing(model, tok, pairs, device)
    for layer_idx in range(2):
        ax = axes[fam_idx][layer_idx]
        w = routing[layer_idx]
        im = ax.imshow(w, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(w.shape[1]))
        ax.set_xticklabels([f"T{i}" for i in range(w.shape[1])], fontsize=8)
        ax.set_ylabel("Example", fontsize=7)
        short_fam = family.replace("rat_", "").replace("poly_", "").replace("_trig", "")
        ax.set_title(f"{short_fam[:28]} — Layer {layer_idx + 1}", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("HyperKAN: Routing Within Same Motif Family\n"
             "Do same-family problems cluster into similar template usage?",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "hyperkan_family_routing.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  done")

# ============================================================
# 3. Average routing per family
# ============================================================
print("3. Average routing per family...")

family_avg = {}
for family in non_terminal["expr_family"].unique():
    rows = non_terminal[non_terminal["expr_family"] == family].head(8)
    pairs = [(r.state_str, r.goal_str) for r in rows.itertuples(index=False)]
    routing = get_routing(model, tok, pairs, device)
    family_avg[family] = [r.mean(axis=0) for r in routing]

families_sorted = sorted(family_avg.keys())
n = len(families_sorted)
fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

for fam_idx, family in enumerate(families_sorted):
    short = family.replace("rat_", "").replace("poly_", "").replace("_trig_", "\n")
    for layer_idx in range(2):
        ax = axes[layer_idx, fam_idx]
        w = family_avg[family][layer_idx].reshape(1, -1)
        im = ax.imshow(w, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(w.shape[1]))
        ax.set_xticklabels([f"T{i}" for i in range(w.shape[1])], fontsize=8)
        ax.set_yticks([])
        if layer_idx == 0:
            ax.set_title(short[:20], fontsize=7)
        ax.set_ylabel(f"L{layer_idx+1}", fontsize=8)

fig.suptitle("HyperKAN: Average Spline Routing by Motif Family\n"
             "Each column = one family; rows = Layer 1 and Layer 2",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "hyperkan_family_avg_routing.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  done")

# ============================================================
# 4. Trajectory-step routing (routing changes as we approach goal)
# ============================================================
print("4. Trajectory-step routing...")

# Find a solved example
solved_row = None
solved_outcome = None
for row in non_terminal.head(20).itertuples(index=False):
    outcome = run_beam_search(
        model=model, tokenizer=tok,
        start_expr=row.state_str, goal_expr=row.goal_str,
        beam_width=4, max_steps=8, max_length=256, device=device,
    )
    if outcome["success"] and outcome["node"] and len(outcome["node"].steps) >= 2:
        solved_row = row
        solved_outcome = outcome
        break

if solved_row and solved_outcome:
    steps = solved_outcome["node"].steps  # ((action, expr), ...)
    # Build state sequence: start, after step 1, after step 2, ...
    state_sequence = [solved_row.state_str] + [expr for _, expr in steps]
    goal = solved_row.goal_str
    step_labels = ["start"] + [f"after {action}" for action, _ in steps]

    pairs = [(s, goal) for s in state_sequence]
    routing = get_routing(model, tok, pairs, device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for layer_idx, ax in enumerate(axes):
        w = routing[layer_idx]
        im = ax.imshow(w, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_yticks(range(len(step_labels)))
        ax.set_yticklabels(step_labels, fontsize=9)
        ax.set_xticks(range(w.shape[1]))
        ax.set_xticklabels([f"T{i}" for i in range(w.shape[1])], fontsize=9)
        ax.set_xlabel("Spline Template")
        ax.set_title(f"Layer {layer_idx + 1}")
        plt.colorbar(im, ax=ax, shrink=0.8)

    short_state = solved_row.state_str[:50]
    short_goal = solved_row.goal_str[:50]
    fig.suptitle(f"HyperKAN: Routing Changes Along Solved Trajectory\n"
                 f"{short_state}… → {short_goal}…",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "hyperkan_trajectory_routing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  done ({len(steps)} steps: {[a for a, _ in steps]})")
else:
    print("  no suitable solved example found for trajectory routing")

print("\nAll HyperKAN routing visualizations done.")
