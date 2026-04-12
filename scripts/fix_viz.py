"""Rebuild corrected visualizations.

Fixes:
  - model_comparison.png: use actual full eval JSON numbers
  - trajectory graphs: green successful path, gray dead branches, shorter labels
"""
from __future__ import annotations

import json
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from pathlib import Path

from data_gen.actions import ACTION_NAMES, action_mask
from models.factory import create_model
from tokenizer.srepr_tokenizer import SReprTokenizer
from search.beam_search import run_beam_search
from data_gen.canonicalize import canonicalize

OUT = Path("docs")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"mlp": "#6366f1", "static_kan": "#f59e0b", "hyperkan": "#10b981"}

# ============================================================
# 1. Corrected model_comparison.png — from actual eval JSONs
# ============================================================
print("Building corrected model_comparison.png...")

eval_data = {
    "mlp":        json.loads(Path("artifacts/logs/eval_mlp.json").read_text()),
    "static_kan": json.loads(Path("artifacts/logs/eval_static_kan.json").read_text()),
    "hyperkan":   json.loads(Path("artifacts/logs/eval_hyperkan.json").read_text()),
}
val_losses = {"mlp": 0.0221, "static_kan": 0.0222, "hyperkan": 0.0122}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Solve rate
solve_rates = {n: d["solve_rate"] for n, d in eval_data.items()}
ax = axes[0]
bars = ax.bar(solve_rates.keys(), solve_rates.values(), color=[COLORS[n] for n in solve_rates])
ax.set_ylabel("Solve Rate")
ax.set_title("Verified Solve Rate\n(274 test problems)")
ax.set_ylim(0, 1)
for bar, (name, val) in zip(bars, solve_rates.items()):
    solved = eval_data[name]["solved"]
    attempts = eval_data[name]["attempts"]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.1%}\n({solved}/{attempts})", ha="center", fontsize=10)

# Mean steps
mean_steps = {n: d["mean_solved_steps"] for n, d in eval_data.items()}
ax = axes[1]
bars = ax.bar(mean_steps.keys(), mean_steps.values(), color=[COLORS[n] for n in mean_steps])
ax.set_ylabel("Mean Steps to Solve")
ax.set_title("Efficiency (lower = better)")
for bar, val in zip(bars, mean_steps.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", fontsize=11)

# Val action loss
ax = axes[2]
bars = ax.bar(val_losses.keys(), val_losses.values(), color=[COLORS[n] for n in val_losses])
ax.set_ylabel("Val Action Loss (BCE)")
ax.set_title("Supervised Loss (lower = better)")
for bar, val in zip(bars, val_losses.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f"{val:.4f}", ha="center", fontsize=11)

fig.suptitle("Model Comparison: MLP vs Static KAN vs HyperKAN\n(full test set, beam-width 4, max 8 steps)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  model_comparison.png done")


# ============================================================
# 2. Improved trajectory plots
# ============================================================

def short_label(expr: str, maxlen: int = 28) -> str:
    s = str(expr).replace("**", "^").replace("*", "·")
    return s[:maxlen] + "…" if len(s) > maxlen else s


def plot_trajectory_clean(
    explored: list[dict],
    successful_path: list[tuple[str, str]] | None,
    output_path: Path,
    title: str,
) -> None:
    """Render beam-search graph with green success path, gray dead branches."""
    graph = nx.DiGraph()
    for edge in explored:
        graph.add_edge(edge["from"], edge["to"], action=edge["action"])

    # Identify nodes/edges on the successful path
    success_nodes: set[str] = set()
    success_edges: set[tuple[str, str]] = set()
    if successful_path:
        prev = None
        for action, expr in successful_path:
            # find the predecessor — walk explored in order
            if prev is None:
                # start node: find by checking edges
                for edge in explored:
                    if edge["to"] == expr and edge["action"] == action:
                        success_nodes.add(edge["from"])
                        success_nodes.add(expr)
                        success_edges.add((edge["from"], expr))
                        prev = expr
                        break
            else:
                success_nodes.add(expr)
                success_edges.add((prev, expr))
                prev = expr

    pos = nx.spring_layout(graph, seed=17, k=2.5)

    # Shorten labels
    label_map = {node: short_label(node) for node in graph.nodes()}

    fig, ax = plt.subplots(figsize=(13, 8))

    # Dead branch nodes and edges
    dead_nodes = [n for n in graph.nodes() if n not in success_nodes]
    success_node_list = list(success_nodes)
    dead_edges = [(u, v) for u, v in graph.edges() if (u, v) not in success_edges]
    success_edge_list = list(success_edges)

    nx.draw_networkx_nodes(graph, pos, nodelist=dead_nodes, node_size=600,
                           node_color="#e2e8f0", ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=success_node_list, node_size=900,
                           node_color="#bbf7d0", ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=dead_edges, arrows=True,
                           edge_color="#cbd5e1", alpha=0.5, ax=ax, width=1.0)
    nx.draw_networkx_edges(graph, pos, edgelist=success_edge_list, arrows=True,
                           edge_color="#16a34a", ax=ax, width=2.5)
    nx.draw_networkx_labels(graph, pos, labels=label_map, font_size=6, ax=ax)

    # Edge labels only on success path
    success_edge_labels = {(u, v): graph[u][v]["action"] for u, v in success_edge_list}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=success_edge_labels,
                                 font_size=7, font_color="#15803d", ax=ax)

    green_patch = mpatches.Patch(color="#bbf7d0", label="Solution path")
    gray_patch = mpatches.Patch(color="#e2e8f0", label="Dead branch")
    ax.legend(handles=[green_patch, gray_patch], loc="lower right", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Load models for trajectory generation
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


print("Loading models for trajectory generation...")
models = {
    "mlp":        load_model("artifacts/checkpoints/mlp/best.pt"),
    "static_kan": load_model("artifacts/checkpoints/static_kan/best.pt"),
    "hyperkan":   load_model("artifacts/checkpoints/hyperkan/best.pt"),
}

test = pd.read_parquet("artifacts/generated/test.parquet")
non_terminal = test[test["distance_to_goal"] > 0].reset_index(drop=True)

print("Running beam search for trajectory examples...")

def run_all(row):
    out = {}
    for name, (model, tok, device) in models.items():
        out[name] = run_beam_search(
            model=model, tokenizer=tok,
            start_expr=row.state_str, goal_expr=row.goal_str,
            beam_width=4, max_steps=8, max_length=256, device=device,
        )
    return out

solved_all_row = None
static_only_row = None
failed_all_row = None

for i, row in enumerate(non_terminal.head(30).itertuples(index=False)):
    outcomes = run_all(row)
    all_solved = all(outcomes[n]["success"] for n in models)
    none_solved = not any(outcomes[n]["success"] for n in models)
    static_only = outcomes["static_kan"]["success"] and not outcomes["mlp"]["success"] and not outcomes["hyperkan"]["success"]

    if all_solved and solved_all_row is None:
        solved_all_row = (row, outcomes)
    if static_only and static_only_row is None:
        static_only_row = (row, outcomes)
    if none_solved and failed_all_row is None:
        failed_all_row = (row, outcomes)

    if solved_all_row and static_only_row and failed_all_row:
        break
    print(f"  row {i}: all_solved={all_solved}, static_only={static_only}, none_solved={none_solved}")

if solved_all_row:
    row, outcomes = solved_all_row
    path = list(outcomes["static_kan"]["node"].steps) if outcomes["static_kan"]["node"] else None
    plot_trajectory_clean(
        outcomes["static_kan"]["explored"], path,
        OUT / "trajectory_solved_all.png",
        title=f"Solved by ALL models — {row.expr_family} (depth {row.distance_to_goal})\n"
              f"{short_label(row.state_str, 60)}  →  {short_label(row.goal_str, 60)}",
    )
    print("  trajectory_solved_all.png done")

if static_only_row:
    row, outcomes = static_only_row
    path = list(outcomes["static_kan"]["node"].steps) if outcomes["static_kan"]["node"] else None
    plot_trajectory_clean(
        outcomes["static_kan"]["explored"], path,
        OUT / "trajectory_static_kan_only.png",
        title=f"Solved ONLY by static_kan — {row.expr_family} (depth {row.distance_to_goal})\n"
              f"{short_label(row.state_str, 60)}  →  {short_label(row.goal_str, 60)}",
    )
    print("  trajectory_static_kan_only.png done")

if failed_all_row:
    row, outcomes = failed_all_row
    plot_trajectory_clean(
        outcomes["mlp"]["explored"], None,
        OUT / "trajectory_failed_all.png",
        title=f"Failed by ALL models — {row.expr_family} (depth {row.distance_to_goal})\n"
              f"{short_label(row.state_str, 60)}  →  {short_label(row.goal_str, 60)}",
    )
    print("  trajectory_failed_all.png done")

print("\nAll fixes done.")
