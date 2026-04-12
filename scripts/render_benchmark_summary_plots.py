from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"mlp": "#4c78a8", "static_kan": "#f58518", "hyperkan": "#54a24b"}
MODEL_ORDER = ["mlp", "static_kan", "hyperkan"]
MODEL_LABELS = {"mlp": "MLP", "static_kan": "Static KAN", "hyperkan": "HyperKAN"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_solve_rate_by_depth(summary: dict, out_path: Path) -> None:
    depths = ["2", "3"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(depths))
    width = 0.25
    for idx, model in enumerate(MODEL_ORDER):
        rates = [summary["models"][model]["per_depth_beam"][d]["solve_rate"] for d in depths]
        ax.bar(x + idx * width, rates, width, label=MODEL_LABELS[model], color=COLORS[model])
        for j, r in enumerate(rates):
            ax.text(x[j] + idx * width, r + 0.02, f"{r:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"depth {d}" for d in depths])
    ax.set_ylabel("Verified Solve Rate (beam)")
    ax.set_title("Solve Rate by Depth (Beam Search)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_greedy_vs_beam(summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    greedy = [summary["models"][m]["greedy"]["solve_rate"] for m in MODEL_ORDER]
    beam = [summary["models"][m]["beam"]["solve_rate"] for m in MODEL_ORDER]
    ax.bar(x - width / 2, greedy, width, label="greedy (beam=1)", color="#9aa0a6")
    ax.bar(x + width / 2, beam, width, label="beam (width=4)", color="#1f77b4")
    for i, (g, b) in enumerate(zip(greedy, beam, strict=False)):
        ax.text(i - width / 2, g + 0.02, f"{g:.1%}", ha="center", fontsize=10)
        ax.text(i + width / 2, b + 0.02, f"{b:.1%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Verified Solve Rate")
    ax.set_title("Greedy vs Beam Solve Rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rescue(summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(MODEL_ORDER))
    rescue = [summary["models"][m]["rescue"]["rate"] for m in MODEL_ORDER]
    ax.bar(x, rescue, color=[COLORS[m] for m in MODEL_ORDER])
    for i, r in enumerate(rescue):
        ax.text(i, r + 0.002, f"+{r:.1%}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Beam − Greedy Solve Rate")
    ax.set_title("Beam Search Rescue")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_family_heatmap(summary: dict, out_path: Path) -> None:
    families = sorted(summary["models"]["mlp"]["per_family_beam"].keys())
    data = np.array([[summary["models"][m]["per_family_beam"][fam]["solve_rate"] for fam in families] for m in MODEL_ORDER])

    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=25, ha="right", fontsize=9)
    ax.set_title("Solve Rate by Motif Family (Beam)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]*100:.0f}%", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("solve rate")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(summary: dict, out_path: Path) -> None:
    # Headline: overall beam solve rate + depth-3 solve rate in one figure.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    beam_rates = [summary["models"][m]["beam"]["solve_rate"] for m in MODEL_ORDER]
    depth3_rates = [summary["models"][m]["per_depth_beam"]["3"]["solve_rate"] for m in MODEL_ORDER]

    ax = axes[0]
    bars = ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], beam_rates, color=[COLORS[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Solve Rate")
    ax.set_title("Overall (Beam)")
    for bar, val in zip(bars, beam_rates, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.1%}", ha="center", fontsize=10)

    ax = axes[1]
    bars = ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], depth3_rates, color=[COLORS[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Solve Rate")
    ax.set_title("Depth-3 (Beam)")
    for bar, val in zip(bars, depth3_rates, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.1%}", ha="center", fontsize=10)

    fig.suptitle("Verified Solve Rate — Model Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README plots from benchmark summary.json")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/shallow_benchmark_parallel/summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    summary = _load(args.summary)
    out = args.out_dir

    plot_model_comparison(summary, out / "model_comparison.png")
    plot_solve_rate_by_depth(summary, out / "solve_rate_by_depth.png")
    plot_family_heatmap(summary, out / "solve_rate_by_family.png")
    plot_greedy_vs_beam(summary, out / "greedy_vs_beam.png")
    plot_rescue(summary, out / "rescue.png")


if __name__ == "__main__":
    main()

