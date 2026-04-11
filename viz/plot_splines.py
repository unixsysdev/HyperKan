from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_goal_conditioned_splines(
    spline_states: list[object],
    output_path: str | Path,
    title: str = "Goal-Conditioned Spline Mixtures",
) -> None:
    fig, axes = plt.subplots(len(spline_states), 1, figsize=(8, 3 * max(1, len(spline_states))))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    else:
        axes = [axes]
    for idx, state in enumerate(spline_states):
        if is_dataclass(state):
            state = asdict(state)
        weights = state["mixture_weights"].detach().cpu().numpy()
        ax = axes[idx]
        ax.imshow(weights, aspect="auto", cmap="viridis")
        ax.set_title(f"Layer {idx + 1}")
        ax.set_xlabel("Template")
        ax.set_ylabel("Example")
    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
