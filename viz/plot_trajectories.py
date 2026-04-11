from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def plot_trajectory_graph(edges: list[dict[str, object]], output_path: str | Path, title: str = "Verified Trajectory Graph") -> None:
    graph = nx.DiGraph()
    for edge in edges:
        graph.add_edge(edge["from"], edge["to"], label=edge["action"], score=edge["score"])

    pos = nx.spring_layout(graph, seed=17)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="#dbeafe", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=True, edge_color="#64748b", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={(u, v): data["label"] for u, v, data in graph.edges(data=True)},
        font_size=7,
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

