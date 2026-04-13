from __future__ import annotations

import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "generated"


def load_json(path: str) -> dict:
    full_path = REPO / path
    with full_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_rate(metrics: dict) -> str:
    return f"{metrics['solved']}/{metrics['attempts']} ({100.0 * metrics['solve_rate']:.1f}%)"


def json_metrics(path: str) -> dict:
    data = load_json(path)
    if "metrics" in data:
        return data["metrics"]
    return data


def table_global() -> list[str]:
    rows = [
        ("MLP", "artifacts/logs/eval_mlp.json", "0/127"),
        ("Static KAN", "artifacts/logs/eval_static_kan.json", "16/127"),
        ("HyperKAN initial", "artifacts/logs/eval_hyperkan.json", "1/127"),
    ]
    lines = [
        "## Global Benchmark",
        "",
        "| Model | Beam solves | Depth-3 solves |",
        "|---|---:|---:|",
    ]
    for name, path, depth3 in rows:
        lines.append(f"| {name} | {fmt_rate(json_metrics(path))} | {depth3} |")
    return lines


def table_structural() -> list[str]:
    rows = [
        ("Default beam-4", "artifacts/scoped_structural_probe_search_checkpoints/hyperkan/test_beam4_best_search.json"),
        (
            "Root penalty 2.0 + hidden-action bonus beam-4",
            "artifacts/scoped_structural_probe_search_checkpoints/hyperkan/test_beam4_root2_hiddenbonus05.json",
        ),
        (
            "Root penalty 2.0 + frontier reranker beam-4",
            "artifacts/scoped_structural_probe_search_checkpoints/hyperkan/test_beam4_root2_frontier_hidden_cancel_05_s3.json",
        ),
    ]
    lines = [
        "## Scoped Structural Probe",
        "",
        "| Condition | Held-out mixed-family solves | Mean expansions |",
        "|---|---:|---:|",
    ]
    for name, path in rows:
        metrics = json_metrics(path)
        lines.append(f"| {name} | {metrics['solved']}/{metrics['attempts']} | {metrics['mean_expansions']:.2f} |")
    return lines


def table_depth7() -> list[str]:
    rows = [
        ("Default beam-4", "artifacts/scoped_depth_expansion_probe_checkpoints/hyperkan/hyperkan/test_beam4_default.json"),
        ("Root penalty 2.0 beam-4", "artifacts/scoped_depth_expansion_probe_checkpoints/hyperkan/hyperkan/test_beam4_root2.json"),
        (
            "Root penalty 2.0 + frontier reranker beam-4",
            "artifacts/scoped_depth_expansion_probe_checkpoints/hyperkan/hyperkan/test_beam4_root2_frontier.json",
        ),
    ]
    lines = [
        "## Scoped Depth-7 Expansion",
        "",
        "| Condition | Held-out depth-7 solves | Mean expansions |",
        "|---|---:|---:|",
    ]
    for name, path in rows:
        metrics = json_metrics(path)
        lines.append(f"| {name} | {metrics['solved']}/{metrics['attempts']} | {metrics['mean_expansions']:.2f} |")
    return lines


def table_bounded_diagnostics() -> list[str]:
    # These bounded diagnostics are intentionally small transfer slices; the
    # paper reports them as negative checks, not headline benchmark results.
    lines = [
        "## Bounded Depth-7 Transfer Diagnostics",
        "",
        "| Model | Condition | Solves | Mean expansions |",
        "|---|---|---:|---:|",
        "| Recovered HyperKAN | Default | 0/12 | 257.17 |",
        "| Recovered HyperKAN | Root penalty 2.0 | 0/12 | 312.17 |",
        "| Recovered HyperKAN | Root penalty 2.0 + heuristic frontier reranker 0.5 | 0/12 | 317.33 |",
        "| Recovered HyperKAN | Learned frontier 0.1, first 4 steps | 0/12 | 258.42 |",
        "| Recovered HyperKAN | Root penalty 2.0 + learned frontier 0.1 | 0/12 | 316.67 |",
        "| Static KAN | Default | 0/12 | 296.83 |",
        "| Static KAN | Root penalty 2.0 | 0/12 | 327.75 |",
        "| Static KAN | Root penalty 2.0 + heuristic frontier reranker 0.5 | 1/12 | 314.08 |",
        "| Static KAN | Learned frontier 0.1, first 4 steps | 0/12 | 305.92 |",
        "| Static KAN | Root penalty 2.0 + learned frontier 0.1 | 0/12 | 324.42 |",
        "",
        "## RL Frontier Controller Diagnostic",
        "",
        "| Condition | Solves | Mean expansions |",
        "|---|---:|---:|",
        "| Supervised learned frontier 0.1, first 4 steps | 0/12 | 258.42 |",
        "| RL frontier 0.1, first 4 steps | 0/12 | 264.17 |",
        "| Root penalty 2.0 + supervised learned frontier 0.1 | 0/12 | 316.67 |",
        "| Root penalty 2.0 + RL frontier 0.1 | 0/12 | 319.00 |",
    ]
    return lines


def copy_figures() -> None:
    figure_names = [
        "greedy_vs_beam.png",
        "hyperkan_same_state_diff_goals.png",
        "model_comparison.png",
        "solve_rate_by_family.png",
        "solve_rate_by_depth.png",
        "spline_comparison.png",
        "trajectory_failed_all.png",
        "trajectory_solved_all.png",
        "trajectory_static_kan_only.png",
    ]
    target_dir = OUT / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in figure_names:
        shutil.copy2(REPO / "docs" / name, target_dir / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = []
    tables.extend(table_global())
    tables.append("")
    tables.extend(table_structural())
    tables.append("")
    tables.extend(table_depth7())
    tables.append("")
    tables.extend(table_bounded_diagnostics())
    tables.append("")

    (OUT / "paper_tables.md").write_text("\n".join(tables), encoding="utf-8")
    copy_figures()

    summary = {
        "generated": [
            "paper/generated/paper_tables.md",
            "paper/generated/figures/model_comparison.png",
            "paper/generated/figures/solve_rate_by_depth.png",
            "paper/generated/figures/solve_rate_by_family.png",
            "paper/generated/figures/greedy_vs_beam.png",
            "paper/generated/figures/hyperkan_same_state_diff_goals.png",
            "paper/generated/figures/spline_comparison.png",
            "paper/generated/figures/trajectory_failed_all.png",
            "paper/generated/figures/trajectory_solved_all.png",
            "paper/generated/figures/trajectory_static_kan_only.png",
        ]
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote paper/generated/paper_tables.md")
    print("wrote paper/generated/manifest.json")
    print("copied paper/generated/figures/*.png")


if __name__ == "__main__":
    main()
