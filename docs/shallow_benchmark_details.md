# Shallow Benchmark: Extra Diagnostics

This page holds the deeper diagnostics for the 274-row shallow benchmark whose headline results are summarized in the repo [README](../README.md). All plots referenced here are generated locally and committed under `docs/` for stable rendering.

Source of truth for metrics: `artifacts/shallow_benchmark_parallel/summary.json` (not committed).

## Action Confusion (15-row diagnostic sample)

![Action confusion](action_confusion.png)

This is a small-sample diagnostic of predicted first action vs. optimal first action (multi-hot shortest-first-action labels). Treat patterns here as suggestive only.

## KAN Spline Geometry

![Spline comparison](spline_comparison.png)

Static KAN uses fixed spline templates by construction. HyperKAN uses a hypernetwork to produce goal-conditioned spline mixture weights.

### Spline mixtures by optimal action (small sample)

![Spline by action](spline_by_action.png)

## HyperKAN Conditional Routing (diagnostics)

These plots probe whether HyperKAN’s routing is meaningfully goal- and family-dependent, even though it does not improve verified solving on the shallow benchmark.

### Same state, different goals

![Same state, different goals](hyperkan_same_state_diff_goals.png)

### Routing within a motif family

![Family routing](hyperkan_family_routing.png)

### Average routing per family

![Average routing per family](hyperkan_family_avg_routing.png)

### Routing along a solved trajectory

![Trajectory routing](hyperkan_trajectory_routing.png)

