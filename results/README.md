# Results (Committed)

This folder contains small, committed JSON artifacts for reproducibility and public inspection.

Phase A (HyperKAN recovery sweep):
- `results/hyperkan_recovery/` contains per-variant `hyperkan_{greedy,beam}_summary.json` plus the training `history.json` for each recovered checkpoint.
- `results/hyperkan_recovery/search_temp_small_hyper/` contains the eval-only policy-temperature sweep beam summaries.

Large artifacts (checkpoints `.pt`, large intermediate logs) remain uncommitted under `artifacts/` by default.

