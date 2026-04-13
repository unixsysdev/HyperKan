# Released Paper Artifacts

This directory is the manifest for the frozen `paper-v1.0` artifact. The large datasets and checkpoints remain in their original checked-in locations so existing scripts keep working, but the paper-facing subset is listed here explicitly.

## Global Benchmark

- Eval summaries:
  - `artifacts/logs/eval_mlp.json`
  - `artifacts/logs/eval_static_kan.json`
  - `artifacts/logs/eval_hyperkan.json`
- Checkpoints:
  - `artifacts/checkpoints/mlp/best.pt`
  - `artifacts/checkpoints/static_kan/best.pt`
  - `artifacts/checkpoints/hyperkan/best.pt`
- Historical figures:
  - `docs/model_comparison.png`
  - `docs/solve_rate_by_depth.png`
  - `docs/solve_rate_by_family.png`
  - `docs/greedy_vs_beam.png`

## Recovered HyperKAN

- Primary recovered checkpoint:
  - `artifacts/checkpoints_recovery/small_hyper/hyperkan/best.pt`
- Summary:
  - `docs/hyperkan_recovery_results.md`

## Scoped Structural Probe

- Dataset:
  - `artifacts/scoped_structural_probe/`
- Config:
  - `configs/scoped_structural_probe.yaml`
- Search-selected checkpoints and evals:
  - `artifacts/scoped_structural_probe_search_checkpoints/`
- Summary:
  - `docs/scoped_structural_results.md`
  - `docs/early_frontier_reranker_results.md`

## Scoped Depth Expansion

- Dataset:
  - `artifacts/scoped_depth_expansion_probe/`
- Config:
  - `configs/scoped_depth_expansion_probe.yaml`
- Checkpoints and evals:
  - `artifacts/scoped_depth_expansion_probe_checkpoints/`
- Summary:
  - `docs/scoped_depth_expansion_results.md`

## Paper

- TeX:
  - `paper/hyperkan_verified_symbolic_rewrite_search.tex`
- PDF:
  - `paper/hyperkan_verified_symbolic_rewrite_search.pdf`
- Regenerated table/figure outputs:
  - `paper/generated/`
