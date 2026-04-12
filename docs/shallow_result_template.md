# Shallow Benchmark Template

Dataset:
- `artifacts/generated/test.parquet`

Models:
- `mlp`
- `static_kan`
- `hyperkan`

Metrics to fill:
- Greedy solve rate
- Beam solve rate
- Rescue rate: `beam_solved - greedy_solved`
- Per-depth solve rate
- Per-family solve rate
- Best validation total loss

Current locked single-run headline:
- `mlp`: `147/274` = `53.6%`
- `static_kan`: `163/274` = `59.5%`
- `hyperkan`: `148/274` = `54.0%`

Three-seed extension:
- Seed 1:
- Seed 2:
- Seed 3:

Notes:
- Keep beam and greedy on the same test set.
- Treat this as the official shallow baseline until a deeper benchmark is validated.
