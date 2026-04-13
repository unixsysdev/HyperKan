## Global Benchmark

| Model | Beam solves | Depth-3 solves |
|---|---:|---:|
| MLP | 147/274 (53.6%) | 0/127 |
| Static KAN | 163/274 (59.5%) | 16/127 |
| HyperKAN initial | 148/274 (54.0%) | 1/127 |

## Scoped Structural Probe

| Condition | Held-out mixed-family solves | Mean expansions |
|---|---:|---:|
| Default beam-4 | 0/60 | 89.38 |
| Root penalty 2.0 + hidden-action bonus beam-4 | 36/60 | 56.38 |
| Root penalty 2.0 + frontier reranker beam-4 | 48/60 | 59.23 |

## Scoped Depth-7 Expansion

| Condition | Held-out depth-7 solves | Mean expansions |
|---|---:|---:|
| Default beam-4 | 0/84 | 171.23 |
| Root penalty 2.0 beam-4 | 18/84 | 144.67 |
| Root penalty 2.0 + frontier reranker beam-4 | 18/84 | 150.18 |
