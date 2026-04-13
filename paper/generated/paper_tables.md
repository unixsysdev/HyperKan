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

## Bounded Depth-7 Transfer Diagnostics

| Model | Condition | Solves | Mean expansions |
|---|---|---:|---:|
| Recovered HyperKAN | Default | 0/12 | 257.17 |
| Recovered HyperKAN | Root penalty 2.0 | 0/12 | 312.17 |
| Recovered HyperKAN | Root penalty 2.0 + heuristic frontier reranker 0.5 | 0/12 | 317.33 |
| Recovered HyperKAN | Learned frontier 0.1, first 4 steps | 0/12 | 258.42 |
| Recovered HyperKAN | Root penalty 2.0 + learned frontier 0.1 | 0/12 | 316.67 |
| Static KAN | Default | 0/12 | 296.83 |
| Static KAN | Root penalty 2.0 | 0/12 | 327.75 |
| Static KAN | Root penalty 2.0 + heuristic frontier reranker 0.5 | 1/12 | 314.08 |
| Static KAN | Learned frontier 0.1, first 4 steps | 0/12 | 305.92 |
| Static KAN | Root penalty 2.0 + learned frontier 0.1 | 0/12 | 324.42 |

## RL Frontier Controller Diagnostic

| Condition | Solves | Mean expansions |
|---|---:|---:|
| Supervised learned frontier 0.1, first 4 steps | 0/12 | 258.42 |
| RL frontier 0.1, first 4 steps | 0/12 | 264.17 |
| Root penalty 2.0 + supervised learned frontier 0.1 | 0/12 | 316.67 |
| Root penalty 2.0 + RL frontier 0.1 | 0/12 | 319.00 |
