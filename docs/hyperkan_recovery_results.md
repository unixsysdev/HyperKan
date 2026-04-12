# HyperKAN Recovery: Phase A Results

This doc summarizes the Phase A goal: make HyperKAN competitive with Static KAN on the shallow verified benchmark (274 non-terminal test rows; beam width 4; max steps 8).

## Baselines (from shallow benchmark)

- Static KAN beam: **163/274**, depth-3: **16/127**
- HyperKAN (original) beam: **148/274**, depth-3: **1/127**

## Recovery Variants (trained + evaluated)

All variants are HyperKAN and use the same dataset split and eval settings; only architecture knobs changed.

| Variant | Key change | Greedy | Beam | Rescue | Depth-3 beam |
|---|---|---:|---:|---:|---:|
| `last_layer_only` | condition only `kan_2` (uniform mix on `kan_1`) | 146/274 | 156/274 | +10 | 9/127 |
| `templates_2` | `spline_templates: 2` | 147/274 | 156/274 | +9 | 9/127 |
| `small_hyper` | `hyper_hidden_dim: 64` | 147/274 | **162/274** | +15 | **15/127** |
| `soft_routing` | `mixture_temperature: 2.0` + `mixture_entropy_weight: 0.02` | 146/274 | 148/274 | +2 | 1/127 |

Winner: **`small_hyper`**. It nearly matches Static KAN (162 vs 163 overall; 15 vs 16 on depth-3).

## Search Temperature Calibration (eval-only)

Applied `policy_temperature` at beam search time on the `small_hyper` checkpoint.

| policy_temperature | Beam | Depth-3 beam |
|---:|---:|---:|
| 0.7 | 162/274 | 15/127 |
| 1.0 | 162/274 | 15/127 |
| 1.3 | 161/274 | 14/127 |
| 1.6 | 156/274 | 9/127 |

Conclusion: temperature is **not** the missing lever; high temperature hurts.

## Commands

Run the recovery sweep (train + greedy/beam eval) inside the toolbox:

```bash
toolbox run -c llama-rocm-7.2 bash -c 'cd $REPO && source scripts/toolbox_env.sh && python3 scripts/run_hyperkan_recovery_sweep.py'
```

