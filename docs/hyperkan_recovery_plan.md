# HyperKAN Recovery Sweep (5 Ablations)

Goal: make **HyperKAN match or exceed Static KAN** on the current 274-row shallow benchmark before spending budget on deeper chains.

Baseline (current shallow benchmark)
- Static KAN beam: 163/274 (59.5%), depth-3: 16/127 (12.6%)
- HyperKAN beam: 148/274 (54.0%), depth-3: 1/127 (0.8%)

## Priority-Ordered Ablations (max 5)

All ablations keep benchmark semantics fixed (same dataset, same beam width/max steps, same scoring), and write outputs to isolated directories.

1. **Condition only last KAN layer** (`last_layer_only`)
   - Disable goal-conditioning for `kan_1`; keep conditioning for `kan_2`.
   - Hypothesis: current hypernetwork modulates too much; restricting conditioning reduces variance/overfit.

2. **Smaller hypernetwork** (`small_hyper`)
   - Reduce hypernetwork hidden size (e.g. `hyper_hidden_dim=64`).
   - Hypothesis: current hypernetwork has more capacity than the shallow benchmark supports.

3. **Fewer templates** (`templates_2`)
   - Reduce `spline_templates` from 4 to 2 for HyperKAN.
   - Hypothesis: smaller template dictionary reduces spurious routing and improves calibration.

4. **Softer routing + routing entropy regularization** (`soft_routing`)
   - Increase mixture softmax temperature (e.g. `mixture_temperature=2.0`).
   - Add mixture entropy regularization during training (`train.mixture_entropy_weight>0`) to discourage near-one-hot routing early.
   - Hypothesis: routing is currently too sharp/unstable and overfits.

5. **Calibrated search temperature (eval-only)** (`search_temp`)
   - Keep checkpoint fixed; apply softmax temperature in beam search (`--policy-temperature`).
   - Hypothesis: HyperKAN logits are miscalibrated for beam ranking; temperature scaling recovers solve rate without retraining.

## How To Run

Run in ROCm toolbox:

```bash
toolbox run -c llama-rocm-7.2 bash -c 'cd $REPO && source scripts/toolbox_env.sh && python3 scripts/run_hyperkan_recovery_sweep.py'
```

Outputs:
- `artifacts/hyperkan_recovery/<variant>/...` per-variant benchmark summaries and status
- `artifacts/checkpoints_recovery/<variant>/hyperkan/best.pt` for trained variants

