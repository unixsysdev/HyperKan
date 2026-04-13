# RL Frontier Controller Diagnostic

Branch: `feature/rl-frontier-controller`

This branch tests a bounded version of the RL idea: keep the supervised recovered HyperKAN policy mostly fixed and fine-tune only the auxiliary `frontier_head` with a REINFORCE-style rollout objective.

This is not full PPO and not end-to-end policy RL.  It is a first search-control diagnostic:

- state: current symbolic expression and goal
- action: one valid scoped rewrite action
- policy score: frozen supervised action logits plus a trainable frontier-head term
- reward: `+1` for verified solve, minus a small expansion penalty
- updated parameters: `frontier_head` only
- rollout data: `artifacts/scoped_depth7_frontier_transfer/train.parquet`
- held-out diagnostic: first 12 non-terminal rows from `artifacts/scoped_depth7_frontier_transfer/test.parquet`

## Command

```bash
PYTHONPATH=. LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
  python3 scripts/rl_finetune_frontier_controller.py \
  --checkpoint artifacts/scoped_depth7_frontier_transfer_checkpoints/hyperkan/best.pt \
  --dataset artifacts/scoped_depth7_frontier_transfer/train.parquet \
  --action-vocab artifacts/scoped_depth7_frontier_transfer/scoped_action_vocab.json \
  --output-dir artifacts/rl_frontier_controller/hyperkan \
  --epochs 2 \
  --max-steps 7 \
  --frontier-weight 0.1 \
  --lr 0.0001 \
  --expansion-penalty 0.01 \
  --entropy-weight 0.01
```

Training completed cleanly:

| Epoch | Episodes | Train rollout solves | Mean reward |
| --- | ---: | ---: | ---: |
| 1 | 130 | 68/130 | 0.4728 |
| 2 | 130 | 68/130 | 0.4729 |

The train-rollout solve rate did not improve over the two epochs, which is already a weak sign.

## Held-Out Diagnostic

The RL-fine-tuned checkpoint was evaluated on the same first 12 non-terminal held-out depth-7 transfer attempts used in the learned-frontier diagnostic:

| Condition | Solves | Mean expansions |
| --- | ---: | ---: |
| Supervised learned frontier 0.1, first 4 steps | 0/12 | 258.42 |
| RL frontier 0.1, first 4 steps | 0/12 | 264.17 |
| Root penalty 2.0 + supervised learned frontier 0.1 | 0/12 | 316.67 |
| Root penalty 2.0 + RL frontier 0.1 | 0/12 | 319.00 |

## Conclusion

This first RL controller test does not rescue the depth-7 transfer slice.  It is useful because it validates the RL plumbing and confirms that a naive frontier-head-only REINFORCE update is not enough.

The result should not change the paper story.  The strongest claim remains:

- moderate-depth heuristic frontier shaping rescues recovered HyperKAN,
- depth-7 exposes a failure boundary,
- supervised frontier labels and this first RL frontier-controller attempt do not yet solve that boundary.
