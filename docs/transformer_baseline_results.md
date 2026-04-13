# Transformer Encoder Baseline

This branch keeps the Transformer check narrow. It swaps the shared BiGRU encoder for a lightweight Transformer encoder while leaving the existing MLP policy/value heads, training loop, scoped beam evaluator, and search-control conditions unchanged. The goal is not to add an architecture zoo; it is to test whether the held-out root-collapse and depth-7 transfer failures disappear under an attention-based encoder.

## Implementation

- Added `encoder_type: transformer` as an opt-in model config field.
- Added `SharedTransformerEncoder` with learned positional embeddings, 2 layers, 4 heads, hidden size 160, and feed-forward size 320 in the baseline configs.
- Added configs:
  - `configs/scoped_structural_probe_transformer.yaml`
  - `configs/scoped_depth7_frontier_transfer_transformer.yaml`

Default configs still use the original BiGRU encoder.

## Structural Probe

Command family:

```bash
toolbox run -c llama-rocm-7.2 bash -lc 'cd /home/marcel/Work/Mathy && source scripts/toolbox_env.sh && python3 -m train.run_experiment --config configs/scoped_structural_probe_transformer.yaml --model-type mlp --output-dir artifacts/scoped_structural_probe_transformer_checkpoints'
```

The 5-epoch Transformer-MLP run selected `artifacts/scoped_structural_probe_transformer_checkpoints/mlp/best.pt`.

Held-out `mixed_trig_hidden` beam-4 results:

| Condition | Solves | Mean expansions |
| --- | ---: | ---: |
| Default | 0/60 | 103.82 |
| Root penalty 2.0 | 3/60 | 101.45 |
| Root penalty 2.0 + heuristic frontier reranker | 24/60 | 80.40 |

Interpretation: the Transformer encoder does not remove default held-out collapse on this small controlled run. Frontier shaping still matters, but the result is weaker than the recovered HyperKAN frontier-reranker result reported in the paper.

## Depth-7 Transfer Slice

Command family:

```bash
toolbox run -c llama-rocm-7.2 bash -lc 'cd /home/marcel/Work/Mathy && source scripts/toolbox_env.sh && python3 -m train.run_experiment --config configs/scoped_depth7_frontier_transfer_transformer.yaml --model-type mlp --output-dir artifacts/scoped_depth7_frontier_transfer_transformer_checkpoints'
```

The 5-epoch Transformer-MLP run selected `artifacts/scoped_depth7_frontier_transfer_transformer_checkpoints/mlp/best.pt`.

First 12 non-terminal held-out `mixed_trig_hidden_apart` attempts:

| Condition | Solves | Mean expansions |
| --- | ---: | ---: |
| Default | 0/12 | 260.75 |
| Root penalty 2.0 | 0/12 | 307.17 |
| Root penalty 2.0 + heuristic frontier reranker | 0/12 | 307.25 |

Interpretation: the bounded depth-7 slice remains unsolved in this Transformer-MLP check. This is not a universal negative result for all attention-based encoders, but it is enough to avoid claiming that the observed failure is only a BiGRU bottleneck artifact.

## Caveat

This is a lightweight baseline, not a tuned Transformer study. It tests one compact attention encoder under the existing data, action vocabulary, and search-control conditions.
