# Depth-7 Learned Frontier First Check

Branch: `feature/depth7-learned-frontier`

This check adds an optional HyperKAN frontier head trained from guided short-horizon labels.  The first run is an in-family mechanism check, not a held-out-family generalization claim: the random split includes `mixed_trig_hidden_apart` rows in train/validation/test so the model can see depth-7 examples during training.

## Dataset

- Raw dataset: `artifacts/scoped_depth7_frontier_infamily_raw/`
- Frontier-target dataset: `artifacts/scoped_depth7_frontier_infamily/`
- Families: `trig_merge`, `hidden_cancel`, `apart_normalize`, `mixed_trig_hidden`, `mixed_trig_hidden_apart`
- Rows: 300 total, 215 train, 42 validation, 43 test
- Action vocabulary: 19 actions
- Frontier target: guided action reaches the inferred non-root hidden-cancel action within the next 3 guided actions
- Frontier positives: 54 train, 9 validation, 9 test

## Training

Command:

```bash
PYTHONPATH=. LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
  python3 train/run_experiment.py \
  --config configs/scoped_depth7_frontier_infamily.yaml \
  --model-type hyperkan \
  --output-dir artifacts/scoped_depth7_frontier_infamily_checkpoints
```

The 5-epoch run finished cleanly on ROCm.  The frontier loss was learnable:

- train frontier loss: 0.464 -> 0.063
- validation frontier loss: 0.267 -> 0.058

## Test Results

All results below use beam width 4 and max steps 7 on `artifacts/scoped_depth7_frontier_infamily/test.parquet`.

| Condition | Overall solves | `mixed_trig_hidden_apart` solves | Mean expansions |
| --- | ---: | ---: | ---: |
| Default | 30/34 | 7/7 | 84.44 |
| Root penalty 2.0 | 27/34 | 7/7 | 100.82 |
| Learned frontier 0.5, first 4 steps | 25/34 | 7/7 | 115.97 |
| Root penalty 2.0 + learned frontier 0.5, first 4 steps | 24/34 | 7/7 | 125.06 |

## Interpretation

This first check answers a narrow question: depth-7 is trainable when depth-7 examples are present in the training split.  It does not yet show that the learned frontier head improves search, because default inference already solves all 7 in-family depth-7 test rows.

The learned-frontier inference score preserves the depth-7 solves but hurts overall solve rate and expansions at weight 0.5.  The next useful experiment is therefore a harder split or curriculum setting where the default policy does not already saturate the depth-7 rows, then a smaller learned-frontier weight sweep such as 0.05, 0.1, and 0.25.

## Transfer Diagnostic

The in-family split above is too easy, so the next check uses a harder curriculum/transfer split:

- Raw dataset: `artifacts/scoped_depth7_frontier_transfer_raw/`
- Frontier-target dataset: `artifacts/scoped_depth7_frontier_transfer/`
- Train/validation families: `trig_merge`, `hidden_cancel`, `apart_normalize`, `mixed_trig_hidden`
- Test family: `mixed_trig_hidden_apart`
- Rows: 300 total, 170 train, 34 validation, 96 test
- Frontier positives: 30 train, 6 validation, 36 test

The recovered HyperKAN model trains cleanly here too:

- train frontier loss: 0.569 -> 0.053
- validation frontier loss: 0.427 -> 0.078

Full held-out depth-7 beam evaluation is slow on this split because failed beams expand heavily.  The first diagnostic slice uses the first 12 non-terminal held-out depth-7 attempts:

| Condition | Solves | Mean expansions |
| --- | ---: | ---: |
| Default | 0/12 | 257.17 |
| Root penalty 2.0 | 0/12 | 312.17 |
| Root penalty 2.0 + heuristic frontier reranker 0.5 | 0/12 | 317.33 |
| Learned frontier 0.05, first 4 steps | 0/12 | 260.42 |
| Learned frontier 0.1, first 4 steps | 0/12 | 258.42 |
| Learned frontier 0.25, first 4 steps | 0/12 | 278.58 |
| Root penalty 2.0 + learned frontier 0.1, first 4 steps | 0/12 | 316.67 |

This is a stronger scientific checkpoint than the in-family run, and it is negative so far.  The learned frontier head learns the moderate-depth target but does not transfer into solved depth-7 held-out rows on this diagnostic slice.  Small learned-frontier weights are mostly neutral at 0.05 and 0.1, while 0.25 starts to increase expansions.  Root penalty and the heuristic reranker also do not solve the first 12 transfer rows for this trained model.

The current conclusion remains: the auxiliary head is technically wired and learnable, but this implementation has not yet improved the depth-7 transfer problem.

## Static KAN Transfer Diagnostic

The same auxiliary frontier head was also added to Static KAN and trained on the same transfer split.

Command:

```bash
PYTHONPATH=. LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
  python3 train/run_experiment.py \
  --config configs/scoped_depth7_frontier_transfer_static.yaml \
  --model-type static_kan \
  --output-dir artifacts/scoped_depth7_frontier_transfer_checkpoints
```

The Static KAN frontier label is also learnable:

- train frontier loss: 0.486 -> 0.046
- validation frontier loss: 0.322 -> 0.067

On the same first 12 non-terminal held-out depth-7 attempts:

| Condition | Solves | Mean expansions |
| --- | ---: | ---: |
| Default | 0/12 | 296.83 |
| Root penalty 2.0 | 0/12 | 327.75 |
| Root penalty 2.0 + heuristic frontier reranker 0.5 | 1/12 | 314.08 |
| Learned frontier 0.05, first 4 steps | 0/12 | 306.50 |
| Learned frontier 0.1, first 4 steps | 0/12 | 305.92 |
| Learned frontier 0.25, first 4 steps | 0/12 | 293.67 |
| Root penalty 2.0 + learned frontier 0.1, first 4 steps | 0/12 | 324.42 |

This adds a small model-comparison insight: the heuristic reranker can still find one shallow Static KAN solve on this diagnostic slice, but the learned frontier head does not.  The learned score slightly reduces expansions at weight 0.25, but it still does not convert transfer rows into solves.

The conclusion is unchanged across recovered HyperKAN and Static KAN: local short-horizon frontier supervision is learnable, but the current label is too weak or too local to solve held-out depth-7 transfer.
