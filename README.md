# HyperKan / Mathy: Verified Symbolic Rewrite Search

This repo trains policy/value models for verified symbolic rewrite search. The original global-action benchmark worked end to end and showed **Static KAN > MLP** on verified solve rate; the initial HyperKAN underperformed, then a reduced-capacity recovered HyperKAN nearly matched Static KAN after ablations. The important follow-up result is that the global SymPy action semantics turned out too shallow for robust depth-4+ composition, so the main branch of the project has moved to **scoped actions**: `action = (site, op)`. Scoped smoke and medium guided benchmarks now train and evaluate end to end, but they are still too guided and too clean to separate Static KAN from recovered HyperKAN.

## What This Project Is

The task is goal-directed symbolic rewriting with formal execution:

- A model reads a current expression and an exact target form.
- The policy predicts rewrite actions.
- SymPy executes the selected rewrite.
- Search verifies whether the exact target form is reached.

There are now two benchmark modes:

- **Global actions:** the original six whole-expression SymPy operations: `expand`, `factor`, `cancel`, `apart`, `together`, `trigsimp`.
- **Scoped actions:** the same operator families applied to deterministic sites, serialized as `site::op`, for example `expr@1::factor` or `add_slice@root[0:2]::together`.

The scoped version is not a new model architecture claim. It is a benchmark semantics change: the policy must choose both **where** and **what** to rewrite.

## Global Benchmark Results

The original verified benchmark has 274 non-terminal test problems. Beam search uses width 4 and max 8 steps.

| Model | Beam solves | Solve rate | Depth-3 solves |
|---|---:|---:|---:|
| MLP | 147/274 | 53.6% | 0/127 |
| Static KAN | 163/274 | 59.5% | 16/127 |
| HyperKAN (initial) | 148/274 | 54.0% | 1/127 |

Depth breakdown:

- All models solve depth-2: `147/147`.
- Depth-3 is the only real discriminator:
  - MLP: `0/127`
  - Static KAN: `16/127`
  - HyperKAN: `1/127`

Conclusion: **Static KAN is the best model on the global benchmark**, but the benchmark is effectively shallow. Depth-2 is saturated, and depth-3 is the only separator.

The plots below are from the earlier global-action runs. They remain useful historical diagnostics, but they are not scoped-benchmark results.

![Model comparison](docs/model_comparison.png)

![Solve rate by depth](docs/solve_rate_by_depth.png)

![Solve rate by family](docs/solve_rate_by_family.png)

![Greedy vs beam](docs/greedy_vs_beam.png)

## HyperKAN Recovery

The HyperKAN recovery branch tested whether the initial HyperKAN underperformance was a capacity/routing issue rather than a fundamental failure.

The best recovery variant was the reduced-capacity HyperKAN:

- Variant: `small_hyper`
- Key change: `hyper_hidden_dim = 64`
- Recovered HyperKAN: `162/274` overall, `15/127` on depth-3
- Static KAN: `163/274` overall, `16/127` on depth-3

Search-temperature calibration did not close the remaining gap. Longer training also did not help reliably: epoch-35 and epoch-50 checks showed verified search can regress even while supervised validation loss improves.

Main takeaway: **search-based checkpoint selection matters**. Supervised validation loss is not sufficient for model selection in this setup.

See [docs/hyperkan_recovery_results.md](docs/hyperkan_recovery_results.md) and `results/hyperkan_recovery/` for the recovery details.

## Why We Moved Away From Global Actions

The depth problem stopped being generator guesswork after local graph mining.

Under global actions, isolated ladders exist:

- `block_a_trig_merge_expand` gives a verified depth-3 ladder.
- `block_b_cancel_expand` gives a verified depth-2 ladder.

But naive additive composition under global SymPy rewrites did **not** produce robust depth-4+ families at useful density. Whole-expression rewrites often simplify multiple latent subproblems at once, so composed depth collapses or becomes too hard to verify reliably.

That is why the project moved toward scoped actions.

Relevant files/artifacts:

- [scripts/mine_family_graphs.py](scripts/mine_family_graphs.py)
- [results/composite_family_mining_initial_scan.json](results/composite_family_mining_initial_scan.json)
- [results/global_family_final_bounded_scan.json](results/global_family_final_bounded_scan.json)

## Scoped-Action Benchmark: Current Status

Scoped actions use:

```text
action = (site, op)
```

The current scoped site model uses deterministic subtree path sites plus grouped `Add` slices. Grouped slices were added because normal SymPy `Add` flattening can erase a logical block as a single subtree.

Examples:

- `expr@root::trigsimp`
- `expr@1::factor`
- `numerator@1::expand`
- `denominator@1::factor`
- `add_slice@root[0:2]::together`

Core implementation:

- [data_gen/scoped_actions.py](data_gen/scoped_actions.py)
- [docs/scoped_action_spec.md](docs/scoped_action_spec.md)

Current scoped verification state:

- Strict single-block scoped verification works:
  - block A verifies at depth `3`
  - block B verifies at depth `2`
- Strict composed verification is still hard.
- Guided first-path composition succeeds for `A3+B1` at scoped depth `4`.
- Strict composed verification and shortest-action tie recovery are still open.

So scoped actions are a live path forward, but this is not the final strict scoped benchmark yet.

## Scoped Smoke Benchmark

The scoped smoke benchmark uses guided single-path labels for the `A3+B1` family.

Artifacts:

- Dataset: `artifacts/scoped_smoke/`
- Config: [configs/scoped_smoke.yaml](configs/scoped_smoke.yaml)
- Action vocab: `artifacts/scoped_smoke/scoped_action_vocab.json`
- Eval: [eval/run_scoped_smoke_eval.py](eval/run_scoped_smoke_eval.py)

Dataset:

- `24` guided `A3+B1` trajectories
- `120` rows
- `4` scoped actions

Results:

| Model | Non-terminal test solves |
|---|---:|
| Static KAN | 16/16 |
| HyperKAN | 16/16 |

Conclusion: the scoped dataset, tokenizer, model head, training loop, and scoped beam inference path work end to end. This is a smoke milestone, not a final benchmark result.

## Scoped Medium Guided Benchmark

The medium guided benchmark scales the same guided `A3+B1` family and uses a harder split.

Artifacts:

- Dataset: `artifacts/scoped_medium/`
- Config: [configs/scoped_medium.yaml](configs/scoped_medium.yaml)
- Checkpoints: `artifacts/scoped_medium_checkpoints/`

Dataset:

- `200` guided `A3+B1` trajectories
- `1000` rows
- Split: `heldout_coeff`
- Test split holds out `(a_coeff_0, a_coeff_1) = (4, 4)`
- `48` non-terminal test attempts

Results:

| Model | Held-out test solves |
|---|---:|
| Static KAN | 48/48 |
| HyperKAN | 48/48 |

Conclusion: the scoped benchmark is alive beyond the tiny smoke run, but guided `A3+B1` alone is still too clean to separate Static KAN and HyperKAN. The next benchmark needs more scoped families and stricter verification, not just more rows from the same guided family.

## One Guided Scoped Trajectory

This is a real trajectory from `artifacts/scoped_medium/test.parquet`:

- `trajectory_id = scoped_smoke_10`
- `parameter_key = 4_4_1_2_1_4_5_7`
- family: guided scoped `A3+B1`

<details>
<summary>Show the guided depth-4 scoped trajectory</summary>

```text
distance 4
(4/(x + 2) + 4/(x + 1))*(sin(y)**2 + cos(y)**2)
  + (z**2 + 12*z + 35)/(((z + 1)*(z + 4)*(z + 5)))

--[expr@1::factor]-->

distance 3
(4/(x + 2) + 4/(x + 1))*(sin(y)**2 + cos(y)**2)
  + (z + 7)/(((z + 1)*(z + 4)))

--[expr@0::trigsimp]-->

distance 2
(4/(x + 2) + 4/(x + 1))
  + (z + 7)/(((z + 1)*(z + 4)))

--[add_slice@root[0:2]::together]-->

distance 1
4*(2*x + 3)/((x + 1)*(x + 2))
  + (z + 7)/(((z + 1)*(z + 4)))

--[expr@1::expand]-->

distance 0 / goal
(8*x/(x**2 + 3*x + 2) + 12/(x**2 + 3*x + 2))
  + (z + 7)/(((z + 1)*(z + 4)))
```

</details>

This illustrates the scoped benchmark’s main semantic change: the model predicts both the site and the operation.

## What Is Proven vs Not Yet Proven

Proven:

- The original global benchmark works end to end.
- Static KAN beats MLP on the global benchmark.
- Recovered HyperKAN can nearly match Static KAN on the global benchmark.
- Verified search metrics can diverge from supervised validation loss.
- Scoped action infrastructure works end to end: dataset generation, training, checkpointing, and scoped beam eval.
- Guided scoped depth-4 trajectories can be trained and solved.

Not yet proven:

- Robust strict depth-4+ density under scoped exact verification.
- Shortest-action multi-label tie recovery for composed scoped cases.
- A scoped benchmark that cleanly separates Static KAN and HyperKAN.
- That guided `A3+B1` performance transfers to less templated or stricter scoped families.

## Next Steps

- Add more scoped families, not just more rows.
- Restore strict composed verification.
- Recover shortest-action ties for accepted scoped rows.
- Use structurally harder held-out splits.
- Compare Static KAN vs recovered HyperKAN again on a harder scoped benchmark.

## Reproduction

All ROCm training/eval commands should run inside the toolbox:

```bash
toolbox run -c llama-rocm-7.2 bash -c 'cd /home/marcel/Work/Mathy && source scripts/toolbox_env.sh && <command>'
```

Build guided scoped datasets:

```bash
python3 scripts/build_scoped_smoke_dataset.py \
  --samples 24 \
  --split-mode random \
  --output-dir artifacts/scoped_smoke

python3 scripts/build_scoped_smoke_dataset.py \
  --samples 200 \
  --split-mode heldout_coeff \
  --output-dir artifacts/scoped_medium
```

Train scoped models:

```bash
python3 -m train.run_experiment \
  --config configs/scoped_smoke.yaml \
  --model-type static_kan \
  --output-dir artifacts/scoped_smoke_checkpoints

python3 -m train.run_experiment \
  --config configs/scoped_smoke.yaml \
  --model-type hyperkan \
  --output-dir artifacts/scoped_smoke_checkpoints

python3 -m train.run_experiment \
  --config configs/scoped_medium.yaml \
  --model-type static_kan \
  --output-dir artifacts/scoped_medium_checkpoints

python3 -m train.run_experiment \
  --config configs/scoped_medium.yaml \
  --model-type hyperkan \
  --output-dir artifacts/scoped_medium_checkpoints
```

Run scoped eval:

```bash
python3 -m eval.run_scoped_smoke_eval \
  --dataset artifacts/scoped_smoke/test.parquet \
  --checkpoint artifacts/scoped_smoke_checkpoints/static_kan/best.pt \
  --action-vocab artifacts/scoped_smoke/scoped_action_vocab.json \
  --output artifacts/scoped_smoke_checkpoints/static_kan/scoped_smoke_eval.json \
  --beam-width 4 \
  --max-steps 4

python3 -m eval.run_scoped_smoke_eval \
  --dataset artifacts/scoped_medium/test.parquet \
  --checkpoint artifacts/scoped_medium_checkpoints/hyperkan/best.pt \
  --action-vocab artifacts/scoped_medium/scoped_action_vocab.json \
  --output artifacts/scoped_medium_checkpoints/hyperkan/scoped_medium_eval.json \
  --beam-width 4 \
  --max-steps 4
```

For the original global benchmark and historical plots, see:

- [docs/shallow_benchmark_details.md](docs/shallow_benchmark_details.md)
- [docs/hyperkan_recovery_results.md](docs/hyperkan_recovery_results.md)
- `artifacts/shallow_benchmark_parallel/`
- `results/hyperkan_recovery/`

## Repository Layout

```text
configs/   training configs
data_gen/  symbolic actions, scoped actions, canonicalization, dataset generation
docs/      project notes and historical figures
eval/      verified global eval and scoped smoke/medium eval
models/    BiGRU encoder, MLP, Static KAN, HyperKAN policy heads
results/   graph-mining and recovery summaries
scripts/   dataset builders, graph miners, diagnostics
search/    global and scoped beam search
tokenizer/ structural expression tokenizer
train/     training loop and losses
viz/       historical plotting utilities
```
