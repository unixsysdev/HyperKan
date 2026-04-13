# HyperKan: Verified Symbolic Rewrite Search

This repo trains policy/value models for verified symbolic rewrite search. The original global-action benchmark worked end to end and showed **Static KAN > MLP** on verified solve rate; the initial HyperKAN underperformed, then a reduced-capacity recovered HyperKAN nearly matched Static KAN after ablations. The important follow-up result is that the global SymPy action semantics turned out too shallow for robust depth-4+ composition, so the main branch of the project has moved to **scoped actions**: `action = (site, op)`. Scoped smoke, medium, and diverse guided benchmarks validated the pipeline, but they were too clean to separate models by solve rate. The new structural scoped probe is the first benchmark slice with real compositional pressure: seen families are learnable, but both models fail completely on a held-out mixed composition family.

## What This Project Is

The task is goal-directed symbolic rewriting with formal execution:

- A model reads a current expression and a target form.
- The policy predicts rewrite actions.
- SymPy executes the selected rewrite.
- Search checks whether the target form is reached.

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

The recovery branch tested whether the initial HyperKAN underperformance was a capacity/routing issue rather than a fundamental failure. The best variant was the reduced-capacity HyperKAN:

- Variant: `small_hyper`
- Key change: `hyper_hidden_dim = 64`
- Recovered HyperKAN: `162/274` overall, `15/127` on depth-3
- Static KAN: `163/274` overall, `16/127` on depth-3

Search-temperature calibration did not close the remaining gap. Longer training also did not help reliably: epoch-35 and epoch-50 checks showed verified search can regress even while supervised validation loss improves.

Main takeaway: **search-based checkpoint selection matters**. Supervised validation loss is not sufficient for model selection in this setup.

See [docs/hyperkan_recovery_results.md](docs/hyperkan_recovery_results.md) and `results/hyperkan_recovery/` for details.

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
- Strict composed verification is not yet tractable enough for dataset-scale use.
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

Conclusion: the scoped data, model head, training loop, and beam inference path work end to end. This is a smoke milestone, not a final benchmark result.

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

Conclusion: the scoped benchmark is alive beyond the tiny smoke run, but guided `A3+B1` alone is still too clean to separate Static KAN and HyperKAN. The held-out coefficient split tests interpolation within one guided family, not full compositional generalization across scoped families. The next benchmark needs more scoped families and stricter verification, not just more rows from the same guided family.

## Scoped Diverse Guided Benchmark

The diverse guided benchmark adds action-order family diversity and uses a structural held-out-family split. It is still guided single-path data, not strict composed verification.

Artifacts:

- Dataset: `artifacts/scoped_diverse/`
- Config: [configs/scoped_diverse.yaml](configs/scoped_diverse.yaml)
- Checkpoints: `artifacts/scoped_diverse_checkpoints/`

Dataset:

- `400` guided trajectories
- `2000` rows
- `4` action-order families: `b_first_a3_b1`, `trig_b_a2`, `trig_together_b_expand`, `a_first_b_last`
- Split: `heldout_family`
- Train families: `a_first_b_last`, `b_first_a3_b1`
- Validation family: `trig_b_a2`
- Test family: `trig_together_b_expand`
- `400` non-terminal held-out test attempts
- `6` scoped actions

Training did show a harder generalization signal: both models fit train quickly while validation loss rose on the held-out validation family. Search still saturated on the held-out test family:

| Model | Beam 4 held-out test solves | Beam 1 held-out test solves |
|---|---:|---:|
| Static KAN | 400/400 | 400/400 |
| HyperKAN | 400/400 | 400/400 |

Conclusion: adding action-order families is a useful next benchmark step, but these variants are still too templated. The current diverse guided split creates validation-loss pressure but does not yet create solve-rate pressure, even with greedy scoped search.

## Scoped Structural Probe

The structural probe replaces action-order variants with different algebraic mechanisms and holds out a composed family at test time.

Artifacts:

- Dataset: `artifacts/scoped_structural_probe/`
- Config: [configs/scoped_structural_probe.yaml](configs/scoped_structural_probe.yaml)
- Builder: [scripts/build_scoped_structural_dataset.py](scripts/build_scoped_structural_dataset.py)
- Checkpoints/evals: `artifacts/scoped_structural_probe_checkpoints/`

Families:

- `trig_merge`: `trigsimp -> together -> expand`
- `hidden_cancel`: `numerator factor -> cancel -> expand`
- `apart_normalize`: `denominator factor -> apart`
- `mixed_trig_hidden`: trig block plus hidden-factor block in one expression

Split:

- Train/val families: `trig_merge`, `hidden_cancel`, `apart_normalize`
- Test family: `mixed_trig_hidden`
- Dataset size: `48` trajectories, `204` rows, `12` scoped actions

Results from the current loss-selected checkpoints:

| Split | Static KAN beam1 | Static KAN beam4 | HyperKAN beam1 | HyperKAN beam4 |
|---|---:|---:|---:|---:|
| Train | 60/80 | 80/80 | 70/80 | 80/80 |
| Val | 12/16 | 16/16 | 14/16 | 16/16 |
| Test (`mixed_trig_hidden`) | 0/60 | 0/60 | 0/60 | 0/60 |

Interpretation:

- The benchmark is no longer saturated.
- Both models learn the seen single-block structural families.
- Beam rescues seen-family performance.
- Both models fail completely on the held-out mixed composition family.
- This is the first scoped result showing a real compositional generalization gap instead of interpolation success.

The current trainer now saves epoch checkpoints, and the repo now includes a search-based checkpoint selector:

- [scripts/select_scoped_checkpoint.py](scripts/select_scoped_checkpoint.py)

That selector was added after the first structural probe run. A rerun with epoch checkpointing and search-based selection is now also available under `artifacts/scoped_structural_probe_search_checkpoints/`.

Search-selected rerun:

- Static KAN selected `epoch_5`
- HyperKAN selected `epoch_2` instead of the loss-selected `best.pt`
- Search selection did **not** rescue held-out composition.
- Static KAN: `0/60` on `mixed_trig_hidden` for both beam 1 and beam 4
- HyperKAN: `0/60` on `mixed_trig_hidden` for both beam 1 and beam 4

The failure mode is now clearer. On held-out mixed-family states, the selected checkpoints default to whole-expression actions instead of local subgoal actions:

- Static KAN top-1 first action: always `expr@root::together`
- HyperKAN top-1 first action: always `expr@root::expand`

So the current limitation is not just lack of beam width. The learned heuristics are not localizing onto the independent subproblems inside the mixed family.

Targeted mixed-family failure analysis plus one minimal localization intervention are now also saved under `artifacts/scoped_structural_probe_search_checkpoints/*/mixed_failure_analysis.json` using [scripts/analyze_mixed_family_failures.py](scripts/analyze_mixed_family_failures.py).

That analysis confirms:

- Static KAN step-2 top action is always `expr@root::expand`
- HyperKAN step-2 top action is always `expr@root::cancel`
- Both greedy rollouts remain entirely on global whole-expression actions and never switch into local blockwise behavior

Minimal intervention: penalize whole-expression root actions at inference.

- Static KAN does not improve:
- Static KAN, root penalty `1.0`: `0/60` greedy, `0/60` beam
- Static KAN, root penalty `2.0`: `0/60` greedy, `0/60` beam
- HyperKAN does improve:
- HyperKAN, root penalty `1.0`: `0/60` greedy, `24/60` beam
- HyperKAN, root penalty `2.0`: `24/60` greedy, `36/60` beam

Interpretation: HyperKAN’s held-out mixed-family failure is at least partly an inference-time global-action bias. A small localization bias at search time unlocks nontrivial mixed-family solves. Static KAN does not show the same behavior.

Localization-aware inference is now wired directly into the main scoped eval path:

- [search/scoped_beam_search.py](search/scoped_beam_search.py)
- [eval/run_scoped_smoke_eval.py](eval/run_scoped_smoke_eval.py)
- [scripts/select_scoped_checkpoint.py](scripts/select_scoped_checkpoint.py)

Training-time localization fix (negative result):

- Added a small `expr@root` avoidance loss and retrained recovered HyperKAN with [configs/scoped_structural_probe_hyper_local.yaml](configs/scoped_structural_probe_hyper_local.yaml).
- Search-based selection picked `epoch_4` from `artifacts/scoped_structural_probe_localized_checkpoints/hyperkan/`.
- Seen-family validation under beam 4:
- default: `15/16`
- root penalty `1.0`: `16/16`
- root penalty `2.0`: `12/16`
- Held-out `mixed_trig_hidden`:
- default: `0/60` greedy, `0/60` beam
- root penalty `1.0`: `0/60` beam
- root penalty `2.0`: `0/60` greedy, `24/60` beam

Compared with the baseline recovered HyperKAN, this is worse: the baseline model reached `36/60` on held-out mixed composition under beam with root penalty `2.0`. So the first training-time localization loss does not improve default compositional generalization, and it weakens the stronger inference-only rescue rather than replacing it.

Second training-time localization fix (site-first factorized head, also negative):

- Added an optional factorized HyperKAN action head that scores each flat scoped action as `site_logit(site) + op_logit(op)`, with auxiliary site/op losses derived from the existing shortest-action labels.
- Config: [configs/scoped_structural_probe_hyper_sitefirst.yaml](configs/scoped_structural_probe_hyper_sitefirst.yaml)
- Artifacts: `artifacts/scoped_structural_probe_sitefirst_checkpoints/hyperkan/`
- Search-based selection picked `epoch_5`.
- Seen-family validation under beam 4 improved to `16/16` with much lower mean expansions (`25.81` vs `51.63` for the previous recovered HyperKAN run).
- Held-out `mixed_trig_hidden` stayed completely unsolved:
- default: `0/60` greedy, `0/60` beam
- root penalty `2.0`: `0/60` greedy, `0/60` beam
- First-action diagnosis on the mixed family did not change:
- step 1 top action: always `expr@root::together`
- step 2 top action: always `expr@root::expand`

Interpretation: the factorized site/op head improved efficiency on seen families but did not fix the held-out compositional failure or even alter the root-action bias on the mixed family. So the next intervention should not be "more site-first loss of the same kind"; it needs a stronger mechanism for localized subgoal choice or subgoal switching.

State-conditional localization inference (also negative):

- Added an official conditional root-penalty mode in scoped search/eval that only downweights root actions on states matching a simple mixed-signature heuristic: a root `Add` with both a trig addend and a separate factorized rational addend.
- This preserves seen-family behavior for recovered HyperKAN:
- validation beam 4, root penalty `2.0`, `mixed_signatures` mode: `16/16`
- mean expansions: `29.31`
- But it does **not** reproduce the held-out mixed-family rescue:
- held-out `mixed_trig_hidden`, root penalty `2.0`, `mixed_signatures` mode: `0/60` greedy, `0/60` beam

So the best mixed-family result is still the simpler always-on inference penalty on recovered HyperKAN (`24/60` greedy, `36/60` beam at penalty `2.0`). This conditional heuristic is too weak: it avoids hurting seen families, but it does not trigger the compositional rescue.

## One Guided Scoped Trajectory

<details>
<summary>Guided depth-4 A3+B1 trajectory from artifacts/scoped_medium/test.parquet</summary>

Metadata:

- `trajectory_id = scoped_smoke_10`
- `parameter_key = 4_4_1_2_1_4_5_7`
- family: guided scoped `A3+B1`

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
- A held-out-family guided scoped split can be built, trained, and evaluated.
- The structural scoped probe is non-saturated and exposes a real held-out composition gap.
- Root-penalized localization-aware inference is a meaningful official eval condition on the scoped structural probe.
- A factorized site/op HyperKAN head improves seen-family efficiency but still fails `0/60` on held-out mixed composition and preserves the same root-action bias.
- A simple state-conditional localization heuristic preserves seen-family behavior but still fails `0/60` on held-out mixed composition.

Not yet proven:

- Robust strict depth-4+ density under scoped verification.
- Shortest-action multi-label tie recovery for composed scoped cases.
- A search-selected scoped benchmark that cleanly separates Static KAN and HyperKAN on held-out composition.
- That guided action-order variants are hard enough to test compositional scoped reasoning.
- That scoped performance transfers to held-out mixed composition once search-based selection is used.
- That a simple training-time `expr@root` avoidance loss can replace inference-time localization bias.
- That a basic factorized site/op head is sufficient to teach mixed-family localization or compositional subgoal choice.
- That a simple mixed-signature conditional inference rule can replace the stronger always-on localization bias.

## Next Steps

- Use search-based checkpoint selection on the structural probe.
- Run failure analysis on held-out `mixed_trig_hidden` trajectories.
- Test localization-biased search or site-first decoding as the next minimal intervention.
- Replace the failed root-avoidance auxiliary loss with a stronger localization mechanism, likely site-first decoding or a more explicit site-selection objective.
- Move past simple site-first factorization and test a mechanism that can change subgoal sequencing, not just the action parametrization.
- Move past simple conditional penalties and test an explicit subgoal-progress signal or another mechanism that can change the default ranking on mixed compositions.
- Add more structurally different scoped families, not just action-order variants or more rows.
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

python3 scripts/build_scoped_smoke_dataset.py \
  --samples 400 \
  --split-mode heldout_family \
  --families b_first_a3_b1 trig_b_a2 trig_together_b_expand a_first_b_last \
  --output-dir artifacts/scoped_diverse

python3 scripts/build_scoped_structural_dataset.py \
  --samples 48 \
  --split-mode heldout_test_family \
  --families trig_merge hidden_cancel apart_normalize mixed_trig_hidden \
  --output-dir artifacts/scoped_structural_probe
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

python3 -m train.run_experiment \
  --config configs/scoped_diverse.yaml \
  --model-type static_kan \
  --output-dir artifacts/scoped_diverse_checkpoints

python3 -m train.run_experiment \
  --config configs/scoped_diverse.yaml \
  --model-type hyperkan \
  --output-dir artifacts/scoped_diverse_checkpoints

python3 -m train.run_experiment \
  --config configs/scoped_structural_probe.yaml \
  --model-type static_kan \
  --output-dir artifacts/scoped_structural_probe_checkpoints

python3 -m train.run_experiment \
  --config configs/scoped_structural_probe.yaml \
  --model-type hyperkan \
  --output-dir artifacts/scoped_structural_probe_checkpoints

python3 -m train.run_experiment \
  --config configs/scoped_structural_probe_hyper_local.yaml \
  --model-type hyperkan \
  --output-dir artifacts/scoped_structural_probe_localized_checkpoints

python3 -m train.run_experiment \
  --config configs/scoped_structural_probe_hyper_sitefirst.yaml \
  --model-type hyperkan \
  --output-dir artifacts/scoped_structural_probe_sitefirst_checkpoints
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

python3 -m eval.run_scoped_smoke_eval \
  --dataset artifacts/scoped_diverse/test.parquet \
  --checkpoint artifacts/scoped_diverse_checkpoints/static_kan/best.pt \
  --action-vocab artifacts/scoped_diverse/scoped_action_vocab.json \
  --output artifacts/scoped_diverse_checkpoints/static_kan/scoped_diverse_eval.json \
  --beam-width 4 \
  --max-steps 4

python3 -m eval.run_scoped_smoke_eval \
  --dataset artifacts/scoped_diverse/test.parquet \
  --checkpoint artifacts/scoped_diverse_checkpoints/hyperkan/best.pt \
  --action-vocab artifacts/scoped_diverse/scoped_action_vocab.json \
  --output artifacts/scoped_diverse_checkpoints/hyperkan/scoped_diverse_eval.json \
  --beam-width 4 \
  --max-steps 4

python3 -m eval.run_scoped_smoke_eval \
  --dataset artifacts/scoped_structural_probe/test.parquet \
  --checkpoint artifacts/scoped_structural_probe_checkpoints/static_kan/best.pt \
  --action-vocab artifacts/scoped_structural_probe/scoped_action_vocab.json \
  --output artifacts/scoped_structural_probe_checkpoints/static_kan/test_beam4.json \
  --beam-width 4 \
  --max-steps 5

python3 -m eval.run_scoped_smoke_eval \
  --dataset artifacts/scoped_structural_probe/test.parquet \
  --checkpoint artifacts/scoped_structural_probe_checkpoints/hyperkan/best.pt \
  --action-vocab artifacts/scoped_structural_probe/scoped_action_vocab.json \
  --output artifacts/scoped_structural_probe_checkpoints/hyperkan/test_beam4.json \
  --beam-width 4 \
  --max-steps 5

python3 -m eval.run_scoped_smoke_eval \
  --dataset artifacts/scoped_structural_probe/test.parquet \
  --checkpoint artifacts/scoped_structural_probe_search_checkpoints/hyperkan/best_search.pt \
  --action-vocab artifacts/scoped_structural_probe/scoped_action_vocab.json \
  --output artifacts/scoped_structural_probe_search_checkpoints/hyperkan/test_beam4_root2.json \
  --beam-width 4 \
  --max-steps 5 \
  --root-action-penalty 2.0
```

Select checkpoints by scoped search metric:

```bash
python3 scripts/select_scoped_checkpoint.py \
  --checkpoint-dir artifacts/scoped_structural_probe_checkpoints/static_kan \
  --dataset artifacts/scoped_structural_probe/val.parquet \
  --action-vocab artifacts/scoped_structural_probe/scoped_action_vocab.json \
  --beam-width 4 \
  --max-steps 5 \
  --output artifacts/scoped_structural_probe_checkpoints/static_kan/search_selection.json \
  --copy-best-to artifacts/scoped_structural_probe_checkpoints/static_kan/best_search.pt

python3 scripts/select_scoped_checkpoint.py \
  --checkpoint-dir artifacts/scoped_structural_probe_localized_checkpoints/hyperkan \
  --dataset artifacts/scoped_structural_probe/val.parquet \
  --action-vocab artifacts/scoped_structural_probe/scoped_action_vocab.json \
  --beam-width 4 \
  --max-steps 5 \
  --include-best-pt \
  --output artifacts/scoped_structural_probe_localized_checkpoints/hyperkan/search_selection_val_beam4.json \
  --copy-best-to artifacts/scoped_structural_probe_localized_checkpoints/hyperkan/best_search.pt

python3 scripts/select_scoped_checkpoint.py \
  --checkpoint-dir artifacts/scoped_structural_probe_sitefirst_checkpoints/hyperkan \
  --dataset artifacts/scoped_structural_probe/val.parquet \
  --action-vocab artifacts/scoped_structural_probe/scoped_action_vocab.json \
  --beam-width 4 \
  --max-steps 5 \
  --include-best-pt \
  --output artifacts/scoped_structural_probe_sitefirst_checkpoints/hyperkan/search_selection_val_beam4.json \
  --copy-best-to artifacts/scoped_structural_probe_sitefirst_checkpoints/hyperkan/best_search.pt
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
