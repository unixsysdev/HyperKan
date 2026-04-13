# HyperKan: Verified Algebraic Rewriting with KAN Policy Heads

> **Branch status:** This README is preserved for the branch historical experiment state. For the frozen submission artifact, use `release/paper-v1` or tag `paper-v1.0`. Later results supersede parts of the branch-local narrative below: the paper story is the moderate-depth frontier-reranker rescue, the depth-7 failure boundary, and negative learned-frontier/RL frontier-controller diagnostics.


Local-first POC on Strix Halo (ROCm 7.2) for learning verified symbolic rewrite policies. (Project name: “Mathy”, repo: **HyperKan**.)

## What this is

The system learns to rewrite algebraic expressions to a goal form using a set of 6 SymPy actions (`expand`, `factor`, `cancel`, `apart`, `together`, `trigsimp`). A BiGRU encoder reads both the current state and the goal, a policy head scores the 6 actions, and a value head estimates distance to goal. At inference, beam search explores rewrite paths and verifies the result exactly via SymPy.

Three policy heads are compared:
- **MLP** — standard dense baseline
- **Static KAN** — Kolmogorov-Arnold Network with fixed spline templates
- **HyperKAN** — goal-conditioned KAN where a hypernetwork generates spline mixture weights from the goal embedding

## Headline result

On the full 274-problem verified test set (beam width 4, max 8 steps), **Static KAN** achieves the best solve rate at **59.5% (163/274)**, outperforming **MLP** at **53.6% (147/274)** and **HyperKAN** at **54.0% (148/274)**.

Depth-2 is saturated (all models solve **147/147** depth-2 problems). The benchmark is now effectively a **depth-3 benchmark**, and that is where the separation lives.

| Model | Greedy | Beam | Depth-3 beam |
|---|---:|---:|---:|
| MLP | 133/274 (48.5%) | 147/274 (53.6%) | 0/127 (0.0%) |
| Static KAN | 143/274 (52.2%) | 163/274 (59.5%) | 16/127 (12.6%) |
| HyperKAN | 147/274 (53.6%) | 148/274 (54.0%) | 1/127 (0.8%) |

HyperKAN improves supervised validation action loss (best `val_action_loss` ≈ **0.0122** vs ≈ **0.0169** for MLP/static), but that advantage does **not** translate into verified search performance on this benchmark.

## Phase A: HyperKAN Recovery (Done)

Goal: make HyperKAN competitive with Static KAN on verified search *before* betting on it for deeper chains.

Result: reducing HyperKAN’s conditioning capacity closes almost the entire gap.

- Static KAN beam: **163/274**, depth-3: **16/127**
- Recovered HyperKAN (`small_hyper`) beam: **162/274**, depth-3: **15/127**

The recovery sweep tested:
- Conditioning only the last KAN layer (`hyperkan_condition_kan1: false`)
- Smaller hypernetwork (`hyper_hidden_dim: 64`) **(best)**
- Fewer templates (`spline_templates: 2`)
- Softer routing + routing entropy regularization (`mixture_temperature`, `mixture_entropy_weight`)
- Eval-only beam softmax temperature (`policy_temperature`) on the best checkpoint (no improvement; high temperature hurts)

Full table + temperature sweep: [docs/hyperkan_recovery_results.md](docs/hyperkan_recovery_results.md).

For public, committed JSON artifacts (per-variant summaries + training histories), see `results/hyperkan_recovery/` (starting point: [results/README.md](results/README.md)).

## Rigor Check: “Maybe It Just Needed More Training?” (No)

We ran a controlled extension to distinguish “undertrained” vs “misaligned/unstable for search”.

Setup:
- Extend training from ~20 → 50 epochs for:
  - recovered HyperKAN (`small_hyper`, best Phase A checkpoint)
  - baseline HyperKAN (`hyperkan_base`, original)
- Beam-eval on a fixed **80-row depth-3 failure challenge set** derived from `small_hyper`’s depth-3 beam failures at epoch 20.
- Net-check on the full **127-row depth-3 test slice** (the real discriminator).

Results:
- Challenge set (80 rows): at epoch 35, `small_hyper` improved (6/80), but by epoch 50 both models collapsed (0/80).
- Depth-3 netcheck (127 rows, beam):
  - Static KAN: **16/127**
  - `small_hyper` epoch-20: **15/127**
  - `small_hyper` epoch-35: **13/127** (regressed)

Conclusion: supervised validation loss kept improving, but verified beam performance did not. Checkpoint selection must be based on verified search metrics (or search-based early stopping), not val loss alone.

Public summary artifacts: `results/hyperkan_extension_check/`.

---

## First real run — results

**Dataset:** 2206 rows, curated motif families, depth 2–3, random 70/15/15 split after exact shortest-path BFS acceptance.  
**Training:** 20 epochs, batch 128, AdamW lr=1e-3, ROCm 7.2, Strix Halo.  
**Eval:** greedy (beam=1) and beam-search (beam width 4), max 8 steps, verified by SymPy, **full test set (274 non-terminal problems)**.

Metrics and plots below are rendered from `artifacts/shallow_benchmark_parallel/summary.json`.

### Model comparison

![Model comparison](docs/model_comparison.png)

*Full test set, 274 problems.*

Overall, Static KAN solves **59.5% (163/274)** of test problems vs **53.6% (147/274)** for MLP and **54.0% (148/274)** for HyperKAN. The entire separation comes from depth-3: Static KAN is the only model with meaningful depth-3 performance (**16/127**).

Interpretation: Static KAN is currently the best inductive bias for verified solving on the only nontrivial slice of this benchmark (depth-3). HyperKAN fits supervised labels better, but does not improve verified search outcomes here.

### Solve rate by depth

![Solve rate by depth](docs/solve_rate_by_depth.png)

*Full test set depth breakdown (beam search).*

Depth-2 is saturated for all models (147/147). Depth-3 is the discriminator: Static KAN solves 16/127, HyperKAN 1/127, MLP 0/127.

### Solve rate by motif family

![Solve rate by family](docs/solve_rate_by_family.png)

*Full test set family breakdown (beam search).*

Depth-3 hardness concentrates in `rat_three_partial_trig_to_expanded` (0% across all models). Static KAN’s gains come primarily from `rat_partial_trig_to_expanded`, where it solves a meaningful slice that MLP and HyperKAN largely miss.

### Greedy vs beam search

![Greedy vs beam](docs/greedy_vs_beam.png)

Beam search rescues some additional solves for MLP and Static KAN, but almost none for HyperKAN (already near its beam score under greedy on this benchmark).

![Beam rescue](docs/rescue.png)

Static KAN gets the largest beam-rescue delta; HyperKAN gets almost none.

---

## Trajectory examples

### Solved by all models — `rat_partial_trig_to_together` (depth 2)

![Trajectory solved by all](docs/trajectory_solved_all.png)

`(2/(x+1) + 3/(x-2))·(sin²y + cos²y) → (5x-1)/((x-2)(x+1))`. Green path: `trigsimp` drops the trig identity, then `together` combines the fractions. All three models find this path — it is the easy case where they agree.

### Solved ONLY by static_kan — `rat_partial_trig_to_expanded` (depth 3)

![Trajectory static KAN only](docs/trajectory_static_kan_only.png)

`(2/(x+5) + 2/(x+2))·(sin²y + cos²y) → 4x/(x²+7x+10) + 14/(x²+7x+10)`. A 3-step problem requiring trigsimp, then expand, then apart. MLP and HyperKAN do not find it; static KAN does. The green path shows the solution; gray edges are dead branches the beam explored and abandoned.

### Failed by all models — `rat_three_partial_trig_to_expanded` (depth 3)

![Trajectory failed by all](docs/trajectory_failed_all.png)

A 3-partial-fraction rational expansion problem. All models exhaust the beam without reaching the goal within 8 steps. The gray graph shows the space explored — broad but not reaching the target.

---

## Why HyperKAN may not win yet

Baseline HyperKAN underperforms Static KAN on verified search, but Phase A shows that this was largely an architecture/config issue: a smaller hypernetwork can recover to near-parity. Remaining plausible reasons HyperKAN does not *win* yet include shallow/templated data, an objective mismatch between multi-label BCE training and softmax-based search ranking, and goal-conditioning not being necessary at depth 2–3.

For the diagnostic plots (action confusion, spline geometry, routing), see [docs/shallow_benchmark_details.md](docs/shallow_benchmark_details.md).

### One routing sanity check (same state, different goals)

![Same state, different goals](docs/hyperkan_same_state_diff_goals.png)

This confirms the hypernetwork responds to goal identity rather than state alone.

---

## Honest conclusions

| Claim | Status |
|-------|--------|
| End-to-end pipeline works on ROCm 7.2 | ✓ proven |
| Static KAN is the best head on verified solve rate | ✓ on this benchmark (59.5% vs 53.6% / 54.0%) |
| HyperKAN > static KAN on verified solve rate | ✗ not demonstrated |
| HyperKAN learns goal-conditioned spline routing | ✓ visible in spline plots |
| Reduced-capacity HyperKAN can be competitive on verified search | ✓ (`small_hyper`: 162/274; 15/127 depth-3) |
| Search temperature calibration is the missing lever | ✗ (no improvement over `t=1.0`) |
| Depth-2 is a useful discriminator | ✗ saturated (147/147 for all models) |
| Depth-3 is the discriminator | ✓ (MLP 0/127, static 16/127, hyper 1/127) |

## Limitations

- Dataset is currently depth 2–3 only; depth 4–6 behavior is unknown
- Motif library is narrow (7 families, polynomial and rational forms only)
- HyperKAN routing is active but sparse; on this benchmark it does not improve depth-3 solving despite improved supervised loss
- The hardest family (`rat_three_partial_trig_to_expanded`) is unsolved by all models, suggesting the benchmark still lacks enough depth pressure

## Next steps

- Promote Static KAN as the baseline head to beat
- Promote `small_hyper` as the recovered HyperKAN baseline (Phase A complete): see [docs/hyperkan_recovery_results.md](docs/hyperkan_recovery_results.md)
- Focus dataset work on depth 4+ (depth-2 is saturated; depth-3 is now mostly the discriminator)
- Build validated depth-4+ exact-form families (deep-chain synthesis) before spending more complexity budget on HyperKAN
- Scale to 10k–20k rows only after depth-4+ examples exist and the benchmark is no longer effectively “depth-3 only”
- Promote to H200/B200 only if local gains hold and depth scaling confirms the result

---

## Layout

```
configs/        config files (local_poc.yaml, toolbox_smoke.yaml, overfit_test.yaml)
data_gen/       symbolic actions, canonicalization, BFS generation, validation
tokenizer/      SReprTokenizer — structural tokenization and vocabulary
models/         BiGRU encoder, MLP, StaticKAN, HyperKAN policy heads
train/          multi-task loss (action BCE + value Huber + entropy), train loop
search/         beam search with verified SymPy rewriting
eval/           end-to-end evaluation entrypoints
viz/            spline mixture and trajectory graph plots
scripts/        run_viz.py, run_full_viz.py, analyze_actions.py, train_local.sh
artifacts/      generated datasets, checkpoints, logs and plots
docs/           visualization outputs committed for README rendering
```

## Running

```bash
# All commands run inside the ROCm 7.2 toolbox
toolbox run -c llama-rocm-7.2 bash -c 'cd $REPO && source scripts/toolbox_env.sh && <command>'

# Generate data (random split for architecture comparison)
python3 -m data_gen.generate_backward --samples 5000 --seed 17 --workers 6 --split-mode random --output-dir artifacts/generated

# Train
python3 -m train.run_experiment --config configs/local_poc.yaml --model-type mlp
python3 -m train.run_experiment --config configs/local_poc.yaml --model-type static_kan
python3 -m train.run_experiment --config configs/local_poc.yaml --model-type hyperkan

# Eval
python3 -m eval.run_verified_eval --dataset artifacts/generated/test.parquet --checkpoint artifacts/checkpoints/hyperkan/best.pt

# Visualize
PYTHONPATH=. python3 scripts/run_full_viz.py
```
