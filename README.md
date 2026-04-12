# Mathy — Goal-conditioned HyperKAN for Verified Algebraic Rewriting

> **Mathy** is the project. The repository is **HyperKan**. Run it from wherever you cloned it — the instructions below use `$REPO` as the root.

A local-first POC on Strix Halo (ROCm 7.2) for learning symbolic rewrite policies using KAN architectures.

## What this is

The system learns to rewrite algebraic expressions to a goal form using a set of 6 SymPy actions (`expand`, `factor`, `cancel`, `apart`, `together`, `trigsimp`). A BiGRU encoder reads both the current state and the goal, a policy head scores the 6 actions, and a value head estimates distance to goal. At inference, beam search explores rewrite paths and verifies the result exactly via SymPy.

Three policy heads are compared:
- **MLP** — standard dense baseline
- **Static KAN** — Kolmogorov-Arnold Network with fixed spline templates
- **HyperKAN** — goal-conditioned KAN where a hypernetwork generates spline mixture weights from the goal embedding

## Headline result

On the full 274-problem verified test set (beam width 4, max 8 steps), **Static KAN** achieves the best solve rate at **59.5% (163/274)**, outperforming **MLP** at **53.6% (147/274)** and **HyperKAN** at **54.0% (148/274)**.

The benchmark is now effectively a **depth-3 benchmark**: depth-2 is saturated (all models solve **147/147** depth-2 problems), and nearly all separation happens at depth-3:
- **MLP:** **0/127** depth-3 solves
- **Static KAN:** **16/127** depth-3 solves
- **HyperKAN:** **1/127** depth-3 solves

HyperKAN still achieves the lowest supervised validation action loss (best `val_action_loss` ≈ **0.0122** vs ≈ **0.0169** for MLP/static), but that advantage does **not** translate into verified search performance. On this benchmark, HyperKAN is currently misaligned with the search objective (or overfitting the supervised labels).

## Phase plan

- **Phase 1 (current):** prove the symbolic environment, baselines, and HyperKAN behavior on Strix Halo
- **Phase 2:** promote to H200/B200 only if the local gate passes
- **Phase 3:** multi-GPU only if single-GPU changes the research scope

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

**What this means:**
- Static KAN outperforms both baselines on the only nontrivial slice of this benchmark (depth-3).
- HyperKAN improves supervised label fit (`val_action_loss`) but not verified search behavior: loss/search mismatch is real here.

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

## Action confusion

![Action confusion](docs/action_confusion.png)

*Predicted vs optimal first action on a 15-problem sample. Small counts — treat patterns as suggestive, not definitive.*

**MLP** — on this sample, often collapses expand/factor decisions and conflates together/trigsimp.

**Static KAN** — directionally cleaner: expand predictions align with expand-optimal cases, together predictions align with together-optimal cases. Some trigsimp confusion remains.

**HyperKAN** — achieves the lowest BCE loss but remains poorly calibrated for first-action prediction on this benchmark. Factor predictions dominate across multiple optimal-action classes, suggesting the model is fitting label correlations rather than the underlying policy.

---

## KAN spline geometry

![Spline comparison](docs/spline_comparison.png)

*Same 8 test examples run through both models.*

**Static KAN** (left) — uses fixed spline templates across all examples by design. That makes it input-invariant in template weights, which is expected by construction. It still produces the best verified solve rate on the current benchmark.

**HyperKAN** (right) — the hypernetwork produces different mixture weights per example. In Layer 1, examples split across two template groups. In Layer 2, individual examples show distinct combinations. This shows active goal-conditioned routing: different goals produce different spline mixture assignments. The routing is sparse (near one-hot rather than smoothly blended) and has not yet translated into better verified solve rates.

### Spline mixtures by optimal action

![Spline by action](docs/spline_by_action.png)

*Very small per-action sample counts — treat as a diagnostic hint, not a robust result.*

Different optimal actions tend to activate different spline templates in HyperKAN: `expand` activates templates 0–1, while `factor` and `together` both lean on template 3. This partial overlap between factor and together in template space is consistent with the confusion seen in the action calibration plot. Whether this reflects genuine action-specific routing structure or is an artifact of the small dataset is not yet clear.

---

## HyperKAN conditional routing

The hypernetwork in HyperKAN generates spline mixture weights from the goal embedding at inference time. The four plots below probe whether this routing is meaningfully goal-conditioned. HyperKAN shows clear goal- and family-dependent routing patterns, but on the current benchmark those routing differences are more visible across goals and families than across steps within a single solved trajectory.

## Why HyperKAN may not win yet

HyperKAN’s extra capacity is real (the routing plots show it), but the current evidence says it is **not translating into verified search performance**. Plausible reasons:
- **Shallow + templated data:** depth-2 is saturated and depth-3 is small (127 rows). Goal-conditioning may simply be unnecessary at this difficulty.
- **Train/search objective mismatch:** the policy is trained with multi-label BCE over all shortest first actions, but inference uses softmax-normalized logits inside beam search. Those imply different decision geometry and calibration pressures.
- **Search scoring sensitivity:** small miscalibration in logits can dominate beam ranking even if BCE improves.
- **Stability/regularization:** the hypernetwork may increase variance and overfit label correlations in a small benchmark, while a static KAN provides a simpler inductive bias that generalizes better here.

### Same state, different goals

![Same state, different goals](docs/hyperkan_same_state_diff_goals.png)

*Fixed current expression, 6 different goal expressions — one row per goal.*

Each row uses the identical state but a different target. Different template assignments are visible across goals, confirming the routing responds to goal identity rather than state alone. The effect is clearest in Layer 1.

### Routing within a motif family

![Family routing](docs/hyperkan_family_routing.png)

*Up to 6 instances per family, 4 families shown.*

Some clustering is visible within family blocks — problems from the same family tend toward similar template combinations rather than pure per-sample chaos. The degree of clustering varies by family.

### Average routing per family

![Average routing per family](docs/hyperkan_family_avg_routing.png)

*Each column = one family; rows = Layer 1 and Layer 2. Averaged over up to 8 examples per family.*

Averaged routing differs across families: some consistently favor one template, others spread weight across two or three. This is the clearest evidence of family-level structure in the spline routing. Sample counts per family are small, so individual family averages may be brittle.

### Routing along a solved trajectory

![Trajectory routing](docs/hyperkan_trajectory_routing.png)

*Routing at each step of one beam-search solution (factor → trigsimp).*

For this solved trajectory, routing stays largely stable across steps — Layer 1 stays on the same template, Layer 2 stays on the same template. This suggests the goal-conditioned configuration may dominate over local state changes on this example. Whether dynamic per-step rerouting occurs on harder, longer trajectories is not established here.

---

## Honest conclusions

| Claim | Status |
|-------|--------|
| End-to-end pipeline works on ROCm 7.2 | ✓ proven |
| Static KAN is the best head on verified solve rate | ✓ on this benchmark (59.5% vs 53.6% / 54.0%) |
| HyperKAN > static KAN on verified solve rate | ✗ not demonstrated |
| HyperKAN learns goal-conditioned spline routing | ✓ visible in spline plots |
| Goal-conditioned routing improves search at depth 3 | ✗ not on current data (1/127) |
| Depth-2 is a useful discriminator | ✗ saturated (147/147 for all models) |
| Depth-3 is the discriminator | ✓ (MLP 0/127, static 16/127, hyper 1/127) |

## Limitations

- Dataset is currently depth 2–3 only; depth 4–6 behavior is unknown
- Motif library is narrow (7 families, polynomial and rational forms only)
- HyperKAN routing is active but sparse; on this benchmark it does not improve depth-3 solving despite improved supervised loss
- The hardest family (`rat_three_partial_trig_to_expanded`) is unsolved by all models, suggesting the benchmark still lacks enough depth pressure

## Next steps

- Promote Static KAN as the baseline head to beat
- Focus dataset work on depth 3+ (depth-2 is saturated)
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
