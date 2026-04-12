# Mathy — Goal-conditioned HyperKAN for Verified Algebraic Rewriting

A local-first POC on Strix Halo (ROCm 7.2) for learning symbolic rewrite policies using KAN architectures.

## What this is

The system learns to rewrite algebraic expressions to a goal form using a set of 6 SymPy actions (`expand`, `factor`, `cancel`, `apart`, `together`, `trigsimp`). A BiGRU encoder reads both the current state and the goal, a policy head scores the 6 actions, and a value head estimates distance to goal. At inference, beam search explores rewrite paths and verifies the result exactly via SymPy.

Three policy heads are compared:
- **MLP** — standard dense baseline
- **Static KAN** — Kolmogorov-Arnold Network with fixed spline templates
- **HyperKAN** — goal-conditioned KAN where a hypernetwork generates spline mixture weights from the goal embedding

## Phase plan

- **Phase 1 (current):** prove the symbolic environment, baselines, and HyperKAN behavior on Strix Halo
- **Phase 2:** promote to H200/B200 only if the local gate passes
- **Phase 3:** multi-GPU only if single-GPU changes the research scope

---

## First real run — results

**Dataset:** 2206 rows, curated motif families, depth 2–3, random 70/15/15 split after exact shortest-path BFS acceptance.  
**Training:** 20 epochs, batch 128, AdamW lr=1e-3, ROCm 7.2, Strix Halo.  
**Eval:** beam-search width 4, max 8 steps, verified by SymPy.

### Model comparison

![Model comparison](docs/model_comparison.png)

**The headline result:** Static KAN solves **59.5%** of test problems vs 53.6% for MLP and HyperKAN. Static KAN also uses more steps (2.28 vs 2.00), meaning it finds harder paths MLP misses. HyperKAN achieves the lowest supervised loss (0.012 BCE) but does not convert that to verified solve rate.

**What this means:**
- Static KAN outperforms MLP on end-to-end verified search — a real architecture result
- HyperKAN improves label fit but not search behavior — a useful negative, not a failure
- The goal-conditioned routing in HyperKAN is not yet helping at depth 2–3; likely needs richer/deeper data to matter

### Solve rate by depth

![Solve rate by depth](docs/solve_rate_by_depth.png)

All models solve depth-2 problems perfectly. The gap opens at depth 3 — only static KAN manages to solve any depth-3 problems in this small subset. MLP and HyperKAN collapse to 0% on depth-3 in the 15-row sample. This is where the architecture difference actually lives.

### Solve rate by motif family

![Solve rate by family](docs/solve_rate_by_family.png)

All models handle `poly_trig_to_factored`, `rat_partial_trig_to_together`, and `rat_three_partial_trig_to_together` well. The hard family is `rat_three_partial_trig_to_expanded` — zero solve rate across all models. This is a depth-3 multi-step rational expansion problem; the current dataset doesn't have enough of these for training.

---

## Trajectory examples

### Solved by all models — `rat_partial_trig_to_together` (depth 2)

![Trajectory solved by all](docs/trajectory_solved_all.png)

`(2/(x+1) + 3/(x-2))*(sin²y + cos²y) → (5x-1)/((x-2)(x+1))`. Beam search finds the path via `trigsimp` (drop the identity) then `together` (combine fractions). All three models agree — this is the easy case.

### Solved ONLY by static_kan — `rat_partial_trig_to_expanded` (depth 3)

![Trajectory static KAN only](docs/trajectory_static_kan_only.png)

`(2/(x+5) + 2/(x+2))*(sin²y + cos²y) → 4x/(x²+7x+10) + 14/(x²+7x+10)`. A 3-step problem: drop the trig identity, combine fractions, then expand the numerator. MLP and HyperKAN fail — static KAN finds it. The beam explores `expand`, `factor`, `trigsimp`, `together`, `apart`, `cancel` paths before landing on the correct sequence.

### Failed by all models — `rat_three_partial_trig_to_expanded` (depth 3)

![Trajectory failed by all](docs/trajectory_failed_all.png)

A 3-partial-fraction rational expansion. The beam explores many paths but none reach the goal within 8 steps. The problem requires coordinating 3 symbolic steps on a more complex expression — the current models and dataset depth aren't enough.

---

## Action confusion

![Action confusion](docs/action_confusion.png)

**MLP** — confused: predicts `factor` for both `expand` and `factor` optimal actions, predicts `together` for `together` and `trigsimp`. Almost never picks `expand` or `trigsimp` correctly.

**Static KAN** — cleaner: mostly predicts `expand` for `expand`, `together` for `together`. Some `trigsimp` confusion but directionally right.

**HyperKAN** — most confused despite lowest BCE loss: `factor` dominates predictions across multiple optimal classes. Lower loss but worse action calibration — the model is fitting label correlations rather than the underlying policy.

---

## KAN spline geometry

![Spline comparison](docs/spline_comparison.png)

**Static KAN** (left) — uniform purple across all examples and both layers. The model uses a single effective template (all weight on one template, same for everyone). This is stable but not expressive.

**HyperKAN** (right) — the hypernetwork produces genuinely different mixture weights per example. In Layer 1, examples cluster into two groups (templates 1–2 vs template 3). In Layer 2, one example (row 2) uses a distinct template combination. This is the goal-conditioning working as intended — different goals produce different spline geometries. The problem is that this expressiveness isn't yet translating to better solve rates.

### Spline mixtures by optimal action

![Spline by action](docs/spline_by_action.png)

Different optimal actions produce different spline template activations in HyperKAN: `expand` lights up templates 0–1, `factor` lights up template 3, `together` also activates template 3. The fact that `factor` and `together` share a template partially explains why those actions get confused in the action confusion matrix.

---

## Honest conclusions

| Claim | Status |
|-------|--------|
| End-to-end pipeline works on ROCm 7.2 | ✓ proven |
| Static KAN > MLP on verified solve rate | ✓ real result (59.5% vs 53.6%) |
| HyperKAN > static KAN on verified solve rate | ✗ not yet |
| HyperKAN learns goal-conditioned spline geometry | ✓ visible in spline plots |
| Goal-conditioning improves search behavior at depth 2–3 | ✗ needs deeper data |

The depth-3 gap is where the interesting comparison lives. Scaling the motif library to depth 4–6 and 10–20k rows is the next gate before any architecture conclusion can be made about HyperKAN.

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
```

## Running

```bash
# All commands run inside the ROCm 7.2 toolbox
toolbox run -c llama-rocm-7.2 bash -c 'cd /path/to/Mathy && source scripts/toolbox_env.sh && <command>'

# Generate data
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
