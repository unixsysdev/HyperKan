# Scoped Structural Results

This note collects the detailed results chain behind the scoped structural probe. The README only keeps the headline conclusions.

## Structural Probe Setup

Artifacts:

- Dataset: `artifacts/scoped_structural_probe/`
- Config: [../configs/scoped_structural_probe.yaml](../configs/scoped_structural_probe.yaml)
- Builder: [../scripts/build_scoped_structural_dataset.py](../scripts/build_scoped_structural_dataset.py)
- Search-selected rerun: `artifacts/scoped_structural_probe_search_checkpoints/`

Families:

- `trig_merge`: `trigsimp -> together -> expand`
- `hidden_cancel`: `numerator factor -> cancel -> expand`
- `apart_normalize`: `denominator factor -> apart`
- `mixed_trig_hidden`: trig block plus hidden-factor block in one expression

Split:

- Train/val families: `trig_merge`, `hidden_cancel`, `apart_normalize`
- Test family: `mixed_trig_hidden`
- Dataset size: `48` trajectories, `204` rows, `12` scoped actions

## Core Structural Probe Result

Loss-selected checkpoints:

| Split | Static KAN beam1 | Static KAN beam4 | HyperKAN beam1 | HyperKAN beam4 |
|---|---:|---:|---:|---:|
| Train | 60/80 | 80/80 | 70/80 | 80/80 |
| Val | 12/16 | 16/16 | 14/16 | 16/16 |
| Test (`mixed_trig_hidden`) | 0/60 | 0/60 | 0/60 | 0/60 |

Interpretation:

- The benchmark is non-saturated.
- Seen single-block families are learnable.
- Beam rescues seen-family performance.
- Both models fail completely on held-out mixed composition.

## Search-Selected Rerun

Search-based checkpoint selection:

- Static KAN selected `epoch_5`
- HyperKAN selected `epoch_2`

Held-out `mixed_trig_hidden` still failed completely:

- Static KAN: `0/60` for beam 1 and beam 4
- HyperKAN: `0/60` for beam 1 and beam 4

Default first-action diagnosis on held-out mixed-family states:

- Static KAN top-1 first action: `expr@root::together`
- HyperKAN top-1 first action: `expr@root::expand`

So the default failure mode is root-collapse, not just weak beam width.

## Inference-Time Localization Rescue

Minimal intervention: penalize whole-expression root actions at inference.

Static KAN:

- root penalty `1.0`: `0/60` greedy, `0/60` beam
- root penalty `2.0`: `0/60` greedy, `0/60` beam

Recovered HyperKAN:

- root penalty `1.0`: `0/60` greedy, `24/60` beam
- root penalty `2.0`: `24/60` greedy, `36/60` beam

This is still the best mixed-family solve-rate condition on the branch.

## Training-Time Localization Fixes

### Root-Avoidance Loss

Config: [../configs/scoped_structural_probe_hyper_local.yaml](../configs/scoped_structural_probe_hyper_local.yaml)

- Search-selected epoch: `epoch_4`
- Seen-family val beam-4:
  - default: `15/16`
  - root penalty `1.0`: `16/16`
  - root penalty `2.0`: `12/16`
- Held-out `mixed_trig_hidden`:
  - default: `0/60` greedy, `0/60` beam
  - root penalty `2.0`: `0/60` greedy, `24/60` beam

Conclusion: weaker than the inference-only baseline.

### Site-First Factorized Head

Config: [../configs/scoped_structural_probe_hyper_sitefirst.yaml](../configs/scoped_structural_probe_hyper_sitefirst.yaml)

- Search-selected epoch: `epoch_5`
- Seen-family val beam-4: `16/16` with mean expansions `25.81`
- Held-out `mixed_trig_hidden`:
  - default: `0/60` greedy, `0/60` beam
  - root penalty `2.0`: `0/60` greedy, `0/60` beam
- Mixed-family first-action diagnosis still collapsed to:
  - step 1: `expr@root::together`
  - step 2: `expr@root::expand`

Conclusion: more efficient on seen families, no compositional gain.

## Conditional Localization Heuristic

Heuristic: only penalize root actions on states matching a simple mixed-signature rule.

- Validation beam-4, recovered HyperKAN, root penalty `2.0`: `16/16`
- Held-out `mixed_trig_hidden`: `0/60` greedy, `0/60` beam

Conclusion: the useful rescue is not captured by this simple conditional rule.

## Early-Frontier Mechanism

Artifacts:

- [early_frontier_hypothesis.md](./early_frontier_hypothesis.md)
- `artifacts/scoped_structural_probe_search_checkpoints/hyperkan/penalty_rescue_analysis_v2.json`

Key finding under recovered HyperKAN with unconditional root penalty `2.0`:

- Default mixed-family top-3 is always:
  - `expr@root::expand`
  - `expr@root::cancel`
  - `expr@root::together`
- Penalized top-1 changes on all 60 rows, but solved and unsolved rows still share the same first penalized action: `expr@0::trigsimp`

The real separator is early hidden-branch access:

- Solved beam rows reaching any hidden-block site within first 3 actions: `36/36`
- Solved beam rows reaching `expr@2::cancel` within first 3 actions: `36/36`
- Unsolved beam rows reaching any hidden-block site within first 3 actions: `12/24`
- Unsolved beam rows reaching `expr@2::cancel` within first 3 actions: `12/24`

So the unconditional rescue works by early frontier reshaping, not by a uniquely correct first local move.

## First Sequencing-Aware Tweak

Inference condition:

- root penalty `2.0`
- early hidden bonus `0.5`
- early hidden bonus window `3`

Recovered HyperKAN:

- val beam-4: `14/16`
- mixed test greedy: `24/60`
- mixed test beam-4: `36/60`
- mixed test mean expansions: `56.38`

Conclusion: no accuracy gain over the unconditional-penalty baseline, but lower mixed-family beam expansions.
