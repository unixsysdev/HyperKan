# Scoped Action Spec

## Why This Exists

The current benchmark uses 6 global macro-actions:

- `expand`
- `factor`
- `cancel`
- `apart`
- `together`
- `trigsimp`

Graph-mining and backward synthesis now show a consistent pattern:

- isolated ladders exist under global semantics
- naive composition does not reliably produce additive depth
- global rewrites often simplify multiple latent subproblems at once

So the next benchmark variant should change **action locality**, not mainly the operator names.

The new action should be:

- `action = (site, op)`

where:

- `site` identifies a specific subexpression
- `op` is still one of the same 6 rewrite families

This is the smallest redesign that can make independent subproblems stay independent.

## Goals

- Preserve the current symbolic story and exact verification workflow.
- Make compositional depth mathematically possible.
- Keep the rest of the pipeline shape as intact as possible.
- Avoid an action-space explosion in the first scoped prototype.

## Non-Goals

- Do not redesign the encoder architecture yet.
- Do not introduce a large catalog of primitive rewrite rules yet.
- Do not remove the current global benchmark; keep it as a baseline.

## Benchmark Split

After this redesign there should be two benchmark modes:

1. `global_actions`
   Existing benchmark with the current 6 whole-expression rewrites.

2. `scoped_actions`
   New benchmark where a rewrite operator is applied only to a selected site.

These are different action spaces. Existing global-action checkpoints remain valid only as baselines for `global_actions`.

## Path-Based Scoped Sites

The next prototype should move from named shallow sites to deterministic subtree paths.

The action should still be:

- `action = (site, op)`

but `site` should now mean:

- a path from the root to a subtree
- plus a view on that subtree: `expr`, `numerator`, or `denominator`

Examples:

- `expr@root`
- `expr@0`
- `expr@0.1`
- `numerator@1`
- `denominator@2.1`
- `add_slice@root[0:2]`

This keeps the action local while avoiding reliance on a fragile top-level block decomposition.

### Site Semantics

For an expression `E`, define candidate sites by recursively enumerating subtree paths from the root.

Version 1 should keep this bounded:

- maximum path depth: `2`
- site types: `expr`, `numerator`, `denominator`, `add_slice`
- subtree whitelist:
  - root
  - `Add`
  - `Mul`
  - rational subtree
  - trig-containing subtree
- add-slice lengths:
  - contiguous `Add` slices of length `2` to `3`

This is the middle ground:

- more expressive than top-level named sites
- still much smaller than arbitrary unrestricted AST surgery
- able to keep acting on a logical additive block after top-level `Add` flattening

## Action Semantics

Each action is a pair:

- `site_id`
- `op_name`

where `op_name` is one of:

- `expand`
- `factor`
- `cancel`
- `apart`
- `together`
- `trigsimp`

### Rewrite Rule

Given expression `E`, site `S`, and operator `op`:

1. Extract the exact subexpression at site `S`.
2. Apply the SymPy rewrite only to that subexpression.
3. Reinsert the rewritten subexpression into `E` without altering the rest of the tree.
4. Reparse and canonicalize the resulting full expression.
5. Accept the action only if:
   - the resulting full expression is structurally different
   - the resulting full expression is algebraically equivalent to `E`

This preserves the current validity discipline while making the action local.

## Recommended Initial Operator Coverage By Site

Not every operator needs to be valid on every site in v1.

Recommended initial matrix:

- `expr@...`
  - all 6 operators

- `numerator@...`
  - `expand`, `factor`, `cancel`, `trigsimp`

- `denominator@...`
  - `expand`, `factor`, `cancel`, `trigsimp`

- `add_slice@...[i:j]`
  - all 6 operators

For the first prototype, it is acceptable to exclude combinations that prove unstable or redundant.

## Canonical Site Enumeration

The site list must be deterministic so dataset labels and search agree.

Recommended ordering:

1. preorder traversal by path
2. at each path: `expr`, then `numerator`, then `denominator` when applicable

Each site should be serialized to a stable string id, for example:

- `expr@root`
- `expr@0`
- `expr@0.1`
- `numerator@1`
- `denominator@2.1`

Then the atomic action id becomes:

- `site_id + "::" + op_name`

## Data Format Changes

The current dataset stores:

- `state_str`
- `goal_str`
- `valid_shortest_actions`
- `distance_to_goal`

For scoped actions, keep the same high-level shape but change the label space:

- `valid_shortest_actions` becomes a multi-hot vector over scoped action ids

Recommended metadata additions:

- `action_vocab`
  - JSON file mapping scoped action ids to indices

- `benchmark_mode`
  - `global_actions` or `scoped_actions`

Optional row-level debugging fields:

- `site_count`
- `valid_scoped_action_ids`

These are optional for training but useful for auditing.

## Search Changes

Beam search stays structurally the same:

- score candidate actions
- apply rewrite
- verify exact target match

What changes is the branching function.

Instead of:

- enumerate 6 global actions

do:

- enumerate candidate sites
  
For exact verification, keep the search aggressively bounded:

- canonical full-expression dedupe
- cached site enumeration and scoped action application
- lazy successor generation so direct-goal successors short-circuit expensive states
- explicit node caps recorded separately from wall-clock timeouts

The current bounded sanity run shows the desired intermediate milestone:

- block A is again tractable at scoped depth `3`
- block B is tractable at scoped depth `2`
- naive composed `A + B` states still time out under exact bounded search
- guided first-path search finds the `A3 + B1` composition at scoped depth `4`

So the live bottleneck is no longer single-block addressability. It is strict composed-state verification and shortest-action tie recovery under the frozen grouped path model.

The smoke-training path now uses guided single-path labels:

- dataset: `artifacts/scoped_smoke`
- action vocabulary: `artifacts/scoped_smoke/scoped_action_vocab.json`
- config: `configs/scoped_smoke.yaml`
- eval: `eval/run_scoped_smoke_eval.py`

This validates scoped policy/eval plumbing but should stay labeled as smoke data until strict composed verification and shortest-action tie recovery are restored.

The medium guided split extends this but does not yet separate models:

- dataset: `artifacts/scoped_medium`
- config: `configs/scoped_medium.yaml`
- split: held-out `(a_coeff_0, a_coeff_1) = (4, 4)` for test
- result: both Static KAN and HyperKAN solve 48/48 non-terminal held-out test rows

So guided `A3 + B1` alone is now a plumbing/regression benchmark, not a model-comparison benchmark.
- enumerate allowed operators for each site
- filter to valid scoped actions

So the policy is effectively ranking:

- `P(site, op | state, goal)`

The simplest implementation is to flatten `(site, op)` into one categorical action space.

## Model Changes

The smallest compatible model change is:

- keep the same encoder
- replace the 6-logit policy head with an `N`-logit head over scoped actions

This means:

- same input format
- same value head idea
- same beam-search structure
- same loss type

Only the policy output dimension changes.

### Checkpoint Compatibility

Global-action checkpoints are **not** reusable as policy checkpoints for the scoped benchmark.

Reason:

- the action vocabulary changes from 6 classes to `N` scoped classes

What remains reusable:

- code structure
- tokenizer format
- dataset row shape at a high level
- evaluation/search pipeline design
- global checkpoints as baseline comparison points on the old benchmark

## Exact Verification Rules

The scoped benchmark should keep the same exactness standard:

- target success is exact structural match to the goal form
- not just algebraic equivalence

This remains essential. If the goal is only semantic equivalence, local depth will still collapse.

## Minimal Implementation Plan

### Phase 1: spec + mechanics

- add site enumeration utility
- add scoped rewrite application utility
- add scoped action vocabulary builder
- add exact validity checks mirroring the current global action pipeline

### Phase 2: local graph sanity checks

- build a miner for scoped families
- verify that depth adds on simple two-block constructions
- compare local graph shape against the current global-action results

### Phase 3: scoped dataset prototype

- generate a small exact verified dataset
- target depth 3 to 5 first
- validate action labels exactly

### Phase 4: scoped model baseline

- retrain MLP and Static KAN baselines on the new action space
- keep HyperKAN out of the critical path until the benchmark itself is working

## Success Criteria

The scoped prototype is worth keeping only if it shows at least one of:

- validated depth-4+ families at nontrivial density
- clear product-like behavior on composed subproblems
- lower shortcut density than the current global benchmark

If scoped actions do not improve any of these, then the next move should be finer primitive rewrites rather than more site engineering.

## Immediate Recommendation

Do one last narrow family-mining pass under global semantics only as a sanity check.

If that still does not produce usable depth-4+ density:

- freeze the global benchmark as the current baseline
- continue the scoped-action implementation
- treat the first path-based scoped version as the next main benchmark track
