# Scoped Depth Expansion Results

This note records the first result on `feature/scoped-depth-expansion`.

## Deeper Held-Out Family

The branch adds a new held-out test family:

- `mixed_trig_hidden_apart`

Guided action chain:

1. `expr@1::trigsimp`
2. `add_slice@root[1:3]::together`
3. `expr@2::expand`
4. `numerator@3::factor`
5. `expr@3::cancel`
6. `denominator@1::factor`
7. `expr@1::apart`

This is a deeper three-block composition built from:

- trig-gated merge
- hidden-factor cancel
- apart-normalize

Artifacts:

- Dataset: `artifacts/scoped_depth_expansion_probe/`
- Builder: [../scripts/build_scoped_structural_dataset.py](../scripts/build_scoped_structural_dataset.py)
- Config: [../configs/scoped_depth_expansion_probe.yaml](../configs/scoped_depth_expansion_probe.yaml)

Dataset summary:

- `48` trajectories
- `228` total rows
- `110` train rows
- `22` val rows
- `96` test rows
- `84` non-terminal held-out test attempts
- `14` scoped actions

Replay sanity check passed on all train, val, and test trajectories.

## Model Run

Model:

- recovered HyperKAN-style settings
- checkpoint family: `artifacts/scoped_depth_expansion_probe_checkpoints/hyperkan/hyperkan/`

Search selection:

- default inference selected `epoch_5`
- root penalty `2.0` selected `epoch_5`
- root penalty `2.0` + frontier reranker selected `epoch_5`

So the depth-expansion comparison isolates inference behavior rather than checkpoint drift.

## Main Result

| Condition | Val beam-4 | Held-out greedy | Held-out beam-4 | Held-out mean expansions |
|---|---:|---:|---:|---:|
| Default | 16/16 | 0/84 | 0/84 | 171.23 |
| Root penalty `2.0` | 16/16 | 0/84 | 18/84 | 144.67 |
| Root penalty `2.0` + frontier reranker | 16/16 | 0/84 | 18/84 | 150.18 |

## Interpretation

This is the first real depth test of the earlier reranker result.

What survived:

- default inference is still dead on the deeper held-out family
- root-penalized inference still helps under beam search
- the gain is now much weaker than on the shallower structural probe

What did not survive:

- the frontier reranker does **not** improve solve rate beyond the root-penalty baseline on this deeper family
- it also does not improve overall held-out mean expansions

Additional signal:

- all three greedy conditions remain `0/84`
- both successful beam conditions have `mean_solved_steps = 1.0`

That strongly suggests the current inference rescue is only carrying over to shallow near-goal states inside the deeper family, not to the full 7-step composition itself.

## Failure Slice

Follow-up path analysis:

- root penalty `2.0`: `18/84`
- root penalty `2.0` + frontier reranker: `18/84`
- both conditions solve only the one-action path `expr@1::apart`
- solved rows by guided distance:
  - distance `1`: `9/12`
  - distance `3`: `9/12`
  - distance `2`, `4`, `5`, `6`, `7`: `0/12` each

Early hidden-branch access:

| Condition | Hidden site within 3 actions | Hidden cancel within 3 actions |
|---|---:|---:|
| Root penalty `2.0` | 0/84 | 0/84 |
| Root penalty `2.0` + frontier reranker | 10/84 | 0/84 |

So the reranker does alter part of the early frontier on failed rows, but it does not reach the hidden-cancel operation early and does not convert any deeper state into a solve. The `18/84` rescue is therefore a shallow shortcut effect inside the held-out family, not a successful traversal of the full depth-7 composition.

## Branch Conclusion So Far

The early-frontier mechanism does not vanish at higher depth, but its benefit appears to collapse sharply:

- on the shallower mixed family, the reranker raised held-out beam solves from `36/60` to `48/60`
- on the deeper family, the reranker no longer improves solve rate over root penalty alone

So the current result points to a depth limit for the present inference-only rescue. The useful follow-up analysis now confirms that the beam is not solving the deeper parts of the 7-step family: the only successful path is the one-action `apart` shortcut, while the hidden-cancel subgoal remains inaccessible early enough to matter.
