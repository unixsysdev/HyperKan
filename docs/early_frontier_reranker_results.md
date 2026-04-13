# Early Frontier Reranker Results

This note records the first completed result on `feature/early-frontier-reranker`.

## Setup

Base model:

- recovered HyperKAN from the scoped structural probe rerun
- checkpoint: `artifacts/scoped_structural_probe_search_checkpoints/hyperkan/best_search.pt`

Reference condition:

- unconditional root-action penalty `2.0`

Reranker condition:

- unconditional root-action penalty `2.0`
- frontier bonus `0.5`
- frontier bonus window `3` steps
- frontier bonus mode `hidden_cancel_access`

Implementation:

- [../search/scoped_beam_search.py](../search/scoped_beam_search.py)
- [../eval/run_scoped_smoke_eval.py](../eval/run_scoped_smoke_eval.py)
- [../scripts/select_scoped_checkpoint.py](../scripts/select_scoped_checkpoint.py)
- [../scripts/analyze_penalty_rescue_paths.py](../scripts/analyze_penalty_rescue_paths.py)

Artifacts:

- `artifacts/scoped_structural_probe_search_checkpoints/hyperkan/val_beam4_root2_frontier_hidden_cancel_05_s3.json`
- `artifacts/scoped_structural_probe_search_checkpoints/hyperkan/test_beam1_root2_frontier_hidden_cancel_05_s3.json`
- `artifacts/scoped_structural_probe_search_checkpoints/hyperkan/test_beam4_root2_frontier_hidden_cancel_05_s3.json`
- `artifacts/scoped_structural_probe_search_checkpoints/hyperkan/frontier_reranker_analysis.json`

## Headline Result

| Condition | Val beam-4 | Mixed greedy | Mixed beam-4 | Mixed beam expansions |
|---|---:|---:|---:|---:|
| Root penalty `2.0` baseline | 16/16 | 24/60 | 36/60 | 66.67 |
| Root penalty `2.0` + frontier reranker | 14/16 | 36/60 | 48/60 | 59.23 |

This is the first condition on the scoped structural probe that beats the previous mixed-family beam baseline of `36/60`.

## Interpretation

- The reranker improves held-out mixed-family performance substantially:
  - greedy rises from `24/60` to `36/60`
  - beam rises from `36/60` to `48/60`
- It also reduces mixed-family beam expansions from `66.67` to `59.23`.
- The tradeoff is worse seen-family validation, from `16/16` to `14/16`.

So this branch satisfies the main success criterion of the plan by improving held-out mixed-family beam solve rate, but it does not dominate the baseline overall.

## Mechanism

The reranker is not just amplifying the earlier hidden-branch story.

From `frontier_reranker_analysis.json`:

- first action changes on all `60` mixed-family test rows
- solved rows split across both blocks:
  - first non-root site `expr@2` on `24` solved rows
  - first non-root site `expr@0` on `12` solved rows
  - first non-root site `expr@1` on `12` solved rows
- solved first-non-root block counts:
  - `hidden`: `24`
  - `trig`: `24`

Most common solved prefixes:

- `expr@2::cancel` on `24` solved rows
- `expr@0::trigsimp -> add_slice@root[0:2]::together -> expr@1::expand` on `12` solved rows
- `expr@1::expand -> expr@2::cancel` on `12` solved rows

Early hidden-branch access within the first 3 actions is still useful, but no longer the whole story:

- solved rows with hidden-site access within 3: `36/48`
- solved rows with hidden-cancel access within 3: `36/48`
- unsolved rows with hidden-site access within 3: `12/12`
- unsolved rows with hidden-cancel access within 3: `12/12`

So the reranker appears to do two things:

- it still unlocks hidden-branch access often enough to matter
- it also creates alternative successful early sequences that are not explained by hidden-branch access alone

That makes this a sequencing result, not just a stronger localization bias.

## Current Branch Conclusion

Best current hierarchy on this branch:

1. Default recovered HyperKAN: dead on held-out mixed composition.
2. Recovered HyperKAN + unconditional root penalty `2.0`: first real rescue, `36/60` beam.
3. Recovered HyperKAN + root penalty `2.0` + frontier reranker: best held-out mixed result so far, `48/60` beam, but worse seen-family validation.

So the next question after this branch is no longer whether early-frontier shaping matters. It does. The next question is how to recover that gain without paying the validation cost.
