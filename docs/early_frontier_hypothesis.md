# Early Frontier Hypothesis

This note records the current best interpretation of the scoped mixed-family result on the held-out `mixed_trig_hidden` probe.

## Main Hypothesis

- Default failure: root-collapse.
- Penalty rescue: early frontier reshaping.
- Solved and unsolved rows share the same first penalized action.
- Separation happens in the next 1-3 steps through hidden-branch access.

This is the strongest current explanation for why recovered HyperKAN improves from `0/60` to `36/60` on held-out mixed composition under beam search with an unconditional root penalty of `2.0`.

## Evidence

Baseline recovered HyperKAN on held-out `mixed_trig_hidden`:

- default greedy: `0/60`
- default beam-4: `0/60`
- unconditional root penalty `2.0`, greedy: `24/60`
- unconditional root penalty `2.0`, beam-4: `36/60`

From `artifacts/scoped_structural_probe_search_checkpoints/hyperkan/penalty_rescue_analysis_v2.json`:

- Default top-3 on every mixed-family row:
  - `expr@root::expand`
  - `expr@root::cancel`
  - `expr@root::together`
- Under unconditional root penalty `2.0`, the first action changes on all 60 rows, and it changes both site and op every time.
- But solved and unsolved rows still share the same penalized top-1 action: `expr@0::trigsimp`.

So the rescue is not explained by first-step localization alone.

## Early Hidden-Branch Metric

The useful discriminator is early hidden-branch access:

- Solved beam rows reaching any hidden-block site within the first 3 actions: `36/36`
- Solved beam rows reaching `expr@2::cancel` within the first 3 actions: `36/36`
- Unsolved beam rows reaching any hidden-block site within the first 3 actions: `12/24`
- Unsolved beam rows reaching `expr@2::cancel` within the first 3 actions: `12/24`

This suggests the bottleneck is not merely selecting a local site. It is getting beam search onto the hidden branch early enough.

## First Sequencing-Aware Tweak

A simple inference-time bonus for hidden-branch actions in the first 3 steps was tested on top of the unconditional root penalty:

- config: no retraining, recovered HyperKAN `best_search.pt`
- search condition:
  - `root_action_penalty = 2.0`
  - `early_hidden_bonus = 0.5`
  - `early_hidden_bonus_steps = 3`

Result:

- validation beam-4: `14/16` with mean expansions `57.06`
- held-out mixed greedy: `24/60` with mean expansions `24.8`
- held-out mixed beam-4: `36/60` with mean expansions `56.38`

Interpretation:

- This first sequencing bonus does not improve the held-out mixed solve count beyond the current best `36/60`.
- It does reduce beam expansions on the mixed family relative to the unconditional-penalty baseline (`56.38` vs `66.67`).
- It hurts seen-family validation, so it is not yet a better overall eval condition.

## Current Conclusion

- The unconditional root penalty remains the best mixed-family rescue condition.
- Early hidden-branch access is now an explicit, measured mechanism.
- The next useful intervention should target subgoal sequencing / early hidden-branch access more precisely, rather than another generic localization loss.
