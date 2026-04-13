# Early Frontier Reranker Plan

Branch: `feature/early-frontier-reranker`

## Thesis

The current scoped structural result suggests that the main bottleneck is not generic localization, but early frontier shaping:

- default failure = root-collapse
- recovered HyperKAN + unconditional root penalty `2.0` = best mixed-family result
- solved and unsolved beam cases share the same first penalized action
- separation happens in the next 1-3 steps through early hidden-branch access

So the next experiment should target **subgoal sequencing / early hidden-branch access**, not another broad architecture change.

## What Stays Fixed

- Keep the current recovered HyperKAN checkpoint family as the base model.
- Keep the scoped structural probe as the main discriminator.
- Keep the unconditional root-penalty baseline as the reference condition:
  - mixed-family greedy: `24/60`
  - mixed-family beam: `36/60`

## What Changes

Add a small learned or rule-light inference-side adjustment for the first 2-3 search steps only.

Candidate form:

- rerank frontier states during the first few expansions
- or add a small learned score for actions / states likely to expose the hidden branch early
- or predict a tiny "which subgoal next?" bias over early beam states

The point is to influence **early branch access**, not to replace the full policy.

## Minimal Success Criteria

This branch is only a success if it does at least one of:

1. Beats `36/60` beam on held-out `mixed_trig_hidden`.
2. Keeps `36/60` beam while reducing expansions without hurting seen-family validation.
3. Reduces dependence on the unconditional root penalty while preserving the mixed-family solve rate.

If it does none of those, stop.

## Diagnostics To Keep

Every experiment on this branch should report:

- held-out mixed-family beam solve rate
- held-out mixed-family mean expansions
- seen-family validation beam solve rate
- early hidden-branch access within the first 3 actions
- early `expr@2::cancel` access within the first 3 actions

## Things Not To Reopen Here

- broad HyperKAN architecture sweeps
- more root-penalty heuristic variants
- more action-order family generation
- strict-verifier plumbing unrelated to the early-frontier mechanism

Those belong to other branches, not this one.
