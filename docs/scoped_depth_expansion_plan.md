# Scoped Depth Expansion Plan

Branch: `feature/scoped-depth-expansion`

Status:

- First result is in [scoped_depth_expansion_results.md](./scoped_depth_expansion_results.md).
- Current finding: the depth jump weakens the earlier rescue sharply. On the new 7-step held-out family, default inference is `0/84`, root penalty reaches `18/84` under beam, and the frontier reranker does not improve solve rate beyond that.

## Thesis

The early-frontier reranker branch established the first strong compositional inference result on the scoped structural probe:

- recovered HyperKAN + root penalty `2.0` + frontier reranker improved held-out mixed-family performance from `36/60` beam to `48/60` beam
- mixed-family greedy improved from `24/60` to `36/60`
- mixed-family beam expansions dropped from `66.67` to `59.23`

That result is strong enough to freeze. The next question is no longer whether early-frontier shaping can help. It can.

The next question is:

**Does the same inference hierarchy still help when the scoped benchmark gets deeper?**

## What This Branch Should Do

Add at least one deeper scoped family beyond the current structural probe, for example:

- `A3 + B2`
- `A3 + B1 + C1`

where:

- `A3` is a trig-gated merge ladder
- `B2` is a hidden-factor cancel ladder
- `C1` is a distinct one-step normalization block, if needed

The point is to increase real compositional depth, not to add more parameter rows from the same shallow family.

## Fixed Comparison Conditions

Evaluate the same recovered HyperKAN checkpoint family under three official inference conditions:

1. default recovered HyperKAN
2. recovered HyperKAN + root penalty `2.0`
3. recovered HyperKAN + root penalty `2.0` + frontier reranker

If Static KAN is included later, treat it as a secondary comparison, not the first branch goal.

## Metrics To Report

For each condition:

- held-out deeper-family greedy solve rate
- held-out deeper-family beam solve rate
- held-out deeper-family mean expansions
- seen-family validation beam solve rate

If the deeper family still supports the earlier mechanism analysis, also keep:

- hidden-branch access within first 3 actions
- early `expr@2::cancel` access where applicable

## Success Criteria

This branch is useful only if it answers at least one of these:

1. The reranker still improves deeper held-out compositional solve rate.
2. The reranker keeps solve rate flat but reduces deeper-family expansions.
3. The reranker fails on deeper families, which marks a clear depth limit for the current inference mechanism.

Any of those is informative. What is not useful is another branch that only re-confirms shallow behavior.

## Things Not To Reopen Here

- broad architecture sweeps
- generic localization losses
- tiny heuristic sweeps on the existing structural probe
- README polishing unrelated to the deeper-family experiment

This branch is for depth expansion, not for revisiting earlier conclusions.
