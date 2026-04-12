# Deep Chain Next Round

Current implementation target:
- Freeze the known depth-3 trig block as block A: `trigsimp -> together -> expand`
- Compose it with a separate rational cancellation block B in `z`
- Keep the row schema unchanged so existing checkpoints and eval code still load the new dataset

What is implemented now:
- `scripts/build_composite_chain_dataset.py`
- `scripts/mine_family_graphs.py`
- `docs/scoped_action_spec.md`
- `data_gen/scoped_actions.py`
- `scripts/scoped_action_sanity.py`
- Family `composite_a3_b1_cancel`: block A depth 3 plus block B depth 1, targeting verified depth 4
- Family `composite_a3_b2_cancel_expand`: block A depth 3 plus block B depth 2, targeting verified depth 5
- Exact prefix verification on every backward-built prefix
- Local graph mining over typed families, with an initial scan saved at `results/composite_family_mining_initial_scan.json`
- Final bounded global-family pass saved at `results/global_family_final_bounded_scan.json`
- A minimal scoped-action redesign spec for the likely next benchmark track
- Path-based scoped-action mechanics for deterministic subtree sites, `(site, op)` ids, scoped rewrites, and scoped shortest-path search

Acceptance rules:
- Exact shortest distance must equal the prefix length
- Intended action must stay in the optimal first-action set at every prefix
- Candidate is rejected immediately on timeout, shortcut collapse, or intended-action loss

Practical note:
- With the current global SymPy `cancel` action, a true `factor -> cancel` gate is not reliable because `cancel` can factor internally.
- So the honest first step is `3 + 1` and `3 + 2`, not pretending we already have a robust `3 + 3`.
- The initial family scan supports the same conclusion: isolated depth-3 and depth-2 ladders exist, but the naive additive composites mostly time out or fail to verify under current global semantics.
- The final bounded global pass did not change that conclusion in a useful way, so the global-family search line should now be considered closed.
- The first shallow scoped prototype exposed a concrete limitation: top-level named sites lose whole-block addressability once `Add` flattening splits a subproblem into multiple direct terms.
- The current scoped direction is path-based subtree addressing plus grouped `Add` slices, which restores more local control and preserves some logical blocks after flattening, but still needs validation on the known ladders and the simple `A + B` composition.
- After tightening scoped BFS, the grouped path model now re-verifies the known ladders individually:
  - block A verifies at scoped distance 3
  - block B verifies at scoped distance 2
- The strict bounded scoped sanity run still times out on the composed `A + B` cases, so exhaustive composed-state verification remains the bottleneck.
- A guided first-path scoped verifier now finds the target `A3 + B1` path at distance 4 with 4 node expansions. This is enough for a scoped smoke-train candidate, but final multi-label dataset generation still needs strict/tie-recovery verification.
- The scoped smoke-train path is implemented:
  - guided `A3 + B1` dataset in `artifacts/scoped_smoke`
  - 4-action scoped vocabulary in `artifacts/scoped_smoke/scoped_action_vocab.json`
  - scoped smoke config in `configs/scoped_smoke.yaml`
  - scoped beam eval in `eval/run_scoped_smoke_eval.py`
- First smoke results:
  - static KAN trained for 3 epochs and solved 16/16 non-terminal scoped smoke test rows
  - HyperKAN trained for 3 epochs and solved 16/16 non-terminal scoped smoke test rows
- These are smoke-pipeline results only. They validate the scoped action head and guided depth-4 path, not the final strict scoped benchmark.
- Medium guided scoped dataset:
  - generated 200 guided `A3 + B1` trajectories / 1000 rows in `artifacts/scoped_medium`
  - held-out coefficient split: train excludes `(a_coeff_0, a_coeff_1) = (4, 4)`, validation uses `(3, 4)` and `(4, 3)`, test uses `(4, 4)`
  - static KAN trained for 5 epochs and solved 48/48 non-terminal held-out test rows
  - HyperKAN trained for 5 epochs and solved 48/48 non-terminal held-out test rows
- The medium guided split is still too clean to separate Static KAN from HyperKAN. The next benchmark-hardening step should add more scoped families and restore strict composed verification / shortest-action tie recovery.

Compatibility note:
- Existing checkpoints remain usable for zero-shot eval and continued training as long as we keep the same parquet columns, same six actions, and the same tokenizer format.
- That compatibility does not carry over to a scoped-action policy head; the action vocabulary changes, so scoped-action training should assume fresh policy checkpoints.
