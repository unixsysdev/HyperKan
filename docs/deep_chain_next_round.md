# Deep Chain Next Round

Bounded target:
- One family only
- One intended 4-step chain
- No scale-up

Candidate family:
- `rat_partial_expanded`

Known clean chain:
- `trigsimp -> together -> expand`

Next intended 4-step attempt:
- exact-form target first
- append one additional inverse step only if exact shortest distance rises from `3` to `4`

Accept criteria:
- Exact shortest distance increases by exactly `1` at each prefix
- Intended action remains in the optimal action set at each prefix
- No shortcut path appears
- Final target stays in the exact intended form, not just an easier equivalent form

Reject criteria:
- Distance stays flat or drops
- Intended action disappears from the optimal action set
- Verification times out repeatedly
- Target collapses to a shallower equivalent form
