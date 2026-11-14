No substantive findings in `cc9c677` and `54a4f0f`.

- The r19 fix correctly reserves room for one promoted Error. A targeted boundary check confirmed that 99 errors triggers retry, while 98 can become 99 without changing the K-th target.
- Promotion preserves badge, level, visibility and eviction consistency. Two-frame burst gaps remain valid.
- Leaderboard tick arithmetic preserves ≥30-frame alert spacing and the complete `[f, f+4]` value window, with different target rows and values at `f-1` and `f+5`.
- The timing remains defensible: exact frame selection and persistent traces let agents verify the event without real-time pressure. Endpoint names are readable at half scale; that name-column crop costs 39,100 pixels and fits comfortably within the budget.
- I found no new reward shortcut from forcing an Error successor; the answer codes remain independently generated.

Read-only checks, all at `nice -n 19`: the cap-boundary check and one reference run per kind passed, using **39,968 / 47,323 pixels**. No full suite or many-seed loops run.