One substantive finding:

- **P2 — [framehunt/scenes/toasts.py:152](/home/user/bidy2/framehunt/scenes/toasts.py:152): successor promotion can crash scene construction.** A timeline containing exactly `MAX_ERRORS` (99) passes the initial filter. Promoting its non-error successor produces 100 errors and raises `RuntimeError` instead of retrying. I reproduced this with a controlled RNG boundary case; I did not search for an affected seed. **Fix:** skip promotion at the cap, or perform promotion and cap validation inside the timeline retry loop.

Otherwise, no substantive fairness, tick-arithmetic, visibility, or reward-hacking issues found. Exact-frame access makes the shorter windows defensible, and the inspected half-scale endpoint column was readable for 39,100 pixels, leaving sufficient budget for search and answer reading.

Four targeted window tests passed across two seeds per kind, using `nice -n 19`. No files modified.