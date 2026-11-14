One substantive finding:

- **Medium — [scripts/build_tasks.py:323](/home/user/bidy2/scripts/build_tasks.py:323): v1 tasks contaminate v2 evaluation.** The builder reuses `tasks/eval` and replaces only selected item directories. An existing v1 checkout—including this workspace—retains the 20 dialogs/dashboard tasks. Running `scripts/eval.sh` therefore evaluates **50 tasks instead of 30**, changing the measured gap and making results depend on checkout history. **Fix:** remove obsolete generated task directories during a full build, or use a separate v2 directory and update `eval.sh` accordingly. Preserve incremental behavior for `--only`.

No other substantive correctness, fairness, integrity, or reference issues found.

Validation: **334 tests passed**, including reference runs on **200 distinct 64-bit seeds per new kind**. Maximum reference costs were **41,048 pixels for toasts** and **47,648 for leaderboard**. No files modified.