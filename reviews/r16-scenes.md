One substantive finding:

- **Medium — [scripts/eval.sh:90](/home/user/bidy2/scripts/eval.sh:90): missing task directory aborts the agents stage.** When `tasks/<split>` does not exist, `[ -d "$TASKS" ]` returns 1. The `HAVE="$(...)"` assignment inherits that status, and `set -e` exits before line 91 can rebuild. Reproduced with the exact shell expression. **Fix:** initialize `HAVE=""` and run the assignment inside an explicit `if [ -d "$TASKS" ]; then ...; fi`. Add a regression check for an absent task directory.

The stale-directory cleanup works as intended in mocked filesystem checks, including preserving other tasks with `--only`. No other substantive correctness, fairness, integrity, or design issues found.

Validation: 332 scene/anti-hack tests passed; reference solved 200/200 cases per new kind, using at most **41,048 pixels** for toasts and **47,648** for leaderboard. Another 1,000 construction checks per kind passed on 64-bit seeds. Build integration tests requiring temporary files were blocked by the read-only filesystem. No files changed.