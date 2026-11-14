Fourth review round of branch verda-skypilot-evals (full diff: `git diff 7820aaa..HEAD`; changes since round 3:
`git diff 01fc8b6..HEAD` minus reviews/). Read the code; do not modify files.

Requirements: as in reviews/r15-skypilot-evals-prompt.md. Earlier findings: reviews/r15..r17-skypilot-evals.md.
Changes for round 3:
- r17 #1: `eval_tick` does nothing with a box in state `launching` (it only notes one older than 30 minutes),
  so a `--resume` launch is never fetched/deleted on its previous job's status.
- r17 #2 (also found in the GPU smoke test, where finished jobs stayed RUNNING): the job-limit helper no
  longer ignores TERM; it signals every child of the job shell except itself; a cleanup trap installed right
  after it kills vLLM, socat, the helper and the helper's sleep on every exit. Checked locally with a reduced
  limit: a hung job is stopped at the limit, a finished job leaves no process behind.
- r17 #3: `scripts/eval.sh --fleet` keeps eval-wait's status, still summarizes, then exits with it.

Check these fixes and look for remaining real correctness, safety, cost-leak or comparability problems. Report
each with file:line, why it matters and a concrete fix. Ignore style and nitpicks, do not re-report fixed or
explicitly accepted items (r16 #2 residual risk). If nothing serious remains, say "no blocking issues".
