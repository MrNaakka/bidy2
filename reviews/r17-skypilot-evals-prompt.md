Third review round of branch verda-skypilot-evals (full diff: `git diff 7820aaa..HEAD`; changes since round 2:
`git diff dd90f82..HEAD` minus reviews/). Read the code; do not modify files.

Requirements: as in reviews/r15-skypilot-evals-prompt.md. Earlier findings: reviews/r15-skypilot-evals.md and
reviews/r16-skypilot-evals.md. Changes for round 2:
- r16 #1: the relay probe container runs as root (`--user 0:0`) so it can read the 0600 header file.
- r16 #2: `eval-up` itself now sweeps every 30 s for 5 minutes after a failed launch (in the smoke test the
  leaked instance appeared within about a minute), so a leak no longer depends on `eval-wait`; `eval-wait`
  sweeps for 30 minutes after the failure time. We accept that an instance booting after that, with no
  watcher, is a residual risk (documented in the README): Verda has no provider-side expiry we could set,
  and a janitor service is out of scope for this hackathon tool.
- r16 #3: `--new` sweeps the old fleet's clusters and carries boxes still in the sweep window as `orphans`
  that every `eval-wait` tick keeps sweeping until they expire.
- r16 #4: exhausted fetch retries give state `fetch_failed` with a "results NOT fetched" message, and
  `eval-wait` exits non-zero.

Check these fixes and look for remaining real correctness, safety, cost-leak or comparability problems. Report
each with file:line, why it matters and a concrete fix. Ignore style and nitpicks, do not re-report fixed or
explicitly accepted items. If nothing serious remains, say "no blocking issues".
