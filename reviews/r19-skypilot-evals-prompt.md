Fifth review round of branch verda-skypilot-evals (full diff: `git diff 7820aaa..HEAD`; changes since round 4:
`git diff 2e6d87f..HEAD` minus reviews/). Read the code; do not modify files.

Requirements: as in reviews/r15-skypilot-evals-prompt.md. Earlier findings: reviews/r15..r18-skypilot-evals.md.
Changes for round 4:
- r18 #1: the job body moved to infra/verda/eval_job.sh (staged with the tasks; part of the comparability
  hash). eval.yaml's run is `exec timeout --kill-after=120 <MAX_RUN_HOURS> bash ~/fh/in/eval_job.sh`. GNU
  timeout puts the job in its own process group and signals the whole group, so descendants cannot outlive
  the limit (checked locally: a grandchild holding stdout is killed and the pipe closes at the limit).
  Harbor's own deadline is 5 minutes earlier, so it normally stops gracefully first. On a normal exit the
  job's trap kills vLLM and socat.
- r18 #2 (key in `docker compose exec -e` arguments inside Harbor 0.25.0): accepted and documented. Harbor is
  the organizers' harness and we do not patch it; the box is single-user root, trial containers have their
  own PID namespace and cannot see host processes; local Harbor runs behave the same.

Check these and look for remaining real correctness, safety, cost-leak or comparability problems. Report each
with file:line, why it matters and a concrete fix. Ignore style and nitpicks, do not re-report fixed or
explicitly accepted items (r16 #2 residual risk, r18 #2). If nothing serious remains, say "no blocking issues".
