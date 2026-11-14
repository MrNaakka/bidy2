Second review round of branch verda-skypilot-evals (full diff: `git diff 7820aaa..HEAD`; changes since the
first round: `git diff 37d2a70~1..HEAD` minus the reviews/ files). Read the code; do not modify files.

Context and requirements are those of round 1, in reviews/r15-skypilot-evals-prompt.md; the round-1 findings are
in reviews/r15-skypilot-evals.md. Changes made since:
- r15 #1: the key reaches curl only through a 0600 header file (`-H @file`), also inside the probe container.
- r15 #2: a background killer stops the whole job after MAX_RUN_HOURS (TERM to children and the job shell, then
  KILL); curl calls have timeouts.
- r15 #3: `sweep_leaks` deletes Verda instances and volumes named `<cluster>-<our SkyPilot user hash>-*` for
  eval clusters SkyPilot does not know; run after failed launches, every `eval-wait` tick (which keeps
  running for 30 minutes after a failed launch), and in `eval-down`. Observed in the smoke test: the fork
  times out on the create call, reports no capacity, and Verda boots the instance anyway.
- r15 #4: `eval-up` refuses a cluster of the same name without a box record in the state file.
- r15 #5: `--new` refuses while any box of the old fleet is live.
- r15 #6: the status rsync must succeed (`--delete`), and a SUCCEEDED job must yield box.json.
- Smoke findings: the vLLM venv uses uv-managed Python (system Python lacks Python.h, which Triton compiles
  against); `eval.yaml` allows any Verda region (instance type still pinned; the region is recorded);
  `eval-up --resume` reruns `sky launch` on existing boxes (after an interrupted eval-up or a failed job).

Check that these fixes are correct and complete, and look for remaining real correctness, safety, cost-leak
or comparability problems (same requirements as round 1). Report each with file:line, why it matters and a
concrete fix. Ignore style and nitpicks, and do not re-report what is fixed. If nothing serious remains, say
"no blocking issues".
