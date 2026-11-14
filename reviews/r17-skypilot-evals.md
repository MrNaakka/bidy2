Three issues remain:

1. **[P1] Watchdog can delete a box during `--resume` — `infra/verda/fh.py:1037–1046`.** `eval-up` records `launching` and releases the lock before submitting the new job. For an existing UP box, `eval_tick` immediately reads the previous job’s terminal status, fetches its results, and deletes the box. I reproduced this interleaving. **Fix:** skip completion processing until submission is confirmed, persist the new job ID, and monitor that specific job.

2. **[P2] Timeout helper survives successful completion — `infra/verda/eval.yaml:55–56,109`.** The helper ignores TERM, but the EXIT trap uses TERM to cancel it. Its sleep also retains the job’s log descriptors. A shortened reproduction kept the log pipe open until the deadline; the pinned SkyPilot runner waits for those streams before reporting completion. Finished evaluations can therefore keep billing until `MAX_RUN_HOURS`. **Fix:** use an outer `timeout --kill-after` around the complete job, or terminate and reap both the watchdog and its sleep on every exit.

3. **[P2] Fleet wrapper masks evaluation failures — `scripts/eval.sh:108`.** `eval-wait ... || echo ...` consumes the new nonzero failure status. The script then summarizes and exits successfully, potentially including stale or partial results after failed retrieval. I reproduced exit status 0 when the watcher returns 1. **Fix:** retain the watcher’s exit status, optionally summarize for diagnostics, then exit nonzero when any run failed.

No files were modified.