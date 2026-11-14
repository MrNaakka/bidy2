# Example Harbor run logs

Full Harbor job outputs, each with its score. The layout is Harbor's own:

- `result.json`, `config.json`, `job.log`: the job's score summary and config.
- `<item>__<id>/result.json`: one trial's result. `verifier/reward.txt` holds its 0/1 score.
- `<item>__<id>/artifacts/fh-evidence/`: the MCP server's own record, collected from the
  sidecar. `result.json` holds the submitted answer, whether it was correct, pixels used
  and the number of calls. `calls.jsonl` logs every tool call with its arguments, cost
  and remaining budget. `verifier/` holds copies of both.
- `<item>__<id>/agent/trajectory.json` and `agent/opencode.txt`: the agent's messages and
  tool calls.

We left out OpenCode's internal state directory (`agent/opencode/`: a SQLite database,
a git snapshot and a log, about 22 MB per job). It duplicates the trajectory.

These logs contain the answers of the instances they ran on. That is safe only because
neither salt is used for our reported results: each is either retired or a throwaway.

## `fh-eval-opencode-run1/`: EVAL v1, Opus run 1 (retired salt)

Opus 5.5 (`anthropic/claude-opus-5-5`, OpenCode 1.19.5) on the **EVAL v1** set: 30 items
(10 each of logstream, dialogs, dashboard), 120,000-pixel budget. Score **21/30, mean
reward 0.70**. It was run under the first EVAL salt, which was retired when EVAL v2 got a
new one, so these are not instances of the current EVAL items.

## `fresh-clone-modal-opus-run1/`: EVAL v2 item ids, fresh-clone salt

Opus 5.5 (OpenCode 1.19.5) run from a fresh clone of this repository with
`scripts/eval.sh --env modal -n 40 --runs 1` on 2026-11-14. The clone generated its own
salt, so these are new instances, not the ones behind our reported results. EVAL still
had 10 toasts items then (since removed), so the folder holds 40 trials; Harbor's
`result.json` reports 28/40 over all of them. On the 30 current EVAL items the score is
**18/30, mean reward 0.600** (logstream 5/10, dialogs 6/10, scandoc 7/10), mean 112k
pixels per item. API cost for the whole 40-item run was $5.72; it took 3 min 50 s on
Modal.
