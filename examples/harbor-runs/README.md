# Example Harbor run log

`fh-eval-opencode-run1/` is the Harbor job output of Opus run 1 (`anthropic/claude-opus-5-5`
through OpenCode 1.19.5). It ran on the 30 EVAL items with a 120,000-pixel budget per
item and scored **21/30, mean reward 0.70**. It was produced by
`scripts/eval.sh --stage agents`.

The layout is Harbor's own:

- `result.json`, `config.json`, `job.log`: the job's score summary and config.
- `<item>__<id>/result.json`: one trial's result. `verifier/reward.txt` holds its 0/1 score.
- `<item>__<id>/artifacts/fh-evidence/`: the MCP server's own record, collected from the
  sidecar. `result.json` holds the submitted answer, whether it was correct, pixels used
  and the number of calls. `calls.jsonl` logs every tool call with its arguments, cost
  and remaining budget. `verifier/` holds copies of both.
- `<item>__<id>/agent/trajectory.json` and `agent/opencode.txt`: the agent's messages and
  tool calls.

We left out OpenCode's internal state directory (`agent/opencode/`: a SQLite database,
a git snapshot and a log), about 22 MB in all. It duplicates the trajectory.

The item seeds come from a secret salt in our checkout. A fresh checkout generates a
new salt, so the items a judge's run produces are different instances of the same 30
item ids. They have the same kinds and difficulty, but different videos and answers.
