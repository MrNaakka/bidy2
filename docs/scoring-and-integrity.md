# Scoring and reward integrity

## The reward

The reward is 1.0 only if the first submitted answer matches exactly (ignoring case and
whitespace). Otherwise it is 0. There is no partial credit and no shaping term to farm.
The pixel budget is a hard constraint, not a reward term.

## Why it holds up

- **Separate verifier.** The MCP server runs in an `fh-mcp` sidecar container. After
  the agent finishes, a hook dumps the server's own record (submitted answer,
  correctness, pixels, every call) through a token-protected route. Harbor copies it
  into a separate verifier container with no network, which writes
  `/logs/verifier/reward.txt`. Nothing the agent's container writes is read.
- **No feedback.** `submit` never says whether the answer was right. Only the first
  submission counts, so guessing again doesn't help.
- **Secret instances.** Generators and item ids are public. Each item's seed is
  `HMAC(salt, split:item)` with a secret per-checkout salt (`secrets/eval.salt`,
  created on the first build, git-ignored). A fresh checkout gets new instances of the
  same kinds and difficulty. Without that checkout's salt, answers can't be computed
  offline from the public generators, and the agent never sees the salt.
- **Baselines score 0.** Do-nothing scores 0/40. Stop-early (one cheap look, then a
  guess in the right format) also scores 0/40. Every look-alike is well-formed, so
  getting the format right earns nothing.

## Tests

`tests/test_antihack.py` and `tests/test_integrity.py` cover verifier strictness,
missing or symlinked evidence, token checks and the absence of secrets in the sidecar
image.

More detail (server routes, task layout, item secrecy): [design.md](design.md).
