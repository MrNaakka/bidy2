You are reviewing a change on branch verda-skypilot-evals (diff: `git diff 7820aaa..HEAD`). Read the code; do not modify files.

## What changed
Parallel Qwen3.8-27B evaluation for FrameHunt on Verda via our SkyPilot fork:
- `infra/verda/eval.yaml` (new): one SkyPilot task = one eval run per box. The box installs docker.io,
  docker-compose-v2, docker-buildx, socat, rsync, uv and Harbor 0.25.0, builds a vLLM 0.33.0 venv, downloads
  Qwen/Qwen3.8-27B at a pinned revision from Hugging Face to its boot disk and verifies every file
  (`verify_weights.py`), serves it with fixed vLLM flags on 127.0.0.1:8000 (key via the `QWEN_API_KEY`
  SkyPilot secret), relays it with socat to the Docker bridge (172.17.0.1:18000) for the trial containers,
  then runs `harbor run` against a staged tasks dir and Harbor config. LoRA evals add the adapter shipped in
  the staged dir. Autostop/autodown after 60 idle minutes; MAX_RUN_HOURS timeout around the job.
- `infra/verda/fh.py`: `eval-up` (stage dir with tasks/cfg/meta/adapter; `sky launch -d` per box in
  parallel; fleet state in `.fh-eval.json`; refuses a relaunch whose definition differs), `eval-status`
  (cluster/job state + ssh probe of progress and vLLM /metrics + cost), `eval-wait` (watchdog: fetch finished
  boxes then `sky down`, report lost/preempted boxes, enforce a fleet cost cap via SkyPilot's cost report),
  `eval-fetch` (rsync runs/<job> and the box's status files back, write runs/<job>/fh-eval-meta.json),
  `eval-down`.
- `infra/verda/fh`: pins fastapi<0.145 (the API server crashed with newer fastapi), adds `fh sky ...`.
- `scripts/eval.sh --fleet` integration; `serve.yaml` gets `--max-num-seqs 32 --kv-cache-dtype fp8`;
  README; mocked tests in `tests/test_fh_eval.py`.

## Requirements to check against
- Safety: the tasks dir holds secret seeds/verify tokens: it must stay private on the box and never be
  committed; the API key must never be printed, logged, put on a command line or committed. The live
  serving box and its `fh-qwen` volume must never be touched (eval boxes must not mount it; `fh` must only
  delete clusters it created).
- Cost: no leaked boxes. Every path (failed launch, failed job, preemption, cost cap, crash of the local
  watcher, eval-down) must end with the box deleted or bounded by the on-box backstops. Spot by default.
- Comparability: every box of a fleet, and the base vs LoRA fleets, must use the same pinned instance
  type, identical vLLM flags (LoRA adds only the adapter flags) and the same -n; this must be recorded with
  each fetched run.
- Correctness: the bash in eval.yaml (set -euo pipefail, background jobs, traps, timeouts), the fleet state
  machine in fh.py, rsync/ssh usage, SkyPilot CLI/SDK usage.

Report only real correctness, safety, cost-leak or design problems, each with file:line, why it matters and
a concrete fix. Ignore style and nitpicks. If you find nothing serious, say "no blocking issues".
