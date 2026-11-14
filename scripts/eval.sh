#!/usr/bin/env bash
# FrameHunt evaluation: build the EVAL tasks, run everything through Harbor, summarize.
#
#   scripts/eval.sh                      # everything: build, reference, nop, stop-early, 3 agent runs, summary
#   scripts/eval.sh --stage baselines    # build + reference + nop + stop-early + summary (no API spend)
#   scripts/eval.sh --stage agents       # the 3 OpenCode runs + summary (needs ANTHROPIC_API_KEY, e.g. in .env)
#   scripts/eval.sh --stage summarize    # only rebuild results/summary.{md,json} from runs/
#   scripts/eval.sh --stage agents --model qwen   # Qwen3.8-27B on our vLLM endpoint (infra/verda; needs
#                                                 # QWEN_BASE_URL and QWEN_API_KEY in .env, endpoint up)
#
# Options: -n N (concurrent trials, default 4), --runs R (agent runs, default 3),
#          --model M (default anthropic/claude-opus-5-5; "qwen" = Qwen3.8-27B via infra/verda,
#          "qwen-fh" = the same with our LoRA adapter), --split S (default eval).
# OpenCode is pinned to $OPENCODE_VERSION (the version the training data was captured with).
# Only agent-install timeouts (before the agent sees anything) are retried, up to 2 times.
# Jobs have fixed names and go to runs/<job>; an existing runs/<job> is replaced.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE=all
N=4
RUNS=3
MODEL="anthropic/claude-opus-5-5"
SPLIT=eval
OPENCODE_VERSION=1.19.5
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    -n) N="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$STAGE" in all|baselines|agents|summarize) ;; *) echo "bad --stage: $STAGE" >&2; exit 2 ;; esac

TASKS="tasks/$SPLIT"
OUT=runs
mkdir -p "$OUT" results
TIMES="results/stage_times.txt"
[ "$STAGE" = summarize ] || echo "# stage wall times, $(date -u +%FT%TZ), stage=$STAGE n=$N" >> "$TIMES"

stage_time() {  # name, seconds
  echo "$1 $2s" | tee -a "$TIMES"
}

job() {  # job name, then harbor run args
  local name="$1"; shift
  rm -rf "$OUT/$name"
  local t0=$SECONDS
  echo "=== $name: harbor run $*"
  # Retry only agent-install timeouts (OpenCode's npm install), which happen before the agent
  # sees anything. Other start/build/healthcheck errors are not retried: with the separate
  # verifier, Harbor can raise them after the agent has run, and a retry would hand the agent
  # a fresh pixel budget. Failures after the agent started count as failures.
  harbor run -p "$TASKS" --job-name "$name" -o "$OUT" -n "$N" -r 2 \
    --retry-include AgentSetupTimeoutError -y "$@"
  stage_time "$name" $((SECONDS - t0))
}

if [ "$STAGE" = all ] || [ "$STAGE" = agents ]; then
  case "$MODEL" in qwen|qwen-fh) NEEDKEY=QWEN_API_KEY ;; *) NEEDKEY=ANTHROPIC_API_KEY ;; esac
  if [ -z "${!NEEDKEY:-}" ] && ! grep -q "^$NEEDKEY=." .env 2>/dev/null; then
    echo "$NEEDKEY is not set (env or .env); needed for the agent runs with --model $MODEL." >&2
    echo "Run 'scripts/eval.sh --stage baselines' to skip them." >&2
    exit 2
  fi
fi

if [ "$STAGE" = all ] || [ "$STAGE" = baselines ]; then
  t0=$SECONDS
  uv run python scripts/build_tasks.py --split "$SPLIT"
  stage_time build $((SECONDS - t0))
  # Reference run: Harbor's oracle agent runs solution/solve.sh (framehunt/reference.py).
  job "fh-$SPLIT-reference" -a oracle
  # Do-nothing baseline: never calls a tool, never submits.
  job "fh-$SPLIT-nop" -a nop
  # Stop-early baseline: info, one low-res look, then a well-formed guess.
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" job "fh-$SPLIT-stop-early" -a framehunt.baselines.stop_early:StopEarly
fi

if [ "$STAGE" = all ] || [ "$STAGE" = agents ]; then
  # Build if missing or if the task dir doesn't hold exactly the split's items (e.g. it
  # was built from an older item list).
  WANT="$(uv run python -c "from framehunt.splits import SPLITS; print(' '.join(sorted(i for i, _, _ in SPLITS['$SPLIT'])))")"
  HAVE="$( [ -d "$TASKS" ] && find "$TASKS" -mindepth 2 -maxdepth 2 -name task.toml -printf '%h\n' | xargs -rn1 basename | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')"
  [ "$WANT" = "$HAVE" ] || uv run python scripts/build_tasks.py --split "$SPLIT"
  # Give Harbor only the one key the model needs, not the whole .env.
  KEYENV="$(mktemp)"; trap 'rm -f "$KEYENV"' EXIT
  envval() { local v="${!1:-}"; [ -n "$v" ] || v="$(grep -m1 "^$1=" .env 2>/dev/null | cut -d= -f2- || true)"; printf '%s' "$v"; }
  case "$MODEL" in
    qwen|qwen-fh)
      SERVED=qwen3.8-27b; LABEL=qwen
      [ "$MODEL" = qwen-fh ] && { SERVED=qwen3.8-27b-fh; LABEL=qwen-fh; }
      printf 'QWEN_API_KEY=%s\n' "$(envval QWEN_API_KEY)" > "$KEYENV"
      CFG="$(mktemp --suffix .json)"
      uv run python infra/verda/verda_cli.py harbor-agent --served-name "$SERVED" --opencode-version "$OPENCODE_VERSION" > "$CFG"
      AGENT_ARGS=(-c "$CFG")
      ;;
    *)
      LABEL=opencode
      printf 'ANTHROPIC_API_KEY=%s\n' "$(envval ANTHROPIC_API_KEY)" > "$KEYENV"
      AGENT_ARGS=(-a opencode -m "$MODEL" --ak "version=$OPENCODE_VERSION")
      ;;
  esac
  for i in $(seq 1 "$RUNS"); do
    job "fh-$SPLIT-$LABEL-run$i" "${AGENT_ARGS[@]}" --env-file "$KEYENV"
  done
fi

uv run python scripts/summarize.py --runs "$OUT" --out results
echo
echo "Full tables: results/summary.md (per item, per kind); raw data: results/summary.json; Harbor jobs: $OUT/"
