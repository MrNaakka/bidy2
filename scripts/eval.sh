#!/usr/bin/env bash
# FrameHunt evaluation: build the EVAL tasks, run everything through Harbor, summarize.
#
#   scripts/eval.sh                      # everything: build, reference, nop, stop-early, 3 agent runs, summary
#   scripts/eval.sh --stage baselines    # build + reference + nop + stop-early + summary (no API spend)
#   scripts/eval.sh --stage agents       # the 3 OpenCode runs + summary (needs ANTHROPIC_API_KEY, e.g. in .env)
#   scripts/eval.sh --stage summarize    # only rebuild results/summary.{md,json} from runs/
#
# Options: -n N (concurrent trials, default 4), --runs R (agent runs, default 3),
#          --model M (default anthropic/claude-opus-5-5), --split S (default eval).
# Jobs have fixed names and go to runs/<job>; an existing runs/<job> is replaced.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE=all
N=4
RUNS=3
MODEL="anthropic/claude-opus-5-5"
SPLIT=eval
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
  harbor run -p "$TASKS" --job-name "$name" -o "$OUT" -n "$N" -y "$@"
  stage_time "$name" $((SECONDS - t0))
}

if [ "$STAGE" = all ] || [ "$STAGE" = agents ]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ] && ! grep -q '^ANTHROPIC_API_KEY=.' .env 2>/dev/null; then
    echo "ANTHROPIC_API_KEY is not set (env or .env); needed for the agent runs." >&2
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
  [ -d "$TASKS" ] || uv run python scripts/build_tasks.py --split "$SPLIT"
  ENVFILE=()
  [ -f .env ] && ENVFILE=(--env-file .env)
  for i in $(seq 1 "$RUNS"); do
    job "fh-$SPLIT-opencode-run$i" -a opencode -m "$MODEL" "${ENVFILE[@]}"
  done
fi

uv run python scripts/summarize.py --runs "$OUT" --out results
echo
echo "Full tables: results/summary.md (per item, per kind); raw data: results/summary.json; Harbor jobs: $OUT/"
