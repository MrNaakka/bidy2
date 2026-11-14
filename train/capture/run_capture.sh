#!/usr/bin/env bash
# Capture OpenCode's exact chat-completions requests for FrameHunt (see train/fhdata.py).
#
#   train/capture/run_capture.sh [scratch_dir]
#
# Builds one EVAL task, starts the scripted logging mock (mock_opencode.py) on the host's
# 127.0.0.1:19190, and runs Harbor's opencode agent against it with the same provider
# overlay the Qwen runs use (image input declared). Rootless Docker (slirp4netns) maps
# the host loopback to 10.0.2.2 inside task containers, so the agent's baseURL is
# http://10.0.2.2:19190/v1 (rootful Docker: use 172.17.0.1 and bind the mock there).
# Requests land in train/capture/requests/req_NN.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SC="${1:-$(mktemp -d)}"
uv run python scripts/build_tasks.py --split eval --out "$SC/tasks" --only logstream-7001
rm -rf train/capture/requests
uv run python train/capture/mock_opencode.py --port 19190 --log train/capture/requests > "$SC/mock.log" 2>&1 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT
sleep 1
QWEN_API_KEY=mock-key harbor run -c train/capture/job_capture.json -p "$SC/tasks" \
  --job-name fh-capture-opencode -o "$SC/runs" -n 1 -y
ls train/capture/requests
