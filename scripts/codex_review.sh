#!/bin/bash
# Headless Codex review. Usage: scripts/codex_review.sh <prompt-file> <out-md>
set -u
cd "$(dirname "$0")/.."
timeout 3000 codex exec -m gpt-6.1-sol -c model_reasoning_effort='"medium"' --sandbox read-only -C "$PWD" \
  -o "$2" "$(cat "$1")" < /dev/null > "$2.log" 2>&1
echo "exit=$?"
