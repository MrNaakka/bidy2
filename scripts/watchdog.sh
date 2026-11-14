#!/bin/bash
# Calls `verda_cli.py <cmd>-ensure` every 2 minutes until the stop file exists.
# Usage: scripts/watchdog.sh serve|train <logfile> <stopfile>
cd "$(dirname "$0")/.."
while [ ! -f "$3" ]; do
  uv run python infra/verda/verda_cli.py "$1-ensure" --wait >> "$2" 2>&1
  sleep 120
done
echo "[$(date +%T)] watchdog stopped" >> "$2"
