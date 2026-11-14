#!/bin/bash
# Calls `infra/verda/fh <cmd>-ensure` every 2 minutes until the stop file exists.
# Usage: scripts/watchdog.sh serve|train <logfile> <stopfile>
cd "$(dirname "$0")/.."
while [ ! -f "$3" ]; do
  infra/verda/fh "$1-ensure" >> "$2" 2>&1
  sleep 120
done
echo "[$(date +%T)] watchdog stopped" >> "$2"
