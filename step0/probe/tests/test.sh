#!/bin/bash
CODE=$(curl -s "http://probe-mcp:8000/_verify?token=step0-verify-7f3a")
GOT=$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null)
echo "expected=$CODE got=$GOT"
if [ -n "$CODE" ] && [ "$CODE" = "$GOT" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
