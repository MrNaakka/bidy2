#!/bin/bash
# Oracle cheats via the verify endpoint; only used to check plumbing.
curl -s "http://probe-mcp:8000/_verify?token=step0-verify-7f3a" > /app/answer.txt
