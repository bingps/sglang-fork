#!/bin/bash
# Kill any sglang launch_server process (by python cmdline), avoiding self-match.
pids=$(pgrep -f "python -m sglang.launch_server")
if [ -n "$pids" ]; then
  echo "killing: $pids"
  kill -9 $pids 2>/dev/null
else
  echo "no sglang server process found"
fi
