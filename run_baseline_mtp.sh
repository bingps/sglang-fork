#!/bin/bash
# Baseline MTP server (no HiSparse) for throughput comparison.
cd "$(dirname "$0")"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/${TS}_baseline_mtp.log"
echo "Baseline MTP log: $LOG"
exec .venv/bin/python -m sglang.launch_server \
  --model-path .models_dsv32 \
  --disable-radix-cache \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tp-size 8 --mem-fraction-static 0.85 \
  --port 30012 --host 127.0.0.1 \
  > "$LOG" 2>&1
