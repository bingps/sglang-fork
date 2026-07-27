#!/bin/bash
# E2E test launcher for HiSparse+MTP staging-ring validation.
cd "$(dirname "$0")"
mkdir -p logs
LOG_FILE="logs/$(date +%Y%m%d_%H%M%S)_e2e_server.log"
echo "server log: $LOG_FILE"
exec .venv/bin/python -m sglang.launch_server \
  --model-path .models_dsv32 \
  --disable-radix-cache --enable-hisparse \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tp-size 8 --mem-fraction-static 0.85 \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 1}' \
  --port 30012 --host 127.0.0.1 \
  > "$LOG_FILE" 2>&1
