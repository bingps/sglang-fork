#!/bin/bash
# Baseline HiSparse (no spec) + DP Attention (dp_size=tp_size=8) for long-context bench.
cd "$(dirname "$0")"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/${TS}_baseline_hisparse_dp.log"
echo "Baseline HiSparse DP log: $LOG"
exec .venv/bin/python -m sglang.launch_server \
  --model-path .models_dsv32 \
  --disable-radix-cache --enable-hisparse \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --mem-fraction-static 0.85 --max-running-requests 24 \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 1}' \
  --cuda-graph-max-bs 32 \
  --port 30012 --host 127.0.0.1 \
  > "$LOG" 2>&1
