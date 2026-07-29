#!/bin/bash
# Phase 3 validation: HiSparse<->MTP hybrid with CUDA graph ENABLED.
#
# Phase 3 added a residency dimension to the decode/verify graph key and
# captures BOTH variants when the hybrid is on (offloaded swap-in path and
# device-resident translate path). Before Phase 3, capture unconditionally
# attached the coordinator, so a "resident" batch replayed the swap-in graph.
#
# Checks per forced mode:
#   - server boots (watch capture time / memory: hybrid captures 2x verify graphs)
#   - burst probe: semantic correctness under graph replay
#   - 30k long context: KV far exceeds device_buffer_size=8192, so resident and
#     offloaded genuinely diverge (short prompts fit in the buffer and mask it)
#   - zero server errors
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
PORT=30012
RESULT="logs/${TS}_hybrid_cg.txt"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

kill_server() {
  local pids=$(pgrep -f "python -m sglang.launch_server" 2>/dev/null || true)
  if [ -n "$pids" ]; then echo "killing: $pids"; kill -9 $pids 2>/dev/null || true; sleep 8; fi
}

wait_server_ready() {
  local log=$1 timeout=1200 elapsed=0
  echo -n "  waiting..."
  while ! grep -q "The server is fired up" "$log" 2>/dev/null; do
    sleep 5; elapsed=$((elapsed + 5))
    if [ $elapsed -ge $timeout ]; then echo " TIMEOUT"; tail -30 "$log"; return 1; fi
    echo -n "."
  done
  echo " ready (${elapsed}s)"
}

SERVER_ARGS=(
  --model-path .models_dsv32 --disable-radix-cache --tp-size 8
  --mem-fraction-static 0.85 --cuda-graph-max-bs 32 --port "$PORT" --host 127.0.0.1
  --enable-hisparse-mtp-hybrid
  --speculative-algorithm EAGLE --speculative-num-steps 2
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 2}'
)

run_mode() {
  local mode=$1
  kill_server
  local log="logs/${TS}_cg_${mode}.log"
  echo "" | tee -a "$RESULT"
  echo "=== CUDA graph ON, FORCE_MODE=$mode  server=$log ===" | tee -a "$RESULT"
  SGLANG_FORCE_HISPARSE_MTP_MODE="$mode" \
    .venv/bin/python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$log" 2>&1 &
  if ! wait_server_ready "$log"; then
    echo "$mode | SERVER FAILED" | tee -a "$RESULT"; kill_server; return 1
  fi
  grep -i "mode pinned" "$log" | tail -1 | tee -a "$RESULT"
  .venv/bin/python e2e_burst_probe.py > "logs/${TS}_cgprobe_${mode}.txt" 2>&1
  echo "  burst: $(tail -1 "logs/${TS}_cgprobe_${mode}.txt")" | tee -a "$RESULT"
  BENCH_PROMPTS_FILE=/cpfs02/user/lgd/sglang-hisparse-mtp/longbench_prompts_30k.jsonl \
    .venv/bin/python bench_longbench_concurrent.py "cg_${mode}_c4" 4 128 4 \
    > "logs/${TS}_cglong_${mode}.txt" 2>&1
  echo "  long30k: $(grep -iE 'SYSTEM decode|success|failed' "logs/${TS}_cglong_${mode}.txt" | tail -2 | tr '\n' ' ')" | tee -a "$RESULT"
  # Confirm graph replay actually happened (not silently eager).
  echo "  cuda-graph batches: $(grep -c 'cuda graph: True' "$log")" | tee -a "$RESULT"
  local errs=$(grep -cE "Traceback|illegal memory|CUDA error|RuntimeError|AssertionError" "$log" 2>/dev/null || true)
  echo "  server error lines: $errs" | tee -a "$RESULT"
  if [ "$errs" != "0" ]; then grep -nE "Traceback|illegal memory|CUDA error|RuntimeError|AssertionError" "$log" | head -5 | tee -a "$RESULT"; fi
  kill_server
}

echo "Phase 3: hybrid + CUDA graph ON, dual residency graph variants ($(date))" | tee "$RESULT"
run_mode mtp
run_mode hisparse
echo "" | tee -a "$RESULT"
echo "CG DONE" | tee -a "$RESULT"
