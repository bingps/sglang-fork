#!/bin/bash
# Phase 0 smoke test for HiSparse<->MTP hybrid (Phase 1+2 end-to-end).
# Boots ONE hybrid server per forced mode and runs the deterministic burst
# probe (temp=0, checks expected substrings + garbled-output ratio):
#   - FORCE_MODE=mtp      -> resident path (no coordinator; translate)
#   - FORCE_MODE=hisparse -> offloaded path (coordinator; swap-in), i.e. the
#                            existing HiSparse+MTP behavior via the hybrid boot
# Validates that both batch-level paths boot and generate coherent output with
# zero server errors. Config: EAGLE [2,1,3], buf=8192, h2d=2.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
PORT=30012
RESULT="logs/${TS}_hybrid_smoke.txt"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

kill_server() {
  local pids=$(pgrep -f "python -m sglang.launch_server" 2>/dev/null || true)
  if [ -n "$pids" ]; then echo "killing: $pids"; kill -9 $pids 2>/dev/null || true; sleep 8; fi
}

wait_server_ready() {
  local log=$1 timeout=900 elapsed=0
  echo -n "  waiting..."
  while ! grep -q "The server is fired up" "$log" 2>/dev/null; do
    sleep 5; elapsed=$((elapsed + 5))
    if grep -qE "Traceback|Error:|error:" "$log" 2>/dev/null && \
       ! grep -q "The server is fired up" "$log" 2>/dev/null; then
      if [ $elapsed -ge 60 ]; then echo " LIKELY FAILED"; tail -30 "$log"; return 1; fi
    fi
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
  local log="logs/${TS}_hybrid_${mode}.log"
  local probe="logs/${TS}_probe_${mode}.txt"
  echo "" | tee -a "$RESULT"
  echo "=== FORCE_MODE=$mode  server=$log ===" | tee -a "$RESULT"
  SGLANG_FORCE_HISPARSE_MTP_MODE="$mode" \
    .venv/bin/python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$log" 2>&1 &
  if ! wait_server_ready "$log"; then
    echo "$mode | SERVER FAILED TO START" | tee -a "$RESULT"; kill_server; return 1
  fi
  grep -i "HybridModeController: mode pinned" "$log" | tail -1 | tee -a "$RESULT"
  .venv/bin/python e2e_burst_probe.py > "$probe" 2>&1
  local rc=$?
  tail -1 "$probe" | tee -a "$RESULT"
  echo "  probe exit=$rc" | tee -a "$RESULT"
  local errs=$(grep -cE "Traceback|illegal memory|CUDA error|RuntimeError|AssertionError" "$log" 2>/dev/null || echo 0)
  echo "  server error lines: $errs" | tee -a "$RESULT"
  if [ "$errs" != "0" ]; then grep -nE "Traceback|illegal memory|CUDA error|RuntimeError|AssertionError" "$log" | head -5 | tee -a "$RESULT"; fi
  kill_server
}

echo "Hybrid smoke: EAGLE[2,1,3] buf=8192 h2d=2, TP=8 ($(date))" | tee "$RESULT"
run_mode mtp
run_mode hisparse
echo "" | tee -a "$RESULT"
echo "SMOKE DONE" | tee -a "$RESULT"
