#!/bin/bash
# Phase 0: isolate the resident (translate) path from the CUDA-graph confound.
#
# decode_cuda_graph_runner.py:814-816 attaches the coordinator UNCONDITIONALLY
# at capture time, so a captured verify graph always contains the offloaded
# swap-in path. With CUDA graph ON, a "resident" batch therefore replays
# swap-in instead of the translate path that Phase 2 gates. Running with
# --disable-cuda-graph forces the eager dispatch, which is the only way to
# validate the resident path itself.
#
# Also uses LONG prompts (30k >> device_buffer_size=8192) so the offloaded path
# would genuinely need host swap-in -- the short burst prompts fit entirely in
# the device buffer and mask the difference.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
PORT=30012
RESULT="logs/${TS}_resident_nocg.txt"
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
    if [ $elapsed -ge $timeout ]; then echo " TIMEOUT"; tail -30 "$log"; return 1; fi
    echo -n "."
  done
  echo " ready (${elapsed}s)"
}

SERVER_ARGS=(
  --model-path .models_dsv32 --disable-radix-cache --tp-size 8
  --mem-fraction-static 0.85 --port "$PORT" --host 127.0.0.1
  --disable-cuda-graph
  --enable-hisparse-mtp-hybrid
  --speculative-algorithm EAGLE --speculative-num-steps 2
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 2}'
)

run_mode() {
  local mode=$1
  kill_server
  local log="logs/${TS}_nocg_${mode}.log"
  local probe="logs/${TS}_nocgprobe_${mode}.txt"
  echo "" | tee -a "$RESULT"
  echo "=== eager (no CG) FORCE_MODE=$mode  server=$log ===" | tee -a "$RESULT"
  SGLANG_FORCE_HISPARSE_MTP_MODE="$mode" \
    .venv/bin/python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$log" 2>&1 &
  if ! wait_server_ready "$log"; then
    echo "$mode | SERVER FAILED" | tee -a "$RESULT"; kill_server; return 1
  fi
  grep -i "mode pinned" "$log" | tail -1 | tee -a "$RESULT"
  # short burst (semantic correctness)
  .venv/bin/python e2e_burst_probe.py > "$probe" 2>&1
  echo "  burst: $(tail -1 "$probe")  exit=$?" | tee -a "$RESULT"
  # long-context single request: 30k prompt far exceeds the 8192 device buffer,
  # so this is where resident vs offloaded genuinely diverge.
  BENCH_PROMPTS_FILE=/cpfs02/user/lgd/sglang-hisparse-mtp/longbench_prompts_30k.jsonl \
    .venv/bin/python bench_longbench_concurrent.py "nocg_${mode}_c2" 2 128 2 \
    > "logs/${TS}_nocglong_${mode}.txt" 2>&1
  echo "  long30k: $(grep -iE 'throughput|success|ok|fail' "logs/${TS}_nocglong_${mode}.txt" | tail -3 | tr '\n' ' ')" | tee -a "$RESULT"
  local errs=$(grep -cE "Traceback|illegal memory|CUDA error|RuntimeError|AssertionError" "$log" 2>/dev/null || true)
  echo "  server error lines: $errs" | tee -a "$RESULT"
  if [ "$errs" != "0" ]; then grep -nE "Traceback|illegal memory|CUDA error|RuntimeError|AssertionError" "$log" | head -5 | tee -a "$RESULT"; fi
  kill_server
}

echo "Resident-vs-offloaded EAGER (no CUDA graph), 30k long ctx ($(date))" | tee "$RESULT"
run_mode mtp
run_mode hisparse
echo "" | tee -a "$RESULT"
echo "NOCG DONE" | tee -a "$RESULT"
