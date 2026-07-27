#!/bin/bash
# A/B for MTP-path optimizations (async backup + event-ordered staging wait).
# Params mirror doc Test 11: TP=8, h2d=2, buf=4096, LongBench 9k-15k prompts,
# max_tokens=2048, 48 requests, c=24/32.
# Doc baselines (pre-opt): HiSparse BL 714/676; HiSparse+MTP [1,1,2] 1082/1098.
set -euo pipefail
cd "$(dirname "$0")"

PORT=30012
RESULT_FILE="logs/bench_opt_ab_$(date +%Y%m%d_%H%M%S).txt"
NUM_REQUESTS=48
MAX_TOKENS=2048

kill_server() {
  local pids=$(pgrep -f "python -m sglang.launch_server" 2>/dev/null || true)
  if [ -n "$pids" ]; then echo "killing: $pids"; kill -9 $pids 2>/dev/null || true; sleep 8; fi
}

wait_server_ready() {
  local log=$1 timeout=600 elapsed=0
  echo -n "  waiting..."
  while ! grep -q "The server is fired up" "$log" 2>/dev/null; do
    sleep 5; elapsed=$((elapsed + 5))
    if [ $elapsed -ge $timeout ]; then echo " TIMEOUT"; tail -20 "$log"; return 1; fi
    echo -n "."
  done
  echo " ready (${elapsed}s)"
}

run_bench() {
  local log=$1 label=$2 conc=$3
  echo ""; echo "--- $label | c=$conc ---"
  local bench_start=$(date '+%Y-%m-%d %H:%M:%S')
  .venv/bin/python bench_longbench_concurrent.py "${label}_c${conc}" "$conc" "$MAX_TOKENS" "$NUM_REQUESTS" 2>&1
  local bench_end=$(date '+%Y-%m-%d %H:%M:%S')
  python3 -c "
import re
values, ar_values = [], []
with open('$log') as f:
    for line in f:
        m_ts = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if not m_ts: continue
        ts = m_ts.group(1)
        if ts < '$bench_start' or ts > '$bench_end': continue
        m_tp = re.search(r'gen throughput \(token/s\): ([\d.]+)', line)
        if m_tp: values.append(float(m_tp.group(1)))
        m_ar = re.search(r'accept rate: ([\d.]+)', line)
        if m_ar: ar_values.append(float(m_ar.group(1)))
if not values:
    tp_str = 'no data'
else:
    values.sort(); n = len(values)
    lo = max(1, n // 10); hi = n - lo
    trimmed = values[lo:hi] if hi > lo else values
    avg = sum(trimmed) / len(trimmed)
    tp_str = f'samples={n} trimmed_avg={avg:.1f} median={values[n//2]:.1f}'
ar_str = f' accept_rate={sum(ar_values)/len(ar_values):.3f}' if ar_values else ''
print(f'  [result] {tp_str}{ar_str}')
with open('$RESULT_FILE', 'a') as f:
    f.write(f'$label | c=$conc | {tp_str}{ar_str}\n')
"
}

run_one_config() {
  local label=$1; shift
  local server_args=("$@")
  kill_server
  local ts=$(date +%Y%m%d_%H%M%S)
  local log="logs/${ts}_${label}.log"
  echo ""; echo "=========================================="; echo "CONFIG: $label"; echo "=========================================="
  .venv/bin/python -m sglang.launch_server "${server_args[@]}" > "$log" 2>&1 &
  wait_server_ready "$log" || { kill_server; return 1; }
  for conc in 24 32; do
    run_bench "$log" "$label" "$conc"
  done
  kill_server
}

mkdir -p logs
echo "OPT A/B (async backup + event staging wait), TP=8 h2d=2 buf=4096, max_tokens=$MAX_TOKENS ($(date))" | tee "$RESULT_FILE"
echo "doc pre-opt: hisparse_bl 714/676, hisparse_mtp112 1082/1098, plain_bl 876/826" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 1. HiSparse+MTP [1,1,2] — the optimized path
run_one_config "opt_hisparse_mtp112" \
  --model-path .models_dsv32 --disable-radix-cache --enable-hisparse \
  --speculative-algorithm EAGLE --speculative-num-steps 1 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
  --tp-size 8 --mem-fraction-static 0.85 \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}' \
  --cuda-graph-max-bs 32 --port $PORT --host 127.0.0.1

# 2. HiSparse BL — untouched path, environment calibration
run_one_config "opt_hisparse_bl" \
  --model-path .models_dsv32 --disable-radix-cache --enable-hisparse \
  --tp-size 8 --mem-fraction-static 0.85 \
  --max-total-tokens 423680 \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}' \
  --cuda-graph-max-bs 32 --port $PORT --host 127.0.0.1

echo ""; echo "ALL DONE"; cat "$RESULT_FILE"
