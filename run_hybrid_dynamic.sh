#!/bin/bash
# Capstone: live dynamic MTP <-> HiSparse switching (Phase 6 end-to-end).
#
# No FORCE_MODE pin -- the HybridModeController drives the mode from real
# device-pool pressure. 30k prompts at concurrency 8 make resident mode consume
# the physical pool (each resident token needs its own physical slot), pushing
# usage past the up-threshold so in-flight requests are offloaded; as requests
# drain, usage falls back under the down-threshold and they are restored.
#
# Pass criteria:
#   - at least one MTP -> HiSparse AND one HiSparse -> MTP transition logged
#   - all requests succeed, output stays coherent (burst probe afterwards)
#   - zero server errors (no IMA / assert / traceback)
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
PORT=30012
LOG="logs/${TS}_hybrid_dynamic.log"
RESULT="logs/${TS}_hybrid_dynamic.txt"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

kill_server() {
  local pids=$(pgrep -f "python -m sglang.launch_server" 2>/dev/null || true)
  if [ -n "$pids" ]; then echo "killing: $pids"; kill -9 $pids 2>/dev/null || true; sleep 8; fi
}

kill_server
echo "Dynamic MTP<->HiSparse switching, 30k prompts ($(date))" | tee "$RESULT"
echo "server log: $LOG" | tee -a "$RESULT"

# Narrow band + small min-bsz so the ramp actually crosses both thresholds.
.venv/bin/python -m sglang.launch_server \
  --model-path .models_dsv32 --disable-radix-cache --tp-size 8 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs 32 --port "$PORT" --host 127.0.0.1 \
  --enable-hisparse-mtp-hybrid \
  --hisparse-mtp-usage-up 0.5 --hisparse-mtp-usage-down 0.25 \
  --hisparse-mtp-min-bsz 2 --hisparse-mtp-max-bsz-for-mtp 1 \
  --hisparse-mtp-cooldown-steps 10 \
  --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 8192, "host_to_device_ratio": 2}' \
  > "$LOG" 2>&1 &

echo -n "  waiting..."
elapsed=0
while ! grep -q "The server is fired up" "$LOG" 2>/dev/null; do
  sleep 5; elapsed=$((elapsed + 5))
  if [ $elapsed -ge 1200 ]; then echo " TIMEOUT"; tail -30 "$LOG"; kill_server; exit 1; fi
  echo -n "."
done
echo " ready (${elapsed}s)"

# Load ramp: 8 concurrent 30k-context requests, long generations to sustain
# pressure, then the pool drains as they finish.
BENCH_PROMPTS_FILE=/cpfs02/user/lgd/sglang-hisparse-mtp/longbench_prompts_30k.jsonl \
  .venv/bin/python bench_longbench_concurrent.py dynamic_c8 8 512 8 \
  > "logs/${TS}_dynlong.txt" 2>&1
RAMP_RC=$?
echo "  ramp: $(grep -iE 'success|SYSTEM decode' "logs/${TS}_dynlong.txt" | tail -2 | tr '\n' ' ') (rc=$RAMP_RC)" | tee -a "$RESULT"
# The benchmark prints per-request success but exits 0 regardless; parse it.
RAMP_OK=$(grep -oE "success: [0-9]+/[0-9]+" "logs/${TS}_dynlong.txt" | tail -1)

# Light load afterwards: usage drops, should be back on MTP; also checks output.
.venv/bin/python e2e_burst_probe.py > "logs/${TS}_dynprobe.txt" 2>&1
PROBE_RC=$?
echo "  burst after ramp: $(tail -1 "logs/${TS}_dynprobe.txt") (rc=$PROBE_RC)" | tee -a "$RESULT"

echo "" | tee -a "$RESULT"
echo "--- mode transitions ---" | tee -a "$RESULT"
grep -oE "Hybrid: (MTP -> HiSparse|HiSparse -> MTP( aborted)?)[^,]*" "$LOG" | sort | uniq -c | tee -a "$RESULT"
TO_HS=$(grep -c "Hybrid: MTP -> HiSparse" "$LOG" || true)
TO_MTP=$(grep -c "Hybrid: HiSparse -> MTP (" "$LOG" || true)
echo "  MTP->HiSparse: $TO_HS   HiSparse->MTP: $TO_MTP" | tee -a "$RESULT"
echo "--- sample transition lines ---" | tee -a "$RESULT"
grep -E "Hybrid: (MTP -> HiSparse|HiSparse -> MTP)" "$LOG" | head -6 | tee -a "$RESULT"

ERRS=$(grep -cE "Traceback|illegal memory|CUDA error|RuntimeError|AssertionError" "$LOG" 2>/dev/null || true)
echo "  server error lines: $ERRS" | tee -a "$RESULT"
if [ "$ERRS" != "0" ]; then grep -nE "Traceback|illegal memory|CUDA error|RuntimeError|AssertionError" "$LOG" | head -8 | tee -a "$RESULT"; fi

# PASS requires: both transition directions, zero server errors, the ramp
# completed with every request succeeding, and the burst probe passed (its
# exit code carries the semantic checks). Anything less is NEEDS REVIEW.
RAMP_FULL_OK=0
case "$RAMP_OK" in
  "success: "*) N=${RAMP_OK#success: }; [ "${N%/*}" = "${N#*/}" ] && RAMP_FULL_OK=1 ;;
esac
if [ "$TO_HS" -ge 1 ] && [ "$TO_MTP" -ge 1 ] && [ "$ERRS" = "0" ] \
   && [ "$RAMP_RC" = "0" ] && [ "$RAMP_FULL_OK" = "1" ] && [ "$PROBE_RC" = "0" ]; then
  echo "DYNAMIC SWITCH: PASS" | tee -a "$RESULT"
else
  echo "DYNAMIC SWITCH: NEEDS REVIEW (to_hs=$TO_HS to_mtp=$TO_MTP errs=$ERRS ramp_rc=$RAMP_RC ramp=$RAMP_OK probe_rc=$PROBE_RC)" | tee -a "$RESULT"
fi
kill_server
