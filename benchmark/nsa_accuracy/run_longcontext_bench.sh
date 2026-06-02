#!/bin/bash
set -e

source /workspace/sglang-fork/.venv/bin/activate
cd /workspace/sglang-fork

MODEL_PATH="/data2/dsv32awq/"
TP=8
MEM_FRACTION="0.88"
CUDA_GRAPH_MAX_BS=2
HOST="127.0.0.1"
PORT=30000
BASE_URL="http://${HOST}:${PORT}"
RESULT_DIR="benchmark/nsa_accuracy/results"
NUM_THREADS=64

# 要跑的 benchmark 列表 (通过环境变量指定，默认两个都跑)
BENCHMARKS="${BENCHMARKS:-longbench_v2 aime25}"

# LongBench v2 配置
LONGBENCH_NUM_EXAMPLES="${LONGBENCH_NUM_EXAMPLES:-0}"  # 0=全部
LONGBENCH_MIN_CTX="${LONGBENCH_MIN_CTX:-}"  # 可设为 20000 筛选长文本

# AIME25 配置
AIME25_NUM_EXAMPLES="${AIME25_NUM_EXAMPLES:-0}"  # 0=全部 (30题)

BASE_ARGS=(
    --model-path "$MODEL_PATH"
    --tp "$TP"
    --mem-fraction-static "$MEM_FRACTION"
    --tool-call-parser deepseekv3
    --reasoning-parser deepseek-v3
    --speculative-algorithm EAGLE
    --speculative-num-steps 1
    --speculative-eagle-topk 1
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --host "$HOST"
    --port "$PORT"
)

kill_server() {
    echo "[INFO] Killing server..."
    bash scripts/killall_sglang.sh 2>/dev/null || true
    sleep 10
}

wait_for_server() {
    local server_pid=$1
    local config_name=$2
    echo "[INFO] Waiting for server..."
    local elapsed=0
    while ! env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u no_proxy -u NO_PROXY \
            curl -sf "${BASE_URL}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "[ERROR] Server died!"; tail -20 "${RESULT_DIR}/server_${config_name}.log"; exit 1
        fi
        if [ $elapsed -ge 600 ]; then echo "[ERROR] Timeout"; exit 1; fi
        if [ $((elapsed % 60)) -eq 0 ]; then echo "[INFO] Still waiting... ${elapsed}s"; fi
    done
    echo "[INFO] Server ready! (${elapsed}s)"
}

run_eval_benchmark() {
    local config_name="$1"
    local eval_name="$2"

    echo "============================================="
    echo "[EVAL] Config: ${config_name} | Benchmark: ${eval_name}"
    echo "============================================="

    local eval_args=(
        --base-url "${BASE_URL}"
        --eval-name "${eval_name}"
        --num-threads "${NUM_THREADS}"
    )

    if [ "$eval_name" = "longbench_v2" ]; then
        if [ "$LONGBENCH_NUM_EXAMPLES" != "0" ]; then
            eval_args+=(--num-examples "$LONGBENCH_NUM_EXAMPLES")
        fi
        if [ -n "$LONGBENCH_MIN_CTX" ]; then
            eval_args+=(--min-context-length "$LONGBENCH_MIN_CTX")
        fi
    elif [ "$eval_name" = "aime25" ]; then
        if [ "$AIME25_NUM_EXAMPLES" != "0" ]; then
            eval_args+=(--num-examples "$AIME25_NUM_EXAMPLES")
        fi
    fi

    # 保留代理(HF下载) + no_proxy 跳过本地
    env no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost" \
        python3 -m sglang.test.run_eval "${eval_args[@]}" \
        2>&1 | tee "${RESULT_DIR}/${eval_name}_${config_name}.log"

    local score=$(grep -oP 'Score: \K[0-9.]+' "${RESULT_DIR}/${eval_name}_${config_name}.log" || echo "N/A")
    echo "[RESULT] ${config_name} | ${eval_name}: ${score}"
    echo "${config_name},${score}" >> "${RESULT_DIR}/${eval_name}_summary.csv"
}

launch_server() {
    local config_name="$1"
    local draft_tokens="$2"
    local extra_env="$3"
    local index_topk_freq="$4"

    kill_server

    # 等端口释放
    local port_wait=0
    while ss -tlnp | grep -q ":${PORT} " 2>/dev/null; do
        sleep 2; port_wait=$((port_wait + 2))
        if [ $port_wait -ge 30 ]; then echo "[ERROR] Port stuck"; exit 1; fi
    done

    local cmd_args=("${BASE_ARGS[@]}" --speculative-num-draft-tokens "$draft_tokens")
    if [ -n "$index_topk_freq" ]; then
        cmd_args+=(--json-model-override-args "{\"index_topk_freq\": ${index_topk_freq}}")
    fi

    if [ -n "$extra_env" ]; then
        env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u no_proxy -u NO_PROXY \
            $extra_env python3 -m sglang.launch_server "${cmd_args[@]}" \
            &> "${RESULT_DIR}/server_${config_name}.log" &
    else
        env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u no_proxy -u NO_PROXY \
            python3 -m sglang.launch_server "${cmd_args[@]}" \
            &> "${RESULT_DIR}/server_${config_name}.log" &
    fi
    local server_pid=$!
    echo "[INFO] Server PID: ${server_pid} for ${config_name}"
    wait_for_server $server_pid "$config_name"
}

# 配置列表: name, draft_tokens, extra_env, index_topk_freq
CONFIGS=(
    "baseline_d2|2||"
    "specret_d2|2|SGLANG_NSA_ENABLE_SPECRET=1|"
    "specret_d3|3|SGLANG_NSA_ENABLE_SPECRET=1|"
    "specret_d4|4|SGLANG_NSA_ENABLE_SPECRET=1|"
    "index_topk_freq_2|2||2"
    "index_topk_freq_3|2||3"
    "index_topk_freq_4|2||4"
)

TOTAL=${#CONFIGS[@]}
mkdir -p "$RESULT_DIR"

echo "================================================"
echo " NSA Accuracy: LongBench v2 + AIME25"
echo " Benchmarks: ${BENCHMARKS}"
echo " Configs: ${TOTAL}"
echo "================================================"

step=0
for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r config_name draft_tokens extra_env index_topk_freq <<< "$config_str"
    step=$((step + 1))

    echo ""
    echo "###############################################"
    echo "# [${step}/${TOTAL}] ${config_name}"
    echo "#   draft_tokens=${draft_tokens} env=${extra_env} freq=${index_topk_freq}"
    echo "###############################################"

    launch_server "$config_name" "$draft_tokens" "$extra_env" "$index_topk_freq"

    for bench in $BENCHMARKS; do
        run_eval_benchmark "$config_name" "$bench"
    done

    kill_server
done

echo ""
echo "========== FINAL RESULTS =========="
for bench in $BENCHMARKS; do
    echo ""
    echo "--- ${bench} ---"
    cat "${RESULT_DIR}/${bench}_summary.csv" 2>/dev/null || echo "(no results)"
done
