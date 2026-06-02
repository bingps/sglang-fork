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
NUM_THREADS=16
EVAL_NAME="${EVAL_NAME:-aime25}"
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"

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

run_eval() {
    local config_name="$1"

    echo "============================================="
    echo "[EVAL] Config: ${config_name} | Benchmark: ${EVAL_NAME}"
    echo "============================================="

    local eval_args=(
        --base-url "${BASE_URL}"
        --eval-name "${EVAL_NAME}"
        --num-threads "${NUM_THREADS}"
    )
    if [ "$NUM_EXAMPLES" != "0" ]; then
        eval_args+=(--num-examples "$NUM_EXAMPLES")
    fi

    # AIME25 需要更长的 reasoning chain
    if [ "$EVAL_NAME" = "aime25" ]; then
        eval_args+=(--max-tokens "${MAX_TOKENS:-32768}")
        eval_args+=(--temperature "${TEMPERATURE:-0.6}")
    else
        eval_args+=(--max-tokens "${MAX_TOKENS:-2048}")
    fi

    # 保留代理(HF下载) + no_proxy 跳过本地
    env no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost" \
        python3 -m sglang.test.run_eval "${eval_args[@]}" \
        2>&1 | tee "${RESULT_DIR}/${EVAL_NAME}_${config_name}.log"

    local score=$(grep -oP 'Score: \K[0-9.]+' "${RESULT_DIR}/${EVAL_NAME}_${config_name}.log" || echo "N/A")
    echo "[RESULT] ${config_name} | ${EVAL_NAME}: ${score}"
    echo "${config_name},${score}" >> "${RESULT_DIR}/${EVAL_NAME}_summary.csv"
}

launch_server() {
    local config_name="$1"
    local draft_tokens="$2"
    local extra_env="$3"
    local index_topk_freq="$4"

    kill_server
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

# 配置列表
CONFIGS=(
    "baseline_d2|2||"
    "specret_d2|2|SGLANG_NSA_ENABLE_SPECRET=1|"
    "specret_d3|3|SGLANG_NSA_ENABLE_SPECRET=1|"
    "specret_d4|4|SGLANG_NSA_ENABLE_SPECRET=1|"
    "index_topk_freq_2|2||2"
    "index_topk_freq_3|2||3"
    "index_topk_freq_4|2||4"
)

# 可通过 START_FROM 跳过已完成的配置
START_FROM=${START_FROM:-1}
TOTAL=${#CONFIGS[@]}
mkdir -p "$RESULT_DIR"

echo "================================================"
echo " NSA Accuracy: ${EVAL_NAME}"
echo " Starting from config #${START_FROM}/${TOTAL}"
echo "================================================"

step=0
for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r config_name draft_tokens extra_env index_topk_freq <<< "$config_str"
    step=$((step + 1))
    if [ $step -lt $START_FROM ]; then continue; fi

    echo ""
    echo "###############################################"
    echo "# [${step}/${TOTAL}] ${config_name}"
    echo "###############################################"

    launch_server "$config_name" "$draft_tokens" "$extra_env" "$index_topk_freq"
    run_eval "$config_name"
    kill_server
done

echo ""
echo "========== FINAL ${EVAL_NAME} RESULTS =========="
cat "${RESULT_DIR}/${EVAL_NAME}_summary.csv" 2>/dev/null || echo "(no results)"
