#!/bin/bash
# =============================================================================
# NSA Accuracy Comparison: Unified Benchmark Script
#
# 对比 Baseline vs SPECRET (d=2,3,4) vs index_topk_freq (2,3,4)
#
# 用法:
#   EVAL_NAME=gsm8k NUM_EXAMPLES=200 bash run_nsa_accuracy.sh
#   EVAL_NAME=ceval bash run_nsa_accuracy.sh
#   EVAL_NAME=aime25 bash run_nsa_accuracy.sh
#   EVAL_NAME="longbench_v2 aime25" bash run_nsa_accuracy.sh  # 多 benchmark
#   START_FROM=5 EVAL_NAME=ceval bash run_nsa_accuracy.sh     # 从第5个配置开始
#
# 支持的 benchmark: mmlu, gsm8k, mgsm_en, ceval, aime25, longbench_v2
# =============================================================================

set -e

# ===================== 配置区 =====================
MODEL_PATH="${MODEL_PATH:-/data2/dsv32awq/}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-2}"

# 评测配置
EVAL_NAME="${EVAL_NAME:-gsm8k}"          # 空格分隔可跑多个
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"      # 0=全部
NUM_THREADS="${NUM_THREADS:-64}"

# AIME25 特定配置
AIME25_MAX_TOKENS="${AIME25_MAX_TOKENS:-32768}"
AIME25_TEMPERATURE="${AIME25_TEMPERATURE:-0.6}"

# LongBench v2 特定配置
LONGBENCH_MIN_CTX="${LONGBENCH_MIN_CTX:-}"  # 设为 20000 筛选 20K+ 长文本

# 从第 N 个配置开始（用于断点续跑）
START_FROM="${START_FROM:-1}"

# 结果输出目录
RESULT_DIR="benchmark/nsa_accuracy/results"
mkdir -p "$RESULT_DIR"

# 公共服务器参数
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

# 7 个对比配置: name|draft_tokens|extra_env|index_topk_freq
CONFIGS=(
    "baseline_d2|2||"
    "specret_d2|2|SGLANG_NSA_ENABLE_SPECRET=1|"
    "specret_d3|3|SGLANG_NSA_ENABLE_SPECRET=1|"
    "specret_d4|4|SGLANG_NSA_ENABLE_SPECRET=1|"
    "index_topk_freq_2|2||2"
    "index_topk_freq_3|2||3"
    "index_topk_freq_4|2||4"
)

# ===================== 辅助函数 =====================

kill_server() {
    echo "[INFO] Killing server..."
    bash scripts/killall_sglang.sh 2>/dev/null || true
    sleep 10
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
        if [ $port_wait -ge 30 ]; then echo "[ERROR] Port ${PORT} stuck"; exit 1; fi
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

    # 等待服务器就绪
    echo "[INFO] Waiting for server..."
    local elapsed=0
    while ! env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u no_proxy -u NO_PROXY \
            curl -sf "${BASE_URL}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "[ERROR] Server died!"; tail -20 "${RESULT_DIR}/server_${config_name}.log"; exit 1
        fi
        if [ $elapsed -ge 600 ]; then echo "[ERROR] Timeout (600s)"; exit 1; fi
        if [ $((elapsed % 60)) -eq 0 ]; then echo "[INFO] Still waiting... ${elapsed}s"; fi
    done
    echo "[INFO] Server ready! (${elapsed}s)"
}

run_eval() {
    local config_name="$1"
    local eval_name="$2"

    echo "============================================="
    echo "[EVAL] Config: ${config_name} | Benchmark: ${eval_name}"
    echo "============================================="

    local log_file="${RESULT_DIR}/${eval_name}_${config_name}.log"

    if [ "$eval_name" = "ceval" ]; then
        # CEval 使用 sglang 原生 API
        local ceval_args=(
            --data-path "ceval/ceval-exam"
            --host "$HOST" --port "$PORT"
            --parallel "$NUM_THREADS"
            --result-file "${RESULT_DIR}/${eval_name}_${config_name}.jsonl"
        )
        if [ "$NUM_EXAMPLES" != "0" ]; then
            ceval_args+=(--num-questions "$NUM_EXAMPLES")
        fi

        env no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost" HF_DATASETS_OFFLINE=1 \
            python3 benchmark/ceval/bench_sglang.py "${ceval_args[@]}" \
            2>&1 | tee "$log_file"

        local score=$(grep -oP 'Accuracy: \K[0-9.]+' "$log_file" || echo "N/A")
    else
        # 其他 benchmark 使用 OpenAI 兼容 API
        local eval_args=(
            --base-url "${BASE_URL}"
            --eval-name "${eval_name}"
            --num-threads "${NUM_THREADS}"
        )
        if [ "$NUM_EXAMPLES" != "0" ]; then
            eval_args+=(--num-examples "$NUM_EXAMPLES")
        fi

        # 按 benchmark 类型设置 max_tokens 和 temperature
        case "$eval_name" in
            aime25)
                eval_args+=(--max-tokens "$AIME25_MAX_TOKENS" --temperature "$AIME25_TEMPERATURE")
                ;;
            longbench_v2)
                eval_args+=(--max-tokens 2048)
                [ -n "$LONGBENCH_MIN_CTX" ] && eval_args+=(--min-context-length "$LONGBENCH_MIN_CTX")
                ;;
            *)
                eval_args+=(--max-tokens 2048)
                ;;
        esac

        env no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost" \
            python3 -m sglang.test.run_eval "${eval_args[@]}" \
            2>&1 | tee "$log_file"

        local score=$(grep -oP 'Score: \K[0-9.]+' "$log_file" || echo "N/A")
    fi

    echo "[RESULT] ${config_name} | ${eval_name}: ${score}"
    echo "${config_name},${score}" >> "${RESULT_DIR}/${eval_name}_summary.csv"
}

# ===================== 主流程 =====================

TOTAL=${#CONFIGS[@]}

echo "================================================"
echo " NSA Accuracy Comparison"
echo " Benchmarks: ${EVAL_NAME}"
echo " Num examples: ${NUM_EXAMPLES} (0=all)"
echo " Configs: ${TOTAL} (starting from #${START_FROM})"
echo "================================================"

step=0
for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r config_name draft_tokens extra_env index_topk_freq <<< "$config_str"
    step=$((step + 1))
    if [ $step -lt $START_FROM ]; then continue; fi

    echo ""
    echo "###############################################"
    echo "# [${step}/${TOTAL}] ${config_name}"
    echo "#   draft_tokens=${draft_tokens} env=${extra_env} freq=${index_topk_freq}"
    echo "###############################################"

    launch_server "$config_name" "$draft_tokens" "$extra_env" "$index_topk_freq"

    for bench in $EVAL_NAME; do
        run_eval "$config_name" "$bench"
    done

    kill_server
done

# ===================== 汇总结果 =====================
echo ""
echo "========== FINAL RESULTS =========="
for bench in $EVAL_NAME; do
    echo ""
    echo "--- ${bench} ---"
    column -t -s',' "${RESULT_DIR}/${bench}_summary.csv" 2>/dev/null || echo "(no results)"
done
echo ""
echo "跳过率对应关系:"
echo "  SPECRET d=2  ↔  index_topk_freq=2  (50% 跳过)"
echo "  SPECRET d=3  ↔  index_topk_freq=3  (66% 跳过)"
echo "  SPECRET d=4  ↔  index_topk_freq=4  (75% 跳过)"
