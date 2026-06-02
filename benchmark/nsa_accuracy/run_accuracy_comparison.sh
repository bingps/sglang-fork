#!/bin/bash
# =============================================================================
# NSA Accuracy Comparison:
#   Baseline vs SPECRET (d=2,3,4) vs index_topk_freq (2,3,4)
#
# 按相同的 indexer 跳过率对比:
#   SPECRET d=N → verify 阶段 (N-1)/N token 复用 topk
#   index_topk_freq=N → 所有阶段 (N-1)/N 层复用 topk
#
# 参考 PR: https://github.com/sgl-project/sglang/pull/21502
# =============================================================================

set -e

# ===================== 配置区 =====================
MODEL_PATH="/data2/dsv32awq/"
TP=8
PORT=30000
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
MEM_FRACTION=0.9
CUDA_GRAPH_MAX_BS=16

# 评测配置
EVAL_NAME="${EVAL_NAME:-mmlu}"       # 可选: mmlu, gsm8k, mgsm_en, longbench_v2
NUM_EXAMPLES="${NUM_EXAMPLES:-5000}" # MMLU 5000, GSM8K 可用 200
NUM_THREADS="${NUM_THREADS:-512}"

# 结果输出目录
RESULT_DIR="benchmark/nsa_accuracy/results"
mkdir -p "$RESULT_DIR"

# 公共服务器参数（不含 speculative-num-draft-tokens，由各配置单独指定）
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

# ===================== 辅助函数 =====================

wait_for_server() {
    echo "[INFO] Waiting for server to be ready..."
    local max_wait=600
    local elapsed=0
    # 使用 curl -sf: -s 静默, -f 对HTTP错误(如503)返回非零退出码
    while ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "[ERROR] Server failed to start within ${max_wait}s"
            tail -50 "${RESULT_DIR}/server_${current_config}.log" 2>/dev/null
            exit 1
        fi
        if [ $((elapsed % 60)) -eq 0 ]; then
            echo "[INFO] Still waiting... ${elapsed}s elapsed"
        fi
    done
    echo "[INFO] Server is ready! (took ${elapsed}s)"
}

kill_server() {
    echo "[INFO] Killing server..."
    bash scripts/killall_sglang.sh 2>/dev/null || true
    sleep 10
}

run_eval() {
    local config_name="$1"
    local result_file="${RESULT_DIR}/${EVAL_NAME}_${config_name}.json"

    echo "============================================="
    echo "[EVAL] Config: ${config_name}"
    echo "[EVAL] Benchmark: ${EVAL_NAME}"
    echo "[EVAL] Num examples: ${NUM_EXAMPLES}"
    echo "============================================="

    if [ "$EVAL_NAME" = "ceval" ]; then
        # CEval 使用 sglang 原生 API
        local ceval_args=(
            --data-path "ceval/ceval-exam"
            --host "$HOST"
            --port "$PORT"
            --parallel "$NUM_THREADS"
            --result-file "${RESULT_DIR}/${EVAL_NAME}_${config_name}.jsonl"
        )
        if [ "$NUM_EXAMPLES" != "0" ] && [ "$NUM_EXAMPLES" != "5000" ]; then
            ceval_args+=(--num-questions "$NUM_EXAMPLES")
        fi

        # 保留代理(用于下载数据集)，但设置 no_proxy 跳过本地连接
        # 使用 HF_DATASETS_OFFLINE=1 强制使用缓存，避免 HF API 限流
        env no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost" \
            HF_DATASETS_OFFLINE=1 \
        python3 benchmark/ceval/bench_sglang.py "${ceval_args[@]}" \
            2>&1 | tee "${RESULT_DIR}/${EVAL_NAME}_${config_name}.log"

        # 提取分数 (CEval 输出格式: "Accuracy: 0.xxx")
        local score=$(grep -oP 'Accuracy: \K[0-9.]+' "${RESULT_DIR}/${EVAL_NAME}_${config_name}.log" || echo "N/A")
    else
        # 其他 benchmark 使用 OpenAI 兼容 API
        env -u HTTP_PROXY -u http_proxy -u https_proxy -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
        python3 -m sglang.test.run_eval \
            --base-url "${BASE_URL}" \
            --eval-name "${EVAL_NAME}" \
            --num-examples "${NUM_EXAMPLES}" \
            --num-threads "${NUM_THREADS}" \
            2>&1 | tee "${RESULT_DIR}/${EVAL_NAME}_${config_name}.log"

        # 提取分数 (run_eval 输出格式: "Score: 0.xxx")
        local score=$(grep -oP 'Score: \K[0-9.]+' "${RESULT_DIR}/${EVAL_NAME}_${config_name}.log" || echo "N/A")
    fi

    echo "[RESULT] ${config_name}: score = ${score}"
    echo "${config_name},${score}" >> "${RESULT_DIR}/${EVAL_NAME}_summary.csv"
}

# 启动服务器的通用函数
# 参数: $1=config_name, $2=draft_tokens, $3=extra_env (可选), $4=index_topk_freq (可选)
launch_and_eval() {
    local config_name="$1"
    local draft_tokens="$2"
    local extra_env="$3"
    local index_topk_freq="$4"
    local step_num="$5"
    local total_steps="$6"

    echo ""
    echo "###############################################"
    echo "# [${step_num}/${total_steps}] ${config_name}"
    echo "#   draft_tokens=${draft_tokens} env=${extra_env} index_topk_freq=${index_topk_freq}"
    echo "###############################################"
    kill_server

    # 确保端口已经释放
    local port_wait=0
    while ss -tlnp | grep -q ":${PORT} " 2>/dev/null; do
        sleep 2
        port_wait=$((port_wait + 2))
        if [ $port_wait -ge 30 ]; then
            echo "[ERROR] Port ${PORT} still in use after 30s"
            exit 1
        fi
    done

    # 设置 current_config 供 wait_for_server 使用
    current_config="$config_name"

    local cmd_args=("${BASE_ARGS[@]}" --speculative-num-draft-tokens "$draft_tokens")
    if [ -n "$index_topk_freq" ]; then
        cmd_args+=(--json-model-override-args "{\"index_topk_freq\": ${index_topk_freq}}")
    fi

    # 启动服务器时清除代理变量（避免 warmup 请求走代理）
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
    echo "[INFO] Server PID: ${server_pid}"

    # 等待服务器就绪，同时检查进程存活
    echo "[INFO] Waiting for server to be ready..."
    local max_wait=600
    local elapsed=0
    while ! env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u no_proxy -u NO_PROXY \
            curl -sf "${BASE_URL}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "[ERROR] Server process died during startup!"
            tail -30 "${RESULT_DIR}/server_${config_name}.log"
            exit 1
        fi
        if [ $elapsed -ge $max_wait ]; then
            echo "[ERROR] Server failed to start within ${max_wait}s"
            tail -50 "${RESULT_DIR}/server_${config_name}.log"
            exit 1
        fi
        if [ $((elapsed % 60)) -eq 0 ]; then
            echo "[INFO] Still waiting... ${elapsed}s elapsed"
        fi
    done
    echo "[INFO] Server is ready! (took ${elapsed}s)"

    run_eval "$config_name"
    kill_server
}

# ===================== 主流程 =====================

TOTAL_STEPS=7
echo "EVAL_NAME=${EVAL_NAME}, NUM_EXAMPLES=${NUM_EXAMPLES}"
echo "Total configurations: ${TOTAL_STEPS}"
echo ""
echo "config_name,score" > "${RESULT_DIR}/${EVAL_NAME}_summary.csv"

# --- 1. Baseline (d=2, 无近似) ---
launch_and_eval "baseline_d2" 2 "" "" 1 $TOTAL_STEPS

# --- 2. SPECRET d=2 (verify 阶段 50% token 复用) ---
launch_and_eval "specret_d2" 2 "SGLANG_NSA_ENABLE_SPECRET=1" "" 2 $TOTAL_STEPS

# --- 3. SPECRET d=3 (verify 阶段 66% token 复用) ---
launch_and_eval "specret_d3" 3 "SGLANG_NSA_ENABLE_SPECRET=1" "" 3 $TOTAL_STEPS

# --- 4. SPECRET d=4 (verify 阶段 75% token 复用) ---
launch_and_eval "specret_d4" 4 "SGLANG_NSA_ENABLE_SPECRET=1" "" 4 $TOTAL_STEPS

# --- 5. index_topk_freq=2 (所有阶段 50% 层复用) ---
launch_and_eval "index_topk_freq_2" 2 "" "2" 5 $TOTAL_STEPS

# --- 6. index_topk_freq=3 (所有阶段 66% 层复用) ---
launch_and_eval "index_topk_freq_3" 2 "" "3" 6 $TOTAL_STEPS

# --- 7. index_topk_freq=4 (所有阶段 75% 层复用) ---
launch_and_eval "index_topk_freq_4" 2 "" "4" 7 $TOTAL_STEPS

# ===================== 汇总结果 =====================
echo ""
echo "============================================="
echo "              ACCURACY SUMMARY"
echo "============================================="
echo ""
echo "跳过率对应关系:"
echo "  SPECRET d=2  ↔  index_topk_freq=2  (50% 跳过)"
echo "  SPECRET d=3  ↔  index_topk_freq=3  (66% 跳过)"
echo "  SPECRET d=4  ↔  index_topk_freq=4  (75% 跳过)"
echo ""
column -t -s',' "${RESULT_DIR}/${EVAL_NAME}_summary.csv"
echo ""
echo "详细结果保存在: ${RESULT_DIR}/"
echo "============================================="
