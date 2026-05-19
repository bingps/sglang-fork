#!/bin/bash

SGLANG_NSA_ENABLE_SPECRET=1 python3 -m sglang.launch_server --model-path /data2/dsv32awq/ --tp 8 --mem-fraction-static=0.9 --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3 --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --cuda-graph-max-bs 16 &> server.log &

until grep -q "The server is fired up and ready to roll!" server.log; do sleep 5; done

for i in 1 2 3; do python3 -m sglang.bench_serving \
  --dataset-name longbench_v2 --dataset-path /data2/LongBench-v2/data.jsonl \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --num-prompts 7 \
  --model /data2/dsv32awq \
  --sharegpt-context-len 150000 \
  --sharegpt-output-len 1024 \
  --output-file bench_longbench2_mtp2_req7_sr.jsonl; 
done

bash scripts/killall_sglang.sh
sleep 10

python3 -m sglang.launch_server --model-path /data2/dsv32awq/ --tp 8 --mem-fraction-static=0.9 --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3 --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --cuda-graph-max-bs 16 &> server.log &

until grep -q "The server is fired up and ready to roll!" server.log; do sleep 5; done

for i in 1 2 3; do python3 -m sglang.bench_serving \
  --dataset-name longbench_v2 --dataset-path /data2/LongBench-v2/data.jsonl \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --num-prompts 7 \
  --model /data2/dsv32awq \
  --sharegpt-context-len 150000 \
  --sharegpt-output-len 1024 \
  --output-file bench_longbench2_mtp2_req7_nosr.jsonl; 
done
