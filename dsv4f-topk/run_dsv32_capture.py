#!/usr/bin/env python3
"""Capture dsv32 (DeepSeek-V3.2 DSA) indexer topk for LongBench-v2 first 100 requests.

DSA: 61 indexer layers, index_topk=2048 (vs dsv4f: 21 layers, 512).
Server must run with --enable-return-indexer-topk --enable-dp-attention.
"""
import base64
import json
import os
import re
import time
import urllib.request

import numpy as np

DATA_PATH = "/cpfs02/user/danqing.zq/data/longbench/longbench_v2.jsonl"
MODEL_CONFIG = "/cpfs02/user/lgd/models/dsv32/config.json"
OUT_DIR = "/cpfs02/user/lgd/sglang-fork/dsv4f-topk/dsv32_topk"
URL = "http://localhost:30000/generate"
N = 100
MAX_NEW_TOKENS = 512
REQUEST_TIMEOUT = 3600

os.makedirs(OUT_DIR, exist_ok=True)

cfg = json.load(open(MODEL_CONFIG))
# DSA: all layers have indexer
NUM_INDEXER_LAYERS = cfg.get("num_hidden_layers", 61)
INDEX_TOPK = int(cfg.get("index_topk", 2048))
print(f"dsv32: num_indexer_layers={NUM_INDEXER_LAYERS}, index_topk={INDEX_TOPK}")

examples = []
with open(DATA_PATH) as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        examples.append(json.loads(line))
print(f"Loaded first {len(examples)} records")


def extract_answer(response):
    response = response.replace("*", "")
    for pattern in [
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
        r"answer\s+is\s*\(?([A-D])\)?",
        r"\b([A-D])\b",
    ]:
        m = re.search(pattern, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def post(payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


manifest = []
t0 = time.time()

for idx, ex in enumerate(examples):
    gold = str(ex.get("answer", "")).strip().upper()
    payload = {
        "text": ex["prompt"],
        "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
        "return_indexer_topk": True,
    }
    res = None
    for attempt in range(3):
        try:
            res = post(payload)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"[{idx + 1}/{N}] ERROR: {str(e)[:120]}", flush=True)
                manifest.append({"idx": idx, "error": str(e)})

    if res is None:
        continue

    mi = res.get("meta_info", {}) or {}
    text = res.get("text", "") or ""
    pred = extract_answer(text)
    ptoks = mi.get("prompt_tokens")
    ctoks = mi.get("completion_tokens")

    rec = {
        "idx": idx, "prompt_tokens": ptoks, "completion_tokens": ctoks,
        "predicted": pred, "answer_gold": gold,
        "is_correct": pred == gold,
    }

    b64 = mi.get("indexer_topk")
    if b64:
        flat = np.frombuffer(base64.b64decode(b64), dtype=np.int32)
        per_row = NUM_INDEXER_LAYERS * INDEX_TOPK
        rows = flat.size // per_row
        topk = flat.reshape(rows, NUM_INDEXER_LAYERS, INDEX_TOPK)
        out_path = os.path.join(OUT_DIR, f"req_{idx:04d}.npz")
        np.savez_compressed(
            out_path, topk_index=topk, idx=idx,
            prompt_tokens=ptoks if ptoks else -1,
            completion_tokens=ctoks if ctoks else -1,
            num_indexer_layers=NUM_INDEXER_LAYERS,
            index_topk=INDEX_TOPK,
        )
        rec["topk_rows"] = int(rows)
        rec["file"] = os.path.basename(out_path)
        print(f"[{idx + 1}/{N}] ok rows={rows} ptoks={ptoks} ctoks={ctoks} "
              f"t={time.time() - t0:.0f}s", flush=True)
    else:
        rec["error"] = "no indexer_topk"
        print(f"[{idx + 1}/{N}] WARN no indexer_topk ptoks={ptoks}", flush=True)

    manifest.append(rec)

with open(os.path.join(OUT_DIR, "manifest.jsonl"), "w") as f:
    for rec in manifest:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

ok = sum(1 for m in manifest if "error" not in m)
print(f"\nDone: {ok}/{N} ok. Total {time.time() - t0:.0f}s. Out: {OUT_DIR}")
