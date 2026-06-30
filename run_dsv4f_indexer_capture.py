#!/usr/bin/env python3
"""Capture dsv4f CSA indexer topk (decode tokens) for the first 100 LongBench-v2 records.

Server must run with --enable-return-indexer-topk --enable-dp-attention.
batch_result_processor._maybe_collect_indexer_topk is patched to return only the
decode-token rows, so meta_info.indexer_topk already covers just the generated tokens.

Per request -> dsv4f-topk/req_XXXX.npz:
  topk_index: int32 (completion_tokens, num_indexer_layers=21, index_topk=512); -1 = padding
  + metadata (idx, prompt/completion tokens, domain, difficulty, gold/pred answer, response)
Also dsv4f-topk/manifest.jsonl with one summary line per request.

NOTE: the runtime only exposes the indexer's topk INDICES, not its scores
(scores are discarded in-kernel). Per the agreed scope, only indices are stored.
"""
import base64
import json
import os
import re
import time
import urllib.request

import numpy as np

DATA_PATH = "/cpfs02/user/danqing.zq/data/longbench/longbench_v2.jsonl"
MODEL_CONFIG = "/cpfs02/user/lgd/models/dsv4f/config.json"
OUT_DIR = "/cpfs02/user/lgd/sglang-fork/dsv4f-topk"
URL = "http://localhost:30000/generate"
N = 100
MAX_NEW_TOKENS = 512
REQUEST_TIMEOUT = 3600  # long prefills (up to ~110k tokens) + 512 decode

os.makedirs(OUT_DIR, exist_ok=True)

cfg = json.load(open(MODEL_CONFIG))
NUM_INDEXER_LAYERS = sum(1 for r in (cfg.get("compress_ratios") or []) if r == 4)
INDEX_TOPK = int(cfg["index_topk"])
print(f"num_indexer_layers={NUM_INDEXER_LAYERS}, index_topk={INDEX_TOPK}")

examples = []
with open(DATA_PATH) as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        examples.append(json.loads(line))
print(f"Loaded first {len(examples)} records (no length filter)")


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
correct = 0
errors = 0
t0 = time.time()

for idx, ex in enumerate(examples):
    gold = str(ex.get("answer", "")).strip().upper()
    rec = {
        "idx": idx,
        "domain": ex.get("domain", ""),
        "difficulty": ex.get("difficulty", ""),
        "answer_gold": gold,
        "qwen3_prompt_tokens": ex.get("token_len", {}).get("qwen3", {}).get("prompt"),
    }
    payload = {
        "text": ex["prompt"],
        "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
        "return_indexer_topk": True,
    }
    res = None
    err = None
    for attempt in range(3):
        try:
            res = post(payload)
            break
        except Exception as e:  # noqa: BLE001
            err = e
            if attempt < 2:
                time.sleep(5)
    if res is None:
        rec["error"] = str(err)
        errors += 1
        manifest.append(rec)
        print(f"[{idx + 1}/{N}] ERROR: {str(err)[:120]}")
        continue

    mi = res.get("meta_info", {}) or {}
    text = res.get("text", "") or ""
    pred = extract_answer(text)
    ptoks = mi.get("prompt_tokens")
    ctoks = mi.get("completion_tokens")
    rec.update(
        {"predicted": pred, "is_correct": pred == gold,
         "prompt_tokens": ptoks, "completion_tokens": ctoks}
    )

    b64 = mi.get("indexer_topk")
    if not b64:
        rec["error"] = "no indexer_topk in meta_info"
        errors += 1
        manifest.append(rec)
        print(f"[{idx + 1}/{N}] WARN: no indexer_topk (ptoks={ptoks})")
        continue

    flat = np.frombuffer(base64.b64decode(b64), dtype=np.int32)
    per_row = NUM_INDEXER_LAYERS * INDEX_TOPK
    if flat.size % per_row != 0:
        rec["error"] = f"size {flat.size} not divisible by {per_row}"
        errors += 1
        manifest.append(rec)
        print(f"[{idx + 1}/{N}] ERROR bad size {flat.size}")
        continue
    rows = flat.size // per_row
    topk = flat.reshape(rows, NUM_INDEXER_LAYERS, INDEX_TOPK)
    rec["topk_rows"] = int(rows)
    rec["decode_only_ok"] = (ctoks is not None and rows == ctoks)

    out_path = os.path.join(OUT_DIR, f"req_{idx:04d}.npz")
    np.savez_compressed(
        out_path,
        topk_index=topk,
        idx=idx,
        prompt_tokens=ptoks if ptoks is not None else -1,
        completion_tokens=ctoks if ctoks is not None else -1,
        domain=rec["domain"],
        difficulty=rec["difficulty"],
        answer_gold=gold,
        predicted=pred or "",
        response=text[:1000],
        num_indexer_layers=NUM_INDEXER_LAYERS,
        index_topk=INDEX_TOPK,
    )
    rec["file"] = os.path.basename(out_path)
    if rec["is_correct"]:
        correct += 1
    manifest.append(rec)
    flag = "" if rec["decode_only_ok"] else f" !!rows({rows})!=ctoks({ctoks})"
    print(
        f"[{idx + 1}/{N}] ok rows={rows} pred={pred} gold={gold} "
        f"ptoks={ptoks} ctoks={ctoks} t={time.time() - t0:.0f}s{flag}",
        flush=True,
    )

with open(os.path.join(OUT_DIR, "manifest.jsonl"), "w") as f:
    for rec in manifest:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

valid = N - errors
print("=" * 50)
print(f"Done: {N} requests, {errors} errors")
if valid:
    print(f"QA accuracy on valid: {correct}/{valid} ({correct / valid * 100:.1f}%)")
print(f"Outputs: {OUT_DIR}/ (req_XXXX.npz + manifest.jsonl)")
print(f"Total time {time.time() - t0:.0f}s")
