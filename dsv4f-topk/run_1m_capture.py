#!/usr/bin/env python3
"""Capture indexer topk for synthetic 1M-token requests.

Server: TP4-only, --enable-return-indexer-topk, SGLANG_DUMP_KV_DIR set.
topk (decode-only) returned via HTTP; C4 KV + indexer K dumped server-side.
"""
import base64
import json
import os
import time
import urllib.request

import numpy as np

DATA = "/cpfs02/user/lgd/data/infinitebench/synth_1m.jsonl"
MODEL_CONFIG = "/cpfs02/user/lgd/models/dsv4f/config.json"
OUT_DIR = "/cpfs02/user/lgd/sglang-fork/dsv4f-topk/topk_1m"
URL = "http://localhost:30000/generate"
MAX_NEW_TOKENS = 128
REQUEST_TIMEOUT = 7200  # 1M prefill is slow

os.makedirs(OUT_DIR, exist_ok=True)
cfg = json.load(open(MODEL_CONFIG))
NL = sum(1 for r in (cfg.get("compress_ratios") or []) if r == 4)
TK = int(cfg["index_topk"])
print(f"num_indexer_layers={NL}, index_topk={TK}")

reqs = [json.loads(l) for l in open(DATA)]
print(f"Loaded {len(reqs)} 1M requests")


def post(payload):
    data = json.dumps(payload).encode()
    r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


manifest = []
t0 = time.time()
for ex in reqs:
    idx = ex["idx"]
    payload = {
        "text": ex["prompt"],
        "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
        "return_indexer_topk": True,
    }
    try:
        res = post(payload)
    except Exception as e:
        print(f"[{idx}] ERROR: {str(e)[:120]}", flush=True)
        manifest.append({"idx": idx, "error": str(e)})
        continue
    mi = res.get("meta_info", {}) or {}
    ptoks = mi.get("prompt_tokens")
    ctoks = mi.get("completion_tokens")
    rid = mi.get("id")
    b64 = mi.get("indexer_topk")
    rec = {"idx": idx, "rid": rid, "prompt_tokens": ptoks, "completion_tokens": ctoks,
           "n_docs": ex["n_docs"]}
    if b64:
        flat = np.frombuffer(base64.b64decode(b64), dtype=np.int32)
        per = NL * TK
        rows = flat.size // per
        topk = flat.reshape(rows, NL, TK)
        out = os.path.join(OUT_DIR, f"req_{idx:02d}.npz")
        np.savez_compressed(out, topk_index=topk, idx=idx,
                            prompt_tokens=ptoks if ptoks else -1,
                            completion_tokens=ctoks if ctoks else -1,
                            rid=rid or "")
        rec["topk_rows"] = int(rows)
        rec["file"] = os.path.basename(out)
        rec["decode_ok"] = (ctoks is not None and rows == ctoks)
        print(f"[{idx}] ok rows={rows} ptoks={ptoks} ctoks={ctoks} "
              f"t={time.time()-t0:.0f}s", flush=True)
    else:
        rec["error"] = "no indexer_topk"
        print(f"[{idx}] WARN no indexer_topk ptoks={ptoks}", flush=True)
    manifest.append(rec)

with open(os.path.join(OUT_DIR, "manifest.jsonl"), "w") as f:
    for r in manifest:
        f.write(json.dumps(r) + "\n")
print(f"\nDone. {len([m for m in manifest if 'error' not in m])}/{len(reqs)} ok. "
      f"Total {time.time()-t0:.0f}s. Out: {OUT_DIR}")
