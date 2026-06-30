#!/usr/bin/env python3
"""Simulate hisparse load/eviction over captured indexer topk -> cache hit rate.

Faithful to the ground-truth swap-in kernel
  python/sglang/jit_kernel/csrc/hisparse.cuh
and the buffer/LRU state in
  python/sglang/srt/managers/hisparse_coordinator.py

Model
-----
Per (request, C4 indexer layer): an LRU "hot buffer" of B = device_buffer_size
COMPRESSED-token slots + 1 reserved slot pinning the newest compressed token.

  compress_ratio = 4 (C4).  compressed_seq_len = full_seq_len // 4.
  Captured topk indices are COMPRESSED positions in [0, compressed_seq_len); -1 = pad.

  Fast path (cs <= B): whole sequence resident in order -> every selected pos is a HIT,
    no host load (hisparse.cuh:207-216).
  Slow path (cs > B):
    * newest = cs-1 is ALWAYS a hit (reserved slot, excluded from LRU; hisparse.cuh:256-267).
    * a selected pos is a HIT iff currently resident in the B-slot buffer, else a MISS.
    * hit/miss counted BEFORE this step's load (a miss is a miss even though it is then
      loaded): total_misses = |selected| - hits (hisparse.cuh:361,409).
    * LRU update: hits -> MRU; misses -> load, evicting the LRU among residents NOT hit
      this step; loaded tokens become MRU (hisparse.cuh:341-456).
  Init residency (staging prefill, alloc_device_buffer arange): slots 0..B-1 hold
    compressed tokens 0..B-1.  init_mode: first_b (default, staging) | empty (RDMA direct).

We captured DECODE rows only: row k -> query at full pos prompt_tokens-1+k,
  full_seq_len = prompt_tokens + k  =>  compressed_seq_len = (prompt_tokens + k)//4.
"""
import argparse
import glob
import json
import os
from collections import OrderedDict

import numpy as np

COMPRESS_RATIO = 4
DIR = "/cpfs02/user/lgd/sglang-fork/dsv4f-topk"


def sim_layer(topk_layer, prompt_tokens, B, init_mode):
    """topk_layer: (rows, index_topk) int32 for one layer. Returns (hits, total)."""
    rows = topk_layer.shape[0]
    resident = None  # OrderedDict token->None, order = LRU(front)..MRU(back); lazy init
    hits = 0
    total = 0
    for k in range(rows):
        cs = (prompt_tokens + k) // COMPRESS_RATIO
        row = topk_layer[k]
        sel = row[(row >= 0) & (row < cs)]
        if sel.size == 0:
            continue
        if cs <= B:
            hits += int(sel.size)
            total += int(sel.size)
            continue
        if resident is None:  # entering slow path; buffer was full with 0..B-1
            if init_mode == "empty":
                resident = OrderedDict()
            else:  # first_b (code default)
                resident = OrderedDict.fromkeys(range(B))
        newest = cs - 1
        step_hits = []
        step_misses = []
        for p in sel.tolist():
            total += 1
            if p == newest:
                hits += 1  # pinned reserved slot
            elif p in resident:
                hits += 1
                step_hits.append(p)
            else:
                step_misses.append(p)
        for p in step_hits:  # hits -> MRU (protected from this step's eviction)
            resident.move_to_end(p)
        for p in step_misses:  # load: evict LRU non-hit, insert MRU
            if len(resident) >= B:
                resident.popitem(last=False)
            resident[p] = None
    return hits, total


def sim_request(topk, prompt_tokens, B, init_mode):
    """topk: (rows, n_layers, index_topk). Returns per-layer list of (hits,total)."""
    n_layers = topk.shape[1]
    return [sim_layer(topk[:, L, :], prompt_tokens, B, init_mode) for L in range(n_layers)]


def load_all():
    files = sorted(glob.glob(os.path.join(DIR, "req_*.npz")))
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        out.append(
            dict(
                file=os.path.basename(f),
                topk=d["topk_index"],
                prompt_tokens=int(d["prompt_tokens"]),
                completion_tokens=int(d["completion_tokens"]),
                domain=str(d["domain"]),
            )
        )
    return out


def run_config(data, B, init_mode):
    """Returns dict with overall, per-layer, and per-request hit rates."""
    n_layers = data[0]["topk"].shape[1]
    layer_hits = np.zeros(n_layers, dtype=np.int64)
    layer_tot = np.zeros(n_layers, dtype=np.int64)
    per_req = []
    tot_h = tot_t = 0
    for ex in data:
        pl = sim_request(ex["topk"], ex["prompt_tokens"], B, init_mode)
        rh = sum(h for h, _ in pl)
        rt = sum(t for _, t in pl)
        for L, (h, t) in enumerate(pl):
            layer_hits[L] += h
            layer_tot[L] += t
        per_req.append(
            dict(
                file=ex["file"],
                prompt_tokens=ex["prompt_tokens"],
                compressed_len=ex["prompt_tokens"] // COMPRESS_RATIO,
                fast_path=(ex["prompt_tokens"] // COMPRESS_RATIO) <= B,
                hit_rate=(rh / rt) if rt else float("nan"),
                selected=rt,
            )
        )
        tot_h += rh
        tot_t += rt
    return dict(
        B=B,
        init_mode=init_mode,
        overall_hit_rate=tot_h / tot_t if tot_t else float("nan"),
        total_selected=tot_t,
        per_layer_hit_rate=[
            (layer_hits[L] / layer_tot[L]) if layer_tot[L] else float("nan")
            for L in range(n_layers)
        ],
        per_request=per_req,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-sizes", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--default-b", type=int, default=4096)
    args = ap.parse_args()

    data = load_all()
    n_layers = data[0]["topk"].shape[1]
    print(f"Loaded {len(data)} requests, {n_layers} indexer layers, compress_ratio={COMPRESS_RATIO}")
    print(f"prompt_tokens: min={min(e['prompt_tokens'] for e in data)} "
          f"max={max(e['prompt_tokens'] for e in data)}")

    results = {}

    # Primary: default B, both init modes
    print("\n=== Primary: B=%d ===" % args.default_b)
    for init_mode in ("first_b", "empty"):
        r = run_config(data, args.default_b, init_mode)
        results[f"B{args.default_b}_{init_mode}"] = r
        nfast = sum(1 for p in r["per_request"] if p["fast_path"])
        print(f"  init={init_mode:8s}  overall hit rate = {r['overall_hit_rate']*100:.2f}%  "
              f"(fast-path reqs: {nfast}/{len(data)}, total selected={r['total_selected']:,})")

    # Buffer-size sweep (staging init)
    print("\n=== B sweep (init=first_b) ===")
    sweep = {}
    for B in args.buffer_sizes:
        r = run_config(data, B, "first_b")
        sweep[B] = r["overall_hit_rate"]
        results[f"B{B}_first_b"] = r
        print(f"  B={B:6d} (={B*COMPRESS_RATIO:7d} full tokens)  hit rate = {r['overall_hit_rate']*100:.2f}%")

    # Per-layer (default B, first_b)
    base = results[f"B{args.default_b}_first_b"]
    print(f"\n=== Per-layer hit rate (B={args.default_b}, first_b) ===")
    pl = base["per_layer_hit_rate"]
    print("  layer: " + " ".join(f"{i:02d}" for i in range(n_layers)))
    print("  rate%: " + " ".join(f"{x*100:.0f}" for x in pl))
    print(f"  min={min(pl)*100:.1f}% max={max(pl)*100:.1f}% mean={np.mean(pl)*100:.1f}%")

    # Per-request buckets by prompt length (default B, first_b)
    print(f"\n=== Hit rate by prompt-length bucket (B={args.default_b}, first_b) ===")
    buckets = [(0, 16384), (16384, 32768), (32768, 65536), (65536, 131072)]
    for lo, hi in buckets:
        sub = [p for p in base["per_request"] if lo <= p["prompt_tokens"] < hi]
        if not sub:
            continue
        wh = sum(p["hit_rate"] * p["selected"] for p in sub)
        ws = sum(p["selected"] for p in sub)
        print(f"  [{lo:6d},{hi:6d}) full tok: {len(sub):3d} reqs, "
              f"hit rate = {wh/ws*100:.2f}% (micro-avg)")

    # save (drop bulky per_request from some to keep file small; keep for default)
    out = {
        "compress_ratio": COMPRESS_RATIO,
        "num_layers": n_layers,
        "num_requests": len(data),
        "default_b": args.default_b,
        "sweep_overall": {str(k): v for k, v in sweep.items()},
        "configs": {
            k: {kk: vv for kk, vv in r.items() if kk != "per_request"}
            for k, r in results.items()
        },
        "default_per_request": base["per_request"],
    }
    outpath = os.path.join(DIR, "hisparse_sim_results.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {outpath}")


if __name__ == "__main__":
    main()
