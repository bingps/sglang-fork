#!/usr/bin/env python3
"""Test: can we approximate greedy_cooc using only a FEW steps (simulating prefill window)?

For each (request, layer): build W from only the first K steps of topk (K=1,2,4,8,16,all),
do greedy reorder, measure block coverage on ALL steps. If K=1 already close to K=all,
then a single prefill chunk's topk suffices.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/cpfs02/user/lgd/sglang-fork")
from sim_hisparse_hitrate import COMPRESS_RATIO, DIR, load_all

BS = 64
DEVICE = "cuda:0"
WARMUP_STEPS = [1, 2, 4, 8, 16, 32]  # how many steps used to build W


def blocks_touched(sel, bs, perm=None):
    if perm is not None:
        sel = perm[sel]
    return len(np.unique(sel // bs))


def oracle_blocks(n, bs):
    return int(np.ceil(n / bs))


def _build_cooc_gpu(topk_L, n, max_steps=None):
    steps = topk_L.shape[0] if max_steps is None else min(max_steps, topk_L.shape[0])
    rows_list, cols_list = [], []
    for s in range(steps):
        sel = topk_L[s]; sel = np.unique(sel[(sel >= 0) & (sel < n)])
        if len(sel) > 512:
            sel = sel[:512]
        r = np.repeat(sel, len(sel))
        c = np.tile(sel, len(sel))
        rows_list.append(r)
        cols_list.append(c)
    if not rows_list:
        return torch.sparse_coo_tensor(
            torch.empty(2, 0, dtype=torch.long), torch.empty(0), (n, n), device=DEVICE)
    rows = np.concatenate(rows_list).astype(np.int64)
    cols = np.concatenate(cols_list).astype(np.int64)
    indices = torch.from_numpy(np.stack([rows, cols])).to(DEVICE)
    values = torch.ones(len(rows), dtype=torch.float32, device=DEVICE)
    cooc = torch.sparse_coo_tensor(indices, values, (n, n), device=DEVICE).coalesce()
    return cooc


def mk_greedy_perm(topk_L, n, bs, max_steps=None):
    freq = np.zeros(n, dtype=np.int32)
    steps = topk_L.shape[0] if max_steps is None else min(max_steps, topk_L.shape[0])
    for s in range(steps):
        sel = topk_L[s]; sel = sel[(sel >= 0) & (sel < n)]
        np.add.at(freq, sel, 1)

    cooc = _build_cooc_gpu(topk_L, n, max_steps)
    freq_gpu = torch.from_numpy(freq).float().to(DEVICE)
    placed = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    order = []

    while len(order) < n:
        masked_freq = freq_gpu.clone()
        masked_freq[placed] = -1
        seed = int(masked_freq.argmax().item())
        if masked_freq[seed] < 0:
            break
        indicator = torch.zeros(n, dtype=torch.float32, device=DEVICE)
        indicator[seed] = 1.0
        scores = torch.sparse.mm(cooc, indicator.unsqueeze(1)).squeeze(1)
        scores = scores + freq_gpu * 0.01
        scores[placed] = -1
        scores[seed] = -1
        need = min(bs - 1, n - len(order) - 1)
        if need > 0:
            _, top_idx = scores.topk(need)
            block_rest = top_idx[scores[top_idx] > -1].cpu().tolist()
        else:
            block_rest = []
        block = [seed] + block_rest
        for b in block:
            placed[b] = True
        order.extend(block)

    unplaced = torch.where(~placed)[0].cpu().tolist()
    order.extend(unplaced)
    perm = np.empty(n, dtype=np.int32)
    for new_pos, orig in enumerate(order):
        perm[orig] = new_pos
    return perm


def mk_freq_perm(topk_L, n):
    freq = np.zeros(n, dtype=np.int32)
    for s in range(topk_L.shape[0]):
        sel = topk_L[s]; sel = sel[(sel >= 0) & (sel < n)]
        np.add.at(freq, sel, 1)
    order = np.argsort(-freq)
    perm = np.empty(n, dtype=np.int32)
    perm[order] = np.arange(n, dtype=np.int32)
    return perm


def main():
    data = load_all()
    N = len(data)
    NL = data[0]["topk"].shape[1]
    print(f"Loaded {N} requests, {NL} layers, bs={BS}")

    # results[k] = list of blocks per (step,layer) evaluated on ALL steps
    results = {k: [] for k in WARMUP_STEPS}
    results["all"] = []
    results["freq"] = []
    results["original"] = []
    results["oracle"] = []

    t_global = time.time()
    for ri in range(N):
        ex = data[ri]
        topk = ex["topk"]
        ct = (ex["prompt_tokens"] + ex["completion_tokens"]) // COMPRESS_RATIO
        total_steps = topk.shape[0]
        t0 = time.time()

        for L in range(NL):
            topk_L = topk[:, L, :]

            # build perms with varying warmup steps
            perms = {}
            for k in WARMUP_STEPS:
                if k <= total_steps:
                    perms[k] = mk_greedy_perm(topk_L, ct, BS, max_steps=k)
                else:
                    perms[k] = None
            perms["all"] = mk_greedy_perm(topk_L, ct, BS, max_steps=None)
            perms["freq"] = mk_freq_perm(topk_L, ct)

            # evaluate ALL perms on ALL steps
            for s in range(total_steps):
                sel = topk_L[s]; sel = sel[(sel >= 0) & (sel < ct)]
                if len(sel) == 0:
                    continue
                results["original"].append(blocks_touched(sel, BS))
                results["oracle"].append(oracle_blocks(len(sel), BS))
                results["freq"].append(blocks_touched(sel, BS, perms["freq"]))
                results["all"].append(blocks_touched(sel, BS, perms["all"]))
                for k in WARMUP_STEPS:
                    if perms[k] is not None:
                        results[k].append(blocks_touched(sel, BS, perms[k]))
                    else:
                        results[k].append(blocks_touched(sel, BS))  # fallback to original

        elapsed = time.time() - t0
        if (ri + 1) % 10 == 0 or ri == 0:
            print(f"  [{ri+1:3d}/{N}] c4={ct:5d} steps={total_steps:3d} ({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t_global
    ns = len(results["original"])
    print(f"\nDone in {total_time:.0f}s. Samples: {ns}")

    print(f"\n{'='*70}")
    print(f"bs={BS}: mean blocks by warmup steps used to build W")
    print(f"{'='*70}")
    om = np.mean(results["original"])
    orc = np.mean(results["oracle"])
    print(f"  {'method':<20} {'blocks':>8} {'reduction':>10} {'scatter':>10}")
    print(f"  {'-'*52}")
    print(f"  {'original':<20} {om:>8.1f} {'—':>10} {om/orc:>9.1f}x")
    print(f"  {'freq':<20} {np.mean(results['freq']):>8.1f} "
          f"{(1-np.mean(results['freq'])/om)*100:>+9.0f}% {np.mean(results['freq'])/orc:>9.1f}x")
    for k in WARMUP_STEPS:
        v = np.mean(results[k])
        print(f"  {'greedy(W from '+str(k)+')':<20} {v:>8.1f} "
              f"{(1-v/om)*100:>+9.0f}% {v/orc:>9.1f}x")
    v_all = np.mean(results["all"])
    print(f"  {'greedy(W from all)':<20} {v_all:>8.1f} "
          f"{(1-v_all/om)*100:>+9.0f}% {v_all/orc:>9.1f}x")
    print(f"  {'oracle':<20} {orc:>8.1f} {'—':>10} {'1.0x':>10}")

    summary = {"bs": BS, "n_samples": ns}
    for key in ["original", "freq", "all", "oracle"] + WARMUP_STEPS:
        v = np.mean(results[key])
        summary[str(key)] = {"mean": float(v), "scatter": float(v / orc),
                             "reduction_pct": float((1 - v / om) * 100) if key != "original" else 0}
    outpath = os.path.join(DIR, "reorder_warmup_steps.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {outpath}")


if __name__ == "__main__":
    main()
