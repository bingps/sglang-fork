#!/usr/bin/env python3
"""Full 100×21 reorder comparison with GPU acceleration.

6 methods: original, freq, kmeans_idx_k, kmeans_c4_kv, spectral_cooc, greedy_cooc
bs=16/32/64. All requests, all layers.

GPU acceleration:
- greedy_cooc: build sparse cooc on GPU (torch.sparse COO), greedy via torch.mv
- kmeans: torch GPU batched distance computation
- spectral: scipy eigsh (CPU, but sparse cooc built fast)
- block counting: torch GPU vectorized
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/cpfs02/user/lgd/sglang-fork")
from sim_hisparse_hitrate import COMPRESS_RATIO, DIR, load_all

BLOCK_SIZES = [16, 32, 64]
DEVICE = "cuda:0"


def blocks_touched_batch(sel_list, bs, perm=None):
    """Vectorized block counting for a list of selections."""
    results = []
    for sel in sel_list:
        if perm is not None:
            s = perm[sel]
        else:
            s = sel
        results.append(len(np.unique(s // bs)))
    return results


def oracle_blocks(n, bs):
    return int(np.ceil(n / bs))


def perm_from_order(order, n):
    perm = np.empty(n, dtype=np.int32)
    perm[order] = np.arange(n, dtype=np.int32)
    return perm


def mk_freq_perm(topk_L, n):
    freq = np.zeros(n, dtype=np.int32)
    for s in range(topk_L.shape[0]):
        sel = topk_L[s]; sel = sel[(sel >= 0) & (sel < n)]
        np.add.at(freq, sel, 1)
    order = np.argsort(-freq)
    return perm_from_order(order, n)


def mk_kmeans_perm_gpu(features_np, n, bs=64):
    """K-means on GPU using torch."""
    nc = max(n // bs, 2)
    feat = torch.from_numpy(features_np).float().to(DEVICE)
    norms = feat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    feat = feat / norms

    # MiniBatch K-means on GPU
    rng = torch.Generator(device=DEVICE).manual_seed(42)
    idx = torch.randperm(n, generator=rng, device=DEVICE)[:nc]
    centers = feat[idx].clone()

    for _ in range(30):
        # assign (batched to limit memory)
        labels = torch.empty(n, dtype=torch.int64, device=DEVICE)
        batch = 4096
        for start in range(0, n, batch):
            end = min(start + batch, n)
            dists = torch.cdist(feat[start:end], centers)
            labels[start:end] = dists.argmin(dim=1)
        # update
        for c in range(nc):
            mask = labels == c
            if mask.any():
                centers[c] = feat[mask].mean(dim=0)

    labels_cpu = labels.cpu().numpy()
    order = np.lexsort((np.arange(n), labels_cpu))
    return perm_from_order(order, n)


def _build_cooc_gpu(topk_L, n):
    """Build sparse co-occurrence on GPU as torch sparse COO tensor."""
    rows_list, cols_list = [], []
    for s in range(topk_L.shape[0]):
        sel = topk_L[s]; sel = np.unique(sel[(sel >= 0) & (sel < n)])
        if len(sel) > 512:
            sel = sel[:512]
        r = np.repeat(sel, len(sel))
        c = np.tile(sel, len(sel))
        rows_list.append(r)
        cols_list.append(c)
    if not rows_list:
        return torch.sparse_coo_tensor(
            torch.empty(2, 0, dtype=torch.long), torch.empty(0), (n, n),
            device=DEVICE)
    rows = np.concatenate(rows_list).astype(np.int64)
    cols = np.concatenate(cols_list).astype(np.int64)
    indices = torch.from_numpy(np.stack([rows, cols])).to(DEVICE)
    values = torch.ones(len(rows), dtype=torch.float32, device=DEVICE)
    cooc = torch.sparse_coo_tensor(indices, values, (n, n), device=DEVICE)
    cooc = cooc.coalesce()
    return cooc


def mk_greedy_cooc_perm_gpu(topk_L, n, bs=64):
    """Greedy block-filling: one sparse.mm per BLOCK (not per token)."""
    freq = np.zeros(n, dtype=np.int32)
    for s in range(topk_L.shape[0]):
        sel = topk_L[s]; sel = sel[(sel >= 0) & (sel < n)]
        np.add.at(freq, sel, 1)

    cooc = _build_cooc_gpu(topk_L, n)
    freq_gpu = torch.from_numpy(freq).float().to(DEVICE)

    placed = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    order = []

    while len(order) < n:
        # seed: unplaced with highest freq
        masked_freq = freq_gpu.clone()
        masked_freq[placed] = -1
        seed = int(masked_freq.argmax().item())
        if masked_freq[seed] < 0:
            break

        # build indicator from seed alone, do ONE sparse.mm to get affinity
        indicator = torch.zeros(n, dtype=torch.float32, device=DEVICE)
        indicator[seed] = 1.0
        scores = torch.sparse.mm(cooc, indicator.unsqueeze(1)).squeeze(1)
        # blend with freq for tie-breaking
        scores = scores + freq_gpu * 0.01
        scores[placed] = -1
        scores[seed] = -1  # already in block

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

    # remaining unplaced tokens
    unplaced = torch.where(~placed)[0].cpu().tolist()
    order.extend(unplaced)
    order.extend(unplaced)

    perm = np.empty(n, dtype=np.int32)
    for new_pos, orig in enumerate(order):
        perm[orig] = new_pos
    return perm


def mk_spectral_perm_gpu(topk_L, n, bs=64, n_ev=32):
    """Spectral clustering: sparse eigsh (CPU) + K-means in spectral space (GPU)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    # build sparse cooc on CPU for eigsh
    rows_list, cols_list = [], []
    for s in range(topk_L.shape[0]):
        sel = topk_L[s]; sel = np.unique(sel[(sel >= 0) & (sel < n)])
        if len(sel) > 512:
            sel = sel[:512]
        r = np.repeat(sel, len(sel))
        c = np.tile(sel, len(sel))
        rows_list.append(r)
        cols_list.append(c)
    if not rows_list:
        return mk_freq_perm(topk_L, n)
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.ones(len(rows), dtype=np.float32)
    cooc = csr_matrix((data, (rows, cols)), shape=(n, n))

    degrees = np.array(cooc.sum(axis=1)).flatten()
    active = np.where(degrees > 0)[0]
    n_active = len(active)
    if n_active < n_ev * 2:
        return mk_freq_perm(topk_L, n)

    sub = cooc[active][:, active]
    L = laplacian(sub, normed=True)
    n_ev_actual = min(n_ev, n_active - 2)
    try:
        _, eigvecs = eigsh(L, k=n_ev_actual, which='SM', maxiter=300)
    except Exception:
        return mk_freq_perm(topk_L, n)

    # K-means on spectral features (GPU)
    feat = torch.from_numpy(eigvecs).float().to(DEVICE)
    norms = feat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    feat = feat / norms

    nc = min(max(n // bs, 2), n_active)
    rng = torch.Generator(device=DEVICE).manual_seed(42)
    idx = torch.randperm(n_active, generator=rng, device=DEVICE)[:nc]
    centers = feat[idx].clone()
    for _ in range(30):
        labels = torch.cdist(feat, centers).argmin(dim=1)
        for c in range(nc):
            mask = labels == c
            if mask.any():
                centers[c] = feat[mask].mean(dim=0)
    labels_active = labels.cpu().numpy()

    labels_full = np.full(n, nc, dtype=np.int32)
    labels_full[active] = labels_active
    order = np.lexsort((np.arange(n), labels_full))
    return perm_from_order(order, n)


def main():
    data = load_all()
    N = len(data)
    NL = data[0]["topk"].shape[1]
    print(f"Loaded {N} requests, {NL} layers, bs={BLOCK_SIZES}, device={DEVICE}")

    # load .pt files for KV features
    pt_dir = os.path.join(DIR, "kv_dump")
    pt_files = glob.glob(os.path.join(pt_dir, "req_*.pt"))
    pt_by_ptoks = {}
    for p in pt_files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        pt_by_ptoks[d["prompt_tokens"]] = p

    methods = ["original", "freq", "kmeans_idx_k", "kmeans_c4_kv", "spectral_cooc", "greedy_cooc"]
    results = {m: {bs: [] for bs in BLOCK_SIZES} for m in methods}
    results["oracle"] = {bs: [] for bs in BLOCK_SIZES}

    t_global = time.time()
    for ri in range(N):
        ex = data[ri]
        topk = ex["topk"]
        ct = (ex["prompt_tokens"] + ex["completion_tokens"]) // COMPRESS_RATIO

        pt_path = pt_by_ptoks.get(ex["prompt_tokens"])
        has_kv = pt_path is not None
        if has_kv:
            pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
            idx_k_all = pt_data["indexer_k"].numpy()
            c4_kv_all = pt_data["c4_kv"].numpy()
        t0 = time.time()

        for L in range(NL):
            topk_L = topk[:, L, :]

            perm_freq = mk_freq_perm(topk_L, ct)

            if has_kv:
                perm_idx_k = mk_kmeans_perm_gpu(idx_k_all[L].astype(np.float32), ct)
                perm_c4_kv = mk_kmeans_perm_gpu(c4_kv_all[L].astype(np.float32), ct)
            else:
                perm_idx_k = perm_freq
                perm_c4_kv = perm_freq

            perm_spectral = mk_spectral_perm_gpu(topk_L, ct)
            perm_greedy = mk_greedy_cooc_perm_gpu(topk_L, ct, bs=64)

            perms = {
                "original": None,
                "freq": perm_freq,
                "kmeans_idx_k": perm_idx_k,
                "kmeans_c4_kv": perm_c4_kv,
                "spectral_cooc": perm_spectral,
                "greedy_cooc": perm_greedy,
            }

            # collect all selections for this layer
            sels = []
            for s in range(topk_L.shape[0]):
                sel = topk_L[s]; sel = sel[(sel >= 0) & (sel < ct)]
                if len(sel) > 0:
                    sels.append(sel)

            for sel in sels:
                for bs in BLOCK_SIZES:
                    for m, perm in perms.items():
                        results[m][bs].append(blocks_touched_batch([sel], bs, perm)[0])
                    results["oracle"][bs].append(oracle_blocks(len(sel), bs))

        elapsed = time.time() - t0
        if (ri + 1) % 5 == 0 or ri == 0:
            print(f"  [{ri+1:3d}/{N}] c4={ct:5d} steps={topk.shape[0]:3d} ({elapsed:.1f}s) "
                  f"samples={len(results['original'][64])}", flush=True)

    total_time = time.time() - t_global
    ns = len(results["original"][64])
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.0f}min). Samples: {ns}")

    print("\n" + "=" * 100)
    print(f"Results: mean blocks per topk selection ({N} req × {NL} layers, {ns} samples)")
    print("=" * 100)

    for bs in BLOCK_SIZES:
        orc = np.mean(results["oracle"][bs])
        print(f"\n  bs={bs}  (oracle={orc:.1f})")
        print(f"  {'method':<16} {'blocks':>8} {'reduction':>10} {'scatter':>10}")
        print(f"  {'-'*48}")
        om = np.mean(results["original"][bs])
        for m in methods:
            v = np.mean(results[m][bs])
            red = (1 - v / om) * 100
            scat = v / orc
            best_v = min(np.mean(results[mm][bs]) for mm in methods if mm != "original")
            flag = " <-- best" if m != "original" and abs(v - best_v) < 0.1 else ""
            print(f"  {m:<16} {v:>8.1f} {red:>+9.0f}% {scat:>9.1f}x{flag}")

    summary = {}
    for bs in BLOCK_SIZES:
        orc = np.mean(results["oracle"][bs])
        om = np.mean(results["original"][bs])
        summary[str(bs)] = {}
        for m in methods + ["oracle"]:
            v = np.mean(results[m][bs])
            entry = {"mean": float(v), "scatter": float(v / orc)}
            if m != "original" and m != "oracle":
                entry["reduction_pct"] = float((1 - v / om) * 100)
            summary[str(bs)][m] = entry

    outpath = os.path.join(DIR, "reorder_all_methods_full.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {outpath}")


if __name__ == "__main__":
    main()
