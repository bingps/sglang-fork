#!/usr/bin/env python3
"""Analyze overlap between adjacent NSA draft-token top-k dumps."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch


def _iter_dump_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.pt")))
        else:
            matches = [Path(p) for p in glob.glob(item)]
            paths.extend(matches or [path])
    return sorted({p for p in paths if p.exists()})


def _valid_set(row: Iterable[int]) -> set[int]:
    return {int(x) for x in row if int(x) >= 0}


def _append_pair(stats: dict, left: set[int], right: set[int]) -> None:
    if not left and not right:
        return
    inter = len(left & right)
    union = len(left | right)
    denom = min(len(left), len(right))
    stats["pairs"] += 1
    stats["overlap_counts"].append(inter)
    stats["overlap_ratios"].append(inter / denom if denom else 0.0)
    stats["jaccards"].append(inter / union if union else 0.0)
    stats["exact_matches"] += int(left == right)


def _analyze_payload(payload: dict) -> dict:
    topk = payload["topk"].to(torch.int64).tolist()
    extend_lens = payload.get("extend_seq_lens") or [len(topk)]
    stats = {
        "pairs": 0,
        "exact_matches": 0,
        "overlap_counts": [],
        "overlap_ratios": [],
        "jaccards": [],
    }

    offset = 0
    for req_len in extend_lens:
        req_len = int(req_len)
        rows = topk[offset : offset + req_len]
        offset += req_len
        for i in range(max(0, len(rows) - 1)):
            _append_pair(stats, _valid_set(rows[i]), _valid_set(rows[i + 1]))

    return stats


def _collect_vs_draft0(payload: dict) -> list[tuple[int, float, float]]:
    """For each (draft_idx, overlap_ratio_vs_draft0, jaccard_vs_draft0) within
    each request in the payload, return a flat list."""
    topk = payload["topk"].to(torch.int64).tolist()
    extend_lens = payload.get("extend_seq_lens") or [len(topk)]
    out: list[tuple[int, float, float]] = []

    offset = 0
    for req_len in extend_lens:
        req_len = int(req_len)
        rows = topk[offset : offset + req_len]
        offset += req_len
        if not rows:
            continue
        base = _valid_set(rows[0])
        for i, row in enumerate(rows):
            cur = _valid_set(row)
            inter = len(base & cur)
            union = len(base | cur)
            denom = min(len(base), len(cur))
            ratio = inter / denom if denom else 0.0
            jacc = inter / union if union else 0.0
            out.append((i, ratio, jacc))
    return out


def _merge_stats(dst: dict, src: dict) -> None:
    dst["pairs"] += src["pairs"]
    dst["exact_matches"] += src["exact_matches"]
    dst["overlap_counts"].extend(src["overlap_counts"])
    dst["overlap_ratios"].extend(src["overlap_ratios"])
    dst["jaccards"].extend(src["jaccards"])


def _empty_stats() -> dict:
    return {
        "files": 0,
        "pairs": 0,
        "exact_matches": 0,
        "overlap_counts": [],
        "overlap_ratios": [],
        "jaccards": [],
    }


def _summarize(stats: dict) -> dict:
    pairs = stats["pairs"]
    return {
        "files": stats.get("files", 0),
        "pairs": pairs,
        "avg_overlap_count": mean(stats["overlap_counts"]) if pairs else 0.0,
        "avg_overlap_ratio": mean(stats["overlap_ratios"]) if pairs else 0.0,
        "avg_jaccard": mean(stats["jaccards"]) if pairs else 0.0,
        "exact_match_ratio": stats["exact_matches"] / pairs if pairs else 0.0,
    }


def _plot_heatmap(
    matrix,
    layers: list[int],
    num_draft: int,
    output_path: Path,
    title: str,
    value_fmt: str = "{:.2f}",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(matrix, dtype=float)  # shape (num_draft, num_layers)
    # drop the meaningless draft0 vs draft0 row (always 1.0)
    if num_draft > 1:
        arr = arr[1:, :]
        row_labels = [f"draft{i} vs draft0" for i in range(1, num_draft)]
    else:
        row_labels = [f"draft{i} vs draft0" for i in range(num_draft)]
    rows = arr.shape[0]

    # split layers into two halves for an upper/lower subplot layout
    mid = (len(layers) + 1) // 2
    splits = [(0, mid), (mid, len(layers))]

    max_cols = max(end - start for start, end in splits) or 1
    fig_w = max(8.0, 0.32 * max_cols + 2)
    fig_h = max(3.5, 0.55 * rows * len(splits) + 2.0)
    fig, axes = plt.subplots(len(splits), 1, figsize=(fig_w, fig_h))
    if len(splits) == 1:
        axes = [axes]

    im = None
    for ax, (start, end) in zip(axes, splits):
        sub = arr[:, start:end]
        sub_layers = layers[start:end]
        im = ax.imshow(sub, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(sub_layers)))
        ax.set_xticklabels([str(l) for l in sub_layers], rotation=90, fontsize=8)
        ax.set_yticks(range(rows))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("layer_id")
        ax.set_ylabel("draft index")
        ax.set_title(f"layers {sub_layers[0]}..{sub_layers[-1]}")

        if sub.shape[1] <= 80:
            for i in range(sub.shape[0]):
                for j in range(sub.shape[1]):
                    v = sub[i, j]
                    if not np.isnan(v):
                        ax.text(
                            j,
                            i,
                            value_fmt.format(v),
                            ha="center",
                            va="center",
                            color="white" if v < 0.5 else "black",
                            fontsize=7,
                        )

    fig.subplots_adjust(hspace=0.6, top=0.92)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze adjacent draft-token top-k overlap from NSA dump files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Dump directory, .pt file, or glob pattern, e.g. /tmp/sglang_nsa_draft_topk",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text summary.",
    )
    parser.add_argument(
        "--heatmap",
        type=str,
        default=None,
        help="Output path prefix for the (num_draft x num_layer) heatmap PNG. "
        "If a directory is provided, files 'overlap_ratio.png' and 'jaccard.png' "
        "are written inside it. Defaults to '<inputs[0]>/heatmap_overlap_ratio.png' "
        "and '<inputs[0]>/heatmap_jaccard.png' when omitted.",
    )
    args = parser.parse_args()

    paths = _iter_dump_paths(args.inputs)
    by_layer: dict[int, dict] = defaultdict(_empty_stats)
    overall = _empty_stats()

    # heatmap accumulators: (layer_id, draft_idx) -> list[ratio], list[jaccard]
    hm_ratio: dict[tuple[int, int], list[float]] = defaultdict(list)
    hm_jacc: dict[tuple[int, int], list[float]] = defaultdict(list)

    for path in paths:
        payload = torch.load(path, map_location="cpu")
        layer_id = int(payload.get("layer_id", -1))
        stats = _analyze_payload(payload)
        stats["files"] = 1
        _merge_stats(by_layer[layer_id], stats)
        by_layer[layer_id]["files"] += 1
        _merge_stats(overall, stats)
        overall["files"] += 1

        for draft_idx, ratio, jacc in _collect_vs_draft0(payload):
            hm_ratio[(layer_id, draft_idx)].append(ratio)
            hm_jacc[(layer_id, draft_idx)].append(jacc)

    result = {
        "num_files": len(paths),
        "overall": _summarize(overall),
        "by_layer": {
            str(layer_id): _summarize(stats)
            for layer_id, stats in sorted(by_layer.items())
        },
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Loaded files: {result['num_files']}")
        print("Overall:")
        print(json.dumps(result["overall"], indent=2, sort_keys=True))
        print("By layer:")
        for layer_id, summary in result["by_layer"].items():
            print(f"Layer {layer_id}: {json.dumps(summary, sort_keys=True)}")

    # build heatmap matrix
    if hm_ratio:
        layers = sorted({lid for lid, _ in hm_ratio.keys()})
        num_draft = max(d for _, d in hm_ratio.keys()) + 1
        ratio_mat = [[float("nan")] * len(layers) for _ in range(num_draft)]
        jacc_mat = [[float("nan")] * len(layers) for _ in range(num_draft)]
        for (lid, didx), vs in hm_ratio.items():
            ratio_mat[didx][layers.index(lid)] = mean(vs)
        for (lid, didx), vs in hm_jacc.items():
            jacc_mat[didx][layers.index(lid)] = mean(vs)

        out_arg = args.heatmap
        if out_arg is None:
            base_dir = Path(args.inputs[0])
            base_dir = base_dir if base_dir.is_dir() else base_dir.parent
            ratio_path = base_dir / "heatmap_overlap_ratio.png"
            jacc_path = base_dir / "heatmap_jaccard.png"
        else:
            out_path = Path(out_arg)
            if out_path.is_dir() or out_arg.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
                ratio_path = out_path / "overlap_ratio.png"
                jacc_path = out_path / "jaccard.png"
            else:
                ratio_path = out_path.with_name(out_path.stem + "_overlap_ratio.png")
                jacc_path = out_path.with_name(out_path.stem + "_jaccard.png")

        _plot_heatmap(
            ratio_mat,
            layers,
            num_draft,
            ratio_path,
            title="Top-k overlap ratio (draft_i vs draft_0)",
        )
        _plot_heatmap(
            jacc_mat,
            layers,
            num_draft,
            jacc_path,
            title="Top-k Jaccard (draft_i vs draft_0)",
        )
        print(f"Saved heatmaps:\n  {ratio_path}\n  {jacc_path}")


if __name__ == "__main__":
    main()
