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
    args = parser.parse_args()

    paths = _iter_dump_paths(args.inputs)
    by_layer: dict[int, dict] = defaultdict(_empty_stats)
    overall = _empty_stats()

    for path in paths:
        payload = torch.load(path, map_location="cpu")
        layer_id = int(payload.get("layer_id", -1))
        stats = _analyze_payload(payload)
        stats["files"] = 1
        _merge_stats(by_layer[layer_id], stats)
        by_layer[layer_id]["files"] += 1
        _merge_stats(overall, stats)
        overall["files"] += 1

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
        return

    print(f"Loaded files: {result['num_files']}")
    print("Overall:")
    print(json.dumps(result["overall"], indent=2, sort_keys=True))
    print("By layer:")
    for layer_id, summary in result["by_layer"].items():
        print(f"Layer {layer_id}: {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
