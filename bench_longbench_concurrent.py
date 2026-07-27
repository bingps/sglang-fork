"""High-concurrency long-context throughput benchmark.

Sends long LongBench prompts at a target concurrency level and measures
aggregate (system) decode throughput, per-request latency, and success rate.
This stresses HiSparse's capacity (contexts exceed the 8192 device buffer, so
host swap-in is exercised) and concurrency scaling.

Usage:
  python bench_longbench_concurrent.py <label> <concurrency> [max_new_tokens] [num_requests] [log_path]
"""

import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = 30012
PROMPTS_FILE = os.environ.get(
    "BENCH_PROMPTS_FILE",
    "/cpfs02/user/lgd/sglang-hisparse-mtp/longbench_prompts.jsonl",
)


def load_prompts():
    out = []
    with open(PROMPTS_FILE) as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def send_one(prompt, max_tokens):
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    url = "http://127.0.0.1:%d/generate" % PORT
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read())
        dt = time.time() - t0
        meta = out["meta_info"]
        return {
            "ok": True,
            "completion_tokens": meta.get("completion_tokens", 0),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "latency": dt,
            "text": out.get("text", ""),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "latency": time.time() - t0,
            "error": str(e)[:200],
        }


def main():
    label = sys.argv[1]
    concurrency = int(sys.argv[2])
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    num_requests = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    prompts = load_prompts()
    work = [prompts[i % len(prompts)]["prompt"] for i in range(num_requests)]

    print(
        "=== %s | concurrency=%d | requests=%d | max_new_tokens=%d ==="
        % (label, concurrency, num_requests, max_tokens)
    )

    results = [None] * num_requests
    lock = threading.Lock()
    done = [0]

    def task(idx, prompt):
        res = send_one(prompt, max_tokens)
        results[idx] = res
        with lock:
            done[0] += 1
            if done[0] % 8 == 0 or done[0] == num_requests:
                print("  progress: %d/%d done" % (done[0], num_requests), flush=True)
        return res

    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(task, i, p) for i, p in enumerate(work)]
        for f in futs:
            f.result()
    wall_total = time.time() - wall_start

    ok = [r for r in results if r and r["ok"]]
    fail = [r for r in results if not r or not r["ok"]]
    total_completion = sum(r["completion_tokens"] for r in ok)
    total_prompt = sum(r["prompt_tokens"] for r in ok)
    latencies = sorted(r["latency"] for r in ok)

    def pct(p):
        if not latencies:
            return 0.0
        k = min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1))))
        return latencies[k]

    sys_tps = total_completion / wall_total if wall_total > 0 else 0
    total_tps = (total_prompt + total_completion) / wall_total if wall_total > 0 else 0

    print("\n--- %s results ---" % label)
    print("  success: %d/%d (%.1f%%)" % (len(ok), num_requests, 100 * len(ok) / num_requests))
    if fail:
        print("  failures: %d, e.g. %s" % (len(fail), fail[0].get("error", "?")))
    print("  wall time: %.1fs" % wall_total)
    print("  total prompt tokens: %d" % total_prompt)
    print("  total completion tokens: %d" % total_completion)
    print("  SYSTEM decode throughput: %.1f tok/s" % sys_tps)
    print("  SYSTEM total throughput: %.1f tok/s (incl prefill)" % total_tps)
    if ok:
        per_req_tps = [r["completion_tokens"] / r["latency"] for r in ok if r["latency"] > 0]
        print("  per-request decode tok/s: avg=%.1f" % (sum(per_req_tps) / len(per_req_tps)))
        print(
            "  latency p50/p90/p99: %.1f / %.1f / %.1f s"
            % (pct(50), pct(90), pct(99))
        )


if __name__ == "__main__":
    main()
