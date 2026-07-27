"""Benchmark decode throughput: send requests, parse server log gen throughput."""

import json
import re
import sys
import time
import urllib.request

PORT = 30012
LOG = sys.argv[2] if len(sys.argv) > 2 else "e2e_server.log"
PROMPT = (
    "Explain the theory of general relativity in detail, covering spacetime "
    "curvature, the equivalence principle, gravitational time dilation, "
    "gravitational waves, and experimental evidence. Be thorough."
)
MAX_TOKENS = 512
NUM_RUNS = 3


def send_request():
    payload = {
        "text": PROMPT,
        "sampling_params": {
            "max_new_tokens": MAX_TOKENS,
            "temperature": 0.0,
        },
    }
    url = "http://127.0.0.1:%d/generate" % PORT
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    meta = out["meta_info"]
    return meta["completion_tokens"], wall


def parse_log_throughput(marker_line_count):
    with open(LOG, "r", errors="replace") as f:
        lines = f.readlines()
    new_lines = lines[marker_line_count:]
    tps = []
    for line in new_lines:
        m = re.search(r"gen throughput \(token/s\): ([\d.]+)", line)
        if m:
            tps.append(float(m.group(1)))
    return tps


def count_log_lines():
    with open(LOG, "r", errors="replace") as f:
        return sum(1 for _ in f)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    print("=== %s benchmark (%d runs, %d tokens each) ===" % (label, NUM_RUNS, MAX_TOKENS))

    wall_results = []
    log_results = []
    for i in range(NUM_RUNS):
        marker = count_log_lines()
        tokens, wall = send_request()
        tps_list = parse_log_throughput(marker)
        wall_tps = tokens / wall if wall > 0 else 0
        avg_log_tps = sum(tps_list) / len(tps_list) if tps_list else 0
        wall_results.append(wall_tps)
        log_results.append(avg_log_tps)
        print(
            "  run %d: %d tokens in %.2fs | wall=%.1f tok/s | log_avg=%.1f tok/s (%d samples)"
            % (i, tokens, wall, wall_tps, avg_log_tps, len(tps_list))
        )

    avg_wall = sum(wall_results) / len(wall_results)
    avg_log = sum(log_results) / len(log_results)
    print("\n  %s average: wall=%.1f tok/s | log=%.1f tok/s" % (label, avg_wall, avg_log))


if __name__ == "__main__":
    main()
