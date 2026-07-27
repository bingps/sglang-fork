"""8-way concurrent chat burst: quality + admission stress."""

import json
from concurrent.futures import ThreadPoolExecutor
import urllib.request

BASE = "http://127.0.0.1:30012"

PROMPTS = [
    ("What is the capital of Japan? One sentence.", 48),
    ("Explain photosynthesis in three sentences.", 160),
    ("Write a 300-word story about a robot learning to paint.", 500),
    ("Count from one to twenty in words, comma separated.", 160),
    ("Name three planets of the solar system and one fact about each.", 200),
    ("Translate 'good morning' into French, German, and Spanish.", 100),
    ("What is 17 * 23? Show the steps.", 250),
    ("List four primary colors of light.", 80),
]


def chat(prompt, max_tokens):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def gram_ratio(text):
    w = text.split()
    if len(w) < 12:
        return 1.0
    g = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return len(set(g)) / len(g)


fails = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(chat, p, m) for p, m in PROMPTS]
    for i, f in enumerate(futs):
        t = f.result()
        r = gram_ratio(t)
        ok = r > 0.5 and len(t.strip()) > 0
        fails += 0 if ok else 1
        print(f"[{i}] 4gram={r:.3f} {'OK ' if ok else 'BAD'} head={t[:90]!r}")

checks = [
    ("Tokyo", 0), ("chloro", 1), (None, 2), ("twenty", 3),
    (None, 4), ("Bonjour|bonjour|Guten", 5), ("391", 6), (None, 7),
]
import re
res = [f.result() for f in futs]
for pat, idx in checks:
    if pat and not re.search(pat, res[idx]):
        fails += 1
        print(f"[{idx}] MISSING expected pattern {pat!r}")

print("BURST", "PASS" if fails == 0 else f"FAIL ({fails})")
if fails != 0:
    raise SystemExit(1)
