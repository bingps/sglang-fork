"""E2E validation for HiSparse+MTP speculative staging ring.

Cases:
1. short prompt + long generation (ring wraps many times)
2. long prompt (repeated content) + summary
3. sequential long generations (footprint stability; failures would surface
   as hisparse alloc RuntimeError / garbage output)
4. concurrent mixed-length requests
"""

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:30012"


def gen(prompt, max_tokens, temperature=0.0):
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    }
    req = urllib.request.Request(
        BASE + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    out["_elapsed"] = time.time() - t0
    return out


def brief(res, name):
    meta = res["meta_info"]
    text = res["text"]
    spec_accept = meta.get("spec_accept_rate", meta.get("spec_verify_ct", "n/a"))
    print(f"--- {name} ---")
    print(
        f"completion_tokens={meta['completion_tokens']} "
        f"elapsed={res['_elapsed']:.1f}s spec_accept={spec_accept}"
    )
    print("head:", repr(text[:200]))
    print("tail:", repr(text[-200:]))
    # crude degeneration check: a healthy completion should not be one token
    # repeated; measure unique 4-grams ratio on the tail half.
    words = text.split()
    if len(words) > 100:
        tail = words[len(words) // 2 :]
        grams = [tuple(tail[i : i + 4]) for i in range(len(tail) - 3)]
        ratio = len(set(grams)) / max(len(grams), 1)
        print(f"unique-4gram-ratio(tail half)={ratio:.3f}")
        assert ratio > 0.3, f"{name}: degenerate repetition detected ({ratio:.3f})"
    print()
    return res


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "short"):
        r = gen("The capital of France is", 32)
        brief(r, "short-sanity")
        assert "Paris" in r["text"], "short-sanity: expected 'Paris'"

    if which in ("all", "longgen"):
        for i in range(3):
            r = gen(
                "Write a detailed, structured essay about the history of "
                "computing, covering mechanical calculators, the invention of "
                "the transistor, mainframes, personal computers, the internet, "
                "and modern AI. Use clear section headings.",
                1200,
            )
            brief(r, f"long-generation-{i}")

    if which in ("all", "longprompt"):
        base_text = (
            "The quick brown fox jumps over the lazy dog near the river bank. "
            "Scientists discovered that honeybees communicate through dances. "
            "The ancient library of Alexandria contained countless scrolls. "
        )
        long_prompt = base_text * 600  # ~ 12k+ tokens
        long_prompt += (
            "\n\nQuestion: List the three distinct topics that were repeated "
            "in the text above. Answer briefly:"
        )
        r = gen(long_prompt, 200)
        brief(r, "long-prompt-summary")
        hits = sum(
            kw in r["text"].lower()
            for kw in ("fox", "bee", "alexandria", "library", "dog", "dance")
        )
        assert hits >= 2, f"long-prompt-summary: content not grounded (hits={hits})"

    if which in ("all", "concurrent"):
        prompts = [
            ("The capital of Japan is", 32),
            ("Explain photosynthesis in three sentences.", 128),
            ("Write a 500-word story about a robot learning to paint.", 700),
            # NOTE: raw-completion imperative prompts like "Count from one to
            # twenty." make the base model echo exercise instructions
            # repeatedly even at bs=1 (verified identical sequentially), so
            # use a natural continuation instead.
            ("The numbers from one to twenty written out in words are: one,", 128),
        ]
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(gen, p, m) for p, m in prompts]
            for i, f in enumerate(futs):
                brief(f.result(), f"concurrent-{i}")

    print("E2E PASS:", which)


if __name__ == "__main__":
    main()
