"""Build a long-context benchmark dataset from LongBench.

Samples prompts with context length in [LEN_MIN, LEN_MAX] tokens from a mix of
English QA / summarization tasks. Each prompt = context + task instruction, so
the input exceeds the 8192 device buffer (exercising host swap-in) while staying
within the default 16384 max_prefill_tokens (single-chunk prefill).
"""

import json
import os
import random

BASE = "/cpfs02/user/lgd/sglang-hisparse-mtp/.longbench/data"
OUT = "/cpfs02/user/lgd/sglang-hisparse-mtp/longbench_prompts.jsonl"

LEN_MIN = 9000
LEN_MAX = 16000
PER_TASK = 12
SEED = 42

# task -> instruction template appended after the context
TASKS = {
    "narrativeqa": "Answer the question based on the above story.\nQuestion: {q}\nAnswer:",
    "musique": "Answer the question based on the above passages.\nQuestion: {q}\nAnswer:",
    "hotpotqa": "Answer the question based on the above passages.\nQuestion: {q}\nAnswer:",
    "qmsum": "Summarize the above meeting transcript based on the query.\nQuery: {q}\nSummary:",
    "gov_report": "Summarize the above government report concisely.\nSummary:",
}

random.seed(SEED)
samples = []
for task, tmpl in TASKS.items():
    path = os.path.join(BASE, task + ".jsonl")
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            L = ex.get("length", 0)
            if LEN_MIN <= L <= LEN_MAX:
                rows.append(ex)
    random.shuffle(rows)
    for ex in rows[:PER_TASK]:
        ctx = ex["context"]
        q = ex.get("input", "")
        if "{q}" in tmpl:
            tail = tmpl.format(q=q)
        else:
            tail = tmpl
        prompt = ctx.strip() + "\n\n" + tail
        samples.append(
            {
                "task": task,
                "length": ex.get("length", 0),
                "prompt": prompt,
                "answers": ex.get("answers", []),
            }
        )

random.shuffle(samples)
with open(OUT, "w") as fh:
    for s in samples:
        fh.write(json.dumps(s, ensure_ascii=False) + "\n")

lens = [s["length"] for s in samples]
lens.sort()
print("total prompts:", len(samples))
print("length min/median/max:", lens[0], lens[len(lens) // 2], lens[-1])
from collections import Counter

print("per task:", dict(Counter(s["task"] for s in samples)))
