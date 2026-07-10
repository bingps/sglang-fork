#!/usr/bin/env python3
"""Build ~1M-token requests by concatenating distinct InfiniteBench longbook_qa_eng docs.

Natural narrative text (not repetitive passkey filler) so indexer topk patterns are
realistic. Each request = several distinct book excerpts concatenated to ~1M tokens,
followed by a real question from one of the constituent books.
"""
import json
import os

from transformers import AutoTokenizer

SRC = "/cpfs02/user/lgd/data/infinitebench/longbook_qa_eng.jsonl"
OUT = "/cpfs02/user/lgd/data/infinitebench/synth_1m.jsonl"
MODEL = "/cpfs02/user/lgd/models/dsv4f"
TARGET_TOKENS = 1_000_000
# model max_position_embeddings = 1048576; leave room for decode + wrapper
HARD_CAP = 990_000
N_REQUESTS = 5
SEP = "\n\n=== NEW DOCUMENT ===\n\n"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# collect distinct (context, question, answer)
seen = set()
docs = []
with open(SRC) as f:
    for line in f:
        d = json.loads(line)
        c = d.get("context", "")
        key = c[:200]
        if key in seen:
            continue
        seen.add(key)
        docs.append({
            "context": c,
            "question": d.get("input", ""),
            "answer": d.get("answer", ""),
        })
print(f"distinct docs: {len(docs)}")

# measure token lengths (cache)
print("measuring token lengths...")
for i, d in enumerate(docs):
    d["ntok"] = len(tok.encode(d["context"]))
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(docs)}")
docs.sort(key=lambda d: -d["ntok"])
print(f"doc token range: {docs[-1]['ntok']}..{docs[0]['ntok']}")

# greedily build N requests, each from distinct docs, no doc reused across requests
requests = []
used = set()
di = 0
for r in range(N_REQUESTS):
    parts = []
    total = 0
    q = None
    ans = None
    while total < TARGET_TOKENS and di < len(docs):
        if di in used:
            di += 1
            continue
        d = docs[di]
        used.add(di)
        di += 1
        parts.append(d["context"])
        total += d["ntok"]
        # use the LAST doc's question (most recent, model attends to end)
        q = d["question"]
        ans = d["answer"]
    if not parts:
        break
    body = SEP.join(parts)
    prompt = (
        f"You are given several documents. Read them all carefully.\n\n"
        f"{body}\n\n"
        f"Based on the documents above, answer this question: {q}\n\nAnswer:"
    )
    # hard-cap to HARD_CAP tokens (model max ctx = 1048576) via token truncation
    ids = tok.encode(prompt)
    if len(ids) > HARD_CAP:
        # keep head (documents) + tail (question); truncate the middle of the body
        tail = tok.encode(f"\n\nBased on the documents above, answer this question: {q}\n\nAnswer:")
        head_budget = HARD_CAP - len(tail)
        ids = ids[:head_budget] + tail
        prompt = tok.decode(ids)
    ntok = len(tok.encode(prompt))
    requests.append({
        "idx": r,
        "prompt": prompt,
        "n_docs": len(parts),
        "prompt_tokens_est": ntok,
        "question": q,
        "answer": ans,
    })
    print(f"request {r}: {len(parts)} docs, {ntok} tokens ({ntok/1e6:.2f}M)")

with open(OUT, "w") as f:
    for r in requests:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nSaved {len(requests)} requests -> {OUT}")
print(f"docs used: {len(used)}/{len(docs)}")
