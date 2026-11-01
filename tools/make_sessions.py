#!/usr/bin/env python3
"""Build continuous typing sessions from held-out text, for `stbench session`.

Each line is one paragraph typed from start to end:
  {"src", "prefix": text already in the document (the paragraph before), "events": [...]}
An event is either {"keys", "clean_keys", "target"} (a chunk of 1-4 words typed as pinyin, with a slip at
--typo-rate) or {"text"} (punctuation, Latin letters, digits: inserted as they are).

usage: make_sessions.py <out.jsonl> <paragraphs> [--typo-rate 0.3] [--seed 1] [--sources zhihu,wiki,...]
"""
import os, sys, json, random, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_evalset import B, HAN, annotate, jieba, typo  # noqa: E402


def session(src, prefix, p, a, rng):
    words = list(jieba.cut(p, HMM=False))
    events = []

    def text(s):
        if events and "text" in events[-1]:
            events[-1]["text"] += s
        else:
            events.append({"text": s})

    i = 0
    while i < len(words):
        if not HAN.match(words[i]):
            text(words[i]); i += 1
            continue
        nw = rng.choices([1, 2, 3, 4], [0.30, 0.35, 0.22, 0.13])[0]
        chunk = []
        for w in words[i:i + nw]:
            if not HAN.match(w):
                break
            chunk.append(w)
        target = "".join(chunk)
        sylls = annotate(p, words, i, chunk) if len(target) <= 12 else None
        if sylls:
            clean = "".join(sylls)
            ev = {"keys": clean, "clean_keys": clean, "target": target}
            if rng.random() < a.typo_rate:
                t = typo(clean, rng)
                if t:
                    ev["keys"] = t[0]
            events.append(ev)
        else:
            text(target)    # no reading in the lexicon: as if pasted
        i += len(chunk)
    return {"src": src, "prefix": prefix, "events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("n", type=int)
    ap.add_argument("--typo-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sources", default="zhihu,wiki,v2ex_topic,v2ex_replies,fineweb")
    ap.add_argument("--min-chars", type=int, default=80)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    paras = []
    for src in a.sources.split(","):
        path = os.path.join(B, "heldout", src + ".txt")
        if not os.path.exists(path):
            continue
        docs = open(path, encoding="utf-8").read().split("\n\n")
        for d in docs[len(docs) // 2:]:          # the test half, like make_evalset.py --split test
            ps = [p for p in d.split("\n") if p.strip()]
            for k, p in enumerate(ps):
                if len(p) >= a.min_chars:
                    paras.append((src, ps[k - 1] if k > 0 else "", p))
    rng.shuffle(paras)
    with open(a.out, "w", encoding="utf-8") as out:
        for src, prefix, p in paras[:a.n]:
            out.write(json.dumps(session(src, prefix, p, a, rng), ensure_ascii=False) + "\n")
    print("sessions", min(a.n, len(paras)), file=sys.stderr)


if __name__ == "__main__":
    main()
