#!/usr/bin/env python3
"""Build typing-simulation examples from held-out text.

Each example: {"src", "context", "target", "keys", "clean_keys", "typo": null|{"kind","pos"}, "syllables"}
The user is simulated as typing a chunk of 1-4 words (2-9 characters) after the real preceding text,
with the pinyin taken from the IME lexicon (disambiguated with pypinyin for polyphones).
Typos are injected with the kinds the engine models, at a configurable rate.

usage: make_evalset.py <out.jsonl> <n> [--typo-rate 0.3] [--seed 1] [--sources zhihu,wiki,...] [--split dev|test|train]
"""
import os, sys, re, json, random, argparse, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "data/build")

import jieba
from pypinyin import lazy_pinyin, Style

jieba.setLogLevel(60)
jieba.set_dictionary(os.path.join(B, "jieba_dict.txt"))
jieba.initialize()

readings = collections.defaultdict(list)
for line in open(os.path.join(B, "lexicon.tsv"), encoding="utf-8"):
    w, py, wt = line.rstrip("\n").split("\t")
    readings[w].append((py.replace(" ", "'"), int(wt)))
for w in readings:
    readings[w].sort(key=lambda x: -x[1])

ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
OFF = [0, 0.25, 0.75]
POS = {}
for r, row in enumerate(ROWS):
    for i, ch in enumerate(row):
        POS[ch] = (i + OFF[r], r)


def neighbours(c):
    x, y = POS[c]
    return [d for d, (a, b) in POS.items() if d != c and ((a - x) ** 2 + (b - y) ** 2) ** 0.5 < 1.3]


HAN = re.compile(r"^[一-鿿]+$")


def word_pinyin(w, ctx_py):
    rs = readings.get(w)
    if not rs:
        return None
    if len(rs) == 1:
        return rs[0][0]
    cand = {r for r, _ in rs}
    if ctx_py in cand:
        return ctx_py
    return rs[0][0]


def typo(keys, rng):
    """Apply one realistic slip; returns (new_keys, kind, pos)."""
    kinds = ["insertRepeat", "insertAdjacent", "delete", "substituteAdjacent", "transpose"]
    weights = [0.28, 0.14, 0.22, 0.22, 0.14]
    for _ in range(20):
        k = rng.choices(kinds, weights)[0]
        n = len(keys)
        if n < 3:
            return None
        i = rng.randrange(1, n)  # never touch the very first key
        if k == "insertRepeat":
            new = keys[:i] + keys[i - 1] + keys[i:]
        elif k == "insertAdjacent":
            new = keys[:i] + rng.choice(neighbours(keys[i - 1])) + keys[i:]
        elif k == "delete":
            if i >= n - 1:  # dropping the final key is indistinguishable from unfinished input; keep it interior
                continue
            new = keys[:i] + keys[i + 1:]
        elif k == "substituteAdjacent":
            new = keys[:i] + rng.choice(neighbours(keys[i])) + keys[i + 1:]
        else:
            if i >= n - 1 or keys[i] == keys[i + 1]:
                continue
            new = keys[:i] + keys[i + 1] + keys[i] + keys[i + 2:]
        if new != keys:
            return new, k, i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("n", type=int)
    ap.add_argument("--typo-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sources", default="zhihu,wiki,v2ex_topic,v2ex_replies,cci3")
    ap.add_argument("--split", default="dev", help="dev = first half of held-out docs, test = second half")
    ap.add_argument("--heldout-dir", default=os.path.join(B, "heldout"))
    ap.add_argument("--input", help="use this text file (docs separated by blank lines) instead of held-out corpora")
    ap.add_argument("--range", default="0:1", help="fraction of documents of --input to use, e.g. 0:0.7")
    ap.add_argument("--sequential", action="store_true", help="chunk the text in order (a usage stream) instead of sampling")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.input:
        docs = [d for d in open(a.input, encoding="utf-8").read().split("\n\n") if d.strip()]
        lo, hi = (float(x) for x in a.range.split(":"))
        docs = docs[int(len(docs) * lo):int(len(docs) * hi)]
        src = os.path.basename(a.input).rsplit(".", 1)[0]
        if a.sequential:
            sequential(docs, src, a, rng)
            return
        paras = [(src, p) for d in docs for p in d.split("\n") if len(p) >= 12]
        sample(paras, a, rng)
        return
    paras = []
    for src in a.sources.split(","):
        path = os.path.join(a.heldout_dir, src + ".txt")
        if not os.path.exists(path):
            continue
        docs = open(path, encoding="utf-8").read().split("\n\n")
        half = len(docs) // 2
        docs = docs[:half] if a.split == "dev" else docs[half:]
        for d in docs:
            for p in d.split("\n"):
                if len(p) >= 12:
                    paras.append((src, p))
    sample(paras, a, rng)


def annotate(p, words, i, chunk):
    """pinyin syllables of `chunk` (list of words) starting at word index i of paragraph p"""
    start_char = len("".join(words[:i]))
    full_py = lazy_pinyin(p[:start_char + len("".join(chunk))], style=Style.NORMAL)
    sylls = []
    off = start_char
    for w in chunk:
        ctx_py = "'".join(x.replace("ü", "v") for x in full_py[off:off + len(w)])
        py = word_pinyin(w, ctx_py)
        if py is None:
            return None
        sylls.extend(py.split("'"))
        off += len(w)
    return sylls


def sequential(docs, src, a, rng):
    """Every Han run of the text, in order, cut into chunks of 1-4 words: a stream of what the user typed."""
    out = open(a.out, "w", encoding="utf-8")
    made = 0
    for d in docs:
        for p in d.split("\n"):
            words = list(jieba.cut(p, HMM=False))
            i = 0
            while i < len(words):
                if not HAN.match(words[i]):
                    i += 1
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
                    ctx = "".join(words[:i])[-120:]
                    clean = "".join(sylls)
                    ex = {"src": src, "context": ctx, "target": target, "clean_keys": clean, "keys": clean, "syllables": sylls, "typo": None}
                    if rng.random() < a.typo_rate:
                        t = typo(clean, rng)
                        if t:
                            ex["keys"], kind, pos = t
                            ex["typo"] = {"kind": kind, "pos": pos}
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    made += 1
                    if made >= a.n:
                        print("examples", made, file=sys.stderr)
                        return
                i += len(chunk)
    print("examples", made, file=sys.stderr)


def sample(paras, a, rng):
    rng.shuffle(paras)
    out = open(a.out, "w", encoding="utf-8")
    made = 0
    for src, p in paras:
        if made >= a.n:
            break
        words = list(jieba.cut(p, HMM=False))
        # candidate chunk starts: a Han word preceded by anything
        starts = [i for i, w in enumerate(words) if HAN.match(w)]
        if not starts:
            continue
        i = rng.choice(starts)
        nw = rng.choices([1, 2, 3, 4], [0.30, 0.35, 0.22, 0.13])[0]
        chunk = []
        for w in words[i:i + nw]:
            if not HAN.match(w):
                break
            chunk.append(w)
        target = "".join(chunk)
        if not (2 <= len(target) <= 9):
            continue
        ctx = "".join(words[:i])[-120:]
        full_py = lazy_pinyin(p[:len("".join(words[:i])) + len(target)], style=Style.NORMAL, v_to_u=False, neutral_tone_with_five=False)
        # pypinyin gives per-character; align to the chunk
        start_char = len("".join(words[:i]))
        sylls = []
        ok = True
        off = start_char
        for w in chunk:
            ctx_py = "'".join(x.replace("ü", "v") for x in full_py[off:off + len(w)])
            py = word_pinyin(w, ctx_py)
            if py is None:
                ok = False
                break
            sylls.extend(py.split("'"))
            off += len(w)
        if not ok:
            continue
        clean = "".join(sylls)
        ex = {"src": src, "context": ctx, "target": target, "clean_keys": clean, "keys": clean,
              "syllables": sylls, "typo": None}
        if rng.random() < a.typo_rate:
            t = typo(clean, rng)
            if t is None:
                continue
            ex["keys"], kind, pos = t
            ex["typo"] = {"kind": kind, "pos": pos}
        out.write(json.dumps(ex, ensure_ascii=False) + "\n")
        made += 1
    print("examples", made, file=sys.stderr)


if __name__ == "__main__":
    main()
