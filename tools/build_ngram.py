#!/usr/bin/env python3
"""Segment the corpus with the IME lexicon and estimate a pruned, interpolated Kneser-Ney word bigram.

Documents are separated by blank lines in data/build/corpus/*.txt. The last HOLDOUT fraction of the
documents of every file is reserved for evaluation and never counted.

Outputs data/build/lm.bin (format in Sources/PinyinEngine/NGram.swift) and data/build/heldout/<src>.txt
"""
import os, re, sys, math, struct, time
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "data/build")
CORPUS = os.path.join(B, "corpus")
HOLDOUT = 0.02
MIN_BIGRAM = int(os.environ.get("MIN_BIGRAM", "2"))

PUNCT_MAP = {",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；", "(": "（", ")": "）",
             "「": "“", "」": "”", "『": "“", "』": "”", "\"": "“", "【": "（", "】": "）", "~": "～", "-": "—", "－": "—"}
FULL_PUNCT = set("，。？！、：；“”‘’（）《》…—·～")

surf_id = None
V = 0
SID = {}


def init_worker():
    global surf_id, V, SID
    import jieba
    jieba.setLogLevel(60)
    jieba.set_dictionary(os.path.join(B, "jieba_dict.txt"))
    jieba.initialize()
    surfaces = [l.rstrip("\n") for l in open(os.path.join(B, "surfaces.txt"), encoding="utf-8")]
    surf_id = {w: i for i, w in enumerate(surfaces)}
    V = len(surfaces)
    for s in ["<s>", "</s>", "<unk>", "<eng>", "<num>"]:
        SID[s] = surf_id[s]


TOKEN_RE = re.compile(r"([一-鿿㐀-䶿]+)|([A-Za-z]+)|([0-9]+(?:\.[0-9]+)?)|(\s+)|(.)")


def tokenize_para(p, jieba):
    out = [SID["<s>"]]
    for m in TOKEN_RE.finditer(p):
        han, eng, num, ws, other = m.groups()
        if han:
            for w in jieba.cut(han, HMM=False):
                i = surf_id.get(w)
                if i is None:
                    for ch in w:
                        out.append(surf_id.get(ch, SID["<unk>"]))
                else:
                    out.append(i)
        elif eng:
            out.append(SID["<eng>"])
        elif num:
            out.append(SID["<num>"])
        elif ws:
            continue
        else:
            c = PUNCT_MAP.get(other, other)
            if c in FULL_PUNCT:
                out.append(surf_id[c])
            else:
                out.append(SID["<unk>"])
    return out


def count_chunk(lines):
    import jieba
    toks = []
    for p in lines:
        toks.extend(tokenize_para(p, jieba))
    a = np.array(toks, dtype=np.int64)
    uni = np.bincount(a, minlength=V)
    # bigrams, not crossing paragraph starts (<s> is only a history)
    prev, cur = a[:-1], a[1:]
    mask = cur != SID["<s>"]
    keys = prev[mask] * V + cur[mask]
    k, c = np.unique(keys, return_counts=True)
    return uni, k, c.astype(np.int64), len(a)


def read_docs(path):
    docs, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    docs.append(cur); cur = []
            else:
                cur.append(line)
    if cur:
        docs.append(cur)
    return docs


def merge(k1, c1, k2, c2):
    k = np.concatenate([k1, k2]); c = np.concatenate([c1, c2])
    order = np.argsort(k, kind="stable")
    k = k[order]; c = c[order]
    uniq, idx = np.unique(k, return_index=True)
    sums = np.add.reduceat(c, idx)
    return uniq, sums


def main():
    init_worker()
    os.makedirs(os.path.join(B, "heldout"), exist_ok=True)
    files = sorted(f for f in os.listdir(CORPUS) if f.endswith(".txt"))
    chunks = []
    for fn in files:
        docs = read_docs(os.path.join(CORPUS, fn))
        n_hold = max(50, int(len(docs) * HOLDOUT))
        train, held = docs[:-n_hold], docs[-n_hold:]
        with open(os.path.join(B, "heldout", fn), "w", encoding="utf-8") as f:
            for d in held:
                f.write("\n".join(d) + "\n\n")
        lines = [p for d in train for p in d]
        print(fn, "docs", len(docs), "train paras", len(lines), file=sys.stderr)
        step = 20000
        chunks += [lines[i:i + step] for i in range(0, len(lines), step)]
    print("chunks", len(chunks), file=sys.stderr)
    uni = np.zeros(V, dtype=np.int64)
    K = np.zeros(0, dtype=np.int64); C = np.zeros(0, dtype=np.int64)
    pendK, pendC = [], []
    total = 0
    t0 = time.time()
    with Pool(max(1, os.cpu_count() - 2), initializer=init_worker) as pool:
        for i, (u, k, c, n) in enumerate(pool.imap_unordered(count_chunk, chunks)):
            uni += u; total += n
            pendK.append(k); pendC.append(c)
            if len(pendK) >= 16:
                K, C = merge(np.concatenate(pendK), np.concatenate(pendC), K, C)
                pendK, pendC = [], []
                print(f"  {i+1}/{len(chunks)} tokens={total/1e6:.1f}M bigrams={len(K)/1e6:.1f}M {time.time()-t0:.0f}s", file=sys.stderr)
    if pendK:
        K, C = merge(np.concatenate(pendK), np.concatenate(pendC), K, C)
    print("tokens", total, "distinct bigrams", len(K), file=sys.stderr)
    np.savez(os.path.join(B, "counts.npz"), uni=uni, K=K, C=C)
    estimate(uni, K, C)


def estimate(uni, K, C):
    prev = (K // V).astype(np.int64)
    cur = (K % V).astype(np.int64)
    # modified KN discounts from count-of-counts
    n1 = np.sum(C == 1); n2 = np.sum(C == 2); n3 = np.sum(C == 3); n4 = np.sum(C == 4)
    Y = n1 / (n1 + 2 * n2)
    D1 = 1 - 2 * Y * n2 / n1; D2 = 2 - 3 * Y * n3 / n2; D3 = 3 - 4 * Y * n4 / n3
    print("discounts", D1, D2, D3, file=sys.stderr)
    # continuation unigram
    cont = np.bincount(cur, minlength=V).astype(np.float64)
    p_cont = cont / cont.sum()
    # blend with lexicon prior so words never seen in the corpus still get sensible mass
    prior = np.fromfile(os.path.join(B, "prior.f32"), dtype=np.float32).astype(np.float64) if os.path.exists(os.path.join(B, "prior.f32")) else None
    if prior is None:
        prior = np.ones(V) / V
    lam = 0.03
    p_uni = (1 - lam) * p_cont + lam * prior
    p_uni /= p_uni.sum()
    # bigram
    c_prev = np.bincount(prev, weights=C, minlength=V)
    D = np.where(C == 1, D1, np.where(C == 2, D2, D3))
    n1v = np.bincount(prev, weights=(C == 1), minlength=V)
    n2v = np.bincount(prev, weights=(C == 2), minlength=V)
    n3v = np.bincount(prev, weights=(C >= 3), minlength=V)
    gamma = np.where(c_prev > 0, (D1 * n1v + D2 * n2v + D3 * n3v) / np.maximum(c_prev, 1), 1.0)
    p_bi = (C - D) / c_prev[prev] + gamma[prev] * p_uni[cur]
    keep = C >= MIN_BIGRAM
    # backoff weight after pruning: alpha(v) = (1 - sum_kept p_bi) / (1 - sum_kept p_uni)
    kept_bi = np.bincount(prev[keep], weights=p_bi[keep], minlength=V)
    kept_uni = np.bincount(prev[keep], weights=p_uni[cur[keep]], minlength=V)
    alpha = np.where(kept_bi > 0, (1 - kept_bi) / np.maximum(1 - kept_uni, 1e-9), 1.0)
    alpha = np.clip(alpha, 1e-6, None)
    prevk, curk, pk = prev[keep], cur[keep], p_bi[keep]
    order = np.lexsort((curk, prevk))
    prevk, curk, pk = prevk[order], curk[order], pk[order]
    start = np.zeros(V + 1, dtype=np.uint32)
    counts = np.bincount(prevk, minlength=V)
    start[1:] = np.cumsum(counts)
    logu = np.log(p_uni).astype(np.float32)
    logb = np.log(alpha).astype(np.float32)
    logp = np.log(pk).astype(np.float16)

    def pad(b):
        return b + b"\0" * ((-len(b)) % 8)
    out = bytearray(b"STLM" + struct.pack("<3I", 1, V, len(pk)))
    out = pad(bytes(out))
    for sec in [logu.tobytes(), logb.tobytes(), start.tobytes(), curk.astype(np.uint32).tobytes(), logp.tobytes()]:
        out += pad(sec)
    with open(os.path.join(B, "lm.bin"), "wb") as f:
        f.write(out)
    print("lm.bin", len(out) / 1e6, "MB bigrams kept", len(pk), file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "estimate":
        init_worker()
        d = np.load(os.path.join(B, "counts.npz"))
        estimate(d["uni"], d["K"], d["C"])
    else:
        main()
