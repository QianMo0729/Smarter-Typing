#!/usr/bin/env python3
"""Compare small LMs as candidate judges on dumped candidate sets (context + candidates + fast features).
For each model: NN-only top-1, and fused top-1 after refitting lm/nn/other weights by 2-fold CV."""
import sys, json, math, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

data = [json.loads(l) for l in open(sys.argv[1])]
data = [d for d in data if d["label"] >= 0 and len(d["cands"]) > 1]
models = sys.argv[2:]
NAMES = ["lm", "reading", "typoCost", "typos", "shapeCost", "words", "nn", "personal", "memory", "nnMissing"]

def score_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    except Exception:
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = m.to(dev, dtype=torch.float16 if dev == "mps" else torch.float32)
    m.eval()
    bos = tok.eos_token_id if tok.bos_token_id is None else tok.bos_token_id
    out = []
    t0 = time.time()
    for d in data:
        ctx = [bos] + tok.encode(d["context"])
        cands = [tok.encode(c) for c in d["cands"]]
        L = max(len(c) for c in cands)
        # fixed shapes (bucketed length, fixed batch) so the GPU kernels are compiled once
        T = ((len(ctx) + L + 31) // 32) * 32
        B = ((len(cands) + 15) // 16) * 16
        ids = torch.full((B, T), tok.eos_token_id or 0)
        for i, c in enumerate(cands):
            ids[i, :len(ctx)] = torch.tensor(ctx); ids[i, len(ctx):len(ctx) + len(c)] = torch.tensor(c)
        bi, pi, ti, ci = [], [], [], []
        for i, c in enumerate(cands):
            for k, t in enumerate(c):
                bi.append(i); pi.append(len(ctx) - 1 + k); ti.append(t); ci.append(i)
        with torch.no_grad():
            logits = m(input_ids=ids.to(m.device)).logits
            b = torch.tensor(bi, device=logits.device); p_ = torch.tensor(pi, device=logits.device); t_ = torch.tensor(ti, device=logits.device)
            rows = logits[b, p_].float()                       # (n_tokens, V) only the rows we need
            vals = (rows.gather(1, t_[:, None])[:, 0] - torch.logsumexp(rows, -1)).cpu().numpy()
        s = [0.0] * len(cands)
        for v, i in zip(vals, ci):
            s[i] += float(v)
        out.append(s)
    print(path, "scored in", round(time.time() - t0), "s", file=sys.stderr)
    return out

def fit_eval(train, test, nnscores):
    # conditional logit on [lm, reading, typoCost, typos, shapeCost, words, nn]
    idx = [0, 1, 2, 3, 4, 5]
    def feats(ds, ns):
        X = []
        for d, n in zip(ds, ns):
            f = np.array(d["f"])[:, idx]
            X.append(np.concatenate([f, np.array(n)[:, None]], 1))
        return X
    Xtr = feats([data[i] for i in train], [nnscores[i] for i in train])
    Ytr = [data[i]["label"] for i in train]
    w = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    with torch.no_grad(): w[:] = torch.tensor([0.5, 0.8, -1, -0.3, -1.7, -0.6, 0.5])
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = 0
        for x, y in zip(Xtr, Ytr):
            s = torch.tensor(x) @ w
            loss = loss - torch.log_softmax(s, 0)[y]
        loss = loss / len(Xtr) + 1e-3 * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    Xte = feats([data[i] for i in test], [nnscores[i] for i in test])
    ok = sum(int(np.argmax(x @ w.detach().numpy()) == data[i]["label"]) for x, i in zip(Xte, test))
    return ok / len(test), w.detach().numpy()

n = len(data)
folds = [list(range(0, n, 2)), list(range(1, n, 2))]
base_ok = sum(int(np.argmax(np.array(d["f"])[:, 0]) == d["label"]) for d in data) / n
print("examples", n, "n-gram only", round(base_ok, 4))
for path in models:
    ns = score_model(path)
    nn_only = sum(int(np.argmax(s) == d["label"]) for s, d in zip(ns, data)) / n
    accs = []
    for k in range(2):
        acc, w = fit_eval(folds[k], folds[1 - k], ns)
        accs.append(acc)
    print(path, "nn-only", round(nn_only, 4), "fused(2-fold)", round(sum(accs) / 2, 4), "weights", np.round(w, 2))
