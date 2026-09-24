#!/usr/bin/env python3
"""Fit the log-linear decision layer (conditional logit over the candidate set) from feature dumps.

usage: fit_weights.py <train_feats.jsonl> <out weights.json> [--base existing weights.json]
Feature order must match Features.names in Candidate.swift.
"""
import sys, json, argparse
import numpy as np
import torch

NAMES = ["lm", "reading", "typoCost", "typos", "shapeCost", "words", "nn", "personal", "memory", "nnMissing", "nnCtx", "lmCtx"]
FIXED = {"personal": 0.87, "memory": 0.73, "nnMissing": 0.0}   # personal trust fitted on simulated users; nnMissing unused


def load(path):
    X, T, Y, TY = [], [], [], []
    for line in open(path):
        d = json.loads(line)
        if d["label"] < 0 or len(d["f"]) < 2:
            continue
        X.append(np.array(d["f"], dtype=np.float32))
        T.append(np.array(d["t"], dtype=np.int32))
        Y.append(d["label"]); TY.append(d["typo"])
    return X, T, Y, TY


def fit(X, Y, use, init, l2=1e-3, iters=300, sample_w=None):
    idx = [NAMES.index(n) for n in use]
    w = torch.tensor([init[n] for n in use], dtype=torch.float64, requires_grad=True)
    maxlen = max(len(x) for x in X)
    F = np.zeros((len(X), maxlen, len(idx)))
    M = np.full((len(X), maxlen), -1e9)
    # features that are not being fitted still contribute with their current weights (fixed offset)
    fixed = np.array([0.0 if n in use else init[n] for n in NAMES])
    for i, x in enumerate(X):
        F[i, :len(x)] = x[:, idx]
        M[i, :len(x)] = x @ fixed
    F = torch.tensor(F); M = torch.tensor(M); Yt = torch.tensor(Y)
    sw = torch.tensor(sample_w if sample_w is not None else np.ones(len(X)))
    w0 = torch.tensor([init[n] for n in use], dtype=torch.float64)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = (F * w).sum(-1) + M
        lp = torch.log_softmax(s, dim=1)
        nll = -(lp[torch.arange(len(Y)), Yt] * sw).sum() / sw.sum()
        loss = nll + l2 * ((w - w0) ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        s = (F * w).sum(-1) + M
        acc = (s.argmax(1) == Yt).double().mean().item()
    out = dict(init)
    for n, v in zip(use, w.detach().numpy()):
        out[n] = float(v)
    return out, acc


def evaluate(X, T, Y, TY, weights, guard):
    wv = np.array([weights[n] for n in NAMES])
    ok = fc = clean = typo_ok = typo_n = 0
    for x, t, y, ty in zip(X, T, Y, TY):
        s = x @ wv
        order = np.argsort(-s)
        top = order[0]
        if t[top] > 0:
            lits = [j for j in order if t[j] == 0]
            if lits and s[top] - s[lits[0]] < guard:
                top = lits[0]
        good = top == y
        ok += good
        if ty:
            typo_n += 1; typo_ok += good
        else:
            clean += 1
            if t[top] > 0 and not good:
                fc += 1
    return ok / len(Y), fc / max(1, clean), typo_ok / max(1, typo_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats"); ap.add_argument("out")
    ap.add_argument("--clean-weight", type=float, default=1.0, help="upweight clean examples to discourage false corrections")
    ap.add_argument("--personal", nargs="*", default=[], help="feature dumps made with learned personal state; fits the trust in personal signals")
    a = ap.parse_args()
    X, T, Y, TY = load(a.feats)
    print("examples with target in list:", len(X))
    sw = np.array([1.0 if ty else a.clean_weight for ty in TY])
    init = {"lm": 1.0, "reading": 1.0, "typoCost": -1.0, "typos": 0.0, "shapeCost": -1.0, "words": -0.2, "nn": 0.0, "nnCtx": 0.0, "lmCtx": 0.0, **FIXED}
    fast, acc_fast = fit(X, Y, ["lm", "reading", "typoCost", "typos", "shapeCost", "words", "lmCtx"], init, sample_w=sw)
    print("fast weights", {k: round(v, 3) for k, v in fast.items()}, "acc(in-list)", round(acc_fast, 4))
    init_full = dict(fast); init_full["nn"] = 1.0; init_full["lm"] = 0.3
    full, acc_full = fit(X, Y, ["lm", "reading", "typoCost", "typos", "shapeCost", "words", "nn", "nnCtx", "lmCtx"], init_full, sample_w=sw)
    print("full weights", {k: round(v, 3) for k, v in full.items()}, "acc(in-list)", round(acc_full, 4))
    if a.personal:
        # fit only the trust in personal signals (base weights stay as fitted on general text)
        PX, PT, PY, PTY = [], [], [], []
        for f in a.personal:
            x, t, y, ty = load(f)
            PX += x; PT += t; PY += y; PTY += ty
        print("personal examples", len(PX))
        full, acc_p = fit(PX, PY, ["personal", "memory"], full, l2=1e-2)
        fast, _ = fit([np.where(np.arange(x.shape[1]) == NAMES.index("nn"), 0, x) for x in PX], PY, ["personal", "memory"], fast, l2=1e-2)
        print("trust in personal signals: full", round(full["personal"], 3), round(full["memory"], 3), "fast", round(fast["personal"], 3), round(fast["memory"], 3), "acc", round(acc_p, 4))
    best = None
    for g in [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        acc, fc, tr = evaluate(X, T, Y, TY, full, g)
        print(f"guard {g}: acc {acc:.4f} false-corr {fc:.4f} typo-recovery {tr:.4f}")
        # prefer fewer false corrections: accept at most 0.2% accuracy loss per 0.5% fc reduction
        score = acc - 0.5 * fc
        if best is None or score > best[0]:
            best = (score, g)
    out = {"fast": [fast[n] for n in NAMES], "full": [full[n] for n in NAMES], "typoGuard": best[1]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, "guard", best[1])


if __name__ == "__main__":
    main()
