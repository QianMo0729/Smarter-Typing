#!/usr/bin/env python3
"""Extract clean Chinese paragraphs from the raw corpora.

Each output line is one paragraph (document context is kept together so that
evaluation can use real preceding context).  Lines are prefixed by the source tag:
    <source>\t<paragraph>
Output: data/build/corpus/<source>.txt
"""
import os, re, json, gzip, sys, random
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/raw")
OUT = os.path.join(ROOT, "data/build/corpus")
os.makedirs(OUT, exist_ok=True)

HAN = re.compile(r"[一-鿿]")
WS = re.compile(r"[ \t　\xa0]+")
URL = re.compile(r"https?://\S+|www\.\S+")
TRAD_HINT = set("們這個來說對時會國過還後從學發開關問題點與於為們愛經見長門間體現")  # crude traditional detector


def clean_para(p):
    p = URL.sub("", p)
    p = p.replace("\r", "")
    p = WS.sub(" ", p).strip()
    if len(p) < 8:
        return None
    han = len(HAN.findall(p))
    if han < 6 or han / len(p) < 0.5:
        return None
    if sum(1 for c in p if c in TRAD_HINT) > 2:
        return None
    return p


def split_paras(text):
    for p in re.split(r"\n+", text):
        p = clean_para(p)
        if p:
            yield p[:2000]


def write(name, gen, limit_chars):
    n = 0
    with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as f:
        for doc in gen:
            paras = list(split_paras(doc))
            if not paras:
                continue
            # keep doc boundaries with an empty line
            for p in paras:
                f.write(p + "\n")
                n += len(p)
            f.write("\n")
            if n >= limit_chars:
                break
    print(name, n, "chars", file=sys.stderr)


def wiki():
    with open(os.path.join(RAW, "wiki.json"), encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        yield d.get("completion") or d.get("text") or ""


def zhihu():
    for fn in ["zhihu0.parquet", "zhihu1.parquet"]:
        path = os.path.join(RAW, fn)
        if not os.path.exists(path):
            continue
        t = pq.read_table(path)
        cols = t.column_names
        for batch in t.to_batches(10000):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                q = d.get("INSTRUCTION", [""] * batch.num_rows)[i] or ""
                a = d.get("RESPONSE", [""] * batch.num_rows)[i] or ""
                yield q + "\n" + a


def v2ex(fn, field_candidates, exclude=()):
    path = os.path.join(RAW, fn)
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if (d.get("member") or {}).get("username") in exclude:
                continue
            parts = []
            for k in field_candidates:
                v = d.get(k)
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, list):
                    parts.extend(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in v)
            yield "\n".join(parts)


def fineweb():
    t = pq.read_table(os.path.join(RAW, "fineweb0.parquet"), columns=["text"])
    for batch in t.to_batches(5000):
        for x in batch.column(0).to_pylist():
            yield x or ""


def v2ex_members():
    """Chinese character count per member (replies)."""
    import collections
    cnt = collections.Counter()
    posts = collections.Counter()
    with gzip.open(os.path.join(RAW, "v2ex_replies.jsonl.gz"), "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = (d.get("member") or {}).get("username")
            if u:
                cnt[u] += len(HAN.findall(d.get("content") or ""))
                posts[u] += 1
    return cnt, posts


def v2ex_replies(exclude):
    """General replies (excluding the simulated users). Replies of the same topic form one document."""
    cur_topic, buf = None, []
    with gzip.open(os.path.join(RAW, "v2ex_replies.jsonl.gz"), "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = (d.get("member") or {}).get("username")
            if u in exclude:
                continue
            t = d.get("topic_id")
            if t != cur_topic and buf:
                yield "\n".join(buf); buf = []
            cur_topic = t
            buf.append(d.get("content") or "")
    if buf:
        yield "\n".join(buf)


def v2ex_user(username):
    """One member's replies in chronological order, each reply a document."""
    rows = []
    with gzip.open(os.path.join(RAW, "v2ex_replies.jsonl.gz"), "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if username not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if (d.get("member") or {}).get("username") == username:
                rows.append((d.get("created") or 0, d.get("content") or ""))
    rows.sort()
    for _, c in rows:
        yield c


if __name__ == "__main__":
    which = sys.argv[1:] or ["wiki", "zhihu", "v2ex_topic", "v2ex_replies", "fineweb"]
    users_file = os.path.join(OUT, "..", "sim_users.txt")
    for w in which:
        if w == "wiki":
            write("wiki", wiki(), 350_000_000)
        elif w == "zhihu":
            write("zhihu", zhihu(), 350_000_000)
        elif w == "v2ex_topic":
            users = [l.split("\t")[0] for l in open(users_file)][:8]
            write("v2ex_topic", v2ex("v2ex_topic.jsonl.gz", ["title", "content", "text"], set(users)), 150_000_000)
        elif w == "v2ex_members":
            cnt, posts = v2ex_members()
            with open(users_file, "w") as f:
                for u, c in cnt.most_common(40):
                    f.write(f"{u}\t{c}\t{posts[u]}\n")
            print(open(users_file).read(), file=sys.stderr)
        elif w == "v2ex_replies":
            users = [l.split("\t")[0] for l in open(users_file)][:8]
            write("v2ex_replies", v2ex_replies(set(users)), 200_000_000)
        elif w.startswith("user:"):
            name = w[5:]
            os.makedirs(os.path.join(OUT, "..", "users"), exist_ok=True)
            n = 0
            with open(os.path.join(OUT, "..", "users", name + ".txt"), "w", encoding="utf-8") as f:
                for doc in v2ex_user(name):
                    paras = list(split_paras(doc))
                    if paras:
                        f.write("\n".join(paras) + "\n\n"); n += sum(len(p) for p in paras)
            print("user", name, n, "chars", file=sys.stderr)
        elif w == "fineweb":
            write("fineweb", fineweb(), 300_000_000)
