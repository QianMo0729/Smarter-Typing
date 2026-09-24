#!/usr/bin/env python3
"""Merge rime-ice dictionaries into a flat lexicon (word, pinyin, weight).

Output:
  data/build/syllables.txt   one toneless syllable per line (index = syllable id)
  data/build/lexicon.tsv     word \t pinyin(space separated) \t weight
  data/build/jieba_dict.txt  word freq   (surface-level, for corpus segmentation)
"""
import os, re, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIME = os.path.join(ROOT, "third_party/rime-ice/cn_dicts")
OUT = os.path.join(ROOT, "data/build")
os.makedirs(OUT, exist_ok=True)

HAN = re.compile(r"^[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f〇]+$")

def read_dict(name):
    body = False
    with open(os.path.join(RIME, name), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not body:
                if line.strip() == "...":
                    body = True
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word, py = parts[0], parts[1]
            w = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else DEFAULT_WEIGHT.get(name, 0)
            yield word, py.strip(), w

entries = collections.defaultdict(int)  # (word, py) -> weight
# 41448: rare characters (no weights) so single-character coverage matches the system IME;
# others: alternative readings of set phrases (阿房宫 e pang gong ...)
DEFAULT_WEIGHT = {"41448.dict.yaml": 1, "others.dict.yaml": 50}
for name in ["8105.dict.yaml", "base.dict.yaml", "ext.dict.yaml", "others.dict.yaml", "41448.dict.yaml"]:
    n = 0
    for word, py, w in read_dict(name):
        if not HAN.match(word):
            continue
        syls = py.split(" ")
        if len(syls) != len(word):
            continue
        key = (word, py)
        if name == "41448.dict.yaml" and any(k[0] == word for k in [key]) and key in entries:
            continue
        entries[key] = max(entries[key], w)
        n += 1
    print(name, n)

# Drop readings that are clearly marginal for single chars (weight tiny vs main reading),
# they add noise without helping (e.g. 行 heng = 1).
by_word = collections.defaultdict(list)
for (word, py), w in entries.items():
    by_word[word].append((py, w))
final = []
for word, rs in by_word.items():
    mx = max(w for _, w in rs)
    for py, w in rs:
        if len(word) == 1 and mx > 1000 and w * 5000 < mx:
            continue
        final.append((word, py, w))

syls = sorted({s for _, py, _ in final for s in py.split(" ")})
with open(os.path.join(OUT, "syllables.txt"), "w") as f:
    f.write("\n".join(syls) + "\n")

final.sort(key=lambda x: (x[1], -x[2]))
with open(os.path.join(OUT, "lexicon.tsv"), "w", encoding="utf-8") as f:
    for word, py, w in final:
        f.write(f"{word}\t{py}\t{w}\n")

surf = collections.defaultdict(int)
for word, py, w in final:
    surf[word] += w
with open(os.path.join(OUT, "jieba_dict.txt"), "w", encoding="utf-8") as f:
    for word, w in surf.items():
        f.write(f"{word} {max(1, w)}\n")

print("syllables", len(syls), "entries", len(final), "surfaces", len(surf))
