#!/usr/bin/env python3
"""Compile lexicon.tsv into the mmap-friendly binary used by the Swift engine.

lex.bin layout (little endian, every section 8-byte aligned):
  header: magic 'STLX', version u32, nSyl, nSurf, nNodes, nEdges, nWords, blobLen (u32 each)
  sylBlob       : syllable spellings joined with '\n' (u8, length in header2)
  surfOffsets   : u32[nSurf+1]
  surfBlob      : u8[blobLen]              UTF-8 of all surfaces
  surfChars     : u8[nSurf]                number of characters of each surface
  prior         : f32[nSurf]               log prior from dictionary weights (fallback unigram)
  nodeChild     : u32[nNodes+1]            child range per trie node
  childSyl      : u16[nEdges]
  childNode     : u32[nEdges]
  nodeWord      : u32[nNodes+1]            word range per trie node
  wordSurf      : u32[nWords]
  wordReading   : f32[nWords]              log P(reading | surface)
Also writes surfaces.txt (id -> surface) used by the n-gram builder.
Special tokens are appended to the surface table (no readings): see SPECIALS.
"""
import os, sys, math, struct, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "data/build")

SPECIALS = ["<s>", "</s>", "<unk>", "<eng>", "<num>",
            "，", "。", "？", "！", "、", "：", "；", "“", "”", "‘", "’", "（", "）", "《", "》", "…", "—", "·", "～"]


def main():
    syls = [l.strip() for l in open(os.path.join(B, "syllables.txt")) if l.strip()]
    sid = {s: i for i, s in enumerate(syls)}
    entries = []
    for line in open(os.path.join(B, "lexicon.tsv"), encoding="utf-8"):
        w, py, wt = line.rstrip("\n").split("\t")
        entries.append((w, tuple(sid[s] for s in py.split(" ")), int(wt)))

    surf_weight = collections.defaultdict(int)
    for w, _, wt in entries:
        surf_weight[w] += wt
    # surface ids ordered by weight desc; when a surfaces.txt already exists (the n-gram model was built on
    # it), keep those ids stable and append new surfaces at the end
    old_path = os.path.join(B, "surfaces.txt")
    if os.path.exists(old_path) and "--renumber" not in sys.argv:
        surfaces = [l.rstrip("\n") for l in open(old_path, encoding="utf-8")]
        known = set(surfaces)
        surfaces += sorted((w for w in surf_weight if w not in known), key=lambda w: -surf_weight[w])
    else:
        surfaces = sorted(surf_weight, key=lambda w: -surf_weight[w])
        surfaces += [s for s in SPECIALS if s not in surf_weight]
    surf_id = {w: i for i, w in enumerate(surfaces)}
    total = sum(surf_weight.values()) + len(surfaces)
    prior = np.array([math.log((surf_weight.get(w, 0) + 1) / total) for w in surfaces], dtype=np.float32)

    # log P(reading | surface)
    reading = []
    for w, py, wt in entries:
        tot = surf_weight[w]
        n_readings = 1
        reading.append(math.log((wt + 1) / (tot + 1)) if tot > 0 else 0.0)

    # build trie over syllable sequences
    children = [dict()]
    words = [[]]
    for (w, py, wt), rl in zip(entries, reading):
        node = 0
        for s in py:
            nxt = children[node].get(s)
            if nxt is None:
                nxt = len(children)
                children[node][s] = nxt
                children.append(dict())
                words.append([])
            node = nxt
        words[node].append((surf_id[w], rl, wt))

    # BFS renumbering for locality
    order = [0]
    new_id = {0: 0}
    i = 0
    while i < len(order):
        n = order[i]; i += 1
        for s in sorted(children[n]):
            c = children[n][s]
            new_id[c] = len(order)
            order.append(c)
    n_nodes = len(order)
    node_child = np.zeros(n_nodes + 1, dtype=np.uint32)
    child_syl, child_node = [], []
    node_word = np.zeros(n_nodes + 1, dtype=np.uint32)
    word_surf, word_read = [], []
    for k, old in enumerate(order):
        node_child[k] = len(child_syl)
        for s in sorted(children[old]):
            child_syl.append(s)
            child_node.append(new_id[children[old][s]])
        node_word[k] = len(word_surf)
        for sidx, rl, wt in sorted(words[old], key=lambda x: -x[2]):
            word_surf.append(sidx)
            word_read.append(rl)
    node_child[n_nodes] = len(child_syl)
    node_word[n_nodes] = len(word_surf)

    blob = bytearray()
    offs = np.zeros(len(surfaces) + 1, dtype=np.uint32)
    chars = np.zeros(len(surfaces), dtype=np.uint8)
    for k, w in enumerate(surfaces):
        offs[k] = len(blob)
        blob += w.encode("utf-8")
        chars[k] = min(255, len(w))
    offs[len(surfaces)] = len(blob)
    syl_blob = "\n".join(syls).encode()

    def pad(b):
        return b + b"\0" * ((-len(b)) % 8)

    out = bytearray()
    out += b"STLX" + struct.pack("<8I", 1, len(syls), len(surfaces), n_nodes, len(child_syl), len(word_surf), len(blob), len(syl_blob))
    out = pad(bytes(out))
    for sec in [syl_blob, offs.tobytes(), bytes(blob), chars.tobytes(), prior.tobytes(),
                node_child.tobytes(), np.array(child_syl, dtype=np.uint16).tobytes(),
                np.array(child_node, dtype=np.uint32).tobytes(), node_word.tobytes(),
                np.array(word_surf, dtype=np.uint32).tobytes(), np.array(word_read, dtype=np.float32).tobytes()]:
        out += pad(sec)
    with open(os.path.join(B, "lex.bin"), "wb") as f:
        f.write(out)
    np.exp(prior.astype(np.float64)).astype(np.float32).tofile(os.path.join(B, "prior.f32"))
    with open(os.path.join(B, "surfaces.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(surfaces) + "\n")
    print("surfaces", len(surfaces), "nodes", n_nodes, "edges", len(child_syl), "words", len(word_surf), "bytes", len(out))


if __name__ == "__main__":
    main()
