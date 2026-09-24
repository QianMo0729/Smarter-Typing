import Foundation

/// A lexicon word placed on a span of keys. Syllables live in the lattice's shared pool.
public struct WordEdge {
    public var start: Int32
    public var end: Int32
    public var word: WordID
    public var sylStart: Int32     // index into Lattice.sylPool
    public var sylCount: Int16
    public var typos: Int16
    public var abbrevs: Int16
    public var partial: Bool
    public var editRef: Int32      // index into Lattice.typoEdges (-1 when no slip is assumed)
    public var reading: Float      // log P(reading | word)
    public var typoCost: Float     // channel cost of keyboard slips
    public var shapeCost: Float    // cost of abbreviated / unfinished syllables
    public var unigram: Float      // cached base unigram (for pruning / ordering)
}

public struct LatticeOptions {
    public var maxWordSyllables = 8
    public var maxTyposPerWord = 1
    public var wordsPerNode = 32
    public var wordsPerSpan = 16          // words read without any assumed slip
    public var typoWordsPerSpan = 6       // words that need a correction get their own, smaller quota
    public var visitBudget = 40_000
    public init() {}
}

/// Word lattice built from a syllable graph and the lexicon trie.
public struct Lattice {
    public let graph: SyllableGraph
    public var edgesFrom: [[WordEdge]]
    public var sylPool: [SylID] = []
    public var typoEdges: [SylEdge] = []
    public var n: Int { graph.n }

    public init(graph: SyllableGraph, lexicon: Lexicon, lm: NGramModel, options: LatticeOptions = LatticeOptions()) {
        self.graph = graph
        edgesFrom = Array(repeating: [], count: graph.n + 1)
        sylPool.reserveCapacity(4096)
        let reach = graph.reachable()
        var scratch = Scratch(n: graph.n)
        for s in 0..<graph.n where reach[s] {
            expand(from: s, lexicon: lexicon, lm: lm, options: options, scratch: &scratch)
        }
    }

    @inline(__always) public func syls(_ e: WordEdge) -> ArraySlice<SylID> {
        sylPool[Int(e.sylStart)..<Int(e.sylStart) + Int(e.sylCount)]
    }

    public func edits(_ e: WordEdge) -> [SylEdge] { e.editRef >= 0 ? [typoEdges[Int(e.editRef)]] : [] }

    /// Every lexicon word that can start the input, read literally (for the full candidate list shown to the
    /// user, like a regular IME's pages of single characters and words). Returns (word, end, syllables).
    public static func allFirstWords(graph: SyllableGraph, lexicon: Lexicon, maxSyllables: Int = 8, limit: Int = 2000) -> [(WordID, Int, [SylID], Float)] {
        var out: [(WordID, Int, [SylID], Float)] = []
        var stack: [(pos: Int, node: Int, syls: [SylID], shape: Float)] = [(0, 0, [], 0)]
        while let f = stack.popLast() {
            if f.syls.count >= maxSyllables || out.count >= limit { continue }
            for e in graph.edgesFrom[f.pos] where e.kind == .exact || e.kind == .partial {
                let syls = graph.syls(e)
                lexicon.forEachChild(f.node, in: syls) { syl, child in
                    let path = f.syls + [syl]
                    let shape = f.shape + e.cost
                    for wi in lexicon.wordRange(child) { out.append((lexicon.wordSurface(wi), e.end, path, shape)) }
                    if e.kind == .exact && e.end < graph.n && lexicon.hasChildren(child) { stack.append((e.end, child, path, shape)) }
                }
            }
        }
        return out
    }

    struct Frame { var pos: Int32; var node: Int32; var parent: Int32; var syl: SylID; var depth: Int16; var typos: Int16; var abbrevs: Int16; var partial: Bool; var editRef: Int32; var typo: Float; var shape: Float }

    struct Scratch {
        var frames: [Frame] = []
        var stack: [Int32] = []
        var bySpan: [[WordEdge]]
        var pathBuf: [SylID] = []
        init(n: Int) { bySpan = Array(repeating: [], count: n + 1) }
    }

    private mutating func expand(from s: Int, lexicon: Lexicon, lm: NGramModel, options: LatticeOptions, scratch: inout Scratch) {
        scratch.frames.removeAll(keepingCapacity: true)
        scratch.stack.removeAll(keepingCapacity: true)
        for i in scratch.bySpan.indices where !scratch.bySpan[i].isEmpty { scratch.bySpan[i].removeAll(keepingCapacity: true) }
        scratch.frames.append(Frame(pos: Int32(s), node: 0, parent: -1, syl: 0, depth: 0, typos: 0, abbrevs: 0, partial: false, editRef: -1, typo: 0, shape: 0))
        scratch.stack.append(0)
        var visits = 0
        let n = graph.n
        outer: while let fi = scratch.stack.popLast() {
            let f = scratch.frames[Int(fi)]
            if Int(f.depth) >= options.maxWordSyllables { continue }
            for e in graph.edgesFrom[Int(f.pos)] {
                let isTypo = e.kind == .typo
                if isTypo && Int(f.typos) >= options.maxTyposPerWord { continue }
                let isAbbrev = e.kind == .abbrev
                if isAbbrev && f.abbrevs >= 4 { continue }
                var typoRef: Int32 = f.editRef
                if isTypo {
                    typoRef = Int32(typoEdges.count)
                    typoEdges.append(e)
                }
                var stop = false
                func visit(_ syl: SylID, _ child: Int) {
                    if stop { return }
                    visits += 1
                    if visits > options.visitBudget { stop = true; return }
                    var nf = Frame(pos: Int32(e.end), node: Int32(child), parent: fi, syl: syl, depth: f.depth + 1,
                                   typos: f.typos + (isTypo ? 1 : 0), abbrevs: f.abbrevs + (isAbbrev ? 1 : 0),
                                   partial: f.partial || e.kind == .partial, editRef: typoRef,
                                   typo: f.typo + (isTypo ? e.cost : 0), shape: f.shape + (isTypo ? 0 : e.cost))
                    if nf.partial && e.kind != .partial { nf.partial = true }
                    let wr = lexicon.wordRange(child)
                    // an abbreviated single syllable is too ambiguous to stand alone as a word
                    if !wr.isEmpty && !(nf.depth == 1 && isAbbrev) {
                        let fid = Int32(scratch.frames.count)
                        scratch.frames.append(nf)
                        let sylStart = Int32(sylPool.count)
                        // write syllables of this path into the pool (walk parents)
                        scratch.pathBuf.removeAll(keepingCapacity: true)
                        var p = fid
                        while p > 0 { scratch.pathBuf.append(scratch.frames[Int(p)].syl); p = scratch.frames[Int(p)].parent }
                        sylPool.append(contentsOf: scratch.pathBuf.reversed())
                        var k = 0
                        for wi in wr {
                            if k >= options.wordsPerNode { break }
                            k += 1
                            let w = lexicon.wordSurface(wi)
                            scratch.bySpan[e.end].append(WordEdge(start: Int32(s), end: Int32(e.end), word: w, sylStart: sylStart, sylCount: nf.depth,
                                typos: nf.typos, abbrevs: nf.abbrevs, partial: nf.partial, editRef: nf.typos > 0 ? nf.editRef : -1,
                                reading: lexicon.wordReadingLogP(wi), typoCost: nf.typo, shapeCost: nf.shape, unigram: lm.unigramLogP(w)))
                        }
                        if e.end < n && lexicon.hasChildren(child) && !nf.partial { scratch.stack.append(fid) }
                    } else if e.end < n && lexicon.hasChildren(child) && !nf.partial {
                        scratch.frames.append(nf)
                        scratch.stack.append(Int32(scratch.frames.count - 1))
                    }
                }
                if e.setRef < 0 {
                    let c = lexicon.child(Int(f.node), e.syl)
                    if c >= 0 { visit(e.syl, c) }
                } else {
                    lexicon.forEachChild(Int(f.node), in: graph.sets[Int(e.setRef)], visit)
                }
                if stop { break outer }
            }
        }
        for end in (s + 1)...n where !scratch.bySpan[end].isEmpty {
            let list = scratch.bySpan[end]
            // cheapest reading per word, then the best few words of the span
            var keyed = [(Float, Int32, Int32)]()   // (local score, word, index)
            keyed.reserveCapacity(list.count)
            for (i, e) in list.enumerated() { keyed.append((e.unigram + e.reading - e.typoCost - e.shapeCost, e.word, Int32(i))) }
            keyed.sort { $0.0 > $1.0 }
            // literal readings and corrected readings have separate quotas, so corrections of common
            // words can never crowd out a word the user actually spelled
            var exact = 0, typo = 0
            var words = Set<WordID>()
            for (_, w, i) in keyed {
                let e = list[Int(i)]
                if e.typos == 0 {
                    if exact >= options.wordsPerSpan || words.contains(w) { continue }
                    exact += 1
                } else {
                    if typo >= options.typoWordsPerSpan || words.contains(w) { continue }
                    typo += 1
                }
                words.insert(w)
                edgesFrom[s].append(e)
                if exact >= options.wordsPerSpan && typo >= options.typoWordsPerSpan { break }
            }
        }
    }
}
