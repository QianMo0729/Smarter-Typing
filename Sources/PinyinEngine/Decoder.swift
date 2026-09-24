import Foundation

/// Log-linear weights of the probabilistic decision layer. `score = w · features`, `P = softmax(score)`.
public struct FusionWeights: Codable, Sendable {
    public var w: [Float]

    /// Older weight files (without the context interactions) are padded with zeros.
    public init(_ w: [Float]) {
        var v = w
        if v.count < Features.count { v += Array(repeating: 0, count: Features.count - v.count) }
        precondition(v.count == Features.count)
        self.w = v
    }

    //                                     lm   read typoC typos shape words  nn  pers  mem  nnMissing nnCtx lmCtx
    public static let fast = FusionWeights([1.0, 1.0, -1.0, 0.0, -1.0, -0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    public static let full = FusionWeights([0.35, 0.6, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    /// Effective n-gram weight for a given amount of context.
    @inline(__always) public func lmWeight(ctx: Float) -> Float { w[0] + w[11] * ctx }

    @inline(__always) public func score(_ f: Features) -> Float {
        w[0] * f.lm + w[1] * f.reading + w[2] * f.typoCost + w[3] * f.typos + w[4] * f.shapeCost
            + w[5] * f.words + w[6] * f.nn + w[7] * f.personal + w[8] * f.memory + w[9] * f.nnMissing
            + (w[10] * f.nn + w[11] * f.lm) * f.ctx
    }
}

/// Word-level bonus supplied by the personal model during search.
public protocol PathBonus {
    /// log-ratio bonus of `word` following `prev` (0 = no opinion)
    func bonus(prev: WordID, word: WordID) -> Float
}

public struct DecoderOptions {
    public var beam = 20
    public var perState = 2
    public var literalSlots = 4        // hypotheses without assumed slips always kept per position
    public var nbest = 20
    public var variantPaths = 6
    public var variantsPerSegment = 5
    public var maxCandidates = 32
    public var maxPrefix = 60
    public var maxTyposTotal = 2
    public init() {}
}

public struct DecodeResult {
    public var full: [Candidate]       // candidates covering all keys (or the longest parsable prefix)
    public var prefix: [Candidate]     // single-word candidates starting at key 0
    public var covered: Int            // keys covered by `full`
    public var n: Int
}

public final class Decoder {
    let lexicon: Lexicon
    let lm: NGramModel
    public var options = DecoderOptions()

    public init(lexicon: Lexicon, lm: NGramModel) { self.lexicon = lexicon; self.lm = lm }

    struct Hyp { var score: Float; var last: WordID; var edgePos: Int32; var edgeIdx: Int32; var back: Int32; var typos: Int16 }

    @inline(__always) func staticScore(_ e: WordEdge, _ w: FusionWeights) -> Float {
        w.w[1] * e.reading + w.w[2] * e.typoCost + w.w[3] * Float(e.typos) + w.w[4] * e.shapeCost + w.w[5]
    }

    /// Context strength of the current decode (see Features.ctx).
    var ctx: Float = 0

    @inline(__always) func edgeScore(_ e: WordEdge, prev: WordID, weights: FusionWeights, bonus: PathBonus?) -> Float {
        var s = weights.lmWeight(ctx: ctx) * lm.logP(e.word, given: prev) + staticScore(e, weights)
        if let bonus { s += weights.w[7] * bonus.bonus(prev: prev, word: e.word) }
        return s
    }

    public func decode(lattice: Lattice, context: WordID, weights: FusionWeights = .fast, bonus: PathBonus? = nil, ctxStrength: Float = 0) -> DecodeResult {
        ctx = ctxStrength
        let n = lattice.n
        var pool: [Hyp] = [Hyp(score: 0, last: context, edgePos: -1, edgeIdx: -1, back: -1, typos: 0)]
        pool.reserveCapacity(4096)
        var beams = [[Int32]](repeating: [], count: n + 1)
        beams[0] = [0]
        var statics: [Float] = []
        for pos in 0..<n {
            if beams[pos].isEmpty { continue }
            prune(&beams[pos], pool: pool, limit: options.beam)
            let edges = lattice.edgesFrom[pos]
            if edges.isEmpty { continue }
            statics.removeAll(keepingCapacity: true)
            for e in edges { statics.append(staticScore(e, weights)) }
            let wLM = weights.lmWeight(ctx: ctx), wP = weights.w[7]
            for hi in beams[pos] {
                let h = pool[Int(hi)]
                for ei in 0..<edges.count {
                    let e = edges[ei]
                    let t = Int(h.typos) + Int(e.typos)
                    if t > options.maxTyposTotal { continue }
                    var s = h.score + statics[ei] + wLM * lm.logP(e.word, given: h.last)
                    if let bonus { s += wP * bonus.bonus(prev: h.last, word: e.word) }
                    pool.append(Hyp(score: s, last: e.word, edgePos: Int32(pos), edgeIdx: Int32(ei), back: hi, typos: Int16(t)))
                    beams[Int(e.end)].append(Int32(pool.count - 1))
                }
            }
        }
        // the furthest position with hypotheses
        var end = n
        while end > 0 && beams[end].isEmpty { end -= 1 }
        var full: [Candidate] = []
        if end > 0 {
            prune(&beams[end], pool: pool, limit: options.beam, perState: options.beam)
            var seen: [String: Int] = [:]
            for hi in beams[end] {
                let c = makeCandidate(path: path(hi, pool: pool, lattice: lattice), lattice: lattice, context: context, end: end, n: n)
                if let j = seen[c.text] {
                    if c.score > full[j].score { full[j] = c }
                } else { seen[c.text] = full.count; full.append(c) }
            }
            full.sort { $0.score > $1.score }
            if full.count > options.nbest { full.removeLast(full.count - options.nbest) }
            for i in full.indices { seen[full[i].text] = i }
            addVariants(&full, seen: &seen, lattice: lattice, context: context, end: end, n: n, weights: weights, bonus: bonus)
            for i in full.indices { full[i].score = weights.score(full[i].features) }
            full.sort { $0.score > $1.score }
            if full.count > options.maxCandidates { full.removeLast(full.count - options.maxCandidates) }
        }
        // prefix (first word) candidates
        var prefix: [Candidate] = []
        var seenP = Set(full.map { $0.text })
        let firstEdges = lattice.edgesFrom[0]
        let firstScores = firstEdges.map { edgeScore($0, prev: context, weights: weights, bonus: bonus) }
        let order = firstEdges.indices.sorted { a, b in
            if firstEdges[a].end != firstEdges[b].end { return firstEdges[a].end > firstEdges[b].end }
            return firstScores[a] > firstScores[b]
        }
        for i in order {
            let e = firstEdges[i]
            let text = lexicon.surface(e.word)
            if seenP.contains(text) { continue }
            seenP.insert(text)
            var c = makeCandidate(path: [e], lattice: lattice, context: context, end: Int(e.end), n: n)
            c.score = weights.score(c.features)
            prefix.append(c)
            if prefix.count >= options.maxPrefix { break }
        }
        return DecodeResult(full: full, prefix: prefix, covered: end, n: n)
    }

    /// Keep the best `limit` hypotheses (at most `perState` per last word), always retaining a few
    /// hypotheses that assume no keyboard slip so the literal reading survives to the end.
    private func prune(_ beam: inout [Int32], pool: [Hyp], limit: Int, perState: Int? = nil) {
        let ps = perState ?? options.perState
        if beam.count <= 1 { return }
        var keyed = beam.map { (pool[Int($0)].score, $0) }
        keyed.sort { $0.0 > $1.0 }
        var kept: [Int32] = []
        kept.reserveCapacity(limit + options.literalSlots)
        var perWord: [WordID: Int] = [:]
        var literal = 0
        for (_, h) in keyed {
            let hyp = pool[Int(h)]
            let isLit = hyp.typos == 0
            if kept.count >= limit {
                // beam is full: only literal hypotheses may still enter, until the reserved slots are used
                if literal >= options.literalSlots { break }
                if !isLit { continue }
            }
            let c = perWord[hyp.last, default: 0]
            if c >= ps { continue }
            perWord[hyp.last] = c + 1
            kept.append(h)
            if isLit { literal += 1 }
        }
        beam = kept
    }

    private func path(_ hi: Int32, pool: [Hyp], lattice: Lattice) -> [WordEdge] {
        var out: [WordEdge] = []
        var i = hi
        while i > 0 {
            let h = pool[Int(i)]
            out.append(lattice.edgesFrom[Int(h.edgePos)][Int(h.edgeIdx)])
            i = h.back
        }
        return out.reversed()
    }

    func makeCandidate(path: [WordEdge], lattice: Lattice, context: WordID, end: Int, n: Int) -> Candidate {
        var f = Features()
        var prev = context
        var text = ""
        var segs: [Segment] = []
        segs.reserveCapacity(path.count)
        for e in path {
            f.lm += lm.logP(e.word, given: prev)
            f.reading += e.reading
            f.typoCost += e.typoCost
            f.typos += Float(e.typos)
            f.shapeCost += e.shapeCost
            f.words += 1
            prev = e.word
            text += lexicon.surface(e.word)
            segs.append(Segment(word: e.word, start: Int(e.start), end: Int(e.end), syls: Array(lattice.syls(e)), typoCost: e.typoCost, shapeCost: e.shapeCost, typos: Int(e.typos), edits: lattice.edits(e)))
        }
        f.ctx = ctx
        var c = Candidate(text: text, segments: segs, end: end, features: f, isFull: end == n)
        c.score = FusionWeights.fast.score(f)
        return c
    }

    /// Make sure plausible alternatives on the same spans are present (e.g. 智能 vs 只能), so that context
    /// can decide between them even when the n-gram strongly prefers one.
    private func addVariants(_ full: inout [Candidate], seen: inout [String: Int], lattice: Lattice, context: WordID, end: Int, n: Int, weights: FusionWeights, bonus: PathBonus?) {
        let base = Array(full.prefix(options.variantPaths))
        for c in base {
            var path: [WordEdge] = []
            for s in c.segments {
                guard let e = lattice.edgesFrom[s.start].first(where: { Int($0.end) == s.end && $0.word == s.word }) else { path = []; break }
                path.append(e)
            }
            if path.isEmpty { continue }
            for (i, seg) in path.enumerated() {
                let prev = i == 0 ? context : path[i - 1].word
                let alts = lattice.edgesFrom[Int(seg.start)].filter { $0.end == seg.end && $0.word != seg.word }
                if alts.isEmpty { continue }
                let scores = alts.map { edgeScore($0, prev: prev, weights: weights, bonus: bonus) }
                let order = alts.indices.sorted { scores[$0] > scores[$1] }
                for k in order.prefix(options.variantsPerSegment) {
                    var p = path
                    p[i] = alts[k]
                    let v = makeCandidate(path: p, lattice: lattice, context: context, end: end, n: n)
                    if seen[v.text] == nil {
                        seen[v.text] = full.count
                        full.append(v)
                    }
                }
            }
        }
    }
}
