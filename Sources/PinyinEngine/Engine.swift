import Foundation

/// Everything the engine knows about the current composition.
public struct Analysis {
    public var raw: String
    public var context: String
    public var app: String?
    public var contextWord: WordID
    public var ctxVec: ContextVector
    public var graph: SyllableGraph
    public var full: [Candidate]
    public var prefix: [Candidate]
    public var covered: Int
    public var n: Int
    public var nnApplied = false
    public var fastMs: Double = 0

    /// The closed decision the small model is asked about: which of these strings did the user mean?
    public var options: [String] { full.map { $0.text } }
    public var best: Candidate? { full.first ?? prefix.first }
    public var confidence: Float { full.first?.prob ?? 0 }

    /// Raw input keys not covered by `full` (e.g. an unparsable tail).
    public var uncoveredRaw: String {
        let bytes = Array(raw.utf8)
        let cut = covered < graph.rawIndex.count ? graph.rawIndex[covered] : bytes.count
        return String(decoding: bytes[min(cut, bytes.count)...], as: UTF8.self)
    }
    /// The typed pinyin split into syllables the way the best reading understands it ("zhi neng shu ru fa"),
    /// for display in the text field. Falls back to the raw keys where it cannot be split cleanly.
    public func segmentedPinyin(table: SyllableTable) -> String {
        guard let best = full.first ?? prefix.first else { return raw }
        let keys = graph.keys
        var pieces: [String] = []
        var covered = 0
        for seg in best.segments {
            guard seg.start == covered, seg.end <= keys.count else { break }
            var pos = seg.start
            for s in seg.syls {
                let sp = Array(table.spelling(s).utf8)
                let end = min(seg.end, pos + sp.count)
                if Array(keys[pos..<end]) == sp { pieces.append(String(decoding: sp, as: UTF8.self)); pos = end }
                else { break }
            }
            if pos < seg.end { pieces.append(String(decoding: keys[pos..<seg.end], as: UTF8.self)) }
            covered = seg.end
        }
        if covered < keys.count { pieces.append(String(decoding: keys[covered...], as: UTF8.self)) }
        return pieces.joined(separator: " ")
    }

    /// Raw keys remaining after committing `c` (for partial commits of prefix candidates).
    public func remainingRaw(after c: Candidate) -> String {
        let bytes = Array(raw.utf8)
        let cut = c.end < graph.rawIndex.count ? graph.rawIndex[c.end] : bytes.count
        var rest = Array(bytes[min(cut, bytes.count)...])
        while rest.first == 39 { rest.removeFirst() }
        return String(decoding: rest, as: UTF8.self)
    }
}

public struct EngineResources {
    public var dir: URL
    public var modelPath: String?
    public init(dir: URL, modelPath: String? = nil) { self.dir = dir; self.modelPath = modelPath }
}

/// Weights file written by tools/fit_weights.py
struct WeightsFile: Codable {
    var fast: [Float]
    var full: [Float]
    var typoScale: Float?
    var typoGuard: Float?
    var partialComplete: Float?
    var partialIncomplete: Float?
    var abbrev: Float?
}

public final class Engine: @unchecked Sendable {
    public let lexicon: Lexicon
    public let lm: NGramModel
    public let typo: TypoModel
    public let decoder: Decoder
    public let contextWords: ContextWords
    public let userLM: UserLM
    public let memory: ContextMemory
    public var weightsFast = FusionWeights.fast
    public var weightsFull = FusionWeights.full
    /// A typo-corrected reading must beat the best literal reading by this many nats to become first.
    public var typoGuard: Float = 0.5
    public var graphOptions = GraphOptions()
    public var latticeOptions = LatticeOptions()
    public var usePersonal = true

    let resourceDir: URL

    /// Forget everything learned about the user (personal LM, memory, typing habits, adapted weights).
    public func resetPersonal() {
        userLM.restore(UserLM.State(uni: [:], bi: [:], prevTotal: [:], total: 0))
        memory.table = [:]
        typo.reset()
        weightsFast = .fast; weightsFull = .full; typoGuard = 0.5
        loadWeights(resourceDir.appendingPathComponent("weights.json"))
    }

    public init(resources: EngineResources) throws {
        resourceDir = resources.dir
        lexicon = try Lexicon(url: resources.dir.appendingPathComponent("lex.bin"))
        let lmURL = resources.dir.appendingPathComponent("lm.bin")
        lm = FileManager.default.fileExists(atPath: lmURL.path) ? try NGramModel(url: lmURL) : NGramModel(lexicon: lexicon)
        lm.oovPrior = lexicon.prior
        typo = TypoModel()
        decoder = Decoder(lexicon: lexicon, lm: lm)
        contextWords = ContextWords(lexicon: lexicon)
        userLM = UserLM(base: lm)
        memory = ContextMemory()
        let wURL = resources.dir.appendingPathComponent("weights.json")
        if let d = try? Data(contentsOf: wURL), let w = try? JSONDecoder().decode(WeightsFile.self, from: d) {
            apply(w)
        }
    }

    func apply(_ w: WeightsFile) {
        if w.fast.count == Features.count { weightsFast = FusionWeights(w.fast) }
        if w.full.count == Features.count { weightsFull = FusionWeights(w.full) }
        if let s = w.typoScale { typo.scale = s }
        if let g = w.typoGuard { typoGuard = g }
        if let p = w.partialComplete { graphOptions.partialCostComplete = p }
        if let p = w.partialIncomplete { graphOptions.partialCostIncomplete = p }
        if let a = w.abbrev { graphOptions.abbrevCost = a }
    }

    public func loadWeights(_ url: URL) {
        if let d = try? Data(contentsOf: url), let w = try? JSONDecoder().decode(WeightsFile.self, from: d) { apply(w) }
    }

    /// Fast, synchronous analysis (no neural model): typo-tolerant parsing, lattice, n-gram search,
    /// personal features. Candidates are ordered by the fast weights.
    public func analyze(raw: String, context: String, app: String? = nil) -> Analysis {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let graph = SyllableGraph(raw: Array(raw.lowercased().utf8), table: lexicon.syllables, typo: typo, options: graphOptions)
        let lattice = Lattice(graph: graph, lexicon: lexicon, lm: lm, options: latticeOptions)
        let cword = contextWords.lastWord(context)
        let bonus: PathBonus? = usePersonal && userLM.total > 0 ? userLM : nil
        let r = decoder.decode(lattice: lattice, context: cword, weights: weightsFast, bonus: bonus, ctxStrength: Engine.contextStrength(context))
        var a = Analysis(raw: raw, context: context, app: app, contextWord: cword, ctxVec: ContextVector(text: context, app: app),
                         graph: graph, full: r.full, prefix: r.prefix, covered: r.covered, n: r.n)
        addPersonalFeatures(&a)
        finalize(&a, weights: weightsFast)
        a.fastMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        return a
    }

    /// 0 with no preceding text, 1 with 30 or more characters.
    public static func contextStrength(_ context: String) -> Float { min(1, Float(context.count) / 30) }

    func addPersonalFeatures(_ a: inout Analysis) {
        guard usePersonal else { return }
        let hasLM = userLM.total > 0, hasMem = !memory.isEmpty
        if !hasLM && !hasMem { return }
        func fill(_ c: inout Candidate) {
            var prev = a.contextWord
            var p: Float = 0, m: Float = 0
            for s in c.segments {
                if hasLM { p += userLM.bonus(prev: prev, word: s.word) }
                prev = s.word
            }
            if hasMem { m = memory.vote(charSyls: c.syllables, text: c.text, ctx: a.ctxVec) }
            c.features.personal = p
            c.features.memory = m
        }
        for i in a.full.indices { fill(&a.full[i]) }
        for i in a.prefix.indices { fill(&a.prefix[i]) }
    }

    /// Fold neural log-probabilities (aligned with `a.full`) into the features and re-rank.
    public func applyNeural(_ a: inout Analysis, scores: [Float]) {
        for i in a.full.indices where i < scores.count && scores[i] > -1e8 {
            a.full[i].features.nn = scores[i]
            a.full[i].features.nnMissing = 0
        }
        a.nnApplied = true
        finalize(&a, weights: weightsFull)
    }

    /// Score, normalise into probabilities (the decision distribution) and apply the conservative typo rule.
    public func finalize(_ a: inout Analysis, weights: FusionWeights) {
        if a.full.isEmpty { return }
        for i in a.full.indices { a.full[i].score = weights.score(a.full[i].features) }
        a.full.sort { $0.score > $1.score }
        let mx = a.full[0].score
        var z: Float = 0
        for c in a.full { z += exp(c.score - mx) }
        for i in a.full.indices { a.full[i].prob = exp(a.full[i].score - mx) / z }
        // Do not let a correction displace what the user literally typed unless it is clearly better.
        // Slips this user makes habitually need less margin.
        if a.full[0].typoCount > 0, let j = a.full.firstIndex(where: { $0.typoCount == 0 }) {
            var fam: Float = 1
            for s in a.full[0].segments { for e in s.edits { if let k = e.edit { fam = min(fam, typo.familiarity(k, typed: e.typed, intended: e.intended)) } } }
            let margin = typoGuard * (1 - 0.8 * fam)
            if a.full[0].score - a.full[j].score < margin {
                let lit = a.full.remove(at: j)
                a.full.insert(lit, at: 0)
            }
        }
        for i in a.prefix.indices { a.prefix[i].score = weightsFast.score(a.prefix[i].features) }
        // prefix candidates: longer spans first, then by score (like a regular IME's word list)
        a.prefix.sort { x, y in
            let lx = x.end, ly = y.end
            if lx != ly { return lx > ly }
            return x.score > y.score
        }
    }

    /// Candidate list for display: the best whole-input conversions first, then every word that can start
    /// the input (longest first, each group ordered by how well it fits the context), like a regular IME.
    public func displayList(_ a: Analysis, maxFull: Int = 3, limit: Int = 1500) -> [Candidate] {
        var out: [Candidate] = []
        var seen = Set<String>()
        for (i, c) in a.full.enumerated() {
            if i >= maxFull { break }
            if i > 0 && c.prob < 0.03 { break }
            out.append(c); seen.insert(c.text)
        }
        // remaining whole-input candidates and first-word candidates from the search
        var rest: [Candidate] = []
        for c in a.full.dropFirst(out.count) where c.typoCount == 0 && !seen.contains(c.text) { rest.append(c) }
        for c in a.prefix where c.typoCount == 0 && !seen.contains(c.text) { rest.append(c) }
        // plus every other word the lexicon has for the start of the input
        let have = Set(rest.map { $0.text }).union(seen)
        for (w, end, syls, shape) in Lattice.allFirstWords(graph: a.graph, lexicon: lexicon) {
            let text = lexicon.surface(w)
            if have.contains(text) { continue }
            var f = Features()
            f.lm = lm.logP(w, given: a.contextWord)
            f.shapeCost = shape
            f.words = 1
            f.ctx = Engine.contextStrength(a.context)
            var c = Candidate(text: text, segments: [Segment(word: w, start: 0, end: end, syls: syls, typoCost: 0, shapeCost: shape, typos: 0, edits: [])],
                              end: end, features: f, isFull: end == a.n)
            c.score = weightsFast.score(f)
            rest.append(c)
        }
        rest.sort { x, y in
            if x.end != y.end { return x.end > y.end }
            return x.score > y.score
        }
        var seenRest = seen
        for c in rest where !seenRest.contains(c.text) {
            out.append(c); seenRest.insert(c.text)
            if out.count >= limit { break }
        }
        return out
    }
}
