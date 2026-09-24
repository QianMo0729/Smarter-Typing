import Foundation

/// A chunk of text the simulated user wants to type.
public struct SimExample: Codable {
    public var src: String
    public var context: String
    public var target: String
    public var keys: String
    public var clean_keys: String
    public var syllables: [String]?
    public var typo: TypoInfo?
    public struct TypoInfo: Codable { public var kind: String; public var pos: Int }
}

public func loadExamples(_ path: String) -> [SimExample] {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
    let dec = JSONDecoder()
    return text.split(separator: "\n").compactMap { try? dec.decode(SimExample.self, from: Data($0.utf8)) }
}

/// Simulates a person using the IME: types chunks, accepts / picks / corrects candidates the way people do,
/// sometimes rewrites text for reasons unrelated to the IME. All learning goes through `Learner`, i.e. the
/// exact code path of the real input method.
public final class UserSimulator {
    public let engine: Engine
    public let nn: NeuralScorer?
    public let learner: Learner?
    public var clock: Double = 1_700_000_000
    var rng: SplitMix

    /// Probability that a user who sees the wanted text in the list picks it directly (otherwise they
    /// accept the first candidate by reflex, notice, delete it and type again).
    public var pPickDirectly = 0.7
    /// Probability of deleting a correct commit and writing something else (change of mind).
    public var pRewrite = 0.03
    /// Personal typing habit applied to the intended keys (e.g. swapping "uo" to "ou").
    public var habit: ((String, inout SplitMix) -> String)?

    public struct Stats {
        public var n = 0
        public var firstChoice = 0          // first candidate was what the user wanted
        public var typoN = 0, typoFirst = 0 // inputs containing a slip
        public var cleanN = 0, cleanFirst = 0
        public var falseCorrections = 0     // clean input turned into something else by a correction
        public var explicitPicks = 0, retypes = 0, assembled = 0, rewrites = 0
        public var accuracy: Double { n == 0 ? 0 : Double(firstChoice) / Double(n) }
        public init() {}
    }
    public var stats = Stats()

    public init(engine: Engine, nn: NeuralScorer?, learner: Learner?, seed: UInt64 = 7) {
        self.engine = engine
        self.nn = nn
        self.learner = learner
        self.rng = SplitMix(seed: seed)
        engine.memory.clock = { [unowned self] in self.clock }
    }

    public func analyze(keys: String, context: String) -> Analysis {
        var a = engine.analyze(raw: keys, context: context)
        if let nn, !a.full.isEmpty {
            nn.setContext(context)
            engine.applyNeural(&a, scores: nn.score(a.options))
        }
        return a
    }

    func record(_ c: Candidate, _ a: Analysis, keys: String, context: String, rank: Int) -> CommitRecord {
        var charSyls: [SylID?] = [], words: [WordID] = [], wordSyls: [[SylID]] = []
        for s in c.segments {
            words.append(s.word); wordSyls.append(s.syls)
            let nChars = engine.lexicon.charCount(s.word)
            if s.syls.count == nChars { charSyls += s.syls.map { Optional($0) } } else { charSyls += Array(repeating: nil, count: nChars) }
        }
        return CommitRecord(time: clock, app: "sim", context: context, raw: keys, text: c.text, charSyls: charSyls, words: words,
                            wordSyls: wordSyls, chosenRank: rank, analysis: a, topProb: a.full.first?.prob ?? 0,
                            usedTypoCorrection: c.typoCount > 0)
    }

    /// Record for text the user had to assemble piece by piece (not offered as a whole).
    func assembledRecord(_ ex: SimExample, keys: String) -> CommitRecord {
        let words = engine.contextWords.tailWords(ex.target, maxWords: 64)
        var syls: [SylID?] = (ex.syllables ?? []).map { engine.lexicon.syllables.idOf[$0] }
        if syls.count != ex.target.count { syls = Array(repeating: nil, count: ex.target.count) }
        var wordSyls: [[SylID]] = []
        var off = 0
        for w in words {
            let n = engine.lexicon.charCount(w)
            let lo = min(off, syls.count), hi = min(syls.count, off + n)
            let s = syls[lo..<hi].compactMap { $0 }
            wordSyls.append(s.count == n ? s : [])
            off += n
        }
        return CommitRecord(time: clock, app: "sim", context: ex.context, raw: keys, text: ex.target, charSyls: syls, words: words,
                            wordSyls: wordSyls, chosenRank: -1, analysis: nil, topProb: 0, usedTypoCorrection: false)
    }

    func advance(_ lo: Double, _ hi: Double) { clock += lo + (hi - lo) * rng.nextDouble() }

    func deleteAll(_ text: String) {
        for _ in 0..<text.count { advance(0.08, 0.15); learner?.didDeleteBackward(time: clock) }
    }

    /// Type one chunk. Returns whether the first candidate was right.
    @discardableResult
    public func type(_ ex: SimExample) -> Bool {
        var keys = ex.keys
        var slipped = ex.typo != nil
        if let habit {
            let k2 = habit(ex.clean_keys, &rng)
            if k2 != ex.clean_keys { keys = k2; slipped = true }
        }
        advance(1.0, 3.0)
        let a = analyze(keys: keys, context: ex.context)
        stats.n += 1
        let list = engine.displayList(a)
        let first = a.full.first?.text ?? list.first?.text
        let ok = first == ex.target
        if ok { stats.firstChoice += 1 }
        if slipped { stats.typoN += 1; if ok { stats.typoFirst += 1 } } else {
            stats.cleanN += 1; if ok { stats.cleanFirst += 1 }
            if !ok, let f = a.full.first, f.typoCount > 0 { stats.falseCorrections += 1 }
        }
        guard let learner else { return ok }
        if ok, let c = a.full.first {
            learner.didCommit(record(c, a, keys: keys, context: ex.context, rank: 0))
            if rng.nextDouble() < pRewrite {
                // change of mind: delete the (correct) text and write something unrelated
                stats.rewrites += 1
                advance(1, 4)
                deleteAll(c.text)
                let other = ["不过", "算了", "其实", "我觉得", "换个说法"][Int(rng.next() % 5)]
                let a2 = analyze(keys: "x", context: ex.context)
                if let c2 = a2.prefix.first ?? a2.full.first {
                    var r = record(c2, a2, keys: "x", context: ex.context, rank: 0)
                    r.text = other; r.charSyls = Array(repeating: nil, count: other.count)
                    learner.didCommit(r)
                }
            }
            return true
        }
        if let k = list.firstIndex(where: { $0.text == ex.target }), !slipped || rng.nextDouble() < 0.5, rng.nextDouble() < pPickDirectly {
            stats.explicitPicks += 1
            learner.didCommit(record(list[k], a, keys: keys, context: ex.context, rank: k))
            return false
        }
        // accept the first candidate by reflex, notice, delete, retype (clean keys this time)
        if let c = a.full.first ?? list.first {
            learner.didCommit(record(c, a, keys: keys, context: ex.context, rank: 0))
            advance(0.4, 2.0)
            deleteAll(c.text)
            stats.retypes += 1
            advance(0.3, 1.0)
            let a2 = analyze(keys: ex.clean_keys, context: ex.context)
            let list2 = engine.displayList(a2)
            if let k = list2.firstIndex(where: { $0.text == ex.target }) {
                learner.didCommit(record(list2[k], a2, keys: ex.clean_keys, context: ex.context, rank: k))
            } else {
                stats.assembled += 1
                learner.didCommit(assembledRecord(ex, keys: ex.clean_keys))
            }
        } else {
            stats.assembled += 1
            learner.didCommit(assembledRecord(ex, keys: ex.clean_keys))
        }
        return false
    }

    /// Accuracy of the current state on a test set without learning from it.
    public func evaluate(_ set: [SimExample], habit: Bool = false) -> Stats {
        var s = Stats()
        var r = SplitMix(seed: 99)
        for ex in set {
            var keys = ex.keys
            var slipped = ex.typo != nil
            if habit, let h = self.habit {
                let k2 = h(ex.clean_keys, &r)
                if k2 != ex.clean_keys { keys = k2; slipped = true }
            }
            let a = analyze(keys: keys, context: ex.context)
            let ok = a.full.first?.text == ex.target
            s.n += 1
            if ok { s.firstChoice += 1 }
            if slipped { s.typoN += 1; if ok { s.typoFirst += 1 } } else {
                s.cleanN += 1; if ok { s.cleanFirst += 1 }
                if !ok, let f = a.full.first, f.typoCount > 0 { s.falseCorrections += 1 }
            }
        }
        return s
    }
}

/// Small deterministic RNG.
public struct SplitMix {
    var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    public mutating func nextDouble() -> Double { Double(next() >> 11) / Double(1 << 53) }
}
