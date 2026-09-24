import Foundation

/// What the IME reports when a composition is committed.
public struct CommitRecord {
    public var id: Int = 0
    public var time: Double
    public var app: String?
    public var context: String            // text before the composition
    public var raw: String                // keys typed for the whole composition
    public var text: String               // committed text
    public var charSyls: [SylID?]         // syllable of every committed character (nil if unknown)
    public var words: [WordID]            // segmentation of the committed text
    public var wordSyls: [[SylID]]        // syllables per word
    public var chosenRank: Int            // 0 = first candidate (space); >0 explicitly chosen; -1 assembled from parts
    public var analysis: Analysis?        // analysis of `raw` at commit time
    public var topProb: Float             // probability of the first candidate when committing
    public var usedTypoCorrection: Bool   // committed candidate assumed a keyboard slip

    public init(time: Double, app: String?, context: String, raw: String, text: String, charSyls: [SylID?], words: [WordID], wordSyls: [[SylID]], chosenRank: Int, analysis: Analysis?, topProb: Float, usedTypoCorrection: Bool) {
        self.time = time; self.app = app; self.context = context; self.raw = raw; self.text = text
        self.charSyls = charSyls; self.words = words; self.wordSyls = wordSyls; self.chosenRank = chosenRank
        self.analysis = analysis; self.topProb = topProb; self.usedTypoCorrection = usedTypoCorrection
    }
}

/// Interpretation of what the user did, for logs / evaluation.
public enum LearningEventKind: String, Codable {
    case explicitChoice      // picked a non-first candidate
    case reselection         // deleted a commit and re-entered the same pinyin with other text
    case typoCorrection      // deleted a commit and re-typed nearly the same keys
    case wrongCorrection     // an automatic typo correction was undone
    case rewrite             // deleted and wrote something else: no judgement about the old text
    case retraction          // deleted without replacement
    case kept                // survived the correction window
}

public struct LearningEvent: Codable {
    public var kind: LearningEventKind
    public var old: String
    public var new: String
    public var weight: Float
    public var time: Double
}

/// Turns ordinary typing behaviour into learning signals, without any labelling by the user.
///
/// It keeps a short "run" of text the IME committed at the cursor. Backspaces remove characters from the
/// run; commits made after deletions form a replacement. When the replacement is complete the episode is
/// classified (same pinyin → word choice error, nearby keys → typo, unrelated → rewrite) and only the
/// confident interpretations update the personal models. Text that survives is a weak positive signal.
public final class Learner {
    struct RunChar { var ch: Character; var syl: SylID?; var record: Int }
    struct Episode {
        var deleted: [RunChar] = []          // original characters removed (in text order)
        var replacement: [RunChar] = []
        var start: Double
        var lastActivity: Double
    }

    public let engine: Engine
    public var keepWindow: Double = 25          // seconds before kept text is credited
    public var episodeTimeout: Double = 45
    public var learningRate: Float = 0.02
    public var minGuard: Float = 0.3
    public var maxGuard: Float = 2.5
    public var onEvent: ((LearningEvent) -> Void)?
    public private(set) var events: [LearningEvent] = []
    public var recordEvents = true

    private var records: [Int: CommitRecord] = [:]
    private var credited = Set<Int>()
    private var deletedRecords = Set<Int>()
    private var run: [RunChar] = []
    private var episode: Episode?
    private var nextId = 1
    private let baseFull: FusionWeights
    private let baseFast: FusionWeights

    public init(engine: Engine) {
        self.engine = engine
        baseFull = engine.weightsFull
        baseFast = engine.weightsFast
    }

    // MARK: inputs from the IME

    public func didCommit(_ r0: CommitRecord) {
        var r = r0
        r.id = nextId; nextId += 1
        tick(r.time)
        records[r.id] = r
        engine.typo.observeKeystrokes(r.raw.count)
        let chars = Array(r.text)
        let newChars = chars.enumerated().map { RunChar(ch: $0.element, syl: $0.offset < r.charSyls.count ? r.charSyls[$0.offset] : nil, record: r.id) }
        run.append(contentsOf: newChars)
        if run.count > 400 { run.removeFirst(run.count - 400) }
        if r.chosenRank > 0 { learnExplicitChoice(r) }
        if var ep = episode {
            ep.replacement.append(contentsOf: newChars)
            ep.lastActivity = r.time
            episode = ep
            if ep.replacement.count >= ep.deleted.count { resolve(now: r.time) }
        }
    }

    /// One backspace at the end of the run.
    public func didDeleteBackward(time: Double) {
        tick(time)
        guard let last = run.popLast() else { return }
        if episode == nil { episode = Episode(start: time, lastActivity: time) }
        if let i = episode!.replacement.lastIndex(where: { $0.record == last.record }), i == episode!.replacement.count - 1 {
            episode!.replacement.removeLast()        // fixing the replacement itself
        } else {
            episode!.deleted.insert(last, at: 0)
            if !run.contains(where: { $0.record == last.record }) { deletedRecords.insert(last.record) }
        }
        episode!.lastActivity = time
    }

    /// The composition replaced a selection that was exactly the tail of the run (select + retype).
    public func didReplaceSelection(_ selected: String, time: Double) {
        let n = selected.count
        guard n > 0, run.count >= n, String(run.suffix(n).map { $0.ch }) == selected else { didBreak(time: time); return }
        for _ in 0..<n { didDeleteBackward(time: time) }
    }

    /// Cursor moved elsewhere, focus changed, or anything we cannot follow.
    public func didBreak(time: Double) {
        resolve(now: time)
        creditAll(now: time)
        run.removeAll()
    }

    /// Periodic housekeeping: credit kept text, close stale episodes.
    public func tick(_ now: Double) {
        if let ep = episode, now - ep.lastActivity > episodeTimeout { resolve(now: now) }
        for (id, r) in records where !credited.contains(id) && now - r.time > keepWindow && episode == nil {
            credit(r)
        }
        if records.count > 200 {
            let old = records.keys.sorted().prefix(records.count - 200)
            for k in old { records[k] = nil; credited.remove(k); deletedRecords.remove(k) }
        }
    }

    public func flush(now: Double) { didBreak(time: now) }

    // MARK: interpretation

    private func emit(_ k: LearningEventKind, _ old: String, _ new: String, _ w: Float, _ t: Double) {
        let e = LearningEvent(kind: k, old: old, new: new, weight: w, time: t)
        if recordEvents { events.append(e) }
        onEvent?(e)
    }

    private func creditAll(now: Double) {
        for (id, r) in records where !credited.contains(id) { credit(r) }
    }

    /// Weak positive evidence for text that was kept.
    private func credit(_ r: CommitRecord) {
        credited.insert(r.id)
        if deletedRecords.contains(r.id) { return }
        let w: Float = r.chosenRank != 0 ? 1.0 : 0.3 + 0.5 * (1 - min(1, r.topProb))
        let prev = engine.contextWords.lastWord(r.context)
        engine.userLM.add(words: r.words, prev: prev, weight: w)
        // remember the decision in context where it was contested by another reading of the same pinyin
        if let a = r.analysis {
            let ctx = ContextVector(text: r.context, app: r.app)
            let rivals = a.full.prefix(6).filter { $0.text != r.text && $0.prob > 0.02 }
            for (span, _) in contested(r, rivals: rivals) {
                addSpan(r, span, ctx: ctx, weight: 0.5 * w)
            }
        }
        if r.usedTypoCorrection, let a = r.analysis, let c = a.full.first(where: { $0.text == r.text }) {
            for s in c.segments { for e in s.edits { if let k = e.edit { engine.typo.observe(kind: k, typed: e.typed, intended: e.intended, weight: 0.5) } } }
        }
        emit(.kept, "", r.text, w, r.time)
    }

    /// Word boundaries (character offsets) of a record / candidate.
    private func boundaries(_ words: [WordID]) -> Set<Int> {
        var b: Set<Int> = [0]
        var off = 0
        for w in words { off += engine.lexicon.charCount(w); b.insert(off) }
        return b
    }

    /// Character spans where the committed text differs from rival readings with identical syllables,
    /// widened to word boundaries of both readings. Returns (span, rival text of that span).
    func contested(_ r: CommitRecord, rivals: [Candidate]) -> [(Range<Int>, String)] {
        let chars = Array(r.text)
        guard r.charSyls.count == chars.count, !r.charSyls.contains(where: { $0 == nil }) else { return [] }
        let syls = r.charSyls.map { $0! }
        let mine = boundaries(r.words)
        var out: [(Range<Int>, String)] = []
        var seen = Set<Range<Int>>()
        for c in rivals {
            guard c.syllables == syls else { continue }
            let rc = Array(c.text)
            guard rc.count == chars.count else { continue }
            let theirs = boundaries(c.segments.map { $0.word })
            var i = 0
            while i < chars.count {
                if chars[i] == rc[i] { i += 1; continue }
                var j = i
                while j < chars.count && chars[j] != rc[j] { j += 1 }
                // widen to boundaries that both segmentations respect
                var lo = i, hi = j
                var changed = true
                while changed {
                    changed = false
                    while lo > 0 && !(mine.contains(lo) && theirs.contains(lo)) { lo -= 1; changed = true }
                    while hi < chars.count && !(mine.contains(hi) && theirs.contains(hi)) { hi += 1; changed = true }
                }
                if hi - lo <= engine.memory.maxSpan && !seen.contains(lo..<hi) {
                    seen.insert(lo..<hi)
                    out.append((lo..<hi, String(rc[lo..<hi])))
                }
                i = j
            }
        }
        return out
    }

    private func addSpan(_ r: CommitRecord, _ span: Range<Int>, ctx: ContextVector, weight: Float, text: String? = nil) {
        let syls = r.charSyls[span].compactMap { $0 }
        guard syls.count == span.count else { return }
        let chars = Array(r.text)
        engine.memory.add(syls: syls, text: text ?? String(chars[span]), ctx: ctx, weight: weight)
    }

    private func learnExplicitChoice(_ r: CommitRecord) {
        guard let a = r.analysis, let top = a.full.first, top.text != r.text else { return }
        let ctx = ContextVector(text: r.context, app: r.app)
        for (span, rivalText) in contested(r, rivals: [top]) {
            addSpan(r, span, ctx: ctx, weight: 1.0)
            addSpan(r, span, ctx: ctx, weight: -0.7, text: rivalText)
        }
        sgd(a, correct: r.text, weight: 1.0)
        emit(.explicitChoice, top.text, r.text, 1.0, r.time)
    }

    private func resolve(now: Double) {
        guard let ep = episode else { return }
        episode = nil
        if ep.deleted.isEmpty { return }
        let oldText = String(ep.deleted.map { $0.ch })
        let newText = String(ep.replacement.map { $0.ch })
        let oldRecs = Array(Set(ep.deleted.map { $0.record })).sorted().compactMap { records[$0] }
        let newRecs = Array(Set(ep.replacement.map { $0.record })).sorted().compactMap { records[$0] }
        guard let firstOld = oldRecs.first else { return }
        let age = ep.start - (oldRecs.last?.time ?? ep.start)
        let timeW: Float = age < 15 ? 1.0 : (age < 60 ? 0.6 : 0.25)
        // context before the deleted text
        let ctxText = contextBefore(ep.deleted.first!, in: firstOld)
        let ctx = ContextVector(text: ctxText, app: firstOld.app)

        if ep.replacement.isEmpty {
            // Deleted without replacement: the user took something back; lower the kept credit only.
            for r in oldRecs where credited.contains(r.id) {
                engine.userLM.add(words: r.words, prev: engine.contextWords.lastWord(r.context), weight: -0.2)
            }
            emit(.retraction, oldText, "", 0, now)
            return
        }
        let oldSyl = ep.deleted.map { $0.syl }
        let newSyl = ep.replacement.map { $0.syl }
        let n = min(oldSyl.count, newSyl.count)
        let sameSyl = oldSyl.count == newSyl.count && zip(oldSyl, newSyl).allSatisfy { $0 != nil && $0 == $1 }
        let oldWhole = oldRecs.allSatisfy { r in ep.deleted.filter { $0.record == r.id }.count == r.text.count }
        let oldRaw = oldRecs.map { $0.raw }.joined(), newRaw = newRecs.map { $0.raw }.joined()
        let keyEdits = oldWhole && !oldRaw.isEmpty && !newRaw.isEmpty ? TypoModel.classify(typed: Array(oldRaw.utf8), intended: Array(newRaw.utf8)) : []

        // "sm" gave 扫描, deleted, typed "shenme" → 什么: the old keys abbreviate the new pinyin, so this corrects
        // the choice for those keys (not a change of mind)
        let abbreviated = !sameSyl && oldWhole && oldRecs.count == 1 && newSyl.allSatisfy { $0 != nil }
            && keysAbbreviate(oldRaw, newSyl.compactMap { $0 }) && oldText != newText
        if abbreviated {
            let w = timeW
            let sylsNew = newSyl.compactMap { $0 }
            engine.memory.add(syls: sylsNew, text: newText, ctx: ctx, weight: 1.0 * w)
            let oldSyls = oldSyl.compactMap { $0 }
            if oldSyls.count == oldSyl.count { engine.memory.add(syls: oldSyls, text: oldText, ctx: ctx, weight: -0.8 * w) }
            if let a = oldRecs[0].analysis { sgd(a, correct: newText, weight: w) }
            for r in oldRecs where credited.contains(r.id) { engine.userLM.add(words: r.words, prev: engine.contextWords.lastWord(r.context), weight: -0.3) }
            emit(.reselection, oldText + "(" + oldRaw + ")", newText, w, now)
        } else if sameSyl && oldText != newText {
            // Same pinyin, different characters: the choice was wrong in this context.
            let w = timeW
            let syls = oldSyl.compactMap { $0 }
            if syls.count == oldSyl.count {
                // store the differing part, widened to the words of the new text when we know them
                let oc = Array(oldText), nc = Array(newText)
                var lo = 0, hi = oc.count
                while lo < hi && oc[lo] == nc[lo] { lo += 1 }
                while hi > lo && oc[hi - 1] == nc[hi - 1] { hi -= 1 }
                if newRecs.count == 1, oldRecs.count == 1, newRecs[0].text == newText {
                    let b = boundaries(newRecs[0].words)
                    while lo > 0 && !b.contains(lo) { lo -= 1 }
                    while hi < nc.count && !b.contains(hi) { hi += 1 }
                }
                if hi - lo < 2 { lo = 0; hi = min(nc.count, max(hi, 2)) }
                if hi - lo <= engine.memory.maxSpan {
                    engine.memory.add(syls: syls[lo..<hi], text: String(oc[lo..<hi]), ctx: ctx, weight: -1.0 * w)
                    engine.memory.add(syls: syls[lo..<hi], text: String(nc[lo..<hi]), ctx: ctx, weight: 1.0 * w)
                }
            }
            for r in oldRecs {
                if credited.contains(r.id) { engine.userLM.add(words: r.words, prev: engine.contextWords.lastWord(r.context), weight: -0.3) }
                if let a = r.analysis, oldRecs.count == 1 { sgd(a, correct: newText, weight: w) }
            }
            emit(.reselection, oldText, newText, w, now)
        } else if !keyEdits.isEmpty && keyEdits.count <= 2 && oldText != newText {
            // Nearly the same keys: the user mistyped. Learn the slip and the intended reading.
            let w = timeW
            for (k, t, i) in keyEdits { engine.typo.observe(kind: k, typed: t, intended: i, weight: Double(w)) }
            if oldRecs.count == 1, let a = oldRecs[0].analysis {
                sgd(a, correct: newText, weight: w)
                // the intended reading was offered but held back by the conservative rule: relax it a little
                if let top = a.full.first, top.typoCount == 0, let c = a.full.first(where: { $0.text == newText }), c.typoCount > 0,
                   c.score > top.score - 2 {
                    engine.typoGuard = max(minGuard, engine.typoGuard - 0.1)
                }
            }
            emit(.typoCorrection, oldText + "(" + oldRaw + ")", newText + "(" + newRaw + ")", w, now)
        } else if oldWhole, oldRecs.count == 1, oldRecs[0].usedTypoCorrection, newRaw == oldRaw || newText.hasPrefix(literalReading(oldRecs[0]) ?? "\u{0}") {
            // An automatic correction was undone: be more conservative.
            engine.typoGuard = min(maxGuard, engine.typoGuard + 0.1)
            if let a = oldRecs[0].analysis { sgd(a, correct: newText, weight: timeW) }
            emit(.wrongCorrection, oldText, newText, timeW, now)
        } else {
            // Different content: a rewrite / change of mind. No judgement on the old text, just stop crediting it.
            var partial = 0
            for i in 0..<n where oldSyl[i] != nil && oldSyl[i] == newSyl[i] { partial += 1 }
            for r in oldRecs where credited.contains(r.id) {
                engine.userLM.add(words: r.words, prev: engine.contextWords.lastWord(r.context), weight: -0.15)
            }
            emit(.rewrite, oldText, newText, 0, now)
            _ = partial
        }
    }

    /// True if `keys` spell `syls` with every syllable given at least by its first letter ("sm" ~ shen me,
    /// "zhinen" ~ zhi neng, "zn" ~ zhi neng). Requires at least one abbreviated syllable to matter.
    func keysAbbreviate(_ keys: String, _ syls: [SylID]) -> Bool {
        let k = Array(keys.utf8).filter { $0 != 39 }
        let sp = syls.map { Array(engine.lexicon.syllables.spelling($0).utf8) }
        guard !sp.isEmpty, k.count < sp.reduce(0, { $0 + $1.count }) else { return false }
        // dynamic programming: dp[i][j] = keys[0..<i] covers syllables[0..<j]
        var dp = [[Bool]](repeating: [Bool](repeating: false, count: sp.count + 1), count: k.count + 1)
        dp[0][0] = true
        for i in 0..<k.count {
            for j in 0..<sp.count where dp[i][j] {
                var len = 1
                while len <= sp[j].count && i + len <= k.count && k[i + len - 1] == sp[j][len - 1] {
                    dp[i + len][j + 1] = true
                    len += 1
                }
            }
        }
        return dp[k.count][sp.count]
    }

    private func literalReading(_ r: CommitRecord) -> String? {
        r.analysis?.full.first(where: { $0.typoCount == 0 })?.text
    }

    private func contextBefore(_ c: RunChar, in r: CommitRecord) -> String {
        // everything in the run before `c`, preceded by the record's own context
        var s = r.context
        if let i = run.firstIndex(where: { $0.record == c.record }) {
            s = records[run[0].record]?.context ?? r.context
            s += String(run[..<i].map { $0.ch })
        }
        return s
    }

    /// One conditional-logit gradient step towards `correct` among the analysed candidates.
    /// Only the trust placed in the personal signals (personal LM, context memory) is adapted online;
    /// the base weights stay as fitted offline, so a few odd events cannot derail general accuracy.
    private func sgd(_ a: Analysis, correct: String, weight: Float) {
        guard let j = a.full.firstIndex(where: { $0.text == correct }), a.full.count > 1 else { return }
        let adapt = [7, 8]   // personal, memory
        func step(_ w: inout FusionWeights, base: FusionWeights) {
            let scores = a.full.map { w.score($0.features) }
            let mx = scores.max()!
            let ex = scores.map { exp($0 - mx) }
            let z = ex.reduce(0, +)
            var grad = [Float](repeating: 0, count: Features.count)
            for (i, c) in a.full.enumerated() {
                let p = ex[i] / z
                let v = c.features.vector
                for k in adapt { grad[k] += ((i == j ? 1 : 0) - p) * v[k] }
            }
            for k in adapt {
                let g = max(-2, min(2, grad[k])) - 0.2 * (w.w[k] - base.w[k])
                w.w[k] = max(0.2, min(3, w.w[k] + learningRate * weight * g))
            }
        }
        if a.nnApplied { step(&engine.weightsFull, base: baseFull) }
        step(&engine.weightsFast, base: baseFast)
    }
}
