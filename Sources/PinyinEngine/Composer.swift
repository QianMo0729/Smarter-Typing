import Foundation

/// Timing policy for the "type, then hit space immediately" experience.
public struct TimingPolicy {
    /// After a keystroke, how long the candidate list may wait for the neural ranking before the fast
    /// ranking is shown. Within this window a late result simply replaces the list before anyone can read it.
    public var panelWaitMs: Double = 45
    /// A displayed list older than this is considered "seen": it is never reordered afterwards and a
    /// commit takes exactly what is shown.
    public var seenMs: Double = 160
    /// On commit, how long we may block waiting for the neural ranking of the exact current input.
    public var commitWaitMs: Double = 90
    public init() {}
}

public struct CommitResult {
    public let text: String
    public let record: CommitRecord
    public let waitedMs: Double
    public let usedNeural: Bool
}

/// One composition: raw keys → candidates → commit. Independent of AppKit so the IME, tests and the
/// latency simulator share exactly the same behaviour. Call from one thread (the main thread in the IME).
public final class Composer {
    public let engine: Engine
    public var neural: NeuralService?
    /// Show the pinyin split into syllables ("zhi neng") in the text field.
    public var spacedPinyin = true
    public var policy = TimingPolicy()
    public var context: String = ""
    public var app: String?
    /// Called whenever the displayed candidates change.
    public var onDisplay: (() -> Void)?
    /// Hook to run work on the main thread (tests can make it synchronous).
    public var runOnMain: (@escaping () -> Void) -> Void = { f in DispatchQueue.main.async(execute: f) }

    public private(set) var raw = ""
    /// Edit position inside `raw` (in characters): letters are inserted here and backspace deletes before it.
    /// Candidates always cover the whole of `raw`, so fixing a letter in the middle immediately updates the
    /// smart first choice for the complete input.
    public private(set) var caret = 0
    public private(set) var parts: [(text: String, cand: Candidate, raw: String, rank: Int)] = []
    public private(set) var analysis: Analysis?
    public private(set) var version = 0
    public private(set) var displayed: [Candidate] = []
    public private(set) var displayedVersion = -1
    public private(set) var displayedAt: UInt64 = 0
    public private(set) var displayedNeural = false
    public var highlighted = 0
    public private(set) var navigated = false
    private var keyTime: UInt64 = 0
    private var firstAnalysis: Analysis?
    private var panelTimer: DispatchWorkItem?

    public init(engine: Engine, neural: NeuralService?) {
        self.engine = engine
        self.neural = neural
    }

    public var isComposing: Bool { !raw.isEmpty || !parts.isEmpty }
    /// Text for the inline marked text: converted parts followed by the remaining pinyin, split into syllables.
    public var markedText: String { parts.map { $0.text }.joined() + displayedPinyin }
    /// Where the caret is drawn inside `markedText` (UTF-16 offset): after the converted parts, then after
    /// `caret` letters of the pinyin, skipping the syllable spaces the display adds.
    public var markedCaret: Int {
        var offset = (parts.map { $0.text }.joined() as NSString).length
        var letters = 0
        for ch in displayedPinyin {
            if letters == caret { break }
            offset += 1
            if ch != " " { letters += 1 }
        }
        return offset
    }
    private var displayedPinyin: String {
        if spacedPinyin, let a = analysis, !raw.contains("'") { return a.segmentedPinyin(table: engine.lexicon.syllables) }
        return raw
    }

    public func begin(context: String, app: String?) {
        reset()
        self.context = context
        self.app = app
    }

    public func reset() {
        raw = ""; caret = 0; parts = []; analysis = nil; displayed = []; displayedVersion = -1
        highlighted = 0; navigated = false; firstAnalysis = nil
        panelTimer?.cancel(); panelTimer = nil
    }

    private var fullContext: String { context + parts.map { $0.text }.joined() }

    // MARK: editing

    /// Insert typed keys at the caret.
    public func insert(_ s: String) {
        raw.insert(contentsOf: s, at: raw.index(raw.startIndex, offsetBy: caret))
        caret += s.count
        refresh()
    }

    /// Delete the key before the caret. At the start of the pinyin, the last converted part is unconverted
    /// back into its keys instead. Returns false when there was nothing to delete.
    @discardableResult
    public func backspace() -> Bool {
        if caret > 0 {
            raw.remove(at: raw.index(raw.startIndex, offsetBy: caret - 1))
            caret -= 1
        } else if let last = parts.popLast() {
            raw = last.raw + raw
            caret = last.raw.count
        } else {
            return false
        }
        if !isComposing { reset(); return true }
        refresh()
        return true
    }

    /// Forward delete: remove the key after the caret. Returns false when the caret is at the end.
    @discardableResult
    public func deleteForward() -> Bool {
        guard caret < raw.count else { return false }
        raw.remove(at: raw.index(raw.startIndex, offsetBy: caret))
        if !isComposing { reset(); return true }
        refresh()
        return true
    }

    /// Move the caret inside the pinyin (← →). Returns false when it was already at that end.
    @discardableResult
    public func moveCaret(_ delta: Int) -> Bool {
        let target = max(0, min(raw.count, caret + delta))
        guard target != caret else { return false }
        caret = target
        return true
    }

    /// Put the caret at the start (Home) or the end (End) of the pinyin.
    public func moveCaret(toEnd: Bool) { caret = toEnd ? raw.count : 0 }

    public func moveHighlight(_ delta: Int) {
        guard !displayed.isEmpty else { return }
        highlighted = max(0, min(displayed.count - 1, highlighted + delta))
        navigated = true
        onDisplay?()
    }

    private func refresh() {
        keyTime = DispatchTime.now().uptimeNanoseconds
        version += 1
        highlighted = 0
        navigated = false
        guard !raw.isEmpty else {
            analysis = nil
            publish([], neural: false)
            return
        }
        let a = engine.analyze(raw: raw, context: fullContext, app: app)
        analysis = a
        let v = version
        if let neural, neural.isReady, !a.full.isEmpty {
            neural.submit(NeuralService.Job(version: v, context: fullContext, options: a.options)) { [weak self] r in
                self?.runOnMain { self?.neuralArrived(r) }
            }
            // show the fast list only if the neural one is late
            panelTimer?.cancel()
            let item = DispatchWorkItem { [weak self] in
                guard let self, self.version == v, self.displayedVersion != v else { return }
                self.publish(self.engine.displayList(a), neural: false)
            }
            panelTimer = item
            DispatchQueue.main.asyncAfter(deadline: .now() + policy.panelWaitMs / 1000, execute: item)
        } else {
            publish(engine.displayList(a), neural: false)
        }
    }

    private func neuralArrived(_ r: NeuralService.Result) {
        guard r.version == version, var a = analysis, a.options == r.options, !a.nnApplied else { return }
        engine.applyNeural(&a, scores: r.scores)
        analysis = a
        let sinceKey = Double(DispatchTime.now().uptimeNanoseconds - keyTime) / 1e6
        let seen = displayedVersion == version && Double(DispatchTime.now().uptimeNanoseconds - displayedAt) / 1e6 > policy.seenMs
        // Never reorder a list the user may already be reading or navigating.
        if displayedVersion != version || (!navigated && !seen && sinceKey < policy.seenMs) {
            panelTimer?.cancel()
            publish(engine.displayList(a), neural: true)
        }
    }

    private func publish(_ list: [Candidate], neural: Bool) {
        displayed = list
        displayedVersion = version
        displayedAt = DispatchTime.now().uptimeNanoseconds
        displayedNeural = neural
        if !navigated { highlighted = 0 }
        onDisplay?()
    }

    // MARK: committing

    /// Space: commit the highlighted candidate, making sure the smart first choice for the exact input is used.
    public func commitHighlighted() -> CommitResult? {
        guard isComposing else { return nil }
        var waited = 0.0
        if raw.isEmpty { return finish(rank: 0, waited: 0) }
        let shownAge = displayedVersion == version ? Double(DispatchTime.now().uptimeNanoseconds - displayedAt) / 1e6 : 0
        let wysiwyg = displayedVersion == version && (displayedNeural || navigated || shownAge > policy.seenMs)
        if !wysiwyg, let neural, var a = analysis, !a.nnApplied, neural.isReady {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let deadlineLeft = max(5, policy.commitWaitMs - Double(t0 - keyTime) / 1e6)
            if let r = neural.result(for: version) ?? neural.wait(for: version, timeout: deadlineLeft / 1000), r.options == a.options {
                engine.applyNeural(&a, scores: r.scores)
                analysis = a
                panelTimer?.cancel()
                publish(engine.displayList(a), neural: true)
            }
            waited = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        } else if displayedVersion != version, let a = analysis {
            publish(engine.displayList(a), neural: a.nnApplied)
        }
        return select(index: highlighted, waited: waited)
    }

    /// Choose the displayed candidate at `index` (number key or space). May be a partial selection.
    public func select(index: Int, waited: Double = 0) -> CommitResult? {
        guard index >= 0, index < displayed.count, let a = analysis else { return nil }
        let c = displayed[index]
        if parts.isEmpty { firstAnalysis = a }
        let rest = a.remainingRaw(after: c)
        let consumed = String(raw.prefix(raw.count - rest.count))
        let rank = index == 0 ? 0 : index
        parts.append((c.text, c, consumed, rank))
        raw = rest
        caret = raw.count
        if raw.isEmpty {
            return finish(rank: parts.count > 1 ? -1 : rank, waited: waited)
        }
        refresh()
        return nil
    }

    /// Return key: commit the typed letters as they are.
    public func commitRaw() -> String {
        let s = parts.map { $0.text }.joined() + raw
        reset()
        return s
    }

    private func finish(rank: Int, waited: Double) -> CommitResult {
        let text = parts.map { $0.text }.joined()
        var charSyls: [SylID?] = []
        var words: [WordID] = []
        var wordSyls: [[SylID]] = []
        var typo = false
        for p in parts {
            for s in p.cand.segments {
                words.append(s.word)
                wordSyls.append(s.syls)
                let nChars = engine.lexicon.charCount(s.word)
                if s.syls.count == nChars { charSyls.append(contentsOf: s.syls.map { Optional($0) }) }
                else { charSyls.append(contentsOf: Array(repeating: nil, count: nChars)) }
                if s.typos > 0 { typo = true }
            }
        }
        let a0 = firstAnalysis ?? analysis
        let rec = CommitRecord(time: Date().timeIntervalSince1970, app: app, context: context, raw: parts.map { $0.raw }.joined(),
                               text: text, charSyls: charSyls, words: words, wordSyls: wordSyls, chosenRank: rank,
                               analysis: a0, topProb: a0?.full.first?.prob ?? 0, usedTypoCorrection: typo)
        let usedNN = a0?.nnApplied ?? false
        let ctx = context + text
        reset()
        context = ctx
        neural?.prepareContext(ctx)
        return CommitResult(text: text, record: rec, waitedMs: waited, usedNeural: usedNN)
    }
}
