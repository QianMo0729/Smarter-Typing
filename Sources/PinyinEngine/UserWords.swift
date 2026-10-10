import Foundation

/// Words the user taught the input method: text that had to be put together by hand because the lexicon
/// does not have it as a word (a name like 林霁雯), or that sits too deep in the lexicon to ever be
/// offered (霁 for "ji"). Each one is a word of its own, with ids above the lexicon's, so the lattice,
/// the decoder and the personal models treat it like any other word.
public final class UserWords: @unchecked Sendable {
    public struct Entry {
        public let id: WordID
        public let text: String
        public let syls: [SylID]
        public var count: Float       // how often it was taught / used (decays, may be lowered again)
        public var time: Double       // last time it was taught or used
    }

    /// User word ids start far above any lexicon id and are never reused, so counts kept elsewhere
    /// (personal LM) can never end up attached to another word.
    public static let firstID: WordID = 1 << 30
    @inline(__always) public static func isUser(_ w: WordID) -> Bool { w >= firstID }

    /// log P of a word taught once. About what the n-gram gives a moderately rare lexicon word: clearly
    /// better than spelling the same text out of unrelated single characters, not better than a common word.
    public var prior: Float = -11
    /// Extra credit (personal feature) for having been taught explicitly, so the word also holds its own
    /// against lexicon homophones the neural judge happens to like better.
    public var boost: Float = 3
    public var maxEntries = 4000
    public var maxLength = 8

    private struct Key: Hashable { let text: String; let syls: [SylID] }
    private struct Node { var children: [SylID: Int32] = [:]; var words: [WordID] = [] }

    private var entries: [WordID: Entry] = [:]
    private var index: [Key: WordID] = [:]
    private var byText: [String: WordID] = [:]
    private var nodes: [Node] = [Node()]
    private var nextID = UserWords.firstID

    public init() {}

    public var isEmpty: Bool { entries.isEmpty }
    public var count: Int { entries.count }
    public var all: [Entry] { entries.values.sorted { $0.id < $1.id } }

    @inline(__always) public func entry(_ w: WordID) -> Entry? { entries[w] }
    public func id(text: String, syls: [SylID]) -> WordID? { index[Key(text: text, syls: syls)] }
    public func id(ofText text: String) -> WordID? { byText[text] }

    // MARK: scoring

    /// Repeated use counts logarithmically, like a frequency: 1 → 0, 3 → 1.1, 20 or more → 3.
    @inline(__always) private func strength(_ c: Float) -> Float { log(min(20, max(0.25, c))) }

    /// Stand-in for the n-gram probability the base model cannot have for this word.
    @inline(__always) public func logP(_ w: WordID) -> Float {
        guard let e = entries[w] else { return -25 }
        return prior + strength(e.count)
    }

    @inline(__always) public func bonus(_ w: WordID) -> Float {
        guard let e = entries[w] else { return 0 }
        return boost + strength(e.count)
    }

    // MARK: updates

    /// Teach `text` for `syls`, or strengthen it when it is already known.
    @discardableResult
    public func add(text: String, syls: [SylID], weight: Float, time: Double) -> WordID? {
        guard weight > 0, !syls.isEmpty, syls.count == text.count, syls.count <= maxLength else { return nil }
        let key = Key(text: text, syls: syls)
        if let id = index[key] {
            entries[id]!.count += weight
            entries[id]!.time = time
            return id
        }
        let id = nextID
        nextID += 1
        entries[id] = Entry(id: id, text: text, syls: syls, count: weight, time: time)
        link(entries[id]!)
        if entries.count > maxEntries { evict() }
        return id
    }

    /// Raise or lower the standing of a word; it is forgotten when nothing is left.
    public func adjust(_ w: WordID, by delta: Float, time: Double? = nil) {
        guard var e = entries[w] else { return }
        e.count += delta
        if let time, delta > 0 { e.time = time }
        if e.count <= 0.05 {
            entries[w] = nil
            rebuild()
        } else {
            entries[w] = e
        }
    }

    public func decay(_ f: Float) {
        for k in entries.keys { entries[k]!.count *= f }
        let n = entries.count
        entries = entries.filter { $0.value.count > 0.05 }
        if entries.count != n { rebuild() }
    }

    public func removeAll() {
        entries = [:]
        rebuild()
    }

    /// Drop the least used tenth (oldest first among equals).
    private func evict() {
        let order = entries.values.sorted { ($0.count, $0.time) < ($1.count, $1.time) }
        for e in order.prefix(max(1, maxEntries / 10)) { entries[e.id] = nil }
        rebuild()
    }

    private func link(_ e: Entry) {
        index[Key(text: e.text, syls: e.syls)] = e.id
        byText[e.text] = e.id
        var n = 0
        for s in e.syls {
            if let c = nodes[n].children[s] { n = Int(c); continue }
            nodes.append(Node())
            nodes[n].children[s] = Int32(nodes.count - 1)
            n = nodes.count - 1
        }
        nodes[n].words.append(e.id)
    }

    private func rebuild() {
        index = [:]; byText = [:]; nodes = [Node()]
        for e in all { link(e) }
    }

    // MARK: trie keyed by syllables (same walk as Lexicon, for the lattice)

    @inline(__always) func hasChildren(_ node: Int) -> Bool { !nodes[node].children.isEmpty }
    @inline(__always) func words(at node: Int) -> [WordID] { nodes[node].words }

    /// Iterate children of `node` whose syllable is in `syls`.
    @inline(__always) func forEachChild(_ node: Int, in syls: [SylID], _ body: (SylID, Int) -> Void) {
        let children = nodes[node].children
        if children.isEmpty { return }
        if syls.count <= children.count {
            for s in syls { if let c = children[s] { body(s, Int(c)) } }
        } else {
            for (s, c) in children where syls.contains(s) { body(s, Int(c)) }
        }
    }

    // MARK: persistence

    /// Syllables are saved by spelling, so a rebuilt lexicon cannot silently change a reading.
    struct State: Codable {
        struct Word: Codable { var id: WordID; var text: String; var pinyin: [String]; var count: Float; var time: Double }
        var next: WordID
        var words: [Word]
    }

    func state(table: SyllableTable) -> State {
        State(next: nextID, words: all.map { State.Word(id: $0.id, text: $0.text, pinyin: $0.syls.map { table.spelling($0) }, count: $0.count, time: $0.time) })
    }

    func restore(_ s: State, table: SyllableTable) {
        entries = [:]
        nextID = max(UserWords.firstID, s.next)
        for w in s.words {
            let syls = w.pinyin.compactMap { table.idOf[$0] }
            guard UserWords.isUser(w.id), syls.count == w.pinyin.count, syls.count == w.text.count, w.count > 0 else { continue }
            entries[w.id] = Entry(id: w.id, text: w.text, syls: syls, count: w.count, time: w.time)
            nextID = max(nextID, w.id + 1)
        }
        rebuild()
    }
}
