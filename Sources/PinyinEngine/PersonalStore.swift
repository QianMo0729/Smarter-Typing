import Foundation

/// Saves / restores everything the engine learned about the user. Files stay on this Mac.
public final class PersonalStore {
    public let dir: URL
    let engine: Engine

    struct Snapshot: Codable {
        var version = 2
        var savedAt: Double
        var lastDecay: Double
        var userLM: UserLM.State
        var memory: [UInt64: [ContextMemory.Entry]]
        var userWords: UserWords.State?
        var typo: Data?
        var weightsFast: [Float]
        var weightsFull: [Float]
        var typoGuard: Float
    }

    public private(set) var lastDecay: Double = Date().timeIntervalSince1970
    /// Half-life of personal word statistics; old habits fade so new ones can take over.
    public var halfLifeDays: Double = 120

    public init(dir: URL, engine: Engine) {
        self.dir = dir
        self.engine = engine
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    var file: URL { dir.appendingPathComponent("personal.json") }

    private let ioQueue = DispatchQueue(label: "smartpinyin.store", qos: .utility)

    /// Snapshot on the calling thread (cheap copy-on-write), encode and write in the background so typing
    /// never stalls on disk I/O. Pass `sync: true` when the process is about to exit.
    public func save(sync: Bool = false) {
        let s = Snapshot(savedAt: Date().timeIntervalSince1970, lastDecay: lastDecay, userLM: engine.userLM.state(),
                         memory: engine.memory.table, userWords: engine.userWords.state(table: engine.lexicon.syllables), typo: engine.typo.encode(),
                         weightsFast: engine.weightsFast.w, weightsFull: engine.weightsFull.w, typoGuard: engine.typoGuard)
        if sync { ioQueue.sync { self.write(s) } } else { ioQueue.async { self.write(s) } }
    }

    private func write(_ s: Snapshot) {
        guard let d = try? JSONEncoder().encode(s) else { return }
        let tmp = dir.appendingPathComponent("personal.json.tmp")
        do {
            try d.write(to: tmp, options: .atomic)
            _ = try FileManager.default.replaceItemAt(file, withItemAt: tmp)
        } catch {
            try? d.write(to: file, options: .atomic)
        }
    }

    public func load() {
        guard let d = try? Data(contentsOf: file), let s = try? JSONDecoder().decode(Snapshot.self, from: d) else { return }
        engine.userLM.restore(s.userLM)
        engine.memory.restore(s.memory)
        if let u = s.userWords { engine.userWords.restore(u, table: engine.lexicon.syllables) } else { adoptWordsFromMemory() }
        if let t = s.typo { engine.typo.load(t) }
        // only the personally adapted trust weights are restored; everything else comes from the shipped model,
        // so an update of the base weights is not shadowed by an old copy
        for k in [7, 8] {
            if s.weightsFast.count > k { engine.weightsFast.w[k] = s.weightsFast[k] }
            if s.weightsFull.count > k { engine.weightsFull.w[k] = s.weightsFull[k] }
        }
        engine.typoGuard = s.typoGuard
        lastDecay = s.lastDecay
        applyDecay()
    }

    /// Exponential forgetting, applied at most daily.
    public func applyDecay(now: Double = Date().timeIntervalSince1970) {
        let days = (now - lastDecay) / 86400
        guard days >= 1 else { return }
        let f = pow(0.5, days / halfLifeDays)
        engine.userLM.decay(Float(f))
        engine.userWords.decay(Float(f))
        engine.typo.decay(f)
        lastDecay = now
    }

    public func erase() {
        try? FileManager.default.removeItem(at: file)
    }

    /// Data saved before user words existed. The memory already holds what the user kept putting together by
    /// hand (it could be remembered there, but never offered): adopt those texts as user words, once.
    func adoptWordsFromMemory() {
        let lexicon = engine.lexicon, user = engine.userWords
        // readings of every character, to find the syllables behind a memory key again
        var readings: [Character: [SylID]] = [:]
        for s in 0..<lexicon.syllables.count {
            let node = lexicon.child(0, SylID(s))
            if node < 0 { continue }
            for wi in lexicon.wordRange(node) {
                let w = lexicon.wordSurface(wi)
                if lexicon.charCount(w) == 1, let c = lexicon.surface(w).first { readings[c, default: []].append(SylID(s)) }
            }
        }
        func syllables(of chars: [Character], key: UInt64) -> [SylID]? {
            var found: [SylID]?
            var path: [SylID] = []
            var tried = 0
            func walk(_ i: Int) {
                if found != nil || tried > 2000 { return }
                if i == chars.count {
                    tried += 1
                    if ContextMemory.key(path) == key { found = path }
                    return
                }
                for s in readings[chars[i]] ?? [] { path.append(s); walk(i + 1); path.removeLast() }
            }
            walk(0)
            return found
        }
        // texts the user wanted that are no word of the lexicon, longest first: the memory also holds pieces
        // of the same text (顾祎 and 祎宁 next to 顾祎宁)
        var wanted: [(text: String, key: UInt64, weight: Float, time: Double)] = []
        for (key, list) in engine.memory.table {
            var byText: [String: (weight: Float, time: Double)] = [:]
            for e in list { byText[e.text] = ((byText[e.text]?.weight ?? 0) + e.weight, max(byText[e.text]?.time ?? 0, e.time)) }
            for (text, v) in byText where v.weight > 0 && text.count >= 2 && text.count <= user.maxLength && lexicon.id(of: text) == nil {
                wanted.append((text, key, v.weight, v.time))
            }
        }
        wanted.sort { ($0.text.count, $0.weight, $0.text) > ($1.text.count, $1.weight, $1.text) }
        var kept: [String] = []
        for w in wanted where !kept.contains(where: { $0.contains(w.text) }) {
            // only what the engine does not arrive at by itself from the pinyin
            guard let syls = syllables(of: Array(w.text), key: w.key), engine.shortfall(of: w.text, syls: syls, context: "") == nil else { continue }
            user.add(text: w.text, syls: syls, weight: min(3, w.weight), time: w.time)
            kept.append(w.text)
        }
    }
}

extension ContextMemory {
    func restore(_ t: [UInt64: [Entry]]) { table = t }
}
