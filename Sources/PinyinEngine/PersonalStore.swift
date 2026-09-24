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
                         memory: engine.memory.table, typo: engine.typo.encode(),
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
        engine.typo.decay(f)
        lastDecay = now
    }

    public func erase() {
        try? FileManager.default.removeItem(at: file)
    }
}

extension ContextMemory {
    func restore(_ t: [UInt64: [Entry]]) { table = t }
}
