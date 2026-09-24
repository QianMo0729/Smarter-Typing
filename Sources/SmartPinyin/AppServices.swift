import AppKit
import PinyinEngine

/// Process-wide state shared by all input controllers: the engine, the neural judge, the learner.
final class AppServices {
    static let shared = AppServices()

    let engine: Engine
    let neural: NeuralService
    let learner: Learner
    let store: PersonalStore
    let stats = Stats()
    var learningPaused = false
    private var timer: Timer?
    private var dirty = false

    static var supportDir: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("SmartPinyin", isDirectory: true)
    }

    static var resourceDir: URL {
        if let env = ProcessInfo.processInfo.environment["SMARTPINYIN_RESOURCES"] { return URL(fileURLWithPath: env) }
        return Bundle.main.resourceURL!.appendingPathComponent("data", isDirectory: true)
    }

    private init() {
        let res = AppServices.resourceDir
        do {
            engine = try Engine(resources: EngineResources(dir: res))
        } catch {
            Log.write("fatal: cannot load engine resources from \(res.path): \(error)")
            fatalError("SmartPinyin resources missing")
        }
        neural = NeuralService()
        learner = Learner(engine: engine)
        learner.recordEvents = false
        store = PersonalStore(dir: AppServices.supportDir, engine: engine)
        store.load()
        learner.onEvent = { [weak self] e in
            self?.dirty = true
            if e.kind != .kept { Log.write("learn \(e.kind.rawValue) w=\(String(format: "%.2f", e.weight))") }
        }
        let model = res.appendingPathComponent("model.gguf").path
        if FileManager.default.fileExists(atPath: model) {
            let t0 = Date()
            neural.load(modelPath: model) {
                Log.write("neural model ready in \(String(format: "%.0f", Date().timeIntervalSince(t0) * 1000))ms")
            }
        } else {
            Log.write("no neural model at \(model); running n-gram only")
        }
        timer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in self?.housekeeping() }
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: .main) { [weak self] _ in
            self?.learner.flush(now: Date().timeIntervalSince1970)
            self?.store.save(sync: true)
            self?.neural.shutdown()
        }
    }

    private var lastSave = Date()
    private func housekeeping() {
        let now = Date().timeIntervalSince1970
        learner.tick(now)
        if dirty && Date().timeIntervalSince(lastSave) > 60 {
            store.applyDecay()
            store.save()
            dirty = false
            lastSave = Date()
        }
    }

    func saveSoon() { dirty = true }

    func resetPersonalData() {
        learner.didBreak(time: Date().timeIntervalSince1970)
        engine.resetPersonal()
        store.erase()
        dirty = false
        Log.write("personal data erased")
    }
}

/// Timing statistics (numbers only, never text) so latency can be checked in real use.
final class Stats {
    var commits = 0
    var neuralCommits = 0
    var waitedCommits = 0
    var totalWait = 0.0
    var maxWait = 0.0
    var compositions = 0
    var docContext = 0          // context read from the app itself
    var localOnly = 0           // only our own record of recent commits
    var noContext = 0
    var contextChars = 0

    func recordContext(fromDocument: Int, local: Int) {
        compositions += 1
        if fromDocument > 0 { docContext += 1; contextChars += fromDocument }
        else if local > 0 { localOnly += 1; contextChars += local }
        else { noContext += 1 }
    }

    func record(_ r: CommitResult) {
        commits += 1
        if r.usedNeural { neuralCommits += 1 }
        if r.waitedMs > 0.5 { waitedCommits += 1 }
        totalWait += r.waitedMs
        maxWait = max(maxWait, r.waitedMs)
        if commits % 50 == 0 { Log.write(summary) }
    }

    var summary: String {
        String(format: "commits %d, smart-ranked %d (%.1f%%), waited on space %d (avg %.1fms, max %.1fms); context from app %d, own %d, none %d",
               commits, neuralCommits, commits > 0 ? 100.0 * Double(neuralCommits) / Double(commits) : 0,
               waitedCommits, waitedCommits > 0 ? totalWait / Double(waitedCommits) : 0, maxWait, docContext, localOnly, noContext)
    }
}

enum Log {
    static let url: URL = {
        let dir = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0].appendingPathComponent("Logs")
        return dir.appendingPathComponent("SmartPinyin.log")
    }()
    static let queue = DispatchQueue(label: "smartpinyin.log")

    static func write(_ s: String) {
        let line = "\(ISO8601DateFormatter().string(from: Date())) \(s)\n"
        queue.async {
            if let h = try? FileHandle(forWritingTo: url) {
                h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close()
            } else {
                try? line.data(using: .utf8)!.write(to: url)
            }
        }
    }
}
