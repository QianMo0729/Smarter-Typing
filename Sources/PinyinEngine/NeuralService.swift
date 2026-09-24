import Foundation

/// Runs the neural judge on its own serial queue with "latest wins" scheduling, so typing never waits
/// on the model and stale work is dropped. Results are tagged with the composition version they belong to.
public final class NeuralService: @unchecked Sendable {
    public struct Job {
        public let version: Int
        public let context: String
        public let options: [String]
    }
    public struct Result {
        public let version: Int
        public let options: [String]
        public let scores: [Float]
        public let ms: Double
        public let queuedAt: UInt64
        public let doneAt: UInt64
    }

    private let queue = DispatchQueue(label: "smartpinyin.neural", qos: .userInteractive)
    private let lock = NSLock()
    private var scorer: NeuralScorer?
    private var pending: (Job, UInt64, (Result) -> Void)?
    private var running = false
    private var results: [Int: Result] = [:]
    private var waiters: [Int: [DispatchSemaphore]] = [:]
    public private(set) var loadError: String?
    public var isReady: Bool { lock.lock(); defer { lock.unlock() }; return scorer != nil }

    public init() {}

    /// Load the model in the background (a few hundred ms warm, longer on first launch).
    public func load(modelPath: String, config: NeuralScorer.Config = NeuralScorer.Config(), warmup: Bool = true, done: (() -> Void)? = nil) {
        queue.async {
            setenv("GGML_METAL_TENSOR_DISABLE", "1", 0)
            do {
                let s = try NeuralScorer(modelPath: modelPath, config: config)
                if warmup {
                    // compile GPU pipelines for the batch shapes we use
                    s.setContext("预热")
                    _ = s.score(["智能输入法", "只能输入法", "智能", "只能"])
                    _ = s.score(["你好"])
                    s.setContext("")
                }
                self.lock.lock(); self.scorer = s; self.lock.unlock()
            } catch {
                self.lock.lock(); self.loadError = "\(error)"; self.lock.unlock()
            }
            done?()
        }
    }

    /// Release the model (call before the process exits so GPU resources are torn down in order).
    public func shutdown() {
        queue.sync {
            self.lock.lock(); self.scorer = nil; self.lock.unlock()
        }
    }

    /// Prefill the context cache ahead of time (e.g. right after a commit).
    public func prepareContext(_ context: String) {
        queue.async {
            self.lock.lock(); let s = self.scorer; self.lock.unlock()
            s?.setContext(context)
        }
    }

    /// Submit a scoring job; a job that has not started yet is replaced by a newer one.
    public func submit(_ job: Job, completion: @escaping (Result) -> Void) {
        lock.lock()
        pending = (job, DispatchTime.now().uptimeNanoseconds, completion)
        let start = !running
        if start { running = true }
        lock.unlock()
        if start { queue.async { self.drain() } }
    }

    private func drain() {
        while true {
            lock.lock()
            guard let (job, queuedAt, completion) = pending else { running = false; lock.unlock(); return }
            pending = nil
            let s = scorer
            lock.unlock()
            guard let s else { continue }
            let t0 = DispatchTime.now().uptimeNanoseconds
            s.setContext(job.context)
            let scores = s.score(job.options)
            let t1 = DispatchTime.now().uptimeNanoseconds
            let r = Result(version: job.version, options: job.options, scores: scores, ms: Double(t1 - t0) / 1e6, queuedAt: queuedAt, doneAt: t1)
            lock.lock()
            results[job.version] = r
            if results.count > 8 { for k in results.keys.sorted().dropLast(8) { results[k] = nil } }
            let ws = waiters.removeValue(forKey: job.version) ?? []
            lock.unlock()
            for w in ws { w.signal() }
            completion(r)
        }
    }

    /// Result for `version` if it is already available.
    public func result(for version: Int) -> Result? {
        lock.lock(); defer { lock.unlock() }
        return results[version]
    }

    /// Block the caller up to `timeout` seconds for the result of `version` (used when the user commits
    /// right after the last keystroke). Returns nil on timeout.
    public func wait(for version: Int, timeout: Double) -> Result? {
        lock.lock()
        if let r = results[version] { lock.unlock(); return r }
        let pendingMatches = pending?.0.version == version
        if !running && !pendingMatches { lock.unlock(); return nil }
        let sem = DispatchSemaphore(value: 0)
        waiters[version, default: []].append(sem)
        lock.unlock()
        _ = sem.wait(timeout: .now() + timeout)
        lock.lock(); defer { lock.unlock() }
        waiters[version] = nil
        return results[version]
    }
}
