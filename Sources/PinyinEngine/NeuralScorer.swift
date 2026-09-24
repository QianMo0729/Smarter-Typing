import Foundation
import Accelerate
import CLlama

/// Small causal LM (GGUF via llama.cpp) used as a probabilistic judge: given the text before the cursor
/// and a closed set of candidate strings, it returns log P(candidate | context) for every candidate
/// in a single batched forward pass. The context is kept in the KV cache (sequence 0) and extended
/// incrementally, candidates are decoded as a token trie so shared prefixes are computed once.
///
/// Not thread-safe: use it from one serial queue.
public final class NeuralScorer {
    public struct Config {
        public var gpu = true
        public var threads = 6
        public var nCtx: UInt32 = 1024
        public var maxContextTokens = 160      // context window kept in the cache
        public var trimTo = 112                // when exceeded, re-prefill the last `trimTo` tokens
        public var maxCandidates = 48
        public init() {}
    }

    public let config: Config
    let model: OpaquePointer
    let ctx: OpaquePointer
    let vocab: OpaquePointer
    public let nVocab: Int
    let docSep: llama_token
    var batch: llama_batch
    let batchCap: Int32 = 512

    private(set) var cached: [llama_token] = []   // tokens in sequence 0
    private var lastRow: [Float] = []             // logits after the last context token
    private var lastLSE: Float = 0
    private var scratch: [Float]
    private var tokenCache: [String: [llama_token]] = [:]
    public private(set) var contextText: String = ""
    public var lastScoreMs: Double = 0
    public var lastScoreTokens = 0

    public static func silenceLogs() {
        llama_log_set({ _, _, _ in }, nil)
    }

    public init(modelPath: String, config: Config = Config()) throws {
        self.config = config
        NeuralScorer.silenceLogs()
        llama_backend_init()
        var mp = llama_model_default_params()
        mp.n_gpu_layers = config.gpu ? -1 : 0
        guard let m = llama_model_load_from_file(modelPath, mp) else { throw NSError(domain: "NeuralScorer", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot load \(modelPath)"]) }
        model = m
        var cp = llama_context_default_params()
        cp.n_ctx = config.nCtx
        cp.n_batch = UInt32(batchCap)
        cp.n_ubatch = UInt32(batchCap)
        cp.n_seq_max = UInt32(config.maxCandidates + 1)
        cp.n_threads = Int32(config.threads)
        cp.n_threads_batch = Int32(config.threads)
        cp.kv_unified = true
        cp.no_perf = true
        cp.offload_kqv = config.gpu
        cp.op_offload = config.gpu
        guard let c = llama_init_from_model(m, cp) else { throw NSError(domain: "NeuralScorer", code: 2, userInfo: [NSLocalizedDescriptionKey: "cannot create context"]) }
        ctx = c
        vocab = llama_model_get_vocab(m)
        nVocab = Int(llama_vocab_n_tokens(vocab))
        docSep = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : llama_vocab_eos(vocab)
        batch = llama_batch_init(batchCap, 0, Int32(config.maxCandidates + 1))
        scratch = [Float](repeating: 0, count: nVocab)
        _ = setContext("")
    }

    deinit {
        llama_batch_free(batch)
        llama_free(ctx)
        llama_model_free(model)
    }

    public func tokenize(_ text: String) -> [llama_token] {
        if let t = tokenCache[text] { return t }
        let utf8 = Array(text.utf8CString)
        var buf = [llama_token](repeating: 0, count: utf8.count + 8)
        var n = llama_tokenize(vocab, utf8, Int32(utf8.count - 1), &buf, Int32(buf.count), false, false)
        if n < 0 {
            buf = [llama_token](repeating: 0, count: Int(-n))
            n = llama_tokenize(vocab, utf8, Int32(utf8.count - 1), &buf, Int32(buf.count), false, false)
        }
        let out = Array(buf.prefix(Int(max(0, n))))
        if tokenCache.count > 20000 { tokenCache.removeAll(keepingCapacity: true) }
        tokenCache[text] = out
        return out
    }

    private var memory: llama_memory_t { llama_get_memory(ctx) }

    private func clearBatch() { batch.n_tokens = 0 }

    private func add(_ token: llama_token, pos: Int, seqs: [Int32], logits: Bool) {
        let i = Int(batch.n_tokens)
        batch.token[i] = token
        batch.pos[i] = llama_pos(pos)
        batch.n_seq_id[i] = Int32(seqs.count)
        for (k, s) in seqs.enumerated() { batch.seq_id[i]![k] = s }
        batch.logits[i] = logits ? 1 : 0
        batch.n_tokens += 1
    }

    /// Log-sum-exp of a logits row.
    private func lse(_ row: UnsafePointer<Float>) -> Float {
        var mx: Float = 0
        vDSP_maxv(row, 1, &mx, vDSP_Length(nVocab))
        var neg = -mx
        var n32 = Int32(nVocab)
        scratch.withUnsafeMutableBufferPointer { s in
            vDSP_vsadd(row, 1, &neg, s.baseAddress!, 1, vDSP_Length(nVocab))
            vvexpf(s.baseAddress!, s.baseAddress!, &n32)
        }
        var sum: Float = 0
        vDSP_sve(scratch, 1, &sum, vDSP_Length(nVocab))
        return mx + log(sum)
    }

    /// Choose the window of `full` to keep in the cache. The window start stays anchored while text is
    /// appended (so each commit only costs the new tokens) and jumps forward when it grows too long.
    func window(for full: String) -> String {
        let maxChars = config.maxContextTokens * 5 / 4, trimChars = config.trimTo * 5 / 4
        if !contextText.isEmpty, let r = full.range(of: contextText, options: .backwards) {
            let extended = contextText + full[r.upperBound...]
            if extended.count <= maxChars && full.hasSuffix(extended) { return extended }
        }
        return full.count <= maxChars ? full : String(full.suffix(trimChars))
    }

    /// Make sequence 0 hold the tail of `full`. Returns false on failure.
    @discardableResult
    public func setContext(_ full: String) -> Bool {
        let text = window(for: full)
        var toks = [docSep] + tokenize(text)
        if toks.count > config.maxContextTokens { toks = [docSep] + Array(toks.suffix(config.trimTo)) }
        // longest common prefix with what is cached
        var common = 0
        while common < min(toks.count, cached.count) && toks[common] == cached[common] { common += 1 }
        if common == toks.count && common == cached.count && !lastRow.isEmpty { contextText = text; return true }
        if common == toks.count { common -= 1 }   // need to recompute logits of the last token
        if common == 0 { llama_memory_clear(memory, true) } else { _ = llama_memory_seq_rm(memory, 0, llama_pos(common), -1) }
        var pos = common
        while pos < toks.count {
            clearBatch()
            let end = min(toks.count, pos + Int(batchCap))
            for i in pos..<end { add(toks[i], pos: i, seqs: [0], logits: i == toks.count - 1) }
            if llama_decode(ctx, batch) != 0 { cached = []; lastRow = []; return false }
            pos = end
        }
        let row = llama_get_logits_ith(ctx, batch.n_tokens - 1)!
        lastRow = Array(UnsafeBufferPointer(start: row, count: nVocab))
        lastLSE = lse(row)
        cached = toks
        contextText = text
        return true
    }

    /// log P(candidate | context) for each candidate (natural log). Candidates are scored in one batch.
    public func score(_ candidates: [String]) -> [Float] {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let cands = Array(candidates.prefix(config.maxCandidates))
        var result = [Float](repeating: -1e9, count: candidates.count)
        if cands.isEmpty || cached.isEmpty { return result }
        // token trie
        struct Node { var token: llama_token; var parent: Int; var depth: Int; var seqs: [Int32]; var children: [llama_token: Int]; var batchIndex: Int32 }
        var nodes: [Node] = [Node(token: -1, parent: -1, depth: 0, seqs: [], children: [:], batchIndex: -1)]
        var paths: [[Int]] = []
        for (ci, c) in cands.enumerated() {
            var cur = 0
            var path: [Int] = []
            for t in tokenize(c) {
                if let nx = nodes[cur].children[t] { cur = nx } else {
                    nodes.append(Node(token: t, parent: cur, depth: nodes[cur].depth + 1, seqs: [], children: [:], batchIndex: -1))
                    nodes[cur].children[t] = nodes.count - 1
                    cur = nodes.count - 1
                }
                nodes[cur].seqs.append(Int32(ci + 1))
                path.append(cur)
            }
            paths.append(path)
        }
        let nNew = nodes.count - 1
        if nNew > Int(batchCap) { return result }
        let base = cached.count
        for ci in 0..<cands.count { llama_memory_seq_cp(memory, 0, llama_seq_id(ci + 1), -1, -1) }
        clearBatch()
        for i in 1..<nodes.count {
            let needLogits = !nodes[i].children.isEmpty
            nodes[i].batchIndex = needLogits ? batch.n_tokens : -1
            add(nodes[i].token, pos: base + nodes[i].depth - 1, seqs: nodes[i].seqs, logits: needLogits)
        }
        let ok = llama_decode(ctx, batch) == 0
        if ok {
            // log-softmax normalizers of every row we need
            var lseByNode = [Int: Float]()
            for i in 1..<nodes.count where nodes[i].batchIndex >= 0 {
                lseByNode[i] = lse(llama_get_logits_ith(ctx, nodes[i].batchIndex)!)
            }
            for (ci, path) in paths.enumerated() {
                var lp: Float = 0
                for (k, nodeIdx) in path.enumerated() {
                    let tok = Int(nodes[nodeIdx].token)
                    if k == 0 {
                        lp += lastRow[tok] - lastLSE
                    } else {
                        let p = path[k - 1]
                        let row = llama_get_logits_ith(ctx, nodes[p].batchIndex)!
                        lp += row[tok] - lseByNode[p]!
                    }
                }
                result[ci] = lp
            }
        }
        for ci in 0..<cands.count { _ = llama_memory_seq_rm(memory, llama_seq_id(ci + 1), -1, -1) }
        lastScoreTokens = nNew
        lastScoreMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        return result
    }
}
