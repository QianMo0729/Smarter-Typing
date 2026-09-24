import Foundation

/// Word bigram language model with backoff (Kneser-Ney estimated offline, see tools/build_ngram.py).
///
/// lm.bin: magic 'STLM', version, V, nBigrams, then 8-byte aligned sections
///   unigram f32[V], backoff f32[V], bigramStart u32[V+1], bigramW2 u32[nB], bigramLogP f16[nB]
public final class NGramModel: @unchecked Sendable {
    public let vocab: Int
    private let reader: BinaryReader?
    private let unigram: UnsafeBufferPointer<Float>
    private let backoff: UnsafeBufferPointer<Float>?
    private let bStart: UnsafeBufferPointer<UInt32>?
    private let bW2: UnsafeBufferPointer<UInt32>?
    private let bLogP: UnsafeBufferPointer<Float16>?
    public let hasBigrams: Bool
    /// Log prior for words added to the lexicon after the n-gram model was estimated (ids ≥ vocab).
    public var oovPrior: UnsafeBufferPointer<Float>?

    public init(url: URL) throws {
        var r = try BinaryReader(url: url, magic: "STLM")
        _ = r.u32()
        let v = Int(r.u32()), nb = Int(r.u32())
        vocab = v
        unigram = r.section(Float.self, count: v)
        backoff = r.section(Float.self, count: v)
        bStart = r.section(UInt32.self, count: v + 1)
        bW2 = r.section(UInt32.self, count: nb)
        bLogP = r.section(Float16.self, count: nb)
        reader = r
        hasBigrams = nb > 0
    }

    /// Unigram-only fallback built from the lexicon prior.
    public init(lexicon: Lexicon) {
        vocab = lexicon.surfaceCount
        unigram = lexicon.prior
        reader = nil; backoff = nil; bStart = nil; bW2 = nil; bLogP = nil
        hasBigrams = false
    }

    @inline(__always) func oov(_ w: WordID) -> Float {
        if let p = oovPrior, Int(w) < p.count { return p[Int(w)] - 2 }
        return -25
    }

    @inline(__always) public func unigramLogP(_ w: WordID) -> Float {
        Int(w) < vocab ? unigram[Int(w)] : oov(w)
    }

    /// log P(w | prev)
    @inline(__always) public func logP(_ w: WordID, given prev: WordID) -> Float {
        let wi = Int(w)
        guard wi < vocab else { return oov(w) }
        guard hasBigrams, prev >= 0, Int(prev) < vocab, let bStart, let bW2, let bLogP, let backoff else { return unigram[wi] }
        var lo = Int(bStart[Int(prev)]), hi = Int(bStart[Int(prev) + 1])
        let target = UInt32(w)
        while lo < hi {
            let mid = (lo + hi) >> 1
            let v = bW2[mid]
            if v == target { return Float(bLogP[mid]) }
            if v < target { lo = mid + 1 } else { hi = mid }
        }
        return backoff[Int(prev)] + unigram[wi]
    }
}
