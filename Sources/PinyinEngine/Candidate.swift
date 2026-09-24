import Foundation

public struct Segment {
    public var word: WordID
    public var start: Int
    public var end: Int
    public var syls: [SylID]
    public var typoCost: Float
    public var shapeCost: Float
    public var typos: Int
    public var edits: [SylEdge]
}

/// Features of a candidate that the probabilistic decision layer combines.
public struct Features {
    public static let names = ["lm", "reading", "typoCost", "typos", "shapeCost", "words", "nn", "personal", "memory", "nnMissing", "nnCtx", "lmCtx"]
    public static let count = names.count

    public var lm: Float = 0          // n-gram log P(candidate | previous word)
    public var reading: Float = 0     // Σ log P(reading | word)
    public var typoCost: Float = 0    // channel cost of assumed keyboard slips (nats)
    public var typos: Float = 0       // number of assumed slips
    public var shapeCost: Float = 0   // abbreviated / unfinished syllables
    public var words: Float = 0       // number of words (segmentation granularity)
    public var nn: Float = 0          // neural log P(candidate | context) (0 when missing)
    public var personal: Float = 0    // personal language model log-ratio
    public var memory: Float = 0      // context memory vote from past choices / corrections
    public var nnMissing: Float = 1   // 1 when the neural score is not available
    /// How much preceding text there is (0 = none … 1 = 30+ characters). The decision layer trusts the
    /// neural judge more and the n-gram less as context grows (interaction features nnCtx, lmCtx).
    public var ctx: Float = 0

    public var vector: [Float] { [lm, reading, typoCost, typos, shapeCost, words, nn, personal, memory, nnMissing, nn * ctx, lm * ctx] }

    public init() {}
}

public struct Candidate {
    public var text: String
    public var segments: [Segment]
    public var end: Int               // number of (clean) keys covered
    public var features = Features()
    public var score: Float = 0       // fused log-linear score
    public var prob: Float = 0        // probability within the candidate set
    public var isFull: Bool           // covers the whole input

    public var typoCount: Int { segments.reduce(0) { $0 + $1.typos } }
    public var syllables: [SylID] { segments.flatMap { $0.syls } }
}
