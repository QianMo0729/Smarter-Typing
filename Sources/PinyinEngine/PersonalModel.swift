import Foundation

/// Personal word statistics learned from what the user keeps. Provides a context-dependent bonus
/// (a smoothed log-ratio against the base n-gram), so frequent personal words win where they fit
/// and do not blindly win everywhere.
public final class UserLM: PathBonus, @unchecked Sendable {
    let base: NGramModel
    public private(set) var uni: [WordID: Float] = [:]
    public private(set) var bi: [UInt64: Float] = [:]
    public private(set) var prevTotal: [WordID: Float] = [:]
    public private(set) var total: Float = 0
    public var uniWeight: Float = 0.6
    public var maxBonus: Float = 8
    public var minBonus: Float = -2
    /// Words the user taught explicitly; they carry a bonus of their own, before any statistics exist.
    public var taught: UserWords?

    public init(base: NGramModel) { self.base = base }

    /// Whether there is anything personal to add to the search.
    public var isActive: Bool { total > 0 || !(taught?.isEmpty ?? true) }

    @inline(__always) static func key(_ a: WordID, _ b: WordID) -> UInt64 { UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b)) }

    public func bonus(prev: WordID, word: WordID) -> Float {
        var b: Float = UserWords.isUser(word) ? taught?.bonus(word) ?? 0 : 0
        if total == 0 { return b }
        // Observed vs expected counts with a small pseudo-count: words the user uses more than the base model
        // expects get a bonus; frequent words the user conspicuously avoids get a penalty; rare unseen words
        // are left alone (their expected count is tiny).
        if let cp = prevTotal[prev], cp > 0 {
            let c = bi[Self.key(prev, word)] ?? 0
            let expected = cp * exp(base.logP(word, given: prev))
            b += log((c + 1) / (expected + 1))
        }
        let cu = uni[word] ?? 0
        let expectedU = total * exp(base.unigramLogP(word))
        b += uniWeight * log((cu + 2) / (expectedU + 2))
        return min(maxBonus, max(minBonus, b))
    }

    /// Add evidence that `words` were written after `prev` with confidence `weight` (may be negative to retract).
    public func add(words: [WordID], prev: WordID, weight: Float) {
        var p = prev
        for w in words {
            uni[w, default: 0] += weight
            bi[Self.key(p, w), default: 0] += weight
            prevTotal[p, default: 0] += weight
            total += weight
            if uni[w]! <= 0.001 { uni[w] = nil }
            if bi[Self.key(p, w)]! <= 0.001 { bi[Self.key(p, w)] = nil }
            if prevTotal[p]! <= 0.001 { prevTotal[p] = nil }
            p = w
        }
        total = max(0, total)
    }

    public func decay(_ f: Float) {
        for k in uni.keys { uni[k]! *= f }
        for k in bi.keys { bi[k]! *= f }
        for k in prevTotal.keys { prevTotal[k]! *= f }
        total *= f
        uni = uni.filter { $0.value > 0.01 }
        bi = bi.filter { $0.value > 0.01 }
        prevTotal = prevTotal.filter { $0.value > 0.01 }
    }

    struct State: Codable { var uni: [WordID: Float]; var bi: [UInt64: Float]; var prevTotal: [WordID: Float]; var total: Float }
    func state() -> State { State(uni: uni, bi: bi, prevTotal: prevTotal, total: total) }
    func restore(_ s: State) { uni = s.uni; bi = s.bi; prevTotal = s.prevTotal; total = s.total }
}

/// Sparse bag of context features (recent characters, character bigrams, application).
public struct ContextVector: Codable {
    public var f: [UInt32: Float]

    public init(text: String, app: String?) {
        var m: [UInt32: Float] = [:]
        let chars = Array(text.suffix(16))
        let n = chars.count
        for (i, c) in chars.enumerated() {
            let dist = Float(n - i)
            let w = 1.0 / (1.0 + 0.15 * dist)
            if c.isWhitespace { continue }
            m[Self.h("c" + String(c)), default: 0] += w
            if i + 1 < n { m[Self.h("b" + String(c) + String(chars[i + 1])), default: 0] += 1.5 * w }
        }
        if let app, !app.isEmpty { m[Self.h("app:" + app), default: 0] += 0.8 }
        var norm: Float = 0
        for v in m.values { norm += v * v }
        norm = norm.squareRoot()
        if norm > 0 { for k in m.keys { m[k]! /= norm } }
        f = m
    }

    static func h(_ s: String) -> UInt32 {
        var x: UInt32 = 2166136261
        for b in s.utf8 { x = (x ^ UInt32(b)) &* 16777619 }
        return x
    }

    public func dot(_ o: ContextVector) -> Float {
        let (a, b) = f.count <= o.f.count ? (f, o.f) : (o.f, f)
        var s: Float = 0
        for (k, v) in a { if let w = b[k] { s += v * w } }
        return s
    }
}

/// Memory of past decisions keyed by pinyin: which characters the user wanted for these syllables in
/// which context (a kNN vote). Only contested spans are stored — places where the chosen text differed
/// from a competing reading of the same pinyin — so common words do not add noise.
public final class ContextMemory: @unchecked Sendable {
    public struct Entry: Codable {
        public var text: String
        public var ctx: ContextVector
        public var weight: Float
        public var time: Double
    }
    public internal(set) var table: [UInt64: [Entry]] = [:]
    public var maxPerKey = 48
    public var maxSpan = 6
    public var halfLifeDays: Double = 90
    /// Clock (seconds since 1970); replaceable for simulations.
    public var clock: () -> Double = { Date().timeIntervalSince1970 }

    public init() {}

    @inline(__always) public static func key<S: Collection>(_ syls: S) -> UInt64 where S.Element == SylID {
        var h: UInt64 = 14695981039346656037
        for s in syls { h = (h ^ UInt64(s)) &* 1099511628211; h = (h ^ 0xff) &* 1099511628211 }
        return h
    }

    public func add<S: Collection>(syls: S, text: String, ctx: ContextVector, weight: Float) where S.Element == SylID {
        let now = clock()
        let k = Self.key(syls)
        var list = table[k] ?? []
        // merge with a very similar existing entry to keep the memory compact
        if let i = list.firstIndex(where: { $0.text == text && $0.ctx.dot(ctx) > 0.9 && ($0.weight > 0) == (weight > 0) }) {
            list[i].weight += weight
            list[i].time = now
        } else {
            list.append(Entry(text: text, ctx: ctx, weight: weight, time: now))
        }
        if list.count > maxPerKey {
            list.sort { abs($0.weight) * decay($0, now) > abs($1.weight) * decay($1, now) }
            list.removeLast(list.count - maxPerKey)
        }
        table[k] = list
    }

    @inline(__always) func decay(_ e: Entry, _ now: Double) -> Float {
        Float(pow(0.5, max(0, now - e.time) / (halfLifeDays * 86400)))
    }

    /// Context-weighted vote for `text` as the reading of `syls` (bounded log-odds style score).
    public func vote<S: Collection>(syls: S, text: Substring, ctx: ContextVector) -> Float where S.Element == SylID {
        guard let list = table[Self.key(syls)] else { return 0 }
        let now = clock()
        var forText: Float = 0, against: Float = 0
        for e in list {
            let sim = max(0, e.ctx.dot(ctx))
            let w = e.weight * decay(e, now) * (0.25 + sim)   // some context-free prior, mostly context-driven
            if e.text == text { forText += w } else if w > 0 { against += w }
        }
        return log1p(max(0, forText)) - log1p(max(0, -forText)) - 0.5 * log1p(against)
    }

    /// Sum of votes over every span (up to `maxSpan` characters) of a candidate whose syllables are known.
    public func vote(charSyls: [SylID], text: String, ctx: ContextVector) -> Float {
        guard !table.isEmpty, charSyls.count == text.count else { return 0 }
        let chars = Array(text)
        var total: Float = 0
        for i in 0..<chars.count {
            for len in 1...min(maxSpan, chars.count - i) {
                let k = Self.key(charSyls[i..<(i + len)])
                if table[k] == nil { continue }
                total += vote(syls: charSyls[i..<(i + len)], text: Substring(String(chars[i..<(i + len)])), ctx: ctx)
            }
        }
        return total
    }

    public var isEmpty: Bool { table.isEmpty }
}
