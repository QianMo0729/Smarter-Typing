import Foundation

/// Kinds of keyboard slips the decoder can undo. Costs are in nats (−log probability relative to a clean keystroke).
public enum EditKind: Int, CaseIterable, Codable, Sendable {
    case insertRepeat = 0      // an extra key identical to a neighbour: "duooyin"
    case insertAdjacent        // an extra key adjacent on the keyboard to a neighbour
    case insertOther
    case delete                // a key is missing: "zhieng"
    case substituteAdjacent    // neighbouring key hit instead: "zhinrng"
    case substituteOther
    case transpose             // two keys swapped: "zhinegn"

    public var name: String {
        switch self {
        case .insertRepeat: return "ins-repeat"
        case .insertAdjacent: return "ins-adjacent"
        case .insertOther: return "ins-other"
        case .delete: return "delete"
        case .substituteAdjacent: return "sub-adjacent"
        case .substituteOther: return "sub-other"
        case .transpose: return "transpose"
        }
    }
}

public enum Keyboard {
    static let rows: [String] = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    static let offsets: [Double] = [0, 0.25, 0.75]
    /// adjacency[a][b] for a, b in 0..<26
    static let adjacency: [[Bool]] = {
        var pos = [(Double, Double)](repeating: (0, 0), count: 26)
        for (r, row) in rows.enumerated() {
            for (i, ch) in row.utf8.enumerated() { pos[Int(ch) - 97] = (Double(i) + offsets[r], Double(r)) }
        }
        var adj = [[Bool]](repeating: [Bool](repeating: false, count: 26), count: 26)
        for a in 0..<26 {
            for b in 0..<26 where a != b {
                let dx = pos[a].0 - pos[b].0, dy = pos[a].1 - pos[b].1
                adj[a][b] = (dx * dx + dy * dy).squareRoot() < 1.3
            }
        }
        return adj
    }()

    @inline(__always) public static func isAdjacent(_ a: UInt8, _ b: UInt8) -> Bool {
        let x = Int(a) &- 97, y = Int(b) &- 97
        guard x >= 0, x < 26, y >= 0, y < 26 else { return false }
        return adjacency[x][y]
    }
}

/// Noisy-channel model of the user's typing. Starts from generic rates and adapts to the edits
/// actually observed when the user corrects typos (see `Learner`).
public final class TypoModel: @unchecked Sendable {
    public static let defaultCosts: [EditKind: Float] = [
        .insertRepeat: 4.0, .insertAdjacent: 5.0, .insertOther: 7.0,
        .delete: 5.2, .substituteAdjacent: 5.0, .substituteOther: 7.5, .transpose: 4.8,
    ]
    public var base: [Float]
    /// Global scale applied to all edit costs (fitted together with the fusion weights).
    public var scale: Float = 1.0

    // personal statistics
    public private(set) var kindCounts = [Double](repeating: 0, count: EditKind.allCases.count)
    public private(set) var pairCounts: [Int: Double] = [:]     // key = kind*1000 + a*26 + b
    public private(set) var keystrokes: Double = 0
    private var costCache: [Float]
    private let lock = NSLock()

    public init() {
        base = EditKind.allCases.map { Self.defaultCosts[$0]! }
        costCache = base
    }

    /// Cost of an edit where the typed key is `typed` and the intended key is `intended` (0 when not applicable).
    @inline(__always) public func cost(_ kind: EditKind, typed: UInt8 = 0, intended: UInt8 = 0) -> Float {
        var c = costCache[kind.rawValue]
        if !pairCounts.isEmpty, typed != 0 || intended != 0 {
            let key = kind.rawValue * 1000 + (Int(typed) & 31) * 26 + (Int(intended) & 31)
            if let pc = pairCounts[key], pc > 0 {
                let kc = max(kindCounts[kind.rawValue], 1)
                // a pair the user repeatedly produces becomes cheaper (up to ~2 nats)
                c -= Float(min(2.0, log(1 + pc * 3 / kc + pc / 2)))
            }
        }
        return max(1.0, c) * scale
    }

    /// How habitual this particular slip is for the user (0 = never seen, → 1 = very common).
    public func familiarity(_ kind: EditKind, typed: UInt8, intended: UInt8) -> Float {
        let key = kind.rawValue * 1000 + (Int(typed) & 31) * 26 + (Int(intended) & 31)
        guard let pc = pairCounts[key] else { return 0 }
        return Float(pc / (pc + 3))
    }

    public static func classify(typed: [UInt8], intended: [UInt8]) -> [(EditKind, UInt8, UInt8)] {
        // Damerau alignment with at most 2 edits; returns the edit list (empty if identical or too far).
        let a = typed, b = intended
        let n = a.count, m = b.count
        if abs(n - m) > 2 { return [] }
        var d = [[Int]](repeating: [Int](repeating: 99, count: m + 1), count: n + 1)
        for i in 0...n { d[i][0] = i }
        for j in 0...m { d[0][j] = j }
        if n > 0 && m > 0 {
            for i in 1...n {
                for j in 1...m {
                    let sub = d[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)
                    var v = min(sub, min(d[i - 1][j] + 1, d[i][j - 1] + 1))
                    if i > 1, j > 1, a[i - 1] == b[j - 2], a[i - 2] == b[j - 1] { v = min(v, d[i - 2][j - 2] + 1) }
                    d[i][j] = v
                }
            }
        }
        if d[n][m] == 0 || d[n][m] > 2 { return [] }
        var edits: [(EditKind, UInt8, UInt8)] = []
        var i = n, j = m
        while i > 0 || j > 0 {
            if i > 0, j > 0, a[i - 1] == b[j - 1], d[i][j] == d[i - 1][j - 1] { i -= 1; j -= 1; continue }
            if i > 1, j > 1, a[i - 1] == b[j - 2], a[i - 2] == b[j - 1], d[i][j] == d[i - 2][j - 2] + 1 {
                edits.append((.transpose, a[i - 1], a[i - 2])); i -= 2; j -= 2; continue
            }
            if i > 0, j > 0, d[i][j] == d[i - 1][j - 1] + 1 {
                edits.append((Keyboard.isAdjacent(a[i - 1], b[j - 1]) ? .substituteAdjacent : .substituteOther, a[i - 1], b[j - 1]))
                i -= 1; j -= 1; continue
            }
            if i > 0, d[i][j] == d[i - 1][j] + 1 {
                let x = a[i - 1]
                let prev: UInt8 = i > 1 ? a[i - 2] : 0, next: UInt8 = i < n ? a[i] : 0
                let kind: EditKind = (x == prev || x == next) ? .insertRepeat : ((Keyboard.isAdjacent(x, prev) || Keyboard.isAdjacent(x, next)) ? .insertAdjacent : .insertOther)
                edits.append((kind, x, 0)); i -= 1; continue
            }
            if j > 0 { edits.append((.delete, 0, b[j - 1])); j -= 1; continue }
            break
        }
        return edits
    }

    // MARK: learning

    public func observeKeystrokes(_ n: Int) {
        lock.lock(); keystrokes += Double(n); lock.unlock()
    }

    public func observe(kind: EditKind, typed: UInt8, intended: UInt8, weight: Double = 1) {
        lock.lock()
        kindCounts[kind.rawValue] += weight
        let key = kind.rawValue * 1000 + (Int(typed) & 31) * 26 + (Int(intended) & 31)
        pairCounts[key, default: 0] += weight
        recompute()
        lock.unlock()
    }

    public func reset() {
        lock.lock()
        kindCounts = [Double](repeating: 0, count: EditKind.allCases.count)
        pairCounts = [:]
        keystrokes = 0
        recompute()
        lock.unlock()
    }

    /// Exponential forgetting so the model follows changing habits.
    public func decay(_ factor: Double) {
        lock.lock()
        for i in kindCounts.indices { kindCounts[i] *= factor }
        for k in pairCounts.keys { pairCounts[k]! *= factor }
        keystrokes *= factor
        recompute()
        lock.unlock()
    }

    private func recompute() {
        // Bayesian blend of the generic prior rate with the user's observed rate.
        let alpha = 30000.0   // pseudo keystrokes behind the prior
        for k in EditKind.allCases {
            let prior = exp(-Double(base[k.rawValue]))
            let rate = (kindCounts[k.rawValue] + alpha * prior) / (keystrokes + alpha)
            costCache[k.rawValue] = Float(-log(rate))
        }
    }

    // MARK: persistence
    struct State: Codable { var kindCounts: [Double]; var pairCounts: [Int: Double]; var keystrokes: Double }

    public func encode() -> Data? {
        lock.lock(); defer { lock.unlock() }
        return try? JSONEncoder().encode(State(kindCounts: kindCounts, pairCounts: pairCounts, keystrokes: keystrokes))
    }

    public func load(_ data: Data) {
        guard let s = try? JSONDecoder().decode(State.self, from: data) else { return }
        lock.lock()
        if s.kindCounts.count == kindCounts.count { kindCounts = s.kindCounts }
        pairCounts = s.pairCounts
        keystrokes = s.keystrokes
        recompute()
        lock.unlock()
    }
}
