import Foundation

public enum SylEdgeKind: UInt8 { case exact, partial, abbrev, typo }

/// One way to read keys[start..<end] as a syllable (or a set of syllables for partial / abbreviated input).
/// Plain data (no references) so it can be copied freely in hot loops.
public struct SylEdge {
    public var start: Int
    public var end: Int
    public var syl: SylID              // the syllable, when setRef < 0
    public var setRef: Int32 = -1      // index into SyllableGraph.sets for partial / abbreviated input
    public var cost: Float             // channel cost in nats
    public var kind: SylEdgeKind
    public var edit: EditKind? = nil
    public var typed: UInt8 = 0
    public var intended: UInt8 = 0
}

public struct GraphOptions {
    public var allowTypos = true
    public var allowAbbrev = true
    public var partialCostComplete: Float = 3.0   // last syllable is complete but could be extended (xian -> xiang)
    public var partialCostIncomplete: Float = 0.7 // last syllable is clearly unfinished (zhin -> zhi/ng...)
    public var abbrevCost: Float = 3.5
    /// consider non-adjacent substitutions / insertions (expensive, rarely right unless the user makes them)
    public var otherEdits = false
    public init() {}
}

/// Graph over key positions whose edges are syllable readings, including readings that undo one keyboard slip.
public struct SyllableGraph {
    public let keys: [UInt8]                 // letters only (apostrophes removed)
    public let rawIndex: [Int]               // keys[i] came from raw[rawIndex[i]]; rawIndex[n] = raw.count
    public let boundaries: Set<Int>          // forced syllable boundaries (from apostrophes)
    public var edgesFrom: [[SylEdge]]
    public var sets: [[SylID]] = []
    public var n: Int { keys.count }

    public func syls(_ e: SylEdge) -> [SylID] { e.setRef < 0 ? [e.syl] : sets[Int(e.setRef)] }

    @inline(__always) func same(_ a: SylEdge, _ b: SylEdge) -> Bool {
        if a.setRef < 0 || b.setRef < 0 { return a.setRef == b.setRef && a.syl == b.syl }
        return sets[Int(a.setRef)] == sets[Int(b.setRef)]
    }

    public init(raw: [UInt8], table: SyllableTable, typo: TypoModel, options: GraphOptions = GraphOptions()) {
        var k: [UInt8] = []
        var idx: [Int] = []
        var b = Set<Int>()
        for (i, c) in raw.enumerated() {
            if c == 39 { b.insert(k.count); continue }   // apostrophe
            if c >= 97 && c <= 122 { k.append(c); idx.append(i) }
        }
        idx.append(raw.count)
        keys = k; rawIndex = idx; boundaries = b
        edgesFrom = Array(repeating: [], count: k.count + 1)
        build(table: table, typo: typo, options: options)
    }

    @inline(__always) private func crosses(_ i: Int, _ j: Int) -> Bool {
        if boundaries.isEmpty { return false }
        for x in boundaries where x > i && x < j { return true }
        return false
    }

    private mutating func add(_ e: SylEdge) {
        // Deduplicate by (end, syls): keep the cheaper reading.
        for (ix, o) in edgesFrom[e.start].enumerated() where o.end == e.end && same(o, e) {
            if e.cost < o.cost { edgesFrom[e.start][ix] = e }
            return
        }
        edgesFrom[e.start].append(e)
    }

    private mutating func build(table: SyllableTable, typo: TypoModel, options: GraphOptions) {
        let n = keys.count
        for i in 0..<n {
            // exact syllables and the trailing partial syllable
            var node = 0
            var j = i
            while j < n {
                if j > i && boundaries.contains(j) { break }
                node = table.child(node, keys[j])
                if node < 0 { break }
                let t = table.nodes[node].terminal
                if t >= 0 { add(SylEdge(start: i, end: j + 1, syl: SylID(t), cost: 0, kind: .exact)) }
                if j + 1 == n {
                    var set = table.nodes[node].prefixSet
                    if t >= 0 { set.removeAll { $0 == SylID(t) } }
                    if !set.isEmpty {
                        let c = t >= 0 ? options.partialCostComplete : options.partialCostIncomplete
                        sets.append(set)
                        add(SylEdge(start: i, end: n, syl: set[0], setRef: Int32(sets.count - 1), cost: c, kind: .partial))
                    }
                }
                j += 1
            }
            // abbreviated syllables (initial only), not at the end of input (covered by partial)
            if options.allowAbbrev {
                for ini in SyllableTable.initials {
                    let ib = Array(ini.utf8)
                    let e = i + ib.count
                    if e >= n || crosses(i, e) { continue }
                    if Array(keys[i..<e]) == ib, let set = table.initialSets[ini], !set.isEmpty {
                        sets.append(set)
                        add(SylEdge(start: i, end: e, syl: set[0], setRef: Int32(sets.count - 1), cost: options.abbrevCost, kind: .abbrev))
                    }
                }
            }
            if options.allowTypos { fuzzy(from: i, table: table, typo: typo, otherEdits: options.otherEdits) }
        }
    }

    /// Enumerate syllables reachable from position `i` with exactly one keyboard edit.
    private mutating func fuzzy(from i: Int, table: SyllableTable, typo: TypoModel, otherEdits: Bool) {
        let n = keys.count
        var out: [SylEdge] = []
        let atSegmentStart = i == 0 || boundaries.contains(i)

        func emit(_ node: Int, _ pos: Int, _ kind: EditKind, _ typed: UInt8, _ intended: UInt8, _ cost: Float) {
            let t = table.nodes[node].terminal
            if t >= 0 && pos > i {
                out.append(SylEdge(start: i, end: pos, syl: SylID(t), cost: cost, kind: .typo, edit: kind, typed: typed, intended: intended))
            }
        }
        // after the edit: only exact matching continues
        func tail(_ node0: Int, _ pos0: Int, _ kind: EditKind, _ typed: UInt8, _ intended: UInt8, _ cost: Float) {
            var node = node0, pos = pos0
            emit(node, pos, kind, typed, intended, cost)
            while pos < n {
                if boundaries.contains(pos) && pos > i { break }
                node = table.child(node, keys[pos])
                if node < 0 { break }
                pos += 1
                emit(node, pos, kind, typed, intended, cost)
            }
        }
        // before the edit: exact prefix walk, trying each edit at each point
        var node = 0
        var pos = i
        while true {
            let depth = pos - i
            if pos < n && !(boundaries.contains(pos) && pos > i) {
                let x = keys[pos]
                // substitution
                for L in UInt8(97)...UInt8(122) where L != x {
                    let c = table.child(node, L)
                    if c < 0 { continue }
                    let kind: EditKind = Keyboard.isAdjacent(x, L) ? .substituteAdjacent : .substituteOther
                    if kind == .substituteOther && !otherEdits { continue }
                    tail(c, pos + 1, kind, x, L, typo.cost(kind, typed: x, intended: L))
                }
                // insertion: keys[pos] is extra
                if depth > 0 || atSegmentStart {
                    let prev: UInt8 = pos > 0 ? keys[pos - 1] : 0
                    let next: UInt8 = pos + 1 < n ? keys[pos + 1] : 0
                    let kind: EditKind = (x == prev || x == next) ? .insertRepeat
                        : ((Keyboard.isAdjacent(x, prev) || Keyboard.isAdjacent(x, next)) ? .insertAdjacent : .insertOther)
                    if (depth > 0 || pos + 1 < n) && (kind != .insertOther || otherEdits) {
                        tail(node, pos + 1, kind, x, 0, typo.cost(kind, typed: x))
                    }
                }
                // transposition
                if pos + 1 < n && keys[pos + 1] != x && !boundaries.contains(pos + 1) {
                    let c1 = table.child(node, keys[pos + 1])
                    if c1 >= 0 {
                        let c2 = table.child(c1, x)
                        if c2 >= 0 { tail(c2, pos + 2, .transpose, x, keys[pos + 1], typo.cost(.transpose, typed: x, intended: keys[pos + 1])) }
                    }
                }
            }
            // deletion: a key is missing before keys[pos] (not at the very end: partial input covers that)
            if pos < n {
                for L in UInt8(97)...UInt8(122) {
                    let c = table.child(node, L)
                    if c < 0 { continue }
                    tail(c, pos, .delete, 0, L, typo.cost(.delete, intended: L))
                }
            }
            if pos >= n { break }
            if boundaries.contains(pos) && pos > i { break }
            node = table.child(node, keys[pos])
            if node < 0 { break }
            pos += 1
        }
        for e in out {
            // an exact reading of the same span/syllable always wins
            if edgesFrom[i].contains(where: { $0.end == e.end && $0.setRef < 0 && $0.syl == e.syl && $0.kind == .exact }) { continue }
            add(e)
        }
    }

    /// Positions reachable from 0 (used to detect unparsable input).
    public func reachable() -> [Bool] {
        var r = [Bool](repeating: false, count: n + 1)
        r[0] = true
        for i in 0..<n where r[i] {
            for e in edgesFrom[i] { r[e.end] = true }
        }
        return r
    }
}
