import Foundation

public typealias SylID = UInt16

/// Toneless pinyin syllable inventory plus a letter trie used for parsing key strings.
public final class SyllableTable {
    public let spellings: [String]
    public let bytes: [[UInt8]]
    public let idOf: [String: SylID]

    // Letter trie. Node 0 is the root.
    struct Node {
        var child = [Int32](repeating: -1, count: 26)
        var terminal: Int32 = -1          // syllable id if a complete syllable ends here
        var prefixSet: [SylID] = []       // syllables that have this node's path as a prefix (inclusive)
        var depth: Int = 0
    }
    var nodes: [Node] = [Node()]

    /// Syllables sharing an initial ("zh", "b", ...). Used for abbreviated input.
    public private(set) var initialSets: [String: [SylID]] = [:]

    public static let initials: [String] = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]

    public init(spellings: [String]) {
        self.spellings = spellings
        self.bytes = spellings.map { Array($0.utf8) }
        var m: [String: SylID] = [:]
        for (i, s) in spellings.enumerated() { m[s] = SylID(i) }
        self.idOf = m
        for (i, b) in bytes.enumerated() { insert(b, id: SylID(i)) }
        // Common alternative spellings typed by users.
        for (alias, target) in [("nue", "nve"), ("lue", "lve")] {
            if let t = m[target], m[alias] == nil { insert(Array(alias.utf8), id: t) }
        }
        for n in nodes.indices { nodes[n].prefixSet.sort() }
        for ini in Self.initials {
            var set: [SylID] = []
            // like the system IME, a single z / c / s also stands for zh / ch / sh ("sm" → 什么)
            for (i, s) in spellings.enumerated() where s.hasPrefix(ini) { set.append(SylID(i)) }
            initialSets[ini] = set
        }
    }

    private func insert(_ b: [UInt8], id: SylID) {
        var n = 0
        nodes[0].prefixSet.append(id)
        for (d, c) in b.enumerated() {
            let k = Int(c) - 97
            if nodes[n].child[k] < 0 {
                var nn = Node()
                nn.depth = d + 1
                nodes.append(nn)
                nodes[n].child[k] = Int32(nodes.count - 1)
            }
            n = Int(nodes[n].child[k])
            if !nodes[n].prefixSet.contains(id) { nodes[n].prefixSet.append(id) }
        }
        nodes[n].terminal = Int32(id)
    }

    @inline(__always) func child(_ node: Int, _ letter: UInt8) -> Int {
        let k = Int(letter) &- 97
        if k < 0 || k >= 26 { return -1 }
        return Int(nodes[node].child[k])
    }

    public var count: Int { spellings.count }

    public func spelling(_ id: SylID) -> String { spellings[Int(id)] }

    /// Syllables whose spelling starts with `prefix`.
    public func completions(of prefix: [UInt8]) -> [SylID] {
        var n = 0
        for c in prefix {
            n = child(n, c)
            if n < 0 { return [] }
        }
        return nodes[n].prefixSet
    }
}
