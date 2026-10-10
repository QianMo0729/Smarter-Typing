import Foundation

public typealias WordID = Int32

enum BinaryError: Error { case badMagic(String), truncated(String) }

/// Read-only view of a memory-mapped binary file with 8-byte aligned sections.
struct BinaryReader {
    let data: Data
    let base: UnsafeRawPointer
    var offset = 0

    init(url: URL, magic: String) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        base = data.withUnsafeBytes { $0.baseAddress! }   // mapping lives as long as `data`
        let m = String(decoding: data.prefix(4), as: UTF8.self)
        if m != magic { throw BinaryError.badMagic(url.lastPathComponent) }
        offset = 4
    }

    mutating func u32() -> UInt32 {
        let v = base.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        offset += 4
        return v
    }

    mutating func align() { offset = (offset + 7) & ~7 }

    mutating func section<T>(_ type: T.Type, count: Int) -> UnsafeBufferPointer<T> {
        align()
        let p = (base + offset).assumingMemoryBound(to: T.self)
        offset += count * MemoryLayout<T>.stride
        return UnsafeBufferPointer(start: p, count: count)
    }
}

/// Pinyin lexicon: surfaces (distinct words) and a trie keyed by syllable sequences.
public final class Lexicon: @unchecked Sendable {
    private let reader: BinaryReader
    public let syllables: SyllableTable
    public let surfaceCount: Int
    let surfOffsets: UnsafeBufferPointer<UInt32>
    let surfBlob: UnsafeBufferPointer<UInt8>
    public let surfChars: UnsafeBufferPointer<UInt8>
    public let prior: UnsafeBufferPointer<Float>
    let nodeChild: UnsafeBufferPointer<UInt32>
    let childSyl: UnsafeBufferPointer<UInt16>
    let childNode: UnsafeBufferPointer<UInt32>
    let nodeWord: UnsafeBufferPointer<UInt32>
    let wordSurf: UnsafeBufferPointer<UInt32>
    let wordReading: UnsafeBufferPointer<Float>
    private var surfaceIndex: [String: WordID]? = nil
    private let indexLock = NSLock()
    /// Words taught by the user. Their ids continue above the lexicon's and resolve through the same calls.
    public let user = UserWords()

    public init(url: URL) throws {
        var r = try BinaryReader(url: url, magic: "STLX")
        _ = r.u32()
        let nSyl = Int(r.u32()), nSurf = Int(r.u32()), nNodes = Int(r.u32()), nEdges = Int(r.u32()), nWords = Int(r.u32()), blobLen = Int(r.u32()), sylBlobLen = Int(r.u32())
        let sylBlob = r.section(UInt8.self, count: sylBlobLen)
        let spellings = String(decoding: sylBlob, as: UTF8.self).split(separator: "\n").map(String.init)
        precondition(spellings.count == nSyl)
        syllables = SyllableTable(spellings: spellings)
        surfaceCount = nSurf
        surfOffsets = r.section(UInt32.self, count: nSurf + 1)
        surfBlob = r.section(UInt8.self, count: blobLen)
        surfChars = r.section(UInt8.self, count: nSurf)
        prior = r.section(Float.self, count: nSurf)
        nodeChild = r.section(UInt32.self, count: nNodes + 1)
        childSyl = r.section(UInt16.self, count: nEdges)
        childNode = r.section(UInt32.self, count: nEdges)
        nodeWord = r.section(UInt32.self, count: nNodes + 1)
        wordSurf = r.section(UInt32.self, count: nWords)
        wordReading = r.section(Float.self, count: nWords)
        reader = r
    }

    @inline(__always) public func surface(_ w: WordID) -> String {
        if UserWords.isUser(w) { return user.entry(w)?.text ?? "" }
        let a = Int(surfOffsets[Int(w)]), b = Int(surfOffsets[Int(w) + 1])
        return String(decoding: UnsafeBufferPointer(rebasing: surfBlob[a..<b]), as: UTF8.self)
    }

    @inline(__always) public func charCount(_ w: WordID) -> Int {
        UserWords.isUser(w) ? user.entry(w)?.syls.count ?? 0 : Int(surfChars[Int(w)])
    }

    /// Child of `node` along syllable `s`, or -1.
    @inline(__always) public func child(_ node: Int, _ s: SylID) -> Int {
        var lo = Int(nodeChild[node]), hi = Int(nodeChild[node + 1])
        while lo < hi {
            let mid = (lo + hi) >> 1
            let v = childSyl[mid]
            if v == s { return Int(childNode[mid]) }
            if v < s { lo = mid + 1 } else { hi = mid }
        }
        return -1
    }

    /// Iterate children of `node` whose syllable is in the sorted set `syls`.
    @inline(__always) public func forEachChild(_ node: Int, in syls: [SylID], _ body: (SylID, Int) -> Void) {
        if syls.count == 1 {
            let c = child(node, syls[0])
            if c >= 0 { body(syls[0], c) }
            return
        }
        var i = Int(nodeChild[node]); let e = Int(nodeChild[node + 1])
        var j = 0
        while i < e && j < syls.count {
            let a = childSyl[i], b = syls[j]
            if a == b { body(a, Int(childNode[i])); i += 1; j += 1 }
            else if a < b { i += 1 } else { j += 1 }
        }
    }

    @inline(__always) public func hasChildren(_ node: Int) -> Bool { nodeChild[node + 1] > nodeChild[node] }

    @inline(__always) public func wordRange(_ node: Int) -> Range<Int> { Int(nodeWord[node])..<Int(nodeWord[node + 1]) }
    @inline(__always) public func wordSurface(_ i: Int) -> WordID { WordID(wordSurf[i]) }
    @inline(__always) public func wordReadingLogP(_ i: Int) -> Float { wordReading[i] }

    /// Look up the trie node for an exact syllable sequence.
    public func node(for syls: [SylID]) -> Int {
        var n = 0
        for s in syls {
            n = child(n, s)
            if n < 0 { return -1 }
        }
        return n
    }

    /// Readings (syllable sequences) of a surface. Slow path (linear scan of a trie walk guided by nothing);
    /// used only for learning / tools, so it relies on an index built lazily.
    public func id(of surface: String) -> WordID? {
        indexLock.lock(); defer { indexLock.unlock() }
        if surfaceIndex == nil {
            var m = [String: WordID](minimumCapacity: surfaceCount)
            for i in 0..<surfaceCount { m[self.surface(WordID(i))] = WordID(i) }
            surfaceIndex = m
        }
        return surfaceIndex![surface] ?? user.id(ofText: surface)
    }
}
