import Foundation

/// Helpers to turn free text (the text before the cursor) into word ids for the n-gram / personal models.
public struct ContextWords {
    let lexicon: Lexicon
    public let bos: WordID
    public let punct: [Character: WordID]
    public let eng: WordID
    public let num: WordID
    public let unk: WordID

    public init(lexicon: Lexicon) {
        self.lexicon = lexicon
        bos = lexicon.id(of: "<s>") ?? 0
        eng = lexicon.id(of: "<eng>") ?? 0
        num = lexicon.id(of: "<num>") ?? 0
        unk = lexicon.id(of: "<unk>") ?? 0
        var p: [Character: WordID] = [:]
        let map: [Character: String] = [",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；", "(": "（", ")": "）", "\"": "“"]
        for s in ["，", "。", "？", "！", "、", "：", "；", "“", "”", "‘", "’", "（", "）", "《", "》", "…", "—", "·", "～"] {
            if let id = lexicon.id(of: s) { p[Character(s)] = id }
        }
        for (a, b) in map { if let id = lexicon.id(of: b) { p[a] = id } }
        punct = p
    }

    /// Segment the tail of `text` into words by backward maximum matching (good enough for history state).
    public func tailWords(_ text: String, maxWords: Int = 4) -> [WordID] {
        let chars = Array(text.suffix(24))
        var out: [WordID] = []
        var j = chars.count
        while j > 0 && out.count < maxWords {
            let c = chars[j - 1]
            if c == "\n" { if out.isEmpty { out.append(bos) }; break }
            if let p = punct[c] { out.append(p); j -= 1; continue }
            if c.isASCII && (c.isLetter) {
                var i = j - 1
                while i > 0 && chars[i - 1].isASCII && chars[i - 1].isLetter { i -= 1 }
                out.append(eng); j = i; continue
            }
            if c.isNumber {
                var i = j - 1
                while i > 0 && chars[i - 1].isNumber { i -= 1 }
                out.append(num); j = i; continue
            }
            if c.isWhitespace { j -= 1; continue }
            var found = false
            for len in stride(from: min(8, j), through: 1, by: -1) {
                let s = String(chars[(j - len)..<j])
                if let id = lexicon.id(of: s) { out.append(id); j -= len; found = true; break }
            }
            if !found { out.append(unk); j -= 1 }
        }
        if out.isEmpty { out.append(bos) }
        return out.reversed()
    }

    public func lastWord(_ text: String) -> WordID {
        if text.isEmpty { return bos }
        return tailWords(text, maxWords: 1).last ?? bos
    }
}
