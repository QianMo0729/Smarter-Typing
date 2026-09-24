import Foundation

/// Full-width punctuation for Chinese mode.
struct Punctuation {
    private var doubleQuoteOpen = true
    private var singleQuoteOpen = true

    mutating func map(_ ch: Character) -> String? {
        switch ch {
        case ",": return "，"
        case ".": return "。"
        case "?": return "？"
        case "!": return "！"
        case ":": return "："
        case ";": return "；"
        case "\\": return "、"
        case "(": return "（"
        case ")": return "）"
        case "<": return "《"
        case ">": return "》"
        case "[": return "【"
        case "]": return "】"
        case "^": return "……"
        case "_": return "——"
        case "~": return "～"
        case "$": return "￥"
        case "\"":
            defer { doubleQuoteOpen.toggle() }
            return doubleQuoteOpen ? "“" : "”"
        case "'":
            defer { singleQuoteOpen.toggle() }
            return singleQuoteOpen ? "‘" : "’"
        default: return nil
        }
    }
}
