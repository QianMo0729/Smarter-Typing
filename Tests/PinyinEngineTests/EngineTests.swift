import XCTest
@testable import PinyinEngine

final class EngineTests: XCTestCase {
    static let resDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("data/build")
    static var shared: Engine?

    func engine() throws -> Engine {
        if let e = Self.shared { return e }
        let e = try Engine(resources: EngineResources(dir: Self.resDir))
        Self.shared = e
        return e
    }

    func freshEngine() throws -> Engine { try Engine(resources: EngineResources(dir: Self.resDir)) }

    // MARK: parsing

    func testExactAndTypoSyllables() throws {
        let e = try engine()
        let g = SyllableGraph(raw: Array("duooyin".utf8), table: e.lexicon.syllables, typo: e.typo)
        let duo = e.lexicon.syllables.idOf["duo"]!
        let fromZero = g.edgesFrom[0]
        XCTAssertTrue(fromZero.contains { $0.kind == .exact && $0.end == 3 && $0.syl == duo })
        let slip = fromZero.first { $0.kind == .typo && $0.end == 4 && $0.syl == duo }
        XCTAssertNotNil(slip, "doubled o should be readable as duo with an extra key")
        XCTAssertEqual(slip?.edit, .insertRepeat)
    }

    func testApostropheForcesBoundary() throws {
        let e = try engine()
        let g = SyllableGraph(raw: Array("xi'an".utf8), table: e.lexicon.syllables, typo: e.typo, options: { var o = GraphOptions(); o.allowTypos = false; return o }())
        XCTAssertFalse(g.edgesFrom[0].contains { $0.end == 4 }, "xian must not be read across the apostrophe")
        XCTAssertEqual(g.rawIndex.count, 5)
    }

    func testClassifyEdits() {
        func kinds(_ a: String, _ b: String) -> [EditKind] { TypoModel.classify(typed: Array(a.utf8), intended: Array(b.utf8)).map { $0.0 } }
        XCTAssertEqual(kinds("duooyin", "duoyin"), [.insertRepeat])
        XCTAssertEqual(kinds("zhinegn", "zhineng"), [.transpose])
        XCTAssertEqual(kinds("zhieng", "zhineng"), [.delete])
        XCTAssertEqual(kinds("zhinrng", "zhineng"), [.substituteAdjacent])
        XCTAssertEqual(kinds("women", "women"), [])
        XCTAssertEqual(kinds("abcdefg", "xyz"), [])
    }

    // MARK: decoding

    func testContextFreeDecoding() throws {
        let e = try engine()
        let a = e.analyze(raw: "zhinengshurufa", context: "我想开发一个")
        XCTAssertEqual(a.full.first?.text, "智能输入法")
        let b = e.analyze(raw: "duooyin", context: "讨论汉字读音时，要注意")
        XCTAssertTrue(b.full.contains { $0.text == "多音" }, "the corrected reading must be offered")
        let probs = a.full.map { $0.prob }
        XCTAssertEqual(probs.reduce(0, +), 1, accuracy: 1e-3)
    }

    func testUnparsableTailIsKept() throws {
        let e = try engine()
        let a = e.analyze(raw: "nihaoqqq", context: "")
        XCTAssertTrue(a.covered < a.n || a.full.first?.text.hasPrefix("你好") == true)
    }

    // MARK: learning

    private func commit(_ e: Engine, _ learner: Learner, keys: String, context: String, text: String, time: Double, rank: Int = 0) {
        let a = e.analyze(raw: keys, context: context)
        let c = a.full.first { $0.text == text } ?? a.prefix.first { $0.text == text }!
        var charSyls: [SylID?] = [], words: [WordID] = [], wordSyls: [[SylID]] = []
        for s in c.segments { words.append(s.word); wordSyls.append(s.syls); charSyls += s.syls.map { Optional($0) } }
        learner.didCommit(CommitRecord(time: time, app: "test", context: context, raw: keys, text: c.text, charSyls: charSyls,
                                       words: words, wordSyls: wordSyls, chosenRank: rank, analysis: a,
                                       topProb: a.full.first?.prob ?? 0, usedTypoCorrection: c.typoCount > 0))
    }

    func testReselectionIsLearnedInContext() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        let ctx = "时间不够，我"
        // the user got 智能, deleted it right away and entered the same pinyin again choosing 只能
        let before = e.analyze(raw: "zhineng", context: ctx)
        let first = before.full.first!.text
        let other = first == "只能" ? "智能" : "只能"
        commit(e, learner, keys: "zhineng", context: ctx, text: first, time: 100)
        learner.didDeleteBackward(time: 101); learner.didDeleteBackward(time: 101.1)
        commit(e, learner, keys: "zhineng", context: ctx, text: other, time: 102, rank: 1)
        XCTAssertTrue(learner.events.contains { $0.kind == .reselection && $0.new == other }, "\(learner.events.map { $0.kind })")
        let after = e.analyze(raw: "zhineng", context: ctx)
        let m = after.full.first { $0.text == other }!.features.memory
        let mOld = after.full.first { $0.text == first }!.features.memory
        XCTAssertGreaterThan(m, mOld)
    }

    func testRewriteIsNotTreatedAsError() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        commit(e, learner, keys: "zhineng", context: "我想开发一个", text: "智能", time: 100)
        learner.didDeleteBackward(time: 104); learner.didDeleteBackward(time: 104.1)
        commit(e, learner, keys: "jiandan", context: "我想开发一个", text: "简单", time: 106)
        XCTAssertTrue(learner.events.contains { $0.kind == .rewrite })
        XCTAssertFalse(learner.events.contains { $0.kind == .reselection || $0.kind == .typoCorrection })
        // no negative memory was written for 智能
        let key = ContextMemory.key(e.analyze(raw: "zhineng", context: "").full.first { $0.text == "智能" }!.syllables)
        XCTAssertFalse((e.memory.table[key] ?? []).contains { $0.text == "智能" && $0.weight < 0 })
    }

    func testTypoCorrectionTeachesErrorModel() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        let before = e.typo.cost(.transpose, typed: UInt8(ascii: "o"), intended: UInt8(ascii: "u"))
        for i in 0..<6 {
            let t = Double(100 + i * 20)
            // typed "shou" (meant "shuo"), got 手, deleted and retyped "shuo" → 说
            commit(e, learner, keys: "shou", context: "我想", text: "手", time: t)
            learner.didDeleteBackward(time: t + 1)
            commit(e, learner, keys: "shuo", context: "我想", text: "说", time: t + 2)
        }
        XCTAssertTrue(learner.events.contains { $0.kind == .typoCorrection })
        let after = e.typo.cost(.transpose, typed: UInt8(ascii: "o"), intended: UInt8(ascii: "u"))
        XCTAssertLessThan(after, before)
    }

    func testAbbreviationRetypeIsACorrection() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        let a = e.analyze(raw: "sm", context: "分不清")
        let wrong = a.full.first { $0.text != "什么" }!.text
        commit(e, learner, keys: "sm", context: "分不清", text: wrong, time: 100)
        for i in 0..<wrong.count { learner.didDeleteBackward(time: 101 + Double(i) * 0.1) }
        commit(e, learner, keys: "shenme", context: "分不清", text: "什么", time: 103)
        XCTAssertTrue(learner.events.contains { $0.kind == .reselection && $0.new == "什么" }, "\(learner.events.map { ($0.kind, $0.old, $0.new) })")
        XCTAssertTrue(learner.keysAbbreviate("sm", [e.lexicon.syllables.idOf["shen"]!, e.lexicon.syllables.idOf["me"]!]))
        XCTAssertFalse(learner.keysAbbreviate("jd", [e.lexicon.syllables.idOf["shen"]!, e.lexicon.syllables.idOf["me"]!]))
    }

    func testKeptTextBuildsPersonalModel() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        commit(e, learner, keys: "shurufa", context: "智能", text: "输入法", time: 100)
        learner.tick(200)
        XCTAssertGreaterThan(e.userLM.total, 0)
        XCTAssertTrue(learner.events.contains { $0.kind == .kept })
    }

    // MARK: composer

    func testComposerCommitAndPartialSelection() throws {
        let e = try engine()
        let c = Composer(engine: e, neural: nil)
        c.begin(context: "我想开发一个", app: nil)
        for ch in "zhinengshurufa" { c.insert(String(ch)) }
        XCTAssertEqual(c.displayed.first?.text, "智能输入法")
        let r = c.commitHighlighted()
        XCTAssertEqual(r?.text, "智能输入法")
        XCTAssertFalse(c.isComposing)
        // partial: pick the first word, then finish the rest
        c.begin(context: "", app: nil)
        for ch in "zhinengshurufa" { c.insert(String(ch)) }
        let idx = c.displayed.firstIndex { $0.text == "智能" || $0.text == "只能" }!
        XCTAssertNil(c.select(index: idx))
        XCTAssertTrue(c.isComposing)
        XCTAssertEqual(c.raw, "shurufa")
        let r2 = c.commitHighlighted()
        XCTAssertTrue(r2?.text.hasSuffix("输入法") ?? false)
        XCTAssertEqual(r2?.record.chosenRank, -1)
    }

    func testComposerBackspaceRestoresParts() throws {
        let e = try engine()
        let c = Composer(engine: e, neural: nil)
        c.begin(context: "", app: nil)
        for ch in "nihao" { c.insert(String(ch)) }
        c.backspace()
        XCTAssertEqual(c.raw, "niha")
        for _ in 0..<4 { c.backspace() }
        XCTAssertFalse(c.isComposing)
    }

    func testComposerCaretEditsInTheMiddle() throws {
        let e = try engine()
        let c = Composer(engine: e, neural: nil)
        c.begin(context: "我想开发一个", app: nil)
        for ch in "zhnengshurufa" { c.insert(String(ch)) }       // the i of zhi was skipped
        XCTAssertEqual(c.caret, 13)
        XCTAssertTrue(c.moveCaret(-11))                          // zh|nengshurufa
        XCTAssertEqual(c.caret, 2)
        c.insert("i")
        XCTAssertEqual(c.raw, "zhinengshurufa")
        XCTAssertEqual(c.caret, 3)
        XCTAssertEqual(c.displayed.first?.text, "智能输入法", "candidates cover the whole input, not just the part before the caret")
        XCTAssertEqual(c.markedText, "zhi neng shu ru fa")
        XCTAssertEqual(c.markedCaret, 3, "the caret is drawn after zhi, before the syllable space")
        XCTAssertTrue(c.moveCaret(1))
        XCTAssertEqual(c.markedCaret, 5, "zhi n|eng")
        c.backspace()                                            // deletes the n before the caret
        XCTAssertEqual(c.raw, "zhiengshurufa")
        XCTAssertEqual(c.caret, 3)
        XCTAssertTrue(c.deleteForward())                         // and the e after it
        XCTAssertEqual(c.raw, "zhingshurufa")
        XCTAssertEqual(c.caret, 3)
        c.moveCaret(toEnd: true)
        XCTAssertFalse(c.deleteForward(), "nothing after the caret at the end")
        c.moveCaret(-9)
        c.insert("e")
        XCTAssertEqual(c.raw, "zhiengshurufa")
        c.moveCaret(toEnd: false)
        XCTAssertEqual(c.caret, 0)
        XCTAssertFalse(c.moveCaret(-1))
        XCTAssertFalse(c.backspace(), "nothing before the caret and nothing converted")
        XCTAssertEqual(c.raw, "zhiengshurufa")
        c.moveCaret(toEnd: true)
        XCTAssertEqual(c.markedCaret, (c.markedText as NSString).length)
        let r = c.commitHighlighted()
        XCTAssertNotNil(r)
        XCTAssertFalse(c.isComposing)
        XCTAssertEqual(c.caret, 0)
    }

    func testComposerCaretAfterPartialSelection() throws {
        let e = try engine()
        let c = Composer(engine: e, neural: nil)
        c.begin(context: "", app: nil)
        for ch in "zhinengshurufa" { c.insert(String(ch)) }
        c.moveCaret(-3)
        let idx = c.displayed.firstIndex { $0.text == "智能" || $0.text == "只能" }!
        XCTAssertNil(c.select(index: idx))
        XCTAssertEqual(c.raw, "shurufa")
        XCTAssertEqual(c.caret, 7, "after a partial selection the caret is at the end of the remaining pinyin")
        XCTAssertEqual(c.markedCaret, (c.markedText as NSString).length)
        c.moveCaret(toEnd: false)
        XCTAssertTrue(c.backspace(), "backspace at the start of the pinyin unconverts the last part")
        XCTAssertTrue(c.parts.isEmpty)
        XCTAssertEqual(c.raw, "zhinengshurufa")
        XCTAssertEqual(c.caret, 7)
    }
}
