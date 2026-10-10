import XCTest
@testable import PinyinEngine

/// Names and other words the lexicon does not have must be learned from the first time they are put together.
final class NameLearningTests: XCTestCase {
    func freshEngine() throws -> Engine { try Engine(resources: EngineResources(dir: EngineTests.resDir)) }

    /// Type `keys` and pick `pieces` one after another from the candidate list, the way a person builds a name
    /// the input method does not know. A piece that is not listed is taken from the top candidate's start.
    @discardableResult
    func assemble(_ e: Engine, _ learner: Learner, keys: String, pieces: [String], context: String = "", time: Double) -> String? {
        let c = Composer(engine: e, neural: nil)
        c.begin(context: context, app: "test")
        for ch in keys { c.insert(String(ch)) }
        var result: CommitResult?
        for p in pieces {
            guard let i = c.displayed.firstIndex(where: { $0.text == p }) else { XCTFail("\(p) is not offered for \(c.raw)"); return nil }
            result = c.select(index: i)
        }
        guard var r = result?.record else { XCTFail("\(pieces) did not use up \(keys)"); return nil }
        r.time = time
        learner.didCommit(r)
        return r.text
    }

    func first(_ e: Engine, _ keys: String, context: String = "") -> String? {
        e.displayList(e.analyze(raw: keys, context: context)).first?.text
    }

    func testAssembledNameIsOfferedNextTime() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        for (keys, pieces) in [("linjiwen", ["林", "霁", "雯"]), ("guyining", ["顾", "祎", "宁"])] {
            let name = pieces.joined()
            XCTAssertNotEqual(first(e, keys), name)
            XCTAssertEqual(assemble(e, learner, keys: keys, pieces: pieces, time: 100), name)
            // right away, in any context, and it survives the keep window
            XCTAssertEqual(first(e, keys), name)
            XCTAssertEqual(first(e, keys, context: "这个问题你可以直接去问我们组的同事，她叫"), name)
            learner.tick(200)
            XCTAssertEqual(first(e, keys), name)
        }
        // as a word among others, unfinished, and with a slip
        XCTAssertEqual(first(e, "wohelinjiwenqule"), "我和林霁雯去了")
        XCTAssertEqual(first(e, "linjiw"), "林霁雯")
        XCTAssertEqual(first(e, "linjiwne"), "林霁雯")
        // accepting it with space is ordinary use from now on
        let c = Composer(engine: e, neural: nil)
        c.begin(context: "", app: "test")
        for ch in "linjiwen" { c.insert(String(ch)) }
        let r = c.commitHighlighted()
        XCTAssertEqual(r?.text, "林霁雯")
        XCTAssertEqual(r?.record.chosenRank, 0)
        XCTAssertEqual(r?.record.charSyls.count, 3)
    }

    /// The engine can spell 王思雨 from characters it offers, it just reads the pinyin as 王死于.
    func testNameOfCommonCharactersIsLearnedToo() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        XCTAssertNotEqual(first(e, "wangsiyu"), "王思雨")
        assemble(e, learner, keys: "wangsiyu", pieces: ["王", "思", "雨"], time: 100)
        XCTAssertEqual(first(e, "wangsiyu"), "王思雨")
        XCTAssertEqual(first(e, "wangsiyu", context: "明天的会议由"), "王思雨")
    }

    /// A rare character dug out from deep in the list comes up from then on.
    func testRareCharacterMovesUp() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        func index() -> Int { e.displayList(e.analyze(raw: "ji", context: "")).firstIndex { $0.text == "霁" }! }
        let before = index()
        XCTAssertGreaterThan(before, 30)
        assemble(e, learner, keys: "ji", pieces: ["霁"], time: 100)
        XCTAssertLessThan(index(), 5)
        // in running text the usual readings stay
        XCTAssertEqual(first(e, "wojide"), "我记得")
        XCTAssertEqual(first(e, "jidian"), "几点")
    }

    func testNameInsideSentenceIsLearnedAlone() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        XCTAssertEqual(assemble(e, learner, keys: "jintianlinjiwenlaile", pieces: ["今天", "林", "霁", "雯", "来了"], time: 100), "今天林霁雯来了")
        XCTAssertNotNil(e.userWords.id(ofText: "林霁雯"), "\(e.userWords.all.map { $0.text })")
        XCTAssertNil(e.userWords.id(ofText: "今天林霁雯来了"))
        XCTAssertEqual(first(e, "linjiwen"), "林霁雯")
        XCTAssertEqual(first(e, "linjiwenshuo", context: "昨天"), "林霁雯说")
    }

    func testOrdinaryChoicesDoNotBecomeWords() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        // a homophone both of which the engine knows: context decides, not a user word
        let a = e.analyze(raw: "zhineng", context: "时间不够，我")
        let other = a.full.first!.text == "只能" ? "智能" : "只能"
        assemble(e, learner, keys: "zhineng", pieces: [other], context: "时间不够，我", time: 100)
        // cutting a sentence the engine already reads correctly into pieces
        assemble(e, learner, keys: "zhinengshurufa", pieces: ["智能", "输入法"], context: "我想开发一个", time: 130)
        // another phrase of words it knows
        XCTAssertEqual(first(e, "tadehua"), "他的话")
        assemble(e, learner, keys: "tadehua", pieces: ["她的", "画"], context: "我很喜欢", time: 160)
        learner.tick(300)
        XCTAssertTrue(e.userWords.isEmpty, "\(e.userWords.all.map { $0.text })")
    }

    func testDeletedNameIsNotKept() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        assemble(e, learner, keys: "linjiwen", pieces: ["林", "玑", "雯"], time: 100)
        XCTAssertEqual(first(e, "linjiwen"), "林玑雯")
        // wrong character: delete it all and build the right one
        for i in 0..<3 { learner.didDeleteBackward(time: 101 + Double(i) * 0.1) }
        assemble(e, learner, keys: "linjiwen", pieces: ["林", "霁", "雯"], time: 103)
        XCTAssertNil(e.userWords.id(ofText: "林玑雯"))
        XCTAssertEqual(first(e, "linjiwen"), "林霁雯")
    }

    func testDeletingWhatFollowsKeepsTheName() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        assemble(e, learner, keys: "linjiwenshuo", pieces: ["林", "霁", "雯", "说"], context: "昨天", time: 100)
        learner.didDeleteBackward(time: 101)   // only 说 goes
        assemble(e, learner, keys: "wen", pieces: ["问"], context: "昨天林霁雯", time: 102)
        XCTAssertEqual(first(e, "linjiwen"), "林霁雯")
        // without the surname too, but never with the verb
        XCTAssertNotNil(e.userWords.id(ofText: "霁雯"))
        XCTAssertNil(e.userWords.id(ofText: "林霁雯说"))
    }

    func testOfferedWordThatIsReplacedFades() throws {
        let e = try freshEngine()
        let learner = Learner(engine: e)
        assemble(e, learner, keys: "linjiwen", pieces: ["林", "玑", "雯"], time: 100)
        learner.didBreak(time: 200)
        // later it is offered, accepted by reflex, deleted and replaced by the name that was meant
        assemble(e, learner, keys: "linjiwen", pieces: ["林玑雯"], time: 300)
        for i in 0..<3 { learner.didDeleteBackward(time: 301 + Double(i) * 0.1) }
        assemble(e, learner, keys: "linjiwen", pieces: ["林", "霁", "雯"], time: 303)
        XCTAssertEqual(first(e, "linjiwen"), "林霁雯")
        XCTAssertNil(e.userWords.id(ofText: "林玑雯"))
    }

    func testUserWordsAreSavedAndRestored() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("smartpinyin-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let e = try freshEngine()
        let learner = Learner(engine: e)
        assemble(e, learner, keys: "linjiwen", pieces: ["林", "霁", "雯"], time: 100)
        learner.flush(now: 200)
        PersonalStore(dir: dir, engine: e).save(sync: true)
        let e2 = try freshEngine()
        PersonalStore(dir: dir, engine: e2).load()
        XCTAssertEqual(first(e2, "linjiwen"), "林霁雯")
        // ids stay what they were, so personal counts keep pointing at the same word
        XCTAssertEqual(e2.userWords.id(ofText: "林霁雯"), e.userWords.id(ofText: "林霁雯"))
        e2.resetPersonal()
        XCTAssertNotEqual(first(e2, "linjiwen"), "林霁雯")
    }

    /// Personal data from before user words existed: the names are in the memory, where they could never be offered.
    func testNamesAreAdoptedFromOlderData() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("smartpinyin-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let e = try freshEngine()
        let T = e.lexicon.syllables
        func syls(_ s: String) -> [SylID] { s.split(separator: " ").map { T.idOf[String($0)]! } }
        let ctx = ContextVector(text: "我和", app: "test")
        for (text, py) in [("顾祎宁", "gu yi ning"), ("顾祎", "gu yi"), ("祎宁", "yi ning"), ("祎", "yi"), ("只能", "zhi neng"), ("的话", "de hua")] {
            e.memory.add(syls: syls(py), text: text, ctx: ctx, weight: 0.5)
        }
        e.memory.add(syls: syls("zhi neng"), text: "智能", ctx: ctx, weight: -0.7)
        let store = PersonalStore(dir: dir, engine: e)
        store.save(sync: true)
        // make it look like a file written by the previous version
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: store.file)) as! [String: Any]
        json["userWords"] = nil
        try JSONSerialization.data(withJSONObject: json).write(to: store.file)
        let e2 = try freshEngine()
        PersonalStore(dir: dir, engine: e2).load()
        XCTAssertEqual(e2.userWords.all.map { $0.text }, ["顾祎宁"])
        XCTAssertEqual(first(e2, "guyining"), "顾祎宁")
    }
}
