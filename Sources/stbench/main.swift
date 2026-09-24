import Foundation
import PinyinEngine

// stbench — command line harness for the engine: decoding, evaluation, feature dumps, simulations.

var args = Array(CommandLine.arguments.dropFirst())
let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

func opt(_ name: String) -> String? {
    if let i = args.firstIndex(of: name), i + 1 < args.count { return args[i + 1] }
    return nil
}
func flag(_ name: String) -> Bool { args.contains(name) }
func ms(_ t: UInt64) -> String { String(format: "%.2fms", Double(t) / 1e6) }
func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

let resDir = URL(fileURLWithPath: opt("--res") ?? root.appendingPathComponent("data/build").path)
let modelPath = opt("--model") ?? root.appendingPathComponent("models/gguf/Qwen3-0.6B-Base-q8_0.gguf").path
let engine = try Engine(resources: EngineResources(dir: resDir))
if let w = opt("--weights") { engine.loadWeights(URL(fileURLWithPath: w)) }
if let g = opt("--guard") { engine.typoGuard = Float(g)! }
if flag("--no-typo") { engine.graphOptions.allowTypos = false }
let lex = engine.lexicon
if let b = opt("--beam") { engine.decoder.options.beam = Int(b)! }
if let b = opt("--wps") { engine.latticeOptions.wordsPerSpan = Int(b)! }
if let b = opt("--wpn") { engine.latticeOptions.wordsPerNode = Int(b)! }
if flag("--no-other") { engine.graphOptions.otherEdits = false }

func makeNN() -> NeuralScorer? {
    if flag("--no-nn") { return nil }
    var cfg = NeuralScorer.Config()
    if flag("--cpu") { cfg.gpu = false }
    setenv("GGML_METAL_TENSOR_DISABLE", "1", 1)
    return try? NeuralScorer(modelPath: modelPath, config: cfg)
}

func describe(_ c: Candidate) -> String {
    c.segments.map { lex.surface($0.word) + "/" + $0.syls.map { lex.syllables.spelling($0) }.joined(separator: "'") + ($0.typos > 0 ? "*" : "") }.joined(separator: " ")
}

struct Example: Codable {
    var src: String
    var context: String
    var target: String
    var keys: String
    var clean_keys: String
    var typo: TypoInfo?
    struct TypoInfo: Codable { var kind: String; var pos: Int }
}

func loadSet(_ path: String) -> [Example] {
    let text = try! String(contentsOfFile: path, encoding: .utf8)
    let dec = JSONDecoder()
    return text.split(separator: "\n").compactMap { try? dec.decode(Example.self, from: Data($0.utf8)) }
}

let cmd = args.first ?? "help"

switch cmd {
case "decode":
    // stbench decode <pinyin> [context] [--nn]
    let input = args[1]
    let context = args.count > 2 && !args[2].hasPrefix("--") ? args[2] : ""
    var a = engine.analyze(raw: input, context: context)
    a = engine.analyze(raw: input, context: context)
    print(String(format: "fast %.2fms  covered %d/%d  ctxWord=%@", a.fastMs, a.covered, a.n, lex.surface(a.contextWord)))
    if flag("--nn"), let nn = makeNN() {
        nn.setContext(context)
        _ = nn.score(a.options)
        let t = now()
        let s = nn.score(a.options)
        print("nn \(ms(now() - t)) (\(nn.lastScoreTokens) tokens)")
        engine.applyNeural(&a, scores: s)
    }
    for c in a.full.prefix(Int(opt("--show") ?? "") ?? 12) {
        let f = c.features
        print(String(format: "  %@  p=%.3f s=%.2f lm=%.2f nn=%.2f typo=%.1f shape=%.1f pers=%.2f mem=%.2f  %@", c.text, c.prob, c.score, f.lm, f.nn, f.typoCost, f.shapeCost, f.personal, f.memory, describe(c)))
    }
    print("  list:", engine.displayList(a).prefix(20).map { $0.text }.joined(separator: " "))

case "eval":
    // stbench eval <set.jsonl> [--nn] [--limit N] [--dump out.jsonl] [--verbose]
    let set = loadSet(args[1])
    let limit = Int(opt("--limit") ?? "") ?? set.count
    let useNN = flag("--nn")
    let nn = useNN ? makeNN() : nil
    if useNN && nn == nil { print("cannot load model"); exit(1) }
    let dumpPath = opt("--dump")
    var dump: FileHandle? = nil
    if let dumpPath {
        FileManager.default.createFile(atPath: dumpPath, contents: nil)
        dump = FileHandle(forWritingAtPath: dumpPath)
    }
    var n = 0, top1 = 0, top3 = 0, inList = 0
    var cleanN = 0, cleanTop1 = 0, falseCorr = 0
    var typoN = 0, typoTop1 = 0
    var byKind: [String: (Int, Int)] = [:]
    var fastMs: [Double] = [], nnMs: [Double] = [], nnTok: [Int] = []
    var bySrc: [String: (Int, Int)] = [:]
    for ex in set.prefix(limit) {
        var a = engine.analyze(raw: ex.keys, context: ex.context)
        fastMs.append(a.fastMs)
        if let nn {
            nn.setContext(ex.context)
            let t = now()
            let s = nn.score(a.options)
            nnMs.append(Double(now() - t) / 1e6); nnTok.append(nn.lastScoreTokens)
            engine.applyNeural(&a, scores: s)
        }
        n += 1
        let texts = a.full.map { $0.text }
        let ok = texts.first == ex.target
        if ok { top1 += 1 }
        if texts.prefix(3).contains(ex.target) { top3 += 1 }
        if texts.contains(ex.target) { inList += 1 }
        bySrc[ex.src, default: (0, 0)].0 += 1
        if ok { bySrc[ex.src, default: (0, 0)].1 += 1 }
        if let t = ex.typo {
            typoN += 1; if ok { typoTop1 += 1 }
            byKind[t.kind, default: (0, 0)].0 += 1
            if ok { byKind[t.kind, default: (0, 0)].1 += 1 }
        } else {
            cleanN += 1; if ok { cleanTop1 += 1 }
            if let f = a.full.first, f.typoCount > 0, !ok { falseCorr += 1 }
        }
        if flag("--verbose") && !ok {
            print("✗ [\(ex.src)] …\(ex.context.suffix(20)) | \(ex.keys) → \(texts.first ?? "-") (want \(ex.target))\(ex.typo != nil ? " typo:" + ex.typo!.kind : "") inList=\(texts.contains(ex.target))")
        }
        if let dump {
            // one line per example: label index and feature vectors of all full candidates
            let label = texts.firstIndex(of: ex.target) ?? -1
            var obj: [String: Any] = ["label": label, "typo": ex.typo != nil, "src": ex.src]
            obj["f"] = a.full.map { $0.features.vector.map { Double($0) } }
            obj["t"] = a.full.map { $0.typoCount }
            if flag("--with-text") { obj["context"] = ex.context; obj["cands"] = texts }
            let d = try! JSONSerialization.data(withJSONObject: obj)
            dump.write(d); dump.write("\n".data(using: .utf8)!)
        }
    }
    func pct(_ a: Int, _ b: Int) -> String { b == 0 ? "-" : String(format: "%.1f%%", 100 * Double(a) / Double(b)) }
    func p(_ xs: [Double], _ q: Double) -> String {
        if xs.isEmpty { return "-" }
        let s = xs.sorted(); return String(format: "%.1f", s[min(s.count - 1, Int(Double(s.count) * q))])
    }
    print("examples \(n)  top1 \(pct(top1, n))  top3 \(pct(top3, n))  inList \(pct(inList, n))")
    print("clean \(cleanN): top1 \(pct(cleanTop1, cleanN))  false-correction \(pct(falseCorr, cleanN))")
    print("typo  \(typoN): recovered \(pct(typoTop1, typoN))  " + byKind.sorted { $0.key < $1.key }.map { "\($0.key) \(pct($0.value.1, $0.value.0))" }.joined(separator: "  "))
    print("by source: " + bySrc.sorted { $0.key < $1.key }.map { "\($0.key) \(pct($0.value.1, $0.value.0))" }.joined(separator: "  "))
    print("fast ms p50 \(p(fastMs, 0.5)) p95 \(p(fastMs, 0.95)) p99 \(p(fastMs, 0.99))   nn ms p50 \(p(nnMs, 0.5)) p95 \(p(nnMs, 0.95)) p99 \(p(nnMs, 0.99))  tokens p50 \(nnTok.sorted().dropFirst(nnTok.count / 2).first ?? 0)")
    try? dump?.close()

case "personalize":
    // stbench personalize <user> [--limit N] [--habit uo] : learning curve on a user's own text
    let user = args[1]
    let stream = loadExamples(root.appendingPathComponent("data/eval/users/\(user).stream.jsonl").path)
    let test = Array(loadExamples(root.appendingPathComponent("data/eval/users/\(user).test.jsonl").path).prefix(Int(opt("--test") ?? "") ?? 800))
    let general = Array(loadExamples(root.appendingPathComponent("data/eval/test.jsonl").path).prefix(Int(opt("--general") ?? "") ?? 800))
    let limit = Int(opt("--limit") ?? "") ?? stream.count
    let nn = makeNN()
    let useHabit = opt("--habit") == "uo"
    // personal habit: often types "ou" where "uo" is meant (shuo→shou, duo→dou): both are valid pinyin
    let habit: (String, inout SplitMix) -> String = { keys, r in
        guard keys.contains("uo") else { return keys }
        var out = "", chars = Array(keys), i = 0
        while i < chars.count {
            if i + 1 < chars.count && chars[i] == "u" && chars[i + 1] == "o" && r.nextDouble() < 0.5 { out += "ou"; i += 2; continue }
            out.append(chars[i]); i += 1
        }
        return out
    }
    func pct(_ a: Int, _ b: Int) -> String { b == 0 ? "-" : String(format: "%.1f%%", 100 * Double(a) / Double(b)) }
    func show(_ name: String, _ s: UserSimulator.Stats) {
        print("  \(name): first-choice \(pct(s.firstChoice, s.n)) (n=\(s.n))  clean \(pct(s.cleanFirst, s.cleanN))  slips \(pct(s.typoFirst, s.typoN)) (n=\(s.typoN))  false-corr \(pct(s.falseCorrections, s.cleanN))")
    }
    let fresh = try Engine(resources: EngineResources(dir: resDir))
    if let w = opt("--weights") { fresh.loadWeights(URL(fileURLWithPath: w)) }
    let base = UserSimulator(engine: fresh, nn: nn, learner: nil)
    if useHabit { base.habit = habit }
    print("before learning:")
    show("user test", base.evaluate(test, habit: useHabit))
    let learnEngine = try Engine(resources: EngineResources(dir: resDir))
    if let w = opt("--weights") { learnEngine.loadWeights(URL(fileURLWithPath: w)) }
    let learner = Learner(engine: learnEngine)
    let sim = UserSimulator(engine: learnEngine, nn: nn, learner: learner)
    if useHabit { sim.habit = habit }
    var block = UserSimulator.Stats()
    let blockSize = Int(opt("--block") ?? "") ?? 1000
    var t0 = now()
    for (i, ex) in stream.prefix(limit).enumerated() {
        let before = sim.stats
        sim.type(ex)
        block.n += 1
        block.firstChoice += sim.stats.firstChoice - before.firstChoice
        if (i + 1) % blockSize == 0 {
            learner.tick(sim.clock + 100)
            let kinds = Dictionary(grouping: learner.events, by: { $0.kind.rawValue }).mapValues { $0.count }
            print("stream \(i + 1): first-choice in last block \(pct(block.firstChoice, block.n))  picks \(sim.stats.explicitPicks) retypes \(sim.stats.retypes) rewrites \(sim.stats.rewrites)  events \(kinds)  [\(ms(now() - t0))]")
            if flag("--checkpoints") { show("   user test now", sim.evaluate(test, habit: useHabit)) }
            block = UserSimulator.Stats()
            t0 = now()
        }
    }
    learner.flush(now: sim.clock + 100)
    print("after learning from \(min(limit, stream.count)) chunks (typo guard \(learnEngine.typoGuard)):")
    show("user test", sim.evaluate(test, habit: useHabit))
    learnEngine.usePersonal = false
    show("user test, personal signals off", sim.evaluate(test, habit: useHabit))
    learnEngine.usePersonal = true
    if flag("--diag") {
        for ex in test {
            let a = sim.analyze(keys: ex.keys, context: ex.context)
            learnEngine.usePersonal = false
            let b = sim.analyze(keys: ex.keys, context: ex.context)
            learnEngine.usePersonal = true
            let x = a.full.first?.text ?? "", y = b.full.first?.text ?? ""
            if x != y {
                let fa = a.full.first!.features
                print("   \(x == ex.target ? "+" : (y == ex.target ? "-" : "?")) …\(ex.context.suffix(12)) [\(ex.keys)] personal→\(x) (p=\(String(format: "%.2f mem=%.2f", fa.personal, fa.memory))) base→\(y) want \(ex.target)")
            }
        }
    }
    show("general test", sim.evaluate(general))
    print("  general test without personal data:")
    show("general test", base.evaluate(general))
    print("typo model costs:", EditKind.allCases.map { "\($0.name)=\(String(format: "%.2f", learnEngine.typo.cost($0)))" }.joined(separator: " "), " transpose(u,o)=\(String(format: "%.2f", learnEngine.typo.cost(.transpose, typed: 111, intended: 117)))")
    print("weights full:", learnEngine.weightsFull.w.map { String(format: "%.2f", $0) })
    if let dumpPath = opt("--dump") {
        // features of the user's test chunks under the learned personal state (for fitting trust weights)
        FileManager.default.createFile(atPath: dumpPath, contents: nil)
        let h = FileHandle(forWritingAtPath: dumpPath)!
        for ex in test {
            let a = sim.analyze(keys: ex.keys, context: ex.context)
            let texts = a.full.map { $0.text }
            let obj: [String: Any] = ["label": texts.firstIndex(of: ex.target) ?? -1, "typo": ex.typo != nil, "src": ex.src,
                                      "f": a.full.map { $0.features.vector.map { Double($0) } }, "t": a.full.map { $0.typoCount }]
            h.write(try! JSONSerialization.data(withJSONObject: obj)); h.write("\n".data(using: .utf8)!)
        }
        try? h.close()
        print("dumped \(test.count) examples to \(dumpPath)")
    }
    if let out = opt("--save") {
        let store = PersonalStore(dir: URL(fileURLWithPath: out), engine: learnEngine)
        store.save()
        print("saved personal state to \(out)")
    }

case "latency":
    // stbench latency [--n 400] [--key-ms 110] [--space-ms 70] : replay typing with real timing through the
    // same Composer + NeuralService the IME uses, pressing space right after the last letter.
    setvbuf(stdout, nil, _IOLBF, 0)
    let set = Array(loadExamples(root.appendingPathComponent("data/eval/test.jsonl").path).prefix(Int(opt("--n") ?? "") ?? 400))
    let keyMs = Double(opt("--key-ms") ?? "") ?? 110
    let spaceMs = Double(opt("--space-ms") ?? "") ?? 70
    let service = NeuralService()
    var ready = false
    service.load(modelPath: modelPath) { ready = true }
    while !ready { RunLoop.main.run(until: Date().addingTimeInterval(0.01)) }
    let composer = Composer(engine: engine, neural: service)
    var rng = SplitMix(seed: 3)
    func pause(_ meanMs: Double) {
        // log-normal-ish human interval around the mean (sd ≈ 35%)
        let u1 = max(1e-9, rng.nextDouble()), u2 = rng.nextDouble()
        let z = (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
        let ms = max(15, meanMs * exp(0.35 * z - 0.06))
        RunLoop.main.run(until: Date().addingTimeInterval(ms / 1000))
    }
    var waits: [Double] = [], smart = 0, n = 0, correct = 0, refCorrect = 0, backspaces = 0
    var shownChanges = 0
    for (i, ex) in set.enumerated() {
        composer.begin(context: ex.context, app: "bench")
        var lastShown: [String] = []
        var lastShownAt = Date()
        var lastVersion = -1
        composer.onDisplay = {
            let now = Date()
            let texts = composer.displayed.prefix(5).map { $0.text }
            // the first candidate of the same input changed after it had been on screen long enough to be read
            if composer.displayedVersion == lastVersion && !lastShown.isEmpty && texts.first != lastShown.first && now.timeIntervalSince(lastShownAt) > 0.16 { shownChanges += 1 }
            if composer.displayedVersion != lastVersion || texts.first != lastShown.first { lastShownAt = now }
            lastShown = texts; lastVersion = composer.displayedVersion
        }
        let keys = Array(ex.keys)
        for (k, ch) in keys.enumerated() {
            composer.insert(String(ch))
            // occasionally the user fixes a key on the fly (backspace + retype)
            if i % 7 == 0 && k == keys.count / 2 {
                pause(keyMs); composer.backspace(); backspaces += 1
                pause(keyMs); composer.insert(String(ch))
            }
            pause(k == keys.count - 1 ? spaceMs : keyMs)
        }
        guard let r = composer.commitHighlighted() else { continue }
        n += 1
        waits.append(r.waitedMs)
        if r.usedNeural { smart += 1 }
        if r.text == ex.target { correct += 1 }
        pause(250)
    }
    let s = waits.sorted()
    func q(_ p: Double) -> Double { s.isEmpty ? 0 : s[min(s.count - 1, Int(Double(s.count) * p))] }
    print(String(format: "commits %d  smart first choice used %.1f%%  top1 %.1f%%  wait on space: p50 %.1fms p90 %.1fms p99 %.1fms max %.1fms  visible reorders after 160ms: %d",
                 n, 100 * Double(smart) / Double(max(1, n)), 100 * Double(correct) / Double(max(1, n)), q(0.5), q(0.9), q(0.99), s.last ?? 0, shownChanges))
    _ = refCorrect; _ = backspaces
    service.shutdown()

case "nncheck":
    // stbench nncheck <pinyin> <context> : batched vs one-by-one neural scores (consistency check)
    let a = engine.analyze(raw: args[1], context: args.count > 2 ? args[2] : "")
    guard let nn = makeNN() else { exit(1) }
    nn.setContext(args.count > 2 ? args[2] : "")
    let batch = nn.score(a.options)
    var maxDiff: Float = 0
    for (i, o) in a.options.enumerated() {
        let single = nn.score([o])[0]
        maxDiff = max(maxDiff, abs(single - batch[i]))
        if i < 8 || abs(single - batch[i]) > 0.05 { print(String(format: "%@ batch=%.2f single=%.2f tokens=%d", o, batch[i], single, nn.tokenize(o).count)) }
    }
    print("candidates \(a.options.count) max diff \(maxDiff)")

case "profloop":
    let input = args[1]
    let t0 = now()
    for _ in 0..<300 { _ = engine.analyze(raw: input, context: "我想") }
    print("avg \(ms((now() - t0) / 300))")

case "prof":
    // stbench prof <pinyin> : timing breakdown of the fast path
    let input = args[1]
    for _ in 0..<3 {
        let t0 = now()
        let g = SyllableGraph(raw: Array(input.utf8), table: lex.syllables, typo: engine.typo, options: engine.graphOptions)
        let t1 = now()
        let lat = Lattice(graph: g, lexicon: lex, lm: engine.lm, options: engine.latticeOptions)
        let t2 = now()
        let r = engine.decoder.decode(lattice: lat, context: 0, weights: engine.weightsFast)
        let t3 = now()
        let nS = g.edgesFrom.reduce(0) { $0 + $1.count }, nW = lat.edgesFrom.reduce(0) { $0 + $1.count }
        let typoEdges = g.edgesFrom.reduce(0) { $0 + $1.filter { $0.kind == .typo }.count }
        var byKind: [String: Int] = [:]
        for l in g.edgesFrom { for e in l where e.kind == .typo { byKind[e.edit!.name, default: 0] += 1 } }
        print(byKind)
        print("graph \(ms(t1-t0)) syl=\(nS) typo=\(typoEdges)  lattice \(ms(t2-t1)) words=\(nW)  decode \(ms(t3-t2)) full=\(r.full.count)")
    }

default:
    print("usage: stbench decode <pinyin> [context] [--nn] | eval <set.jsonl> [--nn] [--dump f] [--limit n] [--verbose]")
}
