import AppKit
import InputMethodKit
import PinyinEngine

/// One controller per client document. Handles keys, shows marked text and candidates, commits text,
/// and reports what happens to committed text (deletions, replacements) to the learner.
@objc(SmartPinyinInputController)
final class SmartPinyinInputController: IMKInputController {
    private let services = AppServices.shared
    private lazy var composer: Composer = {
        let c = Composer(engine: services.engine, neural: services.neural)
        c.onDisplay = { [weak self] in self?.renderCandidates() }
        return c
    }()
    private var settings: Settings { Settings.shared }
    private var vertical: Bool { settings.layout == .vertical }
    private var asciiMode = false
    private var shiftDown = false
    private var shiftUsedWithKey = false
    private var punct = Punctuation()
    private var localContext = ""            // what we committed recently, for apps without context access
    private var expectedLocation: Int?       // cursor location after our last commit (UTF-16)
    private var lastClientID: String?
    private var expanded = false              // candidate grid expanded (↓ or chevron)

    private var appID: String? { (client() as IMKTextInput?)?.bundleIdentifier() }

    override func recognizedEvents(_ sender: Any!) -> Int {
        Int(NSEvent.EventTypeMask.keyDown.rawValue | NSEvent.EventTypeMask.flagsChanged.rawValue)
    }

    override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        let id = appID
        if id != lastClientID { localContext = ""; expectedLocation = nil; lastClientID = id }
    }

    override func deactivateServer(_ sender: Any!) {
        if composer.isComposing, let c = sender as? IMKTextInput { commitComposition(c) }
        CandidatePanel.shared.hide()
        services.learner.didBreak(time: now)
        services.saveSoon()
        super.deactivateServer(sender)
    }

    override func commitComposition(_ sender: Any!) {
        guard composer.isComposing, let c = sender as? IMKTextInput else { return }
        // the system wants the composition gone (focus change etc.): keep what was typed, literally
        let s = composer.commitRaw()
        c.insertText(s, replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        CandidatePanel.shared.hide()
    }

    private var now: Double { Date().timeIntervalSince1970 }

    // MARK: key handling

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event, let client = sender as? IMKTextInput else { return false }
        if event.type == .flagsChanged { return handleFlags(event, client) }
        guard event.type == .keyDown else { return false }
        if shiftDown { shiftUsedWithKey = true }
        let mods = event.modifierFlags.intersection([.command, .control, .option])
        let code = event.keyCode
        let chars = event.characters ?? ""

        if !mods.isEmpty {
            // shortcuts (cmd+z, cmd+v, ...) change text in ways we cannot follow
            if composer.isComposing { commitRawText(client) }
            services.learner.didBreak(time: now)
            expectedLocation = nil
            return false
        }

        if composer.isComposing { return handleComposing(event, code: code, chars: chars, client: client) }

        // not composing
        switch code {
        case 51:  // backspace: deleting text we may have committed
            services.learner.didDeleteBackward(time: now)
            if !localContext.isEmpty { localContext.removeLast() }
            if let e = expectedLocation { expectedLocation = e - 1 }
            return false
        case 117, 123, 124, 125, 126, 115, 119, 116, 121, 48, 36, 76, 53:
            // forward delete, arrows, home/end, page, tab, return, escape: cursor moves / structure changes
            services.learner.didBreak(time: now)
            expectedLocation = nil
            if code == 36 || code == 76 { localContext += "\n" }
            return false
        default: break
        }
        if asciiMode { noteForeignText(chars); return false }
        guard let ch = chars.first, chars.count == 1 else { return false }
        if ch >= "a" && ch <= "z" {
            startComposition(client)
            composer.insert(String(ch))
            updateMarked(client)
            return true
        }
        if settings.fullWidthPunctuation, let p = punct.map(ch) {
            insert(p, client: client, learnable: false)
            return true
        }
        noteForeignText(chars)
        return false
    }

    private func handleComposing(_ event: NSEvent, code: UInt16, chars: String, client: IMKTextInput) -> Bool {
        let ch = chars.first
        switch code {
        case 49: // space
            if let r = composer.commitHighlighted() { finishCommit(r, client: client) } else { updateMarked(client) }
            return true
        case 51: // backspace
            expanded = false
            composer.backspace()
            if composer.isComposing { updateMarked(client) } else { clearMarked(client) }
            return true
        case 36, 76: // return: literal letters
            commitRawText(client)
            return true
        case 53: // escape: collapse the grid, otherwise cancel
            if expanded { expanded = false; renderCandidates(); return true }
            composer.reset()
            clearMarked(client)
            return true
        case 123: // ←
            if vertical { pageBy(-1) } else { composer.moveHighlight(-1) }
            return true
        case 124: // →
            if vertical { pageBy(1) } else { composer.moveHighlight(1) }
            return true
        case 125: // ↓: horizontal — expand into the grid, then move down a row; vertical — next candidate
            if vertical { composer.moveHighlight(1) }
            else if !expanded { expanded = true; renderCandidates() } else { composer.moveHighlight(CandidatePanel.pageSize) }
            return true
        case 126: // ↑: move up a row; collapse at the top
            if vertical { composer.moveHighlight(-1) }
            else if composer.highlighted < CandidatePanel.pageSize { if expanded { expanded = false; renderCandidates() } }
            else { composer.moveHighlight(-CandidatePanel.pageSize) }
            return true
        case 48: // tab: next candidate
            composer.moveHighlight(event.modifierFlags.contains(.shift) ? -1 : 1); return true
        default: break
        }
        guard let ch, chars.count == 1 else { return true }
        if ch >= "a" && ch <= "z" || (ch == "'" && !composer.raw.isEmpty) {
            expanded = false
            composer.insert(String(ch))
            updateMarked(client)
            return true
        }
        if let d = ch.wholeNumberValue, d >= 1 && d <= 9 {
            let page = composer.highlighted / CandidatePanel.pageSize
            let idx = page * CandidatePanel.pageSize + d - 1
            if let r = composer.select(index: idx) { finishCommit(r, client: client) } else { updateMarked(client) }
            return true
        }
        if ch == "-" || ch == "=" || ch == "[" || ch == "]" {
            pageBy(ch == "=" || ch == "]" ? 1 : -1)
            return true
        }
        if ch.isUppercase && ch.isLetter {
            // an uppercase letter ends the composition: keep the typed letters, then the letter itself
            commitRawText(client)
            return false
        }
        if !ch.isLetter && !ch.isNumber && !ch.isWhitespace {
            // punctuation commits the smart first choice, then the punctuation
            if let r = composer.commitHighlighted() { finishCommit(r, client: client) }
            if !composer.isComposing {
                if settings.fullWidthPunctuation, let p = punct.map(ch) { insert(p, client: client, learnable: false) }
                else { insert(String(ch), client: client, learnable: false) }
            }
            return true
        }
        return true
    }

    private func pageBy(_ dir: Int) {
        let rows = expanded && !vertical ? CandidatePanel.expandedRows : 1
        let ps = CandidatePanel.pageSize
        let start = (composer.highlighted / ps) * ps
        let target = dir > 0 ? start + rows * ps : max(0, start - rows * ps)
        if target < composer.displayed.count { composer.moveHighlight(target - composer.highlighted) }
    }

    private func handleFlags(_ event: NSEvent, _ client: IMKTextInput) -> Bool {
        let shift = event.modifierFlags.contains(.shift)
        let others = !event.modifierFlags.intersection([.command, .control, .option]).isEmpty
        if shift && !shiftDown {
            shiftDown = true
            shiftUsedWithKey = others
        } else if !shift && shiftDown {
            shiftDown = false
            if !shiftUsedWithKey && !others && settings.shiftToggles {
                // a lone Shift tap toggles Chinese / English
                if composer.isComposing { commitRawText(client) }
                asciiMode.toggle()
                var rect = NSRect.zero
                client.attributes(forCharacterIndex: 0, lineHeightRectangle: &rect)
                CandidatePanel.shared.flash(asciiMode ? "英" : "中", anchor: rect)
            }
        }
        return false
    }

    // MARK: composition plumbing

    private func startComposition(_ client: IMKTextInput) {
        let sel = client.selectedRange()
        var context = ""
        if sel.location != NSNotFound {
            if let e = expectedLocation, e != sel.location {
                services.learner.didBreak(time: now)   // cursor moved (mouse, other tools)
                localContext = ""
            }
            if sel.length > 0, let s = client.attributedSubstring(from: sel)?.string, !s.isEmpty {
                services.learner.didReplaceSelection(s, time: now)
            }
            if sel.location > 0 {
                let len = min(sel.location, 160)
                if let s = client.attributedSubstring(from: NSRange(location: sel.location - len, length: len))?.string {
                    context = s
                }
            }
        }
        services.stats.recordContext(fromDocument: context.count, local: localContext.count)
        if context.isEmpty { context = localContext }
        composer.neural = settings.smartRanking ? services.neural : nil
        composer.spacedPinyin = settings.spacedPinyin
        composer.begin(context: String(context.suffix(160)), app: appID)
    }

    private func updateMarked(_ client: IMKTextInput) {
        let text = composer.markedText
        let attrs: [NSAttributedString.Key: Any] = [.underlineStyle: NSUnderlineStyle.single.rawValue]
        client.setMarkedText(NSAttributedString(string: text, attributes: attrs),
                             selectionRange: NSRange(location: (text as NSString).length, length: 0),
                             replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        renderCandidates()
    }

    private func clearMarked(_ client: IMKTextInput) {
        expanded = false
        client.setMarkedText("", selectionRange: NSRange(location: 0, length: 0), replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        CandidatePanel.shared.hide()
    }

    private func renderCandidates() {
        guard let client = client() as IMKTextInput?, composer.isComposing, !composer.displayed.isEmpty else {
            CandidatePanel.shared.hide(); return
        }
        var rect = NSRect.zero
        client.attributes(forCharacterIndex: 0, lineHeightRectangle: &rect)
        let panel = CandidatePanel.shared
        panel.onSelect = { [weak self] i in
            guard let self, let c = self.client() as IMKTextInput? else { return }
            if let r = self.composer.select(index: i) { self.finishCommit(r, client: c) } else { self.updateMarked(c) }
        }
        panel.onToggleExpand = { [weak self] in self?.expanded.toggle(); self?.renderCandidates() }
        panel.show(candidates: composer.displayed, highlighted: composer.highlighted, expanded: expanded, anchor: rect)
    }

    private func finishCommit(_ r: CommitResult, client: IMKTextInput) {
        clearMarked(client)
        insert(r.text, client: client, learnable: true)
        services.stats.record(r)
        if settings.learning && !Privacy.isSensitive(appID) {
            services.learner.didCommit(r.record)
        }
    }

    private func commitRawText(_ client: IMKTextInput) {
        let s = composer.commitRaw()
        clearMarked(client)
        if !s.isEmpty { insert(s, client: client, learnable: false) }
    }

    private func insert(_ text: String, client: IMKTextInput, learnable: Bool) {
        let sel = client.selectedRange()
        client.insertText(text, replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        localContext = String((localContext + text).suffix(200))
        if sel.location != NSNotFound { expectedLocation = sel.location + (text as NSString).length } else { expectedLocation = nil }
        if !learnable {
            // punctuation / literal text we typed: the learner treats it as a boundary it cannot interpret
            services.learner.didBreak(time: now)
        }
        services.neural.prepareContext(localContext)
    }

    private func noteForeignText(_ chars: String) {
        localContext = String((localContext + chars).suffix(200))
        if let e = expectedLocation { expectedLocation = e + (chars as NSString).length }
    }

    // MARK: menu

    override func menu() -> NSMenu! {
        let m = NSMenu()
        let prefs = NSMenuItem(title: "智能拼音设置…", action: #selector(showSettings(_:)), keyEquivalent: "")
        prefs.target = self
        m.addItem(prefs)
        m.addItem(.separator())
        let ready = NSMenuItem(title: services.neural.isReady ? "智能判断：已就绪" : "智能判断：加载中", action: nil, keyEquivalent: "")
        ready.isEnabled = false
        m.addItem(ready)
        let stats = NSMenuItem(title: services.stats.summary, action: nil, keyEquivalent: "")
        stats.isEnabled = false
        m.addItem(stats)
        return m
    }

    @objc func showSettings(_ sender: Any?) {
        let services = self.services
        SettingsWindow.shared.show {
            let alert = NSAlert()
            alert.messageText = "清除个人学习数据？"
            alert.informativeText = "输入法从你的输入中学到的用词、搭配和误触习惯都会被删除，无法恢复。"
            alert.addButton(withTitle: "清除")
            alert.addButton(withTitle: "取消")
            if alert.runModal() == .alertFirstButtonReturn { services.resetPersonalData() }
        }
    }
}

enum Privacy {
    /// Apps whose text should never be learned from.
    static let blocked: Set<String> = ["com.apple.keychainaccess", "com.1password.1password", "com.agilebits.onepassword7", "com.bitwarden.desktop", "com.apple.Passwords"]
    static func isSensitive(_ app: String?) -> Bool { app.map { blocked.contains($0) } ?? false }
}
