import AppKit
import PinyinEngine

/// Candidate window modelled on the system Pinyin input method: a row of numbered candidates with a
/// disclosure chevron; ↓ or a click on the chevron expands it into a grid that keeps paging down.
/// Colour, size, page size, layout and appearance follow `Settings` (default: SUSTech orange).
final class CandidatePanel {
    static let shared = CandidatePanel()
    static var pageSize: Int { Settings.shared.pageSize }
    static let expandedRows = 6

    private let panel: NSPanel
    private let container: NSView
    private let view: CandidateView
    /// Called when the user clicks a candidate (absolute index) or the chevron.
    var onSelect: ((Int) -> Void)?
    var onToggleExpand: (() -> Void)?

    private init() {
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 300, height: 36), styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.popUpMenuWindow)))
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true
        panel.becomesKeyOnlyIfNeeded = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient, .ignoresCycle]
        view = CandidateView()
        if #available(macOS 26.0, *) {
            let glass = NSGlassEffectView()
            glass.cornerRadius = CandidateView.cornerRadius
            glass.contentView = view
            container = glass
        } else {
            let effect = NSVisualEffectView()
            effect.material = .popover
            effect.state = .active
            effect.wantsLayer = true
            effect.layer?.cornerRadius = CandidateView.cornerRadius
            effect.layer?.masksToBounds = true
            effect.layer?.borderWidth = 0.5
            effect.layer?.borderColor = NSColor.separatorColor.cgColor
            effect.addSubview(view)
            container = effect
        }
        panel.contentView = container
        view.onClick = { [weak self] hit in
            switch hit {
            case .candidate(let i): self?.onSelect?(i)
            case .chevron: self?.onToggleExpand?()
            }
        }
    }

    func show(candidates: [Candidate], highlighted: Int, expanded: Bool, anchor: NSRect) {
        guard !candidates.isEmpty else { hide(); return }
        let settings = Settings.shared
        panel.appearance = settings.nsAppearance
        let ps = Self.pageSize
        let row = highlighted / ps
        var rows: [[(Int, String)]] = []
        if settings.layout == .vertical {
            let start = row * ps
            for i in start..<min(candidates.count, start + ps) { rows.append([(i, candidates[i].text)]) }
            view.update(rows: rows, highlighted: highlighted, expanded: false, canExpand: false, vertical: true)
        } else {
            let firstRow = expanded ? max(0, row - Self.expandedRows / 2) : row
            let rowCount = expanded ? Self.expandedRows : 1
            for r in firstRow..<(firstRow + rowCount) {
                let start = r * ps
                if start >= candidates.count { break }
                rows.append((start..<min(candidates.count, start + ps)).map { ($0, candidates[$0].text) })
            }
            view.update(rows: rows, highlighted: highlighted, expanded: expanded,
                        canExpand: candidates.count > (row + 1) * ps || expanded || row > 0, vertical: false)
        }
        let size = view.fittingSize
        view.frame = NSRect(origin: .zero, size: size)
        container.frame = NSRect(origin: .zero, size: size)
        // just below the text line, like the system IME; flipped above near the bottom of the screen
        var origin = NSPoint(x: anchor.minX - CandidateView.padX + 2, y: anchor.minY - size.height - 6)
        if let screen = NSScreen.screens.first(where: { $0.frame.contains(anchor.origin) }) ?? NSScreen.main {
            let vf = screen.visibleFrame
            if origin.y < vf.minY { origin.y = anchor.maxY + 6 }
            if origin.x + size.width > vf.maxX { origin.x = vf.maxX - size.width - 4 }
            origin.x = max(vf.minX + 4, origin.x)
        }
        panel.setFrame(NSRect(origin: origin, size: size), display: true)
        panel.orderFrontRegardless()
    }

    func hide() { panel.orderOut(nil) }

}

final class CandidateView: NSView {
    enum Hit { case candidate(Int), chevron }

    static let cornerRadius: CGFloat = 10
    static let padX: CGFloat = 8
    private var font: NSFont { NSFont.systemFont(ofSize: CGFloat(Settings.shared.fontSize)) }
    private var numberFont: NSFont { NSFont.monospacedDigitSystemFont(ofSize: max(10, CGFloat(Settings.shared.fontSize) * 0.7), weight: .regular) }
    private var rowHeight: CGFloat { ceil(CGFloat(Settings.shared.fontSize) * 2) }
    private let itemGap: CGFloat = 6
    private let chevronWidth: CGFloat = 28

    private var rows: [[(Int, String)]] = []
    private var highlighted = 0
    private var expanded = false
    private var canExpand = false
    private var vertical = false
    private var hitRects: [(NSRect, Hit)] = []
    var onClick: ((Hit) -> Void)?

    override var isFlipped: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    func update(rows: [[(Int, String)]], highlighted: Int, expanded: Bool, canExpand: Bool, vertical: Bool = false) {
        self.rows = rows; self.highlighted = highlighted; self.expanded = expanded; self.canExpand = canExpand; self.vertical = vertical
        needsDisplay = true
    }

    private func isCurrentRow(_ row: [(Int, String)]) -> Bool {
        guard let first = row.first?.0 else { return false }
        return highlighted / CandidatePanel.pageSize == first / CandidatePanel.pageSize
    }

    private func label(_ i: Int) -> Int { i % CandidatePanel.pageSize + 1 }

    private func itemParts(_ n: Int, _ text: String, selected: Bool, numbered: Bool) -> (NSAttributedString, NSAttributedString) {
        let numColor = selected ? NSColor.white.withAlphaComponent(0.85) : NSColor.secondaryLabelColor
        let textColor = selected ? NSColor.white : NSColor.labelColor
        let num = NSAttributedString(string: numbered ? "\(n)" : " ", attributes: [.font: numberFont, .foregroundColor: numColor])
        let txt = NSAttributedString(string: text, attributes: [.font: font, .foregroundColor: textColor])
        return (num, txt)
    }

    private func itemWidth(_ num: NSAttributedString, _ txt: NSAttributedString) -> CGFloat {
        8 + num.size().width + 5 + txt.size().width + 9
    }

    override var fittingSize: NSSize {
        var maxW: CGFloat = 0
        for row in rows {
            let current = isCurrentRow(row) || vertical
            var w = Self.padX
            for (i, t) in row {
                let (n, s) = itemParts(label(i), t, selected: i == highlighted, numbered: current || !expanded)
                w += itemWidth(n, s) + itemGap
            }
            maxW = max(maxW, w - itemGap)
        }
        let width = maxW + (canExpand ? chevronWidth + 4 : 0) + Self.padX
        return NSSize(width: max(width, vertical ? 90 : 120), height: CGFloat(max(1, rows.count)) * rowHeight + 8)
    }

    override func draw(_ dirtyRect: NSRect) {
        hitRects.removeAll()
        let accent = Settings.shared.accent
        var y: CGFloat = 4
        let fullWidth = bounds.width - 2 * Self.padX
        for row in rows {
            let current = isCurrentRow(row) || vertical
            var x = Self.padX
            for (i, t) in row {
                let selected = i == highlighted
                let (num, txt) = itemParts(label(i), t, selected: selected, numbered: current || !expanded)
                let w = vertical ? fullWidth : itemWidth(num, txt)
                let r = NSRect(x: x, y: y + 3, width: w, height: rowHeight - 6)
                if selected {
                    accent.setFill()
                    NSBezierPath(roundedRect: r, xRadius: 6, yRadius: 6).fill()
                }
                let ts = txt.size(), ns = num.size()
                let baseline = r.midY - ts.height / 2
                num.draw(at: NSPoint(x: r.minX + 8, y: baseline + (ts.height - ns.height) / 2 + 1))
                txt.draw(at: NSPoint(x: r.minX + 8 + ns.width + 5, y: baseline))
                hitRects.append((r, .candidate(i)))
                x += w + itemGap
            }
            y += rowHeight
        }
        if canExpand {
            let cx = bounds.maxX - Self.padX - chevronWidth
            NSColor.separatorColor.setFill()
            NSRect(x: cx - 2, y: 10, width: 0.5, height: rowHeight - 12).fill()
            let r = NSRect(x: cx, y: 4 + 3, width: chevronWidth, height: rowHeight - 6)
            if let img = NSImage(systemSymbolName: expanded ? "chevron.up" : "chevron.down", accessibilityDescription: nil)?
                .withSymbolConfiguration(.init(pointSize: 12, weight: .semibold)) {
                let tinted = img.tinted(NSColor.secondaryLabelColor)
                let s = tinted.size
                tinted.draw(in: NSRect(x: r.midX - s.width / 2, y: r.midY - s.height / 2, width: s.width, height: s.height))
            }
            hitRects.append((r, .chevron))
        }
    }

    override func mouseDown(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        if let hit = hitRects.first(where: { $0.0.contains(p) })?.1 { onClick?(hit) }
    }
}

private extension NSImage {
    func tinted(_ color: NSColor) -> NSImage {
        let img = NSImage(size: size, flipped: false) { rect in
            self.draw(in: rect)
            color.set()
            rect.fill(using: .sourceAtop)
            return true
        }
        img.isTemplate = false
        return img
    }
}
