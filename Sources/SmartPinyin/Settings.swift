import AppKit
import SwiftUI

/// User preferences (stored in UserDefaults of the input method). Changes apply immediately.
final class Settings: ObservableObject {
    static let shared = Settings()
    static let sustechOrange = NSColor(srgbRed: 237 / 255, green: 108 / 255, blue: 0, alpha: 1)

    enum ThemeColor: String, CaseIterable, Identifiable {
        case sustech, system, blue, green, purple, pink, graphite, custom
        var id: String { rawValue }
        var title: String {
            switch self {
            case .sustech: return "南科大橙"
            case .system: return "跟随系统强调色"
            case .blue: return "蓝色"
            case .green: return "绿色"
            case .purple: return "紫色"
            case .pink: return "粉色"
            case .graphite: return "石墨色"
            case .custom: return "自定义"
            }
        }
    }
    enum Layout: String, CaseIterable, Identifiable {
        case horizontal, vertical
        var id: String { rawValue }
        var title: String { self == .horizontal ? "横排" : "竖排" }
    }
    enum Appearance: String, CaseIterable, Identifiable {
        case system, light, dark
        var id: String { rawValue }
        var title: String { ["system": "跟随系统", "light": "浅色", "dark": "深色"][rawValue]! }
    }

    private let d = UserDefaults.standard

    @Published var theme: ThemeColor { didSet { d.set(theme.rawValue, forKey: "theme") } }
    @Published var customColor: Color { didSet { saveCustom() } }
    @Published var fontSize: Double { didSet { d.set(fontSize, forKey: "fontSize") } }
    @Published var pageSize: Int { didSet { d.set(pageSize, forKey: "pageSize") } }
    @Published var layout: Layout { didSet { d.set(layout.rawValue, forKey: "layout") } }
    @Published var appearance: Appearance { didSet { d.set(appearance.rawValue, forKey: "appearance") } }
    @Published var spacedPinyin: Bool { didSet { d.set(spacedPinyin, forKey: "spacedPinyin") } }
    @Published var smartRanking: Bool { didSet { d.set(smartRanking, forKey: "smartRanking") } }
    @Published var learning: Bool { didSet { d.set(learning, forKey: "learning") } }
    @Published var fullWidthPunctuation: Bool { didSet { d.set(fullWidthPunctuation, forKey: "fullWidthPunctuation") } }

    private init() {
        d.register(defaults: ["theme": "sustech", "fontSize": 17.0, "pageSize": 9, "layout": "horizontal", "appearance": "system",
                              "spacedPinyin": true, "smartRanking": true, "learning": true, "fullWidthPunctuation": true])
        theme = ThemeColor(rawValue: d.string(forKey: "theme") ?? "") ?? .sustech
        fontSize = d.double(forKey: "fontSize")
        pageSize = max(3, min(10, d.integer(forKey: "pageSize")))
        layout = Layout(rawValue: d.string(forKey: "layout") ?? "") ?? .horizontal
        appearance = Appearance(rawValue: d.string(forKey: "appearance") ?? "") ?? .system
        spacedPinyin = d.bool(forKey: "spacedPinyin")
        smartRanking = d.bool(forKey: "smartRanking")
        learning = d.bool(forKey: "learning")
        fullWidthPunctuation = d.bool(forKey: "fullWidthPunctuation")
        if let a = d.array(forKey: "customColor") as? [Double], a.count == 3 {
            customColor = Color(.sRGB, red: a[0], green: a[1], blue: a[2])
        } else {
            customColor = Color(nsColor: Settings.sustechOrange)
        }
    }

    private func saveCustom() {
        if let c = NSColor(customColor).usingColorSpace(.sRGB) {
            d.set([Double(c.redComponent), Double(c.greenComponent), Double(c.blueComponent)], forKey: "customColor")
        }
    }

    /// Colour of the selected candidate.
    var accent: NSColor {
        switch theme {
        case .sustech: return Settings.sustechOrange
        case .system: return .controlAccentColor
        case .blue: return .systemBlue
        case .green: return .systemGreen
        case .purple: return .systemPurple
        case .pink: return .systemPink
        case .graphite: return .systemGray
        case .custom: return NSColor(customColor)
        }
    }

    var nsAppearance: NSAppearance? {
        switch appearance {
        case .system: return nil
        case .light: return NSAppearance(named: .aqua)
        case .dark: return NSAppearance(named: .darkAqua)
        }
    }
}

struct SettingsView: View {
    @ObservedObject var s = Settings.shared
    var onClearData: () -> Void

    var body: some View {
        Form {
            Section("候选窗") {
                Picker("主题色", selection: $s.theme) {
                    ForEach(Settings.ThemeColor.allCases) { t in
                        HStack {
                            Circle().fill(Color(nsColor: color(t))).frame(width: 10, height: 10)
                            Text(t.title)
                        }.tag(t)
                    }
                }
                if s.theme == .custom {
                    ColorPicker("自定义颜色", selection: $s.customColor, supportsOpacity: false)
                }
                Picker("排列", selection: $s.layout) {
                    ForEach(Settings.Layout.allCases) { Text($0.title).tag($0) }
                }.pickerStyle(.segmented)
                Picker("外观", selection: $s.appearance) {
                    ForEach(Settings.Appearance.allCases) { Text($0.title).tag($0) }
                }.pickerStyle(.segmented)
                HStack {
                    Text("字号")
                    Slider(value: $s.fontSize, in: 13...24, step: 1)
                    Text("\(Int(s.fontSize))").monospacedDigit().frame(width: 28)
                }
                Stepper("每页候选数：\(s.pageSize)", value: $s.pageSize, in: 3...10)
                CandidatePreview(s: s).frame(height: CGFloat(s.fontSize) * 2 + (s.layout == .vertical ? 90 : 20))
            }
            Section("输入") {
                Toggle("拼音按音节分隔显示（zhi neng）", isOn: $s.spacedPinyin)
                Toggle("中文标点使用全角", isOn: $s.fullWidthPunctuation)
                Text("中/英键（Caps Lock）切换输入法：在“系统设置 › 键盘 › 输入法”中打开“使用中/英键切换”。")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Section("智能") {
                Toggle("智能判断（本机小模型结合上下文排序）", isOn: $s.smartRanking)
                Toggle("从日常输入中自动学习", isOn: $s.learning)
                Button("清除个人学习数据…", role: .destructive) { onClearData() }
                Text("学习数据只保存在本机：~/Library/Application Support/SmartPinyin，不保存原文。")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 460)
    }

    func color(_ t: Settings.ThemeColor) -> NSColor {
        let saved = s.theme
        _ = saved
        switch t {
        case .sustech: return Settings.sustechOrange
        case .system: return .controlAccentColor
        case .blue: return .systemBlue
        case .green: return .systemGreen
        case .purple: return .systemPurple
        case .pink: return .systemPink
        case .graphite: return .systemGray
        case .custom: return NSColor(s.customColor)
        }
    }
}

/// Live preview of the candidate window inside the settings.
struct CandidatePreview: NSViewRepresentable {
    @ObservedObject var s: Settings
    func makeNSView(context: Context) -> CandidateView { CandidateView() }
    func updateNSView(_ v: CandidateView, context: Context) {
        let words = ["智能", "只能", "职能", "智", "只", "之", "知", "直", "指", "至"]
        let n = min(s.pageSize, words.count)
        v.update(rows: [(0..<n).map { ($0, words[$0]) }], highlighted: 0, expanded: false, canExpand: true)
        v.needsDisplay = true
    }
}

final class SettingsWindow {
    static let shared = SettingsWindow()
    private var window: NSWindow?

    func show(onClearData: @escaping () -> Void) {
        if window == nil {
            let host = NSHostingController(rootView: SettingsView(onClearData: onClearData))
            let w = NSWindow(contentViewController: host)
            w.title = "智能拼音 设置"
            w.styleMask = [.titled, .closable]
            w.isReleasedWhenClosed = false
            w.center()
            window = w
        }
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }
}
