import AppKit
// Keep the system ABC icon's 22×16 point canvas, but draw only the character.
// macOS shares this artwork between the menu bar and the badge beside the cursor.
// An opaque glyph with a fine dark outline stays visible on light, dark and blue
// backgrounds without a keycap. Keep it non-template for the cursor badge.
let out = CommandLine.arguments[1]
let size = NSSize(width: 22, height: 16)
var reps: [NSBitmapImageRep] = []
for scale in [1, 2] {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width) * scale, pixelsHigh: Int(size.height) * scale, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = size
    NSGraphicsContext.saveGraphicsState()
    let ctx = NSGraphicsContext(bitmapImageRep: rep)!
    NSGraphicsContext.current = ctx
    let bounds = NSRect(origin: .zero, size: size)
    NSColor.clear.setFill()
    bounds.fill(using: .copy)
    let s = NSAttributedString(string: "智", attributes: [
        .font: NSFont.systemFont(ofSize: 13.5, weight: .semibold),
        .foregroundColor: NSColor.white,
    ])
    let outline = NSMutableAttributedString(attributedString: s)
    outline.addAttributes([
        .strokeColor: NSColor(srgbRed: 0.15, green: 0.15, blue: 0.15, alpha: 1),
        .strokeWidth: 5.0,
    ], range: NSRange(location: 0, length: outline.length))
    let sz = s.size()
    let origin = NSPoint(x: bounds.midX - sz.width / 2, y: bounds.midY - sz.height / 2 + 0.5)
    outline.draw(at: origin)
    s.draw(at: origin)
    NSGraphicsContext.restoreGraphicsState()
    reps.append(rep)
}
let data = NSBitmapImageRep.tiffRepresentationOfImageReps(in: reps, using: .lzw, factor: 0)!
try! data.write(to: URL(fileURLWithPath: out))
