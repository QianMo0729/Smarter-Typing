import AppKit
// Menu bar icon in the style of the system input sources: a full-size rounded square with the glyph
// knocked out, drawn as a template (the system tints it for light / dark menu bars).
let out = CommandLine.arguments[1]
var reps: [NSBitmapImageRep] = []
for px in [16, 32] {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = NSSize(width: 16, height: 16)
    NSGraphicsContext.saveGraphicsState()
    let ctx = NSGraphicsContext(bitmapImageRep: rep)!
    NSGraphicsContext.current = ctx
    NSColor.black.setFill()
    NSBezierPath(roundedRect: NSRect(x: 0, y: 0, width: 16, height: 16), xRadius: 3.5, yRadius: 3.5).fill()
    ctx.compositingOperation = .destinationOut
    let s = NSAttributedString(string: "智", attributes: [.font: NSFont.systemFont(ofSize: 11.5, weight: .medium), .foregroundColor: NSColor.black])
    let sz = s.size()
    s.draw(at: NSPoint(x: 8 - sz.width / 2, y: 8 - sz.height / 2 + 0.3))
    NSGraphicsContext.restoreGraphicsState()
    reps.append(rep)
}
let data = NSBitmapImageRep.tiffRepresentationOfImageReps(in: reps, using: .lzw, factor: 0)!
try! data.write(to: URL(fileURLWithPath: out))
