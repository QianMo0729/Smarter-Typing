import AppKit
import InputMethodKit
import Carbon

// SmartPinyin input method entry point.
//   SmartPinyin            run as the input method server (launched by macOS)
//   SmartPinyin --install  register + enable the input source for the current user

let bundle = Bundle.main

if CommandLine.arguments.contains("--install") {
    let url = bundle.bundleURL as CFURL
    var status = TISRegisterInputSource(url)
    print("register:", status)
    if let list = TISCreateInputSourceList(nil, true)?.takeRetainedValue() as? [TISInputSource] {
        for src in list {
            guard let idPtr = TISGetInputSourceProperty(src, kTISPropertyInputSourceID) else { continue }
            let id = Unmanaged<CFString>.fromOpaque(idPtr).takeUnretainedValue() as String
            if id.hasPrefix(bundle.bundleIdentifier ?? "com.smarttyping.inputmethod.SmartPinyin") {
                status = TISEnableInputSource(src)
                print("enable \(id):", status)
            }
        }
    }
    exit(0)
}

let connection = bundle.object(forInfoDictionaryKey: "InputMethodConnectionName") as? String ?? "SmartPinyin_Connection"
let app = NSApplication.shared
guard let server = IMKServer(name: connection, bundleIdentifier: bundle.bundleIdentifier) else {
    Log.write("cannot create IMKServer")
    exit(1)
}
_ = server
// Load the engine and start the model before the first keystroke arrives.
_ = AppServices.shared
Log.write("SmartPinyin started")
app.run()
