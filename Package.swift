// swift-tools-version:6.0
import PackageDescription
import Foundation

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let llamaBuild = "\(root)/third_party/llama.cpp/build"

let llamaLinker: [LinkerSetting] = [
    .unsafeFlags([
        "-L\(llamaBuild)/src",
        "-L\(llamaBuild)/ggml/src",
        "-L\(llamaBuild)/ggml/src/ggml-metal",
        "-L\(llamaBuild)/ggml/src/ggml-blas",
    ]),
    .linkedLibrary("llama"),
    .linkedLibrary("ggml"),
    .linkedLibrary("ggml-base"),
    .linkedLibrary("ggml-cpu"),
    .linkedLibrary("ggml-metal"),
    .linkedLibrary("ggml-blas"),
    .linkedLibrary("c++"),
    .linkedFramework("Accelerate"),
    .linkedFramework("Metal"),
    .linkedFramework("MetalKit"),
    .linkedFramework("Foundation"),
]

let package = Package(
    name: "SmartPinyin",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "SmartPinyin", targets: ["SmartPinyin"]),
        .executable(name: "stbench", targets: ["stbench"]),
        .library(name: "PinyinEngine", targets: ["PinyinEngine"]),
    ],
    targets: [
        .target(name: "CLlama", path: "Sources/CLlama", linkerSettings: llamaLinker),
        .target(
            name: "PinyinEngine",
            dependencies: ["CLlama"],
            swiftSettings: [.unsafeFlags(["-wmo"], .when(configuration: .release))]
        ),
        .executableTarget(
            name: "SmartPinyin",
            dependencies: ["PinyinEngine"],
            linkerSettings: [.linkedFramework("InputMethodKit"), .linkedFramework("AppKit"), .linkedFramework("Carbon")]
        ),
        .executableTarget(name: "stbench", dependencies: ["PinyinEngine"]),
        .testTarget(name: "PinyinEngineTests", dependencies: ["PinyinEngine"]),
    ],
    swiftLanguageModes: [.v5]
)
