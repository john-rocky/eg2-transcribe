import SwiftUI

/// The entry point. On the Mac, `EG2Transcribe.app/Contents/MacOS/EG2Transcribe --selftest <command> …` from a
/// terminal runs the self-test headless and exits (0 = DONE, 1 = error); anything else opens the app. On the
/// iPhone the app always opens and runs a self-test or an autoplay from its launch arguments (see Autoplay.swift).
@main
enum EG2TranscribeMain {
    @MainActor static func main() {
        #if os(macOS)
        let base = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        if let test = SelfTest(arguments: Array(CommandLine.arguments.dropFirst()), base: base) {
            Task.detached {
                do {
                    let done = try await test.run { FileHandle.standardError.write(Data($0.utf8)) }
                    print(done)
                    exit(0)
                } catch {
                    exit(1)
                }
            }
            dispatchMain()
        }
        #endif
        EG2TranscribeApp.main()
    }
}

struct EG2TranscribeApp: App {
    @State private var model = AppModel()
    @State private var launch = LaunchRunner()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .task { await launch.run(model) }
        }
        #if os(macOS)
        .defaultSize(width: 560, height: 720)
        #endif
    }
}
