import Foundation
import Observation
#if os(iOS)
import UIKit
#else
import AppKit
#endif

/// What the app does at launch besides showing itself, from its launch arguments (with devicectl, after `--`):
///
/// - `--selftest <command> …` (iPhone): the headless self-test of SelfTest.swift, its output in
///   Documents/out/<tag>/. The screen stays on while it runs (a locked phone suspends GPU work).
/// - `-autoplay <file>`: once the model is loaded, transcribe the file as if it had been chosen (a path in
///   Documents, or absolute), after `-hold <s>` seconds (default 2), then write <out>/<tag>/{transcript.txt,
///   stats.json, log.txt} (`-out <dir>`, default Documents/out; `-tag <name>`, default autoplay-<time>).
///   `-overlap 0|1` and `-form <id>` set the settings first, `-models <dir>` picks the model folder. For screen
///   recordings and hands-off checks; it presses the same button a person would. On the Mac the log names the
///   window's number (for `screencapture -l`).
@MainActor
@Observable
final class LaunchRunner {
    private var started = false

    func run(_ model: AppModel) async {
        guard !started else { return }
        started = true
        #if os(iOS)
        if let test = SelfTest(arguments: Array(CommandLine.arguments.dropFirst()), base: URL.documentsDirectory) {
            UIApplication.shared.isIdleTimerDisabled = true
            model.selfTestLine = "Self-test \(test.command) (\(test.tag)) running"
            let done = await Task.detached {
                try? await test.run { line in Task { @MainActor in model.selfTestLine = line } }
            }.value
            model.selfTestLine = done.map { "DONE \($0)" } ?? "ERROR: see out/\(test.tag)/log.txt"
            return
        }
        #endif
        await model.start()
        if let file = Autoplay.file { await Autoplay.run(model, file: file) }
    }
}

@MainActor
enum Autoplay {
    private static var defaults: UserDefaults { .standard }
    static var file: String? { defaults.string(forKey: "autoplay") }

    static func run(_ model: AppModel, file: String) async {
        let tag = defaults.string(forKey: "tag") ?? "autoplay-\(Int(Date().timeIntervalSince1970))"
        let outRoot = defaults.string(forKey: "out").map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? URL.documentsDirectory.appending(path: "out", directoryHint: .isDirectory)
        let directory = outRoot.appending(path: tag, directoryHint: .isDirectory)
        guard let log = try? RunLog(directory: directory, sink: { _ in }) else { return }
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = true
        #else
        if let window = NSApplication.shared.windows.first(where: \.isVisible) {
            log.write("WINDOW \(window.windowNumber)")
        }
        #endif
        let url = file.hasPrefix("/") ? URL(fileURLWithPath: file) : URL.documentsDirectory.appending(path: file)
        log.write("autoplay \(url.lastPathComponent) tag=\(tag)")
        if defaults.object(forKey: "overlap") != nil { model.overlap = defaults.bool(forKey: "overlap") }
        if let form = defaults.string(forKey: "form") { await model.selectForm(form) }
        guard model.canTranscribe else {
            log.write("ERROR: the model is not ready (\(model.phase))")
            return
        }
        let hold = defaults.object(forKey: "hold") == nil ? 2 : defaults.double(forKey: "hold")
        try? await Task.sleep(for: .seconds(hold))
        log.write("TAP Choose audio: \(url.lastPathComponent)")
        guard let result = await model.transcribe(url) else {
            log.write("ERROR: \(model.phase)")
            return
        }
        var stats: [String: Any] = [
            "file": url.lastPathComponent, "form": model.formID, "overlap": result.run.overlap,
            "audio_s": result.audioSeconds, "total_s": result.totalSeconds, "x_realtime": result.realtimeFactor,
            "decode_s": result.decodeSeconds, "log_mel_s": result.logMelSeconds, "windows": result.run.windows,
            "windows_wall_s": result.run.wallSeconds, "stage_ms_median": result.run.stageMilliseconds.map { Stats.median($0) },
            "host_s": result.run.host.total, "log_mel_cpu_s": SelfTest.logMelRecord(result.logMel),
            "words": result.run.words.count,
        ]
        stats["device"] = await DeviceProbe.record()
        do {
            try (result.text + "\n").write(to: directory.appending(path: "transcript.txt"), atomically: true, encoding: .utf8)
            try JSONOut.write(stats, to: directory.appending(path: "stats.json"))
            #if os(iOS)
            // The app's own window, after the screen shows the result (a system alert of another app stays out of it).
            try? await Task.sleep(for: .milliseconds(600))
            if let png = windowSnapshot() {
                try png.write(to: directory.appending(path: "screen.png"))
                log.write("SCREEN screen.png")
            }
            #endif
            log.write(String(format: "DONE autoplay %@: %.1f s audio in %.3f s (x%.0f), %ld words", url.lastPathComponent,
                             result.audioSeconds, result.totalSeconds, result.realtimeFactor, result.run.words.count))
        } catch {
            log.write("ERROR: \(error)")
        }
    }

    #if os(iOS)
    /// The key window drawn into a PNG at the screen's scale.
    private static func windowSnapshot() -> Data? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        guard let window = scenes.flatMap(\.windows).first(where: \.isKeyWindow) ?? scenes.first?.windows.first else {
            return nil
        }
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
        return renderer.pngData { _ in _ = window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
    }
    #endif
}
