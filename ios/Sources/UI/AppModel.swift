import Foundation
import Observation

/// What one transcription took, for the stats line.
struct RunStats: Sendable, Equatable {
    let file: String
    let audioSeconds: Double
    let totalSeconds: Double
    let decodeSeconds: Double
    let logMelSeconds: Double
    let windowSeconds: Double
    /// Per stage: name and the sum of its calls' seconds.
    let stageSeconds: [String]
    let stageSums: [Double]
    /// The host's share of the windows: inputs, conversions, labels, collapse.
    let hostSeconds: Double
    let windows: Int
    let form: String
    let overlap: Bool

    var realtimeFactor: Double { audioSeconds / totalSeconds }
}

/// The app's state: the model folder, the loaded pipeline, the last transcript and its numbers.
@MainActor
@Observable
final class AppModel {
    enum Phase: Equatable {
        case starting
        case noModel(String)
        case downloading(Double)
        case loading(String)
        case ready
        case recording
        case transcribing(window: Int, of: Int)
        case failed(String)
    }

    var phase: Phase = .starting
    private(set) var folder: ModelFolder?
    var forms: [PipelineForm] = []
    var formID = ""
    var overlap = true
    var transcript = ""
    var stats: RunStats?
    var selfTestLine: String?
    /// The transcript's text size in points (0 = the platform's body size), kept in the user defaults
    /// (`-textSize 44` as a launch argument sets it for one run).
    var textSize = UserDefaults.standard.double(forKey: "textSize") {
        didSet { UserDefaults.standard.set(textSize, forKey: "textSize") }
    }

    private var transcriber: Transcriber?
    private let recorder = AudioRecorder()
    private static let folderKey = "modelFolderPath"

    var canTranscribe: Bool {
        if case .ready = phase { return transcriber != nil }
        return false
    }

    var isBusy: Bool {
        switch phase {
        case .loading, .downloading, .transcribing: true
        default: false
        }
    }

    var form: PipelineForm? { forms.first { $0.id == formID } }

    // MARK: - Model folder

    /// Opens the first model folder found (a `-models <dir>` launch argument, the one picked last time,
    /// Documents/models/eg2-transcribe, the download) and loads the default form.
    func start() async {
        let defaults = UserDefaults.standard
        let preferred = (defaults.string(forKey: "models") ?? defaults.string(forKey: Self.folderKey))
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
        await open(ModelStore.locate(preferred: preferred))
    }

    func open(_ root: URL?) async {
        guard let root else {
            phase = .noModel(ModelFiles.isPublished
                ? "Download the model, or copy the model folder to \(ModelStore.localFolder.path)."
                : "Copy the model folder to \(ModelStore.localFolder.path).")
            return
        }
        do {
            let folder = try ModelFolder(root: root)
            self.folder = folder
            forms = folder.availableForms
            guard let form = folder.defaultForm else {
                phase = .failed("No form of pipeline.json can run on this device with the bundles in \(root.lastPathComponent).")
                return
            }
            formID = form.id
            overlap = folder.defaultOverlap
            UserDefaults.standard.set(root.path, forKey: Self.folderKey)
            await load()
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Loads (or reloads, after a settings change) the selected form's stages.
    func load() async {
        guard let folder, let form else { return }
        transcriber = nil
        phase = .loading("Loading \(form.title)")
        do {
            transcriber = try await Transcriber.load(folder: folder, form: form) { message in
                Task { @MainActor in self.phase = .loading(message) }
            }
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func selectForm(_ id: String) async {
        guard id != formID, forms.contains(where: { $0.id == id }) else { return }
        formID = id
        await load()
    }

    func download() async {
        phase = .downloading(0)
        do {
            let root = try await ModelStore.download { fraction in
                Task { @MainActor in self.phase = .downloading(fraction) }
            }
            await open(root)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    // MARK: - Transcribing

    /// Transcribes one audio file with the loaded form; returns the result (nil after an error, shown on screen).
    @discardableResult
    func transcribe(_ url: URL) async -> Transcriber.Result? {
        guard let transcriber else { return nil }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        phase = .transcribing(window: 0, of: 0)
        transcript = ""
        do {
            let result = try await transcriber.transcribe(url, overlap: overlap) { done, total in
                Task { @MainActor in self.phase = .transcribing(window: done, of: total) }
            }
            transcript = result.text
            let names = transcriber.stages.map(\.spec.name)
            let sums = result.run.stageMilliseconds.map { $0.filter(\.isFinite).reduce(0, +) / 1000 }
            stats = RunStats(file: url.lastPathComponent, audioSeconds: result.audioSeconds, totalSeconds: result.totalSeconds,
                             decodeSeconds: result.decodeSeconds, logMelSeconds: result.logMelSeconds,
                             windowSeconds: result.run.wallSeconds, stageSeconds: names, stageSums: sums,
                             hostSeconds: result.run.host.total, windows: result.run.windows,
                             form: transcriber.form.title, overlap: result.run.overlap)
            phase = .ready
            return result
        } catch {
            phase = .failed(error.localizedDescription)
            return nil
        }
    }

    var isRecording: Bool { phase == .recording }

    func toggleRecording() async {
        if isRecording {
            phase = .ready
            if let url = recorder.stop() { await transcribe(url) }
            return
        }
        guard canTranscribe else { return }
        guard await recorder.permission() else {
            phase = .failed("Microphone access is off: allow it in Settings to record.")
            return
        }
        do {
            try recorder.start()
            phase = .recording
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Back to ready after an error, when a pipeline is loaded.
    func dismissError() {
        if transcriber != nil { phase = .ready }
    }
}
