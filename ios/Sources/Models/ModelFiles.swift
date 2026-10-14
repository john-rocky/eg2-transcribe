import CoreAI
import Foundation

/// The model folder, laid out as the Hugging Face repo is:
///
///     pipeline.json        the forms (stage lists) the app can run, the default form and overlap
///     reader_config.json   the reader's window contract and vocabulary file
///     vocab.txt            one word per line (line number = the reader's word index)
///     mel_filters.bin      the log-mel filter bank, 257 × 128 float64
///     SHA256SUMS           the SHA-256 of every file (optional; part of the folder's fingerprint, see ModelCache)
///     macos/               bundles for the Mac: .aimodel, specialized for the Mac's chip at the first load
///     ios/                 the same .aimodel bundles for any iPhone
///     ios-<arch>/          the same bundles compiled ahead of time for one iPhone chip (.aimodelc), e.g. ios-h19p
///
/// A bundle is looked up by name in the platform's folders, most specific first: on an iPhone `ios-<arch>/` (if
/// the folder has this chip's AOT build) and then `ios/`; on the Mac `macos/`.
struct ModelFolder: Sendable {
    let root: URL
    let pipeline: PipelineFile
    let reader: ReaderConfig
    /// Existing platform folders, most specific first.
    let bundleFolders: [URL]

    /// "macos" or "ios": which forms of pipeline.json apply here.
    static var platformFamily: String {
        #if os(macOS)
        "macos"
        #else
        "ios"
        #endif
    }

    /// The platform folder names to try, most specific first.
    static var platformFolderNames: [String] {
        #if os(macOS)
        ["macos"]
        #else
        ["ios-\(AIModel.deviceArchitectureName)", "ios"]
        #endif
    }

    init(root: URL) throws {
        self.root = root
        let pipelineURL = root.appending(path: "pipeline.json")
        guard FileManager.default.fileExists(atPath: pipelineURL.path) else {
            throw TranscribeError.model("no pipeline.json in \(root.path)")
        }
        pipeline = try JSONDecoder().decode(PipelineFile.self, from: Data(contentsOf: pipelineURL))
        reader = try ReaderConfig(url: root.appending(path: "reader_config.json"))
        bundleFolders = Self.platformFolderNames
            .map { root.appending(path: $0, directoryHint: .isDirectory) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !bundleFolders.isEmpty else {
            throw TranscribeError.model("none of \(Self.platformFolderNames.map { $0 + "/" }) in \(root.path)")
        }
    }

    /// `<folder>/<name>.aimodelc` or `<folder>/<name>.aimodel` in the first platform folder that has either.
    /// The path is resolved to a real directory: Core AI refuses an AOT bundle opened through a symbolic link.
    func bundleURL(_ name: String) -> URL? {
        for folder in bundleFolders {
            for ext in ["aimodelc", "aimodel"] {
                let url = folder.appending(path: "\(name).\(ext)", directoryHint: .isDirectory)
                if FileManager.default.fileExists(atPath: url.path) { return url.resolvingSymlinksInPath() }
            }
        }
        return nil
    }

    /// The forms of pipeline.json meant for this platform whose bundles are all in the folder.
    var availableForms: [PipelineForm] {
        pipeline.forms.filter { form in
            (form.platforms?.contains(Self.platformFamily) ?? true) && form.stages.allSatisfy { bundleURL($0.bundle) != nil }
        }
    }

    var defaultForm: PipelineForm? {
        let forms = availableForms
        return forms.first { $0.id == pipeline.defaultForm.value } ?? forms.first
    }

    /// Whether windows overlap by default on this platform.
    var defaultOverlap: Bool { pipeline.overlap.value ?? false }
}

/// Where the model folder comes from: Hugging Face (one repo id) or a folder on this device.
enum ModelFiles {
    /// The Hugging Face repo that holds the model folder (published 2026-10-14). A folder at
    /// `ModelStore.localFolder` takes precedence when it is present.
    static let repo = "mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI"
    static let revision = "main"
    static var isPublished: Bool { !repo.hasPrefix("TODO") }

    /// Files every platform needs, and the platform's bundle folders.
    static let sharedFiles = ["pipeline.json", "reader_config.json", "vocab.txt", "mel_filters.bin", "SHA256SUMS"]
}
