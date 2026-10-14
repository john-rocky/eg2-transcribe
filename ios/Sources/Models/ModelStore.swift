import Foundation
import Hub

/// Finds the model folder on this device, or downloads it from Hugging Face.
enum ModelStore {
    /// A folder put on the device by hand: `Documents/models/eg2-transcribe` (on the iPhone through the Files app or
    /// `devicectl device copy to`; on the Mac `~/Documents/models/eg2-transcribe`).
    static var localFolder: URL {
        URL.documentsDirectory.appending(path: "models/eg2-transcribe", directoryHint: .isDirectory)
    }

    /// Downloads go to Caches: they can be fetched again, so they are not backed up.
    static let downloadBase = URL.cachesDirectory.appending(path: "huggingface", directoryHint: .isDirectory)

    static var downloadedFolder: URL {
        HubApi(downloadBase: downloadBase).localRepoLocation(HubApi.Repo(id: ModelFiles.repo))
    }

    /// The first folder that holds a model folder: an explicit one (a launch argument or the folder picked on
    /// the Mac), then the hand-copied one, then the download.
    static func locate(preferred: URL? = nil) -> URL? {
        let candidates = [preferred, localFolder, ModelFiles.isPublished ? downloadedFolder : nil].compactMap { $0 }
        return candidates.first { FileManager.default.fileExists(atPath: $0.appending(path: "pipeline.json").path) }
    }

    /// Downloads the shared files and this platform's bundle folder (on an iPhone the AOT folder for its chip when
    /// the repo has one, else `ios/`). `progress` gets the fraction done.
    static func download(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        guard ModelFiles.isPublished else {
            throw TranscribeError.download("no Hugging Face repo yet (ModelFiles.repo); copy the model folder to \(localFolder.path)")
        }
        let hub = HubApi(downloadBase: downloadBase, cache: nil, useOfflineMode: false)
        let repo = HubApi.Repo(id: ModelFiles.repo)
        let names = try await hub.getFilenames(from: repo, revision: ModelFiles.revision)
        guard let folder = ModelFolder.platformFolderNames.first(where: { name in names.contains { $0.hasPrefix(name + "/") } }) else {
            throw TranscribeError.download("\(ModelFiles.repo) has none of \(ModelFolder.platformFolderNames)")
        }
        let url = try await hub.snapshot(from: repo, revision: ModelFiles.revision,
                                         matching: ModelFiles.sharedFiles + ["\(folder)/*"]) { @Sendable p in
            progress(p.fractionCompleted)
        }
        return url
    }
}
