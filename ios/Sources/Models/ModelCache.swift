import CoreAI
import CryptoKit
import Darwin
import Foundation

/// Core AI keeps what it specializes from a bundle in a cache keyed by the bundle's `main.hash`, and for a bundle
/// compiled ahead of time that hash covers the program, not the weights: a bundle replaced by one with the same program
/// and new weights gets the old weights' specialization back, and with it the old weights' transcripts. So the app
/// remembers a fingerprint of the model folder it loaded and, when the folder it is about to load has another one,
/// deletes its Core AI cache before it loads anything.
enum ModelCache {
    enum Mode: String, Sendable {
        /// Clear the cache when the fingerprint differs from the stored one (or none is stored). The app's mode.
        case check
        /// Clear it whatever the fingerprint (a self-test's cold start).
        case clear
        /// Leave the cache and the stored fingerprint alone (a self-test that shows what a stale cache does).
        case off
    }

    /// What one check found and did.
    struct Check: Sendable {
        let mode: Mode
        let fingerprint: String?
        let previous: String?
        let cleared: Bool
        /// Fingerprint and clearing, milliseconds.
        let milliseconds: Double

        var record: [String: Any] {
            ["mode": mode.rawValue, "fingerprint": fingerprint ?? NSNull(), "previous": previous ?? NSNull(),
             "cleared": cleared, "ms": milliseconds]
        }

        var summary: String {
            switch mode {
            case .off: "model cache: not checked"
            default: "model cache: fingerprint \(fingerprint?.prefix(12) ?? "-") (last \(previous?.prefix(12) ?? "none")), "
                + (cleared ? "Core AI cache cleared" : "unchanged") + String(format: ", %.0f ms", milliseconds)
            }
        }
    }

    static let fingerprintKey = "modelFolderFingerprint"

    /// Compares the folder's fingerprint with the one stored at the last load and, when they differ (or `mode` is
    /// `clear`), deletes this app's Core AI cache; then stores the new fingerprint. Throws when the cache cannot be
    /// cleared: loading on would risk the old weights' results.
    static func prepare(_ folder: ModelFolder, mode: Mode = .check) throws -> Check {
        let start = DispatchTime.now().uptimeNanoseconds
        let defaults = UserDefaults.standard
        let previous = defaults.string(forKey: fingerprintKey)
        guard mode != .off else {
            return Check(mode: mode, fingerprint: nil, previous: previous, cleared: false, milliseconds: 0)
        }
        let current = fingerprint(folder)
        let clearing = mode == .clear || current != previous
        if clearing { try clear(folder) }
        defaults.set(current, forKey: fingerprintKey)
        return Check(mode: mode, fingerprint: current, previous: previous, cleared: clearing,
                     milliseconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
    }

    /// SHA-256 over the folder's SHA256SUMS (the SHA-256 of every file, as published) and the identity of every file
    /// in its bundle folders: path, size, modification time and file number. A file copied over, cloned in or
    /// downloaded again gets a new file number, so a bundle swapped by hand changes the fingerprint too. Only file
    /// metadata is read, not the bundles' bytes.
    static func fingerprint(_ folder: ModelFolder) -> String {
        var hash = SHA256()
        if let sums = try? Data(contentsOf: folder.root.appending(path: "SHA256SUMS")) {
            hash.update(data: sums)
        }
        let root = folder.root.resolvingSymlinksInPath().path
        for bundleFolder in folder.bundleFolders {
            for path in files(under: bundleFolder.resolvingSymlinksInPath()) {
                var info = stat()
                guard stat(path, &info) == 0 else { continue }
                let relative = path.hasPrefix(root) ? String(path.dropFirst(root.count)) : path
                let mtime = "\(info.st_mtimespec.tv_sec).\(info.st_mtimespec.tv_nsec)"
                hash.update(data: Data("\(relative)\t\(info.st_size)\t\(mtime)\t\(info.st_ino)\n".utf8))
            }
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Every regular file below `url`, sorted.
    private static func files(under url: URL) -> [String] {
        guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey]) else {
            return []
        }
        var paths: [String] = []
        while let file = walker.nextObject() as? URL {
            if (try? file.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true { paths.append(file.path) }
        }
        return paths.sorted()
    }

    /// Deletes every Core AI cache entry of this app; if that fails, the entries of the folder's bundles.
    private static func clear(_ folder: ModelFolder) throws {
        do {
            try AIModelCache.default.deleteAll()
        } catch {
            do {
                for url in bundles(folder) { try AIModelCache.default.deleteEntries(for: url) }
            } catch {
                throw TranscribeError.model("could not clear Core AI's cache for the new model folder: \(error)")
            }
        }
    }

    /// The .aimodel / .aimodelc bundles in the folder's bundle folders.
    private static func bundles(_ folder: ModelFolder) -> [URL] {
        folder.bundleFolders.flatMap { dir in
            ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
                .filter { ["aimodel", "aimodelc"].contains($0.pathExtension) }
                .map { $0.resolvingSymlinksInPath() }
        }
    }
}
