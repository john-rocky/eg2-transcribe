import Foundation

enum TranscribeError: LocalizedError {
    case audio(String)
    case model(String)
    case contract(String)
    case download(String)

    var errorDescription: String? {
        switch self {
        case .audio(let message): "Audio: \(message)"
        case .model(let message): "Model folder: \(message)"
        case .contract(let message): "Model contract: \(message)"
        case .download(let message): "Download: \(message)"
        }
    }
}
