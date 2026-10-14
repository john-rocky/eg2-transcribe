import Foundation

/// One Core AI bundle of a pipeline form: which bundle (its name without extension, looked up in the platform's
/// subfolder of the model folder) and which compute unit to prefer when Core AI specializes it.
struct StageSpec: Codable, Sendable, Hashable {
    let name: String
    let bundle: String
    let compute: String
}

/// A way to run the transcriber: the stages in order. Values flow between stages by tensor name (the encoder's
/// `soft_tokens` into the reader, a split encoder's `hidden` from its front to its body).
struct PipelineForm: Codable, Sendable, Hashable, Identifiable {
    let id: String
    let title: String
    let stages: [StageSpec]
    /// Platform families the form is meant for ("macos", "ios"); nil = every platform.
    let platforms: [String]?
}

/// pipeline.json at the root of the model folder.
struct PipelineFile: Codable, Sendable {
    let schema: Int
    /// The form to start with: one id, or one per platform family (`{"macos": "gpu-ane", "ios": "gpu"}`).
    let defaultForm: PlatformValue<String>
    /// Whether consecutive windows overlap by default, likewise (two bundles at once help the Mac and not the
    /// iPhone, whose GPU is shared by both calls).
    let overlap: PlatformValue<Bool>
    let forms: [PipelineForm]

    enum CodingKeys: String, CodingKey {
        case schema, overlap, forms
        case defaultForm = "default_form"
    }
}

/// A setting given once for every platform, or per platform family ("macos", "ios").
enum PlatformValue<Value: Codable & Sendable>: Codable, Sendable {
    case all(Value)
    case perPlatform([String: Value])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Value.self) {
            self = .all(value)
        } else {
            self = .perPlatform(try container.decode([String: Value].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .all(let value): try container.encode(value)
        case .perPlatform(let values): try container.encode(values)
        }
    }

    /// The value for this device's platform family (nil when the family is not listed).
    var value: Value? {
        switch self {
        case .all(let value): value
        case .perPlatform(let values): values[ModelFolder.platformFamily]
        }
    }
}

/// The parts of reader_config.json the app uses: the window contract, the vocabulary file and its size.
struct ReaderConfig: Sendable {
    let version: String
    let windowSeconds: Double
    let leftOverlapTokens: Int
    let rightExtraTokens: Int
    let vocabFile: String
    let vocabSize: Int?

    init(url: URL) throws {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let window = json["window"] as? [String: Any],
              let seconds = (window["seconds"] as? NSNumber)?.doubleValue,
              let left = (window["left_overlap_tokens"] as? NSNumber)?.intValue,
              let right = (window["right_extra_tokens"] as? NSNumber)?.intValue else {
            throw TranscribeError.model("reader_config.json has no window {seconds, left_overlap_tokens, right_extra_tokens}")
        }
        version = json["version"] as? String ?? "?"
        windowSeconds = seconds
        leftOverlapTokens = left
        rightExtraTokens = right
        vocabFile = json["vocab"] as? String ?? "vocab.txt"
        vocabSize = (json["n_vocab"] as? NSNumber)?.intValue
    }

    /// The window contract for an encoder that takes `frames` log-mel frames per call.
    func contract(encoderFrames frames: Int) throws -> WindowContract {
        let expected = Int((windowSeconds * 100).rounded())
        guard frames == expected else {
            throw TranscribeError.contract("the encoder takes \(frames) frames, reader_config.json says \(windowSeconds) s = \(expected)")
        }
        return WindowContract(frames: frames, leftOverlap: leftOverlapTokens, dropLast: rightExtraTokens)
    }
}

enum Vocabulary {
    /// vocab.txt: one word per line; the line number is the word's index, blank = the word count.
    static func load(_ url: URL) throws -> [String] {
        let text = try String(contentsOf: url, encoding: .utf8)
        return text.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }
}
