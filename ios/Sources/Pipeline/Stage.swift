import CoreAI
import Foundation

/// The shape and dtype of one input or output of a Core AI function.
struct TensorSpec: Sendable, Equatable {
    let shape: [Int]
    let scalarType: NDArray.ScalarType

    var count: Int { shape.reduce(1, *) }
}

/// What a stage reads: a host-built value (log-mel window, masks) or another stage's output.
enum StageValue: Sendable {
    case host([Float], shape: [Int])
    case tensor(NDArray)
}

/// One Core AI bundle of the pipeline, loaded once.
///
/// A function never gets two calls at the same time: running the same function twice concurrently corrupts its
/// outputs (measured on the GPU and the Neural Engine alike). The actor alone does not prevent that (an actor is
/// re-entered at `run`'s suspension point), so calls also take a permit, one at a time. Different stages may run
/// concurrently; that is how the transcriber overlaps the encoder of one window with the reader of the previous one.
actor Stage {
    nonisolated let spec: StageSpec
    nonisolated let url: URL
    nonisolated let inputs: [String: TensorSpec]
    nonisolated let outputs: [String: TensorSpec]
    nonisolated let loadSeconds: Double
    /// AOT (`.aimodelc`, compiled ahead of time for this chip) or JIT (`.aimodel`, specialized at load).
    nonisolated var compiledAhead: Bool { url.pathExtension == "aimodelc" }

    private let model: AIModel        // kept alive for as long as the function is used
    private let function: InferenceFunction
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    private init(spec: StageSpec, url: URL, model: AIModel, function: InferenceFunction,
                 inputs: [String: TensorSpec], outputs: [String: TensorSpec], loadSeconds: Double) {
        self.spec = spec
        self.url = url
        self.model = model
        self.function = function
        self.inputs = inputs
        self.outputs = outputs
        self.loadSeconds = loadSeconds
    }

    static func specialization(_ compute: String) throws -> SpecializationOptions {
        switch compute {
        case "gpu": return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case "ane": return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case "cpu": return SpecializationOptions(preferredComputeUnitKind: .cpu)
        case "default": return .default
        default: throw TranscribeError.model("compute is gpu | ane | cpu | default, not '\(compute)'")
        }
    }

    /// Opens (AOT) or specializes (JIT) the bundle and reads its function's contract.
    static func load(_ spec: StageSpec, url: URL) async throws -> Stage {
        let options = try specialization(spec.compute)
        let start = DispatchTime.now().uptimeNanoseconds
        let model = try await AIModel(contentsOf: url, options: options)
        guard let function = try model.loadFunction(named: "main") else {
            throw TranscribeError.contract("\(url.lastPathComponent) has no function 'main'")
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let descriptor = function.descriptor
        guard descriptor.stateNames.isEmpty else {
            throw TranscribeError.contract("\(url.lastPathComponent): stateful functions are not supported")
        }
        var inputs: [String: TensorSpec] = [:]
        for name in descriptor.inputNames {
            guard case .ndArray(let d) = descriptor.inputDescriptor(of: name), !d.hasDynamicShape else {
                throw TranscribeError.contract("\(url.lastPathComponent): input \(name) is not a static array")
            }
            inputs[name] = TensorSpec(shape: d.shape, scalarType: d.scalarType)
        }
        var outputs: [String: TensorSpec] = [:]
        for name in descriptor.outputNames {
            guard case .ndArray(let d) = descriptor.outputDescriptor(of: name) else {
                throw TranscribeError.contract("\(url.lastPathComponent): output \(name) is not an array")
            }
            outputs[name] = TensorSpec(shape: d.shape, scalarType: d.scalarType)
        }
        return Stage(spec: spec, url: url, model: model, function: function, inputs: inputs, outputs: outputs,
                     loadSeconds: seconds)
    }

    /// The function's inputs from the window's values, each a fresh array of the function's dtype (a float32
    /// output of one stage becomes float16 here when the next stage takes float16).
    nonisolated func feed(_ values: [String: StageValue]) throws -> [String: NDArray] {
        var feed: [String: NDArray] = [:]
        for (name, spec) in inputs {
            guard let value = values[name] else {
                throw TranscribeError.contract("\(spec.shape) input '\(name)' of \(self.spec.bundle): no stage or host value")
            }
            switch value {
            case .host(let floats, let shape):
                guard shape == spec.shape else {
                    throw TranscribeError.contract("\(name): host shape \(shape), \(self.spec.bundle) takes \(spec.shape)")
                }
                feed[name] = try Tensor.make(floats, shape: shape, as: spec.scalarType)
            case .tensor(let array):
                guard array.shape == spec.shape else {
                    throw TranscribeError.contract("\(name): \(array.shape) in, \(self.spec.bundle) takes \(spec.shape)")
                }
                feed[name] = try Tensor.copy(array, as: spec.scalarType)
            }
        }
        return feed
    }

    /// One call (waits for the previous call of this function to finish first). Returns the outputs and the
    /// wall milliseconds of the call itself.
    func run(_ feed: [String: NDArray]) async throws -> (outputs: [String: NDArray], milliseconds: Double) {
        await acquire()
        defer { release() }
        let start = DispatchTime.now().uptimeNanoseconds
        var result = try await function.run(inputs: feed)
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        var outputs: [String: NDArray] = [:]
        for name in self.outputs.keys {
            guard let array = result.remove(name)?.ndArray else {
                throw TranscribeError.contract("\(spec.bundle) returned no '\(name)'")
            }
            outputs[name] = array
        }
        return (outputs, milliseconds)
    }

    /// One call on neutral inputs (masks all ones, everything else zero), right after the load: the first call of a
    /// freshly specialized function is the slow one, and it should not land on the user's first recording.
    func warmUp() async throws -> Double {
        var values: [String: StageValue] = [:]
        for (name, spec) in inputs {
            let fill: Float = name.contains("mask") ? 1 : 0
            values[name] = .host([Float](repeating: fill, count: spec.count), shape: spec.shape)
        }
        return try await run(feed(values)).milliseconds
    }

    private func acquire() async {
        if !busy {
            busy = true
            return
        }
        await withCheckedContinuation { waiting.append($0) }
    }

    private func release() {
        if waiting.isEmpty {
            busy = false
        } else {
            waiting.removeFirst().resume()
        }
    }
}
