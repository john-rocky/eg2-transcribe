import CoreAI
import Foundation

/// The whole transcriber: audio file → 16 kHz mono → log-mel (whole file, once) → 30 s windows → the pipeline's
/// stages per window → labels of each window's kept tokens → greedy CTC → words.
///
/// Stages run in order on a window. With `overlap`, consecutive windows are staggered: stage k + 1 of window n runs
/// at the same time as stage k of window n + 1 (two windows in flight for an encoder + reader pipeline), each stage
/// still one call at a time. The labels are the same either way; only the wall clock changes.
final class Transcriber: Sendable {
    let form: PipelineForm
    let stages: [Stage]
    let contract: WindowContract
    let vocabulary: [String]
    let extractor: LogMelExtractor
    /// Load (open or specialize) seconds per stage, then the warm-up call's milliseconds per stage.
    let loadSeconds: [Double]
    let warmUpMilliseconds: [Double]
    /// The model folder's fingerprint check before the load (and whether Core AI's cache was cleared).
    let cacheCheck: ModelCache.Check

    private init(form: PipelineForm, stages: [Stage], contract: WindowContract, vocabulary: [String],
                 extractor: LogMelExtractor, loadSeconds: [Double], warmUpMilliseconds: [Double],
                 cacheCheck: ModelCache.Check) {
        self.form = form
        self.stages = stages
        self.contract = contract
        self.vocabulary = vocabulary
        self.extractor = extractor
        self.loadSeconds = loadSeconds
        self.warmUpMilliseconds = warmUpMilliseconds
        self.cacheCheck = cacheCheck
    }

    var blankID: Int { vocabulary.count }

    /// Loads every stage of `form` from the model folder, checks the contracts end to end, and warms each stage up.
    /// Before anything is loaded, Core AI's cache is cleared if the folder is not the one loaded last (ModelCache).
    static func load(folder: ModelFolder, form: PipelineForm, cache: ModelCache.Mode = .check,
                     status: @Sendable (String) -> Void = { _ in }) async throws -> Transcriber {
        let cacheCheck = try ModelCache.prepare(folder, mode: cache)
        status(cacheCheck.summary)
        let vocabulary = try Vocabulary.load(folder.root.appending(path: folder.reader.vocabFile))
        if let expected = folder.reader.vocabSize, expected != vocabulary.count {
            throw TranscribeError.model("vocab.txt has \(vocabulary.count) words, reader_config.json says \(expected)")
        }
        let extractor = try LogMelExtractor(filtersURL: folder.root.appending(path: LogMelExtractor.filtersFile))
        var stages: [Stage] = []
        for spec in form.stages {
            guard let url = folder.bundleURL(spec.bundle) else {
                throw TranscribeError.model("no \(spec.bundle).aimodelc or .aimodel in \(folder.bundleFolders.map(\.lastPathComponent))")
            }
            status("Loading \(spec.name) (\(url.lastPathComponent), \(spec.compute))")
            stages.append(try await Stage.load(spec, url: url))
        }
        guard let features = stages.first?.inputs["features"], features.shape.count == 3,
              features.shape[2] == LogMelExtractor.mels else {
            throw TranscribeError.contract("the first stage takes no features [1, frames, \(LogMelExtractor.mels)]")
        }
        let contract = try folder.reader.contract(encoderFrames: features.shape[1])
        guard let last = stages.last else { throw TranscribeError.model("form \(form.id) has no stages") }
        for name in ["top1_index", "top1_logit", "blank_logit"] {
            guard last.outputs[name]?.shape == [1, contract.tokens] else {
                throw TranscribeError.contract("the last stage returns no \(name) [1, \(contract.tokens)]")
            }
        }
        var warm: [Double] = []
        for stage in stages {
            status("Warming up \(stage.spec.name)")
            warm.append(try await stage.warmUp())
        }
        return Transcriber(form: form, stages: stages, contract: contract, vocabulary: vocabulary,
                           extractor: extractor, loadSeconds: stages.map(\.loadSeconds), warmUpMilliseconds: warm,
                           cacheCheck: cacheCheck)
    }

    // MARK: - One recording

    struct Result: Sendable {
        let audioSeconds: Double
        let decodeSeconds: Double
        let logMelSeconds: Double
        let logMel: LogMelExtractor.Breakdown
        let frames: Int
        let run: Run

        /// Decode + log-mel + windows.
        var totalSeconds: Double { decodeSeconds + logMelSeconds + run.wallSeconds }
        /// Audio seconds per wall second, end to end.
        var realtimeFactor: Double { audioSeconds / totalSeconds }
        var text: String { run.text }
    }

    func transcribe(_ url: URL, overlap: Bool, progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in })
        async throws -> Result {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let audio = try AudioDecoder.decode(url)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let (features, breakdown) = extractor.featuresTimed(audio.samples)
        let t2 = DispatchTime.now().uptimeNanoseconds
        let frames = features.count / LogMelExtractor.mels
        let run = try await self.run(features: features, frames: frames, overlap: overlap, progress: progress)
        return Result(audioSeconds: audio.seconds, decodeSeconds: Double(t1 - t0) / 1e9,
                      logMelSeconds: Double(t2 - t1) / 1e9, logMel: breakdown, frames: frames, run: run)
    }

    // MARK: - The windows

    /// The host's share of the windows, seconds: building each window's inputs (log-mel rows, masks), handing values
    /// to the stages (dtype conversion and copies), reading the labels, and the final collapse into words.
    struct HostTiming: Sendable {
        var inputs = 0.0
        var feed = 0.0
        var labels = 0.0
        var collapse = 0.0

        var total: Double { inputs + feed + labels + collapse }
    }

    struct Run: Sendable {
        let windows: Int
        let tokens: Int
        let overlap: Bool
        /// One label per token of the recording (blank = the vocabulary size).
        let labels: [Int]
        /// Per token: the reader's top-1 word logit and blank logit (what each label was decided from).
        let topLogits: [Float]
        let blankLogits: [Float]
        let words: [Int]
        let text: String
        let wallSeconds: Double
        /// Per stage, per window: the call's wall milliseconds.
        let stageMilliseconds: [[Double]]
        let host: HostTiming
        /// Outputs kept for inspection (`capture`), per window.
        let captured: [[String: NDArray]]
    }

    /// Runs the windows of `frames` log-mel rows ([frames × 128] row-major) through the stages.
    func run(features: [Float], frames: Int, overlap: Bool, capture: Set<String> = [],
             progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in }) async throws -> Run {
        let plan = Windowing.plan(frames: frames, contract: contract)
        let total = Windowing.totalTokens(plan)
        var labels = [Int](repeating: -1, count: total)
        var tops = [Float](repeating: .nan, count: total)
        var blanks = [Float](repeating: .nan, count: total)
        var milliseconds = [[Double]](repeating: [Double](repeating: .nan, count: plan.count), count: stages.count)
        var captured = [[String: NDArray]](repeating: [:], count: capture.isEmpty ? 0 : plan.count)
        var host = HostTiming()
        let last = stages.count - 1
        let start = DispatchTime.now().uptimeNanoseconds
        func seconds(since t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9 }

        func finish(_ values: [String: StageValue], _ window: Window) throws {
            let t = DispatchTime.now().uptimeNanoseconds
            try readLabels(values, window, labels: &labels, tops: &tops, blanks: &blanks)
            host.labels += seconds(since: t)
            if !capture.isEmpty {
                for name in capture { if case .tensor(let a) = values[name] { captured[window.index][name] = a } }
            }
            progress(window.index + 1, plan.count)
        }

        if !overlap || stages.count == 1 {
            for window in plan {
                var t = DispatchTime.now().uptimeNanoseconds
                var values = hostValues(window, features: features)
                host.inputs += seconds(since: t)
                for (k, stage) in stages.enumerated() {
                    t = DispatchTime.now().uptimeNanoseconds
                    let feed = try stage.feed(values)
                    host.feed += seconds(since: t)
                    let (outputs, ms) = try await stage.run(feed)
                    milliseconds[k][window.index] = ms
                    for (name, array) in outputs { values[name] = .tensor(array) }
                }
                try finish(values, window)
            }
        } else {
            // Wavefront: at step t, stage k works on window t − k; every stage of a step runs at once.
            var pending: [Int: [String: StageValue]] = [:]
            for step in 0..<(plan.count + last) {
                var t = DispatchTime.now().uptimeNanoseconds
                if step < plan.count { pending[step] = hostValues(plan[step], features: features) }
                host.inputs += seconds(since: t)
                t = DispatchTime.now().uptimeNanoseconds
                var jobs: [(stage: Int, window: Int, feed: [String: NDArray])] = []
                for k in 0...last {
                    let n = step - k
                    if n >= 0, n < plan.count, let values = pending[n] {
                        jobs.append((k, n, try stages[k].feed(values)))
                    }
                }
                host.feed += seconds(since: t)
                let stages = self.stages
                let results = try await withThrowingTaskGroup(
                    of: (stage: Int, window: Int, outputs: [String: NDArray], ms: Double).self
                ) { group in
                    for job in jobs {
                        let stage = stages[job.stage]
                        group.addTask {
                            let (outputs, ms) = try await stage.run(job.feed)
                            return (job.stage, job.window, outputs, ms)
                        }
                    }
                    var done: [(stage: Int, window: Int, outputs: [String: NDArray], ms: Double)] = []
                    for try await result in group { done.append(result) }
                    return done
                }
                for result in results.sorted(by: { $0.stage > $1.stage }) {
                    milliseconds[result.stage][result.window] = result.ms
                    guard var values = pending[result.window] else { continue }
                    for (name, array) in result.outputs { values[name] = .tensor(array) }
                    if result.stage == last {
                        try finish(values, plan[result.window])
                        pending[result.window] = nil
                    } else {
                        pending[result.window] = values
                    }
                }
            }
        }
        guard !labels.contains(-1) else {
            throw TranscribeError.contract("the windows left \(labels.filter { $0 < 0 }.count) tokens without a label")
        }
        let t = DispatchTime.now().uptimeNanoseconds
        let words = Greedy.collapse(labels, blank: blankID)
        let text = words.map { vocabulary[$0] }.joined(separator: " ")
        host.collapse = seconds(since: t)
        let wall = seconds(since: start)
        return Run(windows: plan.count, tokens: total, overlap: overlap && stages.count > 1, labels: labels,
                   topLogits: tops, blankLogits: blanks, words: words, text: text, wallSeconds: wall,
                   stageMilliseconds: milliseconds, host: host, captured: captured)
    }

    /// The host's inputs of one window: its log-mel rows (zero-padded to the window), half_mask (1 on the
    /// ceil(frames / 2) rows the first convolution keeps), token_mask (1 on the valid tokens).
    private func hostValues(_ window: Window, features: [Float]) -> [String: StageValue] {
        let mels = LogMelExtractor.mels
        var x = [Float](repeating: 0, count: contract.frames * mels)
        let from = window.frame0 * mels
        let count = window.frames * mels
        x.withUnsafeMutableBufferPointer { dst in
            features.withUnsafeBufferPointer { src in
                dst.baseAddress!.update(from: src.baseAddress! + from, count: count)
            }
        }
        let halfRows = (contract.frames + 1) / 2
        var half = [Float](repeating: 0, count: halfRows)
        for i in 0..<((window.frames + 1) / 2) { half[i] = 1 }
        var tokens = [Float](repeating: 0, count: contract.tokens)
        for i in 0..<window.validTokens { tokens[i] = 1 }
        return ["features": .host(x, shape: [1, contract.frames, mels]),
                "half_mask": .host(half, shape: [1, halfRows]),
                "token_mask": .host(tokens, shape: [1, contract.tokens])]
    }

    private func readLabels(_ values: [String: StageValue], _ window: Window, labels: inout [Int], tops: inout [Float],
                            blanks: inout [Float]) throws {
        guard case .tensor(let index) = values["top1_index"], case .tensor(let top) = values["top1_logit"],
              case .tensor(let blank) = values["blank_logit"] else {
            throw TranscribeError.contract("the last stage gave no top1_index / top1_logit / blank_logit")
        }
        let indices = try Tensor.int32s(index)
        let topValues = try Tensor.floats(top)
        let blankValues = try Tensor.floats(blank)
        for p in window.keep {
            guard indices[p] >= 0, Int(indices[p]) < blankID else {
                throw TranscribeError.contract("top1_index \(indices[p]) outside the \(blankID)-word vocabulary")
            }
            let token = window.token0 + p
            labels[token] = Greedy.label(index: indices[p], top: topValues[p], blank: blankValues[p], blankID: blankID)
            tops[token] = topValues[p]
            blanks[token] = blankValues[p]
        }
    }
}
