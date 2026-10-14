import CoreAI
import Foundation

/// Headless runs for checks and measurements, from launch arguments: the Mac app from the command line, the iPhone
/// app through `xcrun devicectl device process launch … com.coreai.eg2transcribe -- --selftest <command> …`.
/// The same code the app's screen uses; only the inputs and the outputs differ.
///
///   --selftest probe                  the device, thermal and power facts
///   --selftest thermal [--cap S]      the thermal state every 10 s until nominal (at most S s, default 600)
///   --selftest decode --audio F --dump D
///                                     the 16 kHz mono samples of F as float32 little-endian (compare with Python)
///   --selftest cache                  the model folder's fingerprint check alone (with --model-cache clear: clear)
///   --selftest transcribe --audio F   load, warm up, transcribe F; seconds from the process start to the transcript
///   --selftest gate --fixtures DIR [--expect TSV] [--dump-features D]
///                                     every utterance of DIR/MANIFEST.json: the Swift log-mel vs features.npy, the
///                                     soft tokens vs soft_tokens.npy (from the Swift features and from features.npy),
///                                     the transcripts (and TSV's `torch` column: the reference path's transcripts);
///                                     D receives each utterance's log-mel as float32 [frames × 128]
///   --selftest long --audio F [--features NPY] [--soft-reference NPY] [--reference-hyp TXT]
///                   [--reference-labels NPY] [--truth TXT] [--overlap both|on|off] [--passes N] [--compare-minutes M]
///                                     a long recording sequentially and with overlap, N passes each: wall clock,
///                                     milliseconds per window and stage, labels against a reference, word differences;
///                   [--order on,off,… [--cool S]]
///                                     instead, single passes in that order, each started from the thermal state
///                                     nominal (waiting at most S s): overlap on / off compared at the same heat
///
/// Common: --models DIR (default: the app's model folder), --form ID (default: pipeline.json's), --overlap on|off,
/// --model-cache check|clear|off (default check: Core AI's cache is cleared when the folder's fingerprint changed;
/// clear: always; off: neither checked nor cleared, which shows what a stale cache does),
/// --tag NAME (default: the command and the time), --out DIR (default: Documents/out on the iPhone, ./out on the Mac).
/// Relative paths are taken from Documents on the iPhone and from the working directory on the Mac. Every run writes
/// <out>/<tag>/log.txt, flushed line by line, whose last line is `DONE …` or `ERROR: …`, and <out>/<tag>/result.json.
struct SelfTest: Sendable {
    let command: String
    let values: [String: String]
    let base: URL

    /// nil when the arguments hold no `--selftest`.
    init?(arguments: [String], base: URL) {
        var values: [String: String] = [:]
        var rest = arguments.filter { $0 != "--" }[...]
        var command: String?
        while let arg = rest.popFirst() {
            guard arg.hasPrefix("--") else { continue }
            let key = String(arg.dropFirst(2))
            let value = rest.first.map { $0.hasPrefix("--") ? "" : $0 } ?? ""
            if !value.isEmpty { rest = rest.dropFirst() }
            if key == "selftest" { command = value } else { values[key] = value }
        }
        guard let command else { return nil }
        self.command = command
        self.values = values
        self.base = base
    }

    var tag: String {
        values["tag"] ?? "\(command)-\(Int(Date().timeIntervalSince1970))"
    }

    var outDirectory: URL {
        url("out") ?? base.appending(path: "out", directoryHint: .isDirectory)
    }

    func url(_ key: String) -> URL? {
        guard let value = values[key], !value.isEmpty else { return nil }
        return value.hasPrefix("/") ? URL(fileURLWithPath: value) : base.appending(path: value)
    }

    func required(_ key: String) throws -> URL {
        guard let url = url(key) else { throw TranscribeError.model("--\(key) is required for --selftest \(command)") }
        return url
    }

    /// Runs the command; returns the text of the DONE line. Errors are logged as `ERROR: …` and rethrown.
    func run(status: @escaping @Sendable (String) -> Void = { _ in }) async throws -> String {
        let directory = outDirectory.appending(path: tag, directoryHint: .isDirectory)
        let log = try RunLog(directory: directory, sink: status)
        var record: [String: Any] = ["schema": "eg2transcribe-selftest/1", "command": command, "tag": tag,
                                     "arguments": values, "started": ISO8601DateFormatter().string(from: Date())]
        record["device"] = await DeviceProbe.record()
        log.write("selftest \(command) tag=\(tag) device=\(DeviceProbe.machine) arch=\(AIModel.deviceArchitectureName) os=\(DeviceProbe.osBuild) thermal=\(DeviceProbe.thermal())")
        let sampler = RunSampler()
        sampler.begin()
        do {
            let done: String
            switch command {
            case "probe": done = "probe ok"
            case "thermal": done = try await thermal(&record, log)
            case "decode": done = try decode(&record, log)
            case "cache": done = try cache(&record, log)
            case "transcribe": done = try await transcribe(&record, log, directory: directory)
            case "gate": done = try await gate(&record, log)
            case "long": done = try await long(&record, log, directory: directory)
            default: throw TranscribeError.model("unknown --selftest command '\(command)'")
            }
            record["sampler"] = sampler.end()
            record["power_end"] = await DeviceProbe.power()
            record["finished"] = ISO8601DateFormatter().string(from: Date())
            try JSONOut.write(record, to: directory.appending(path: "result.json"))
            log.write("DONE \(done)")
            return done
        } catch {
            record["sampler"] = sampler.end()
            record["error"] = "\(error)"
            record["finished"] = ISO8601DateFormatter().string(from: Date())
            try? JSONOut.write(record, to: directory.appending(path: "result.json"))
            log.write("ERROR: \(error)")
            throw error
        }
    }

    // MARK: - Loading

    private func modelFolder() throws -> ModelFolder {
        guard let root = url("models") ?? ModelStore.locate() else {
            throw TranscribeError.model("no model folder (--models, or \(ModelStore.localFolder.path))")
        }
        return try ModelFolder(root: root)
    }

    private var cacheMode: ModelCache.Mode {
        get throws {
            guard let value = values["model-cache"], !value.isEmpty else { return .check }
            guard let mode = ModelCache.Mode(rawValue: value) else {
                throw TranscribeError.model("--model-cache is check | clear | off, not '\(value)'")
            }
            return mode
        }
    }

    private func load(_ log: RunLog) async throws -> (ModelFolder, Transcriber) {
        let folder = try modelFolder()
        let form: PipelineForm
        if let id = values["form"] {
            guard let f = folder.availableForms.first(where: { $0.id == id }) else {
                throw TranscribeError.model("form '\(id)' is not available here (\(folder.availableForms.map(\.id)))")
            }
            form = f
        } else {
            guard let f = folder.defaultForm else { throw TranscribeError.model("no form of pipeline.json can run here") }
            form = f
        }
        let transcriber = try await Transcriber.load(folder: folder, form: form, cache: try cacheMode) { log.write($0) }
        for (stage, (seconds, warm)) in zip(transcriber.stages, zip(transcriber.loadSeconds, transcriber.warmUpMilliseconds)) {
            log.write(String(format: "stage %@: %@ (%@) load %.3f s, warm-up call %.1f ms", stage.spec.name,
                             stage.url.lastPathComponent, stage.spec.compute, seconds, warm))
        }
        return (folder, transcriber)
    }

    private func overlapFlag(default value: Bool) -> Bool {
        switch values["overlap"] {
        case "on": true
        case "off": false
        default: value
        }
    }

    private func modelRecord(_ folder: ModelFolder, _ t: Transcriber) -> [String: Any] {
        ["folder": folder.root.lastPathComponent, "platform_folders": folder.bundleFolders.map(\.lastPathComponent),
         "reader_config_version": folder.reader.version, "form": t.form.id,
         "window": ["frames": t.contract.frames, "tokens": t.contract.tokens, "left_overlap": t.contract.leftOverlap,
                    "drop_last": t.contract.dropLast],
         "vocabulary": t.vocabulary.count, "cache": t.cacheCheck.record,
         "stages": zip(t.stages, zip(t.loadSeconds, t.warmUpMilliseconds)).map { stage, lw in
             ["name": stage.spec.name, "bundle": stage.url.lastPathComponent, "compute": stage.spec.compute,
              "aot": stage.compiledAhead, "load_s": lw.0, "warm_up_ms": lw.1] as [String: Any]
         }]
    }

    // MARK: - cache

    /// The fingerprint check of `--models` (or the app's folder) without loading anything.
    private func cache(_ record: inout [String: Any], _ log: RunLog) throws -> String {
        let folder = try modelFolder()
        let check = try ModelCache.prepare(folder, mode: try cacheMode)
        record["model"] = ["folder": folder.root.lastPathComponent, "platform_folders": folder.bundleFolders.map(\.lastPathComponent),
                           "cache": check.record]
        log.write(check.summary)
        return check.summary
    }

    // MARK: - thermal

    /// The thermal state every 10 s until nominal or `--cap` seconds (default 600): the cool-down before a device
    /// is handed back.
    private func thermal(_ record: inout [String: Any], _ log: RunLog) async throws -> String {
        let cap = Double(values["cap"] ?? "") ?? 600
        let start = Date()
        var states: [[String: Any]] = []
        while true {
            let state = DeviceProbe.thermal()
            let power = await DeviceProbe.power()
            let elapsed = Date().timeIntervalSince(start)
            states.append(["t_s": elapsed, "state": state, "power": power])
            log.write("thermal \(state) after \(Int(elapsed)) s, power \(power)")
            if state == "nominal" || elapsed >= cap { break }
            try await Task.sleep(for: .seconds(10))
        }
        record["thermal_wait"] = states
        return "thermal=\(DeviceProbe.thermal()) after \(Int(Date().timeIntervalSince(start))) s"
    }

    // MARK: - decode

    private func decode(_ record: inout [String: Any], _ log: RunLog) throws -> String {
        let audio = try required("audio")
        let t0 = DispatchTime.now().uptimeNanoseconds
        let decoded = try AudioDecoder.decode(audio)
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        if let dump = url("dump") {
            try decoded.samples.withUnsafeBytes { try Data($0).write(to: dump) }
        }
        record["decode"] = ["audio": audio.lastPathComponent, "samples": decoded.samples.count, "seconds": decoded.seconds,
                            "source_rate": decoded.sourceSampleRate, "source_channels": decoded.sourceChannels,
                            "converted": decoded.converted, "decode_s": seconds]
        return "decode \(audio.lastPathComponent): \(decoded.samples.count) samples in \(String(format: "%.3f", seconds)) s"
    }

    // MARK: - transcribe (and the time from launch)

    private func transcribe(_ record: inout [String: Any], _ log: RunLog, directory: URL) async throws -> String {
        let audio = try required("audio")
        let launched = DeviceProbe.processStart
        let loadStart = Date()
        let (folder, transcriber) = try await load(log)
        let loaded = Date()
        let overlap = overlapFlag(default: folder.defaultOverlap)
        let result = try await transcriber.transcribe(audio, overlap: overlap)
        let done = Date()
        try (result.text + "\n").write(to: directory.appending(path: "transcript.txt"), atomically: true, encoding: .utf8)
        record["model"] = modelRecord(folder, transcriber)
        record["result"] = resultRecord(result)
        var milestones: [String: Double] = ["load_and_warm_up": loaded.timeIntervalSince(loadStart),
                                            "transcribe": done.timeIntervalSince(loaded)]
        if let launched {
            milestones["process_start_to_load_start"] = loadStart.timeIntervalSince(launched)
            milestones["process_start_to_transcript"] = done.timeIntervalSince(launched)
        }
        record["milestones_s"] = milestones
        log.write("transcript (\(result.run.words.count) words): \(result.text.prefix(200))")
        let fromLaunch = launched.map { String(format: "%.2f s from launch", done.timeIntervalSince($0)) } ?? "launch time unknown"
        return String(format: "transcribe %@: %.1f s audio in %.3f s (x%.0f), %@", audio.lastPathComponent,
                      result.audioSeconds, result.totalSeconds, result.realtimeFactor, fromLaunch)
    }

    private func resultRecord(_ r: Transcriber.Result) -> [String: Any] {
        ["audio_s": r.audioSeconds, "frames": r.frames, "decode_s": r.decodeSeconds, "log_mel_s": r.logMelSeconds,
         "log_mel_cpu_s": Self.logMelRecord(r.logMel), "total_s": r.totalSeconds, "x_realtime": r.realtimeFactor,
         "text": r.text, "run": runRecord(r.run, audio: r.audioSeconds)]
    }

    static func logMelRecord(_ b: LogMelExtractor.Breakdown) -> [String: Any] {
        ["fft": b.fftSeconds, "mel": b.melSeconds, "log": b.logSeconds, "blocks": b.blocks,
         "note": "CPU seconds summed over the parallel blocks of 512 frames: framing + FFT + |X|, mel product, log"]
    }

    private func runRecord(_ run: Transcriber.Run, audio: Double) -> [String: Any] {
        let perWindow = 1000 / Double(max(1, run.windows))
        let host: [String: Double] = ["inputs": run.host.inputs, "feed": run.host.feed, "labels": run.host.labels,
                                      "collapse": run.host.collapse, "total": run.host.total]
        return ["windows": run.windows, "tokens": run.tokens, "words": run.words.count, "overlap": run.overlap,
                "wall_s": run.wallSeconds, "x_realtime_windows": audio / run.wallSeconds,
                // the reader's scale shows in its largest logit: which weights produced the labels
                "top1_logit_max": run.topLogits.filter(\.isFinite).max() ?? Float.nan,
                "blank_logit_max": run.blankLogits.filter(\.isFinite).max() ?? Float.nan,
                "stage_ms_median": run.stageMilliseconds.map { Stats.median($0) },
                "stage_ms_sum": run.stageMilliseconds.map { $0.reduce(0, +) },
                "stage_ms": run.stageMilliseconds.map { $0.map { ($0 * 1000).rounded() / 1000 } },
                "host_s": host, "host_ms_per_window": host.mapValues { $0 * perWindow }]
    }

    // MARK: - gate (fixtures)

    private func gate(_ record: inout [String: Any], _ log: RunLog) async throws -> String {
        let fixtures = try required("fixtures")
        let (folder, transcriber) = try await load(log)
        record["model"] = modelRecord(folder, transcriber)
        let expected = try url("expect").map(ExpectedTranscripts.init) ?? ExpectedTranscripts()
        guard let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appending(path: "MANIFEST.json"))) as? [String: Any],
              let entries = manifest["entries"] as? [[String: Any]] else {
            throw TranscribeError.model("\(fixtures.path)/MANIFEST.json has no entries")
        }
        var rows: [[String: Any]] = []
        var featureMax = 0.0
        var swiftCos: [Double] = [], referenceCos: [Double] = []
        var sameTorch = 0, sameTorchReference = 0, sameLabels = 0, withTorch = 0
        for entry in entries {
            guard let id = entry["id"] as? String, let path = entry["path"] as? String else { continue }
            let dir = fixtures.appending(path: path, directoryHint: .isDirectory)
            let audio = try AudioDecoder.decode(dir.appending(path: "audio.flac"))
            let features = transcriber.extractor.features(audio.samples)
            let frames = features.count / LogMelExtractor.mels
            if let dump = url("dump-features") {
                try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true)
                try features.withUnsafeBytes { try Data($0).write(to: dump.appending(path: "\(id).f32")) }
            }
            let refFeaturesFile = try NPY(dir.appending(path: "features.npy"))
            let refFeatures = try refFeaturesFile.floats()
            let refFrames = refFeaturesFile.shape.first ?? 0
            var maxDiff = 0.0, sumDiff = 0.0, at = 0
            if frames == refFrames {
                for i in 0..<features.count {
                    let d = Double(abs(features[i] - refFeatures[i]))
                    sumDiff += d
                    if d > maxDiff { maxDiff = d; at = i }
                }
            } else {
                maxDiff = .infinity
            }
            featureMax = max(featureMax, maxDiff)
            let softFile = try NPY(dir.appending(path: "soft_tokens.npy"))
            let tokens = softFile.shape.first ?? 0
            let soft = try softFile.floats()
            let swiftRun = try await transcriber.run(features: features, frames: frames, overlap: false, capture: ["soft_tokens"])
            let refRun = try await transcriber.run(features: refFeatures, frames: refFrames, overlap: false, capture: ["soft_tokens"])
            let c1 = try Self.tokenCosines(swiftRun, reference: soft, tokens: tokens)
            let c2 = try Self.tokenCosines(refRun, reference: soft, tokens: tokens)
            swiftCos += c1
            referenceCos += c2
            let torch = expected.torch[id]
            if torch != nil { withTorch += 1 }
            let matchTorch = torch.map { $0 == swiftRun.text }
            let matchTorchRef = torch.map { $0 == refRun.text }
            if matchTorch == true { sameTorch += 1 }
            if matchTorchRef == true { sameTorchReference += 1 }
            let labelsSame = swiftRun.labels == refRun.labels
            if labelsSame { sameLabels += 1 }
            var row: [String: Any] = ["id": id, "samples": audio.samples.count, "frames": frames, "frames_reference": refFrames]
            row["samples_manifest"] = entry["n_samples"] ?? NSNull()
            row["features_max_abs_diff"] = maxDiff
            row["features_mean_abs_diff"] = frames == refFrames ? sumDiff / Double(max(1, features.count)) : Double.nan
            row["features_max_at"] = ["frame": at / LogMelExtractor.mels, "mel": at % LogMelExtractor.mels]
            row["tokens"] = tokens
            row["soft_cos_mean"] = Stats.mean(c1)
            row["soft_cos_min"] = c1.min() ?? Double.nan
            row["soft_cos_mean_from_reference_features"] = Stats.mean(c2)
            row["soft_cos_min_from_reference_features"] = c2.min() ?? Double.nan
            row["transcript"] = swiftRun.text
            row["transcript_from_reference_features"] = refRun.text
            row["labels_same_both_features"] = labelsSame
            row["labels_differing_both_features"] = zip(swiftRun.labels, refRun.labels).filter { $0 != $1 }.count
            row["transcript_torch"] = torch ?? NSNull()
            row["same_as_torch"] = matchTorch ?? NSNull()
            row["same_as_torch_from_reference_features"] = matchTorchRef ?? NSNull()
            row["reference_text"] = expected.reference[id] ?? NSNull()
            row["labels"] = swiftRun.labels
            row["labels_from_reference_features"] = refRun.labels
            row["top1_logit"] = swiftRun.topLogits.map { Double(($0 * 10_000).rounded() / 10_000) }
            row["blank_logit"] = swiftRun.blankLogits.map { Double(($0 * 10_000).rounded() / 10_000) }
            row["stage_ms"] = swiftRun.stageMilliseconds.map { $0.first ?? Double.nan }
            rows.append(row)
            log.write(String(format: "%@: features max|d| %.3g, soft cos mean %.7f min %.6f (reference features: min %.6f), torch %@, %@",
                             id, maxDiff, Stats.mean(c1), c1.min() ?? .nan, c2.min() ?? .nan,
                             matchTorch.map { $0 ? "same" : "DIFFERENT" } ?? "-", swiftRun.text.prefix(60).description))
        }
        let summary: [String: Any] = [
            "utterances": rows.count, "features_max_abs_diff": featureMax,
            "soft_cos_mean": Stats.mean(swiftCos), "soft_cos_min": swiftCos.min() ?? .nan,
            "soft_cos_mean_from_reference_features": Stats.mean(referenceCos),
            "soft_cos_min_from_reference_features": referenceCos.min() ?? .nan,
            "transcripts_same_as_torch": sameTorch, "transcripts_same_as_torch_from_reference_features": sameTorchReference,
            "utterances_with_torch_transcript": withTorch, "labels_same_both_features": sameLabels,
            "bars": ["features_max_abs_diff": 1e-3, "soft_cos_mean": 0.999, "soft_cos_min": 0.99],
        ]
        record["summary"] = summary
        record["utterances"] = rows
        return String(format: "gate %ld utterances: features max|d| %.3g, soft cos mean %.7f min %.6f (from features.npy: min %.6f), transcripts = torch %ld/%ld (from features.npy %ld/%ld), labels equal both features %ld/%ld",
                      rows.count, featureMax, Stats.mean(swiftCos), swiftCos.min() ?? .nan, referenceCos.min() ?? .nan,
                      sameTorch, withTorch, sameTorchReference, withTorch, sameLabels, rows.count)
    }

    /// Per-token cosine (float64) of a one-window run's soft tokens against reference rows [tokens × 512].
    static func tokenCosines(_ run: Transcriber.Run, reference: [Float], tokens: Int) throws -> [Double] {
        guard run.windows == 1, let array = run.captured.first?["soft_tokens"] else {
            throw TranscribeError.contract("the gate needs one window with soft_tokens (got \(run.windows) windows)")
        }
        let soft = try Tensor.floats(array)
        let dim = array.shape.last ?? 0
        guard reference.count == tokens * dim else { throw TranscribeError.contract("soft_tokens.npy is not [\(tokens), \(dim)]") }
        return (0..<tokens).map { t in Stats.cosine(soft, t * dim, reference, t * dim, dim) }
    }

    // MARK: - long recording

    private func long(_ record: inout [String: Any], _ log: RunLog, directory: URL) async throws -> String {
        let audio = try required("audio")
        let (folder, transcriber) = try await load(log)
        record["model"] = modelRecord(folder, transcriber)
        let passes = Int(values["passes"] ?? "") ?? 2
        let modes: [Bool]
        switch values["overlap"] {
        case "on": modes = [true]
        case "off": modes = [false]
        default: modes = [false, true]
        }

        let t0 = DispatchTime.now().uptimeNanoseconds
        let decoded = try AudioDecoder.decode(audio)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let (features, breakdown) = transcriber.extractor.featuresTimed(decoded.samples)
        let t2 = DispatchTime.now().uptimeNanoseconds
        let frames = features.count / LogMelExtractor.mels
        let decodeS = Double(t1 - t0) / 1e9, logMelS = Double(t2 - t1) / 1e9
        let seconds = decoded.seconds
        log.write(String(format: "%@: %.2f s audio, %ld frames; decode %.3f s, log-mel %.3f s (CPU: FFT %.3f, mel %.3f, log %.3f s)",
                         audio.lastPathComponent, seconds, frames, decodeS, logMelS, breakdown.fftSeconds,
                         breakdown.melSeconds, breakdown.logSeconds))
        record["input"] = ["audio": audio.lastPathComponent, "audio_s": seconds, "samples": decoded.samples.count,
                           "frames": frames, "decode_s": decodeS, "log_mel_s": logMelS,
                           "log_mel_cpu_s": Self.logMelRecord(breakdown),
                           "source_rate": decoded.sourceSampleRate, "converted": decoded.converted]

        let referenceLabels = try url("reference-labels").map { try NPY($0).ints() }
        let referenceHyp = try url("reference-hyp").map { try String(contentsOf: $0, encoding: .utf8) }
        let truth = try url("truth").map { try String(contentsOf: $0, encoding: .utf8) }
        let compareTokens = Int((Double(values["compare-minutes"] ?? "") ?? 10) * 60 * 25)

        func compare(_ run: Transcriber.Run) throws -> [String: Any] {
            var out: [String: Any] = [:]
            if let referenceLabels {
                let differing = zip(run.labels, referenceLabels).enumerated().filter { $0.element.0 != $0.element.1 }.map(\.offset)
                out["labels_differing_from_reference"] = differing.count + abs(run.labels.count - referenceLabels.count)
                out["labels_differing_positions"] = Array(differing.prefix(50))
            }
            if let referenceHyp {
                let words = Words.of(referenceHyp)
                out["word_distance_to_reference_hyp"] = WordDistance.distance(Words.of(run.text), words)
                out["transcript_same_as_reference_hyp"] = Words.of(run.text) == words
            }
            if let truth {
                let ref = Words.of(truth)
                let errors = WordDistance.distance(Words.of(run.text), ref)
                out["wer"] = ["errors": errors, "reference_words": ref.count, "wer": Double(errors) / Double(max(1, ref.count))]
            }
            return out
        }

        // The Swift features against Python's (whole file and the first minutes), and the windows from Python's
        // features: the isolation path (same features in, so the labels must equal the Python pipeline's).
        if let featuresURL = url("features") {
            let refFile = try NPY(featuresURL)
            let refFrames = refFile.shape.first ?? 0
            let ref = try refFile.floats()
            var maxAll = 0.0, maxHead = 0.0
            let head = min(frames, compareTokens * 4) * LogMelExtractor.mels
            if refFrames == frames {
                for i in 0..<features.count {
                    let d = Double(abs(features[i] - ref[i]))
                    if d > maxAll { maxAll = d }
                    if i < head, d > maxHead { maxHead = d }
                }
            }
            let isolated = try await transcriber.run(features: ref, frames: refFrames, overlap: modes.last ?? true)
            record["reference_features"] = [
                "file": featuresURL.lastPathComponent, "frames": refFrames, "frames_swift": frames,
                "features_max_abs_diff": refFrames == frames ? maxAll : Double.infinity,
                "features_max_abs_diff_first_minutes": refFrames == frames ? maxHead : Double.infinity,
                "run_from_reference_features": runRecord(isolated, audio: seconds).merging(try compare(isolated)) { $1 },
                "transcript_from_reference_features": isolated.text,
            ]
            log.write(String(format: "features vs %@: max|d| %.3g (first %ld frames %.3g); windows from them: %@",
                             featuresURL.lastPathComponent, maxAll, head / LogMelExtractor.mels, maxHead,
                             (try compare(isolated)).description))
        }

        // `--order on,off,…`: single passes in this order, each started from the thermal state nominal when `--cool S`
        // is given (waiting at most S s): an overlap comparison that heat does not decide.
        if let order = values["order"]?.split(separator: ",").map({ $0 == "on" }), !order.isEmpty {
            let cool = Double(values["cool"] ?? "")
            var runs: [[String: Any]] = []
            var firstLabels: [Int]?
            for (index, overlap) in order.enumerated() {
                var waited = 0.0
                if let cool { waited = try await waitForNominal(cap: cool, log: log) }
                let before = DeviceProbe.thermal()
                let run = try await transcriber.run(features: features, frames: frames, overlap: overlap)
                var r = runRecord(run, audio: seconds)
                r["index"] = index
                r["cooled_s"] = waited
                r["thermal_start"] = before
                r["thermal_end"] = DeviceProbe.thermal()
                r["labels_differing_from_first"] = firstLabels.map { p in zip(p, run.labels).filter { $0 != $1 }.count } ?? 0
                r.merge(try compare(run)) { $1 }
                if firstLabels == nil {
                    firstLabels = run.labels
                    try (run.text + "\n").write(to: directory.appending(path: "transcript.txt"), atomically: true, encoding: .utf8)
                }
                log.write(String(format: "order %ld overlap %@: wall %.3f s (x%.1f), stage medians %@ ms, thermal %@ -> %@, cooled %.0f s",
                                 index, overlap ? "on" : "off", run.wallSeconds, seconds / run.wallSeconds,
                                 run.stageMilliseconds.map { String(format: "%.2f", Stats.median($0)) }.joined(separator: "/"),
                                 before, DeviceProbe.thermal(), waited))
                runs.append(r)
            }
            record["order"] = runs
            let summary = runs.map { r in
                String(format: "%@ %.3f s", (r["overlap"] as? Bool ?? false) ? "on" : "off", r["wall_s"] as? Double ?? .nan)
            }
            return "long \(audio.lastPathComponent) in order: " + summary.joined(separator: ", ")
        }

        var modeRecords: [[String: Any]] = []
        var firstLabels: [Int]?
        var transcript = ""
        for overlap in modes {
            var runs: [[String: Any]] = []
            var passLabels: [Int]?
            for pass in 1...max(1, passes) {
                let capture: Set<String> = (url("soft-reference") != nil && pass == 1 && overlap == modes.first) ? ["soft_tokens"] : []
                let run = try await transcriber.run(features: features, frames: frames, overlap: overlap, capture: capture)
                var r = runRecord(run, audio: seconds)
                r["pass"] = pass
                r["x_realtime_end_to_end"] = seconds / (decodeS + logMelS + run.wallSeconds)
                r["labels_differing_from_pass1"] = passLabels.map { p in zip(p, run.labels).filter { $0 != $1 }.count } ?? 0
                r["labels_differing_from_first_mode"] = firstLabels.map { p in zip(p, run.labels).filter { $0 != $1 }.count } ?? 0
                r.merge(try compare(run)) { $1 }
                if !capture.isEmpty, let softURL = url("soft-reference") {
                    let plan = Windowing.plan(frames: frames, contract: transcriber.contract)
                    r["soft_tokens_vs_reference"] = try softCompare(run, plan: plan, reference: softURL, tokens: compareTokens)
                }
                if passLabels == nil { passLabels = run.labels }
                if firstLabels == nil { firstLabels = run.labels; transcript = run.text }
                log.write(String(format: "overlap %@ pass %ld: %ld windows, wall %.3f s (x%.1f windows, x%.1f end to end), stage medians %@ ms, host %.2f ms/window, %@",
                                 overlap ? "on" : "off", pass, run.windows, run.wallSeconds, seconds / run.wallSeconds,
                                 seconds / (decodeS + logMelS + run.wallSeconds),
                                 run.stageMilliseconds.map { String(format: "%.2f", Stats.median($0)) }.joined(separator: "/"),
                                 run.host.total * 1000 / Double(max(1, run.windows)),
                                 (try compare(run)).filter { $0.key != "labels_differing_positions" }.description))
                runs.append(r)
            }
            modeRecords.append(["overlap": overlap, "passes": runs])
        }
        record["modes"] = modeRecords
        try (transcript + "\n").write(to: directory.appending(path: "transcript.txt"), atomically: true, encoding: .utf8)
        let summary = modeRecords.map { m -> String in
            let p = (m["passes"] as? [[String: Any]])?.first ?? [:]
            return String(format: "overlap %@ %.3f s (x%.0f)", (m["overlap"] as? Bool ?? false) ? "on" : "off",
                          p["wall_s"] as? Double ?? .nan, p["x_realtime_windows"] as? Double ?? .nan)
        }
        return "long \(audio.lastPathComponent): " + summary.joined(separator: ", ")
    }

    /// Waits (polling every 5 s) until the thermal state is nominal, at most `cap` seconds; returns the seconds waited.
    private func waitForNominal(cap: Double, log: RunLog) async throws -> Double {
        let start = Date()
        while DeviceProbe.thermal() != "nominal", Date().timeIntervalSince(start) < cap {
            try await Task.sleep(for: .seconds(5))
        }
        let waited = Date().timeIntervalSince(start)
        if waited > 1 { log.write(String(format: "cooled %.0f s: thermal %@", waited, DeviceProbe.thermal())) }
        return waited
    }

    /// Kept tokens of the windows within the first `tokens` tokens vs a continuous reference [n × 512] (.npy).
    private func softCompare(_ run: Transcriber.Run, plan: [Window], reference: URL, tokens: Int) throws -> [String: Any] {
        let ref = try NPY(reference)
        let dim = ref.shape.last ?? 512
        var cos: [Double] = []
        for window in plan where window.token0 + window.keep.lowerBound < tokens {
            guard let array = run.captured[window.index]["soft_tokens"] else { continue }
            let soft = try Tensor.floats(array)
            for p in window.keep where window.token0 + p < tokens {
                let row = try ref.floats(from: (window.token0 + p) * dim, count: dim)
                cos.append(Stats.cosine(soft, p * dim, row, 0, dim))
            }
        }
        return ["tokens": cos.count, "cos_mean": Stats.mean(cos), "cos_min": cos.min() ?? .nan,
                "reference": reference.lastPathComponent]
    }
}

// MARK: - Helpers

/// `<dir>/log.txt`, one line per event, flushed as it happens (the device poller reads it while the run goes).
final class RunLog: @unchecked Sendable {
    private let lock = NSLock()
    private let handle: FileHandle
    private let start = Date()
    private let sink: @Sendable (String) -> Void

    init(directory: URL, sink: @escaping @Sendable (String) -> Void) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "log.txt")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        self.sink = sink
    }

    func write(_ message: String) {
        let line = String(format: "t=%8.3f | ", Date().timeIntervalSince(start)) + message + "\n"
        lock.lock()
        defer { lock.unlock() }
        handle.write(Data(line.utf8))
        try? handle.synchronize()
        sink(line)
    }
}

enum JSONOut {
    /// NaN and ±inf become null (JSON has no such numbers).
    static func clean(_ value: Any) -> Any {
        switch value {
        case let d as Double: return d.isFinite ? d : NSNull()
        case let f as Float: return f.isFinite ? Double(f) : NSNull()
        case let a as [Double]: return a.map { $0.isFinite ? $0 as Any : NSNull() }
        case let a as [Any]: return a.map(clean)
        case let o as [String: Any]: return o.mapValues(clean)
        default: return value
        }
    }

    static func write(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: clean(object), options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }
}

enum Stats {
    static func median(_ values: [Double]) -> Double {
        let v = values.filter(\.isFinite).sorted()
        guard !v.isEmpty else { return .nan }
        return v.count % 2 == 1 ? v[v.count / 2] : (v[v.count / 2 - 1] + v[v.count / 2]) / 2
    }

    static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? .nan : values.reduce(0, +) / Double(values.count)
    }

    static func cosine(_ a: [Float], _ ia: Int, _ b: [Float], _ ib: Int, _ n: Int) -> Double {
        var ab = 0.0, aa = 0.0, bb = 0.0
        for k in 0..<n {
            let x = Double(a[ia + k]), y = Double(b[ib + k])
            ab += x * y
            aa += x * x
            bb += y * y
        }
        return ab / (aa.squareRoot() * bb.squareRoot() + 1e-30)
    }
}

enum Words {
    /// Lower case, split on whitespace (the WER normalization of the reader's evaluation).
    static func of(_ text: String) -> [String] {
        text.lowercased().split(whereSeparator: \.isWhitespace).map(String.init)
    }
}

enum WordDistance {
    /// Word-level Levenshtein distance (substitutions + deletions + insertions).
    static func distance(_ hyp: [String], _ ref: [String]) -> Int {
        var ids: [String: Int32] = [:]
        let h = hyp.map { w in ids[w] ?? { let id = Int32(ids.count); ids[w] = id; return id }() }
        let r = ref.map { w in ids[w] ?? { let id = Int32(ids.count); ids[w] = id; return id }() }
        if h.isEmpty || r.isEmpty { return max(h.count, r.count) }
        var previous = [Int32](0...Int32(h.count))
        var current = [Int32](repeating: 0, count: h.count + 1)
        for i in 1...r.count {
            current[0] = Int32(i)
            let ri = r[i - 1]
            for j in 1...h.count {
                let substitution = previous[j - 1] + (h[j - 1] == ri ? 0 : 1)
                current[j] = min(substitution, previous[j] + 1, current[j - 1] + 1)
            }
            swap(&previous, &current)
        }
        return Int(previous[h.count])
    }
}

/// hyp_*.tsv of the reader's gate: set, id, reference, torch, coreai (the torch column = the reference path). Only
/// the rows of the `fixture` set are read when the file has a set column (the same id can also be a test-clean row).
struct ExpectedTranscripts {
    var torch: [String: String] = [:]
    var reference: [String: String] = [:]

    init() {}

    init(_ url: URL) throws {
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map(String.init)
        guard let header = lines.first?.split(separator: "\t", omittingEmptySubsequences: false).map(String.init),
              let idColumn = header.firstIndex(of: "id"), let torchColumn = header.firstIndex(of: "torch") else {
            throw TranscribeError.model("\(url.lastPathComponent): no id / torch columns")
        }
        let referenceColumn = header.firstIndex(of: "reference")
        let setColumn = header.firstIndex(of: "set")
        for line in lines.dropFirst() {
            let cells = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard cells.count > max(idColumn, torchColumn) else { continue }
            if let setColumn, cells.count > setColumn, cells[setColumn] != "fixture" { continue }
            torch[cells[idColumn]] = cells[torchColumn]
            if let referenceColumn, cells.count > referenceColumn { reference[cells[idColumn]] = cells[referenceColumn] }
        }
    }
}
