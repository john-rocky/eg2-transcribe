import Accelerate
import Foundation

/// The log-mel features of the EmbeddingGemma 2 audio encoder: `Gemma4AudioFeatureExtractor` of transformers
/// (`models/gemma4/feature_extraction_gemma4.py`, defaults of google/embeddinggemma-2's preprocessor_config.json),
/// step for step, for one 16 kHz mono waveform of any length (no 30 s truncation):
///
///   1. 160 zeros in front of the waveform (semicausal: frame i is centred on sample 160 i).
///   2. Frame i = padded samples [160 i, 160 i + 320) (the extractor unfolds 321 and keeps the first 320; no
///      preemphasis), times the periodic Hann window of length 320 (`window_function(320)`, float64 → float32),
///      the product in float32.
///   3. `rfft` of the frame zero-padded to 512; the magnitude |X| of the 257 bins (not the power), held as float32.
///      The extractor's numpy FFT runs in float32 with errors near float32 rounding; vDSP's float32 FFT is coarser
///      (up to 4e-4 apart in the log-mel on quiet bins), so the FFT here runs in float64 and |X| is rounded to float32:
///      within 5e-7 of the extractor's log-mel.
///   4. Times the mel filter bank [257, 128] (`mel_filter_bank`: HTK scale, 0–8 kHz, no norm; read from
///      mel_filters.bin, float64 as the extractor builds it) in float64, `log(mel + 0.001)` in float64, then float32.
///   5. Only the valid frames: frame i is valid when its whole analysis window is real audio, so
///      n_frames = floor((n − 161) / 160) + 1 (the extractor's `input_features_mask`).
///
/// No dither, no per-bin normalization (both off in the checkpoint's config).
struct LogMelExtractor: Sendable {
    static let frameLength = 320
    static let hopLength = 160
    static let fftLength = 512
    static let bins = 257
    static let mels = 128
    static let melFloor = 1e-3
    static let filtersFile = "mel_filters.bin"

    /// float32 periodic Hann window, 320 values.
    let window: [Float]
    /// The mel filter bank, [257 bins × 128 mels] row-major, float64.
    let filters: [Double]

    /// `filters`: mel_filters.bin = 257 × 128 float64, little-endian, row-major (bin-major), the matrix
    /// `mel_filter_bank(257, 128, 0, 8000, 16000, norm=None, mel_scale="htk")` (tools/make_mel_filters.py).
    init(filtersURL: URL) throws {
        let data = try Data(contentsOf: filtersURL)
        let count = Self.bins * Self.mels
        guard data.count == count * MemoryLayout<Double>.size else {
            throw TranscribeError.model("\(filtersURL.lastPathComponent): \(data.count) bytes, expected \(count * 8) (257 × 128 float64)")
        }
        filters = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [Double] in
            (0..<count).map { Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 8, as: UInt64.self))) }
        }
        window = Self.periodicHann(Self.frameLength)
    }

    /// `numpy.hanning(n + 1)[:-1]` as float32: 0.5 + 0.5 cos(π m / n) for m = −n, −n + 2, … (numpy's own formula).
    static func periodicHann(_ n: Int) -> [Float] {
        let length = n + 1
        return (0..<n).map { k in
            let m = Double(1 - length + 2 * k)
            return Float(0.5 + 0.5 * cos(Double.pi * m / Double(length - 1)))
        }
    }

    /// Valid frames of a waveform of `samples` samples: floor((n − 161) / 160) + 1, or 0. Frame i unfolds 321 padded
    /// samples, the last of which is sample 160 i + 160 of the waveform; it has to be real audio.
    static func frameCount(samples: Int) -> Int {
        let pad = frameLength / 2
        return samples < pad + 1 ? 0 : (samples - pad - 1) / hopLength + 1
    }

    /// The log-mel features of the valid frames, [frames × 128] row-major float32.
    func features(_ samples: [Float]) -> [Float] { featuresTimed(samples).features }

    /// Where the extractor's time goes, summed over the parallel blocks (CPU seconds): framing + FFT + |X|, the mel
    /// product, the log (with the float32 rounding of the output).
    struct Breakdown: Sendable {
        var fftSeconds = 0.0
        var melSeconds = 0.0
        var logSeconds = 0.0
        var blocks = 0
    }

    func featuresTimed(_ samples: [Float]) -> (features: [Float], breakdown: Breakdown) {
        let frames = Self.frameCount(samples: samples.count)
        guard frames > 0 else { return ([], Breakdown()) }
        var out = [Float](repeating: 0, count: frames * Self.mels)
        guard let setup = vDSP_create_fftsetupD(vDSP_Length(9), FFTRadix(kFFTRadix2)) else { return ([], Breakdown()) }
        defer { vDSP_destroy_fftsetupD(setup) }
        let block = 512
        let blocks = (frames + block - 1) / block
        let clock = BlockClock()
        samples.withUnsafeBufferPointer { wave in
            out.withUnsafeMutableBufferPointer { dst in
                // Blocks of frames are independent: one per core, each with its own buffers. Every block reads the
                // waveform and the FFT setup (read-only) and writes only its own rows of the output.
                let shared = SharedBuffers(wave: wave, out: dst.baseAddress!, setup: setup, clock: clock)
                DispatchQueue.concurrentPerform(iterations: blocks) { b in
                    let f0 = b * block
                    computeBlock(shared, frames: f0..<min(frames, f0 + block))
                }
            }
        }
        var breakdown = clock.total
        breakdown.blocks = blocks
        return (out, breakdown)
    }

    /// The buffers the concurrent blocks share (see `featuresTimed`).
    private struct SharedBuffers: @unchecked Sendable {
        let wave: UnsafeBufferPointer<Float>
        let out: UnsafeMutablePointer<Float>
        let setup: FFTSetupD
        let clock: BlockClock
    }

    /// Adds up the phases' seconds of every block.
    private final class BlockClock: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var total = Breakdown()

        func add(fft: Double, mel: Double, log: Double) {
            lock.lock()
            total.fftSeconds += fft
            total.melSeconds += mel
            total.logSeconds += log
            lock.unlock()
        }
    }

    private func computeBlock(_ shared: SharedBuffers, frames: Range<Int>) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let wave = shared.wave
        let n = frames.count
        let half = Self.fftLength / 2
        var frame = [Double](repeating: 0, count: Self.fftLength)
        var real = [Double](repeating: 0, count: half)
        var imag = [Double](repeating: 0, count: half)
        var magnitude = [Double](repeating: 0, count: n * Self.bins)
        var mel = [Double](repeating: 0, count: n * Self.mels)
        let pad = Self.frameLength / 2

        for (row, i) in frames.enumerated() {
            // frame = waveform[160 i − 160 ..< 160 i + 160] (zeros before the start) × window in float32, zero-padded
            let start = i * Self.hopLength - pad
            for j in 0..<Self.frameLength {
                let s = start + j
                frame[j] = s >= 0 ? Double(wave[s] * window[j]) : 0
            }
            real.withUnsafeMutableBufferPointer { re in
                imag.withUnsafeMutableBufferPointer { im in
                    var split = DSPDoubleSplitComplex(realp: re.baseAddress!, imagp: im.baseAddress!)
                    frame.withUnsafeBufferPointer { f in
                        f.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self, capacity: half) {
                            vDSP_ctozD($0, 2, &split, 1, vDSP_Length(half))
                        }
                    }
                    // In-place real FFT; vDSP returns 2 × the DFT, packed: realp[0] = X[0], imagp[0] = X[256].
                    vDSP_fft_zripD(shared.setup, &split, 1, vDSP_Length(9), FFTDirection(kFFTDirection_Forward))
                }
            }
            // |X| as float32, as the extractor holds it
            let base = row * Self.bins
            magnitude[base] = Double(Float(abs(real[0] * 0.5)))
            magnitude[base + half] = Double(Float(abs(imag[0] * 0.5)))
            for k in 1..<half {
                magnitude[base + k] = Double(Float(hypot(real[k] * 0.5, imag[k] * 0.5)))
            }
        }
        let t1 = DispatchTime.now().uptimeNanoseconds
        // mel = |X| [n × 257] · filters [257 × 128] in float64, then log(mel + 0.001)
        magnitude.withUnsafeBufferPointer { a in
            filters.withUnsafeBufferPointer { b in
                mel.withUnsafeMutableBufferPointer { c in
                    vDSP_mmulD(a.baseAddress!, 1, b.baseAddress!, 1, c.baseAddress!, 1,
                               vDSP_Length(n), vDSP_Length(Self.mels), vDSP_Length(Self.bins))
                }
            }
        }
        let t2 = DispatchTime.now().uptimeNanoseconds
        var floor = Self.melFloor
        var count = Int32(mel.count)
        mel.withUnsafeMutableBufferPointer { m in
            vDSP_vsaddD(m.baseAddress!, 1, &floor, m.baseAddress!, 1, vDSP_Length(m.count))
            vvlog(m.baseAddress!, m.baseAddress!, &count)
        }
        let offset = frames.lowerBound * Self.mels
        for k in 0..<(n * Self.mels) { shared.out[offset + k] = Float(mel[k]) }
        let t3 = DispatchTime.now().uptimeNanoseconds
        shared.clock.add(fft: Double(t1 - t0) / 1e9, mel: Double(t2 - t1) / 1e9, log: Double(t3 - t2) / 1e9)
    }
}
