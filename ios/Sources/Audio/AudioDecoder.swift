import AVFoundation
import Foundation

/// Decodes an audio file into what the feature extractor takes: 16 kHz, mono, Float32 samples in [-1, 1).
///
/// Any format Core Audio reads works (WAV, FLAC, M4A / AAC, MP3, CAF, AIFF). A 16 kHz mono file is read as is,
/// so its samples are the decoder's own (a 16-bit FLAC gives sample / 32768, as Python's soundfile does);
/// anything else goes through `AVAudioConverter` (sample-rate conversion at the highest quality, channels
/// downmixed). The file is read in chunks, so a long recording never sits in memory at its source rate.
enum AudioDecoder {
    static let sampleRate = 16_000.0

    struct Decoded: Sendable {
        let samples: [Float]
        let sourceSampleRate: Double
        let sourceChannels: Int
        let converted: Bool

        var seconds: Double { Double(samples.count) / AudioDecoder.sampleRate }
    }

    static func decode(_ url: URL) throws -> Decoded {
        let file = try AVAudioFile(forReading: url)
        let source = file.processingFormat
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1,
                                         interleaved: false) else {
            throw TranscribeError.audio("cannot describe 16 kHz mono Float32")
        }
        let chunk: AVAudioFrameCount = 1 << 16
        var samples: [Float] = []
        samples.reserveCapacity(Int(Double(file.length) * sampleRate / source.sampleRate) + Int(chunk))

        if source.sampleRate == sampleRate, source.channelCount == 1, source.commonFormat == .pcmFormatFloat32 {
            guard let buffer = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: chunk) else {
                throw TranscribeError.audio("cannot allocate a read buffer")
            }
            while file.framePosition < file.length {
                try file.read(into: buffer, frameCount: chunk)
                if buffer.frameLength == 0 { break }
                guard let channel = buffer.floatChannelData?[0] else { throw TranscribeError.audio("no sample data") }
                samples.append(contentsOf: UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
            }
            return Decoded(samples: samples, sourceSampleRate: source.sampleRate,
                           sourceChannels: Int(source.channelCount), converted: false)
        }

        guard let converter = AVAudioConverter(from: source, to: target) else {
            throw TranscribeError.audio("no converter from \(source) to 16 kHz mono")
        }
        converter.downmix = true
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        let outputCapacity = AVAudioFrameCount((Double(chunk) * sampleRate / source.sampleRate).rounded(.up)) + 1024
        guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: chunk),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outputCapacity) else {
            throw TranscribeError.audio("cannot allocate conversion buffers")
        }
        let reader = ChunkReader(file: file, buffer: input, chunk: chunk)
        while true {
            output.frameLength = 0
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
                reader.next(inputStatus)
            }
            if let error = reader.error { throw error }
            if status == .error {
                throw conversionError ?? TranscribeError.audio("conversion failed")
            }
            if output.frameLength > 0, let channel = output.floatChannelData?[0] {
                samples.append(contentsOf: UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
            }
            if status == .endOfStream || (status == .inputRanDry && reader.ended) { break }
        }
        return Decoded(samples: samples, sourceSampleRate: source.sampleRate,
                       sourceChannels: Int(source.channelCount), converted: true)
    }

    /// Hands the converter one chunk of the file per request, then end-of-stream.
    private final class ChunkReader: @unchecked Sendable {
        let file: AVAudioFile
        let buffer: AVAudioPCMBuffer
        let chunk: AVAudioFrameCount
        var ended = false
        var error: Error?

        init(file: AVAudioFile, buffer: AVAudioPCMBuffer, chunk: AVAudioFrameCount) {
            self.file = file
            self.buffer = buffer
            self.chunk = chunk
        }

        func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
            if ended || file.framePosition >= file.length {
                ended = true
                status.pointee = .endOfStream
                return nil
            }
            do {
                try file.read(into: buffer, frameCount: chunk)
            } catch {
                self.error = error
                ended = true
                status.pointee = .endOfStream
                return nil
            }
            if buffer.frameLength == 0 {
                ended = true
                status.pointee = .endOfStream
                return nil
            }
            status.pointee = .haveData
            return buffer
        }
    }
}
