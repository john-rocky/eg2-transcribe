import AVFoundation
import Foundation

/// Records the microphone to a 16 kHz mono 16-bit WAV file in the temporary folder; the file then goes through the
/// same path as a chosen file.
@MainActor
final class AudioRecorder {
    private var recorder: AVAudioRecorder?
    private(set) var url: URL?

    var isRecording: Bool { recorder?.isRecording ?? false }

    /// Asks for the microphone once (the system shows its dialog the first time).
    func permission() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    func start() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)
        #endif
        let url = FileManager.default.temporaryDirectory.appending(path: "recording-\(Int(Date().timeIntervalSince1970)).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: AudioDecoder.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let recorder = try AVAudioRecorder(url: url, settings: settings)
        guard recorder.record() else { throw TranscribeError.audio("the microphone did not start") }
        self.recorder = recorder
        self.url = url
    }

    /// Stops and returns the recorded file.
    func stop() -> URL? {
        recorder?.stop()
        recorder = nil
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
        return url
    }
}
