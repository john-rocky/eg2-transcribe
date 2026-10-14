import SwiftUI

/// The numbers of the last transcription: audio length, wall clock and times real time in one line at the
/// transcript's size, then where the time went in small print.
struct StatsView: View {
    let stats: RunStats
    /// The transcript's text size (0 = the platform's body size); the headline follows it.
    let size: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(headline)
                .font(size > 0 ? .system(size: size * 0.8, weight: .semibold) : .title3.weight(.semibold))
                .monospacedDigit()
            Text(line)
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    /// "0:10 audio · 152 ms · ×66": the end-to-end time from the file to the words.
    private var headline: String {
        "\(Self.clock(stats.audioSeconds)) audio · \(Self.duration(stats.totalSeconds)) · "
            + String(format: "×%.0f", stats.realtimeFactor)
    }

    private var line: String {
        var parts = [
            "decode " + Self.duration(stats.decodeSeconds),
            "log-mel " + Self.duration(stats.logMelSeconds),
        ]
        parts += zip(stats.stageSeconds, stats.stageSums).map { "\($0) " + Self.duration($1) }
        parts.append("host " + Self.duration(stats.hostSeconds))
        parts.append("\(stats.windows) window\(stats.windows == 1 ? "" : "s")")
        parts.append("\(stats.form), overlap \(stats.overlap ? "on" : "off")")
        return parts.joined(separator: " · ")
    }

    /// Milliseconds below a second, seconds above.
    static func duration(_ seconds: Double) -> String {
        seconds < 1 ? String(format: "%.0f ms", seconds * 1000) : String(format: "%.2f s", seconds)
    }

    static func clock(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
                         : String(format: "%d:%02d", s / 60, s % 60)
    }
}
