import Foundation

/// How a recording is cut into encoder windows (reader_config.json `window`).
///
/// The encoder is causal with a few seconds of reach: the first tokens of a window lack their left context, and
/// the last token of a window lacks one input frame. So each window starts `leftOverlap` tokens before the end of
/// the previous one's kept part and drops its last `dropLast` tokens; only the new tokens are kept.
struct WindowContract: Sendable, Equatable, Codable {
    /// Log-mel frames per window (100 per second: 3000 = 30 s); the encoder's input length.
    let frames: Int
    /// Tokens of left context repeated from the previous window.
    let leftOverlap: Int
    /// Tokens dropped at the end of every window but the last.
    let dropLast: Int

    /// Soft tokens per window (25 per second: 750 for 30 s).
    var tokens: Int { frames / 4 }
    var stepTokens: Int { tokens - leftOverlap - dropLast }
    var seconds: Double { Double(frames) / 100 }
}

struct Window: Sendable, Equatable {
    let index: Int
    /// The window's first token in the recording (frame0 = 4 token0).
    let token0: Int
    let frame0: Int
    /// Valid frames in the window (the last window is shorter; the rest is zero padding and half_mask 0).
    let frames: Int
    /// Valid tokens in the window: ceil(ceil(frames / 2) / 2).
    let validTokens: Int
    /// The tokens of this window that go into the transcript, as window positions.
    let keep: Range<Int>
}

enum Windowing {
    /// Soft tokens for n valid log-mel frames: ceil(ceil(n / 2) / 2) (two stride-2 convolutions).
    static func tokens(frames n: Int) -> Int { ((n + 1) / 2 + 1) / 2 }

    /// Windows of `contract.frames` frames stepping by (tokens − leftOverlap − dropLast) tokens. Window i keeps its
    /// tokens [leftOverlap, tokens − dropLast) (the first window from 0); the window that reaches the end keeps
    /// through its last valid token. Together the kept ranges cover the recording's tokens once, in order.
    static func plan(frames nFrames: Int, contract: WindowContract) -> [Window] {
        let total = tokens(frames: nFrames)
        guard total > 0 else { return [] }
        let T = contract.tokens
        let step = contract.stepTokens
        precondition(step > 0, "window contract: no new tokens per window")
        var plan: [Window] = []
        var i = 0
        while true {
            let a = i * step
            let nf = min(contract.frames, nFrames - 4 * a)
            let m = tokens(frames: nf)
            let final = a + m >= total
            let keep = (i == 0 ? 0 : contract.leftOverlap)..<(final ? m : T - contract.dropLast)
            plan.append(Window(index: i, token0: a, frame0: 4 * a, frames: nf, validTokens: m, keep: keep))
            if final { break }
            i += 1
        }
        return plan
    }

    /// Tokens covered by a plan (the recording's token count).
    static func totalTokens(_ plan: [Window]) -> Int {
        plan.last.map { $0.token0 + $0.keep.upperBound } ?? 0
    }
}
