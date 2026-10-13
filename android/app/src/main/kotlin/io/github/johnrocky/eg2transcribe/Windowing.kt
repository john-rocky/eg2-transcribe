package io.github.johnrocky.eg2transcribe

/**
 * How a recording of [frames] mel frames is cut into fixed encoder windows of [windowFrames] frames (1000 = 10 s,
 * 3000 = 30 s; 4 frames = 1 token = 40 ms).
 *
 * Token j reads frames 4j - 3 .. 4j + 3 (two stride-2 convolutions of width 3), and the encoder keeps no state
 * across calls. So:
 * - Left: a window starts without the context the continuous encoder would have, and its first tokens come out wrong
 *   (about the first 96, measured on a Mac). Window k therefore starts [overlapTokens] tokens (4 x overlapTokens
 *   frames) before the end of window k - 1 and keeps only its tokens after that overlap; window 0 keeps all of them.
 * - Right: a window's last token needs its 3 frames ahead. The log-mel is computed on the continuous recording and a
 *   window is a slice of those frames, so every window but the last has them (it is never cut out of a shorter piece
 *   of audio, which would lose them). At the end of the recording a token whose frames run past the last frame is
 *   dropped: the recording keeps floor(frames / 4) tokens.
 *
 * Window k covers frames [k * stride, k * stride + windowFrames) with stride = windowFrames - 4 * overlapTokens, the
 * last window is padded with masked frames, and the kept tokens of all windows are global tokens 0 until [tokens],
 * each exactly once.
 */
class WindowPlan(val frames: Int, val windowFrames: Int, val overlapTokens: Int) {
    init {
        require(frames >= 0) { "frames $frames < 0" }
        require(windowFrames > 0 && windowFrames % 4 == 0) { "window of $windowFrames frames: expected a positive multiple of 4" }
        require(overlapTokens in 0..maxOverlap(windowFrames)) { "overlap of $overlapTokens tokens: expected 0..${maxOverlap(windowFrames)} for a window of $windowFrames frames" }
    }

    val stride: Int = windowFrames - 4 * overlapTokens

    /** Frames up to the end of the last whole token. */
    val wholeFrames: Int = frames / 4 * 4

    /** Tokens of the whole recording: the whole ones. */
    val tokens: Int = wholeFrames / 4

    /** Windows: 1 when the whole tokens fit in one, else enough strides to reach the last of them. */
    val count: Int = if (wholeFrames <= windowFrames) 1 else (wholeFrames - windowFrames + stride - 1) / stride + 1

    /** Encoder work relative to one pass without overlap. */
    val computeFactor: Double get() = windowFrames.toDouble() / stride

    fun start(k: Int): Int = k * stride

    /** Real (unmasked) frames of window k: all of them up to the end of the recording. */
    fun validFrames(k: Int): Int = minOf(windowFrames, frames - start(k))

    /** Tokens the encoder marks valid in window k (ceil of its real frames / 4). */
    fun validTokens(k: Int): Int = tokensFor(validFrames(k))

    /** Index, within window k, of its first kept token. */
    fun keepFrom(k: Int): Int = if (k == 0) 0 else overlapTokens

    /** End (exclusive), within window k, of its kept tokens: the whole ones. */
    fun keepUntil(k: Int): Int = minOf(windowFrames, wholeFrames - start(k)) / 4

    /** Global index of window k's token 0. */
    fun firstToken(k: Int): Int = start(k) / 4

    /** Global tokens window k contributes. */
    fun kept(k: Int): IntRange = (firstToken(k) + keepFrom(k)) until (firstToken(k) + keepUntil(k))

    companion object {
        const val FRAMES_10S = 1000
        const val FRAMES_30S = 3000
        const val DEFAULT_OVERLAP = 96

        /** Tokens the encoder marks valid for [frames] real frames: ceil(ceil(frames / 2) / 2). */
        fun tokensFor(frames: Int): Int = (frames + 3) / 4

        /** At most half a window of overlap (twice the encoder work). */
        fun maxOverlap(windowFrames: Int): Int = windowFrames / 8
    }
}
