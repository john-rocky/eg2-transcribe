package io.github.johnrocky.eg2transcribe

/**
 * How a recording of [frames] mel frames is cut into fixed encoder windows of [windowFrames] frames (1000 = 10 s,
 * 3000 = 30 s; 4 frames = 1 token = 40 ms), and which tokens of each window make the transcript. This is the window
 * contract of the reader (`reader/reader_config.json` `window`: 30 s, 96 tokens of left overlap, 1 token dropped at the
 * right; the same plan as the Core AI port's):
 *
 * Token j reads frames 4j - 3 .. 4j + 3 (two stride-2 convolutions of width 3), and the encoder keeps no state across
 * calls. The log-mel is computed on the continuous recording and a window is a slice of those frames.
 * - Window k starts at token k x [step], step = T - [overlapTokens] - [dropLast] (T = windowFrames / 4 tokens; 653 for
 *   30 s windows).
 * - Left: a window starts without the context the continuous encoder would have, so its first tokens come out wrong
 *   (about the first 96, measured on a Mac). Window k > 0 keeps its tokens from [overlapTokens] on; window 0 from 0.
 * - Right: every window but the last stops [dropLast] tokens before its end (the next window keeps that token).
 * - The window that reaches the end of the recording keeps through its last valid token: the recording keeps
 *   ceil(frames / 4) tokens, the encoder's valid count (the last one is computed with the frames past the end masked,
 *   as the reader's training data and its released fixture outputs were).
 *
 * The reader reads each window's tokens as one block (both ways in time) and the kept ranges of all windows make one
 * stream of global tokens 0 until [tokens], each exactly once.
 */
class WindowPlan(val frames: Int, val windowFrames: Int, val overlapTokens: Int, val dropLast: Int = DEFAULT_DROP_LAST) {
    init {
        require(frames >= 0) { "frames $frames < 0" }
        require(windowFrames > 0 && windowFrames % 4 == 0) { "window of $windowFrames frames: expected a positive multiple of 4" }
        require(overlapTokens in 0..maxOverlap(windowFrames)) { "overlap of $overlapTokens tokens: expected 0..${maxOverlap(windowFrames)} for a window of $windowFrames frames" }
        require(dropLast in 0..1) { "dropLast $dropLast: expected 0 or 1" }
    }

    /** Tokens of one window. */
    val windowTokens: Int = windowFrames / 4

    /** Tokens from one window's start to the next one's. */
    val step: Int = windowTokens - overlapTokens - dropLast

    /** Frames from one window's start to the next one's. */
    val stride: Int = 4 * step

    /** Tokens of the whole recording: ceil(frames / 4). */
    val tokens: Int = tokensFor(frames)

    /** Windows: 1 when the recording fits in one, else enough steps for a window to reach the last token. */
    val count: Int = if (tokens <= windowTokens) 1 else (tokens - windowTokens + step - 1) / step + 1

    /** Encoder work relative to one pass without overlap. */
    val computeFactor: Double get() = windowTokens.toDouble() / step

    fun start(k: Int): Int = k * stride

    /** Real (unmasked) frames of window k: all of them up to the end of the recording. */
    fun validFrames(k: Int): Int = minOf(windowFrames, frames - start(k))

    /** Tokens the encoder marks valid in window k (ceil of its real frames / 4). */
    fun validTokens(k: Int): Int = tokensFor(validFrames(k))

    /** Index, within window k, of its first kept token. */
    fun keepFrom(k: Int): Int = if (k == 0) 0 else overlapTokens

    /** End (exclusive), within window k, of its kept tokens. */
    fun keepUntil(k: Int): Int = if (k == count - 1) validTokens(k) else windowTokens - dropLast

    /** Global index of window k's token 0. */
    fun firstToken(k: Int): Int = k * step

    /** Global tokens window k contributes. */
    fun kept(k: Int): IntRange = (firstToken(k) + keepFrom(k)) until (firstToken(k) + keepUntil(k))

    companion object {
        const val FRAMES_10S = 1000
        const val FRAMES_30S = 3000
        const val DEFAULT_OVERLAP = 96
        const val DEFAULT_DROP_LAST = 1

        /** The window setting "auto" (not a window length): each recording gets its length from [autoFrames]. */
        const val AUTO = 0

        /**
         * The longest recording, in tokens, that "auto" reads with the 10 s files: one 10 s window with its right-edge
         * token to spare (250 - 1 = 249 tokens = 9.96 s; the contract's token at the right of a window).
         */
        const val AUTO_MAX_TOKENS_10S = FRAMES_10S / 4 - DEFAULT_DROP_LAST

        /** Tokens the encoder marks valid for [frames] real frames: ceil(ceil(frames / 2) / 2). */
        fun tokensFor(frames: Int): Int = (frames + 3) / 4

        /** At most half a window of overlap (twice the encoder work). */
        fun maxOverlap(windowFrames: Int): Int = windowFrames / 8

        /**
         * The window length "auto" picks for a recording of [tokens] tokens: 10 s when the recording fits one 10 s window
         * with its right-edge token to spare ([AUTO_MAX_TOKENS_10S]) and the 10 s files are there ([has10s]: the
         * encoder's 10 s file and the reader's), else 30 s when the encoder has that file ([has30s]), else 10 s. A
         * recording that needs more than one window always gets 30 s windows (fewer window edges).
         */
        fun autoFrames(tokens: Int, has10s: Boolean, has30s: Boolean): Int = when {
            has10s && tokens <= AUTO_MAX_TOKENS_10S -> FRAMES_10S
            has30s -> FRAMES_30S
            has10s -> FRAMES_10S
            else -> FRAMES_30S
        }
    }
}
