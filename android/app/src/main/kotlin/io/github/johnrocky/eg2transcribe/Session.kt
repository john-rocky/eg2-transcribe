package io.github.johnrocky.eg2transcribe

import java.io.File
import java.util.Locale

/**
 * What the screen shows, kept in the process: the worker threads write it, the activity renders it ([listener] posts
 * a render on the main thread), and an activity that is created again finds the running job's state here.
 */
object Session {
    enum class Tone { IDLE, WORK, LIVE, OK, WARN }

    // Settings.
    @Volatile var backend = Backend.GPU_FP32
    @Volatile var modelName = Engine.DEFAULT_MODEL
    @Volatile var windowFrames = WindowPlan.FRAMES_30S
    @Volatile var overlapTokens = WindowPlan.DEFAULT_OVERLAP
    @Volatile var minRms = Engine.MIN_RMS

    // State.
    @Volatile var pill = "STARTING"
    @Volatile var tone = Tone.IDLE
    @Volatile var status = ""
    @Volatile var statusOpensSettings = false
    @Volatile var modelLine = ""
    @Volatile var readerLine = ""
    @Volatile var readerIsStub = true
    @Volatile var engineLine = ""
    @Volatile var fallbackLine: String? = null
    @Volatile var inputLine = ""
    @Volatile var progressLine = ""
    @Volatile var progress = 0.0
    @Volatile var detailLine = ""
    @Volatile var note: String? = null
    @Volatile var loading = false
    @Volatile var recording = false
    @Volatile var decoding = false
    @Volatile var transcribing = false
    @Volatile var lastInput: Input? = null
    @Volatile var recordMode = false

    /** System.nanoTime() when the current model load started, 0 when none runs (the screen shows the time it takes). */
    @Volatile var loadingSince = 0L

    /** The window length a run uses: the loaded encoder's when it has no file for the setting (the NPU: 10 s), else the setting. */
    fun effectiveWindowFrames(): Int {
        val sizes = Engine.loaded?.encoder?.windowSizes
        return when {
            sizes != null -> if (windowFrames in sizes) windowFrames else sizes.max()
            backend == Backend.NPU -> WindowPlan.FRAMES_10S
            else -> windowFrames
        }
    }

    /** A decoded input kept for "Transcribe again": the PCM file and what it was. */
    class Input(val pcm: File, val label: String, val json: org.json.JSONObject)

    val busy: Boolean get() = loading || recording || decoding || transcribing

    @Volatile var listener: (() -> Unit)? = null

    fun changed() {
        listener?.invoke()
    }

    // The transcript: lines of words, a new line with a timestamp every 10 s of audio.
    private val text = StringBuilder()
    private var lineStart = Double.NEGATIVE_INFINITY
    private var words = 0

    fun clearTranscript() = synchronized(text) {
        text.setLength(0)
        lineStart = Double.NEGATIVE_INFINITY
        words = 0
    }

    fun addWords(ws: List<Word>) = synchronized(text) {
        for (w in ws) {
            if (w.startSeconds >= lineStart + LINE_SECONDS) {
                if (text.isNotEmpty()) text.append('\n')
                text.append('[').append(clock(w.startSeconds)).append("] ")
                lineStart = w.startSeconds
            } else {
                text.append(' ')
            }
            text.append(w.text)
            words++
        }
    }

    /** The transcript from character [from] on, and the current length. */
    fun transcriptSince(from: Int): Pair<String, Int> = synchronized(text) {
        val f = from.coerceIn(0, text.length)
        text.substring(f) to text.length
    }

    fun transcriptLength(): Int = synchronized(text) { text.length }

    fun wordCount(): Int = synchronized(text) { words }

    private const val LINE_SECONDS = 10.0

    fun clock(seconds: Double): String {
        val s = seconds.toLong()
        return if (s >= 3600) String.format(Locale.US, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
        else String.format(Locale.US, "%02d:%02d", s / 60, s % 60)
    }
}
