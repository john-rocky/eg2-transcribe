package io.github.johnrocky.eg2transcribe

import java.io.File

/**
 * Which encoder file runs which window. The export gives one single-signature file per window length:
 * `<name>_w30.tflite` (signature `audio_30s`) and `<name>_w10.tflite` (`audio_10s`). The two files of a name are one
 * model, picked by either file name, and a window length is offered when its file is there. Any other `.tflite` is one
 * file whose signatures (`audio_30s`, `audio_10s` or both, like the first two-signature export) decide its windows
 * when it is opened.
 */
class ModelChoice private constructor(
    /** The model's name: `<name>` of a pair, or the file name of a single file. */
    val name: String,
    private val w30: File?,
    private val w10: File?,
    /** The file of a model that is not a `_w30` / `_w10` pair (it may be missing). */
    val single: File?,
) {
    val isPair: Boolean get() = single == null

    /** The files that are there. */
    val files: List<File> get() = listOfNotNull(w30, w10, single?.takeIf { it.isFile })

    /** True when no file of this model is on the phone. */
    val missing: Boolean get() = files.isEmpty()

    /** The file for windows of [frames] mel frames: a pair's own file for that length, or the single file. */
    fun fileFor(frames: Int): File? = when {
        single != null -> single.takeIf { it.isFile }
        frames == WindowPlan.FRAMES_30S -> w30
        frames == WindowPlan.FRAMES_10S -> w10
        else -> null
    }

    /** The window lengths this model has a file for; a single file may have either (its signatures decide). */
    val windows: Set<Int>
        get() = if (single != null) (if (single.isFile) setOf(WindowPlan.FRAMES_30S, WindowPlan.FRAMES_10S) else emptySet())
        else listOfNotNull(w30?.let { WindowPlan.FRAMES_30S }, w10?.let { WindowPlan.FRAMES_10S }).toSet()

    /** The NPU compiles only a single-signature 10 s file (the 30 s graph runs a phone out of memory in the compile). */
    val npuFile: File? get() = w10

    /** One line for the screen, e.g. "eg2_audio_f16safe_fp16w · 30 s + 10 s files · 1.18 GB". */
    fun describe(): String {
        if (missing) return "$name · not on this phone"
        val parts = if (single != null) "one file" else listOfNotNull(w30?.let { "30 s" }, w10?.let { "10 s" }).joinToString(" + ") + if (w30 != null && w10 != null) " files" else " file only"
        return "$name · $parts · ${megabytes(files.sumOf { it.length() })}"
    }

    companion object {
        private val PAIR = Regex("^(.+)_w(10|30)\\.tflite$")

        /** The model [chosen] names in [dir]: a `_w30` / `_w10` file name stands for its pair. */
        fun of(dir: File, chosen: String): ModelChoice {
            val m = PAIR.matchEntire(chosen) ?: return ModelChoice(chosen, null, null, File(dir, chosen))
            val base = m.groupValues[1]
            return ModelChoice(
                base,
                File(dir, "${base}_w30.tflite").takeIf { it.isFile },
                File(dir, "${base}_w10.tflite").takeIf { it.isFile },
                null,
            )
        }

        /** Every model in [dir] (a pair once), as the file name that picks it, sorted by name. */
        fun all(dir: File): List<String> {
            val names = dir.listFiles { f -> f.isFile && f.name.endsWith(".tflite") }?.map { it.name }?.sorted() ?: return emptyList()
            val seen = HashSet<String>()
            val out = ArrayList<String>()
            for (n in names) {
                val key = PAIR.matchEntire(n)?.groupValues?.get(1)?.let { "pair:$it" } ?: "file:$n"
                if (seen.add(key)) out += n
            }
            return out
        }

        /** The model's name for a file name: `<name>` of `<name>_w30.tflite` / `<name>_w10.tflite`, else the file name. */
        fun nameOf(fileName: String): String = PAIR.matchEntire(fileName)?.groupValues?.get(1) ?: fileName
    }
}

/** "589 MB", "1.18 GB". */
internal fun megabytes(bytes: Long): String =
    if (bytes >= 1L shl 30) String.format(java.util.Locale.US, "%.2f GB", bytes / 1e9) else String.format(java.util.Locale.US, "%.0f MB", bytes / 1e6)

/**
 * A model file as the screen and the run report name it: its name, its size and its SHA-256, read once when the file
 * is loaded (work thread), and the ms that read took. [sha256] is null when the file could not be read for it.
 */
class FileStamp(val file: File, val bytes: Long, val sha256: String?, val sha256Ms: Double?) {
    /** "reader_w30.tflite (387 MB, sha256 129d6aa2d829)": the first 12 hex digits, as the NPU cache names its entries. */
    val label: String get() = "${file.name} (${megabytes(bytes)}" + (sha256?.let { ", sha256 ${it.take(12)}" } ?: "") + ")"

    companion object {
        /** Reads [file] to its end for the SHA-256; a read error leaves the hash out. */
        fun of(file: File): FileStamp {
            val t = System.nanoTime()
            val sha = runCatching { Npu.sha256(file) }.getOrNull()
            return FileStamp(file, file.length(), sha, if (sha != null) (System.nanoTime() - t) / 1e6 else null)
        }
    }
}
