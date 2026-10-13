package io.github.johnrocky.eg2transcribe

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ThreadFactory
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONArray
import org.json.JSONObject

/**
 * The process-wide model state: the compiled encoder, the reader and word table, the worker threads, and the one
 * transcription that may run. An activity that is recreated finds them here.
 */
object Engine {
    const val TAG = "eg2transcribe"
    const val DEFAULT_MODEL = "eg2_audio_fp32.tflite"
    const val CPU_THREADS = 4
    const val SCAN_THREADS = 4

    /** Shorter recordings are refused (the encoder needs some context; one word is about half a second). */
    const val MIN_SECONDS = 1.0

    /**
     * The loudness floor: RMS of the whole input, 1.0 = full scale. 0.005 (-46 dBFS) sits above two silent recordings
     * made through the microphone of a Galaxy S26 (RMS 0.0034 and 0.0009, the EmbeddingGemma 2 photo-search sample of
     * hfmodels-android, 2026-10-08). Quieter input is refused as "no speech" before the encoder runs.
     */
    const val MIN_RMS = 0.005

    private fun named(name: String) = ThreadFactory { r -> Thread(r, name).apply { isDaemon = true } }

    /** Jobs in order: load, decode, transcribe (each may hand its native work to the LiteRT thread). */
    val work: ExecutorService = Executors.newSingleThreadExecutor(named("work"))
    private val prep: ExecutorService = Executors.newSingleThreadExecutor(named("mel"))
    private val decode: ExecutorService = Executors.newSingleThreadExecutor(named("decode"))
    private val scan: ExecutorService = Executors.newFixedThreadPool(SCAN_THREADS, named("scan"))

    val cancel = AtomicBoolean(false)

    class Loaded(
        val encoder: Encoder,
        val requested: Backend,
        val fallback: String?,
        val delegateLines: List<String>,
    )

    class ReaderSet(val reader: Reader, val table: WordTable, val note: String)

    @Volatile var loaded: Loaded? = null
        private set
    @Volatile var readers: ReaderSet? = null
        private set

    fun filesDir(context: Context): File = requireNotNull(context.getExternalFilesDir(null)) { "no external files dir" }

    /** The reader in files/reader/ when it is complete and readable, else the stub (with the reason in [ReaderSet.note]). */
    fun ensureReader(context: Context): ReaderSet = readers ?: synchronized(this) {
        readers ?: loadReader(context).also { readers = it }
    }

    private fun loadReader(context: Context): ReaderSet {
        val dir = File(filesDir(context), "reader")
        val files = listOf("reader.tflite", "vocab.txt")
        val table = listOf("word_table.npy", "word_table.bin").map { File(dir, it) }.firstOrNull { it.isFile }
        var why = "no trained reader yet"
        if (dir.isDirectory && (File(dir, "reader.tflite").isFile || table != null)) {
            val missing = files.filterNot { File(dir, it).isFile } + (if (table == null) listOf("word_table.npy or word_table.bin") else emptyList())
            if (missing.isEmpty()) {
                try {
                    val t = WordTable.load(table!!, File(dir, "vocab.txt"))
                    val r = LiteRt.call { TfliteReader.open(context, dir, CPU_THREADS) }
                    if (r.dim != t.dim) {
                        LiteRt.call { r.close() }
                        throw IllegalArgumentException("reader.tflite gives ${r.dim}-d vectors, the word table has ${t.dim}")
                    }
                    Log.i(TAG, "READER tflite dim=${r.dim} words=${t.size}")
                    return ReaderSet(r, t, "reader.tflite, ${t.size} words")
                } catch (e: Throwable) {
                    why = "reader/ could not be used: ${e.message}"
                    Log.w(TAG, "READER_FALLBACK $why", e)
                }
            } else {
                why = "reader/ is missing ${missing.joinToString(", ")}"
            }
        }
        val vocab = context.assets.open("stub_vocab.txt").bufferedReader().readLines().map { it.trim() }.filter { it.isNotEmpty() }
        Log.i(TAG, "READER stub words=${vocab.size} ($why)")
        return ReaderSet(Stub.reader(), Stub.table(vocab), "stub: words are placeholders ($why)")
    }

    /**
     * Compiles [file] for [backend] unless that is what is loaded. A GPU that fails to compile or to run, or gives
     * non-finite values in the warm-up, falls back to the CPU and says so; the NPU does not fall back.
     */
    fun ensureEncoder(context: Context, file: File, backend: Backend): Loaded {
        loaded?.let { l -> if (l.requested == backend && l.encoder.file == file && l.encoder.file.length() == file.length()) return l }
        closeEncoder()
        val since = OwnLog.now()
        var fallback: String? = null
        val enc = try {
            Encoder.open(context, file, backend, CPU_THREADS).also { e ->
                if (backend.isGpu && !e.warmupFinite) {
                    LiteRt.call { e.close() }
                    throw EncoderException("not_finite", "${backend.label} gave NaN or infinite values in a test run")
                }
            }
        } catch (e: EncoderException) {
            if (!backend.isGpu || e.code !in setOf("compile_failed", "run_failed", "not_finite")) throw e
            fallback = "${e.text}. Running on the CPU instead."
            Log.w(TAG, "FALLBACK backend=${backend.id} -> cpu: ${e.text}", e.cause)
            Encoder.open(context, file, Backend.CPU, CPU_THREADS)
        }
        val lines = OwnLog.delegateLines(since)
        for (l in lines) Log.i(TAG, "DELEGATE $l")
        return Loaded(enc, backend, fallback, lines).also { loaded = it }
    }

    fun closeEncoder() {
        val l = loaded ?: return
        loaded = null
        LiteRt.call { l.encoder.close() }
    }

    /**
     * Runs the pipeline on the LiteRT thread. The word search gets 4 threads when the encoder is on the GPU or the NPU,
     * and 2 when it shares the CPU with XNNPACK's 4.
     */
    fun transcribe(pcm: PcmSource, loaded: Loaded, readers: ReaderSet, windowFrames: Int, overlapTokens: Int, listener: Pipeline.Listener, dump: Pipeline.Dump?): Pipeline.Result {
        cancel.set(false)
        val scanThreads = if (loaded.encoder.backend == Backend.CPU) 2 else SCAN_THREADS
        val pipeline = Pipeline(loaded.encoder, readers.reader, readers.table, prep, decode, scan, scanThreads)
        return LiteRt.call { pipeline.run(pcm, windowFrames, overlapTokens, listener, { cancel.get() }, dump) }
    }

    fun deviceJson(context: Context): JSONObject = JSONObject()
        .put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("soc", Npu.soc())
        .put("android", Build.VERSION.RELEASE).put("sdk", Build.VERSION.SDK_INT).put("build", Build.DISPLAY)
        .put("thermal_status", thermal(context))

    fun thermal(context: Context): Int =
        if (Build.VERSION.SDK_INT >= 29) (context.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus else -1

    /** VmHWM / VmRSS of this process, kB. */
    fun memoryJson(): JSONObject = runCatching {
        val o = JSONObject()
        for (line in File("/proc/self/status").readLines()) {
            val parts = line.split(Regex("\\s+"))
            if (parts.size >= 2 && parts[0] in listOf("VmHWM:", "VmRSS:")) o.put(parts[0].removeSuffix(":").lowercase() + "_kb", parts[1].toLong())
        }
        o
    }.getOrElse { JSONObject().put("readable", false) }
}

/** This process's own log lines (an app may read its own log without a permission). */
object OwnLog {
    private val stamp = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)

    fun now(): String = synchronized(stamp) { stamp.format(Date()) }

    /** Lines since [since] that say where a graph runs: LiteRT, TFLite, delegate, accelerator, XNNPACK, OpenCL, QNN. */
    fun delegateLines(since: String, max: Int = 60): List<String> = runCatching {
        val p = ProcessBuilder("logcat", "-d", "-v", "time", "-T", since, "--pid=${android.os.Process.myPid()}").redirectErrorStream(true).start()
        val lines = p.inputStream.bufferedReader().readLines()
        p.waitFor()
        val keys = listOf("delegate", "replacing", "accelerator", "xnnpack", "opencl", "qnn", "dispatch", "litert", "tflite", "npu")
        lines.filter { l -> !l.contains(Engine.TAG) && keys.any { l.lowercase().contains(it) } }.takeLast(max)
    }.getOrElse { listOf("logcat could not be read: ${it.message}") }
}

/** JSON helpers for the run report. */
internal fun jsonOf(vararg pairs: Pair<String, Any?>): JSONObject = JSONObject().apply { for ((k, v) in pairs) put(k, v ?: JSONObject.NULL) }

internal fun jsonArray(xs: Iterable<Any?>): JSONArray = JSONArray().apply { xs.forEach { put(it ?: JSONObject.NULL) } }
