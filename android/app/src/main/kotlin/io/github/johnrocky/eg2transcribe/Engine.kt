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

    /** The default model: the fp16-weight pair `eg2_audio_f16safe_fp16w_w30.tflite` + `_w10.tflite` (README.md, The model files). */
    const val DEFAULT_MODEL = "eg2_audio_f16safe_fp16w_w30.tflite"
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
        /** The file asked for and its size (a fallback may run another window's file: [Encoder.file]). */
        val wanted: File,
        val wantedBytes: Long,
        val fallback: String?,
        val delegateLines: List<String>,
        /** The NPU attempt behind this load (also when it fell back); null when the NPU was not asked for. */
        val npu: NpuAttempt?,
        /** The file that runs ([Encoder.file]) with its size and SHA-256, read in this load. */
        val stamp: FileStamp,
        /** The screen's Encoder line without the compile time ([encoderNote]); the run report's `model.note`. */
        val note: String,
    )

    /** An NPU attempt: the cache, the compile, whether LiteRT's dispatch delegate took the graph, and the error if it failed. */
    class NpuAttempt(
        val prepared: Npu.Prepared?,
        /** True: "Replacing … with delegate (DispatchDelegate)" in the log; false: the graph went to the CPU instead; null: the log did not say. */
        val dispatch: Boolean?,
        val compileMs: Double?,
        val cacheBytesAfter: Long?,
        val keyLines: List<String>,
        val error: String?,
    )

    /**
     * The reader in use and its vocabulary. [requested] = the accelerator asked for (the encoder's), [fallback] = why
     * it runs elsewhere; [note] = the screen's line.
     */
    class ReaderSet(
        val reader: Reader,
        val vocab: List<String>,
        val note: String,
        val requested: Backend?,
        val fallback: String?,
        /** The window length (frames) this set was picked for, and the files it was picked from (name -> bytes). */
        val windowFrames: Int,
        val key: List<Pair<String, Long>>,
        /** The GPU precision asked for (fp32 unless a launch extra set another). */
        val precision: ReaderPrecision = ReaderPrecision.FP32,
        /** The reader file with its size and SHA-256, read in this load; null for the stub. */
        val stamp: FileStamp? = null,
        /** True: the reader's graph lives on the reader's own thread and reads window k while the encoder runs k + 1. */
        val ownThread: Boolean = false,
    ) {
        val tflite: TfliteReader? get() = reader as? TfliteReader
    }

    @Volatile var loaded: Loaded? = null
        private set
    @Volatile var readers: ReaderSet? = null
        private set

    fun filesDir(context: Context): File = requireNotNull(context.getExternalFilesDir(null)) { "no external files dir" }

    /** The app's reader directory: files/reader/ (reader_w30.tflite, reader_w10.tflite, vocab.txt). */
    fun readerDir(context: Context): File = File(filesDir(context), "reader")

    /** What a reader load depends on: the files of the reader dir (name, size). */
    private fun readerKey(dir: File): List<Pair<String, Long>> =
        (TfliteReader.FILES.values + "vocab.txt").map { File(dir, it) }.filter { it.isFile }.map { it.name to it.length() }

    /** The accelerator the reader runs on for an encoder on [encoder]: the same one, the NPU's encoder gets GPU fp32. */
    fun readerBackend(encoder: Backend): Backend = if (encoder == Backend.NPU) Backend.GPU_FP32 else encoder

    /**
     * Whether the reader gets its own thread for an encoder on [encoder]: on the NPU, unless [serial] (launch extra
     * `serial_reader`). Then window k's reader runs on the GPU while the encoder runs window k + 1 on the NPU; on the
     * CPU and the GPU the encoder and the reader share the hardware, and both run on the LiteRT thread, one after the
     * other.
     */
    fun readerOwnThread(encoder: Backend, serial: Boolean): Boolean = encoder == Backend.NPU && !serial

    /**
     * The trained reader of files/reader/ for windows of [windowFrames] frames, compiled for [backend] (a GPU -> the
     * CPU when it cannot compile or run it) on the reader's own thread when [ownThread], or the stub with the reason in
     * [ReaderSet.note] when the files are not all there or cannot be used. Kept while the files, the accelerator, the
     * window length and the thread stay the same.
     */
    fun ensureReader(context: Context, backend: Backend, windowFrames: Int, precision: ReaderPrecision = ReaderPrecision.FP32, ownThread: Boolean = false, onStage: (String) -> Unit = {}): ReaderSet {
        val dir = readerDir(context)
        val key = readerKey(dir)
        readers?.takeIf { current(it, key, backend, windowFrames, precision, ownThread) }?.let { return it }
        closeReader()
        return loadReader(context, dir, key, backend, windowFrames, precision, ownThread, onStage).also { readers = it }
    }

    private fun loadReader(context: Context, dir: File, key: List<Pair<String, Long>>, backend: Backend, windowFrames: Int, precision: ReaderPrecision, ownThread: Boolean, onStage: (String) -> Unit): ReaderSet {
        val vocabFile = File(dir, "vocab.txt")
        val file = TfliteReader.fileFor(dir, windowFrames)
        var why = "no trained reader in files/reader/"
        if (dir.isDirectory && (file != null || vocabFile.isFile || TfliteReader.FILES.values.any { File(dir, it).isFile })) {
            val missing = listOfNotNull(
                if (file == null) TfliteReader.FILES[windowFrames] ?: TfliteReader.FILES.values.first() else null,
                if (!vocabFile.isFile) "vocab.txt" else null,
            )
            if (missing.isEmpty()) {
                try {
                    val vocab = WordTable.readVocab(vocabFile)
                    var fallback: String? = null
                    val want = readerBackend(backend)
                    onStage("Compiling the reader for ${want.label}…")
                    val r = try {
                        TfliteReader.open(context, file!!, want, vocab.size, CPU_THREADS, precision, ownThread)
                    } catch (e: EncoderException) {
                        if (!want.isGpu || e.code !in setOf("compile_failed", "run_failed", "not_finite")) throw e
                        fallback = "${e.text}. The reader runs on the CPU instead."
                        Log.w(TAG, "FALLBACK reader ${want.id} -> cpu: ${e.text}", e.cause)
                        onStage("Compiling the reader for the CPU…")
                        TfliteReader.open(context, file!!, Backend.CPU, vocab.size, CPU_THREADS, ownThread = ownThread)
                    }
                    onStage("Reading ${r.file.name} for its SHA-256…")
                    val stamp = FileStamp.of(r.file)
                    Log.i(TAG, "READER file=${r.file.name} bytes=${stamp.bytes} sha256=${stamp.sha256} sha256_ms=${stamp.sha256Ms?.let { Jobs.fmt(it) }} backend=${r.backend.id} precision=${r.precision} tokens=${r.tokens} slices=${r.slices} words=${vocab.size} compile_ms=${Jobs.fmt(r.compileMs)} warmup_ms=${Jobs.fmt(r.warmupMs)} fallback=${fallback != null} own_thread=${r.ownThread}")
                    return ReaderSet(r, vocab, readerNote(r.backend, r.precision, stamp, vocab.size), want, fallback, windowFrames, key, precision, stamp, r.ownThread)
                } catch (e: Throwable) {
                    why = "files/reader/ could not be used: ${e.message}"
                    Log.w(TAG, "READER_FALLBACK $why", e)
                }
            } else {
                why = "files/reader/ is missing ${missing.joinToString(" and ")}"
            }
        }
        val vocab = context.assets.open("stub_vocab.txt").bufferedReader().readLines().map { it.trim() }.filter { it.isNotEmpty() }
        Log.i(TAG, "READER stub words=${vocab.size} ($why)")
        val scanThreads = if (backend == Backend.CPU) 2 else SCAN_THREADS
        return ReaderSet(Stub.reader(vocab, scan, scanThreads), vocab, "Stub reader: the words are placeholders ($why)", readerBackend(backend), null, windowFrames, key, precision)
    }

    /** The Reader line: "Reader on GPU fp32 · reader_w30.tflite (387 MB, sha256 129d6aa2d829) · 200,000 words". */
    fun readerNote(backend: Backend, precision: String, stamp: FileStamp, words: Int): String =
        "Reader on ${backend.label}" + (if (precision == ReaderPrecision.FP16_ACC32.id) " (fp16, fp32 accumulation)" else "") +
            " · ${stamp.label} · ${"%,d".format(Locale.US, words)} words"

    /** The Encoder line: "Encoder on GPU fp32 · eg2_audio_f16safe_fp16w_w30.tflite (589 MB, sha256 f0f6c37ee62b) · signature audio_30s". */
    fun encoderNote(backend: Backend, threads: Int, stamp: FileStamp, signatures: List<String>): String =
        "Encoder on ${backend.label}" + (if (backend == Backend.CPU) " ($threads threads)" else "") +
            " · ${stamp.label} · " + (if (signatures.size == 1) "signature " else "signatures ") + signatures.joinToString(", ")

    /**
     * The same files, accelerator, window length, precision and thread: a reader (or a stub after a failed load) is not
     * loaded again. A stub has no thread of its own: it stands for either.
     */
    private fun current(r: ReaderSet, key: List<Pair<String, Long>>, backend: Backend, windowFrames: Int, precision: ReaderPrecision, ownThread: Boolean): Boolean =
        r.key == key && r.requested == readerBackend(backend) && r.windowFrames == windowFrames && r.precision == precision &&
            (r.tflite == null || r.ownThread == ownThread)

    /** True when [ensureReader] would return the loaded reader without work. */
    fun isReaderLoaded(context: Context, backend: Backend, windowFrames: Int, precision: ReaderPrecision = ReaderPrecision.FP32, ownThread: Boolean = false): Boolean =
        readers?.let { current(it, readerKey(readerDir(context)), backend, windowFrames, precision, ownThread) } ?: false

    /** True when files/reader/ has the reader file of that window length itself (not a longer one that pads). */
    fun hasReaderFile(context: Context, windowFrames: Int): Boolean = TfliteReader.FILES[windowFrames]?.let { File(readerDir(context), it).isFile } ?: false

    fun closeReader() {
        val r = readers ?: return
        readers = null
        LiteRt.on(r.ownThread) { r.reader.close() }
    }

    /** The file [backend] runs for windows of [windowFrames] frames: the NPU only the 10 s file, the others the asked length first. */
    fun fileFor(choice: ModelChoice, backend: Backend, windowFrames: Int): File? =
        if (backend == Backend.NPU) choice.npuFile
        else choice.fileFor(windowFrames) ?: choice.fileFor(if (windowFrames == WindowPlan.FRAMES_30S) WindowPlan.FRAMES_10S else WindowPlan.FRAMES_30S)

    /**
     * Compiles the file of [choice] that [backend] runs for windows of [windowFrames] frames ([fileFor]) unless that is
     * what is loaded. Fallbacks, each said on the screen: the NPU -> GPU fp32 (the asked window's file) when the model
     * has no 10 s file, the NPU is not available here, cannot compile or run the graph, or did not get the graph
     * (LiteRT 2.2.0 moves a graph its NPU compiler cannot take to the CPU without an error); a GPU -> the CPU when it
     * cannot compile or run the graph or gives non-finite values in the warm-up. [onStage] gets the screen's sentence
     * for each stage.
     */
    fun ensureEncoder(context: Context, choice: ModelChoice, backend: Backend, windowFrames: Int, onStage: (String) -> Unit = {}): Loaded {
        val wanted = fileFor(choice, backend, windowFrames)
        isLoaded(choice, backend, windowFrames)?.let { return it }
        closeEncoder()
        val since = OwnLog.now()
        var fallback: String? = null
        var npu: NpuAttempt? = null
        var enc: Encoder? = null
        if (backend == Backend.NPU) {
            if (wanted == null) {
                if (choice.missing) throw EncoderException("model_missing", "The encoder file ${choice.name} is not on this phone.")
                val why = "The NPU runs only the 10 s encoder file (${choice.name}_w10.tflite), which is not on this phone"
                npu = NpuAttempt(null, null, null, null, emptyList(), why)
            } else {
                val (e, attempt) = tryNpu(context, wanted, onStage)
                enc = e
                npu = attempt
            }
            if (enc == null) {
                fallback = "${npu!!.error!!.removeSuffix(".")}. Running on GPU fp32 instead."
                Log.w(TAG, "FALLBACK backend=npu -> gpu_fp32: ${npu.error}")
            }
        }
        if (enc == null) {
            val gpu = if (backend == Backend.NPU) Backend.GPU_FP32 else backend
            val file = fileFor(choice, gpu, windowFrames) ?: throw EncoderException("model_missing", "The encoder file ${choice.name} is not on this phone.")
            onStage("Compiling the encoder for ${gpu.label}…")
            enc = try {
                Encoder.open(context, file, gpu, CPU_THREADS).also { e ->
                    if (gpu.isGpu && !e.warmupFinite) {
                        LiteRt.call { e.close() }
                        throw EncoderException("not_finite", "${gpu.label} gave NaN or infinite values in a test run")
                    }
                }
            } catch (e: EncoderException) {
                if (!gpu.isGpu || e.code !in setOf("compile_failed", "run_failed", "not_finite")) throw e
                fallback = (fallback?.let { "$it " } ?: "") + "${e.text}. Running on the CPU instead."
                Log.w(TAG, "FALLBACK backend=${gpu.id} -> cpu: ${e.text}", e.cause)
                onStage("Compiling the encoder for the CPU…")
                Encoder.open(context, file, Backend.CPU, CPU_THREADS)
            }
        }
        val lines = OwnLog.delegateLines(since)
        for (l in lines) Log.i(TAG, "DELEGATE $l")
        val e = enc!!
        // The SHA-256 of the file that runs, once per load; an NPU load has it from Npu.prepare (its cache's link name).
        val stamp = npu?.prepared?.takeIf { it.file == e.file }?.let { FileStamp(e.file, e.file.length(), it.sha256, it.sha256Ms) }
            ?: run {
                onStage("Reading ${e.file.name} for its SHA-256…")
                FileStamp.of(e.file)
            }
        val w = loadKey(choice, backend, windowFrames) ?: e.file
        return Loaded(e, backend, w, w.length(), fallback, lines, npu, stamp, encoderNote(e.backend, e.threads, stamp, e.signatureNames)).also { loaded = it }
    }

    /** The file a load for these settings is known by: the one [fileFor] picks, or for the NPU without a 10 s file the GPU's. */
    private fun loadKey(choice: ModelChoice, backend: Backend, windowFrames: Int): File? =
        fileFor(choice, backend, windowFrames) ?: if (backend == Backend.NPU) fileFor(choice, Backend.GPU_FP32, windowFrames) else null

    /** The loaded encoder when it is the one these settings ask for (same file, same size: not replaced since), else null. */
    fun isLoaded(choice: ModelChoice, backend: Backend, windowFrames: Int): Loaded? {
        val l = loaded ?: return null
        val key = loadKey(choice, backend, windowFrames) ?: return null
        return l.takeIf { it.requested == backend && it.wanted == key && it.wantedBytes == key.length() }
    }

    /** One NPU compile of [file]: the encoder, or null and the reason in the attempt's error. */
    private fun tryNpu(context: Context, file: File, onStage: (String) -> Unit): Pair<Encoder?, NpuAttempt> {
        val since = OwnLog.now()
        var prepared: Npu.Prepared? = null
        fun attempt(dispatch: Boolean?, compileMs: Double?, error: String?): NpuAttempt {
            val log = OwnLog.lines(since)
            val cacheBytes = prepared?.let { p -> listOf(p.cacheDir, p.plainCacheDir).distinct().sumOf { Npu.bytes(it) } }
            return NpuAttempt(prepared, dispatch ?: OwnLog.dispatchDelegate(log), compileMs, cacheBytes, OwnLog.npuKeyLines(log), error)
        }
        return try {
            Npu.unavailable(context)?.let { return null to attempt(null, null, it) }
            onStage("Checking the NPU cache for ${file.name}…")
            val p = Npu.prepare(context, file)
            prepared = p
            Log.i(TAG, "NPU_CACHE sha256=${p.sha256} sha256_ms=${Jobs.fmt(p.sha256Ms)} path=${p.path} cache_dir=${p.cacheDir} cached_before=${p.cachedBefore}" + (p.note?.let { " note=$it" } ?: ""))
            onStage(if (p.cachedBefore) "Loading the NPU encoder from its cache…" else "Compiling the encoder for the NPU (first time, about 2–3 minutes)…")
            val e = Encoder.open(context, file, Backend.NPU, CPU_THREADS, p)
            val dispatch = OwnLog.dispatchDelegate(OwnLog.lines(since))
            when {
                dispatch == false -> {
                    LiteRt.call { e.close() }
                    null to attempt(false, e.compileMs, "The NPU compiler did not take the encoder graph (LiteRT put it on the CPU)")
                }
                !e.warmupFinite -> {
                    LiteRt.call { e.close() }
                    null to attempt(dispatch, e.compileMs, "The NPU gave NaN or infinite values in a test run")
                }
                else -> e to attempt(dispatch, e.compileMs, null)
            }
        } catch (e: EncoderException) {
            null to attempt(null, null, e.text)
        } catch (t: Throwable) {
            Log.e(TAG, "ERROR npu ${t.message}", t)
            null to attempt(null, null, "The NPU failed: ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    fun closeEncoder() {
        val l = loaded ?: return
        loaded = null
        LiteRt.call { l.encoder.close() }
    }

    /**
     * Runs the pipeline on the LiteRT thread (the encoder, and the reader's graph unless it has its own thread: then
     * window k's reader runs there while the encoder runs window k + 1), the labels on the decode thread.
     */
    fun transcribe(pcm: PcmSource, loaded: Loaded, readers: ReaderSet, windowFrames: Int, overlapTokens: Int, listener: Pipeline.Listener, dump: Pipeline.Dump?): Pipeline.Result {
        cancel.set(false)
        val pipeline = Pipeline(loaded.encoder, readers.reader, readers.vocab, prep, decode, if (readers.ownThread) LiteRt.readerExecutor else null)
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

    /** This process's log lines since [since] (logcat's time format), or null when logcat cannot be read. */
    fun lines(since: String): List<String>? = runCatching {
        val p = ProcessBuilder("logcat", "-d", "-v", "time", "-T", since, "--pid=${android.os.Process.myPid()}").redirectErrorStream(true).start()
        val lines = p.inputStream.bufferedReader().readLines()
        p.waitFor()
        lines
    }.getOrNull()

    private val KEYS = listOf("delegate", "replacing", "accelerator", "xnnpack", "opencl", "qnn", "dispatch", "litert", "tflite", "npu")

    /**
     * Lines since [since] that say where a graph runs (LiteRT, TFLite, delegate, accelerator, XNNPACK, OpenCL, QNN):
     * every "Replacing … node(s) with delegate" line once, the NPU compiler's summary lines, then the last ones up to
     * [max] (an NPU compile logs thousands of QNN lines).
     */
    fun delegateLines(since: String, max: Int = 60): List<String> {
        val all = lines(since) ?: return listOf("logcat could not be read")
        val keyed = all.filter { l -> !l.contains(Engine.TAG) && KEYS.any { l.lowercase().contains(it) } }
        val firsts = LinkedHashMap<String, String>()
        for (l in keyed) if (l.contains("Replacing ") || isNpuSummary(l)) firsts.putIfAbsent(l.substringAfter("): ", l), l)
        val head = firsts.values.take(max * 2 / 3)
        return head + keyed.takeLast(maxOf(0, max - head.size)).filter { it !in head }
    }

    private fun isNpuSummary(l: String) = listOf(
        "caching enabled", "Partitioned subgraph", "compiler plugins were applied", "Failed to apply compiler plugins", "Plugin errs",
        "initialized from cached model", "Flatbuffer model initialized", "NPU accelerator", "QnnDsp <E>",
    ).any { l.contains(it) }

    /** The NPU compiler's lines in [log] (cache, partition, plugin result, dispatch), at most 40. */
    fun npuKeyLines(log: List<String>?): List<String> =
        log?.filter { l -> !l.contains(Engine.TAG) && (isNpuSummary(l) || (l.contains("Replacing ") && l.contains("delegate"))) }?.distinct()?.take(40) ?: emptyList()

    /**
     * Whether LiteRT's dispatch delegate took a graph, from [log]: true when "Replacing … with delegate
     * (DispatchDelegate)" is there; false when only CPU / GPU delegates replaced nodes of the main graph; null when
     * the log says neither (or could not be read).
     */
    fun dispatchDelegate(log: List<String>?): Boolean? {
        if (log == null) return null
        val replacing = log.filter { it.contains("Replacing ") && it.contains("with delegate") }
        if (replacing.any { it.contains("(DispatchDelegate)") }) return true
        if (replacing.any { it.contains("subgraph 0") }) return false
        return null
    }
}

/** JSON helpers for the run report. */
internal fun jsonOf(vararg pairs: Pair<String, Any?>): JSONObject = JSONObject().apply { for ((k, v) in pairs) put(k, v ?: JSONObject.NULL) }

internal fun jsonArray(xs: Iterable<Any?>): JSONArray = JSONArray().apply { xs.forEach { put(it ?: JSONObject.NULL) } }
