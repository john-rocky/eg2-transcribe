package io.github.johnrocky.eg2transcribe

import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

/**
 * The work behind the buttons, on [Engine.work] (one job at a time): load the model, decode a file or record the
 * microphone, refuse what is too short or too quiet, run the pipeline, write the run report. Every outcome, failures
 * included, ends in [Session] (the screen) and one RESULT log line; with a report name it is also written as JSON.
 */
object Jobs {
    /** [report]: a file name under the app's files dir for the run JSON (the device check sets it); [dump]: also per-token files. */
    class Request(val report: String?, val dump: Boolean)

    private const val TAG = Engine.TAG

    fun modelChoice(context: Context): ModelChoice = ModelChoice.of(Engine.filesDir(context), Session.modelName)

    /** Loads the reader and the encoder for the current settings (work thread). True when the encoder is ready. */
    fun loadModel(context: Context): Boolean {
        val readers = Engine.ensureReader(context)
        Session.readerLine = readers.note
        Session.readerIsStub = readers.reader.isStub
        val choice = modelChoice(context)
        Session.modelLine = choice.describe()
        if (Engine.isLoaded(choice, Session.backend, Session.windowFrames) != null) return true
        Session.loading = true
        Session.loadingSince = System.nanoTime()
        Session.pill = "LOADING MODEL"
        Session.tone = Session.Tone.WORK
        Session.status = "Compiling the encoder for ${Session.backend.label}…"
        Session.fallbackLine = null
        Session.changed()
        try {
            val l = Engine.ensureEncoder(context, choice, Session.backend, Session.windowFrames) { stage ->
                Session.status = stage
                Session.changed()
            }
            val e = l.encoder
            val npu = l.npu
            Session.engineLine = "Encoder on ${e.backend.label}" + (if (e.backend == Backend.CPU) " (${e.threads} threads)" else "") +
                " · " + (if (e.backend == Backend.NPU && npu?.prepared?.cachedBefore == true) "loaded from its cache in ${secs(e.compileMs)}"
                else "compiled in ${secs(e.compileMs)}" + (if (e.backend == Backend.NPU) " (first time; later starts read the cache)" else "")) +
                " · ${e.file.name} (${e.signatureNames.joinToString(", ")})"
            Session.fallbackLine = l.fallback
            Log.i(TAG, "ENGINE_READY requested=${l.requested.id} used=${e.backend.id} compile_ms=${fmt(e.compileMs)} warmup_ms=${e.warmupMs} model=${e.file.name} bytes=${e.file.length()} fallback=${l.fallback != null}" +
                (npu?.let { " npu_dispatch=${it.dispatch} npu_cached_before=${it.prepared?.cachedBefore} npu_cache_bytes=${it.cacheBytesAfter} npu_error=${it.error}" } ?: ""))
            Session.pill = "READY"
            Session.tone = Session.Tone.IDLE
            Session.status = if (Session.lastInput == null) "Record, or open an audio file." else "Ready."
            Session.note = null
            return true
        } catch (e: EncoderException) {
            Log.e(TAG, "ERROR ${e.code} ${e.text}", e.cause)
            Session.engineLine = ""
            when (e.code) {
                "model_missing" -> {
                    Session.pill = "MODEL MISSING"
                    Session.status = "The encoder files are not on this phone yet (steps below)."
                    Session.note = missingModelNote(context)
                }
                else -> {
                    Session.pill = "LOAD FAILED"
                    Session.status = e.text
                }
            }
            Session.tone = Session.Tone.LIVE
            lastLoadError = e
            return false
        } catch (t: Throwable) {
            Log.e(TAG, "ERROR load ${t.message}", t)
            Session.pill = "LOAD FAILED"
            Session.tone = Session.Tone.LIVE
            Session.status = "${t.javaClass.simpleName}: ${t.message}"
            lastLoadError = EncoderException("load_failed", Session.status, t)
            return false
        } finally {
            Session.loading = false
            Session.loadingSince = 0L
            Session.changed()
        }
    }

    @Volatile private var lastLoadError: EncoderException? = null

    fun missingModelNote(context: Context): String {
        val dir = sdcard(Engine.filesDir(context))
        val base = ModelChoice.nameOf(Engine.DEFAULT_MODEL)
        return "The encoder is two separate files, one per window length: ${base}_w30.tflite and ${base}_w10.tflite (589 MB each; " +
            "README.md describes them). The NPU runs the 10 s file, the CPU and the GPU the 30 s file.\n\n" +
            "Either push them with adb, then come back to this screen:\n" +
            "adb push ${base}_w30.tflite $dir/\nadb push ${base}_w10.tflite $dir/\n\n" +
            "or tap Change next to Model and choose a file on this phone (it is copied into the app; copy both files of a pair)."
    }

    /** /storage/emulated/0/... as adb's /sdcard/... */
    fun sdcard(f: File): String = f.absolutePath.replace(Regex("^/storage/emulated/0"), "/sdcard")

    /** Decodes an audio file, then transcribes it. */
    fun transcribeUri(context: Context, uri: Uri, label: String, req: Request, cancel: AtomicBoolean) {
        Session.lastInput = null // input.pcm is about to be overwritten
        Session.decoding = true
        Session.pill = "READING FILE"
        Session.tone = Session.Tone.WORK
        Session.status = "Reading $label…"
        Session.progress = 0.0
        Session.progressLine = ""
        Session.changed()
        val pcm = File(context.cacheDir, "input.pcm")
        val decoded = try {
            AudioDecoder.decode(context, uri, pcm, { cancel.get() }) { f ->
                Session.progress = f
                Session.changed()
            }
        } catch (e: Throwable) {
            Session.decoding = false
            val text = if (cancel.get()) "Stopped." else "This file could not be read: ${e.message ?: e.javaClass.simpleName}"
            finish(context, req, if (cancel.get()) "cancelled" else "decode_failed", if (cancel.get()) "STOPPED" else "CANNOT READ", text, input = jsonOf("source" to "file", "label" to label))
            return
        }
        Session.decoding = false
        val json = jsonOf(
            "source" to "file", "label" to label, "samples" to decoded.samples, "seconds" to decoded.samples / 16000.0,
            "format" to decoded.format, "source_rate" to decoded.sourceRate, "channels" to decoded.channels,
            "resampled" to decoded.resampled, "decode_ms" to decoded.ms,
            "expected_samples" to decoded.expectedSamples, "missing_samples" to decoded.expectedSamples?.let { it - decoded.samples },
        )
        Log.i(TAG, "INPUT source=file label=$label samples=${decoded.samples} expected=${decoded.expectedSamples} format=${decoded.format} rate=${decoded.sourceRate} channels=${decoded.channels} resampled=${decoded.resampled} decode_ms=${fmt(decoded.ms)}")
        run(context, Session.Input(pcm, label, json), req)
    }

    /** Records until [stop] (or [ms] when set, the check's press without a finger), then transcribes. */
    fun transcribeMic(context: Context, ms: Long?, req: Request, stop: AtomicBoolean) {
        if (!Mic.granted(context)) {
            Log.i(TAG, "MIC_PERMISSION granted=false")
            Session.statusOpensSettings = true
            finish(context, req, "mic_denied", "MIC OFF", Mic.deniedText(context), input = jsonOf("source" to "mic"))
            return
        }
        Session.statusOpensSettings = false
        Session.lastInput = null // mic.pcm is about to be overwritten
        Session.recording = true
        Session.pill = "RECORDING"
        Session.tone = Session.Tone.LIVE
        Session.status = "Listening… tap Stop when you are done."
        Session.progressLine = ""
        Session.changed()
        val file = File(context.cacheDir, "mic.pcm")
        val t0 = System.nanoTime()
        val r = Mic.record(context, file, { !stop.get() && (ms == null || (System.nanoTime() - t0) / 1e6 < ms) }) { rms, s ->
            Session.progressLine = "${Session.clock(s)} recorded · level ${dbfs(rms)}"
            Session.changed()
        }
        Session.recording = false
        when (r) {
            is Mic.Result.Denied -> {
                Session.statusOpensSettings = true
                finish(context, req, "mic_denied", "MIC OFF", r.text, input = jsonOf("source" to "mic"))
            }
            is Mic.Result.Failed -> finish(context, req, "mic_failed", "MIC FAILED", "The microphone failed: ${r.error.message}", input = jsonOf("source" to "mic"))
            is Mic.Result.Recorded -> {
                Log.i(TAG, "INPUT source=mic samples=${r.samples} rms=${"%.5f".format(Locale.US, r.rms)}")
                val json = jsonOf("source" to "mic", "label" to "microphone", "samples" to r.samples, "seconds" to r.samples / 16000.0, "format" to "pcm16 16 kHz mono", "source_rate" to 16000, "channels" to 1, "resampled" to false)
                run(context, Session.Input(file, "Recording", json), req)
            }
        }
    }

    fun transcribeAgain(context: Context, req: Request) {
        val input = Session.lastInput ?: return
        run(context, input, req)
    }

    /** Checks length and loudness, loads the model, runs the pipeline and reports. */
    private fun run(context: Context, input: Session.Input, req: Request) {
        Session.lastInput = input
        Session.progress = 0.0
        Session.progressLine = ""
        Session.detailLine = ""
        val pcm = try { Pcm16File(input.pcm) } catch (e: Throwable) {
            finish(context, req, "decode_failed", "CANNOT READ", "The audio could not be opened: ${e.message}", input = input.json)
            return
        }
        pcm.use { src ->
            val seconds = src.seconds
            val rms = Loudness.rms(src)
            input.json.put("rms", rms)
            Session.inputLine = "${input.label} · ${Session.clock(seconds)}" + (input.json.optString("format").takeIf { it.isNotEmpty() }?.let { " · $it" } ?: "") +
                (if (input.json.optBoolean("resampled")) " (resampled to 16 kHz)" else "")
            if (seconds < Engine.MIN_SECONDS) {
                finish(context, req, "too_short", "TOO SHORT", "The audio is ${"%.1f".format(Locale.US, seconds)} s long; at least ${Engine.MIN_SECONDS.toInt()} s is needed.", input = input.json)
                return
            }
            if (rms < Session.minRms) {
                finish(context, req, "no_speech", "NO SPEECH", "No speech detected: the audio is at ${dbfs(rms)}, below the ${dbfs(Session.minRms)} floor. Speak closer to the microphone, or open another file.", input = input.json)
                return
            }
            if (!loadModel(context)) {
                val e = lastLoadError
                finish(context, req, e?.code ?: "load_failed", Session.pill, Session.status, input = input.json, keepScreen = true)
                return
            }
            val loaded = Engine.loaded!!
            val readers = Engine.ensureReader(context)
            val windowFrames = if (Session.windowFrames in loaded.encoder.windowSizes) Session.windowFrames else loaded.encoder.windowSizes.max()
            val overlap = Session.overlapTokens.coerceAtMost(WindowPlan.maxOverlap(windowFrames))
            Session.clearTranscript()
            Session.transcribing = true
            Session.pill = "TRANSCRIBING"
            Session.tone = Session.Tone.WORK
            Session.status = "Transcribing on ${loaded.encoder.backend.label}…"
            Session.progress = 0.0
            Session.detailLine = ""
            Session.changed()
            val stem = req.report?.removeSuffix(".json")
            val dumpSoft = if (req.dump && stem != null) File(Engine.filesDir(context), "$stem.soft.f32") else null
            val dumpTokens = if (req.dump && stem != null) File(Engine.filesDir(context), "$stem.tokens.bin") else null
            val softOut = dumpSoft?.let { it.parentFile?.mkdirs(); BufferedOutputStream(FileOutputStream(it), 1 shl 20) }
            val tokOut = dumpTokens?.let { BufferedOutputStream(FileOutputStream(it), 1 shl 16) }
            val result = try {
                Engine.transcribe(src, loaded, readers, windowFrames, overlap, object : Pipeline.Listener {
                    override fun onWindow(k: Int, count: Int, audioSeconds: Double, elapsedMs: Double) {
                        val xrt = audioSeconds / (elapsedMs / 1000.0)
                        Session.progress = (k + 1).toDouble() / count
                        Session.progressLine = "Window ${k + 1} of $count · ${secs(elapsedMs)} · ${"%.1f".format(Locale.US, xrt)}× real time"
                        Log.i(TAG, "WINDOW ${k + 1}/$count audio_s=${fmt(audioSeconds)} elapsed_ms=${fmt(elapsedMs)}")
                        Session.changed()
                    }

                    override fun onWords(words: List<Word>) {
                        Session.addWords(words)
                        Session.changed()
                    }
                }, if (softOut != null || tokOut != null) Pipeline.Dump(softOut, tokOut) else null)
            } catch (e: Throwable) {
                Session.transcribing = false
                Log.e(TAG, "ERROR run ${e.message}", e)
                finish(context, req, "run_failed", "FAILED", "The encoder failed on ${loaded.encoder.backend.label}: ${e.javaClass.simpleName}: ${e.message}", input = input.json, loaded = loaded)
                return
            } finally {
                runCatching { softOut?.close() }
                runCatching { tokOut?.close() }
            }
            Session.transcribing = false
            val r = result
            val xrt = seconds / (r.elapsedMs / 1000.0)
            val decodeMs = input.json.optDouble("decode_ms", 0.0)
            val enc = r.encoderMs.sorted()
            val median = if (enc.isEmpty()) 0.0 else enc[enc.size / 2]
            Session.progressLine = "Window ${r.encoderMs.size} of ${r.plan.count} · ${secs(r.elapsedMs)} · ${"%.1f".format(Locale.US, xrt)}× real time"
            Session.detailLine = "${r.plan.count} windows of ${windowFrames / 100} s (overlap $overlap tokens) · encoder median ${fmt(median)} ms a window · " +
                "log-mel ${secs(r.melMs)} · reader + words ${secs(r.decodeMs)} · ${r.plan.tokens} tokens · ${r.words.size} words" +
                (if (decodeMs > 0) " · file read ${secs(decodeMs)}" else "")
            val text = r.words.joinToString(" ") { it.text }
            val status = if (r.cancelled) "cancelled" else "ok"
            val pill = if (r.cancelled) "STOPPED" else "DONE"
            val line = if (r.cancelled) "Stopped after ${r.encoderMs.size} of ${r.plan.count} windows." else
                "${Session.clock(seconds)} of audio in ${secs(r.elapsedMs)} (${"%.1f".format(Locale.US, xrt)}× real time) on ${loaded.encoder.backend.label}."
            val report = jsonOf(
                "window" to jsonOf(
                    "frames" to windowFrames, "seconds" to windowFrames / 100, "overlap_tokens" to overlap, "overlap_s" to overlap * 0.04,
                    "stride_frames" to r.plan.stride, "count" to r.plan.count, "run" to r.encoderMs.size, "compute_factor" to r.plan.computeFactor,
                    "frames_per_call" to jsonArray(r.windowFrames),
                ),
                "timing" to jsonOf(
                    "transcribe_ms" to r.elapsedMs, "xrt" to xrt, "xrt_with_file_read" to seconds / ((r.elapsedMs + decodeMs) / 1000.0),
                    "encoder_ms" to jsonArray(r.encoderMs), "encoder_ms_median" to median, "mel_ms" to r.melMs, "reader_words_ms" to r.decodeMs,
                    "file_read_ms" to decodeMs,
                ),
                "tokens" to r.plan.tokens, "blanks" to r.blanks, "words" to r.words.size, "transcript" to text,
                "warnings" to jsonArray(r.warnings),
                "dump" to if (dumpSoft != null) jsonOf("soft" to dumpSoft.name, "tokens" to dumpTokens?.name, "token_record" to "int32 global index, int32 row (-1 blank), float64 best cos, float64 second cos, float64 blank logit; little-endian") else null,
            )
            finish(context, req, status, pill, line, input = input.json, loaded = loaded, extra = report, tone = if (r.cancelled) Session.Tone.WARN else Session.Tone.OK)
        }
    }

    /** Sets the screen's end state, logs RESULT and writes the run report when one was asked for. */
    private fun finish(
        context: Context, req: Request, status: String, pill: String, text: String, input: JSONObject? = null,
        loaded: Engine.Loaded? = null, extra: JSONObject? = null, tone: Session.Tone = Session.Tone.LIVE, keepScreen: Boolean = false,
    ) {
        Session.pill = pill
        Session.tone = tone
        Session.status = text
        Session.statusOpensSettings = status == "mic_denied"
        if (!keepScreen) Session.changed()
        val l = loaded ?: Engine.loaded
        val readers = Engine.readers
        val o = jsonOf(
            "app" to "eg2transcribe", "version" to BuildConfig.VERSION_NAME, "litert" to BuildConfig.LITERT_VERSION,
            "status" to status, "text" to text, "time" to OwnLog.now(),
            "device" to Engine.deviceJson(context), "input" to input,
            "model" to modelChoice(context).let { c ->
                jsonOf(
                    "chosen" to Session.modelName, "name" to c.name, "file" to l?.encoder?.file?.name, "bytes" to l?.encoder?.file?.length(),
                    "files" to jsonArray(c.files.map { jsonOf("name" to it.name, "bytes" to it.length()) }),
                )
            },
            "backend" to jsonOf(
                "requested" to Session.backend.id, "used" to l?.encoder?.backend?.id, "fallback" to l?.fallback,
                "compile_ms" to l?.encoder?.compileMs, "warmup_ms" to l?.encoder?.warmupMs?.let { JSONObject(it as Map<*, *>) },
                "threads" to l?.encoder?.threads, "signatures" to l?.encoder?.signatureNames?.let { jsonArray(it) },
                "npu_libraries_packaged" to LiteRt.npuPackaged,
                "npu" to l?.npu?.let { n ->
                    jsonOf(
                        "hexagon" to Npu.hexagon(), "dispatch_delegate" to n.dispatch, "compile_ms" to n.compileMs, "error" to n.error,
                        "sha256" to n.prepared?.sha256, "sha256_ms" to n.prepared?.sha256Ms, "compiled_path" to n.prepared?.path?.absolutePath,
                        "cache_dir" to n.prepared?.cacheDir?.absolutePath, "cached_before" to n.prepared?.cachedBefore, "cache_bytes_after" to n.cacheBytesAfter,
                        "cache_dirs_with_graph" to n.prepared?.let { p -> jsonArray(p.dirsWithGraph().map { it.absolutePath }) },
                        "link_note" to n.prepared?.note, "key_lines" to jsonArray(n.keyLines),
                    )
                },
            ),
            "delegate_lines" to l?.delegateLines?.let { jsonArray(it) },
            "reader" to readers?.let { jsonOf("name" to it.reader.name, "stub" to it.reader.isStub, "dim" to it.reader.dim, "words" to it.table.size, "note" to it.note) },
            "memory" to Engine.memoryJson(),
        )
        extra?.keys()?.forEach { k -> o.put(k, extra.get(k)) }
        Log.i(TAG, "RESULT status=$status backend=${l?.encoder?.backend?.id ?: "-"} " + (extra?.optJSONObject("timing")?.let { t ->
            "windows=${extra.optJSONObject("window")?.optInt("run")} tokens=${extra.optInt("tokens")} words=${extra.optInt("words")} transcribe_ms=${fmt(t.optDouble("transcribe_ms"))} xrt=${fmt(t.optDouble("xrt"))} "
        } ?: "") + "text=${text.take(160)}")
        if (req.report != null) {
            val f = File(Engine.filesDir(context), req.report)
            runCatching {
                f.parentFile?.mkdirs()
                f.writeText(o.toString(1))
                Log.i(TAG, "RUN_JSON path=${f.absolutePath}")
            }.onFailure { Log.e(TAG, "ERROR report ${it.message}", it) }
        }
        Log.i(TAG, "RUN_DONE status=$status")
        Session.changed()
    }

    fun fmt(x: Double) = String.format(Locale.US, "%.1f", x)
    fun secs(ms: Double) = String.format(Locale.US, if (ms < 10_000) "%.2f s" else "%.1f s", ms / 1000.0)
    fun mb(bytes: Long) = megabytes(bytes)
    fun dbfs(rms: Double) = if (rms <= 0.0) "-inf dBFS" else String.format(Locale.US, "%.0f dBFS", 20 * Math.log10(rms))
}
