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

    /** The window length a run with encoder [l] uses for windows of [frames] frames: that length when the encoder has it, else its longest. */
    fun windowFor(l: Engine.Loaded, frames: Int): Int = if (frames in l.encoder.windowSizes) frames else l.encoder.windowSizes.max()

    /**
     * A run's window length ([frames]) and why: [setting] = "auto", "30 s", "10 s" or "npu"; [reason] = the screen's and
     * the report's words for it; [tokens] = the recording's tokens it was picked for (null: picked before a recording).
     */
    class WindowPick(val frames: Int, val setting: String, val reason: String, val tokens: Int?) {
        fun json(): JSONObject = jsonOf(
            "setting" to setting, "picked_frames" to frames, "picked_seconds" to frames / 100, "reason" to reason,
            "tokens" to tokens, "auto_max_tokens_10s" to WindowPlan.AUTO_MAX_TOKENS_10S,
        )
    }

    /**
     * The window for a recording of [tokens] tokens (null = before a recording: what to load at start). "auto" takes
     * the 10 s files for a recording of at most [WindowPlan.AUTO_MAX_TOKENS_10S] tokens when the encoder's 10 s file
     * and `reader_w10.tflite` are both there, else the 30 s files ([WindowPlan.autoFrames]); before a recording it loads
     * the 10 s files when they are there (the microphone and the example are short). The NPU always runs 10 s windows.
     */
    fun pickWindow(context: Context, tokens: Int?): WindowPick {
        if (Session.backend == Backend.NPU) return WindowPick(WindowPlan.FRAMES_10S, "npu", "the NPU runs 10 s windows", tokens)
        val setting = Session.windowFrames
        if (setting != WindowPlan.AUTO) return WindowPick(setting, "${setting / 100} s", "the window setting", tokens)
        val have = modelChoice(context).windows
        val has10 = WindowPlan.FRAMES_10S in have && Engine.hasReaderFile(context, WindowPlan.FRAMES_10S)
        val has30 = WindowPlan.FRAMES_30S in have
        val frames = WindowPlan.autoFrames(tokens ?: 0, has10, has30)
        val max = WindowPlan.AUTO_MAX_TOKENS_10S
        val reason = when {
            tokens == null -> if (frames == WindowPlan.FRAMES_10S) "the 10 s files, for a short recording" else "no 10 s files: the 30 s files"
            frames == WindowPlan.FRAMES_10S && tokens <= max -> "$tokens tokens fit one 10 s window (at most $max)"
            frames == WindowPlan.FRAMES_10S -> "$tokens tokens, and the model has no 30 s file"
            tokens <= max -> "$tokens tokens would fit one 10 s window, but ${missing10(context)} is not on this phone"
            else -> "$tokens tokens are more than one 10 s window holds ($max)"
        }
        return WindowPick(frames, "auto", reason, tokens)
    }

    /** Which 10 s file "auto" misses: the encoder's or the reader's. */
    private fun missing10(context: Context): String =
        if (WindowPlan.FRAMES_10S !in modelChoice(context).windows) "${ModelChoice.nameOf(Session.modelName)}_w10.tflite" else "reader/reader_w10.tflite"

    /** Loads the encoder and then the reader for windows of [frames] frames (work thread). True when both are ready. */
    fun loadModel(context: Context, frames: Int = pickWindow(context, null).frames): Boolean {
        val choice = modelChoice(context)
        Session.modelLine = choice.describe()
        val have = Engine.isLoaded(choice, Session.backend, frames)
        if (have != null && Engine.isReaderLoaded(context, have.encoder.backend, windowFor(have, frames), Session.readerPrecision)) return true
        Session.loading = true
        Session.loadingSince = System.nanoTime()
        Session.pill = "LOADING MODEL"
        Session.tone = Session.Tone.WORK
        Session.status = if (have == null) "Compiling the encoder for ${Session.backend.label}…" else "Loading the reader…"
        if (have == null) Session.fallbackLine = null
        Session.changed()
        // A reader for another window length or accelerator goes before the next encoder compiles: the old reader and
        // the new encoder do not have to fit in memory together (two models of each length would not fit an 8 GB phone).
        Engine.readers?.let { r -> if (r.windowFrames != frames || r.requested != Engine.readerBackend(Session.backend) || r.precision != Session.readerPrecision) Engine.closeReader() }
        try {
            val l = have ?: Engine.ensureEncoder(context, choice, Session.backend, frames) { stage ->
                Session.status = stage
                Session.changed()
            }.also { l ->
                val e = l.encoder
                val npu = l.npu
                Session.engineLine = "Encoder on ${e.backend.label}" + (if (e.backend == Backend.CPU) " (${e.threads} threads)" else "") +
                    " · " + (if (e.backend == Backend.NPU && npu?.prepared?.cachedBefore == true) "loaded from its cache in ${secs(e.compileMs)}"
                    else "compiled in ${secs(e.compileMs)}" + (if (e.backend == Backend.NPU) " (first time; later starts read the cache)" else "")) +
                    " · ${e.file.name} (${e.signatureNames.joinToString(", ")})"
                Session.fallbackLine = l.fallback
                Log.i(TAG, "ENGINE_READY requested=${l.requested.id} used=${e.backend.id} compile_ms=${fmt(e.compileMs)} warmup_ms=${e.warmupMs} model=${e.file.name} bytes=${e.file.length()} fallback=${l.fallback != null}" +
                    (npu?.let { " npu_dispatch=${it.dispatch} npu_cached_before=${it.prepared?.cachedBefore} npu_cache_bytes=${it.cacheBytesAfter} npu_error=${it.error}" } ?: ""))
            }
            val rs = Engine.ensureReader(context, l.encoder.backend, windowFor(l, frames), Session.readerPrecision) { stage ->
                Session.status = stage
                Session.changed()
            }
            Session.readerLine = rs.note + (rs.tflite?.let { " · compiled in ${secs(it.compileMs)}" } ?: "")
            Session.readerIsStub = rs.reader.isStub
            if (rs.fallback != null) Session.fallbackLine = listOfNotNull(l.fallback, rs.fallback).joinToString(" ")
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
            "README.md describes them). On the CPU and the GPU a recording of up to 9.96 s runs on the 10 s file and a longer one on the 30 s file; the NPU runs the 10 s file.\n\n" +
            "Either push them with adb, then come back to this screen:\n" +
            "adb push ${base}_w30.tflite $dir/\nadb push ${base}_w10.tflite $dir/\n\n" +
            "or tap Change next to Model and choose a file on this phone (it is copied into the app; copy both files of a pair)."
    }

    /** /storage/emulated/0/... as adb's /sdcard/... */
    fun sdcard(f: File): String = f.absolutePath.replace(Regex("^/storage/emulated/0"), "/sdcard")

    /** Audio file extensions the batch mode reads from a directory. */
    private val AUDIO = setOf("flac", "wav", "mp3", "m4a", "ogg", "opus", "aac")

    /**
     * Decodes [uri] to cache/input.pcm: the input, or the outcome that ends the job (cancelled, unreadable). [capMs]: stop
     * after that much time (the decode probe; the input then holds what was decoded and `decode.capped`).
     */
    private fun decodeFile(context: Context, uri: Uri, label: String, cancel: AtomicBoolean, mode: AudioDecoder.Mode = Session.decodeMode, capMs: Long? = null): Pair<Session.Input?, Outcome?> {
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
            AudioDecoder.decode(context, uri, pcm, { cancel.get() }, mode, capMs) { f ->
                Session.progress = f
                Session.changed()
            }
        } catch (e: Throwable) {
            Session.decoding = false
            val text = if (cancel.get()) "Stopped." else "This file could not be read: ${e.message ?: e.javaClass.simpleName}"
            return null to Outcome(if (cancel.get()) "cancelled" else "decode_failed", if (cancel.get()) "STOPPED" else "CANNOT READ", text, jsonOf("source" to "file", "label" to label))
        }
        Session.decoding = false
        val json = jsonOf(
            "source" to "file", "label" to label, "samples" to decoded.samples, "seconds" to decoded.samples / 16000.0,
            "format" to decoded.format, "source_rate" to decoded.sourceRate, "channels" to decoded.channels,
            "resampled" to decoded.resampled, "decode_ms" to decoded.ms,
            "expected_samples" to decoded.expectedSamples, "missing_samples" to decoded.expectedSamples?.let { it - decoded.samples },
            "decode" to decoded.stats,
        )
        Log.i(TAG, "INPUT source=file label=$label samples=${decoded.samples} expected=${decoded.expectedSamples} format=${decoded.format} rate=${decoded.sourceRate} channels=${decoded.channels} resampled=${decoded.resampled} decode_ms=${fmt(decoded.ms)} decode_mode=${decoded.stats.optString("mode")} capped=${decoded.stats.optBoolean("capped")}")
        return Session.Input(pcm, label, json) to null
    }

    /**
     * The decode probe (launch extra `decode_only`): reads [uri] the way a transcription would and reports how long it
     * took and where the time went (`input.decode`), without loading or running a model.
     */
    fun decodeOnly(context: Context, uri: Uri, label: String, req: Request, cancel: AtomicBoolean, mode: AudioDecoder.Mode, capMs: Long?) {
        val (input, failed) = decodeFile(context, uri, label, cancel, mode, capMs)
        if (input == null) {
            val o = failed!!
            finish(context, req, o.status, o.pill, o.text, input = o.input)
            return
        }
        val j = input.json
        // The decoded samples' SHA-256 (16-bit little-endian, as ffmpeg's s16le output), to compare a whole read bit for bit.
        runCatching { sha256(input.pcm) }.onSuccess { (hex, hashMs) -> j.put("pcm_sha256", hex); j.put("pcm_sha256_ms", hashMs) }
        val samples = j.optLong("samples")
        val ms = j.optDouble("decode_ms")
        val d = j.optJSONObject("decode")
        val capped = d?.optBoolean("capped") == true
        val audioS = if (capped) d?.optDouble("source_position_s", samples / 16000.0) ?: (samples / 16000.0) else samples / 16000.0
        val text = (if (capped) "Stopped after ${secs(ms)}: " else "") +
            "read ${Session.clock(samples / 16000.0)} of audio (${"%,d".format(Locale.US, samples)} samples) in ${secs(ms)} " +
            "(${"%.1f".format(Locale.US, audioS / (ms / 1000.0))}× real time), ${mode.label}."
        Session.inputLine = "$label · ${Session.clock(samples / 16000.0)} · ${j.optString("format")}"
        Session.detailLine = d?.let { decodeLine(it) } ?: ""
        finish(context, req, if (capped) "capped" else "ok", "READ", text, input = j, tone = Session.Tone.OK)
    }

    /** SHA-256 of a file (hex) and the ms it took. */
    private fun sha256(f: File): Pair<String, Double> {
        val t = System.nanoTime()
        val md = java.security.MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s ->
            val b = ByteArray(1 shl 20)
            while (true) {
                val n = s.read(b)
                if (n < 0) break
                md.update(b, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) } to (System.nanoTime() - t) / 1e6
    }

    /** One line of where a decode's time went, from [AudioDecoder.Stats.json]. */
    private fun decodeLine(d: JSONObject): String = listOf("flac_decode_ms" to "FLAC frames", "read_ms" to "extractor reads", "queue_in_ms" to "queueing input",
        "dequeue_in_ms" to "input slots", "dequeue_out_ms" to "output polls", "wait_ms" to "waiting", "convert_ms" to "samples out", "release_ms" to "releasing")
        .filter { d.has(it.first) }.joinToString(" · ") { (k, name) -> "$name ${secs(d.optDouble(k))}" } +
        " · ${d.optInt("inputs")} buffers in, ${d.optInt("outputs")} out"

    /** Decodes an audio file, then transcribes it. */
    fun transcribeUri(context: Context, uri: Uri, label: String, req: Request, cancel: AtomicBoolean) {
        val (input, failed) = decodeFile(context, uri, label, cancel)
        if (input == null) {
            val o = failed!!
            finish(context, req, o.status, o.pill, o.text, input = o.input)
            return
        }
        run(context, input, req)
    }

    /**
     * Every audio file of [dir], with the same model: one report with an item per file, by name (its status,
     * transcript, timing; per-file dumps `<report stem>.<file stem>.tokens.bin` / `.soft.f32` with [Request.dump]).
     * The device check's fixture mode (the 20 test-clean utterances in one run). With "auto" a file may need the other
     * window length's files: the files that fit the loaded length run first, in name order, and the others after one
     * switch (their audio is read again then), so a batch compiles each length at most once.
     */
    fun transcribeDir(context: Context, dir: File, req: Request, cancel: AtomicBoolean) {
        val files = dir.listFiles { f -> f.isFile && f.extension.lowercase() in AUDIO }?.sortedBy { it.name } ?: emptyList()
        val dirJson = jsonOf("source" to "dir", "label" to dir.name, "files" to files.size)
        if (files.isEmpty()) {
            finish(context, req, "no_files", "NO FILES", "No audio files in ${dir.name}.", input = dirJson)
            return
        }
        val t0 = System.nanoTime()
        val items = arrayOfNulls<JSONObject>(files.size)
        val order = ArrayList<String>()
        var ok = 0
        var last: Outcome? = null
        var stopped = false
        fun one(i: Int, input: Session.Input?, failed: Outcome?) {
            val f = files[i]
            val o = failed ?: transcribeOnce(context, input!!, req.dump, req.report?.removeSuffix(".json")?.let { "$it.${f.nameWithoutExtension}" })
            last = o
            val x = o.extra
            items[i] = jsonOf(
                "file" to f.name, "status" to o.status, "text" to o.text, "input" to (input?.json ?: o.input),
                "transcript" to x?.optString("transcript"), "words" to x?.optInt("words"), "tokens" to x?.optInt("tokens"),
                "window" to x?.optJSONObject("window"), "timing" to x?.optJSONObject("timing"), "dump" to x?.opt("dump"),
                "encoder" to x?.optJSONObject("encoder_run"), "reader" to x?.optJSONObject("reader_run"),
            )
            order += f.name
            if (o.status == "ok") ok++
            if (o.status in STOPPING) stopped = true
        }
        val later = ArrayList<Int>()
        for ((i, f) in files.withIndex()) {
            if (stopped || cancel.get() || Engine.cancel.get()) break
            Log.i(TAG, "DIR_FILE ${i + 1}/${files.size} ${f.name}")
            val (input, failed) = decodeFile(context, Uri.fromFile(f), f.name, cancel)
            val loadedFor = Engine.readers?.windowFrames
            if (input != null && loadedFor != null) {
                val pick = pickWindow(context, WindowPlan.tokensFor(MelFrontEnd.frames(input.json.optLong("samples"))))
                if (pick.frames != loadedFor) {
                    Log.i(TAG, "DIR_LATER ${f.name}: ${pick.frames / 100} s windows (${pick.reason}), after the files for the loaded ${loadedFor / 100} s")
                    later += i
                    continue
                }
            }
            one(i, input, failed)
        }
        for (i in later) {
            if (stopped || cancel.get() || Engine.cancel.get()) break
            val f = files[i]
            Log.i(TAG, "DIR_FILE ${i + 1}/${files.size} ${f.name} (second pass)")
            val (input, failed) = decodeFile(context, Uri.fromFile(f), f.name, cancel)
            one(i, input, failed)
        }
        val ms = (System.nanoTime() - t0) / 1e6
        val status = if (ok == files.size) "ok" else if (ok > 0) "partial" else last?.status ?: "failed"
        val text = "$ok of ${files.size} files transcribed in ${secs(ms)}."
        finish(
            context, req, status, if (ok == files.size) "DONE" else "PARTLY DONE", text, input = dirJson, loaded = Engine.loaded,
            extra = jsonOf(
                "items" to jsonArray(items.filterNotNull()), "run_order" to jsonArray(order),
                "dir_timing" to jsonOf("total_ms" to ms, "files" to files.size, "ok" to ok, "second_pass" to later.size),
            ),
            tone = if (ok == files.size) Session.Tone.OK else Session.Tone.WARN,
        )
    }

    /** Outcomes after which the next file of a batch would fail the same way. */
    private val STOPPING = setOf("load_failed", "model_missing", "compile_failed", "run_failed", "not_finite", "not_an_encoder", "cancelled")

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

    /** How a transcription ended: what [finish] shows and reports. */
    private class Outcome(
        val status: String, val pill: String, val text: String, val input: JSONObject? = null, val extra: JSONObject? = null,
        val tone: Session.Tone = Session.Tone.LIVE, val keepScreen: Boolean = false, val loaded: Engine.Loaded? = null,
    )

    /** Checks length and loudness, loads the model, runs the pipeline and reports. */
    private fun run(context: Context, input: Session.Input, req: Request) {
        val o = transcribeOnce(context, input, req.dump, req.report?.removeSuffix(".json"))
        finish(context, req, o.status, o.pill, o.text, input = input.json, loaded = o.loaded, extra = o.extra, tone = o.tone, keepScreen = o.keepScreen)
    }

    /** One transcription of [input]; [dumpStem] = the per-token files' name without extension (with [dump]). */
    private fun transcribeOnce(context: Context, input: Session.Input, dump: Boolean, dumpStem: String?): Outcome {
        Session.lastInput = input
        Session.progress = 0.0
        Session.progressLine = ""
        Session.detailLine = ""
        val pcm = try { Pcm16File(input.pcm) } catch (e: Throwable) {
            return Outcome("decode_failed", "CANNOT READ", "The audio could not be opened: ${e.message}", input.json)
        }
        pcm.use { src ->
            val seconds = src.seconds
            val rms = Loudness.rms(src)
            input.json.put("rms", rms)
            Session.inputLine = "${input.label} · ${Session.clock(seconds)}" + (input.json.optString("format").takeIf { it.isNotEmpty() }?.let { " · $it" } ?: "") +
                (if (input.json.optBoolean("resampled")) " (resampled to 16 kHz)" else "")
            if (seconds < Engine.MIN_SECONDS) {
                return Outcome("too_short", "TOO SHORT", "The audio is ${"%.1f".format(Locale.US, seconds)} s long; at least ${Engine.MIN_SECONDS.toInt()} s is needed.", input.json)
            }
            if (rms < Session.minRms) {
                return Outcome("no_speech", "NO SPEECH", "No speech detected: the audio is at ${dbfs(rms)}, below the ${dbfs(Session.minRms)} floor. Speak closer to the microphone, or open another file.", input.json)
            }
            val pick = pickWindow(context, WindowPlan.tokensFor(MelFrontEnd.frames(src.length)))
            if (!loadModel(context, pick.frames)) {
                val e = lastLoadError
                return Outcome(e?.code ?: "load_failed", Session.pill, Session.status, input.json, keepScreen = true)
            }
            val loaded = Engine.loaded!!
            val readers = Engine.readers!!
            val windowFrames = windowFor(loaded, pick.frames)
            val overlap = Session.overlapTokens.coerceAtMost(WindowPlan.maxOverlap(windowFrames))
            Session.lastWindowNote = "${windowFrames / 100} s"
            Log.i(TAG, "WINDOW_PICK setting=${pick.setting} frames=${pick.frames} used=$windowFrames tokens=${pick.tokens} reason=${pick.reason}")
            Session.clearTranscript()
            Session.transcribing = true
            Session.pill = "TRANSCRIBING"
            Session.tone = Session.Tone.WORK
            Session.status = "Transcribing on ${loaded.encoder.backend.label}…"
            Session.progress = 0.0
            Session.detailLine = ""
            Session.changed()
            val dumpSoft = if (dump && dumpStem != null) File(Engine.filesDir(context), "$dumpStem.soft.f32") else null
            val dumpTokens = if (dump && dumpStem != null) File(Engine.filesDir(context), "$dumpStem.tokens.bin") else null
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
                return Outcome("run_failed", "FAILED", "The encoder or the reader failed on ${loaded.encoder.backend.label}: ${e.javaClass.simpleName}: ${e.message}", input.json, loaded = loaded)
            } finally {
                runCatching { softOut?.close() }
                runCatching { tokOut?.close() }
            }
            Session.transcribing = false
            val r = result
            val xrt = seconds / (r.elapsedMs / 1000.0)
            val decodeMs = input.json.optDouble("decode_ms", 0.0)
            val median = median(r.encoderMs)
            val readerMedian = median(r.readerMs)
            val callMedians = r.calls.all().mapValues { median(it.value) }
            fun split(p: String) = "write ${fmt(callMedians.getValue("${p}_write_ms"))}, run ${fmt(callMedians.getValue("${p}_run_ms"))}, read ${fmt(callMedians.getValue("${p}_read_ms"))}"
            Session.progressLine = "Window ${r.encoderMs.size} of ${r.plan.count} · ${secs(r.elapsedMs)} · ${"%.1f".format(Locale.US, xrt)}× real time"
            Session.detailLine = (if (r.plan.count == 1) "1 window of ${windowFrames / 100} s" else "${r.plan.count} windows of ${windowFrames / 100} s (overlap $overlap tokens)") +
                (if (pick.setting == "auto") " (auto: ${pick.reason})" else "") +
                " · encoder median ${fmt(median)} ms (${split("encoder")}), reader median ${fmt(readerMedian)} ms (${split("reader")}) a window · " +
                "log-mel ${secs(r.melMs)} · labels ${secs(r.decodeMs)} · ${r.plan.tokens} tokens · ${r.words.size} words" +
                (if (decodeMs > 0) " · file read ${secs(decodeMs)}" else "")
            val text = r.words.joinToString(" ") { it.text }
            val status = if (r.cancelled) "cancelled" else "ok"
            val pill = if (r.cancelled) "STOPPED" else "DONE"
            val line = if (r.cancelled) "Stopped after ${r.encoderMs.size} of ${r.plan.count} windows." else
                "${Session.clock(seconds)} of audio in ${secs(r.elapsedMs)} (${"%.1f".format(Locale.US, xrt)}× real time) on ${loaded.encoder.backend.label}."
            val tr = readers.tflite
            val report = jsonOf(
                "window" to jsonOf(
                    "frames" to windowFrames, "seconds" to windowFrames / 100, "overlap_tokens" to overlap, "overlap_s" to overlap * 0.04,
                    "drop_last_tokens" to r.plan.dropLast, "step_tokens" to r.plan.step, "stride_frames" to r.plan.stride,
                    "count" to r.plan.count, "run" to r.encoderMs.size, "compute_factor" to r.plan.computeFactor,
                    "frames_per_call" to jsonArray(r.windowFrames), "pick" to pick.json(),
                ),
                "timing" to jsonOf(
                    "transcribe_ms" to r.elapsedMs, "xrt" to xrt, "xrt_with_file_read" to seconds / ((r.elapsedMs + decodeMs) / 1000.0),
                    "encoder_ms" to jsonArray(r.encoderMs), "encoder_ms_median" to median,
                    "reader_ms" to jsonArray(r.readerMs), "reader_ms_median" to readerMedian,
                    "mel_ms" to r.melMs, "labels_ms" to r.decodeMs, "file_read_ms" to decodeMs,
                    // Per window: each call's input write / run / output read (encoder_ms = the three; reader_ms = the
                    // reader's three on the LiteRT thread).
                    "calls" to JSONObject().apply { for ((k, v) in r.calls.all()) put(k, jsonArray(v)) },
                    "call_medians" to JSONObject().apply { for ((k, v) in callMedians) put(k, v) },
                ),
                "encoder_run" to jsonOf("file" to loaded.encoder.file.name, "used" to loaded.encoder.backend.id, "compile_ms" to loaded.encoder.compileMs,
                    "signatures" to jsonArray(loaded.encoder.signatureNames)),
                "reader_run" to jsonOf("file" to (tr?.file?.name ?: readers.reader.name), "stub" to readers.reader.isStub, "used" to tr?.backend?.id,
                    "precision" to tr?.precision, "tokens" to tr?.tokens, "compile_ms" to tr?.compileMs, "fallback" to readers.fallback),
                "tokens" to r.plan.tokens, "blanks" to r.blanks, "words" to r.words.size, "transcript" to text,
                "warnings" to jsonArray(r.warnings),
                "dump" to if (dumpSoft != null) jsonOf("soft" to dumpSoft.name, "tokens" to dumpTokens?.name, "token_record" to Pipeline.TOKEN_RECORD) else null,
            )
            return Outcome(status, pill, line, input.json, report, if (r.cancelled) Session.Tone.WARN else Session.Tone.OK, loaded = loaded)
        }
    }

    private fun median(xs: List<Double>): Double {
        if (xs.isEmpty()) return 0.0
        val s = xs.sorted()
        return s[s.size / 2]
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
            "reader" to readers?.let { rs ->
                val t = rs.tflite
                jsonOf(
                    "name" to rs.reader.name, "stub" to rs.reader.isStub, "words" to rs.vocab.size, "note" to rs.note,
                    "file" to t?.file?.name, "bytes" to t?.file?.length(), "requested" to rs.requested?.id, "used" to t?.backend?.id,
                    "fallback" to rs.fallback, "tokens" to t?.tokens, "slices" to t?.slices, "compile_ms" to t?.compileMs,
                    "warmup_ms" to t?.warmupMs, "threads" to t?.threads, "window_frames" to rs.windowFrames, "precision" to t?.precision,
                    "files" to jsonArray(rs.key.map { (n, b) -> jsonOf("name" to n, "bytes" to b) }),
                )
            },
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
