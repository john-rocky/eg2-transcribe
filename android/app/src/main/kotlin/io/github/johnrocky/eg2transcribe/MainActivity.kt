package io.github.johnrocky.eg2transcribe

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.StatFs
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.provider.Settings
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import org.json.JSONObject

/**
 * Speech to text on the phone: record with the microphone or open an audio file, and the words appear with their
 * time. The audio encoder of EmbeddingGemma 2 runs on LiteRT (CPU, GPU or the Qualcomm NPU); a reader turns each
 * 40 ms token into a vector and the nearest word of a table is read out. Until a trained reader is installed the app
 * uses a stub whose words are placeholders (the screen says so). Nothing leaves the phone: the app has no INTERNET
 * permission.
 *
 * Files (the app's external files dir, `files/` below):
 *   files/eg2_audio_f16safe_fp16w_w30.tflite, _w10.tflite
 *                                 the encoder, one file per window length (README.md: how to make them, how to push
 *                                 them); other .tflite files there can be picked under Model
 *   files/reader/                 a trained reader: reader.tflite, word_table.npy or .bin, vocab.txt (README.md)
 *   files/<report>                with --es report: the run report (JSON)
 * and in the app's internal cache: npu_jit/ (the NPU's compiled graphs, about 0.6 GB per model file) and npu_models/
 * (links to the model files named with their SHA-256).
 *
 * Launch extras (the device check, README.md "How the device check works"): --es file <path or name under files/>
 * transcribes that file, --ei mic_ms <ms> records that long from the microphone and transcribes it; with either,
 * --es report <name.json> writes the run report and --ez dump true also the per-token files. Settings: --es backend
 * cpu|gpu_fp32|gpu_default|npu, --es model <file> (either file of a _w30 / _w10 pair), --ei window_s 0|30|10 (0 = auto,
 * the default; the NPU always runs 10 s windows), --ei overlap <tokens>, --ef min_rms <rms>. Measurements only, not on
 * the screen: --es reader_precision fp32|fp16acc32 (the reader on a GPU), --es decode_mode flac|sync|async|legacy (flac,
 * the default: FLAC files by the app's own reader, MediaCodec for the rest and as its fallback; the others send every
 * compressed file through MediaCodec), and --ez decode_only true with --es file: read the file, report where the time
 * went and stop (no model; --ei decode_cap_s <s> stops the read after that long).
 * A start with file or mic_ms shows over the lock screen and keeps the display on (a phone asleep behind a lock keeps
 * the app on its slow cores).
 *
 * Log tag eg2transcribe: STEP, NPU_CACHE, ENGINE_READY, DELEGATE, FALLBACK, READER, INPUT, WINDOW, RESULT, RUN_JSON,
 * RUN_DONE, BUSY, MIC_PERMISSION, ROTATION, ERROR.
 */
class MainActivity : Activity() {
    private val main = Handler(Looper.getMainLooper())
    private var renderPending = false
    private var renderedChars = 0
    private var shownNote: String? = null
    private val myListener: () -> Unit = { scheduleRender() }

    private lateinit var pill: TextView
    private lateinit var status: TextView
    private lateinit var modelValue: TextView
    private lateinit var readerValue: TextView
    private lateinit var windowValue: TextView
    private lateinit var engineLine: TextView
    private lateinit var fallbackLine: TextView
    private lateinit var modelChange: Button
    private lateinit var windowChange: Button
    private val backendButtons = LinkedHashMap<Backend, Button>()
    private lateinit var recordButton: Button
    private lateinit var openButton: Button
    private lateinit var againButton: Button
    private lateinit var stopButton: Button
    private lateinit var exampleButton: Button
    private lateinit var inputLine: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var progressLine: TextView
    private lateinit var scroll: ScrollView
    private lateinit var transcript: TextView
    private lateinit var detailLine: TextView

    companion object {
        private const val TAG = Engine.TAG
        private const val REQ_MIC = 1
        private const val REQ_AUDIO = 2
        private const val REQ_MODEL = 3

        /** Jobs submitted and not finished, and among them the transcriptions (process-wide: a recreated activity sees the same counts). */
        private val pending = AtomicInteger(0)
        private val pendingRuns = AtomicInteger(0)
        private val stopMic = AtomicBoolean(false)
        private val stopDecode = AtomicBoolean(false)

        private const val C_TEXT = 0xFF202124.toInt()
        private const val C_SUB = 0xFF5F6368.toInt()
        private const val C_ACCENT = 0xFF1A73E8.toInt()
        private const val C_WARN = 0xFFB06000.toInt()
        private const val C_ERR = 0xFFC5221F.toInt()
        private const val C_OK = 0xFF188038.toInt()
        private const val C_IDLE = 0xFF5F6368.toInt()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        File(Engine.filesDir(this), "audio").mkdirs()
        setContentView(buildUi())
        applySettings(intent)
        if (Session.recordMode) recordMode() // a window made again in record mode
        Session.listener = myListener
        if (Session.pill == "STARTING") {
            Session.pill = "LOADING"
            Session.status = "Starting…"
        }
        render()
        // A decode-only start (the decode measurement) leaves the model alone.
        val decodeOnly = savedInstanceState == null && intent.getBooleanExtra("decode_only", false)
        if (pending.get() == 0 && !Session.busy && Engine.loaded == null && !decodeOnly) submit { Jobs.loadModel(applicationContext) }
        // A recreated activity (savedInstanceState set) gets its first intent again: that start already ran.
        if (savedInstanceState == null) runFromIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        applySettings(intent)
        runFromIntent(intent)
    }

    override fun onResume() {
        super.onResume()
        // Back from pushing the model with adb: load it if it is there now.
        if (Session.pill == "MODEL MISSING" && pending.get() == 0 && !Jobs.modelChoice(this).missing) submit { Jobs.loadModel(applicationContext) }
    }

    override fun onDestroy() {
        if (Session.listener === myListener) Session.listener = null
        if (isFinishing) {
            // Leaving the app: stop what runs and free the model (1 to 3 GB of memory).
            Engine.cancel.set(true)
            stopMic.set(true)
            stopDecode.set(true)
            submit { Engine.closeReader(); Engine.closeEncoder() }
        }
        super.onDestroy()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        Log.i(TAG, "ROTATION orientation=${if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE) "landscape" else "portrait"} busy=${Session.busy}")
        render()
    }

    private fun submit(job: () -> Unit) = submit(false, job)

    /** [run]: a transcription (a second one is refused while it waits or runs). */
    private fun submit(run: Boolean, job: () -> Unit) {
        pending.incrementAndGet()
        if (run) pendingRuns.incrementAndGet()
        Engine.work.submit {
            try {
                job()
            } catch (t: Throwable) {
                Log.e(TAG, "ERROR job ${t.message}", t)
                Session.pill = "ERROR"
                Session.tone = Session.Tone.LIVE
                Session.status = "${t.javaClass.simpleName}: ${t.message}"
                Session.changed()
            } finally {
                if (run) pendingRuns.decrementAndGet()
                pending.decrementAndGet()
                Session.changed()
            }
        }
    }

    /** Anything running or queued (the buttons wait for it). */
    private val isBusy: Boolean get() = Session.busy || pending.get() > 0

    /** A transcription running or queued, or the microphone open (a second start is refused). */
    private val isRunning: Boolean get() = Session.recording || Session.decoding || Session.transcribing || pendingRuns.get() > 0

    // ---- launch extras -----------------------------------------------------------------------------------------

    private fun applySettings(i: Intent) {
        Backend.parse(i.getStringExtra("backend"))?.let { Session.backend = it }
        i.getStringExtra("model")?.let { Session.modelName = it }
        if (i.hasExtra("window_s")) Session.windowFrames = when (i.getIntExtra("window_s", 0)) {
            10 -> WindowPlan.FRAMES_10S
            30 -> WindowPlan.FRAMES_30S
            else -> WindowPlan.AUTO
        }
        // "auto" runs one 10 s window or 30 s windows: the overlap is bounded by the 30 s window.
        val longest = if (Session.windowFrames == WindowPlan.AUTO) WindowPlan.FRAMES_30S else Session.windowFrames
        if (i.hasExtra("overlap")) Session.overlapTokens = i.getIntExtra("overlap", WindowPlan.DEFAULT_OVERLAP).coerceIn(0, WindowPlan.maxOverlap(longest))
        if (i.hasExtra("min_rms")) Session.minRms = i.getFloatExtra("min_rms", Engine.MIN_RMS.toFloat()).toDouble()
        ReaderPrecision.parse(i.getStringExtra("reader_precision"))?.let { Session.readerPrecision = it }
        AudioDecoder.Mode.parse(i.getStringExtra("decode_mode"))?.let { Session.decodeMode = it }
        i.getStringExtra("picker_dir")?.let { Session.pickerDir = it }
        i.getStringExtra("picker_report")?.let { Session.pickerReport = it }
        // The recording harness without a job: show over the lock screen and keep the display on, as a check's start does.
        if (i.getBooleanExtra("keep_visible", false)) recordMode()
    }

    private fun runFromIntent(i: Intent) {
        val file = i.getStringExtra("file")
        val dir = i.getStringExtra("dir")
        val micMs = if (i.hasExtra("mic_ms")) i.getIntExtra("mic_ms", 0).toLong() else null
        if (file == null && dir == null && micMs == null) return
        recordMode()
        val req = Jobs.Request(i.getStringExtra("report"), i.getBooleanExtra("dump", false))
        val decodeOnly = i.getBooleanExtra("decode_only", false)
        val capMs = if (i.hasExtra("decode_cap_s")) i.getIntExtra("decode_cap_s", 0) * 1000L else null
        // A start that is handled once: a later onNewIntent or a recreated activity does not run it again.
        i.removeExtra("file")
        i.removeExtra("dir")
        i.removeExtra("mic_ms")
        i.removeExtra("decode_only")
        if (isRunning) {
            Log.i(TAG, "BUSY ignored a start (report=${req.report}): one transcription at a time")
            req.report?.let { name ->
                runCatching {
                    val f = File(Engine.filesDir(this), name)
                    f.parentFile?.mkdirs()
                    f.writeText(JSONObject().put("status", "busy").put("text", "A transcription is running; this start was ignored.").put("time", OwnLog.now()).toString(1))
                }
            }
            return
        }
        if (dir != null) {
            // The device check's fixture mode: every audio file of a directory in turn, one report.
            val d = if (dir.startsWith("/")) File(dir) else File(Engine.filesDir(this), dir)
            stopDecode.set(false)
            submit(true) { Jobs.transcribeDir(applicationContext, d, req, stopDecode) }
        } else if (file != null) {
            val f = if (file.startsWith("/")) File(file) else File(Engine.filesDir(this), file)
            stopDecode.set(false)
            if (decodeOnly) submit(true) { Jobs.decodeOnly(applicationContext, Uri.fromFile(f), f.name, req, stopDecode, Session.decodeMode, capMs) }
            else submit(true) { Jobs.transcribeUri(applicationContext, Uri.fromFile(f), f.name, req, stopDecode) }
        } else {
            stopMic.set(false)
            submit(true) { Jobs.transcribeMic(applicationContext, micMs, req, stopMic) }
        }
    }

    private var windowRecordMode = false

    /** For the device check: show over the lock screen, light the display and keep it on (this window, and later ones). */
    private fun recordMode() {
        Session.recordMode = true
        if (windowRecordMode) return
        windowRecordMode = true
        if (Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    // ---- buttons ----------------------------------------------------------------------------------------------

    private fun onRecord() {
        if (Session.recording) {
            stopMic.set(true)
            return
        }
        if (!Mic.granted(this)) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC)
            return
        }
        stopMic.set(false)
        Session.recording = true // the button turns into Stop at once
        Session.statusOpensSettings = false
        render()
        submit(true) { Jobs.transcribeMic(applicationContext, null, Jobs.Request(null, false), stopMic) }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQ_MIC) return
        val granted = grantResults.isNotEmpty() && grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED
        Log.i(TAG, "MIC_PERMISSION granted=$granted")
        if (granted) {
            onRecord()
        } else {
            Session.pill = "MIC OFF"
            Session.tone = Session.Tone.LIVE
            Session.status = Mic.deniedText(this)
            Session.statusOpensSettings = true
            render()
        }
    }

    /** The bundled example (an 8 s LibriSpeech utterance in the APK), copied out once and transcribed like an opened file. */
    private fun onExample() {
        if (isBusy) return
        stopDecode.set(false)
        submit(true) {
            val f = Examples.file(applicationContext)
            Jobs.transcribeUri(applicationContext, Uri.fromFile(f), Examples.NAME, Jobs.Request(Session.pickerReport, false), stopDecode)
        }
    }

    private fun onOpenFile() {
        val i = Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("audio/*")
            .putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "application/ogg", "video/mp4"))
        // The recording harness can open the picker in one folder of the shared storage (launch extra picker_dir).
        Session.pickerDir?.let { d ->
            if (Build.VERSION.SDK_INT >= 26) i.putExtra(DocumentsContract.EXTRA_INITIAL_URI, DocumentsContract.buildDocumentUri("com.android.externalstorage.documents", "primary:$d"))
        }
        runCatching { startActivityForResult(i, REQ_AUDIO) }.onFailure { showStatus("No file picker on this phone: ${it.message}") }
    }

    private fun onChooseModelFile() {
        val i = Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
        runCatching { startActivityForResult(i, REQ_MODEL) }.onFailure { showStatus("No file picker on this phone: ${it.message}") }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (resultCode != RESULT_OK || uri == null) {
            if (requestCode == REQ_AUDIO) showStatus("No file picked.")
            return
        }
        val (name, size) = nameAndSize(uri)
        when (requestCode) {
            REQ_AUDIO -> {
                stopDecode.set(false)
                submit(true) { Jobs.transcribeUri(applicationContext, uri, name, Jobs.Request(Session.pickerReport, false), stopDecode) }
            }
            REQ_MODEL -> {
                if (!name.endsWith(".tflite")) {
                    showStatus("$name is not a .tflite file.")
                    return
                }
                submit { importModel(uri, name, size) }
            }
        }
    }

    private fun nameAndSize(uri: Uri): Pair<String, Long> {
        var name = uri.lastPathSegment ?: "audio"
        var size = -1L
        runCatching {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    c.getString(0)?.let { name = it }
                    if (!c.isNull(1)) size = c.getLong(1)
                }
            }
        }
        return name to size
    }

    /** Copies a picked model into the files dir (work thread), then loads it. */
    private fun importModel(uri: Uri, name: String, size: Long) {
        val dir = Engine.filesDir(applicationContext)
        val free = StatFs(dir.path).availableBytes
        if (size > 0 && free < size + (64L shl 20)) {
            Session.pill = "NO SPACE"
            Session.tone = Session.Tone.LIVE
            Session.status = "$name needs ${Jobs.mb(size)}; this phone has ${Jobs.mb(free)} free."
            Session.changed()
            return
        }
        Session.loading = true
        Session.pill = "COPYING MODEL"
        Session.tone = Session.Tone.WORK
        Session.changed()
        val tmp = File(dir, "$name.part")
        try {
            contentResolver.openInputStream(uri)!!.use { input ->
                tmp.outputStream().buffered(1 shl 20).use { out ->
                    val buf = ByteArray(1 shl 20)
                    var done = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        Session.status = "Copying $name: ${Jobs.mb(done)}" + (if (size > 0) " of ${Jobs.mb(size)}" else "")
                        if (size > 0) Session.progress = done.toDouble() / size
                        Session.changed()
                    }
                }
            }
            val dst = File(dir, name)
            if (Engine.loaded?.encoder?.file == dst) Engine.closeEncoder()
            dst.delete()
            check(tmp.renameTo(dst)) { "could not rename ${tmp.name}" }
            Session.modelName = name
        } catch (t: Throwable) {
            tmp.delete()
            Session.pill = "COPY FAILED"
            Session.tone = Session.Tone.LIVE
            Session.status = "The model could not be copied: ${t.message}"
            return
        } finally {
            Session.loading = false
            Session.changed()
        }
        Jobs.loadModel(applicationContext)
    }

    private fun onBackend(b: Backend) {
        if (isBusy || b == Session.backend && Engine.loaded?.requested == b) return
        Session.backend = b
        render()
        submit { Jobs.loadModel(applicationContext) }
    }

    private fun onModelChange() {
        val dir = Engine.filesDir(this)
        val names = ModelChoice.all(dir)
        val current = ModelChoice.nameOf(Session.modelName)
        val items = names.map { n -> ModelChoice.of(dir, n).describe() + if (ModelChoice.nameOf(n) == current) " (in use)" else "" } + "Choose a file on this phone…"
        AlertDialog.Builder(this).setTitle("Encoder file").setItems(items.toTypedArray()) { _, which ->
            if (which == names.size) onChooseModelFile() else {
                Session.modelName = names[which]
                render()
                submit { Jobs.loadModel(applicationContext) }
            }
        }.setNegativeButton("Cancel", null).show()
    }

    private class WindowChoice(val frames: Int, val overlap: Int, val label: String)

    private fun onWindowChange() {
        val have = Jobs.modelChoice(this).windows
        val all = listOf(
            WindowChoice(WindowPlan.AUTO, 96, "Auto: the 10 s files for a recording of up to 9.96 s, 30 s windows for a longer one — default"),
            WindowChoice(WindowPlan.FRAMES_30S, 96, "30 s windows, 96-token overlap (3.84 s)"),
            WindowChoice(WindowPlan.FRAMES_30S, 128, "30 s windows, 128-token overlap (5.12 s)"),
            WindowChoice(WindowPlan.FRAMES_30S, 0, "30 s windows, no overlap (the start of each window is damaged)"),
            WindowChoice(WindowPlan.FRAMES_10S, 96, "10 s windows, 96-token overlap"),
            WindowChoice(WindowPlan.FRAMES_10S, 0, "10 s windows, no overlap"),
        )
        // The NPU runs the 10 s file only; a model offers the lengths it has a file for ("auto" when it has both).
        val choices = all.filter { c ->
            if (Session.backend == Backend.NPU) c.frames == WindowPlan.FRAMES_10S
            else if (c.frames == WindowPlan.AUTO) have.isEmpty() || have.size == 2
            else have.isEmpty() || c.frames in have
        }
        val title = if (Session.backend == Backend.NPU) "Windows (the NPU runs 10 s windows)" else "Windows"
        AlertDialog.Builder(this).setTitle(title).setItems(choices.map { it.label }.toTypedArray()) { _, which ->
            val before = Jobs.pickWindow(applicationContext, null).frames
            Session.windowFrames = choices[which].frames
            Session.overlapTokens = choices[which].overlap
            Session.lastWindowNote = null
            render()
            // Another window length is another file of the pair: compile it now (the NPU keeps its 10 s file).
            if (Jobs.pickWindow(applicationContext, null).frames != before && Session.backend != Backend.NPU && !isBusy) submit { Jobs.loadModel(applicationContext) }
        }.setNegativeButton("Cancel", null).show()
    }

    private fun onStopTapped() {
        Engine.cancel.set(true)
        stopDecode.set(true)
        stopMic.set(true)
    }

    private fun showStatus(text: String) {
        Session.status = text
        render()
    }

    // ---- screen -----------------------------------------------------------------------------------------------

    private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

    private fun label(sp: Float, color: Int = C_TEXT, bold: Boolean = false) = TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        setTextColor(color)
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun button(text: String, onClick: () -> Unit) = Button(this).apply {
        this.text = text
        isAllCaps = false
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
        setOnClickListener { onClick() }
    }

    private fun row(name: String, value: TextView, action: Button?): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        addView(label(14f, C_SUB).apply { text = name; minWidth = dp(72f) })
        addView(value, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        action?.let { addView(it) }
    }

    private fun buildUi(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
        }
        val pad = dp(16f)
        root.addView(label(22f, bold = true).apply { text = "Eg2 Transcribe" })
        root.addView(label(13f, C_SUB).apply {
            text = "${Build.MANUFACTURER.replaceFirstChar { it.titlecase(Locale.US) }} ${Build.MODEL} · LiteRT ${BuildConfig.LITERT_VERSION} · audio stays on this phone"
        })
        pill = label(13f, Color.WHITE, bold = true).apply {
            setPadding(dp(10f), dp(3f), dp(10f), dp(3f))
            letterSpacing = 0.05f
        }
        root.addView(LinearLayout(this).apply {
            setPadding(0, dp(8f), 0, dp(4f))
            addView(pill)
        })
        status = label(15f).apply {
            setPadding(0, 0, 0, dp(6f))
            setOnClickListener {
                if (Session.statusOpensSettings) runCatching {
                    startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null)))
                }
            }
        }
        root.addView(status)
        modelValue = label(14f)
        modelChange = button("Change") { onModelChange() }
        root.addView(row("Model", modelValue, modelChange))
        readerValue = label(14f)
        root.addView(row("Reader", readerValue, null))
        val backends = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for (b in Backend.entries) {
            val btn = button(b.label) { onBackend(b) }
            backendButtons[b] = btn
            backends.addView(btn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        root.addView(backends)
        windowValue = label(14f)
        windowChange = button("Change") { onWindowChange() }
        root.addView(row("Windows", windowValue, windowChange))
        engineLine = label(12f, C_SUB)
        root.addView(engineLine)
        fallbackLine = label(13f, C_ERR)
        root.addView(fallbackLine)
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        recordButton = button("● Record") { onRecord() }
        openButton = button("Open audio file") { onOpenFile() }
        againButton = button("Again") { if (!isBusy) submit(true) { Jobs.transcribeAgain(applicationContext, Jobs.Request(null, false)) } }
        stopButton = button("Stop") { onStopTapped() }
        for (b in listOf(recordButton, openButton, againButton, stopButton)) actions.addView(b, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(actions)
        exampleButton = button("Try") { onExample() }
        root.addView(row("Example", label(14f).apply { text = Examples.LABEL }, exampleButton))
        inputLine = label(13f, C_SUB)
        root.addView(inputLine)
        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 1000 }
        root.addView(progressBar)
        progressLine = label(14f, C_TEXT, bold = true)
        root.addView(progressLine)
        detailLine = label(12f, C_SUB)
        root.addView(detailLine)
        transcript = label(15f).apply {
            setTextIsSelectable(true)
            setLineSpacing(0f, 1.15f)
            setPadding(0, dp(6f), 0, dp(6f))
        }
        root.addView(transcript)
        // One scroll for the whole screen, so the controls and the transcript fit any window shape (landscape, split
        // screen); the system bars and the cutout are kept clear on every side (landscape puts the navigation bar on one).
        scroll = ScrollView(this).apply {
            setBackgroundColor(Color.WHITE)
            isFillViewport = true
            clipToPadding = false
            addView(root)
            setOnApplyWindowInsetsListener { v, insets ->
                if (Build.VERSION.SDK_INT >= 30) {
                    val b = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
                    v.setPadding(b.left + pad, b.top + dp(8f), b.right + pad, b.bottom + dp(8f))
                } else {
                    @Suppress("DEPRECATION")
                    v.setPadding(insets.systemWindowInsetLeft + pad, insets.systemWindowInsetTop + dp(8f), insets.systemWindowInsetRight + pad, insets.systemWindowInsetBottom + dp(8f))
                }
                insets
            }
        }
        return scroll
    }

    private var tickPending = false

    /** While a model loads: render once a second so the elapsed time moves (an NPU compile takes minutes). */
    private fun scheduleTick() {
        if (tickPending) return
        tickPending = true
        main.postDelayed({
            tickPending = false
            if (!isDestroyed && Session.loading) render()
        }, 1000)
    }

    private fun scheduleRender() {
        synchronized(this) {
            if (renderPending) return
            renderPending = true
        }
        main.postDelayed({
            synchronized(this) { renderPending = false }
            if (!isDestroyed) render()
        }, 100)
    }

    private fun render() {
        val busy = isBusy
        pill.text = Session.pill
        pill.background = GradientDrawable().apply {
            cornerRadius = dp(12f).toFloat()
            setColor(when (Session.tone) {
                Session.Tone.IDLE -> C_IDLE
                Session.Tone.WORK -> C_ACCENT
                Session.Tone.LIVE -> C_ERR
                Session.Tone.OK -> C_OK
                Session.Tone.WARN -> C_WARN
            })
        }
        val since = Session.loadingSince
        status.text = if (Session.loading && since > 0L) "${Session.status} ${Session.clock((System.nanoTime() - since) / 1e9)}" else Session.status
        if (Session.loading && since > 0L) scheduleTick()
        status.setTextColor(if (Session.statusOpensSettings) C_ACCENT else C_TEXT)
        modelValue.text = Session.modelLine.ifEmpty { Session.modelName }
        readerValue.text = Session.readerLine.ifEmpty { "…" }
        readerValue.setTextColor(if (Session.readerIsStub) C_WARN else C_TEXT)
        val windowS = Session.effectiveWindowFrames() / 100
        val overlapText = "overlap ${Session.overlapTokens} tokens (${String.format(Locale.US, "%.2f", Session.overlapTokens * 0.04)} s)"
        windowValue.text = if (Session.windowFrames == WindowPlan.AUTO && Session.backend != Backend.NPU) {
            "auto: 10 s files up to 9.96 s, 30 s windows above" + (Session.lastWindowNote?.let { " · last run $it" } ?: " · $windowS s files loaded")
        } else {
            "$windowS s, $overlapText" + if (Session.backend == Backend.NPU || Engine.loaded?.encoder?.backend == Backend.NPU) " · the NPU runs 10 s windows" else ""
        }
        engineLine.text = Session.engineLine
        engineLine.visibility = if (Session.engineLine.isEmpty()) View.GONE else View.VISIBLE
        fallbackLine.text = Session.fallbackLine ?: ""
        fallbackLine.visibility = if (Session.fallbackLine == null) View.GONE else View.VISIBLE
        for ((b, btn) in backendButtons) {
            btn.isEnabled = !busy
            val selected = b == Session.backend
            btn.setTypeface(null, if (selected) Typeface.BOLD else Typeface.NORMAL)
            btn.setTextColor(if (selected) C_ACCENT else C_SUB)
        }
        modelChange.isEnabled = !busy
        windowChange.isEnabled = !busy
        recordButton.text = if (Session.recording) "■ Stop" else "● Record"
        recordButton.isEnabled = Session.recording || !busy
        openButton.isEnabled = !busy
        exampleButton.isEnabled = !busy
        againButton.isEnabled = !busy && Session.lastInput != null
        stopButton.isEnabled = Session.decoding || Session.transcribing
        inputLine.text = Session.inputLine
        progressBar.progress = (Session.progress * 1000).toInt()
        progressLine.text = Session.progressLine
        detailLine.text = Session.detailLine
        if (busy || Session.recordMode) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val note = Session.note
        if (note != null) {
            if (shownNote != note) {
                transcript.text = note
                shownNote = note
                renderedChars = -1
            }
            return
        }
        if (shownNote != null) {
            shownNote = null
            renderedChars = -1
        }
        val length = Session.transcriptLength()
        if (renderedChars < 0 || length < renderedChars) {
            val (all, n) = Session.transcriptSince(0)
            transcript.text = all
            renderedChars = n
        } else if (length > renderedChars) {
            val atBottom = scroll.getChildAt(0).bottom <= scroll.height + scroll.scrollY + dp(48f)
            val (more, n) = Session.transcriptSince(renderedChars)
            transcript.append(more)
            renderedChars = n
            if (atBottom) scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }
}
