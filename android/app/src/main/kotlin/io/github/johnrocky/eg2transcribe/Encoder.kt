package io.github.johnrocky.eg2transcribe

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Where the encoder runs. GPU fp32 / GPU default = LiteRT's GPU accelerator with fp32 or its default precision. */
enum class Backend(val id: String, val label: String) {
    CPU("cpu", "CPU"),
    GPU_FP32("gpu_fp32", "GPU fp32"),
    GPU_DEFAULT("gpu_default", "GPU default"),
    NPU("npu", "NPU");

    val isGpu: Boolean get() = this == GPU_FP32 || this == GPU_DEFAULT

    companion object {
        fun parse(s: String?): Backend? = when (s?.lowercase()) {
            null -> null
            "gpu" -> GPU_FP32
            else -> entries.firstOrNull { it.id == s.lowercase() }
        }
    }
}

/**
 * The LiteRT threads and the process-wide Environment. Every CompiledModel call runs on the LiteRT thread, except the
 * reader's when the encoder runs on the NPU: that reader graph is compiled, run and closed on its own thread
 * ([readerExecutor]), so that it reads window k on the GPU while the encoder runs window k + 1 on the NPU
 * ([Engine.readerOwnThread], [Pipeline]). An APK that carries the Qualcomm NPU libraries gets an Environment that
 * knows their directory (dispatch and JIT compiler plugin) and keeps compiled NPU graphs in [Npu.cacheRoot], and every
 * graph it compiles, on any accelerator, asks for the HTP's BURST mode: LiteRT 2.2.0 starts the NPU runtime once per
 * process from the first graph's options.
 *
 * LiteRT logs "NPU accelerator could not be loaded and registered: kLiteRtStatusErrorInvalidArgument" once per
 * process, about a second after this Environment registered the NPU accelerator: that is a second, internal
 * Environment LiteRT 2.2.0 makes without options for its older C++ paths (`TensorBuffer::GetDefaultEnvironment()`),
 * which has no dispatch library directory. It does not change this one.
 */
object LiteRt {
    val executor = Executors.newSingleThreadExecutor { r -> Thread(r, THREAD).apply { isDaemon = true } }

    /** The reader's own thread, for a reader next to an encoder on the NPU. */
    val readerExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, READER_THREAD).apply { isDaemon = true } }
    private const val THREAD = "litert"
    private const val READER_THREAD = "litert-reader"
    @Volatile private var env: Environment? = null
    @Volatile var npuPackaged = false
        private set

    fun environment(context: Context): Environment = env ?: synchronized(this) {
        env ?: run {
            npuPackaged = Npu.packaged(context)
            val e = if (npuPackaged) {
                val dir = context.applicationInfo.nativeLibraryDir
                Environment.create(Npu.cacheContext(context), mapOf(Environment.Option.DispatchLibraryDir to dir, Environment.Option.CompilerPluginLibraryDir to dir))
            } else Environment.create()
            Log.i(Engine.TAG, "STEP environment npu_libraries=$npuPackaged accelerators=${runCatching { e.getAvailableAccelerators() }.getOrNull()}")
            env = e
            e
        }
    }

    fun <T> call(block: () -> T): T = callOn(executor, THREAD, block)

    /** [block] on the reader's own thread when [readerThread], else on the LiteRT thread. */
    fun <T> on(readerThread: Boolean, block: () -> T): T = if (readerThread) callOn(readerExecutor, READER_THREAD, block) else call(block)

    private fun <T> callOn(exec: ExecutorService, name: String, block: () -> T): T {
        if (Thread.currentThread().name == name) return block()
        try {
            return exec.submit(Callable { block() }).get()
        } catch (e: ExecutionException) {
            throw e.cause ?: e
        }
    }
}

/** Thrown when the encoder cannot be used as asked; [text] is the screen's sentence. */
class EncoderException(val code: String, val text: String, cause: Throwable? = null) : Exception(text, cause)

/**
 * The EmbeddingGemma 2 audio encoder exported to LiteRT: a .tflite with the signature `audio_10s` (mel
 * [1, 1000, 128] + mask [1, 1000] -> soft [1, 250, 512] + mask [1, 250]) or `audio_30s` (3000 frames -> 750 tokens),
 * or both, masks in float32 (1 = a real frame / token). The export gives one file per window ([ModelChoice]).
 * Buffers are made once per signature and reused; one window runs at a time, on the LiteRT thread.
 */
class Encoder private constructor(
    val file: File,
    val backend: Backend,
    private val model: CompiledModel,
    private val sigs: Map<Int, Sig>,
    val compileMs: Double,
    val threads: Int,
    /** The NPU's cache and link facts ([Npu.prepare]); null on the CPU and the GPU. */
    val npu: Npu.Prepared?,
) : EncoderRunner, Closeable {
    private class Sig(val name: String, val frames: Int, val mel: TensorBuffer, val mask: TensorBuffer, val soft: TensorBuffer, val outMask: TensorBuffer) {
        val inputs = mapOf("mel" to mel, "mask" to mask)
        val outputs = mapOf("soft" to soft, "mask" to outMask)
        fun close() = listOf(mel, mask, soft, outMask).forEach { runCatching { it.close() } }
    }

    private val closed = AtomicBoolean(false)
    override val windowSizes: Set<Int> = sigs.keys
    val signatureNames: List<String> get() = sigs.values.map { it.name }

    /** Warm-up time per signature name (one run of a silent window, right after the compile). */
    val warmupMs = LinkedHashMap<String, Double>()

    /** False when a warm-up run gave a NaN or an infinity. */
    var warmupFinite = true
        private set

    override fun run(frames: Int, mel: FloatArray, mask: FloatArray): EncoderOutput {
        check(!closed.get()) { "the encoder is closed" }
        val s = sigs[frames] ?: throw IllegalArgumentException("no signature for $frames frames (has ${sigs.keys})")
        val t = System.nanoTime()
        s.mel.writeFloat(mel)
        s.mask.writeFloat(mask)
        val t1 = System.nanoTime()
        model.run(s.inputs, s.outputs, s.name)
        val t2 = System.nanoTime()
        val soft = s.soft.readFloat()
        val m = s.outMask.readFloat()
        val t3 = System.nanoTime()
        return EncoderOutput(soft, m, (t3 - t) / 1e6, (t1 - t) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6)
    }

    private fun warmUp() {
        for ((frames, s) in sigs) {
            val mel = FloatArray(frames * MelFrontEnd.NMEL) { SILENCE }
            val mask = FloatArray(frames) { 1f }
            val out = run(frames, mel, mask)
            warmupMs[s.name] = out.ms
            if (out.soft.any { !it.isFinite() }) warmupFinite = false
        }
    }

    /** Idempotent. Call it on the LiteRT thread, after the last [run]. */
    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        sigs.values.forEach { it.close() }
        runCatching { model.close() }
    }

    companion object {
        val SIGNATURES = linkedMapOf(WindowPlan.FRAMES_30S to "audio_30s", WindowPlan.FRAMES_10S to "audio_10s")

        /** log(0 + 0.001): the log-mel of digital silence. */
        private const val SILENCE = -6.9077554f

        /**
         * Compiles [file] for [backend] and warms it up (LiteRT thread; [prepared] = [Npu.prepare] of [file], on the
         * NPU). The NPU gets the NPU and the CPU accelerators, as the native runner of the measurements did: the
         * Qualcomm plugin takes the whole graph, and the CPU is there for an op it would leave out.
         */
        fun open(context: Context, file: File, backend: Backend, cpuThreads: Int, prepared: Npu.Prepared? = null): Encoder = LiteRt.call {
            if (!file.isFile || file.length() == 0L) throw EncoderException("model_missing", "The encoder file ${file.name} is not on this phone.")
            if (backend == Backend.NPU) Npu.unavailable(context)?.let { throw EncoderException("npu_unavailable", it) }
            val env = LiteRt.environment(context)
            if (backend == Backend.NPU && Accelerator.NPU !in env.getAvailableAccelerators()) {
                throw EncoderException("npu_unavailable", "LiteRT has no NPU accelerator in this process: the Qualcomm runtime did not load (logcat has the reason).")
            }
            val burst = if (LiteRt.npuPackaged) CompiledModel.QualcommOptions(htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST) else null
            val options = when (backend) {
                Backend.CPU -> CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(numThreads = cpuThreads) }
                Backend.GPU_FP32 -> CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32) }
                Backend.GPU_DEFAULT -> CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.DEFAULT) }
                Backend.NPU -> CompiledModel.Options(Accelerator.NPU, Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(numThreads = cpuThreads) }
            }.apply { qualcommOptions = burst }
            val path = if (backend == Backend.NPU) (prepared?.path ?: file) else file
            Log.i(Engine.TAG, "STEP compile backend=${backend.id} path=${path.absolutePath}")
            val t = System.nanoTime()
            val model = try {
                CompiledModel.create(path.absolutePath, options, env)
            } catch (e: Throwable) {
                throw EncoderException("compile_failed", "${backend.label} could not compile ${file.name}: ${e.javaClass.simpleName}: ${e.message}", e)
            }
            val compileMs = (System.nanoTime() - t) / 1e6
            Log.i(Engine.TAG, "STEP compiled backend=${backend.id} ms=${"%.1f".format(java.util.Locale.US, compileMs)}")
            val sigs = LinkedHashMap<Int, Sig>()
            val enc = try {
                for ((frames, name) in SIGNATURES) {
                    val ok = runCatching {
                        model.getInputTensorType("mel", name).layout?.dimensions == listOf(1, frames, MelFrontEnd.NMEL) &&
                            model.getInputTensorType("mask", name).layout?.dimensions == listOf(1, frames) &&
                            model.getOutputTensorType("soft", name).layout?.dimensions == listOf(1, frames / 4, SOFT_DIM) &&
                            model.getOutputTensorType("mask", name).layout?.dimensions == listOf(1, frames / 4)
                    }.getOrDefault(false)
                    if (!ok) continue
                    sigs[frames] = Sig(
                        name, frames,
                        model.createInputBuffer("mel", name), model.createInputBuffer("mask", name),
                        model.createOutputBuffer("soft", name), model.createOutputBuffer("mask", name),
                    )
                }
                if (sigs.isEmpty()) throw EncoderException(
                    "not_an_encoder",
                    "${file.name} is not the EmbeddingGemma 2 audio encoder export: no signature audio_30s or audio_10s with mel [1, F, 128] + mask [1, F] -> soft [1, F/4, 512] + mask [1, F/4].",
                )
                Log.i(Engine.TAG, "STEP buffers signatures=${sigs.values.map { it.name }}")
                Encoder(file, backend, model, sigs, compileMs, cpuThreads, if (backend == Backend.NPU) prepared else null)
            } catch (e: Throwable) {
                sigs.values.forEach { it.close() }
                runCatching { model.close() }
                throw e
            }
            // From here the encoder owns the model and the buffers: close them once, through it.
            try {
                enc.warmUp()
            } catch (e: Throwable) {
                enc.close()
                throw EncoderException("run_failed", "${backend.label} compiled ${file.name} but the first run failed: ${e.javaClass.simpleName}: ${e.message}", e)
            }
            Log.i(Engine.TAG, "STEP warmed_up ${enc.warmupMs}")
            enc
        }
    }
}
