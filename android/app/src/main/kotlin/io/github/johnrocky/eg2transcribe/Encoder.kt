package io.github.johnrocky.eg2transcribe

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
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
 * The LiteRT thread and the process-wide Environment. Every CompiledModel call runs on this one thread. An APK that
 * carries the Qualcomm NPU libraries gets an Environment that knows their directory (dispatch and JIT compiler
 * plugin), and every graph it compiles, on any accelerator, asks for the HTP's BURST mode: LiteRT 2.2.0 starts the
 * NPU runtime once per process from the first graph's options.
 */
object LiteRt {
    val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "litert").apply { isDaemon = true } }
    @Volatile private var env: Environment? = null
    @Volatile var npuPackaged = false
        private set

    fun environment(context: Context): Environment = env ?: synchronized(this) {
        env ?: run {
            npuPackaged = Npu.packaged(context)
            val e = if (npuPackaged) {
                val dir = context.applicationInfo.nativeLibraryDir
                Environment.create(context, mapOf(Environment.Option.DispatchLibraryDir to dir, Environment.Option.CompilerPluginLibraryDir to dir))
            } else Environment.create()
            env = e
            e
        }
    }

    fun <T> call(block: () -> T): T {
        if (Thread.currentThread().name == "litert") return block()
        try {
            return executor.submit(Callable { block() }).get()
        } catch (e: ExecutionException) {
            throw e.cause ?: e
        }
    }
}

/** Thrown when the encoder cannot be used as asked; [text] is the screen's sentence. */
class EncoderException(val code: String, val text: String, cause: Throwable? = null) : Exception(text, cause)

/**
 * The EmbeddingGemma 2 audio encoder exported to LiteRT: one .tflite with the signatures `audio_10s` (mel
 * [1, 1000, 128] + mask [1, 1000] -> soft [1, 250, 512] + mask [1, 250]) and `audio_30s` (3000 frames -> 750 tokens),
 * masks in float32 (1 = a real frame / token). Either one alone also works. Buffers are made once per signature
 * and reused; one window runs at a time, on the LiteRT thread.
 */
class Encoder private constructor(
    val file: File,
    val backend: Backend,
    private val model: CompiledModel,
    private val sigs: Map<Int, Sig>,
    val compileMs: Double,
    val threads: Int,
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
        model.run(s.inputs, s.outputs, s.name)
        val soft = s.soft.readFloat()
        val m = s.outMask.readFloat()
        return EncoderOutput(soft, m, (System.nanoTime() - t) / 1e6)
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

        /** Compiles [file] for [backend] and warms it up. On the LiteRT thread. */
        fun open(context: Context, file: File, backend: Backend, cpuThreads: Int): Encoder = LiteRt.call {
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
                Backend.NPU -> CompiledModel.Options(Accelerator.NPU)
            }.apply { qualcommOptions = burst }
            val t = System.nanoTime()
            val model = try {
                CompiledModel.create(file.absolutePath, options, env)
            } catch (e: Throwable) {
                throw EncoderException("compile_failed", "${backend.label} could not compile ${file.name}: ${e.javaClass.simpleName}: ${e.message}", e)
            }
            val compileMs = (System.nanoTime() - t) / 1e6
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
                Encoder(file, backend, model, sigs, compileMs, cpuThreads)
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
            enc
        }
    }
}
