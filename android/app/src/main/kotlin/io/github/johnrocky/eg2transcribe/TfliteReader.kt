package io.github.johnrocky.eg2transcribe

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The trained reader (the version `reader/RELEASE.md` describes) as one LiteRT graph per window length, word table and argmax
 * included: `reader_w30.tflite` (signature `reader_30s`, 750 tokens) and `reader_w10.tflite` (`reader_10s`, 250).
 *
 * Inputs `soft` [1, T, 512] (the encoder's window) and `mask` [1, T] (1 = a real token); outputs `ids` [1, T] int32 (the
 * best of the 200,000 words: argmax of the reader's scale x cosine to the word table; `scale` in `reader_config.json`, 91.218 for v1 and 216.351 for v2), `best` [1, T] (that score) and `blank`
 * [1, T] (the blank's score). A graph exported with the table in K slices gives `ids` / `best` as [1, T, K], one argmax
 * per slice of vocab / K words; the slice with the largest score wins (the lowest slice on a tie = the lowest word, as
 * torch's argmax). A token is blank when its blank score is above its best word's (the release's greedy rule).
 * Runs on the LiteRT thread, on the accelerator the encoder uses (the NPU's encoder gets a GPU fp32 reader).
 */
class TfliteReader private constructor(
    val file: File,
    val backend: Backend,
    private val model: CompiledModel,
    private val signature: String,
    private val softBuf: TensorBuffer,
    private val maskBuf: TensorBuffer,
    private val idsBuf: TensorBuffer,
    private val bestBuf: TensorBuffer,
    private val blankBuf: TensorBuffer,
    /** Tokens per call (T). */
    val tokens: Int,
    /** Table slices (K); 1 = one argmax over the whole vocabulary. */
    val slices: Int,
    private val vocabSize: Int,
    val compileMs: Double,
    val threads: Int,
    /** The GPU precision asked for ([ReaderPrecision.id]), or "default" / "cpu". */
    val precision: String,
) : Reader {
    override val name: String = file.name
    override val isStub = false
    override val windowTokens: Set<Int> = setOf(tokens)
    private val inputs = mapOf("soft" to softBuf, "mask" to maskBuf)
    private val outputs = mapOf("ids" to idsBuf, "best" to bestBuf, "blank" to blankBuf)
    private val soft = FloatArray(tokens * SOFT_DIM)
    private val mask = FloatArray(tokens)
    private val closed = AtomicBoolean(false)

    /** Warm-up time (one run of a silent window, right after the compile). */
    var warmupMs = 0.0
        private set

    override fun read(soft: FloatArray, tokens: Int, valid: Int, from: Int, until: Int): Read {
        check(!closed.get()) { "the reader is closed" }
        require(tokens <= this.tokens && valid <= tokens) { "a window of $tokens tokens ($valid valid) does not fit the reader's ${this.tokens}" }
        val t0 = System.nanoTime()
        System.arraycopy(soft, 0, this.soft, 0, valid * SOFT_DIM)
        java.util.Arrays.fill(this.soft, valid * SOFT_DIM, this.soft.size, 0f)
        java.util.Arrays.fill(mask, 0, valid, 1f)
        java.util.Arrays.fill(mask, valid, mask.size, 0f)
        softBuf.writeFloat(this.soft)
        maskBuf.writeFloat(mask)
        val t1 = System.nanoTime()
        model.run(inputs, outputs, signature)
        val t2 = System.nanoTime()
        val n = maxOf(0, minOf(until, valid) - from)
        return GraphRead(idsBuf.readInt(), bestBuf.readFloat(), blankBuf.readFloat(), slices, vocabSize, from, n).also {
            val t3 = System.nanoTime()
            it.writeMs = (t1 - t0) / 1e6
            it.runMs = (t2 - t1) / 1e6
            it.readMs = (t3 - t2) / 1e6
        }
    }

    private fun warmUp(): Boolean {
        val t = System.nanoTime()
        val r = read(FloatArray(tokens * SOFT_DIM), tokens, tokens, 0, tokens)
        val ids = IntArray(r.n); val word = IntArray(r.n); val best = DoubleArray(r.n); val blank = DoubleArray(r.n)
        r.labels(ids, word, best, blank)
        warmupMs = (System.nanoTime() - t) / 1e6
        return best.all { it.isFinite() } && blank.all { it.isFinite() } && word.all { it in 0 until vocabSize }
    }

    /** Idempotent. Call it on the LiteRT thread, after the last [read]. */
    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        listOf(softBuf, maskBuf, idsBuf, bestBuf, blankBuf).forEach { runCatching { it.close() } }
        runCatching { model.close() }
    }

    companion object {
        /** File names by window length in mel frames, and their signatures. */
        val FILES = linkedMapOf(WindowPlan.FRAMES_30S to "reader_w30.tflite", WindowPlan.FRAMES_10S to "reader_w10.tflite")
        val SIGNATURES = mapOf("reader_w30.tflite" to "reader_30s", "reader_w10.tflite" to "reader_10s")

        /** The reader file in [dir] for windows of [windowFrames] frames: that length's own, else a longer one (it pads). */
        fun fileFor(dir: File, windowFrames: Int): File? {
            FILES[windowFrames]?.let { File(dir, it) }?.takeIf { it.isFile }?.let { return it }
            return FILES.entries.filter { it.key > windowFrames }.map { File(dir, it.value) }.firstOrNull { it.isFile }
        }

        /**
         * Compiles [file] for [backend] (LiteRT thread) and checks its signature: inputs soft [1, T, 512] + mask [1, T],
         * outputs ids [1, T(, K)] int32 + best [1, T(, K)] + blank [1, T]; then one warm-up run. On GPU fp32 (and for
         * the NPU's encoder) [precision] picks fp32 or fp16 arithmetic with fp32 accumulation. Throws [EncoderException]
         * with the screen's sentence.
         */
        fun open(context: Context, file: File, backend: Backend, vocabSize: Int, cpuThreads: Int, precision: ReaderPrecision = ReaderPrecision.FP32): TfliteReader = LiteRt.call {
            val sig = SIGNATURES[file.name] ?: throw EncoderException("not_a_reader", "${file.name} is not a reader file name (reader_w30.tflite or reader_w10.tflite).")
            val env = LiteRt.environment(context)
            val gpuPrecision = if (precision == ReaderPrecision.FP16_ACC32) CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM else CompiledModel.GpuOptions.Precision.FP32
            val options = when (backend) {
                Backend.CPU -> CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(numThreads = cpuThreads) }
                Backend.GPU_FP32, Backend.NPU -> CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = CompiledModel.GpuOptions(precision = gpuPrecision) }
                Backend.GPU_DEFAULT -> CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.DEFAULT) }
            }
            val precisionId = when (backend) {
                Backend.CPU -> "cpu"
                Backend.GPU_DEFAULT -> "default"
                else -> precision.id
            }
            Log.i(Engine.TAG, "STEP reader_compile backend=${backend.id} precision=$precisionId path=${file.absolutePath}")
            val t = System.nanoTime()
            val model = try {
                CompiledModel.create(file.absolutePath, options, env)
            } catch (e: Throwable) {
                throw EncoderException("compile_failed", "${backend.label} could not compile ${file.name}: ${e.javaClass.simpleName}: ${e.message}", e)
            }
            val compileMs = (System.nanoTime() - t) / 1e6
            val bufs = ArrayList<TensorBuffer>()
            val reader = try {
                val soft = model.getInputTensorType("soft", sig).layout?.dimensions
                val mask = model.getInputTensorType("mask", sig).layout?.dimensions
                val ids = model.getOutputTensorType("ids", sig).layout?.dimensions
                val best = model.getOutputTensorType("best", sig).layout?.dimensions
                val blank = model.getOutputTensorType("blank", sig).layout?.dimensions
                val tokens = soft?.getOrNull(1) ?: -1
                val slices = if (ids != null && ids.size == 3) ids[2] else 1
                val ok = soft == listOf(1, tokens, SOFT_DIM) && mask == listOf(1, tokens) && blank == listOf(1, tokens) &&
                    ids == best && (ids == listOf(1, tokens) || ids == listOf(1, tokens, slices)) && slices >= 1 && vocabSize % slices == 0
                if (!ok) throw EncoderException(
                    "not_a_reader",
                    "${file.name} does not have the reader's signature $sig: soft [1, T, $SOFT_DIM] + mask [1, T] -> ids, best [1, T] + blank [1, T] (has soft $soft, mask $mask, ids $ids, best $best, blank $blank).",
                )
                listOf("soft", "mask").forEach { bufs += model.createInputBuffer(it, sig) }
                listOf("ids", "best", "blank").forEach { bufs += model.createOutputBuffer(it, sig) }
                TfliteReader(file, backend, model, sig, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], tokens, slices, vocabSize, compileMs, cpuThreads, precisionId)
            } catch (e: Throwable) {
                bufs.forEach { runCatching { it.close() } }
                runCatching { model.close() }
                throw e
            }
            val finite = try {
                reader.warmUp()
            } catch (e: Throwable) {
                reader.close()
                throw EncoderException("run_failed", "${backend.label} compiled ${file.name} but the first run failed: ${e.javaClass.simpleName}: ${e.message}", e)
            }
            if (!finite) {
                reader.close()
                throw EncoderException("not_finite", "${backend.label} gave NaN, infinite or out-of-range values in the reader's test run")
            }
            Log.i(Engine.TAG, "STEP reader_ready backend=${backend.id} precision=$precisionId tokens=${reader.tokens} slices=${reader.slices} compile_ms=${"%.1f".format(java.util.Locale.US, compileMs)} warmup_ms=${"%.1f".format(java.util.Locale.US, reader.warmupMs)}")
            reader
        }
    }
}

/**
 * The reader graph's arithmetic on a GPU: fp32 (the default: the labels of the PyTorch reader on every fixture token),
 * or fp16 with fp32 accumulation (a hidden setting, launch extra `reader_precision`, to measure it on a phone).
 */
enum class ReaderPrecision(val id: String) {
    FP32("fp32"),
    FP16_ACC32("fp16acc32");

    companion object {
        fun parse(s: String?): ReaderPrecision? = entries.firstOrNull { it.id == s?.lowercase() }
    }
}
