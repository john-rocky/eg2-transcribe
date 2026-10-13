package io.github.johnrocky.eg2transcribe

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import org.json.JSONObject

/**
 * A trained reader as `reader.tflite` in the app's `reader/` dir, run on the CPU from the decode thread. The graph is
 * read by shape: one input of T x 512 floats (T soft tokens), two outputs of T x d and T floats (the vectors and the
 * blank logits); tokens go through in blocks of T, the last one zero-padded. `reader_config.json` may set
 * `blank_logit_scale` (a token is blank when its logit is above scale x its best word cosine; without it, above 0).
 * The released reader's own contract (reader/RELEASE.md) decides the final form; a graph of another shape is refused
 * and the app keeps the stub.
 */
class TfliteReader private constructor(
    override val name: String,
    private val model: CompiledModel,
    private val input: TensorBuffer,
    private val outputs: List<TensorBuffer>,
    private val vectorSlot: Int,
    private val blankSlot: Int,
    private val block: Int,
    override val dim: Int,
    override val blankScale: Double?,
) : Reader {
    override val isStub = false
    private val buf = FloatArray(block * SOFT_DIM)

    override fun read(soft: FloatArray, off: Int, n: Int, vectors: FloatArray, blank: DoubleArray) {
        var t = 0
        while (t < n) {
            val m = minOf(block, n - t)
            System.arraycopy(soft, off + t * SOFT_DIM, buf, 0, m * SOFT_DIM)
            java.util.Arrays.fill(buf, m * SOFT_DIM, buf.size, 0f)
            input.writeFloat(buf)
            model.run(listOf(input), outputs)
            val v = outputs[vectorSlot].readFloat()
            val b = outputs[blankSlot].readFloat()
            System.arraycopy(v, 0, vectors, t * dim, m * dim)
            for (i in 0 until m) blank[t + i] = b[i].toDouble()
            t += m
        }
    }

    override fun close() {
        (outputs + input).forEach { runCatching { it.close() } }
        runCatching { model.close() }
    }

    companion object {
        /** The reader in [dir], or an exception whose message says why it cannot be used. */
        fun open(context: Context, dir: File, threads: Int): TfliteReader {
            val file = File(dir, "reader.tflite")
            val config = File(dir, "reader_config.json").takeIf { it.isFile }?.let { JSONObject(it.readText()) }
            val scale = config?.optDouble("blank_logit_scale", Double.NaN)?.takeIf { !it.isNaN() }
            val model = CompiledModel.create(file.absolutePath, CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(numThreads = threads) }, LiteRt.environment(context))
            var ins: List<TensorBuffer> = emptyList()
            var outs: List<TensorBuffer> = emptyList()
            try {
                ins = model.createInputBuffers()
                outs = model.createOutputBuffers()
                val inSizes = ins.map { it.readFloat().size }
                val outSizes = outs.map { it.readFloat().size }
                require(ins.size == 1 && inSizes[0] > 0 && inSizes[0] % SOFT_DIM == 0 && outs.size == 2) {
                    "reader.tflite has inputs of $inSizes floats and outputs of $outSizes; expected one T x $SOFT_DIM input and two outputs (T x d, T)"
                }
                val t = inSizes[0] / SOFT_DIM
                val blankSlot = outSizes.indexOf(t)
                val vectorSlot = 1 - blankSlot
                require(blankSlot >= 0 && outSizes[vectorSlot] % t == 0 && outSizes[vectorSlot] / t >= 2) {
                    "reader.tflite outputs $outSizes floats for $t tokens; expected T x d and T"
                }
                return TfliteReader(file.name, model, ins[0], outs, vectorSlot, blankSlot, t, outSizes[vectorSlot] / t, scale)
            } catch (e: Throwable) {
                (ins + outs).forEach { runCatching { it.close() } }
                runCatching { model.close() }
                throw e
            }
        }
    }
}
