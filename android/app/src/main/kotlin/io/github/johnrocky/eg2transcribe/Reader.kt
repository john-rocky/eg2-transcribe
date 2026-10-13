package io.github.johnrocky.eg2transcribe

import java.io.Closeable
import kotlin.math.sqrt

/**
 * The reader: soft tokens [n, 512] (the encoder's output) to one vector per token in the word table's space [n, dim]
 * plus a blank logit per token. A token is blank when [isBlank] says so; otherwise its word is the table row nearest
 * to its vector ([WordTable.nearest]). [Ctc] then collapses the stream.
 */
interface Reader : Closeable {
    val name: String
    val dim: Int
    val isStub: Boolean

    /** Soft tokens soft[off until off + n * 512] to [vectors] ([n, dim] row-major) and [blank] ([n]). */
    fun read(soft: FloatArray, off: Int, n: Int, vectors: FloatArray, blank: DoubleArray)

    /**
     * The blank decision from the token's blank logit and its best word cosine. [blankScale] null: blank when the
     * logit is above 0; else blank when it is above blankScale x the best cosine (the stub: scale 1, its logit is the
     * cosine to the blank vector, so a token is blank when the blank vector is nearer than every word).
     */
    val blankScale: Double?

    fun isBlank(logit: Double, bestCos: Double): Boolean = blankScale?.let { logit > it * bestCos } ?: (logit > 0.0)

    override fun close() {}
}

/**
 * The placeholder reader used until a trained reader is in place: identity (d = 512), a table of [Stub.WORDS]
 * random rows for the words of `stub_vocab.txt`, and a random blank vector. Its words have nothing to do with the
 * speech; the pipeline, timing and windowing around it are the real ones.
 */
class StubReader(private val blankVector: FloatArray) : Reader {
    override val name = "stub"
    override val dim = Stub.DIM
    override val isStub = true
    override val blankScale = 1.0

    override fun read(soft: FloatArray, off: Int, n: Int, vectors: FloatArray, blank: DoubleArray) {
        System.arraycopy(soft, off, vectors, 0, n * dim)
        for (t in 0 until n) {
            var dot = 0.0
            var norm2 = 0.0
            for (i in 0 until dim) {
                val x = soft[off + t * dim + i].toDouble()
                dot += x * blankVector[i]
                norm2 += x * x
            }
            blank[t] = if (norm2 == 0.0) 0.0 else dot / sqrt(norm2)
        }
    }
}

/**
 * The stub's random numbers: SplitMix64 from [SEED], each output's top 24 bits as a float in [-1, 1) (exact in
 * float32, so `tools/stub_table.py` and this code produce the same values bit for bit). The table takes the first
 * WORDS x DIM values (row-major), the blank vector the next DIM. Rows are L2-normalized in float64.
 */
object Stub {
    const val SEED = 20261013L
    const val WORDS = 1000
    const val DIM = 512

    fun uniform(count: Int, skip: Long = 0, seed: Long = SEED): FloatArray {
        var state = seed + skip * GAMMA
        return FloatArray(count) {
            state += GAMMA
            var z = state
            z = (z xor (z ushr 30)) * M1
            z = (z xor (z ushr 27)) * M2
            z = z xor (z ushr 31)
            (z ushr 40).toInt() / 8388608f - 1f
        }
    }

    fun table(vocab: List<String>): WordTable {
        require(vocab.size == WORDS) { "the stub vocabulary has ${vocab.size} words, expected $WORDS" }
        return WordTable.fromRaw(vocab, DIM, uniform(WORDS * DIM))
    }

    fun reader(): StubReader {
        val b = uniform(DIM, skip = WORDS.toLong() * DIM)
        var s = 0.0
        for (x in b) s += x.toDouble() * x
        val norm = sqrt(s)
        return StubReader(FloatArray(DIM) { (b[it] / norm).toFloat() })
    }

    private val GAMMA = 0x9E3779B97F4A7C15uL.toLong()
    private val M1 = 0xBF58476D1CE4E5B9uL.toLong()
    private val M2 = 0x94D049BB133111EBuL.toLong()
}
