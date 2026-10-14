package io.github.johnrocky.eg2transcribe

import java.io.Closeable
import java.util.concurrent.ExecutorService
import kotlin.math.sqrt

/**
 * The reader: one encoder window's soft tokens to a word or a blank per token. The work comes in two parts so that the
 * graph runs on the thread that owns the LiteRT models and the CPU work overlaps the next window:
 * [read] runs on the encoder's thread, right after the encoder, on the whole window (the reader looks both ways within
 * it); [Read.labels] then gives the labels of the window's kept tokens, on the decode thread.
 */
interface Reader : Closeable {
    val name: String
    val isStub: Boolean

    /** The window sizes, in tokens, this reader reads in one call (an encoder window of fewer tokens is padded and masked). */
    val windowTokens: Set<Int>

    /**
     * The window's soft tokens soft[0 until tokens * 512], the first [valid] of them real (the rest padding), to the
     * results for its tokens [from] until [until].
     */
    fun read(soft: FloatArray, tokens: Int, valid: Int, from: Int, until: Int): Read

    override fun close() {}
}

/**
 * One window's reader results for its kept tokens, [n] of them. A graph reader also says how its call was spent:
 * [writeMs] = the window into the input buffers, [runMs] = the graph's run call, [readMs] = the outputs back (on a GPU
 * the run call can return before the work is done, and the read then waits for it); all 0 for the stub.
 */
abstract class Read(val n: Int) {
    var writeMs = 0.0
    var runMs = 0.0
    var readMs = 0.0

    /**
     * Fills, per token: [ids] = a word (the vocabulary's line) or [Ctc.BLANK]; [word] = the best word even where the token
     * is blank; [best] = that word's score; [blank] = the blank's score (a token is blank when blank > best).
     */
    abstract fun labels(ids: IntArray, word: IntArray, best: DoubleArray, blank: DoubleArray)
}

/**
 * The trained reader's graph outputs for one window ([TfliteReader]) to labels: per token, [outIds] / [outBest] hold
 * [slices] values (the argmax and its score within each slice of vocabSize / slices words, slice-major per token) and
 * [outBlank] one. The slice with the largest score wins (the first on a tie = the lowest word, torch's argmax); the
 * token is blank when its blank score is above that word's (greedy_labels' rule, compared in float32 as torch does).
 * Tokens [from] until from + n of the window.
 */
class GraphRead(
    private val outIds: IntArray,
    private val outBest: FloatArray,
    private val outBlank: FloatArray,
    private val slices: Int,
    private val vocabSize: Int,
    private val from: Int,
    n: Int,
) : Read(n) {
    override fun labels(ids: IntArray, word: IntArray, best: DoubleArray, blank: DoubleArray) {
        val k = slices
        val per = vocabSize / k
        for (t in 0 until n) {
            val g = from + t
            var s = 0
            var b = outBest[g * k]
            for (c in 1 until k) {
                val v = outBest[g * k + c]
                if (v > b) { b = v; s = c }
            }
            val w = outIds[g * k + s] + s * per
            val bl = outBlank[g]
            word[t] = w
            best[t] = b.toDouble()
            blank[t] = bl.toDouble()
            ids[t] = if (bl > b) Ctc.BLANK else w
        }
    }
}

/**
 * The placeholder reader used until a trained reader is in place: identity (d = 512), a table of [Stub.WORDS] random
 * rows for the words of `stub_vocab.txt`, and a random blank vector; a token is blank when the blank vector is nearer
 * than every word. Its words have nothing to do with the speech; the pipeline, timing and windowing around it are the
 * real ones. The table scan runs on the CPU in [Read.labels] (on [pool] with [threads]).
 */
class StubReader(private val blankVector: FloatArray, val table: WordTable, private val pool: ExecutorService?, private val threads: Int) : Reader {
    override val name = "stub"
    override val isStub = true
    override val windowTokens: Set<Int> = emptySet() // any

    override fun read(soft: FloatArray, tokens: Int, valid: Int, from: Int, until: Int): Read {
        val n = maxOf(0, minOf(until, valid) - from)
        val rows = soft.copyOfRange(from * SOFT_DIM, (from + n) * SOFT_DIM)
        return object : Read(n) {
            override fun labels(ids: IntArray, word: IntArray, best: DoubleArray, blank: DoubleArray) {
                val cos = DoubleArray(n)
                val second = DoubleArray(n)
                table.nearestAll(rows, n, pool, threads, word, cos, second)
                for (t in 0 until n) {
                    var dot = 0.0
                    var norm2 = 0.0
                    for (i in 0 until SOFT_DIM) {
                        val x = rows[t * SOFT_DIM + i].toDouble()
                        dot += x * blankVector[i]
                        norm2 += x * x
                    }
                    blank[t] = if (norm2 == 0.0) 0.0 else dot / sqrt(norm2)
                    best[t] = cos[t]
                    ids[t] = if (word[t] < 0 || blank[t] > cos[t]) Ctc.BLANK else word[t]
                }
            }
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

    fun blankVector(): FloatArray {
        val b = uniform(DIM, skip = WORDS.toLong() * DIM)
        var s = 0.0
        for (x in b) s += x.toDouble() * x
        val norm = sqrt(s)
        return FloatArray(DIM) { (b[it] / norm).toFloat() }
    }

    fun reader(vocab: List<String>, pool: ExecutorService?, threads: Int): StubReader = StubReader(blankVector(), table(vocab), pool, threads)

    private val GAMMA = 0x9E3779B97F4A7C15uL.toLong()
    private val M1 = 0xBF58476D1CE4E5B9uL.toLong()
    private val M2 = 0x94D049BB133111EBuL.toLong()
}
