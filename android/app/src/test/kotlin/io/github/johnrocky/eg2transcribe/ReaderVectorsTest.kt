package io.github.johnrocky.eg2transcribe

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The app's trained-reader path after the graph: the reader graph's outputs for a fixture utterance (computed on a Mac,
 * `src/test/resources/reader/<id>.txt`) through [GraphRead] (blank when the blank score is above the best word's) and
 * [Ctc] give the transcript of reader/RELEASE.md; also through [Pipeline] with a fake encoder, so the window plan's
 * kept range and the token offsets are the app's.
 */
class ReaderVectorsTest {
    private class Vector(val tokens: Int, val ids: IntArray, val best: FloatArray, val blank: FloatArray, val words: Map<Int, String>, val transcript: String)

    private fun load(f: File): Vector {
        val m = HashMap<String, String>()
        val words = HashMap<Int, String>()
        for (line in f.readLines()) {
            if (line.startsWith("#") || line.isBlank()) continue
            val key = line.substringBefore(' ')
            val rest = line.substringAfter(' ')
            if (key == "word") words[rest.substringBefore(' ').toInt()] = rest.substringAfter(' ') else m[key] = rest
        }
        val n = m.getValue("tokens").toInt()
        fun ints(k: String) = m.getValue(k).split(' ').map { it.toInt() }.toIntArray()
        fun floats(k: String) = m.getValue(k).split(' ').map { it.toFloat() }.toFloatArray()
        return Vector(n, ints("ids"), floats("best"), floats("blank"), words, m.getValue("transcript"))
    }

    /** A vocabulary of [VOCAB] lines with the vector's words at their rows (the others are never read). */
    private fun vocab(v: Vector): List<String> = List(VOCAB) { v.words[it] ?: "#$it" }

    private val files = File("src/test/resources/reader").listFiles { f -> f.name.endsWith(".txt") }?.sortedBy { it.name } ?: emptyList()

    @Test
    fun graphOutputsReadAsTheReleaseTranscript() {
        assertTrue("no reader test vectors", files.isNotEmpty())
        for (f in files) {
            val v = load(f)
            val n = v.tokens
            val read = GraphRead(v.ids, v.best, v.blank, 1, VOCAB, 0, n)
            val ids = IntArray(n); val word = IntArray(n); val best = DoubleArray(n); val blank = DoubleArray(n)
            read.labels(ids, word, best, blank)
            val words = Ctc(vocab(v)).feed(ids, n, 0)
            assertEquals(f.name, v.transcript, words.joinToString(" ") { it.text })
            assertEquals(v.ids.toList(), word.toList())
        }
    }

    @Test
    fun throughThePipeline() {
        val prep = java.util.concurrent.Executors.newSingleThreadExecutor()
        val decode = java.util.concurrent.Executors.newSingleThreadExecutor()
        try {
            for (f in files) {
                val v = load(f)
                // A recording of exactly the utterance's tokens: 4 x tokens frames (one 30 s window, all kept).
                val frames = 4 * v.tokens
                val pcm = FloatPcm(FloatArray((frames - 1) * 160 + 161))
                assertEquals(frames, MelFrontEnd.frames(pcm.length))
                val enc = object : EncoderRunner {
                    override val windowSizes = setOf(WindowPlan.FRAMES_30S)
                    override fun run(frames: Int, mel: FloatArray, mask: FloatArray): EncoderOutput =
                        EncoderOutput(FloatArray(frames / 4 * SOFT_DIM), FloatArray(frames / 4) { if (it < v.tokens) 1f else 0f }, 1.0)
                }
                val reader = object : Reader {
                    override val name = "vectors"
                    override val isStub = false
                    override val windowTokens = setOf(750)
                    override fun read(soft: FloatArray, tokens: Int, valid: Int, from: Int, until: Int): Read {
                        assertEquals(750, tokens)
                        assertEquals(v.tokens, valid)
                        return GraphRead(v.ids, v.best, v.blank, 1, VOCAB, from, minOf(until, valid) - from)
                    }
                }
                val r = Pipeline(enc, reader, vocab(v), prep, decode).run(pcm, WindowPlan.FRAMES_30S, 96, object : Pipeline.Listener {})
                assertEquals(f.name, v.transcript, r.words.joinToString(" ") { it.text })
                assertEquals(v.tokens, r.plan.tokens)
            }
        } finally {
            prep.shutdown(); decode.shutdown()
        }
    }

    companion object {
        const val VOCAB = 200_000
    }
}
