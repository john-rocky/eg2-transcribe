package io.github.johnrocky.eg2transcribe

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The window loop with a fake encoder whose token p of the window starting at frame s reads as table row
 * (s / 4 + p) % 997 + 1, except the tokens inside the left overlap, which read as row 0: the stitched stream must be
 * every global token once, in order, and never row 0.
 */
class PipelineTest {
    private val vocab = WordTable.readVocab(File("src/main/assets/stub_vocab.txt"))
    private val table = Stub.table(vocab)
    private val prep = Executors.newSingleThreadExecutor()
    private val decode = Executors.newSingleThreadExecutor()
    private val scan = Executors.newFixedThreadPool(2)

    @After
    fun stop() {
        prep.shutdown(); decode.shutdown(); scan.shutdown()
    }

    private inner class FakeEncoder(val stride: Int, val overlap: Int, override val windowSizes: Set<Int>) : EncoderRunner {
        var calls = 0
        val frames = ArrayList<Int>()

        override fun run(frames: Int, mel: FloatArray, mask: FloatArray): EncoderOutput {
            assertEquals(frames * MelFrontEnd.NMEL, mel.size)
            assertEquals(frames, mask.size)
            this.frames += frames
            val k = calls++
            val start = k * stride
            val valid = mask.count { it > 0.5f }
            val tokens = WindowPlan.tokensFor(valid)
            val soft = FloatArray(frames / 4 * SOFT_DIM)
            val outMask = FloatArray(frames / 4)
            for (p in 0 until tokens) {
                val row = if (k > 0 && p < overlap) 0 else (start / 4 + p) % 997 + 1
                table.row(row).forEachIndexed { i, x -> soft[p * SOFT_DIM + i] = 30f * x }
                outMask[p] = 1f
            }
            return EncoderOutput(soft, outMask, 1.0)
        }
    }

    private fun records(bytes: ByteArray): List<Pair<Int, Int>> {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = ArrayList<Pair<Int, Int>>()
        while (b.remaining() >= 32) {
            val g = b.int
            val id = b.int
            b.double; b.double; b.double
            out += g to id
        }
        return out
    }

    @Test
    fun stitchesEveryTokenOnceAndUsesTheShortSignatureLast() {
        // 5,732 frames: windows at 0, 2,616 and 5,232; the last has 500 real frames and runs on the 10 s signature.
        val samples = (5732 - 1) * 160 + 161
        val pcm = FloatPcm(FloatArray(samples))
        assertEquals(5732, MelFrontEnd.frames(pcm.length))
        val enc = FakeEncoder(2616, 96, setOf(1000, 3000))
        val windows = ArrayList<Int>()
        val tokens = ByteArrayOutputStream()
        val soft = ByteArrayOutputStream()
        val r = Pipeline(enc, Stub.reader(), table, prep, decode, scan, 2).run(pcm, 3000, 96, object : Pipeline.Listener {
            override fun onWindow(k: Int, count: Int, audioSeconds: Double, elapsedMs: Double) { windows += k }
        }, dump = Pipeline.Dump(soft, tokens))
        assertEquals(listOf(0, 1, 2), windows)
        assertEquals(listOf(3000, 3000, 1000), enc.frames)
        assertEquals(listOf(3000, 3000, 1000), r.windowFrames)
        assertEquals(1433, r.plan.tokens)
        assertTrue(r.warnings.toString(), r.warnings.isEmpty())
        val rec = records(tokens.toByteArray())
        assertEquals((0 until 1433).toList(), rec.map { it.first })
        assertEquals((0 until 1433).map { it % 997 + 1 }, rec.map { it.second })
        assertEquals(1433 * SOFT_DIM * 4, soft.size())
        assertEquals(1433, r.words.size)
        assertEquals(vocab[1], r.words[0].text)
        assertEquals(1432, r.words.last().firstToken)
        assertEquals(0, r.blanks)
    }

    @Test
    fun withoutA10sSignatureTheLastWindowIsPadded() {
        val pcm = FloatPcm(FloatArray((5732 - 1) * 160 + 161))
        val enc = FakeEncoder(2616, 96, setOf(3000))
        val r = Pipeline(enc, Stub.reader(), table, prep, decode, null, 1).run(pcm, 3000, 96, object : Pipeline.Listener {})
        assertEquals(listOf(3000, 3000, 3000), r.windowFrames)
        assertEquals(1433, r.words.size)
    }

    @Test
    fun cancelStopsBetweenWindows() {
        val pcm = FloatPcm(FloatArray((5732 - 1) * 160 + 161))
        val enc = FakeEncoder(2616, 96, setOf(1000, 3000))
        var stop = false
        val r = Pipeline(enc, Stub.reader(), table, prep, decode, scan, 2).run(pcm, 3000, 96, object : Pipeline.Listener {
            override fun onWindow(k: Int, count: Int, audioSeconds: Double, elapsedMs: Double) { stop = true }
        }, cancel = { stop })
        assertTrue(r.cancelled)
        assertEquals(1, enc.calls)
        assertEquals(750, r.words.size)
    }

    @Test
    fun dropsTheLastTokenWhenItsFramesRunOut() {
        // 30.000 s = 2,999 frames: the encoder marks 750 tokens valid, token 749 lacks frame 2,999 and is not kept.
        val pcm = FloatPcm(FloatArray(480_000))
        val enc = FakeEncoder(2616, 96, setOf(1000, 3000))
        val tokens = ByteArrayOutputStream()
        val r = Pipeline(enc, Stub.reader(), table, prep, decode, scan, 2).run(pcm, 3000, 96, object : Pipeline.Listener {}, dump = Pipeline.Dump(null, tokens))
        assertTrue(r.warnings.toString(), r.warnings.isEmpty())
        assertEquals(749, r.plan.tokens)
        assertEquals((0 until 749).toList(), records(tokens.toByteArray()).map { it.first })
        assertEquals(749, r.words.size)
    }

    @Test
    fun tenSecondWindows() {
        // 10 s windows with the default overlap: stride 616 frames.
        val pcm = FloatPcm(FloatArray((2600 - 1) * 160 + 161))
        val enc = FakeEncoder(616, 96, setOf(1000, 3000))
        val r = Pipeline(enc, Stub.reader(), table, prep, decode, scan, 2).run(pcm, 1000, 96, object : Pipeline.Listener {})
        assertEquals(WindowPlan(2600, 1000, 96).count, enc.calls)
        assertTrue(r.windowFrames.all { it == 1000 })
        assertEquals(650, r.words.size)
    }
}
