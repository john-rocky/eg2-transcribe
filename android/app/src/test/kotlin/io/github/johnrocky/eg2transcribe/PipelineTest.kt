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
 * (s / 4 + p) % 997 + 1, except the tokens a window must not keep, which read as row 0: the ones inside the left
 * overlap and, in every window but the last, the dropped last token. The stitched stream must be every global token
 * once, in order, and never row 0.
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

    private inner class FakeEncoder(val plan: WindowPlan, override val windowSizes: Set<Int>) : EncoderRunner {
        var calls = 0
        val frames = ArrayList<Int>()

        override fun run(frames: Int, mel: FloatArray, mask: FloatArray): EncoderOutput {
            assertEquals(frames * MelFrontEnd.NMEL, mel.size)
            assertEquals(frames, mask.size)
            this.frames += frames
            val k = calls++
            val start = plan.start(k)
            val valid = mask.count { it > 0.5f }
            assertEquals(plan.validFrames(k), valid)
            val tokens = WindowPlan.tokensFor(valid)
            val soft = FloatArray(frames / 4 * SOFT_DIM)
            val outMask = FloatArray(frames / 4)
            val lastWindow = k == plan.count - 1
            for (p in 0 until tokens) {
                val notKept = (k > 0 && p < plan.overlapTokens) || (!lastWindow && p >= plan.windowTokens - plan.dropLast)
                val row = if (notKept) 0 else (start / 4 + p) % 997 + 1
                table.row(row).forEachIndexed { i, x -> soft[p * SOFT_DIM + i] = 30f * x }
                outMask[p] = 1f
            }
            return EncoderOutput(soft, outMask, 1.0)
        }
    }

    private fun records(bytes: ByteArray): List<Pair<Int, Int>> {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = ArrayList<Pair<Int, Int>>()
        while (b.remaining() >= Pipeline.TOKEN_RECORD_BYTES) {
            val g = b.int
            val label = b.int
            b.int; b.double; b.double
            out += g to label
        }
        return out
    }

    private fun stub() = Stub.reader(vocab, scan, 2)

    @Test
    fun stitchesEveryTokenOnceAndUsesTheShortSignatureLast() {
        // 5,732 frames = 1,433 tokens: windows at tokens 0, 653 and 1,306; the last has 508 real frames and runs on the
        // 10 s signature.
        val samples = (5732 - 1) * 160 + 161
        val pcm = FloatPcm(FloatArray(samples))
        assertEquals(5732, MelFrontEnd.frames(pcm.length))
        val plan = WindowPlan(5732, 3000, 96)
        val enc = FakeEncoder(plan, setOf(1000, 3000))
        val windows = ArrayList<Int>()
        val tokens = ByteArrayOutputStream()
        val soft = ByteArrayOutputStream()
        val r = Pipeline(enc, stub(), vocab, prep, decode).run(pcm, 3000, 96, object : Pipeline.Listener {
            override fun onWindow(k: Int, count: Int, audioSeconds: Double, elapsedMs: Double) { windows += k }
        }, dump = Pipeline.Dump(soft, tokens))
        assertEquals(listOf(0, 1, 2), windows)
        assertEquals(listOf(3000, 3000, 1000), enc.frames)
        assertEquals(listOf(3000, 3000, 1000), r.windowFrames)
        assertEquals(3, r.readerMs.size)
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
        val enc = FakeEncoder(WindowPlan(5732, 3000, 96), setOf(3000))
        val r = Pipeline(enc, stub(), vocab, prep, decode).run(pcm, 3000, 96, object : Pipeline.Listener {})
        assertEquals(listOf(3000, 3000, 3000), r.windowFrames)
        assertEquals(1433, r.words.size)
    }

    @Test
    fun cancelStopsBetweenWindows() {
        val pcm = FloatPcm(FloatArray((5732 - 1) * 160 + 161))
        val enc = FakeEncoder(WindowPlan(5732, 3000, 96), setOf(1000, 3000))
        var stop = false
        val r = Pipeline(enc, stub(), vocab, prep, decode).run(pcm, 3000, 96, object : Pipeline.Listener {
            override fun onWindow(k: Int, count: Int, audioSeconds: Double, elapsedMs: Double) { stop = true }
        }, cancel = { stop })
        assertTrue(r.cancelled)
        assertEquals(1, enc.calls)
        assertEquals(749, r.words.size)
    }

    @Test
    fun keepsTheLastValidTokenOfTheRecording() {
        // 30.000 s = 2,999 frames: 750 tokens, the last one with its missing frame masked; one window keeps all of them.
        val pcm = FloatPcm(FloatArray(480_000))
        val enc = FakeEncoder(WindowPlan(2999, 3000, 96), setOf(1000, 3000))
        val tokens = ByteArrayOutputStream()
        val r = Pipeline(enc, stub(), vocab, prep, decode).run(pcm, 3000, 96, object : Pipeline.Listener {}, dump = Pipeline.Dump(null, tokens))
        assertTrue(r.warnings.toString(), r.warnings.isEmpty())
        assertEquals(750, r.plan.tokens)
        assertEquals((0 until 750).toList(), records(tokens.toByteArray()).map { it.first })
        assertEquals(750, r.words.size)
    }

    @Test
    fun tenSecondWindows() {
        // 10 s windows with the default overlap: step 153 tokens (612 frames).
        val pcm = FloatPcm(FloatArray((2600 - 1) * 160 + 161))
        val plan = WindowPlan(2600, 1000, 96)
        val enc = FakeEncoder(plan, setOf(1000, 3000))
        val r = Pipeline(enc, stub(), vocab, prep, decode).run(pcm, 1000, 96, object : Pipeline.Listener {})
        assertEquals(plan.count, enc.calls)
        assertEquals(4, plan.count)
        assertTrue(r.windowFrames.all { it == 1000 })
        assertEquals(650, r.words.size)
    }

    @Test
    fun reportsEachCallsWriteRunAndRead() {
        // The encoder's and the reader's write / run / read times, one value per window, for the run report.
        val pcm = FloatPcm(FloatArray((5732 - 1) * 160 + 161))
        val plan = WindowPlan(5732, 3000, 96)
        val inner = FakeEncoder(plan, setOf(3000))
        val enc = object : EncoderRunner {
            override val windowSizes = inner.windowSizes
            override fun run(frames: Int, mel: FloatArray, mask: FloatArray): EncoderOutput {
                val o = inner.run(frames, mel, mask)
                return EncoderOutput(o.soft, o.mask, 4.0, 1.0, 2.0, 1.0)
            }
        }
        val reader = object : Reader {
            override val name = "fake"
            override val isStub = false
            override val windowTokens = setOf(750)
            override fun read(soft: FloatArray, tokens: Int, valid: Int, from: Int, until: Int): Read =
                GraphRead(IntArray(tokens) { 1 }, FloatArray(tokens) { 10f }, FloatArray(tokens) { 9f }, 1, 1000, from, minOf(until, valid) - from)
                    .also { it.writeMs = 0.5; it.runMs = 0.25; it.readMs = 0.75 }
        }
        val r = Pipeline(enc, reader, vocab, prep, decode).run(pcm, 3000, 96, object : Pipeline.Listener {})
        assertEquals(listOf(1.0, 1.0, 1.0), r.calls.encoderWriteMs)
        assertEquals(listOf(2.0, 2.0, 2.0), r.calls.encoderRunMs)
        assertEquals(listOf(1.0, 1.0, 1.0), r.calls.encoderReadMs)
        assertEquals(listOf(0.5, 0.5, 0.5), r.calls.readerWriteMs)
        assertEquals(listOf(0.25, 0.25, 0.25), r.calls.readerRunMs)
        assertEquals(listOf(0.75, 0.75, 0.75), r.calls.readerReadMs)
        assertEquals(listOf("encoder_write_ms", "encoder_run_ms", "encoder_read_ms", "reader_write_ms", "reader_run_ms", "reader_read_ms"), r.calls.all().keys.toList())
        assertEquals(listOf(4.0, 4.0, 4.0), r.encoderMs)
    }

    @Test
    fun aGraphReaderIsCalledOnTheWholeWindowAndKeepsTheKeptRange() {
        // A fake trained reader: it sees every window whole (750 tokens, the valid count), and its outputs give token
        // p of window k the word (firstToken(k) + p) % 5 + 1, blank where that is 3.
        val pcm = FloatPcm(FloatArray((5732 - 1) * 160 + 161))
        val plan = WindowPlan(5732, 3000, 96)
        val enc = FakeEncoder(plan, setOf(3000))
        val seen = ArrayList<Pair<Int, Int>>()
        val reader = object : Reader {
            override val name = "fake"
            override val isStub = false
            override val windowTokens = setOf(750)
            var k = 0
            override fun read(soft: FloatArray, tokens: Int, valid: Int, from: Int, until: Int): Read {
                seen += tokens to valid
                val g0 = plan.firstToken(k++)
                val ids = IntArray(tokens) { (g0 + it) % 5 + 1 }
                val best = FloatArray(tokens) { 10f }
                val blank = FloatArray(tokens) { if ((g0 + it) % 5 + 1 == 3) 11f else 9f }
                return GraphRead(ids, best, blank, 1, 1000, from, minOf(until, valid) - from)
            }
        }
        val tokens = ByteArrayOutputStream()
        val r = Pipeline(enc, reader, vocab, prep, decode).run(pcm, 3000, 96, object : Pipeline.Listener {}, dump = Pipeline.Dump(null, tokens))
        assertEquals(listOf(750 to 750, 750 to 750, 750 to 127), seen)
        val rec = records(tokens.toByteArray())
        assertEquals((0 until 1433).toList(), rec.map { it.first })
        assertEquals((0 until 1433).map { g -> (g % 5 + 1).let { if (it == 3) Ctc.BLANK else it } }, rec.map { it.second })
        assertEquals(1433 / 5 * 4 + (1433 % 5).let { if (it >= 3) it - 1 else it }, r.words.size)
    }
}
