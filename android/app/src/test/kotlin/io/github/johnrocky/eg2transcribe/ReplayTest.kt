package io.github.johnrocky.eg2transcribe

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * A developer check, skipped unless `-Peg2.replay=<dir>` is given: the app's pipeline (Kotlin log-mel, window plan,
 * stub reader, word table, CTC) on real encoder outputs computed on a computer for the same windows, so a run on a
 * computer and this code can be compared word by word without the encoder's numerics in between.
 *
 * Each subdirectory of <dir> holds `pcm16` (16 kHz 16-bit samples), `mel.f32` (the extractor's log-mel rows, float32
 * [frames, 128]), `plan.txt` (line 1: label samples frames window overlap; line 2: the frames of each encoder call)
 * and `soft.f32` / `mask.f32` (each call's encoder outputs, concatenated). The test checks the Kotlin log-mel of every
 * window against `mel.f32` (max |delta| <= 1e-3) and the calls' window sizes, and writes `jvm.json` (the run report
 * fields a comparison reads) with `jvm.tokens.bin` and `jvm.soft.f32` (the app's dump format).
 */
class ReplayTest {
    private val root = System.getProperty("eg2.replay")?.let { File(it) }

    @Test
    fun replay() {
        assumeTrue("set -Peg2.replay=<dir> to run", root != null && root.isDirectory)
        val dirs = root!!.listFiles { f -> File(f, "plan.txt").isFile }!!.sortedBy { it.name }
        assertTrue("no replay inputs in $root", dirs.isNotEmpty())
        val vocab = WordTable.readVocab(File("src/main/assets/stub_vocab.txt"))
        val prep = Executors.newSingleThreadExecutor()
        val decode = Executors.newSingleThreadExecutor()
        val scan = Executors.newFixedThreadPool(4)
        try {
            for (d in dirs) replayOne(d, vocab, prep, decode, scan)
        } finally {
            prep.shutdown(); decode.shutdown(); scan.shutdown()
        }
    }

    private fun floats(raf: RandomAccessFile, count: Int): FloatArray {
        val b = ByteArray(count * 4).also { raf.readFully(it) }
        val f = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(count).also { f.get(it) }
    }

    private fun replayOne(d: File, vocab: List<String>, prep: ExecutorService, decode: ExecutorService, scan: ExecutorService) {
        val lines = File(d, "plan.txt").readLines()
        val (label, samples, frames, window, overlap) = lines[0].split(" ")
        val calls = lines[1].trim().split(" ").map { it.toInt() }
        val plan = WindowPlan(frames.toInt(), window.toInt(), overlap.toInt())
        assertEquals(plan.count, calls.size)
        val melRef = RandomAccessFile(File(d, "mel.f32"), "r")
        val soft = RandomAccessFile(File(d, "soft.f32"), "r")
        val maskRef = RandomAccessFile(File(d, "mask.f32"), "r")
        var maxAbs = 0.0
        val enc = object : EncoderRunner {
            var k = 0
            override val windowSizes = setOf(WindowPlan.FRAMES_10S, WindowPlan.FRAMES_30S)

            override fun run(frames: Int, mel: FloatArray, mask: FloatArray): EncoderOutput {
                assertEquals("window $k frames", calls[k], frames)
                val n = plan.validFrames(k)
                melRef.seek(plan.start(k).toLong() * MelFrontEnd.NMEL * 4)
                val ref = floats(melRef, n * MelFrontEnd.NMEL)
                for (i in ref.indices) maxAbs = maxOf(maxAbs, Math.abs(mel[i].toDouble() - ref[i]))
                for (i in n * MelFrontEnd.NMEL until mel.size) assertEquals(0f, mel[i])
                k++
                return EncoderOutput(floats(soft, frames / 4 * SOFT_DIM), floats(maskRef, frames / 4), 0.0)
            }
        }
        val stem = File(d, "jvm").path
        val tokens = BufferedOutputStream(FileOutputStream("$stem.tokens.bin"), 1 shl 16)
        val softOut = BufferedOutputStream(FileOutputStream("$stem.soft.f32"), 1 shl 20)
        val src = Pcm16File(File(d, "pcm16"))
        val t0 = System.nanoTime()
        val r = try {
            Pipeline(enc, Stub.reader(vocab, scan, 4), vocab, prep, decode).run(src, plan.windowFrames, plan.overlapTokens, object : Pipeline.Listener {}, dump = Pipeline.Dump(softOut, tokens))
        } finally {
            tokens.close(); softOut.close(); src.close(); melRef.close(); soft.close(); maskRef.close()
        }
        val ms = (System.nanoTime() - t0) / 1e6
        assertEquals(samples.toLong(), src.length)
        assertTrue(r.warnings.toString(), r.warnings.isEmpty())
        assertEquals(calls, r.windowFrames)
        val text = r.words.joinToString(" ") { it.text }
        File("$stem.json").writeText(
            "{\"status\": \"ok\", \"input\": {\"label\": \"$label\", \"samples\": $samples}, \"backend\": {\"used\": \"replay\"}, " +
                "\"window\": {\"frames\": ${plan.windowFrames}, \"overlap_tokens\": ${plan.overlapTokens}, \"count\": ${plan.count}, " +
                "\"run\": ${r.encoderMs.size}, \"frames_per_call\": [${r.windowFrames.joinToString(", ")}]}, " +
                "\"tokens\": ${plan.tokens}, \"blanks\": ${r.blanks}, \"words\": ${r.words.size}, \"mel_max_abs\": $maxAbs, " +
                "\"transcript\": \"${text.replace("\\", "\\\\").replace("\"", "\\\"")}\"}\n",
        )
        println("REPLAY ${d.name} label=$label windows=${plan.count} tokens=${plan.tokens} words=${r.words.size} blanks=${r.blanks} mel_max_abs=${"%.3e".format(maxAbs)} ms=${"%.0f".format(ms)}")
        assertTrue("${d.name}: log-mel max |delta| $maxAbs > 1e-3", maxAbs <= 1e-3)
        val cont = File(d, "continuous.f32")
        if (cont.isFile) checkWindowEdges(d.name, plan, File("$stem.soft.f32"), cont)
    }

    /**
     * With `continuous.f32` (the recording encoded as one stream, float32 [tokens, 512]): each window's last kept token
     * and, for windows after the first, its first kept token, against the continuous stream. The last token of a window
     * reads 3 frames ahead; a window cut out of a shorter piece of audio loses them (cos about 0.5 on the Core AI side).
     */
    private fun checkWindowEdges(name: String, plan: WindowPlan, soft: File, cont: File) {
        RandomAccessFile(soft, "r").use { s ->
            RandomAccessFile(cont, "r").use { c ->
                fun row(raf: RandomAccessFile, g: Int): FloatArray {
                    raf.seek(g.toLong() * SOFT_DIM * 4)
                    return floats(raf, SOFT_DIM)
                }
                fun cos(g: Int): Double {
                    val x = row(s, g)
                    val y = row(c, g)
                    var xy = 0.0; var xx = 0.0; var yy = 0.0
                    for (i in 0 until SOFT_DIM) { xy += x[i].toDouble() * y[i]; xx += x[i].toDouble() * x[i]; yy += y[i].toDouble() * y[i] }
                    return xy / Math.sqrt(xx * yy)
                }
                val ends = (0 until plan.count).map { k -> cos(plan.kept(k).last) }
                val firsts = (1 until plan.count).map { k -> cos(plan.kept(k).first) }
                println("REPLAY $name window_end_cos min=${"%.6f".format(ends.min())} mean=${"%.9f".format(ends.average())} (n=${ends.size}) " +
                    "window_first_kept_cos min=${"%.6f".format(firsts.minOrNull() ?: 1.0)} (n=${firsts.size})")
                assertTrue("$name: a window's last token against the continuous stream, cos ${ends.min()} < 0.99", ends.min() >= 0.99)
            }
        }
    }

}
