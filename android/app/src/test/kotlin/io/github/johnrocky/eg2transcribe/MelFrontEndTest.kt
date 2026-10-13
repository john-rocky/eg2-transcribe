package io.github.johnrocky.eg2transcribe

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Kotlin log-mel against transformers' Gemma4AudioFeatureExtractor on two LibriSpeech test-clean utterances
 * (src/test/resources/mel, written by the lane's make_test_vectors.py from the reader fixtures: the FLAC's own 16-bit
 * samples and features.npy).
 */
class MelFrontEndTest {
    private val dir = File(javaClass.classLoader!!.getResource("mel/vectors.txt")!!.toURI()).parentFile

    private fun pcm(id: String): FloatArray {
        val b = ByteBuffer.wrap(File(dir, "$id.pcm16").readBytes()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(b.remaining()) { b.get(it) / 32768f }
    }

    private fun mel(id: String): FloatArray {
        val b = ByteBuffer.wrap(File(dir, "$id.mel.f32").readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(b.remaining()).also { b.get(it) }
    }

    @Test
    fun matchesTheExtractor() {
        val lines = File(dir, "vectors.txt").readLines().filter { it.isNotBlank() }
        assertEquals(2, lines.size)
        for (line in lines) {
            val (id, samples, frames) = line.split(" ")
            val x = pcm(id)
            assertEquals(samples.toInt(), x.size)
            val n = MelFrontEnd.frames(x.size.toLong())
            assertEquals("frames of $id", frames.toInt(), n)
            val ref = mel(id)
            assertEquals(n * MelFrontEnd.NMEL, ref.size)
            val out = FloatArray(n * MelFrontEnd.NMEL)
            MelFrontEnd().compute(FloatPcm(x), 0, n, out)
            var maxAbs = 0.0
            var sum = 0.0
            for (i in out.indices) {
                val d = Math.abs(out[i].toDouble() - ref[i])
                maxAbs = maxOf(maxAbs, d)
                sum += d
            }
            println("MEL $id samples=${x.size} frames=$n max_abs=${"%.3e".format(maxAbs)} mean_abs=${"%.3e".format(sum / out.size)}")
            assertTrue("$id: max |delta| $maxAbs > 1e-3", maxAbs <= 1e-3)
        }
    }

    @Test
    fun windowByWindowEqualsOnePass() {
        val id = File(dir, "vectors.txt").readLines().first().split(" ")[0]
        val src = FloatPcm(pcm(id))
        val n = MelFrontEnd.frames(src.length)
        val whole = FloatArray(n * MelFrontEnd.NMEL)
        MelFrontEnd().compute(src, 0, n, whole)
        val parts = FloatArray(n * MelFrontEnd.NMEL)
        val front = MelFrontEnd()
        var f = 0
        for (len in listOf(1, 99, 200, n)) {
            val c = minOf(len, n - f)
            if (c <= 0) break
            front.compute(src, f.toLong(), c, parts, f * MelFrontEnd.NMEL)
            f += c
        }
        assertEquals(n, f)
        for (i in whole.indices) assertEquals("value $i", whole[i].toBits(), parts[i].toBits())
    }

    @Test
    fun frameCount() {
        assertEquals(0, MelFrontEnd.frames(0))
        assertEquals(0, MelFrontEnd.frames(160))
        assertEquals(1, MelFrontEnd.frames(161))
        assertEquals(1, MelFrontEnd.frames(320))
        assertEquals(2, MelFrontEnd.frames(321))
        assertEquals(2999, MelFrontEnd.frames(480_000)) // 30 s
        assertEquals(325_744, MelFrontEnd.frames(52_119_092)) // the 54 min fixture
    }
}
