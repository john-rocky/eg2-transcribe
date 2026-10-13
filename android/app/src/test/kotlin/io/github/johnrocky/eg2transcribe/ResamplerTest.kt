package io.github.johnrocky.eg2transcribe

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ResamplerTest {
    private fun resample(x: FloatArray, src: Int, block: Int = 4096): FloatArray {
        val out = ArrayList<Float>()
        val r = Resampler(src, 16000) { y, n -> for (i in 0 until n) out += y[i] }
        var i = 0
        while (i < x.size) {
            val n = minOf(block, x.size - i)
            r.push(x.copyOfRange(i, i + n), n)
            i += n
        }
        r.finish()
        return out.toFloatArray()
    }

    private fun tone(hz: Double, rate: Int, seconds: Double, amp: Double = 0.5) =
        FloatArray((rate * seconds).toInt()) { (amp * sin(2 * PI * hz * it / rate)).toFloat() }

    private fun rms(x: FloatArray, from: Int, to: Int): Double {
        var s = 0.0
        for (i in from until to) s += x[i].toDouble() * x[i]
        return sqrt(s / (to - from))
    }

    @Test
    fun keepsSpeechBandTonesAndLength() {
        for (src in listOf(48000, 44100, 22050, 8000)) {
            val x = tone(1000.0, src, 2.0)
            val y = resample(x, src, block = 1000 + src % 977)
            val expected = Math.ceil(x.size * 16000.0 / src).toInt()
            assertTrue("$src: ${y.size} samples, expected about $expected", abs(y.size - expected) <= 1)
            // Away from the edges the 1 kHz tone keeps its level and matches the ideal 16 kHz tone sample by sample.
            val a = 2000; val b = y.size - 2000
            assertEquals("$src: rms", 0.5 / sqrt(2.0), rms(y, a, b), 0.005)
            var maxErr = 0.0
            for (i in a until b) maxErr = maxOf(maxErr, abs(y[i] - 0.5 * sin(2 * PI * 1000.0 * i / 16000)))
            assertTrue("$src: max error $maxErr", maxErr < 0.01)
        }
    }

    @Test
    fun removesWhatAbove8kHzWouldFoldBack() {
        for (src in listOf(48000, 44100)) {
            val y = resample(tone(12000.0, src, 1.0), src)
            val level = 20 * log10(rms(y, 1000, y.size - 1000) / (0.5 / sqrt(2.0)))
            assertTrue("$src: a 12 kHz tone comes out at $level dB", level < -40)
        }
    }
}
