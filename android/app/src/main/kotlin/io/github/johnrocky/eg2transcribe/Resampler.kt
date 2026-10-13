package io.github.johnrocky.eg2transcribe

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin

/**
 * Streaming band-limited resampler (windowed sinc, Blackman window, 16 zero crossings, cutoff at 95 % of the lower
 * Nyquist rate) for audio files that are not 16 kHz. Positions are kept as exact fractions of the two rates, so a
 * one-hour file does not drift. A 16 kHz file never goes through it ([AudioDecoder] copies its samples as they are).
 */
class Resampler(private val srcRate: Int, private val dstRate: Int, private val emit: (FloatArray, Int) -> Unit) {
    private val cutoff = 0.95 * minOf(1.0, dstRate.toDouble() / srcRate) // of the source Nyquist rate
    private val halfWidth = ZERO_CROSSINGS / cutoff // in source samples
    private val table: FloatArray = FloatArray((halfWidth * OVERSAMPLE).toInt() + 2) { k ->
        val d = k.toDouble() / OVERSAMPLE
        if (d >= halfWidth) 0f else {
            val x = PI * cutoff * d
            val sinc = if (d == 0.0) 1.0 else sin(x) / x
            val w = 0.42 + 0.5 * cos(PI * d / halfWidth) + 0.08 * cos(2 * PI * d / halfWidth)
            (cutoff * sinc * w).toFloat()
        }
    }
    private var buf = FloatArray(1 shl 16)
    private var bufStart = 0L // source index of buf[0]
    private var bufLen = 0
    private var outIndex = 0L // next output sample
    private val out = FloatArray(4096)
    private var outLen = 0

    fun push(x: FloatArray, n: Int) {
        if (bufLen + n > buf.size) compact()
        if (bufLen + n > buf.size) buf = buf.copyOf(maxOf(buf.size * 2, bufLen + n))
        System.arraycopy(x, 0, buf, bufLen, n)
        bufLen += n
        produce(final = false)
    }

    fun finish() {
        produce(final = true)
        if (outLen > 0) emit(out, outLen)
        outLen = 0
    }

    private fun compact() {
        // Keep the samples the next outputs still need.
        val need = floor(position(outIndex) - halfWidth).toLong() - 1
        val drop = (need - bufStart).coerceIn(0L, bufLen.toLong()).toInt()
        if (drop > 0) {
            System.arraycopy(buf, drop, buf, 0, bufLen - drop)
            bufLen -= drop
            bufStart += drop
        }
    }

    /** Source position of output sample m (exact numerator / denominator, then a double). */
    private fun position(m: Long): Double = (m * srcRate / dstRate) + ((m * srcRate) % dstRate).toDouble() / dstRate

    private fun produce(final: Boolean) {
        val end = bufStart + bufLen
        while (true) {
            val p = position(outIndex)
            val hi = floor(p + halfWidth).toLong()
            if (!final && hi >= end) return
            if (final && floor(p).toLong() >= end) return
            val lo = floor(p - halfWidth).toLong() + 1
            var acc = 0.0
            var i = maxOf(lo, bufStart)
            val last = minOf(hi, end - 1)
            while (i <= last) {
                val d = Math.abs(p - i) * OVERSAMPLE
                val k = d.toInt()
                if (k + 1 < table.size) {
                    val f = d - k
                    val h = table[k] + (table[k + 1] - table[k]) * f
                    acc += buf[(i - bufStart).toInt()] * h
                }
                i++
            }
            out[outLen++] = acc.toFloat()
            if (outLen == out.size) {
                emit(out, outLen)
                outLen = 0
            }
            outIndex++
        }
    }

    companion object {
        private const val ZERO_CROSSINGS = 16.0
        private const val OVERSAMPLE = 512
    }
}
