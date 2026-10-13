package io.github.johnrocky.eg2transcribe

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The log-mel front-end of the EmbeddingGemma 2 audio encoder, written from transformers 5.19.0
 * `Gemma4AudioFeatureExtractor` with google/embeddinggemma-2's preprocessor_config.json:
 *
 * - 16 kHz mono samples in [-1, 1), 160 zero samples in front ("semicausal" padding);
 * - frame i = samples [160 i - 160, 160 i + 160), times a periodic Hann window of 320;
 * - |rfft| over 512 points (the magnitude, not the power);
 * - 128 triangular HTK mel filters from 0 to 8000 Hz, no area normalization;
 * - log(mel + 0.001). No pre-emphasis, no dither, no per-bin mean or deviation.
 *
 * A frame is valid when sample 160 i + 160 exists, so n samples give floor((n - 161) / 160) + 1 frames ([frames]).
 * The extractor computes every frame from the samples alone, so frames of a long recording can be computed window
 * by window ([compute]) and match the whole-recording rows exactly. One instance is not thread-safe (scratch
 * buffers); use one per thread.
 */
class MelFrontEnd {
    companion object {
        const val SAMPLE_RATE = 16000
        const val FRAME = 320
        const val HOP = 160
        const val NFFT = 512
        const val NBIN = NFFT / 2 + 1
        const val NMEL = 128
        const val MEL_FLOOR = 1e-3
        const val MIN_HZ = 0.0
        const val MAX_HZ = 8000.0

        /** Valid frames for [samples] samples: floor((n - 161) / 160) + 1, and 0 below 161 samples. */
        fun frames(samples: Long): Int = if (samples < FRAME / 2 + 1) 0 else (Math.floorDiv(samples - (FRAME / 2 + 1), HOP.toLong()) + 1).toInt()

        /** `window_function(320)` of transformers (periodic Hann, `np.hanning(321)[:-1]`), cast to float32 as the extractor does. */
        fun window(): FloatArray = FloatArray(FRAME) { n -> (0.5 - 0.5 * cos(2.0 * PI * n / FRAME)).toFloat() }

        private fun hzToMel(f: Double) = 2595.0 * log10(1.0 + f / 700.0)
        private fun melToHz(m: Double) = 700.0 * (10.0.pow(m / 2595.0) - 1.0)

        /**
         * `mel_filter_bank(257, 128, 0, 8000, 16000, norm=None, mel_scale="htk")` of transformers, as [NMEL][NBIN]
         * row-major (the transpose of its [NBIN][NMEL] matrix). Corner frequencies come from `np.linspace` over the mel
         * range (start + i * step, the last one set to the end) and the bin frequencies are i * 8000 / 256.
         */
        fun melFilters(): DoubleArray {
            val melMin = hzToMel(MIN_HZ)
            val melMax = hzToMel(MAX_HZ)
            val n = NMEL + 2
            val step = (melMax - melMin) / (n - 1)
            val f = DoubleArray(n) { i -> melToHz(if (i == n - 1) melMax else i * step + melMin) }
            val binHz = (SAMPLE_RATE / 2).toDouble() / (NBIN - 1)
            val out = DoubleArray(NMEL * NBIN)
            for (m in 0 until NMEL) {
                for (k in 0 until NBIN) {
                    val hz = k * binHz
                    val down = (hz - f[m]) / (f[m + 1] - f[m])
                    val up = (f[m + 2] - hz) / (f[m + 2] - f[m + 1])
                    out[m * NBIN + k] = max(0.0, min(down, up))
                }
            }
            return out
        }
    }

    private val win = window()
    private val filters = melFilters()

    // Non-zero span of each filter, so a frame costs about two passes over the bins instead of 128.
    private val firstBin = IntArray(NMEL)
    private val lastBin = IntArray(NMEL)

    init {
        for (m in 0 until NMEL) {
            var a = NBIN
            var b = -1
            for (k in 0 until NBIN) if (filters[m * NBIN + k] > 0.0) {
                if (a == NBIN) a = k
                b = k
            }
            firstBin[m] = a
            lastBin[m] = b // b < a for a filter with no bin (the top one: it falls between two bins)
        }
    }

    // A 512-point real FFT as a 256-point complex FFT of the packed even/odd samples.
    private val half = NFFT / 2
    private val bitrev = IntArray(half).also { r ->
        val bits = Integer.numberOfTrailingZeros(half)
        for (i in 0 until half) r[i] = Integer.reverse(i) ushr (32 - bits)
    }
    private val cosHalf = DoubleArray(half / 2) { cos(2.0 * PI * it / half) }
    private val sinHalf = DoubleArray(half / 2) { sin(2.0 * PI * it / half) }
    private val cosFull = DoubleArray(half + 1) { cos(2.0 * PI * it / NFFT) }
    private val sinFull = DoubleArray(half + 1) { sin(2.0 * PI * it / NFFT) }
    private val re = DoubleArray(half)
    private val im = DoubleArray(half)
    private val mag = DoubleArray(NBIN)
    private var samples = FloatArray(0)

    /**
     * Frames [first, first + count) of [pcm] into [out] from [outOffset], [count][NMEL] row-major. Samples before 0
     * read as 0 (the semicausal padding); a frame past the valid ones also reads zeros past the end (the encoder
     * masks such rows anyway).
     */
    fun compute(pcm: PcmSource, first: Long, count: Int, out: FloatArray, outOffset: Int = 0) {
        if (count <= 0) return
        val start = first * HOP - HOP // first sample of frame `first`
        val need = (count - 1) * HOP + FRAME
        if (samples.size < need) samples = FloatArray(need)
        pcm.read(start, samples, 0, need)
        for (t in 0 until count) frame(samples, t * HOP, out, outOffset + t * NMEL)
    }

    /** One frame from x[off until off + 320] into out[o until o + 128]. */
    private fun frame(x: FloatArray, off: Int, out: FloatArray, o: Int) {
        // z[n] = x[2n] + i x[2n + 1], windowed in float32 like the extractor (frames * window, both float32).
        for (n in 0 until half) {
            val a = 2 * n
            if (a + 1 < FRAME) {
                re[bitrev[n]] = (x[off + a] * win[a]).toDouble()
                im[bitrev[n]] = (x[off + a + 1] * win[a + 1]).toDouble()
            } else {
                re[bitrev[n]] = 0.0
                im[bitrev[n]] = 0.0
            }
        }
        fft256()
        // Split: X[k] = E[k] + W^k O[k], with E, O the transforms of the even and odd samples.
        for (k in 0..half) {
            val k1 = k % half
            val k2 = (half - k) % half
            val zr = re[k1]
            val zi = im[k1]
            val cr = re[k2]
            val ci = -im[k2]
            val er = 0.5 * (zr + cr)
            val ei = 0.5 * (zi + ci)
            val dr = 0.5 * (zr - cr)
            val di = 0.5 * (zi - ci)
            // O[k] = (Z[k] - conj Z[N-k]) / 2i = (di, -dr)
            val or = di
            val oi = -dr
            val wr = cosFull[k]
            val wi = -sinFull[k]
            val xr = er + (wr * or - wi * oi)
            val xi = ei + (wr * oi + wi * or)
            mag[k] = sqrt(xr * xr + xi * xi)
        }
        for (m in 0 until NMEL) {
            var acc = 0.0
            val base = m * NBIN
            for (k in firstBin[m]..lastBin[m]) acc += mag[k] * filters[base + k]
            out[o + m] = ln(acc + MEL_FLOOR).toFloat()
        }
    }

    /** In-place iterative radix-2 FFT of (re, im), already in bit-reversed order. */
    private fun fft256() {
        var len = 2
        while (len <= half) {
            val h = len shr 1
            val step = half / len
            var i = 0
            while (i < half) {
                var t = 0
                for (j in i until i + h) {
                    val wr = cosHalf[t]
                    val wi = -sinHalf[t]
                    val xr = re[j + h] * wr - im[j + h] * wi
                    val xi = re[j + h] * wi + im[j + h] * wr
                    re[j + h] = re[j] - xr
                    im[j + h] = im[j] - xi
                    re[j] += xr
                    im[j] += xi
                    t += step
                }
                i += len
            }
            len = len shl 1
        }
    }
}
