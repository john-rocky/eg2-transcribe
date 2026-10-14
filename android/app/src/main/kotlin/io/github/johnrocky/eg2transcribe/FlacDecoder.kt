package io.github.johnrocky.eg2transcribe

import java.io.InputStream

/**
 * A FLAC reader in Kotlin (the format of RFC 9639): STREAMINFO read and the other metadata skipped, then every frame
 * decoded (CONSTANT, VERBATIM, FIXED and LPC subframes, partitioned Rice residuals with 4- or 5-bit parameters and
 * escapes, wasted bits, the three stereo decorrelations) into one Int array per channel. Every frame's header CRC-8 and
 * frame CRC-16 are checked; a frame without the sync code, a CRC that does not match or a stream that ends inside a
 * frame is a [FlacException]. It cannot seek. It is the app's way to read FLAC ([AudioDecoder.Mode.FLAC]): MediaCodec
 * costs about 3 ms of calls per FLAC frame on a Pixel 8a (409 s for a 54 min file, this reader 5 s).
 */
class FlacDecoder(input: InputStream) {
    /** STREAMINFO: block sizes, rate, channels, bits per sample and the total samples per channel (0 = unknown). */
    class Info(val minBlock: Int, val maxBlock: Int, val sampleRate: Int, val channels: Int, val bitsPerSample: Int, val totalSamples: Long)

    class FlacException(message: String) : Exception(message)

    private val r = Bits(input)
    val info: Info = readHeader()

    /** Frames decoded so far. */
    var frames = 0
        private set

    private var residual = IntArray(0)

    /**
     * Decodes the next frame into [out] (one array per channel, grown as needed; the caller keeps the arrays) and returns
     * its samples per channel, or 0 at the end of the stream.
     */
    fun next(out: Array<IntArray>): Int {
        if (!r.skipToByteBoundary()) return 0
        r.startCapture()
        val sync = r.bitsOrEnd(14) ?: return 0
        if (sync != 0x3FFE) throw FlacException("FLAC frame ${frames + 1}: no sync code (0x${Integer.toHexString(sync)})")
        r.bits(1) // reserved
        r.bits(1) // blocking strategy (fixed / variable): the frame says its own size either way
        val bsCode = r.bits(4)
        val srCode = r.bits(4)
        val chCode = r.bits(4)
        val ssCode = r.bits(3)
        r.bits(1) // reserved
        r.utf8() // frame or sample number
        val blockSize = when (bsCode) {
            1 -> 192
            in 2..5 -> 576 shl (bsCode - 2)
            6 -> r.bits(8) + 1
            7 -> r.bits(16) + 1
            in 8..15 -> 256 shl (bsCode - 8)
            else -> throw FlacException("FLAC frame ${frames + 1}: reserved block size code")
        }
        when (srCode) {
            12 -> r.bits(8)
            13, 14 -> r.bits(16)
        }
        val bps = when (ssCode) {
            0 -> info.bitsPerSample
            1 -> 8
            2 -> 12
            4 -> 16
            5 -> 20
            6 -> 24
            7 -> 32
            else -> throw FlacException("FLAC frame ${frames + 1}: reserved sample size code")
        }
        val headerBytes = r.consumed()
        if (r.bits(8) != Crc.crc8(r.captured(), headerBytes)) throw FlacException("FLAC frame ${frames + 1}: the header's CRC-8 does not match")
        val channels = if (chCode < 8) chCode + 1 else if (chCode <= 10) 2 else throw FlacException("FLAC frame ${frames + 1}: reserved channel code")
        if (channels > out.size) throw FlacException("FLAC frame ${frames + 1}: $channels channels, the reader was given ${out.size}")
        for (c in 0 until channels) {
            if (out[c].size < blockSize) out[c] = IntArray(blockSize)
            // The side channel of a stereo pair carries one bit more.
            val extra = when (chCode) {
                8 -> if (c == 1) 1 else 0
                9 -> if (c == 0) 1 else 0
                10 -> if (c == 1) 1 else 0
                else -> 0
            }
            subframe(out[c], blockSize, bps + extra)
        }
        when (chCode) {
            8 -> { val l = out[0]; val s = out[1]; for (i in 0 until blockSize) s[i] = l[i] - s[i] }
            9 -> { val s = out[0]; val rr = out[1]; for (i in 0 until blockSize) s[i] = s[i] + rr[i] }
            10 -> {
                val m = out[0]; val s = out[1]
                for (i in 0 until blockSize) {
                    val mid = (m[i] shl 1) or (s[i] and 1)
                    m[i] = (mid + s[i]) shr 1
                    s[i] = (mid - s[i]) shr 1
                }
            }
        }
        r.skipToByteBoundary()
        val frameBytes = r.consumed()
        if (r.bits(16) != Crc.crc16(r.captured(), frameBytes)) throw FlacException("FLAC frame ${frames + 1}: the frame's CRC-16 does not match")
        frames++
        return blockSize
    }

    /** FLAC's CRCs: CRC-8 (polynomial 0x07) over a frame header, CRC-16 (0x8005) over a whole frame; both start at 0, MSB first. */
    private object Crc {
        private val T8 = IntArray(256) { i -> var c = i; repeat(8) { c = if (c and 0x80 != 0) (c shl 1) xor 0x07 else c shl 1 }; c and 0xff }
        private val T16 = IntArray(256) { i -> var c = i shl 8; repeat(8) { c = if (c and 0x8000 != 0) (c shl 1) xor 0x8005 else c shl 1 }; c and 0xffff }

        fun crc8(b: ByteArray, n: Int): Int {
            var c = 0
            for (i in 0 until n) c = T8[c xor (b[i].toInt() and 0xff)]
            return c
        }

        fun crc16(b: ByteArray, n: Int): Int {
            var c = 0
            for (i in 0 until n) c = ((c shl 8) xor T16[(c ushr 8) xor (b[i].toInt() and 0xff)]) and 0xffff
            return c
        }
    }

    private fun subframe(s: IntArray, n: Int, bitsIn: Int) {
        r.bits(1) // zero padding
        val type = r.bits(6)
        var wasted = 0
        if (r.bits(1) == 1) wasted = r.unary() + 1
        val bps = bitsIn - wasted
        when {
            type == 0 -> { val v = r.sbits(bps); java.util.Arrays.fill(s, 0, n, v) }
            type == 1 -> for (i in 0 until n) s[i] = r.sbits(bps)
            type in 8..12 -> fixed(s, n, bps, type - 8)
            type >= 32 -> lpc(s, n, bps, type - 31)
            else -> throw FlacException("FLAC frame ${frames + 1}: reserved subframe type $type")
        }
        if (wasted > 0) for (i in 0 until n) s[i] = s[i] shl wasted
    }

    private fun fixed(s: IntArray, n: Int, bps: Int, order: Int) {
        for (i in 0 until order) s[i] = r.sbits(bps)
        val e = residuals(n, order)
        when (order) {
            0 -> System.arraycopy(e, 0, s, 0, n)
            1 -> for (i in 1 until n) s[i] = e[i - 1] + s[i - 1]
            2 -> for (i in 2 until n) s[i] = e[i - 2] + 2 * s[i - 1] - s[i - 2]
            3 -> for (i in 3 until n) s[i] = e[i - 3] + 3 * s[i - 1] - 3 * s[i - 2] + s[i - 3]
            4 -> for (i in 4 until n) s[i] = e[i - 4] + 4 * s[i - 1] - 6 * s[i - 2] + 4 * s[i - 3] - s[i - 4]
        }
    }

    private fun lpc(s: IntArray, n: Int, bps: Int, order: Int) {
        for (i in 0 until order) s[i] = r.sbits(bps)
        val precision = r.bits(4) + 1
        if (precision == 16) throw FlacException("FLAC frame ${frames + 1}: invalid LPC precision")
        val shift = r.sbits(5)
        if (shift < 0) throw FlacException("FLAC frame ${frames + 1}: negative LPC shift")
        val coef = IntArray(order) { r.sbits(precision) }
        val e = residuals(n, order)
        for (i in order until n) {
            var sum = 0L
            for (j in 0 until order) sum += coef[j].toLong() * s[i - 1 - j]
            s[i] = e[i - order] + (sum shr shift).toInt()
        }
    }

    /** The residual of a subframe after [order] warm-up samples: n - order values into [residual]. */
    private fun residuals(n: Int, order: Int): IntArray {
        if (residual.size < n) residual = IntArray(n)
        val e = residual
        val method = r.bits(2)
        if (method > 1) throw FlacException("FLAC frame ${frames + 1}: reserved residual coding method")
        val paramBits = if (method == 0) 4 else 5
        val escape = if (method == 0) 15 else 31
        val partOrder = r.bits(4)
        val parts = 1 shl partOrder
        var k = 0
        for (p in 0 until parts) {
            val count = (n shr partOrder) - if (p == 0) order else 0
            val param = r.bits(paramBits)
            if (param == escape) {
                val width = r.bits(5)
                for (i in 0 until count) e[k++] = if (width == 0) 0 else r.sbits(width)
            } else {
                for (i in 0 until count) e[k++] = r.rice(param)
            }
        }
        return e
    }

    private fun readHeader(): Info {
        var b0 = r.bits(8); var b1 = r.bits(8); var b2 = r.bits(8)
        if (b0 == 'I'.code && b1 == 'D'.code && b2 == '3'.code) {
            // An ID3v2 tag in front: version (2), flags (1), size (4 bytes of 7 bits).
            r.bits(16); r.bits(8)
            var size = 0
            repeat(4) { size = (size shl 7) or (r.bits(8) and 0x7f) }
            repeat(size) { r.bits(8) }
            b0 = r.bits(8); b1 = r.bits(8); b2 = r.bits(8)
        }
        val b3 = r.bits(8)
        if (b0 != 'f'.code || b1 != 'L'.code || b2 != 'a'.code || b3 != 'C'.code) throw FlacException("not a FLAC stream")
        var info: Info? = null
        while (true) {
            val last = r.bits(1) == 1
            val type = r.bits(7)
            val length = r.bits(24)
            if (type == 0) {
                val minBlock = r.bits(16); val maxBlock = r.bits(16)
                r.bits(24); r.bits(24) // frame sizes
                val rate = r.bits(20)
                val channels = r.bits(3) + 1
                val bps = r.bits(5) + 1
                val total = (r.bits(4).toLong() shl 32) or (r.bits(32).toLong() and 0xffffffffL)
                repeat(16) { r.bits(8) } // MD5
                repeat(length - 34) { r.bits(8) }
                info = Info(minBlock, maxBlock, rate, channels, bps, total)
            } else {
                repeat(length) { r.bits(8) }
            }
            if (last) break
        }
        return info ?: throw FlacException("FLAC stream without STREAMINFO")
    }

    /**
     * A big-endian bit reader over a stream, refilled 64 KB at a time. From [startCapture] (at a byte boundary) it keeps
     * a copy of every byte of the frame, for the CRCs: [consumed] of them are read, the rest wait in the bit cache.
     */
    private class Bits(private val input: InputStream) {
        private val buf = ByteArray(1 shl 16)
        private var pos = 0
        private var len = 0
        private var cache = 0L
        private var count = 0
        private var cap = ByteArray(1 shl 16)
        private var capLen = 0
        private var capturing = false

        private fun byte(): Int {
            if (pos >= len) {
                len = input.read(buf)
                pos = 0
                if (len <= 0) { len = 0; return -1 }
            }
            val b = buf[pos++].toInt() and 0xff
            if (capturing) {
                if (capLen == cap.size) cap = cap.copyOf(cap.size * 2)
                cap[capLen++] = b.toByte()
            }
            return b
        }

        /** Starts the copy at the current byte boundary: the whole bytes still unread in the cache come first. */
        fun startCapture() {
            capLen = 0
            capturing = true
            val k = count / 8
            for (i in k - 1 downTo 0) cap[capLen++] = ((cache ushr (8 * i)) and 0xff).toByte()
        }

        /** Bytes of the copy that are read (at a byte boundary). */
        fun consumed(): Int = capLen - count / 8

        fun captured(): ByteArray = cap

        /** [n] bits (0..32) as an unsigned value; the end of the stream inside them is an error. */
        fun bits(n: Int): Int {
            if (n == 0) return 0
            while (count < n) {
                val b = byte()
                if (b < 0) throw FlacException("FLAC stream ends inside a frame")
                cache = (cache shl 8) or b.toLong()
                count += 8
            }
            count -= n
            return ((cache ushr count) and ((1L shl n) - 1)).toInt()
        }

        /** Like [bits], but null when the stream ends before the first of them (the end between frames). */
        fun bitsOrEnd(n: Int): Int? {
            if (count == 0) {
                val b = byte()
                if (b < 0) return null
                cache = b.toLong()
                count = 8
            }
            return bits(n)
        }

        fun sbits(n: Int): Int {
            if (n == 0) return 0
            val v = bits(n)
            return (v shl (32 - n)) shr (32 - n)
        }

        /** Zeros before the next 1 bit (the 1 is read too). */
        fun unary(): Int {
            var q = 0
            while (true) {
                if (count == 0) {
                    val b = byte()
                    if (b < 0) throw FlacException("FLAC stream ends inside a frame")
                    cache = b.toLong()
                    count = 8
                }
                val v = cache and ((1L shl count) - 1)
                if (v == 0L) {
                    q += count
                    count = 0
                    continue
                }
                val zeros = java.lang.Long.numberOfLeadingZeros(v) - (64 - count)
                q += zeros
                count -= zeros + 1
                return q
            }
        }

        fun rice(k: Int): Int {
            val q = unary()
            val u = (q shl k) or bits(k)
            return (u ushr 1) xor -(u and 1)
        }

        /** The UTF-8-like coded frame or sample number (up to 7 bytes), skipped. */
        fun utf8() {
            val b = bits(8)
            var more = 0
            var m = 0x80
            while (b and m != 0 && more < 7) { more++; m = m shr 1 }
            repeat(maxOf(0, more - 1)) { bits(8) }
        }

        /** Drops the rest of the current byte; false when the stream has ended (nothing left). */
        fun skipToByteBoundary(): Boolean {
            count -= count % 8
            if (count > 0) return true
            if (pos < len) return true
            val b = byte()
            if (b < 0) return false
            cache = b.toLong()
            count = 8
            return true
        }
    }
}
