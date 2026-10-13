package io.github.johnrocky.eg2transcribe

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/** 16 kHz mono samples in [-1, 1), read by range. Reads outside [0, length) give zeros. */
interface PcmSource {
    val length: Long

    fun read(start: Long, dst: FloatArray, off: Int, count: Int)

    val seconds: Double get() = length.toDouble() / MelFrontEnd.SAMPLE_RATE
}

/** Samples held in memory (tests, short clips). */
class FloatPcm(private val x: FloatArray) : PcmSource {
    override val length: Long get() = x.size.toLong()

    override fun read(start: Long, dst: FloatArray, off: Int, count: Int) {
        for (i in 0 until count) {
            val p = start + i
            dst[off + i] = if (p in 0 until x.size) x[p.toInt()] else 0f
        }
    }
}

/**
 * A file of 16-bit little-endian mono samples at 16 kHz (what [AudioDecoder] and [Mic] write), memory-mapped: a
 * one-hour recording is 115 MB on disk and does not sit on the Java heap. A sample s reads as s / 32768, the same
 * value soundfile gives for a 16-bit file.
 */
class Pcm16File(val file: File) : PcmSource, Closeable {
    private val raf = RandomAccessFile(file, "r")
    private val maps: List<ShortBuffer>
    override val length: Long = raf.length() / 2

    init {
        // One map per 1 GiB (a MappedByteBuffer is indexed by Int).
        val chunk = 1L shl 30
        val out = ArrayList<ShortBuffer>()
        var pos = 0L
        val bytes = length * 2
        while (pos < bytes) {
            val n = minOf(chunk, bytes - pos)
            out += raf.channel.map(FileChannel.MapMode.READ_ONLY, pos, n).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            pos += n
        }
        maps = out
    }

    override fun read(start: Long, dst: FloatArray, off: Int, count: Int) {
        val perMap = (1L shl 30) / 2
        for (i in 0 until count) {
            val p = start + i
            dst[off + i] = if (p < 0 || p >= length) 0f else maps[(p / perMap).toInt()].get((p % perMap).toInt()) / 32768f
        }
    }

    override fun close() = raf.close()
}

object Loudness {
    /** RMS of the whole source, 1.0 = full scale (one pass in blocks of 64k samples). */
    fun rms(pcm: PcmSource): Double {
        if (pcm.length == 0L) return 0.0
        val buf = FloatArray(1 shl 16)
        var sum = 0.0
        var p = 0L
        while (p < pcm.length) {
            val n = minOf(buf.size.toLong(), pcm.length - p).toInt()
            pcm.read(p, buf, 0, n)
            for (i in 0 until n) sum += buf[i].toDouble() * buf[i]
            p += n
        }
        return sqrt(sum / pcm.length)
    }
}
