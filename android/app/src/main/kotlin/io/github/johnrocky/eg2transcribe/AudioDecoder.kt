package io.github.johnrocky.eg2transcribe

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * An audio file to 16 kHz mono 16-bit PCM in a file ([Pcm16File] reads it). WAV (PCM 8/16/24/32-bit or float) is read
 * directly; anything else (FLAC, MP3, M4A/AAC, Ogg/Opus, ...) goes through MediaExtractor and MediaCodec. Channels are
 * averaged; a rate other than 16 kHz is resampled ([Resampler]). A 16 kHz mono 16-bit file keeps its samples bit for
 * bit (FLAC is lossless, so its samples are the ones soundfile reads on a computer).
 */
object AudioDecoder {
    class Decoded(val samples: Long, val sourceRate: Int, val channels: Int, val format: String, val resampled: Boolean, val ms: Double)

    class DecodeException(text: String, cause: Throwable? = null) : Exception(text, cause)

    fun decode(context: Context, uri: Uri, out: File, cancel: () -> Boolean = { false }, progress: (Double) -> Unit = {}): Decoded {
        val t = System.nanoTime()
        val head = ByteArray(12)
        val n = context.contentResolver.openInputStream(uri)?.use { readFully(it, head) } ?: throw DecodeException("The file could not be opened.")
        val isWav = n == 12 && String(head, 0, 4, Charsets.US_ASCII) == "RIFF" && String(head, 8, 4, Charsets.US_ASCII) == "WAVE"
        val r = if (isWav) decodeWav(context, uri, out, cancel, progress) else decodeMedia(context, uri, out, cancel, progress)
        return Decoded(r.samples, r.sourceRate, r.channels, r.format, r.resampled, (System.nanoTime() - t) / 1e6)
    }

    private fun readFully(s: InputStream, b: ByteArray): Int {
        var n = 0
        while (n < b.size) {
            val k = s.read(b, n, b.size - n)
            if (k < 0) break
            n += k
        }
        return n
    }

    /** Mono float samples at the source rate in, 16 kHz 16-bit samples to the file out. */
    private class Sink(file: File, val sourceRate: Int) : AutoCloseable {
        private val os = BufferedOutputStream(FileOutputStream(file), 1 shl 16)
        private val bytes = ByteBuffer.allocate(4096 * 2).order(ByteOrder.LITTLE_ENDIAN)
        var written = 0L
        val resampled = sourceRate != MelFrontEnd.SAMPLE_RATE
        private val resampler = if (resampled) Resampler(sourceRate, MelFrontEnd.SAMPLE_RATE) { x, n -> write(x, n) } else null
        private var closed = false

        fun push(x: FloatArray, n: Int) = if (resampler != null) resampler.push(x, n) else write(x, n)

        private fun write(x: FloatArray, n: Int) {
            var i = 0
            while (i < n) {
                bytes.clear()
                val m = minOf(4096, n - i)
                for (k in 0 until m) bytes.putShort((x[i + k] * 32768f).roundToInt().coerceIn(-32768, 32767).toShort())
                os.write(bytes.array(), 0, m * 2)
                i += m
            }
            written += n
        }

        /** Flushes the resampler and closes the file; a second call does nothing. */
        override fun close() {
            if (closed) return
            closed = true
            resampler?.finish()
            os.close()
        }
    }

    private class Result(val samples: Long, val sourceRate: Int, val channels: Int, val format: String, val resampled: Boolean)

    private fun decodeWav(context: Context, uri: Uri, out: File, cancel: () -> Boolean, progress: (Double) -> Unit): Result {
        val input = context.contentResolver.openInputStream(uri)?.buffered(1 shl 16) ?: throw DecodeException("The file could not be opened.")
        input.use { s ->
            fun le(b: ByteArray, o: Int, n: Int): Long { var v = 0L; for (k in n - 1 downTo 0) v = (v shl 8) or (b[o + k].toLong() and 0xff); return v }
            val riff = ByteArray(12)
            if (readFully(s, riff) != 12) throw DecodeException("The WAV file is truncated.")
            var fmtTag = 0
            var channels = 0
            var rate = 0
            var bits = 0
            val hdr = ByteArray(8)
            while (true) {
                if (readFully(s, hdr) != 8) throw DecodeException("The WAV file has no data chunk.")
                val id = String(hdr, 0, 4, Charsets.US_ASCII)
                val size = le(hdr, 4, 4)
                if (id == "fmt ") {
                    val f = ByteArray(size.toInt())
                    if (readFully(s, f) != f.size) throw DecodeException("The WAV fmt chunk is truncated.")
                    if (size % 2 == 1L) s.skip(1)
                    fmtTag = le(f, 0, 2).toInt()
                    channels = le(f, 2, 2).toInt()
                    rate = le(f, 4, 4).toInt()
                    bits = le(f, 14, 2).toInt()
                    if (fmtTag == 0xFFFE && f.size >= 26) fmtTag = le(f, 24, 2).toInt() // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID's first 2 bytes
                } else if (id == "data") {
                    if (channels <= 0 || rate <= 0) throw DecodeException("The WAV data chunk comes before its format.")
                    val float = fmtTag == 3
                    if (!(fmtTag == 1 && bits in listOf(8, 16, 24, 32)) && !(float && bits == 32)) throw DecodeException("WAV format $fmtTag with $bits-bit samples is not supported (PCM 8/16/24/32-bit or 32-bit float).")
                    val frameBytes = channels * bits / 8
                    val total = size / frameBytes
                    Sink(out, rate).use { sink ->
                        val chunk = ByteArray(frameBytes * 8192)
                        val mono = FloatArray(8192)
                        var done = 0L
                        while (done < total) {
                            if (cancel()) throw DecodeException("cancelled")
                            val want = (minOf(8192L, total - done) * frameBytes).toInt()
                            val got = readInto(s, chunk, want)
                            val frames = got / frameBytes
                            if (frames == 0) break
                            for (fr in 0 until frames) {
                                var acc = 0f
                                for (c in 0 until channels) {
                                    val o = fr * frameBytes + c * bits / 8
                                    acc += when {
                                        float -> java.lang.Float.intBitsToFloat(le(chunk, o, 4).toInt())
                                        bits == 8 -> ((chunk[o].toInt() and 0xff) - 128) / 128f
                                        bits == 16 -> (le(chunk, o, 2).toInt().toShort()) / 32768f
                                        bits == 24 -> ((le(chunk, o, 3).toInt() shl 8) shr 8) / 8388608f
                                        else -> le(chunk, o, 4).toInt() / 2147483648f
                                    }
                                }
                                mono[fr] = if (channels == 1) acc else acc / channels
                            }
                            sink.push(mono, frames)
                            done += frames
                            progress(done.toDouble() / total)
                            if (got < want) break
                        }
                        sink.close()
                        return Result(sink.written, rate, channels, if (float) "wav float32" else "wav pcm$bits", sink.resampled)
                    }
                } else {
                    s.skip(size + (size % 2))
                }
            }
        }
    }

    private fun readInto(s: InputStream, b: ByteArray, n: Int): Int {
        var k = 0
        while (k < n) {
            val r = s.read(b, k, n - k)
            if (r < 0) break
            k += r
        }
        return k
    }

    private fun decodeMedia(context: Context, uri: Uri, out: File, cancel: () -> Boolean, progress: (Double) -> Unit): Result {
        val ex = MediaExtractor()
        try {
            try {
                ex.setDataSource(context, uri, null)
            } catch (e: Exception) {
                throw DecodeException("This file could not be read as audio (${e.message ?: e.javaClass.simpleName}).", e)
            }
            val track = (0 until ex.trackCount).firstOrNull { ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true }
                ?: throw DecodeException("This file has no audio track.")
            ex.selectTrack(track)
            val inFormat = ex.getTrackFormat(track)
            val mime = inFormat.getString(MediaFormat.KEY_MIME)!!
            val durationUs = if (inFormat.containsKey(MediaFormat.KEY_DURATION)) inFormat.getLong(MediaFormat.KEY_DURATION) else 0L
            val codec = try {
                MediaCodec.createDecoderByType(mime)
            } catch (e: Exception) {
                throw DecodeException("This phone has no decoder for $mime.", e)
            }
            var rate = inFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = inFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var encoding = AudioFormat.ENCODING_PCM_16BIT
            var sink: Sink? = null
            try {
                codec.configure(inFormat, null, null, 0)
                codec.start()
                val info = MediaCodec.BufferInfo()
                var inputDone = false
                var mono = FloatArray(0)
                var lastProgress = -1.0
                // Every free input buffer is filled and every ready output buffer drained before the loop waits (5 ms, only
                // when nothing came out). One buffer in and a 10 ms wait for one buffer out per turn read the 54 min FLAC
                // fixture in 616 s on a Galaxy S26 (5.3 x real time, the app at 23 % of one core).
                var waitUs = 0L
                while (true) {
                    if (cancel()) throw DecodeException("cancelled")
                    while (!inputDone) {
                        val i = codec.dequeueInputBuffer(0)
                        if (i < 0) break
                        val buf = codec.getInputBuffer(i)!!
                        val size = ex.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(i, 0, size, ex.sampleTime, 0)
                            if (durationUs > 0) {
                                val f = ex.sampleTime.toDouble() / durationUs
                                if (f - lastProgress >= 0.005) { progress(f); lastProgress = f }
                            }
                            ex.advance()
                        }
                    }
                    val o = codec.dequeueOutputBuffer(info, waitUs)
                    if (o == MediaCodec.INFO_TRY_AGAIN_LATER) {
                        waitUs = 5_000
                        continue
                    }
                    waitUs = 0
                    if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        val f = codec.outputFormat
                        rate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        continue
                    }
                    if (o < 0) continue
                    val buf = codec.getOutputBuffer(o)!!.order(ByteOrder.LITTLE_ENDIAN)
                    buf.position(info.offset)
                    buf.limit(info.offset + info.size)
                    val s = sink ?: Sink(out, rate).also { sink = it }
                    val bytesPer = when (encoding) {
                        AudioFormat.ENCODING_PCM_FLOAT -> 4
                        AudioFormat.ENCODING_PCM_8BIT -> 1
                        else -> 2
                    }
                    val frames = info.size / (bytesPer * channels)
                    if (mono.size < frames) mono = FloatArray(frames)
                    for (fr in 0 until frames) {
                        var acc = 0f
                        for (c in 0 until channels) acc += when (encoding) {
                            AudioFormat.ENCODING_PCM_FLOAT -> buf.float
                            AudioFormat.ENCODING_PCM_8BIT -> ((buf.get().toInt() and 0xff) - 128) / 128f
                            else -> buf.short / 32768f
                        }
                        mono[fr] = if (channels == 1) acc else acc / channels
                    }
                    s.push(mono, frames)
                    codec.releaseOutputBuffer(o, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
                val s = sink ?: throw DecodeException("The decoder gave no audio for this file.")
                s.close()
                progress(1.0)
                return Result(s.written, rate, channels, mime, s.resampled)
            } finally {
                runCatching { sink?.close() }
                runCatching { codec.stop() }
                codec.release()
            }
        } finally {
            ex.release()
        }
    }
}
