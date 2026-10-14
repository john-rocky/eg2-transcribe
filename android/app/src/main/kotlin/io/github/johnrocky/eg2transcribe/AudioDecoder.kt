package io.github.johnrocky.eg2transcribe

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.roundToInt
import org.json.JSONObject

/**
 * An audio file to 16 kHz mono 16-bit PCM in a file ([Pcm16File] reads it). WAV (PCM 8/16/24/32-bit or float) is read
 * directly; anything else (FLAC, MP3, M4A/AAC, Ogg/Opus, ...) goes through MediaExtractor and MediaCodec. Channels are
 * averaged; a rate other than 16 kHz is resampled ([Resampler]). A 16 kHz mono 16-bit file keeps its samples bit for
 * bit (FLAC is lossless, so its samples are the ones soundfile reads on a computer).
 */
object AudioDecoder {
    /**
     * [expectedSamples]: the length the file declares (WAV: its data size; others: the track duration), at 16 kHz, or
     * null when it declares none. A decoder that returns fewer samples shows up as the difference (the Galaxy S26's
     * FLAC decoder returned the 54 min fixture 998 samples short in an earlier build: the file's last 3 FLAC frames,
     * which are 402 + 402 + 194 samples long). [stats]: where the time went ([Stats.json]).
     */
    class Decoded(val samples: Long, val sourceRate: Int, val channels: Int, val format: String, val resampled: Boolean, val ms: Double, val expectedSamples: Long?, val stats: JSONObject)

    class DecodeException(text: String, cause: Throwable? = null) : Exception(text, cause)

    /**
     * How a file is read (launch extra `decode_mode`). [FLAC], the default: a FLAC file with [FlacDecoder], anything else
     * with MediaCodec's [SYNC] loop. [SYNC], [ASYNC] and [LEGACY] send every compressed file through MediaCodec (kept to
     * measure against). A WAV file is always read directly.
     */
    enum class Mode(val id: String, val label: String) {
        /** One thread: every free input buffer filled, every ready output buffer drained, a wait (10 ms) only when neither moved. */
        SYNC("sync", "sync loop"),

        /** MediaCodec's callbacks on a thread of their own: a buffer is filled or drained the moment the codec offers it. */
        ASYNC("async", "async callbacks"),

        /** The loop of the builds before this one: free inputs filled, then one output buffer a turn, a 5 ms wait after an empty poll. */
        LEGACY("legacy", "earlier loop"),

        /**
         * A FLAC file read by [FlacDecoder] in this process, without MediaExtractor and MediaCodec; one it cannot read (a
         * CRC that does not match, a stream that ends early, a form it does not know) goes to MediaCodec instead, and
         * the report says why. Other files: [SYNC].
         */
        FLAC("flac", "Kotlin FLAC reader");

        companion object {
            fun parse(s: String?): Mode? = entries.firstOrNull { it.id == s?.lowercase() }
        }
    }

    /** Which reader a file goes to. */
    enum class Path(val id: String) { WAV("wav"), FLAC("flac reader"), MEDIACODEC("mediacodec") }

    /**
     * The reader for a file that starts with [head] ([n] bytes read) and is named [name]: WAV read directly; with
     * [Mode.FLAC], a FLAC file (the `fLaC` marker, or an ID3 tag in front of a file named `.flac`) by [FlacDecoder];
     * everything else by MediaCodec.
     */
    fun pathFor(head: ByteArray, n: Int, name: String?, mode: Mode): Path {
        fun ascii(o: Int, k: Int) = if (n >= o + k) String(head, o, k, Charsets.US_ASCII) else ""
        return when {
            ascii(0, 4) == "RIFF" && ascii(8, 4) == "WAVE" -> Path.WAV
            mode == Mode.FLAC && ascii(0, 4) == "fLaC" -> Path.FLAC
            mode == Mode.FLAC && ascii(0, 3) == "ID3" && name?.lowercase()?.endsWith(".flac") == true -> Path.FLAC
            else -> Path.MEDIACODEC
        }
    }

    /**
     * Where a decode's time went, summed over the whole file (nanoseconds; the report gives ms): the extractor's
     * readSampleData + advance, queueInputBuffer, dequeueInputBuffer, dequeueOutputBuffer without a wait, the waits,
     * the samples out (to mono floats and into the PCM file), releaseOutputBuffer; and a time line ([curve]: every 2 s
     * the output buffers, samples and source position so far), to tell a constant rate from one that falls.
     */
    class Stats(var mode: String) {
        var readNs = 0L; var queueInNs = 0L; var dequeueInNs = 0L; var dequeueOutNs = 0L; var waitNs = 0L
        var convertNs = 0L; var releaseNs = 0L

        /** The reader that gave the samples ([Path.id]), and why the FLAC reader handed a file to MediaCodec (null: it did not). */
        var path = ""
        var fallbackReason: String? = null

        /** [Mode.FLAC]: reading and decoding the frames in Kotlin (the MediaCodec stages stay 0). */
        var flacNs = 0L
        var inputs = 0; var outputs = 0; var emptyPolls = 0; var waits = 0; var formatChanges = 0; var maxInFlight = 0
        var capped = false
        var sourcePositionUs = 0L
        var codec: String? = null
        var codecSoftware: Boolean? = null
        val curve = ArrayList<DoubleArray>()
        private val t0 = System.nanoTime()
        private var lastPoint = 0L

        fun inFlight() { maxInFlight = maxOf(maxInFlight, inputs - outputs) }

        /** Adds a time-line point when 2 s passed since the last one (or [force]). */
        fun point(samples: Long, force: Boolean = false) {
            val now = System.nanoTime()
            if (!force && now - lastPoint < 2_000_000_000L) return
            lastPoint = now
            curve += doubleArrayOf((now - t0) / 1e6, outputs.toDouble(), samples.toDouble(), sourcePositionUs / 1000.0)
        }

        fun json(): JSONObject = JSONObject().apply {
            put("mode", mode)
            put("path", path)
            fallbackReason?.let { put("fallback_reason", it) }
            codec?.let { put("codec", it) }
            codecSoftware?.let { put("codec_software_only", it) }
            for ((k, v) in listOf("read_ms" to readNs, "queue_in_ms" to queueInNs, "dequeue_in_ms" to dequeueInNs, "dequeue_out_ms" to dequeueOutNs,
                "wait_ms" to waitNs, "convert_ms" to convertNs, "release_ms" to releaseNs)) put(k, v / 1e6)
            if (flacNs > 0) put("flac_decode_ms", flacNs / 1e6)
            put("inputs", inputs); put("outputs", outputs); put("empty_polls", emptyPolls); put("waits", waits)
            put("format_changes", formatChanges); put("max_in_flight", maxInFlight)
            put("capped", capped); put("source_position_s", sourcePositionUs / 1e6)
            put("curve_fields", "elapsed_ms, output_buffers, samples, source_position_ms")
            put("curve", org.json.JSONArray().apply { for (p in curve) put(org.json.JSONArray().apply { p.forEach { put(it) } }) })
        }
    }

    /**
     * Decodes [uri] to [out]. [mode]: how MediaCodec is driven (not used for WAV); [capMs]: stop after that much time and
     * keep what was decoded (the decode probe; [Stats.capped]).
     */
    fun decode(context: Context, uri: Uri, out: File, cancel: () -> Boolean = { false }, mode: Mode = Mode.FLAC, capMs: Long? = null, progress: (Double) -> Unit = {}): Decoded {
        val t = System.nanoTime()
        val head = ByteArray(12)
        val n = context.contentResolver.openInputStream(uri)?.use { readFully(it, head) } ?: throw DecodeException("The file could not be opened.")
        val path = pathFor(head, n, uri.lastPathSegment, mode)
        val media = if (mode == Mode.FLAC) Mode.SYNC else mode
        val stats = Stats(when (path) { Path.WAV -> "wav"; Path.FLAC -> Mode.FLAC.id; Path.MEDIACODEC -> media.id })
        stats.path = path.id
        val deadline = capMs?.let { t + it * 1_000_000L }
        val r = when (path) {
            Path.WAV -> decodeWav(context, uri, out, cancel, progress)
            Path.FLAC -> try {
                decodeFlac(context, uri, out, cancel, progress, stats, deadline)
            } catch (e: Exception) {
                if (cancel()) throw e
                // A FLAC file the reader cannot read goes to MediaCodec, which writes the PCM file again from the start.
                stats.fallbackReason = if (e is DecodeException) e.message ?: "DecodeException" else "${e.javaClass.simpleName}: ${e.message}"
                stats.path = Path.MEDIACODEC.id
                stats.mode = media.id
                try {
                    decodeMedia(context, uri, out, cancel, progress, media, stats, deadline)
                } catch (e2: Exception) {
                    if (cancel()) throw e2
                    throw DecodeException("the FLAC reader could not read it (${stats.fallbackReason}), and MediaCodec could not either (${e2.message ?: e2.javaClass.simpleName})", e2)
                }
            }
            Path.MEDIACODEC -> decodeMedia(context, uri, out, cancel, progress, media, stats, deadline)
        }
        return Decoded(r.samples, r.sourceRate, r.channels, r.format, r.resampled, (System.nanoTime() - t) / 1e6, r.expectedSamples, stats.json())
    }

    /** [Mode.FLAC]: the frames decoded by [FlacDecoder], channels averaged, into the same PCM file as MediaCodec's path. */
    private fun decodeFlac(context: Context, uri: Uri, out: File, cancel: () -> Boolean, progress: (Double) -> Unit, st: Stats, deadline: Long?): Result {
        val input = context.contentResolver.openInputStream(uri) ?: throw DecodeException("The file could not be opened.")
        input.use { s ->
            fun bad(e: FlacDecoder.FlacException) = DecodeException("This FLAC file could not be read (${e.message}).", e)
            val d = try { FlacDecoder(s) } catch (e: FlacDecoder.FlacException) { throw bad(e) }
            val info = d.info
            st.codec = "kotlin flac ${info.bitsPerSample}-bit ${info.channels} ch ${info.sampleRate} Hz"
            val ch = Array(info.channels) { IntArray(maxOf(info.maxBlock, 1)) }
            var mono = FloatArray(0)
            val scale = 1f / (1 shl (info.bitsPerSample - 1))
            var decoded = 0L
            var lastProgress = -1.0
            Sink(out, info.sampleRate).use { sink ->
                while (true) {
                    if (cancel()) throw DecodeException("cancelled")
                    if (deadline != null && System.nanoTime() > deadline) { st.capped = true; break }
                    var t = System.nanoTime()
                    val n = try { d.next(ch) } catch (e: FlacDecoder.FlacException) { throw bad(e) }
                    st.flacNs += System.nanoTime() - t
                    if (n == 0) break
                    t = System.nanoTime()
                    if (mono.size < n) mono = FloatArray(n)
                    val c = info.channels
                    if (c == 1) {
                        val x = ch[0]
                        for (i in 0 until n) mono[i] = x[i] * scale
                    } else {
                        for (i in 0 until n) {
                            var acc = 0f
                            for (k in 0 until c) acc += ch[k][i] * scale
                            mono[i] = acc / c
                        }
                    }
                    sink.push(mono, n)
                    st.convertNs += System.nanoTime() - t
                    st.inputs++; st.outputs++
                    decoded += n
                    st.sourcePositionUs = decoded * 1_000_000L / info.sampleRate
                    st.point(sink.written)
                    if (info.totalSamples > 0) {
                        val f = decoded.toDouble() / info.totalSamples
                        if (f - lastProgress >= 0.005) { progress(f); lastProgress = f }
                    }
                }
                // A stream that ends on a frame boundary before the length its STREAMINFO declares is cut short too.
                if (!st.capped && info.totalSamples > 0 && decoded != info.totalSamples) {
                    throw DecodeException("This FLAC file could not be read (it ends after $decoded of the ${info.totalSamples} samples it declares).")
                }
                sink.close()
                st.point(sink.written, force = true)
                progress(1.0)
                val expected = if (info.totalSamples > 0) at16k(info.totalSamples, info.sampleRate) else null
                return Result(sink.written, info.sampleRate, info.channels, "audio/flac", sink.resampled, expected)
            }
        }
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

    private class Result(val samples: Long, val sourceRate: Int, val channels: Int, val format: String, val resampled: Boolean, val expectedSamples: Long?)

    /** [n] samples at [rate] as samples at 16 kHz. */
    private fun at16k(n: Long, rate: Int): Long = if (rate == MelFrontEnd.SAMPLE_RATE) n else Math.round(n.toDouble() * MelFrontEnd.SAMPLE_RATE / rate)

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
                        return Result(sink.written, rate, channels, if (float) "wav float32" else "wav pcm$bits", sink.resampled, at16k(total, rate))
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

    private fun decodeMedia(context: Context, uri: Uri, out: File, cancel: () -> Boolean, progress: (Double) -> Unit, mode: Mode, st: Stats, deadline: Long?): Result {
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
            st.codec = runCatching { codec.name }.getOrNull()
            if (Build.VERSION.SDK_INT >= 29) st.codecSoftware = runCatching { codec.codecInfo.isSoftwareOnly }.getOrNull()
            val run = MediaRun(codec, ex, out, inFormat, durationUs, st, progress)
            try {
                when (mode) {
                    Mode.ASYNC -> run.async(cancel, deadline)
                    Mode.SYNC, Mode.LEGACY, Mode.FLAC -> {
                        codec.configure(inFormat, null, null, 0)
                        codec.start()
                        if (mode == Mode.LEGACY) run.legacy(cancel, deadline) else run.sync(cancel, deadline)
                    }
                }
                run.stopCodec()
                val s = run.sink ?: throw DecodeException("The decoder gave no audio for this file.")
                s.close()
                st.point(s.written, force = true)
                progress(1.0)
                val expected = if (durationUs > 0) Math.round(durationUs * MelFrontEnd.SAMPLE_RATE / 1e6) else null
                return Result(s.written, run.rate, run.channels, mime, s.resampled, expected)
            } finally {
                run.stopCodec()
                runCatching { run.sink?.close() }
                codec.release()
            }
        } finally {
            ex.release()
        }
    }

    /** The sync loop's wait when neither an input nor an output buffer moved (it returns as soon as an output is ready). */
    private const val WAIT_US = 10_000L

    /**
     * One MediaCodec decode of an extracted track into the PCM file: the output format as the codec reports it, and
     * the three ways to drive the codec ([Mode]). Everything runs on one thread, the caller's (sync, legacy) or the
     * callback thread (async), which is stopped before the caller reads the result.
     */
    private class MediaRun(
        val codec: MediaCodec, val ex: MediaExtractor, val file: File, val inFormat: MediaFormat, val durationUs: Long,
        val st: Stats, val progress: (Double) -> Unit,
    ) {
        var rate = inFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = inFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var encoding = AudioFormat.ENCODING_PCM_16BIT
        var sink: Sink? = null
        private var mono = FloatArray(0)
        private var lastProgress = -1.0
        private var inputDone = false
        private var thread: HandlerThread? = null
        private val stopped = AtomicBoolean(false)

        /** Fills input buffer [i] with the extractor's next sample, or marks the end of the stream. */
        fun feed(i: Int) {
            val buf = codec.getInputBuffer(i)!!
            var t = System.nanoTime()
            val size = ex.readSampleData(buf, 0)
            val time = ex.sampleTime
            st.readNs += System.nanoTime() - t
            t = System.nanoTime()
            if (size < 0) {
                codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                inputDone = true
            } else {
                codec.queueInputBuffer(i, 0, size, time, 0)
            }
            st.queueInNs += System.nanoTime() - t
            st.inputs++
            st.inFlight()
            if (size < 0) return
            t = System.nanoTime()
            ex.advance()
            st.readNs += System.nanoTime() - t
            st.sourcePositionUs = time
            if (durationUs > 0) {
                val f = time.toDouble() / durationUs
                if (f - lastProgress >= 0.005) { progress(f); lastProgress = f }
            }
        }

        fun formatChanged(f: MediaFormat) {
            rate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
            st.formatChanges++
        }

        /** Output buffer [o] into the PCM file and back to the codec; true at the end of the stream. */
        fun drain(o: Int, info: MediaCodec.BufferInfo): Boolean {
            var t = System.nanoTime()
            val buf = codec.getOutputBuffer(o)!!.order(ByteOrder.LITTLE_ENDIAN)
            buf.position(info.offset)
            buf.limit(info.offset + info.size)
            val s = sink ?: Sink(file, rate).also { sink = it }
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
            st.convertNs += System.nanoTime() - t
            t = System.nanoTime()
            codec.releaseOutputBuffer(o, false)
            st.releaseNs += System.nanoTime() - t
            st.outputs++
            st.point(s.written)
            return info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
        }

        /** Every free input buffer filled and every ready output buffer drained; a wait only when neither moved. */
        fun sync(cancel: () -> Boolean, deadline: Long?) {
            val info = MediaCodec.BufferInfo()
            var idle = false
            while (true) {
                if (cancel()) throw DecodeException("cancelled")
                if (deadline != null && System.nanoTime() > deadline) { st.capped = true; return }
                var moved = false
                while (!inputDone) {
                    val t = System.nanoTime()
                    val i = codec.dequeueInputBuffer(0)
                    st.dequeueInNs += System.nanoTime() - t
                    if (i < 0) break
                    feed(i)
                    moved = true
                }
                var wait = if (idle) WAIT_US else 0L
                while (true) {
                    val t = System.nanoTime()
                    val o = codec.dequeueOutputBuffer(info, wait)
                    val dt = System.nanoTime() - t
                    if (wait > 0) { st.waits++; st.waitNs += dt } else st.dequeueOutNs += dt
                    wait = 0
                    if (o == MediaCodec.INFO_TRY_AGAIN_LATER) { st.emptyPolls++; break }
                    moved = true
                    if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) { formatChanged(codec.outputFormat); continue }
                    if (o < 0) continue
                    if (drain(o, info)) return
                }
                idle = !moved
            }
        }

        /**
         * The builds before this one: free inputs filled, then one output buffer a turn, a 5 ms wait after an empty poll.
         * (The loop before that, one input and a 10 ms wait for one output a turn, read the 54 min FLAC fixture in 616 s
         * on a Galaxy S26; this one in 44 s there and 417.8 s on a Pixel 8a.)
         */
        fun legacy(cancel: () -> Boolean, deadline: Long?) {
            val info = MediaCodec.BufferInfo()
            var waitUs = 0L
            while (true) {
                if (cancel()) throw DecodeException("cancelled")
                if (deadline != null && System.nanoTime() > deadline) { st.capped = true; return }
                while (!inputDone) {
                    val t = System.nanoTime()
                    val i = codec.dequeueInputBuffer(0)
                    st.dequeueInNs += System.nanoTime() - t
                    if (i < 0) break
                    feed(i)
                }
                val t = System.nanoTime()
                val o = codec.dequeueOutputBuffer(info, waitUs)
                val dt = System.nanoTime() - t
                if (waitUs > 0) { st.waits++; st.waitNs += dt } else st.dequeueOutNs += dt
                if (o == MediaCodec.INFO_TRY_AGAIN_LATER) { st.emptyPolls++; waitUs = 5_000; continue }
                waitUs = 0
                if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) { formatChanged(codec.outputFormat); continue }
                if (o < 0) continue
                if (drain(o, info)) return
            }
        }

        /** MediaCodec's callbacks on a thread of their own; the calling thread only waits for the end, a stop or the cap. */
        fun async(cancel: () -> Boolean, deadline: Long?) {
            val th = HandlerThread("decode-codec").also { it.start(); thread = it }
            val done = CountDownLatch(1)
            val error = AtomicReference<Throwable?>(null)
            // A callback that was running when the codec stopped (the cap, a stop) fails on the stopped codec: not an error.
            fun fail(t: Throwable) { if (!stopped.get() && error.compareAndSet(null, t)) done.countDown() }
            codec.setCallback(object : MediaCodec.Callback() {
                override fun onInputBufferAvailable(c: MediaCodec, index: Int) {
                    if (stopped.get() || inputDone) return
                    try { feed(index) } catch (t: Throwable) { fail(t) }
                }

                override fun onOutputBufferAvailable(c: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
                    if (stopped.get()) return
                    try { if (drain(index, info)) done.countDown() } catch (t: Throwable) { fail(t) }
                }

                override fun onOutputFormatChanged(c: MediaCodec, format: MediaFormat) = formatChanged(format)

                override fun onError(c: MediaCodec, e: MediaCodec.CodecException) = fail(e)
            }, Handler(th.looper))
            codec.configure(inFormat, null, null, 0)
            codec.start()
            while (!done.await(100, TimeUnit.MILLISECONDS)) {
                if (cancel()) { stopCodec(); throw DecodeException("cancelled") }
                if (deadline != null && System.nanoTime() > deadline) { st.capped = true; break }
            }
            stopCodec()
            error.get()?.let { throw DecodeException("The decoder failed: ${it.message ?: it.javaClass.simpleName}", it) }
        }

        /** Stops the codec and, in async mode, its callback thread, so that nothing writes the PCM file any more. Idempotent. */
        fun stopCodec() {
            if (!stopped.compareAndSet(false, true)) return
            runCatching { codec.stop() }
            thread?.let { it.quitSafely(); it.join(2000) }
        }
    }
}
