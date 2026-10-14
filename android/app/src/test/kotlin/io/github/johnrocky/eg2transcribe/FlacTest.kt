package io.github.johnrocky.eg2transcribe

import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * [FlacDecoder] against ffmpeg: the samples of a FLAC file as 16-bit little-endian bytes must have the SHA-256 of
 * `ffmpeg -i <file> -f s16le -acodec pcm_s16le -` (FLAC is lossless: soundfile and ffmpeg give the same samples).
 * Always: the bundled example. With `-Peg2.flac=<list>` (lines "<path> <samples> <sha256>", the reference computed on a
 * computer with ffmpeg): every file of the list, e.g. the 20 test-clean fixtures and the 54 min LibriVox recording.
 */
class FlacTest {
    private fun decode(f: File): Pair<Long, String> {
        val md = MessageDigest.getInstance("SHA-256")
        var n = 0L
        f.inputStream().buffered(1 shl 16).use { s ->
            val d = FlacDecoder(s)
            val ch = Array(d.info.channels) { IntArray(d.info.maxBlock) }
            val bytes = ByteArray(2 * 65536)
            while (true) {
                val k = d.next(ch)
                if (k == 0) break
                assertEquals("16-bit mono", 1, d.info.channels)
                for (i in 0 until k) {
                    val v = ch[0][i]
                    bytes[2 * i] = (v and 0xff).toByte()
                    bytes[2 * i + 1] = ((v shr 8) and 0xff).toByte()
                }
                md.update(bytes, 0, 2 * k)
                n += k
            }
            assertEquals("${f.name}: STREAMINFO total samples", d.info.totalSamples, n)
        }
        return n to md.digest().joinToString("") { "%02x".format(it) }
    }

    @Test
    fun theBundledExampleBitForBit() {
        // LibriSpeech test-clean 1284-1180-0000 (the app's example): 129,920 samples; ffmpeg's s16le SHA-256.
        val (n, sha) = decode(File("src/main/assets/examples/1284-1180-0000.flac"))
        assertEquals(129_920L, n)
        assertEquals("afd1b2e3f459ad1cb2e2274b621bff26236dd044e0f8a78e4dfb18a7ba8959ef", sha)
    }

    /** What the reader makes of [bytes]: "ok <samples>", "short <n> of <declared>" (AudioDecoder refuses that too), or the error. */
    private fun outcome(bytes: ByteArray): String = try {
        val d = FlacDecoder(bytes.inputStream())
        val ch = Array(d.info.channels) { IntArray(d.info.maxBlock) }
        var n = 0L
        while (true) {
            val k = d.next(ch)
            if (k == 0) break
            n += k
        }
        if (d.info.totalSamples > 0 && n != d.info.totalSamples) "short $n of ${d.info.totalSamples}" else "ok $n"
    } catch (e: Exception) {
        "error ${e.javaClass.simpleName}: ${e.message}"
    }

    @Test
    fun aCutOrDamagedFileIsNotReadAsWhole() {
        // The app hands such a file to MediaCodec (AudioDecoder.decode) and says why in the report.
        val whole = File("src/main/assets/examples/1284-1180-0000.flac").readBytes()
        assertEquals("ok 129920", outcome(whole))
        val cut = outcome(whole.copyOf(whole.size * 6 / 10)) // the first 60 %
        assertTrue(cut, cut == "error FlacException: FLAC stream ends inside a frame" || cut.startsWith("short"))
        val flipped = whole.copyOf().also { it[whole.size / 2] = (it[whole.size / 2].toInt() xor 0x10).toByte() } // one bit
        val damaged = outcome(flipped)
        assertTrue(damaged, damaged.startsWith("error") || damaged.startsWith("short"))
        println("FLAC cut: $cut; one bit flipped: $damaged")
    }

    @Test
    fun whichReaderAFileGoesTo() {
        fun head(s: String) = s.toByteArray(Charsets.US_ASCII).copyOf(12)
        val flac = head("fLaC")
        val wav = head("RIFF0000WAVE")
        val id3 = head("ID3")
        val mp4 = head("0000ftypM4A ")
        assertEquals(AudioDecoder.Path.FLAC, AudioDecoder.pathFor(flac, 12, "a.flac", AudioDecoder.Mode.FLAC))
        assertEquals(AudioDecoder.Path.FLAC, AudioDecoder.pathFor(flac, 12, null, AudioDecoder.Mode.FLAC)) // by its marker, whatever the name
        assertEquals(AudioDecoder.Path.FLAC, AudioDecoder.pathFor(id3, 12, "a.FLAC", AudioDecoder.Mode.FLAC))
        assertEquals(AudioDecoder.Path.MEDIACODEC, AudioDecoder.pathFor(id3, 12, "a.mp3", AudioDecoder.Mode.FLAC))
        assertEquals(AudioDecoder.Path.WAV, AudioDecoder.pathFor(wav, 12, "a.wav", AudioDecoder.Mode.FLAC))
        assertEquals(AudioDecoder.Path.WAV, AudioDecoder.pathFor(wav, 12, "a.wav", AudioDecoder.Mode.SYNC))
        assertEquals(AudioDecoder.Path.MEDIACODEC, AudioDecoder.pathFor(mp4, 12, "a.m4a", AudioDecoder.Mode.FLAC))
        // The measurement modes send FLAC through MediaCodec too.
        for (m in listOf(AudioDecoder.Mode.SYNC, AudioDecoder.Mode.ASYNC, AudioDecoder.Mode.LEGACY)) {
            assertEquals(AudioDecoder.Path.MEDIACODEC, AudioDecoder.pathFor(flac, 12, "a.flac", m))
        }
        assertEquals(AudioDecoder.Path.MEDIACODEC, AudioDecoder.pathFor(flac, 3, "a.flac", AudioDecoder.Mode.FLAC)) // too short to tell
    }

    @Test
    fun aListOfFilesBitForBit() {
        val list = System.getProperty("eg2.flac")?.let { File(it) }
        assumeTrue("set -Peg2.flac=<list> to run", list != null && list.isFile)
        for (line in list!!.readLines().filter { it.isNotBlank() }) {
            val (path, samples, sha) = line.trim().split(Regex("\\s+"))
            val t = System.nanoTime()
            val (n, got) = decode(File(path))
            println("FLAC ${File(path).parentFile?.name}/${File(path).name} samples=$n sha256=${if (got == sha) "equal" else "DIFF"} ms=${"%.0f".format((System.nanoTime() - t) / 1e6)}")
            assertEquals("$path: samples", samples.toLong(), n)
            assertEquals("$path: SHA-256 of the samples", sha, got)
        }
    }
}
