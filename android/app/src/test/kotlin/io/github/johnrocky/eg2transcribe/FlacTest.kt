package io.github.johnrocky.eg2transcribe

import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
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
