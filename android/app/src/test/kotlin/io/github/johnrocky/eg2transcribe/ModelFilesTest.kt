package io.github.johnrocky.eg2transcribe

import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelFilesTest {
    private val dir: File = Files.createTempDirectory("eg2models").toFile()

    @After
    fun clean() {
        dir.deleteRecursively()
    }

    private fun touch(name: String, bytes: Int = 10) = File(dir, name).apply { writeBytes(ByteArray(bytes)) }

    @Test
    fun aPairIsOneModelPickedByEitherFile() {
        val w30 = touch("eg2_audio_f16safe_fp16w_w30.tflite")
        val w10 = touch("eg2_audio_f16safe_fp16w_w10.tflite")
        for (picked in listOf(w30.name, w10.name)) {
            val c = ModelChoice.of(dir, picked)
            assertEquals("eg2_audio_f16safe_fp16w", c.name)
            assertTrue(c.isPair)
            assertFalse(c.missing)
            assertEquals(w30, c.fileFor(WindowPlan.FRAMES_30S))
            assertEquals(w10, c.fileFor(WindowPlan.FRAMES_10S))
            assertEquals(w10, c.npuFile)
            assertEquals(setOf(WindowPlan.FRAMES_30S, WindowPlan.FRAMES_10S), c.windows)
        }
        assertEquals(listOf(w10.name), ModelChoice.all(dir)) // listed once
    }

    @Test
    fun halfAPair() {
        touch("m_w30.tflite")
        val c = ModelChoice.of(dir, "m_w10.tflite") // picked by the missing half
        assertEquals("m", c.name)
        assertFalse(c.missing)
        assertEquals(setOf(WindowPlan.FRAMES_30S), c.windows)
        assertNull(c.fileFor(WindowPlan.FRAMES_10S))
        assertNull(c.npuFile) // the NPU needs the 10 s file
        assertTrue(c.describe().contains("30 s file only"))
    }

    @Test
    fun aSingleFileDecidesItsWindowsWhenOpened() {
        val f = touch("eg2_audio_fp32.tflite")
        val c = ModelChoice.of(dir, f.name)
        assertFalse(c.isPair)
        assertEquals(f, c.fileFor(WindowPlan.FRAMES_30S))
        assertEquals(f, c.fileFor(WindowPlan.FRAMES_10S))
        assertNull(c.npuFile) // a two-signature file would make the NPU compile its 30 s graph too
        assertEquals(setOf(WindowPlan.FRAMES_30S, WindowPlan.FRAMES_10S), c.windows)
    }

    @Test
    fun missing() {
        val pair = ModelChoice.of(dir, Engine.DEFAULT_MODEL)
        assertTrue(pair.missing)
        assertTrue(pair.windows.isEmpty())
        assertTrue(pair.describe().endsWith("not on this phone"))
        val single = ModelChoice.of(dir, "x.tflite")
        assertTrue(single.missing)
        assertNull(single.fileFor(WindowPlan.FRAMES_30S))
    }

    @Test
    fun listsEveryModelOnce() {
        touch("a_w10.tflite"); touch("a_w30.tflite"); touch("b.tflite"); touch("c_w30.tflite"); touch("notes.txt")
        assertEquals(listOf("a_w10.tflite", "b.tflite", "c_w30.tflite"), ModelChoice.all(dir))
        assertEquals("a", ModelChoice.nameOf("a_w30.tflite"))
        assertEquals("b.tflite", ModelChoice.nameOf("b.tflite"))
    }

    @Test
    fun npuLinkNames() {
        val sha = "0123456789abcdef".repeat(4)
        assertEquals("eg2_audio_f16safe_fp16w_w10-0123456789ab", Npu.linkName("eg2_audio_f16safe_fp16w_w10", sha))
    }

    @Test
    fun theLinesNameTheFilesBySizeAndSha256() {
        val f = File(dir, "reader_w30.tflite").apply { writeText("abc") }
        val s = FileStamp.of(f)
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", s.sha256) // SHA-256 of "abc" (FIPS 180-2)
        assertEquals(3L, s.bytes)
        assertTrue(s.sha256Ms!! >= 0.0)
        assertEquals("reader_w30.tflite (0 MB, sha256 ba7816bf8f01)", s.label)
        // The published files (README.md, The model files).
        val rd = FileStamp(File("reader_w30.tflite"), 386_726_384, "129d6aa2d829dc4418675d01659ccc6fc42e33b84c2486f36faa9407d45c8716", 900.0)
        assertEquals("Reader on GPU fp32 · reader_w30.tflite (387 MB, sha256 129d6aa2d829) · 200,000 words", Engine.readerNote(Backend.GPU_FP32, "fp32", rd, 200_000))
        assertEquals(
            "Reader on GPU fp32 (fp16, fp32 accumulation) · reader_w30.tflite (387 MB, sha256 129d6aa2d829) · 200,000 words",
            Engine.readerNote(Backend.GPU_FP32, ReaderPrecision.FP16_ACC32.id, rd, 200_000),
        )
        val enc = FileStamp(File("eg2_audio_f16safe_fp16w_w30.tflite"), 589_273_696, "f0f6c37ee62b51372ab28013617aad53793a84aaddd96e1e844d0cea0eb2f11d", 1300.0)
        assertEquals(
            "Encoder on GPU fp32 · eg2_audio_f16safe_fp16w_w30.tflite (589 MB, sha256 f0f6c37ee62b) · signature audio_30s",
            Engine.encoderNote(Backend.GPU_FP32, 4, enc, listOf("audio_30s")),
        )
        assertEquals(
            "Encoder on CPU (4 threads) · eg2_audio_f16safe_fp16w_w30.tflite (589 MB, sha256 f0f6c37ee62b) · signatures audio_30s, audio_10s",
            Engine.encoderNote(Backend.CPU, 4, enc, listOf("audio_30s", "audio_10s")),
        )
        // A file that could not be read for its hash is named by its size alone.
        assertEquals("reader_w30.tflite (387 MB)", FileStamp(File("reader_w30.tflite"), 386_726_384, null, null).label)
        val gone = FileStamp.of(File(dir, "gone.tflite"))
        assertNull(gone.sha256)
        assertNull(gone.sha256Ms)
    }

    @Test
    fun theResultLinesSayTheLengthTheTimeTheSpeedAndTheAccelerators() {
        // The 54-minute fixture (52,119,092 samples) and the bundled example (8.1 s), with the run report's numbers.
        assertEquals(
            "Read 54:17 of audio in 55.2 s" to "59× real time · encoder on NPU, reader on GPU fp32",
            Jobs.resultLines(52_119_092 / 16000.0, 55_234.5, 58.97, "NPU", "GPU fp32"),
        )
        assertEquals(
            "Read 00:08 of audio in 0.26 s" to "31× real time · encoder on GPU fp32, reader on GPU fp32",
            Jobs.resultLines(8.12, 258.4, 31.42, "GPU fp32", "GPU fp32"),
        )
        assertEquals("16× real time · encoder on CPU, stub reader", Jobs.resultLines(30.0, 1875.0, 16.0, "CPU", null).second)
        assertEquals("Read 1:02:03 of audio in 61.0 s", Jobs.resultLines(3723.0, 61_000.0, 61.0, "NPU", "GPU fp32").first)
    }
}
