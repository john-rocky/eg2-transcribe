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
}
