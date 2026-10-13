package io.github.johnrocky.eg2transcribe

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class WordTableTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun npy(file: File, descr: String, shape: String, data: ByteArray) {
        var header = "{'descr': '$descr', 'fortran_order': False, 'shape': ($shape), }"
        val total = 10 + header.length + 1
        header += " ".repeat((64 - total % 64) % 64) + "\n"
        val b = ByteBuffer.allocate(10 + header.length + data.size).order(ByteOrder.LITTLE_ENDIAN)
        b.put(0x93.toByte()).put("NUMPY".toByteArray()).put(1).put(0).putShort(header.length.toShort())
        b.put(header.toByteArray(Charsets.ISO_8859_1)).put(data)
        file.writeBytes(b.array())
    }

    @Test
    fun halfToFloat() {
        assertEquals(0f, Npy.halfToFloat(0x0000))
        assertEquals(-0f, Npy.halfToFloat(0x8000))
        assertEquals(1f, Npy.halfToFloat(0x3c00))
        assertEquals(-2f, Npy.halfToFloat(0xc000))
        assertEquals(65504f, Npy.halfToFloat(0x7bff))
        assertEquals(6.1035156e-5f, Npy.halfToFloat(0x0400)) // smallest normal
        assertEquals(5.9604645e-8f, Npy.halfToFloat(0x0001)) // smallest subnormal
        assertEquals(3.0517578e-5f, Npy.halfToFloat(0x0200))
        assertEquals(Float.POSITIVE_INFINITY, Npy.halfToFloat(0x7c00))
        assertTrue(Npy.halfToFloat(0x7e00).isNaN())
        assertEquals(0.33325195f, Npy.halfToFloat(0x3555))
    }

    @Test
    fun readsNpyAndBin() {
        val vocab = tmp.newFile("vocab.txt").apply { writeText("cat\ndog\r\nbird\n") }
        // Rows (3, 0, 0, 4), (0, 2, 0, 0), (1, 1, 1, 1) as float32 and float16.
        val raw = floatArrayOf(3f, 0f, 0f, 4f, 0f, 2f, 0f, 0f, 1f, 1f, 1f, 1f)
        val f32 = ByteBuffer.allocate(raw.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { raw.forEach { putFloat(it) } }.array()
        val halves = mapOf(0f to 0x0000, 1f to 0x3c00, 2f to 0x4000, 3f to 0x4200, 4f to 0x4400)
        val f16 = ByteBuffer.allocate(raw.size * 2).order(ByteOrder.LITTLE_ENDIAN).apply { raw.forEach { putShort(halves.getValue(it).toShort()) } }.array()
        val a = tmp.newFile("word_table.npy").also { npy(it, "<f4", "3, 4", f32) }
        val b = tmp.newFile("t16.npy").also { npy(it, "<f2", "3, 4", f16) }
        val c = tmp.newFile("word_table.bin").apply { writeBytes(f16) }
        val expected = floatArrayOf(0.6f, 0f, 0f, 0.8f, 0f, 1f, 0f, 0f, 0.5f, 0.5f, 0.5f, 0.5f)
        for (f in listOf(a, b, c)) {
            val t = WordTable.load(f, vocab)
            assertEquals(listOf("cat", "dog", "bird"), t.vocab)
            assertEquals(4, t.dim)
            val r = WordTable.Nearest()
            val scratch = FloatArray(3)
            for (j in 0 until 3) {
                // Each raw row, scaled, is nearest to itself with cosine 1.
                val v = FloatArray(4) { raw[j * 4 + it] * 7f }
                t.nearest(v, 0, scratch, r)
                assertEquals(f.name, j, r.row)
                assertEquals(1.0, r.cos, 1e-6)
            }
            val v = floatArrayOf(0.6f, 0f, 0f, 0.8f)
            t.nearest(v, 0, scratch, r)
            assertEquals(0.5 * 0.6 + 0.5 * 0.8, r.second, 1e-6) // "bird"
            for (j in 0 until 3) assertArrayEquals(f.name, expected.copyOfRange(j * 4, j * 4 + 4), t.row(j), 1e-7f)
        }
    }

    @Test
    fun tiesGoToTheLowerRowAndZeroIsNoRow() {
        val t = WordTable.fromRaw(listOf("a", "b", "c"), 2, floatArrayOf(1f, 0f, 1f, 0f, 0f, 1f))
        val r = WordTable.Nearest()
        t.nearest(floatArrayOf(5f, 0f), 0, FloatArray(3), r)
        assertEquals(0, r.row)
        t.nearest(floatArrayOf(0f, 0f), 0, FloatArray(3), r)
        assertEquals(-1, r.row)
    }

    @Test
    fun parallelScanEqualsSerial() {
        val vocab = List(Stub.WORDS) { "w$it" }
        val t = Stub.table(vocab)
        val n = 300
        val v = Stub.uniform(n * Stub.DIM, skip = 9_999_999)
        val rowsA = IntArray(n); val cosA = DoubleArray(n); val secA = DoubleArray(n)
        val rowsB = IntArray(n); val cosB = DoubleArray(n); val secB = DoubleArray(n)
        t.nearestAll(v, n, null, 1, rowsA, cosA, secA)
        val pool = Executors.newFixedThreadPool(4)
        try {
            t.nearestAll(v, n, pool, 4, rowsB, cosB, secB)
        } finally {
            pool.shutdown()
        }
        assertArrayEquals(rowsA, rowsB)
        assertArrayEquals(cosA, cosB, 0.0)
        // And the row is the float64 argmax.
        for (k in 0 until n) {
            var best = Double.NEGATIVE_INFINITY
            var arg = -1
            for (j in 0 until Stub.WORDS) {
                var d = 0.0
                for (i in 0 until Stub.DIM) d += v[k * Stub.DIM + i].toDouble() * rowValue(t, j, i)
                if (d > best) { best = d; arg = j }
            }
            assertEquals("token $k", arg, rowsA[k])
        }
    }

    private val rowCache = HashMap<Int, FloatArray>()

    /** Row j of the stub table, recomputed the way Stub.table builds it. */
    private fun rowValue(t: WordTable, j: Int, i: Int): Float {
        val row = rowCache.getOrPut(j) {
            val raw = Stub.uniform(Stub.DIM, skip = j.toLong() * Stub.DIM)
            var s = 0.0
            for (x in raw) s += x.toDouble() * x
            val norm = Math.sqrt(s)
            FloatArray(Stub.DIM) { (raw[it] / norm).toFloat() }
        }
        return row[i]
    }
}
