package io.github.johnrocky.eg2transcribe

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Test

/** The stub's numbers are the ones tools/stub_table.py prints (the Mac side of the device check uses that script). */
class StubTest {
    private val vocab = WordTable.readVocab(File("src/main/assets/stub_vocab.txt"))

    private fun sha256(x: FloatArray): String {
        val b = ByteBuffer.allocate(x.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        x.forEach { b.putFloat(it) }
        return MessageDigest.getInstance("SHA-256").digest(b.array()).joinToString("") { "%02x".format(it) }
    }

    @Test
    fun sameNumbersAsThePythonScript() {
        val u = Stub.uniform(4)
        assertEquals(listOf(0xbda7e920.toInt(), 0xbcc0de80.toInt(), 0xbe759c50.toInt(), 0x3ed44cbc), u.map { it.toBits() })
        val t = Stub.table(vocab)
        assertEquals((-0.006362096406519413f).toBits(), t.row(0)[0].toBits())
        assertEquals((-0.001826941967010498f).toBits(), t.row(0)[1].toBits())
        assertEquals(0.07338463515043259f.toBits(), t.row(999)[511].toBits())
        val all = FloatArray(Stub.WORDS * Stub.DIM).also { a -> for (j in 0 until Stub.WORDS) t.row(j).copyInto(a, j * Stub.DIM) }
        assertEquals("2fdb7be74d5c544863079b979c64ea4206cf188ed73cf191f8c020f226335241", sha256(all))
    }

    @Test
    fun knownSoftTokensReadAsKnownWords() {
        assertEquals(1000, vocab.size)
        val table = Stub.table(vocab)
        val reader = Stub.reader()
        // The blank vector, recovered the way Stub.reader builds it, for a blank token.
        val braw = Stub.uniform(Stub.DIM, skip = Stub.WORDS.toLong() * Stub.DIM)
        val bnorm = Math.sqrt(braw.sumOf { it.toDouble() * it })
        val blankVec = FloatArray(Stub.DIM) { (braw[it] / bnorm).toFloat() }
        assertEquals("215a5dd815fe409a280b5222df07a5b8ce8933edc6948c142b1fe2c091c29d97", sha256(blankVec))
        val rows = listOf(17, 17, 17, -1, 42, 42, 999, 0)
        val n = rows.size
        val soft = FloatArray(n * SOFT_DIM)
        rows.forEachIndexed { t, j -> (if (j < 0) blankVec else table.row(j)).forEachIndexed { i, x -> soft[t * SOFT_DIM + i] = x * 29f } }
        val vectors = FloatArray(n * reader.dim)
        val blank = DoubleArray(n)
        reader.read(soft, 0, n, vectors, blank)
        val r = IntArray(n); val cos = DoubleArray(n); val second = DoubleArray(n)
        table.nearestAll(vectors, n, null, 1, r, cos, second)
        val ids = IntArray(n) { if (reader.isBlank(blank[it], cos[it])) Ctc.BLANK else r[it] }
        assertEquals(listOf(17, 17, 17, Ctc.BLANK, 42, 42, 999, 0), ids.toList())
        val words = Ctc(vocab).feed(ids, n, 0)
        assertEquals(listOf(vocab[17], vocab[42], vocab[999], vocab[0]), words.map { it.text })
        assertEquals(listOf(0, 4, 6, 7), words.map { it.firstToken })
    }
}
