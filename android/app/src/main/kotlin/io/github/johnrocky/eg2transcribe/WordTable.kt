package io.github.johnrocky.eg2transcribe

import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Future
import kotlin.math.sqrt

/**
 * The words the transcript can use: row j of an N x d table is the text model's vector for line j of vocab.txt. Rows
 * are L2-normalized here, so a token's word is the row of largest dot product (= cosine) with the reader's vector.
 *
 * The scan runs in float32 and the rows within 1e-4 x |v| of the best (the float32 error bound for d = 768 is about
 * 5e-5 x |v|) are scored again in float64, so the chosen row is the float64 argmax: the same row numpy picks from the
 * same vectors (ties go to the lower row).
 */
class WordTable(val vocab: List<String>, val dim: Int, private val rows: FloatArray) {
    val size: Int get() = vocab.size

    init {
        require(rows.size == vocab.size * dim) { "table has ${rows.size} values, expected ${vocab.size} x $dim" }
    }

    /** Row j, normalized (a copy). */
    fun row(j: Int): FloatArray = rows.copyOfRange(j * dim, (j + 1) * dim)

    /** Result of one token's scan. */
    class Nearest {
        var row = -1
        var cos = 0.0
        var second = 0.0
    }

    /** The nearest row to v[off until off + dim]. [scratch] holds [size] floats. A zero vector gives row -1. */
    fun nearest(v: FloatArray, off: Int, scratch: FloatArray, out: Nearest) {
        var norm2 = 0.0
        for (i in 0 until dim) norm2 += v[off + i].toDouble() * v[off + i]
        val norm = sqrt(norm2)
        if (norm == 0.0) {
            out.row = -1; out.cos = 0.0; out.second = 0.0
            return
        }
        var best = Float.NEGATIVE_INFINITY
        var bestRow = 0
        var second = Float.NEGATIVE_INFINITY
        for (j in 0 until size) {
            val s = dot32(v, off, j * dim)
            scratch[j] = s
            if (s > best) {
                second = best; best = s; bestRow = j
            } else if (s > second) second = s
        }
        val eps = (1e-4 * norm).toFloat()
        var exact = Double.NEGATIVE_INFINITY
        var exactRow = bestRow
        var exactSecond = Double.NEGATIVE_INFINITY
        for (j in 0 until size) {
            if (scratch[j] < best - eps) continue
            val d = dot64(v, off, j * dim)
            if (d > exact) {
                exactSecond = exact; exact = d; exactRow = j
            } else if (d > exactSecond) exactSecond = d
        }
        out.row = exactRow
        out.cos = exact / norm
        out.second = (if (exactSecond > Double.NEGATIVE_INFINITY) exactSecond else second.toDouble()) / norm
    }

    private fun dot32(v: FloatArray, off: Int, r: Int): Float {
        var a0 = 0f; var a1 = 0f; var a2 = 0f; var a3 = 0f
        var i = 0
        val n4 = dim and 3.inv()
        while (i < n4) {
            a0 += v[off + i] * rows[r + i]
            a1 += v[off + i + 1] * rows[r + i + 1]
            a2 += v[off + i + 2] * rows[r + i + 2]
            a3 += v[off + i + 3] * rows[r + i + 3]
            i += 4
        }
        while (i < dim) { a0 += v[off + i] * rows[r + i]; i++ }
        return (a0 + a1) + (a2 + a3)
    }

    private fun dot64(v: FloatArray, off: Int, r: Int): Double {
        var a = 0.0
        for (i in 0 until dim) a += v[off + i].toDouble() * rows[r + i]
        return a
    }

    /**
     * Scans [n] vectors (row-major, width [dim]) on [pool] (null = this thread), filling [rowOut], [cosOut] and
     * [secondOut].
     */
    fun nearestAll(v: FloatArray, n: Int, pool: ExecutorService?, threads: Int, rowOut: IntArray, cosOut: DoubleArray, secondOut: DoubleArray) {
        fun span(a: Int, b: Int) {
            val scratch = FloatArray(size)
            val r = Nearest()
            for (t in a until b) {
                nearest(v, t * dim, scratch, r)
                rowOut[t] = r.row; cosOut[t] = r.cos; secondOut[t] = r.second
            }
        }
        if (pool == null || threads <= 1 || n < 32) {
            span(0, n)
            return
        }
        val per = (n + threads - 1) / threads
        val jobs = ArrayList<Future<*>>()
        var a = 0
        while (a < n) {
            val b = minOf(n, a + per)
            val (x, y) = a to b
            jobs += pool.submit { span(x, y) }
            a = b
        }
        for (j in jobs) j.get()
    }

    companion object {
        /** Tables larger than this are refused: the scan runs on the CPU in this version. */
        const val MAX_VALUES = 32L * 1024 * 1024

        /** A table from raw rows (normalized here, in float64, then stored as float32). */
        fun fromRaw(vocab: List<String>, dim: Int, raw: FloatArray): WordTable {
            require(raw.size == vocab.size * dim) { "table has ${raw.size} values, expected ${vocab.size} x $dim" }
            val rows = FloatArray(raw.size)
            for (j in vocab.indices) {
                var s = 0.0
                for (i in 0 until dim) s += raw[j * dim + i].toDouble() * raw[j * dim + i]
                val norm = sqrt(s)
                for (i in 0 until dim) rows[j * dim + i] = if (norm == 0.0) 0f else (raw[j * dim + i] / norm).toFloat()
            }
            return WordTable(vocab, dim, rows)
        }

        /** One word per line, UTF-8; row j = line j. */
        fun readVocab(file: File): List<String> = file.readLines(Charsets.UTF_8).map { it.trimEnd('\r') }.also { v ->
            require(v.isNotEmpty()) { "${file.name} is empty" }
            v.forEachIndexed { i, w -> require(w.isNotBlank()) { "${file.name} line ${i + 1} is empty" } }
        }

        /**
         * `word_table.npy` (float16 or float32, shape (N, d), C order) or `word_table.bin` (raw little-endian float16,
         * N x d with N = the vocab's lines), with `vocab.txt`.
         */
        fun load(table: File, vocabFile: File): WordTable {
            val vocab = readVocab(vocabFile)
            val (dim, raw) = if (table.name.endsWith(".npy")) {
                val npy = Npy.read(table)
                require(npy.shape.size == 2) { "${table.name}: shape ${npy.shape} is not (N, d)" }
                require(npy.shape[0] == vocab.size) { "${table.name} has ${npy.shape[0]} rows, ${vocabFile.name} has ${vocab.size} words" }
                npy.shape[1] to npy.data
            } else {
                val bytes = table.length()
                require(bytes % (2L * vocab.size) == 0L) { "${table.name}: $bytes bytes is not ${vocab.size} rows of float16" }
                val dim = (bytes / (2L * vocab.size)).toInt()
                require(vocab.size.toLong() * dim <= MAX_VALUES) { "${table.name}: ${vocab.size} x $dim values is more than this version scans on the CPU ($MAX_VALUES)" }
                dim to Npy.halves(table.readBytes())
            }
            return fromRaw(vocab, dim, raw)
        }
    }
}

/** A minimal reader for `.npy` files (format 1.0 / 2.0 / 3.0), float16 or float32, little-endian, C order. */
object Npy {
    class Array(val shape: List<Int>, val data: FloatArray)

    fun read(file: File): Array {
        DataInputStream(file.inputStream().buffered()).use { s ->
            val magic = ByteArray(6).also { s.readFully(it) }
            require(magic[0] == 0x93.toByte() && String(magic, 1, 5, Charsets.US_ASCII) == "NUMPY") { "${file.name} is not a .npy file" }
            val major = s.readUnsignedByte()
            s.readUnsignedByte()
            val headerLen = if (major == 1) {
                val b = ByteArray(2).also { s.readFully(it) }
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xffff
            } else {
                val b = ByteArray(4).also { s.readFully(it) }
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).int
            }
            val header = String(ByteArray(headerLen).also { s.readFully(it) }, Charsets.ISO_8859_1)
            val descr = Regex("'descr'\\s*:\\s*'([^']*)'").find(header)?.groupValues?.get(1) ?: error("${file.name}: no descr in the header")
            val fortran = Regex("'fortran_order'\\s*:\\s*(True|False)").find(header)?.groupValues?.get(1) ?: error("${file.name}: no fortran_order")
            require(fortran == "False") { "${file.name} is in Fortran order" }
            val shapeText = Regex("'shape'\\s*:\\s*\\(([^)]*)\\)").find(header)?.groupValues?.get(1) ?: error("${file.name}: no shape")
            val shape = shapeText.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.toInt() }
            val count = shape.fold(1L) { a, b -> a * b }
            require(count <= WordTable.MAX_VALUES) { "${file.name}: $count values is more than this version scans on the CPU (${WordTable.MAX_VALUES})" }
            val n = count.toInt()
            return when (descr) {
                "<f2" -> Array(shape, halves(ByteArray(n * 2).also { s.readFully(it) }))
                "<f4" -> {
                    val b = ByteBuffer.wrap(ByteArray(n * 4).also { s.readFully(it) }).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                    Array(shape, FloatArray(n).also { b.get(it) })
                }
                else -> error("${file.name}: dtype $descr (expected <f2 or <f4)")
            }
        }
    }

    /** Little-endian float16 values to float32. */
    fun halves(bytes: ByteArray): FloatArray {
        val n = bytes.size / 2
        return FloatArray(n) { i -> halfToFloat((bytes[2 * i].toInt() and 0xff) or ((bytes[2 * i + 1].toInt() and 0xff) shl 8)) }
    }

    /** IEEE 754 binary16 bits to float32 (exact: every half is a float). */
    fun halfToFloat(h: Int): Float {
        val sign = (h ushr 15) and 1
        val exp = (h ushr 10) and 0x1f
        val mant = h and 0x3ff
        val bits = when {
            exp == 0 && mant == 0 -> sign shl 31
            exp == 0 -> { // subnormal: normalize
                var e = 127 - 15 + 1
                var m = mant
                while (m and 0x400 == 0) { m = m shl 1; e-- }
                (sign shl 31) or (e shl 23) or ((m and 0x3ff) shl 13)
            }
            exp == 0x1f -> (sign shl 31) or (0xff shl 23) or (mant shl 13)
            else -> (sign shl 31) or ((exp - 15 + 127) shl 23) or (mant shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }
}
