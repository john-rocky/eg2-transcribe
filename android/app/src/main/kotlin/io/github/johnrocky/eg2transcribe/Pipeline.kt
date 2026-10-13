package io.github.johnrocky.eg2transcribe

import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import java.util.concurrent.ExecutorService
import java.util.concurrent.Future

/** Width of a soft token (the encoder's output, the reader's input). */
const val SOFT_DIM = 512

/** One encoder call: mel [frames, 128] and mask [frames] in, soft tokens [frames / 4, 512] and their mask out. */
interface EncoderRunner {
    /** The window sizes, in mel frames, this encoder has a signature for (1000 = 10 s, 3000 = 30 s). */
    val windowSizes: Set<Int>

    fun run(frames: Int, mel: FloatArray, mask: FloatArray): EncoderOutput
}

/** [ms] = writing the inputs + run + reading the outputs (a GPU run returns before the work; the read waits for it). */
class EncoderOutput(val soft: FloatArray, val mask: FloatArray, val ms: Double)

/**
 * PCM to words, window by window ([WindowPlan]): the log-mel of window k + 1 is computed on [prep] while the encoder
 * runs window k on the calling thread (the thread that owns the encoder), and the reader, the table scan and the CTC
 * collapse of window k run on [decode] (one thread, so words arrive in order) with the scan split over [scan].
 */
class Pipeline(
    private val encoder: EncoderRunner,
    private val reader: Reader,
    private val table: WordTable,
    private val prep: ExecutorService,
    private val decode: ExecutorService,
    private val scan: ExecutorService?,
    private val scanThreads: Int,
) {
    interface Listener {
        /** After window [k] of [count]: [audioSeconds] of the recording are encoded, [elapsedMs] since the start. */
        fun onWindow(k: Int, count: Int, audioSeconds: Double, elapsedMs: Double) {}

        /** New words, in order (on the decode thread). */
        fun onWords(words: List<Word>) {}
    }

    /** Optional per-token outputs for checking a run against another pipeline. */
    class Dump(val soft: OutputStream?, val tokens: OutputStream?)

    class Result(
        val plan: WindowPlan,
        val words: List<Word>,
        val encoderMs: List<Double>,
        val windowFrames: List<Int>,
        val melMs: Double,
        val decodeMs: Double,
        val elapsedMs: Double,
        val cancelled: Boolean,
        val blanks: Int,
        val warnings: List<String>,
    )

    fun run(pcm: PcmSource, windowFrames: Int, overlapTokens: Int, listener: Listener, cancel: () -> Boolean = { false }, dump: Dump? = null): Result {
        require(windowFrames in encoder.windowSizes) { "the encoder has no ${windowFrames / 100} s signature (has ${encoder.windowSizes.sorted().map { it / 100 }} s)" }
        val t0 = System.nanoTime()
        val plan = WindowPlan(MelFrontEnd.frames(pcm.length), windowFrames, overlapTokens)
        val ctc = Ctc(table.vocab)
        val words = ArrayList<Word>()
        val warnings = ArrayList<String>()
        val encoderMs = ArrayList<Double>()
        val usedFrames = ArrayList<Int>()
        var melNs = 0L
        var decodeNs = 0L
        var blanks = 0
        val front = ThreadLocal.withInitial { MelFrontEnd() }
        val mels = Array(2) { FloatArray(windowFrames * MelFrontEnd.NMEL) }
        val masks = Array(2) { FloatArray(windowFrames) }

        fun prepare(k: Int): Int {
            val a = System.nanoTime()
            val slot = k % 2
            val n = plan.validFrames(k)
            val mel = mels[slot]
            front.get().compute(pcm, plan.start(k).toLong(), n, mel)
            java.util.Arrays.fill(mel, n * MelFrontEnd.NMEL, mel.size, 0f)
            java.util.Arrays.fill(masks[slot], 0, n, 1f)
            java.util.Arrays.fill(masks[slot], n, windowFrames, 0f)
            synchronized(this) { melNs += System.nanoTime() - a }
            return slot
        }

        fun decodeWindow(k: Int, soft: FloatArray, valid: Int) {
            val a = System.nanoTime()
            val from = plan.keepFrom(k)
            val n = minOf(valid, plan.keepUntil(k)) - from
            if (n <= 0) return
            val vectors = FloatArray(n * reader.dim)
            val blank = DoubleArray(n)
            reader.read(soft, from * SOFT_DIM, n, vectors, blank)
            val rows = IntArray(n)
            val cos = DoubleArray(n)
            val second = DoubleArray(n)
            table.nearestAll(vectors, n, scan, scanThreads, rows, cos, second)
            val ids = IntArray(n) { t -> if (rows[t] < 0 || reader.isBlank(blank[t], cos[t])) Ctc.BLANK else rows[t] }
            blanks += ids.count { it == Ctc.BLANK }
            val first = plan.firstToken(k) + from
            val fresh = ctc.feed(ids, n, first)
            words += fresh
            dump?.soft?.let { out ->
                val b = ByteBuffer.allocate(n * SOFT_DIM * 4).order(ByteOrder.LITTLE_ENDIAN)
                b.asFloatBuffer().put(soft, from * SOFT_DIM, n * SOFT_DIM)
                out.write(b.array())
            }
            dump?.tokens?.let { out ->
                // Per token: global index (int32), row or -1 (int32), best cos, second cos, blank logit (float64).
                val b = ByteBuffer.allocate(n * 32).order(ByteOrder.LITTLE_ENDIAN)
                for (t in 0 until n) {
                    b.putInt(first + t); b.putInt(ids[t]); b.putDouble(cos[t]); b.putDouble(second[t]); b.putDouble(blank[t])
                }
                out.write(b.array())
            }
            decodeNs += System.nanoTime() - a
            if (fresh.isNotEmpty()) listener.onWords(fresh)
        }

        val pending = ArrayDeque<Future<*>>()
        var cancelled = false
        var next: Future<Int> = prep.submit<Int> { prepare(0) }
        try {
            for (k in 0 until plan.count) {
                if (cancel()) {
                    cancelled = true
                    next.cancel(true)
                    break
                }
                val slot = next.get()
                if (k + 1 < plan.count) {
                    val k1 = k + 1
                    next = prep.submit<Int> { prepare(k1) }
                }
                // The last window runs on the 10 s signature when its real frames fit there (same tokens: the padded
                // frames are masked and the encoder looks only back in time).
                val frames = if (windowFrames > WindowPlan.FRAMES_10S && plan.validFrames(k) <= WindowPlan.FRAMES_10S && WindowPlan.FRAMES_10S in encoder.windowSizes) WindowPlan.FRAMES_10S else windowFrames
                val mel = if (frames == windowFrames) mels[slot] else mels[slot].copyOf(frames * MelFrontEnd.NMEL)
                val mask = if (frames == windowFrames) masks[slot] else masks[slot].copyOf(frames)
                val out = encoder.run(frames, mel, mask)
                encoderMs += out.ms
                usedFrames += frames
                var valid = 0
                for (i in 0 until frames / 4) if (out.mask[i] > 0.5f) valid++
                val expected = plan.validTokens(k)
                if (valid != expected) {
                    warnings += "window $k: the encoder marked $valid tokens valid, the plan expects $expected"
                    valid = minOf(valid, expected)
                }
                while (pending.size >= 2) pending.removeFirst().get() // bounded queue: at most 2 windows wait for decoding
                val v = valid
                pending.addLast(decode.submit { decodeWindow(k, out.soft, v) })
                val done = plan.kept(k).last + 1
                listener.onWindow(k, plan.count, done * Ctc.SECONDS_PER_TOKEN, (System.nanoTime() - t0) / 1e6)
            }
            while (pending.isNotEmpty()) pending.removeFirst().get()
        } catch (t: Throwable) {
            next.cancel(true)
            for (f in pending) f.cancel(true)
            throw t
        }
        dump?.soft?.flush()
        dump?.tokens?.flush()
        val elapsed = (System.nanoTime() - t0) / 1e6
        return Result(plan, words, encoderMs, usedFrames, melNs / 1e6, decodeNs / 1e6, elapsed, cancelled, blanks, warnings)
    }
}

