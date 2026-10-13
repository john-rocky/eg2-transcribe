package io.github.johnrocky.eg2transcribe

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import kotlin.math.sqrt

/**
 * Microphone recording: 16 kHz mono 16-bit from the VOICE_RECOGNITION source, written to a file while it records (an
 * hour is 115 MB), until stopped or [MAX_SECONDS]. Without RECORD_AUDIO it records nothing and returns [Result.Denied].
 */
object Mic {
    const val MAX_SECONDS = 3600

    sealed interface Result {
        class Denied(val text: String) : Result
        class Failed(val error: Throwable) : Result
        class Recorded(val file: File, val samples: Long, val rms: Double) : Result
    }

    fun granted(context: Context) = context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    fun deniedText(context: Context): String {
        val label = context.applicationInfo.loadLabel(context.packageManager)
        return "The microphone is off for this app. Allow it when Android asks, or in Settings > Apps > $label > Permissions > Microphone (tap here)."
    }

    /** Records into [out] on the calling thread until [keepGoing] is false. [onLevel] gets each 50 ms block's RMS and the seconds so far. */
    @SuppressLint("MissingPermission") // granted() is checked first
    fun record(context: Context, out: File, keepGoing: () -> Boolean, onLevel: (Double, Double) -> Unit = { _, _ -> }): Result {
        if (!granted(context)) return Result.Denied(deniedText(context))
        var samples = 0L
        var sum = 0.0
        try {
            val rate = MelFrontEnd.SAMPLE_RATE
            val minBuf = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, rate))
            try {
                check(rec.state == AudioRecord.STATE_INITIALIZED) { "the microphone did not open (another app may be using it)" }
                val block = ShortArray(rate / 20)
                val bytes = ByteArray(block.size * 2)
                val max = rate.toLong() * MAX_SECONDS
                BufferedOutputStream(FileOutputStream(out), 1 shl 16).use { os ->
                    rec.startRecording()
                    try {
                        while (keepGoing() && samples < max) {
                            val n = rec.read(block, 0, block.size)
                            if (n < 0) error("AudioRecord.read returned $n")
                            var s = 0.0
                            for (i in 0 until n) {
                                val v = block[i].toInt()
                                bytes[2 * i] = v.toByte()
                                bytes[2 * i + 1] = (v shr 8).toByte()
                                s += (v / 32768.0) * (v / 32768.0)
                            }
                            os.write(bytes, 0, n * 2)
                            samples += n
                            sum += s
                            if (n > 0) onLevel(sqrt(s / n), samples.toDouble() / rate)
                        }
                    } finally {
                        rec.stop()
                    }
                }
            } finally {
                rec.release()
            }
        } catch (t: Throwable) {
            return Result.Failed(t)
        }
        return Result.Recorded(out, samples, if (samples == 0L) 0.0 else sqrt(sum / samples))
    }
}
