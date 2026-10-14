package io.github.johnrocky.eg2transcribe

import android.content.Context
import java.io.File

/**
 * The example bundled in the APK: LibriSpeech test-clean 1284-1180-0000 (8.1 s of read speech, CC BY 4.0), so the app
 * can be tried without an audio file of one's own. It is copied to the cache once and then read like an opened file.
 */
object Examples {
    const val NAME = "1284-1180-0000.flac"
    const val LABEL = "8 s of read speech (LibriSpeech test-clean 1284-1180-0000)"
    private const val ASSET = "examples/$NAME"

    fun file(context: Context): File {
        val f = File(context.cacheDir, "example_$NAME")
        if (!f.isFile || f.length() == 0L) {
            val tmp = File(context.cacheDir, "example_$NAME.tmp")
            context.assets.open(ASSET).use { input -> tmp.outputStream().use { input.copyTo(it) } }
            if (!tmp.renameTo(f)) throw IllegalStateException("could not write ${f.name}")
        }
        return f
    }
}
