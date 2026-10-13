package io.github.johnrocky.eg2transcribe

import android.content.Context
import android.content.ContextWrapper
import android.os.Build
import android.system.Os
import com.google.ai.edge.litert.NpuCompatibilityChecker
import java.io.File
import java.security.MessageDigest

/**
 * The Qualcomm HTP (Hexagon NPU) path of LiteRT 2.2.0: the graph is compiled on the phone (JIT) by LiteRT's Qualcomm
 * compiler plugin and run through its dispatch library and Qualcomm's QNN runtime. Qualcomm's license lets those
 * libraries ship only inside an application, so they are not in this repository: tools/fetch_npu_libs.sh puts them
 * in app/src/main/jniLibs/arm64-v8a before the build (README.md, NPU). This object says, before a compile, whether
 * this phone and this APK can try the NPU, and if not, why; and it keeps the compiled graphs (the JIT cache).
 */
object Npu {
    /** The SoCs LiteRT 2.2.0 lists for Qualcomm and their Hexagon version. */
    private val HEXAGON = mapOf("SM8550" to "V73", "SM8650" to "V75", "SM8750" to "V79", "SM8850" to "V81")

    /** LiteRT's dispatch library and JIT compiler plugin, then QNN's; QnnHtpPrepare, QnnIr and QnnSaver are the JIT compiler's. */
    private val COMMON = listOf(
        "libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Qualcomm.so",
        "libQnnHtp.so", "libQnnSystem.so", "libQnnHtpPrepare.so", "libQnnIr.so", "libQnnSaver.so",
    )

    fun soc(): String = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else "unknown (Android ${Build.VERSION.RELEASE})"

    /** This phone's Hexagon version, or null when LiteRT has no Qualcomm NPU path for its SoC. */
    fun hexagon(): String? {
        if (Build.VERSION.SDK_INT < 31) return null
        val v = HEXAGON[Build.SOC_MODEL] ?: return null
        return v.takeIf { runCatching { NpuCompatibilityChecker.Qualcomm.isDeviceSupported() }.getOrDefault(false) }
    }

    fun libraries(hexagon: String): List<String> = COMMON + listOf("libQnnHtp${hexagon}Stub.so", "libQnnHtp${hexagon}Skel.so")

    /** True when this APK carries every library this phone's NPU path loads (extracted to the native library dir). */
    fun packaged(context: Context): Boolean {
        val hexagon = hexagon() ?: return false
        val dir = File(context.applicationInfo.nativeLibraryDir)
        return libraries(hexagon).all { File(dir, it).isFile }
    }

    /** Null when the NPU can be tried here; otherwise the sentence for the screen. */
    fun unavailable(context: Context): String? {
        val hexagon = hexagon() ?: return "LiteRT ${BuildConfig.LITERT_VERSION} has no NPU path for this phone's chip (${soc()}); it lists Qualcomm ${HEXAGON.keys.joinToString(", ")}."
        val dir = File(context.applicationInfo.nativeLibraryDir)
        val missing = libraries(hexagon).filterNot { File(dir, it).isFile }
        if (missing.isNotEmpty()) return "This build of the app does not carry the Qualcomm NPU libraries for Hexagon $hexagon (missing ${missing.size}: ${missing.joinToString(", ")}). Build it after tools/fetch_npu_libs.sh (README.md, NPU)."
        return null
    }

    // ---- the JIT cache -------------------------------------------------------------------------------------------

    /**
     * The root of LiteRT's JIT cache. LiteRT 2.2.0 keeps one compiled graph per model file as
     * `<root>/<model file stem>/<hash of the model's bytes>/<hash of the options>.tflite` and takes the root from the
     * Context its Environment is created with (Context.getCacheDir()); [cacheContext] points that at this directory,
     * so the app's own files in cache/ (the decoded input) stay out of it.
     */
    fun cacheRoot(context: Context): File = File(context.cacheDir, "npu_jit")

    /** A Context whose cache dir is [cacheRoot], for Environment.create (the only thing it reads from the Context). */
    fun cacheContext(context: Context): Context {
        val root = cacheRoot(context).apply { mkdirs() }
        return object : ContextWrapper(context) {
            override fun getCacheDir(): File = root
        }
    }

    /** Links to the model files, named with their SHA-256 (cache/npu_models/<stem>-<sha256[0:12]>.tflite). */
    private fun linkDir(context: Context): File = File(context.cacheDir, "npu_models")

    /** What [prepare] found: the path the NPU compiles, and where its cache entry lives. */
    class Prepared(
        val file: File,
        /** The path given to LiteRT: a link named `<stem>-<sha256[0:12]>.tflite`, or [file] when no link could be made. */
        val path: File,
        val sha256: String,
        val sha256Ms: Double,
        /** LiteRT's cache directory for [path] (`<root>/<stem of path>`). */
        val cacheDir: File,
        /** True when that directory (or one named after the file itself) already holds a compiled graph. */
        val cachedBefore: Boolean,
        val note: String?,
        /** The directory named after [file] itself, where LiteRT would keep the graph if it named it by the link's target. */
        val plainCacheDir: File,
    ) {
        val sha12: String get() = sha256.take(12)

        /** The cache directories that hold a compiled graph now. */
        fun dirsWithGraph(): List<File> = listOf(cacheDir, plainCacheDir).distinct().filter { holdsGraph(it) }
    }

    /**
     * Gets [file] ready for an NPU compile (work thread): its SHA-256 (kept next to the link with the file's size and
     * time, so it is read once per file), a link `<stem>-<sha256[0:12]>.tflite` to it, so LiteRT's cache directory of
     * this file carries the hash of the bytes it was compiled from; and the cache entries of earlier contents of the
     * same file name are removed (each is about 0.6 GB).
     */
    fun prepare(context: Context, file: File): Prepared {
        val links = linkDir(context).apply { mkdirs() }
        val t = System.nanoTime()
        val sha = sha256Cached(file, File(links, file.name + ".sha256"))
        val shaMs = (System.nanoTime() - t) / 1e6
        val stem = file.name.removeSuffix(".tflite")
        val linkName = linkName(stem, sha)
        val link = File(links, "$linkName.tflite")
        var note: String? = null
        val path = runCatching {
            if (runCatching { Os.readlink(link.path) }.getOrNull() != file.absolutePath) {
                link.delete()
                Os.symlink(file.absolutePath, link.path)
            }
            link
        }.getOrElse { e ->
            note = "no link (${e.message}); the cache entry is named after the file"
            file
        }
        val root = cacheRoot(context)
        // Entries and links of earlier contents of this file name.
        root.listFiles { f -> f.isDirectory && f.name.startsWith("$stem-") && f.name != linkName && isLinkName(f.name, stem) }?.forEach { it.deleteRecursively() }
        links.listFiles { f -> f.name.startsWith("$stem-") && f.name != link.name && isLinkName(f.name.removeSuffix(".tflite"), stem) }?.forEach { it.delete() }
        val cacheDir = File(root, path.name.removeSuffix(".tflite"))
        val plain = File(root, stem)
        return Prepared(file, path, sha, shaMs, cacheDir, holdsGraph(cacheDir) || (plain != cacheDir && holdsGraph(plain)), note, plain)
    }

    /** `<stem>-<first 12 hex digits of the SHA-256>`. */
    fun linkName(stem: String, sha256: String): String = "$stem-${sha256.take(12)}"

    private fun isLinkName(name: String, stem: String): Boolean = name.length == stem.length + 13 && name.substring(stem.length + 1).all { it in '0'..'9' || it in 'a'..'f' }

    /** True when [dir] holds a compiled graph (a .tflite at any depth). */
    fun holdsGraph(dir: File): Boolean = dir.isDirectory && dir.walkTopDown().any { it.isFile && it.name.endsWith(".tflite") }

    /** Bytes under [dir]. */
    fun bytes(dir: File): Long = if (!dir.exists()) 0L else dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    private fun sha256Cached(file: File, memo: File): String {
        val key = "${file.length()} ${file.lastModified()}"
        runCatching {
            val (k1, k2, sha) = memo.readText().trim().split(" ")
            if ("$k1 $k2" == key && sha.length == 64) return sha
        }
        val sha = sha256(file)
        runCatching { memo.writeText("$key $sha\n") }
        return sha
    }

    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { s ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = s.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
