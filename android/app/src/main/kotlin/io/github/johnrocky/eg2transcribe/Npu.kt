package io.github.johnrocky.eg2transcribe

import android.content.Context
import android.os.Build
import com.google.ai.edge.litert.NpuCompatibilityChecker
import java.io.File

/**
 * The Qualcomm HTP (Hexagon NPU) path of LiteRT 2.2.0: the graph is compiled on the phone (JIT) by LiteRT's Qualcomm
 * compiler plugin and run through its dispatch library and Qualcomm's QNN runtime. Qualcomm's license lets those
 * libraries ship only inside an application, so they are not in this repository: tools/fetch_npu_libs.sh puts them
 * in app/src/main/jniLibs/arm64-v8a before the build (README.md, NPU). This object says, before a compile, whether
 * this phone and this APK can try the NPU, and if not, why.
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
}
