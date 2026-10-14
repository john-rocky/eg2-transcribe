// Eg2 Transcribe: English speech to text on the phone. Microphone or audio file -> log-mel (Kotlin) -> the audio
// encoder of EmbeddingGemma 2 on LiteRT (CPU, GPU or NPU) -> reader -> nearest words -> transcript.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

val litertVersion: String = providers.gradleProperty("litert.version").get()
val litertAtLeast23 = litertVersion.split('.').let { v -> v[0].toInt() > 2 || (v[0].toInt() == 2 && v[1].toInt() >= 3) }

android {
    namespace = "io.github.johnrocky.eg2transcribe"
    compileSdk = 36

    defaultConfig {
        applicationId = "io.github.johnrocky.eg2transcribe"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
        buildConfigField("String", "LITERT_VERSION", "\"$litertVersion\"")
    }

    buildFeatures { buildConfig = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs {
            // The NPU path loads Qualcomm's libraries by path from the app's native library dir, so they are extracted
            // at install; and kept byte for byte (the strip step changes some of them).
            useLegacyPackaging = true
            keepDebugSymbols += "**/*.so"
        }
    }

    testOptions {
        unitTests.all {
            it.testLogging { events("passed", "failed", "skipped"); showStandardStreams = true }
            // -Peg2.replay=<dir>: ReplayTest runs the pipeline on encoder outputs computed on a computer (its KDoc).
            providers.gradleProperty("eg2.replay").orNull?.let { dir -> it.systemProperty("eg2.replay", dir) }
            // -Peg2.flac=<list>: FlacTest decodes every file of the list against ffmpeg's samples (its KDoc).
            providers.gradleProperty("eg2.flac").orNull?.let { list -> it.systemProperty("eg2.flac", list) }
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        // LiteRT 2.3.0 is compiled with Kotlin 2.4 (metadata 2.4.0), which this build's Kotlin 2.2.21 refuses without the flag.
        if (litertAtLeast23) freeCompilerArgs.add("-Xskip-metadata-version-check")
    }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:$litertVersion")
    // From 2.3.0 the GPU accelerator is its own artifact.
    if (litertAtLeast23) implementation("com.google.ai.edge.litert:litert-gpu:$litertVersion")
    testImplementation(libs.junit)
}
