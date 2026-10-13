# Eg2 Transcribe for Android (LiteRT)

Record with the microphone or open an audio file, and the app writes down the English words with their times, on the phone. The audio encoder of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) runs on LiteRT (CPU, GPU or the Qualcomm NPU) and turns every 40 ms of audio into a vector; a reader maps each vector into the model's text space and the nearest word of a word table is read out. Until a trained reader is installed, the app runs a stub reader whose words are placeholders: everything around it (log-mel, windows, encoder, timing) is the real pipeline.

The app has no INTERNET permission; audio, model and transcript stay on the phone.

## Build and install

```sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Android 8.0 (API 26) or newer, arm64. LiteRT 2.2.0 (`com.google.ai.edge.litert:litert`, set in `gradle.properties`). `./gradlew -Plitert.version=2.3.0 :app:assembleDebug` builds against 2.3.0: from 2.3.0 the GPU accelerator is the separate `litert-gpu` artifact, which `app/build.gradle.kts` then adds, and the 2.3.0 classes carry Kotlin 2.4 metadata, which this build's Kotlin 2.2.21 reads with `-Xskip-metadata-version-check` (added for 2.3.0 only). The 2.3.0 build has not been run on a phone, and its NPU libraries come from another source (NPU below).

## The model files

The encoder is two separate files, one per window length: the audio tower and `embed_audio` of `google/embeddinggemma-2` exported with litert-torch, with every RMSNorm rescaled so that its sums stay inside the float16 range (in float32 the rescaled graph gives the same numbers bit for bit; the GPU at float16 and the NPU need the rescaling) and the weights of the fully connected and convolution layers stored as float16:

| file | bytes | signature | inputs | outputs |
|---|---|---|---|---|
| `eg2_audio_f16safe_fp16w_w30.tflite` | 589,273,696 | `audio_30s` | `mel` [1, 3000, 128], `mask` [1, 3000] | `soft` [1, 750, 512], `mask` [1, 750] |
| `eg2_audio_f16safe_fp16w_w10.tflite` | 589,015,376 | `audio_10s` | `mel` [1, 1000, 128], `mask` [1, 1000] | `soft` [1, 250, 512], `mask` [1, 250] |

The CPU and the GPU run the 30 s file, the NPU runs the 10 s file (Windows, below). Masks are float32, 1 for a real frame or token. The files are not in this repository.

The float16 weights halve the files; the graph turns them back into float32 before use, so the CPU and the GPU at fp32 precision compute in float32 with rounded weights. Measured against the float32 PyTorch encoder on test windows: on the GPU of a Pixel 8a at fp32 precision the 30 s file's tokens are within 5.1 × 10⁻⁴ (cosine 0.9999999955 at the lowest, 7 windows, 2,306 tokens); on the NPU of a Galaxy S26 the 10 s file's tokens reach a cosine of 0.9999936 on average and 0.9975 at the lowest (9 windows, 1,806 tokens). The float32-weight pair `eg2_audio_f16safe_w30.tflite` / `_w10.tflite` (1,175,047,892 / 1,174,789,572 bytes) is used the same way.

A file named `<name>_w30.tflite` and one named `<name>_w10.tflite` are one model: the app offers a window length when its file is there. Any other `.tflite` with one or both of the signatures (such as an earlier export that holds both) is a model of its own; the NPU runs only a `_w10` file.

Put them on the phone in one of two ways:

- **adb**: start the app once (it creates its files directory and shows "MODEL MISSING"), then
  `adb push eg2_audio_f16safe_fp16w_w30.tflite /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/`
  `adb push eg2_audio_f16safe_fp16w_w10.tflite /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/`
  and come back to the app. A directory made by `adb shell` before the app has started once is not one the app can read.
- **On the phone**: tap **Change** next to Model, then **Choose a file on this phone…** and pick a file; the app copies it into its files directory (589 MB each; copy both files of a pair).

Other `.tflite` files in that directory show up under Model → Change.

## Using it

- **Record** starts the microphone (16 kHz mono); **Stop** ends the recording and the transcription starts. **Open audio file** takes WAV, FLAC, MP3, M4A/AAC, Ogg and whatever else the phone can decode; other sample rates are resampled to 16 kHz, stereo is mixed down. **Again** transcribes the last input again (for example on another backend), **Stop** ends a transcription after the current window.
- While it runs, the screen shows "Window k of N", the time since the start and the speed as a multiple of real time (audio seconds per second). The transcript is lower case without punctuation, a line with its time every 10 s of audio.
- Input shorter than 1 s is refused ("TOO SHORT"), and so is input quieter than RMS 0.005 (−46 dBFS, "NO SPEECH": the floor sits above two silent recordings made through a Galaxy S26 microphone). Without the microphone permission the screen says how to turn it on; a tap on that line opens the app's settings.

## Backends

| button | what runs the encoder |
|---|---|
| CPU | LiteRT's CPU accelerator (XNNPACK), 4 threads |
| GPU fp32 | LiteRT's GPU accelerator with fp32 precision |
| GPU default | LiteRT's GPU accelerator with its default precision (fp16 on most phones) |
| NPU | the Qualcomm Hexagon NPU with the 10 s file; the graph is compiled on the phone once per model file (needs the NPU libraries in the APK, below) |

GPU fp32 is the default. When the GPU cannot compile the graph, cannot run it, or gives non-finite values in a test run, the app runs the encoder on the CPU and says so in red under the backend buttons. When the NPU cannot be used (an unsupported chip, an APK without the libraries, no `_w10` file, a failed compile, or a graph LiteRT did not hand to the NPU), the app runs GPU fp32 instead and says why in the same place. The log-mel, the reader and the word search always run on the CPU.

## Windows

The encoder is called on fixed windows of 30 s (or 10 s), and a call knows nothing about the audio before its window, so the tokens at its start come out wrong. Each window therefore starts 96 tokens (3.84 s) before the end of the previous one and keeps only its tokens after that overlap; window 0 keeps all of them. On the right, token j reads mel frames 4j − 3 to 4j + 3, so a window's last token needs 3 frames past its 30 s (`right_extra_tokens: 1`, 640 samples, in the shared window contract): the app computes the log-mel of the whole recording and cuts windows out of those frames, so every window but the last has them, and at the end of the recording a token whose frames run past the last one is dropped (a recording of F frames keeps floor(F / 4) tokens). A window cut out of exactly 30 s of audio instead would lose its last token (cosine 0.49 on average to the continuous stream on the 54-minute fixture, measured for the iOS port). Measured on a Mac with the opening 10 minutes of the 54-minute fixture against the encoder run on the whole stream: 30 s windows with a 96-token overlap give a minimum token cosine of 0.99994 (0.99999994 on average) for 1.147× the encoder work; without the overlap the minimum is −0.16. **Windows → Change** offers the lengths the model has a file for and a few overlaps. With a file that holds both signatures, a last window with at most 10 s of audio runs on `audio_10s`.

The NPU runs 10 s windows. Its compiler builds the graph on the phone, and for the 30 s graph that step needed more memory than a Galaxy S26 (12 GB) had free (6.8 GB, the run was stopped; the 10 s graph compiles). With the same 96-token overlap a 10 s window keeps 154 new tokens of its 250, so the encoder does 1.623× the work of one pass over the audio instead of 1.147× with 30 s windows.

## The reader

The stub reader (now): identity (the 512-d soft token is its own vector), a table of 1,000 placeholder words with random rows, and a random blank vector; a token is blank when the blank vector is nearer than every word. The words are the 1,000 most frequent words of `reader/fixtures/long-speckled-band/transcript.txt` (`tools/make_stub_vocab.py`), the random numbers a fixed SplitMix64 sequence (`tools/stub_table.py` prints the same table). The screen says "stub: words are placeholders".

A trained reader replaces it with three files in `/sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader/`: `reader.tflite` (one input of T × 512 soft tokens, outputs T × d vectors and T blank logits), `word_table.npy` (float16 or float32, N × d) or `word_table.bin` (raw float16), and `vocab.txt` (N words, one per line, row order). `reader_config.json` may set `blank_logit_scale`. If anything does not match, the app keeps the stub and the Reader line says why. The word search runs on the CPU and takes tables up to 32 M values in this version.

## NPU

Qualcomm's license lets its runtime libraries ship only inside an application, so they are not in this repository. Before building an APK that can use the NPU:

```sh
tools/fetch_npu_libs.sh v81      # SM8850 (Galaxy S26); v79 SM8750, v75 SM8650, v73 SM8550
```

It downloads LiteRT's v2.2.0 dispatch library and JIT compiler plugin and QAIRT 2.47.0.260601 (about 2.4 GB) and puts 10 files in `app/src/main/jniLibs/arm64-v8a/` (ignored by git). `tools/fetch_npu_libs.sh --from <dir> v81` copies them from a directory that already has them. The app extracts them at install (`useLegacyPackaging`) and keeps them byte for byte (`keepDebugSymbols`), and its manifest declares `libcdsprpc.so`, the library the NPU is reached through. With LiteRT 2.3.0 the matching files come from the Maven artifact `com.google.ai.edge.litert:litert-npu-runtime-qualcomm` and QAIRT 2.50.0.260828 instead; this project has not been run that way.

LiteRT 2.2.0 has an NPU path for the Qualcomm SM8550, SM8650, SM8750 and SM8850. When the NPU is picked for a model file that has not been compiled on this phone yet, LiteRT's Qualcomm compiler builds the graph on the phone (the screen says it compiles the encoder for the NPU, about 2–3 minutes, and counts the time), with the CPU accelerator for any operation the NPU would not take, and keeps the result in the app's cache: `cache/npu_jit/<file name>-<12 hex digits of the file's SHA-256>/`. Later starts read it from there. A file with other bytes under the same name gets a new entry, and the old one is removed; Android's "Clear cache" for the app removes them all. The app counts a run as NPU only when LiteRT's log says its dispatch delegate took the graph (a graph the compiler could not take would otherwise run on the CPU without an error). On a Galaxy S26 (SM8850) compiling `eg2_audio_f16safe_fp16w_w10.tflite` took 121.6 s and left a 620 MB entry; a later start read it in 0.6 s.

The LiteRT 2.2.0 AAR declares FOREGROUND_SERVICE, WAKE_LOCK, ACCESS_NETWORK_STATE and RECEIVE_BOOT_COMPLETED for its Play model delivery, which this app does not use; they are in the merged manifest.

## What it did on a Galaxy S26 (2026-10-13)

One run of the device check below with this version: the fp16-weight pair above, the stub reader, LiteRT 2.2.0, a debug build:

| input | backend | time | speed |
|---|---|---|---|
| 30 s | CPU (4 threads) | 0.74 s | 40.6× real time |
| 30 s | GPU fp32 | 0.52 s | 57.7× |
| 30 s | NPU, 10 s windows | 0.56 s, after compiling for 121.6 s | 54.0× |
| 6.6 s (FLAC) | NPU, the compiled graph read from the cache in 0.66 s | 0.27 s | 24.3× |
| 54 min 17 s (the LibriVox fixture, FLAC) | NPU, 529 windows of 10 s | 52.5 s | 62.0× |
| 54 min 17 s | GPU fp32, 125 windows of 30 s | 66.9 s | 48.7× |

The times cover everything from the log-mel of window 1 to the last word (log-mel, encoder, reader and word search). The encoder took a median of 98.6 ms per 10 s window on the NPU and 572.7 ms per 30 s window on the GPU; the phone was at thermal status 3 by the two 54-minute runs (an earlier build with the float32-weight file did the same 54 minutes in 42.1 s on GPU fp32 at thermal status 1 to 2). Reading the file came on top: the 54-minute FLAC took 43.9 s and 44.4 s to decode (the earlier build's decoder loop, which waited up to 10 ms for each decoded block, took 616 s). The phone's FLAC decoder returned that file 998 samples (62 ms) short both times: its last 3 FLAC frames, which are 402, 402 and 194 samples long; the run report gives the declared and the decoded length.

On the CPU and the GPU the words matched, token for token, the same pipeline run on a computer with the same file and the same placeholder table, and the soft tokens matched the encoder run on the whole recording at once: on the 54 minutes on GPU fp32, cosine 0.9999998 on average and 0.9977 at the lowest; the last token of the 30 s recording, the one whose frames end with the recording, 0.99999999999.

The NPU is a choice, not the default. It computes in float16, and it misses the strict per-token check at quiet moments: on the 30 s recording its tokens reached a cosine of 0.999996 on average and 0.9998 at the lowest, but over the 54 minutes 37 of 81,434 tokens fell under 0.99 (0.45 at the lowest, at 37:16), none of them digital silence and most in quiet passages (a median mel level of −4.6 against −2.9 for all tokens), while the GPU fp32 run of the same recording stayed at 0.9977 or above at those tokens. With the placeholder table, 385 of the 81,434 tokens read another word than on the computer. It stays in the app to compare word error rates once the trained reader is in. Compiling for the NPU took the app to a peak of 4,761,784 kB of memory (VmHWM, 4.5 GiB); the 54-minute run that read the cache, 1,536,140 kB (1.5 GiB).

A start with the phone in landscape showed every row and button inside the window, clear of the navigation bar, the whole screen scrolling as one. Silence, a 0.5 s file, a missing model file and the microphone with and without its permission each ended with their sentence on the screen; turning the phone during a run and a second start during a run were handled. A trained reader with a larger word table will take more time than the 1,000-word placeholder.

## How the device check works

A start with extras runs the same code as the buttons and writes a JSON report: `adb shell am start -n io.github.johnrocky.eg2transcribe/.MainActivity --es file audio/f30.wav --es backend gpu_fp32 --es report r_gpu.json` (`--ei mic_ms 3000` records from the microphone instead; `--ez dump true` also writes the soft tokens and the per-token words). The report (`files/<report>`) holds the backend asked for and the one used, the compile time, LiteRT's own log lines about the delegate, the window count, the times, ×RT, the transcript, or the error. A script on the computer runs the same windows through the same `.tflite` with the same stub table and compares the words and the soft tokens token by token. Such a start shows over the lock screen and keeps the display on.

## Code

`MelFrontEnd.kt` (log-mel, written from transformers' `Gemma4AudioFeatureExtractor`), `Windowing.kt` (the window plan), `Encoder.kt` (LiteRT `CompiledModel`), `Pipeline.kt` (the window loop: log-mel of the next window while the encoder runs, reader and words on another thread), `Reader.kt` / `TfliteReader.kt` / `WordTable.kt` / `Ctc.kt` (vectors to words), `AudioDecoder.kt` / `Resampler.kt` / `Mic.kt` (input), `Jobs.kt` / `Engine.kt` / `Session.kt` / `MainActivity.kt` (the app). `./gradlew :app:testDebugUnitTest` runs the JVM tests: the log-mel against the extractor's output for two LibriSpeech utterances (`app/src/test/resources/mel`), the window plan, the word table, the stub's numbers, the collapse and the window loop with a fake encoder.

## Credits and licenses

- Code: Apache-2.0.
- EmbeddingGemma 2 (Google): Apache-2.0; use must follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- `app/src/test/resources/mel`: two LibriSpeech test-clean utterances (2300-131720-0000, 121-121726-0001) as 16-bit samples, and their log-mel rows. LibriSpeech ASR corpus, V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books", ICASSP 2015, https://www.openslr.org/12, CC BY 4.0.
- `app/src/main/assets/stub_vocab.txt`: words from Project Gutenberg eBook #1661, *The Adventures of Sherlock Holmes* (public domain in the USA).
- The idea comes from Shahir's post (@TslShahir, 2026-10-13) and the *Embedding Gemma Transcribe* Mac app. This is an independent project, not affiliated with Google.
