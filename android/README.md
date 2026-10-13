# Eg2 Transcribe for Android (LiteRT)

Record with the microphone or open an audio file, and the app writes down the English words with their times, on the phone. The audio encoder of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) runs on LiteRT (CPU, GPU or the Qualcomm NPU) and turns every 40 ms of audio into a vector; a reader maps each vector into the model's text space and the nearest word of a word table is read out. Until a trained reader is installed, the app runs a stub reader whose words are placeholders: everything around it (log-mel, windows, encoder, timing) is the real pipeline.

The app has no INTERNET permission; audio, model and transcript stay on the phone.

## Build and install

```sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Android 8.0 (API 26) or newer, arm64. LiteRT 2.2.0 (`com.google.ai.edge.litert:litert`, set in `gradle.properties`). `./gradlew -Plitert.version=2.3.0 :app:assembleDebug` builds against 2.3.0: from 2.3.0 the GPU accelerator is the separate `litert-gpu` artifact, which `app/build.gradle.kts` then adds, and the 2.3.0 classes carry Kotlin 2.4 metadata, which this build's Kotlin 2.2.21 reads with `-Xskip-metadata-version-check` (added for 2.3.0 only). The 2.3.0 build has not been run on a phone, and its NPU libraries come from another source (NPU below).

## The model file

The encoder is a separate file, `eg2_audio_fp32.tflite` (1,176,601,684 bytes): the audio tower and `embed_audio` of `google/embeddinggemma-2` exported with litert-torch in float32, with two signatures that share the weights:

| signature | inputs | outputs |
|---|---|---|
| `audio_30s` | `mel` [1, 3000, 128], `mask` [1, 3000] | `soft` [1, 750, 512], `mask` [1, 750] |
| `audio_10s` | `mel` [1, 1000, 128], `mask` [1, 1000] | `soft` [1, 250, 512], `mask` [1, 250] |

Masks are float32, 1 for a real frame or token. Any `.tflite` with these signatures (one of them is enough) can be used. The file is not in this repository.

Put it on the phone in one of two ways:

- **adb**: start the app once (it creates its files directory and shows "MODEL MISSING"), then
  `adb push eg2_audio_fp32.tflite /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/`
  and come back to the app. A directory made by `adb shell` before the app has started once is not one the app can read.
- **On the phone**: tap **Change** next to Model, then **Choose a file on this phone…** and pick the file; the app copies it into its files directory (it needs 1.2 GB free).

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
| NPU | the Qualcomm Hexagon NPU; the graph is compiled on the phone when the NPU is picked (needs the NPU libraries in the APK, below) |

When the GPU cannot compile the graph, cannot run it, or gives non-finite values in a test run, the app runs the encoder on the CPU and says so in red under the backend buttons. The NPU does not fall back: the screen says why it cannot run (an unsupported chip, or an APK without the libraries). The log-mel, the reader and the word search always run on the CPU.

## Windows

The encoder is called on fixed windows of 30 s (or 10 s), and a call knows nothing about the audio before its window, so the tokens at its start come out wrong. Each window therefore starts 96 tokens (3.84 s) before the end of the previous one and keeps only its tokens after that overlap; window 0 keeps all of them. On the right, token j reads mel frames 4j − 3 to 4j + 3, so a window's last token needs 3 frames past its 30 s (`right_extra_tokens: 1`, 640 samples, in the shared window contract): the app computes the log-mel of the whole recording and cuts windows out of those frames, so every window but the last has them, and at the end of the recording a token whose frames run past the last one is dropped (a recording of F frames keeps floor(F / 4) tokens). A window cut out of exactly 30 s of audio instead would lose its last token (cosine 0.49 on average to the continuous stream on the 54-minute fixture, measured for the iOS port). Measured on a Mac with the opening 10 minutes of the 54-minute fixture against the encoder run on the whole stream: 30 s windows with a 96-token overlap give a minimum token cosine of 0.99994 (0.99999994 on average) for 1.147× the encoder work; without the overlap the minimum is −0.16. **Windows → Change** offers 30 s or 10 s and a few overlaps. A last window with at most 10 s of audio runs on `audio_10s`.

## The reader

The stub reader (now): identity (the 512-d soft token is its own vector), a table of 1,000 placeholder words with random rows, and a random blank vector; a token is blank when the blank vector is nearer than every word. The words are the 1,000 most frequent words of `reader/fixtures/long-speckled-band/transcript.txt` (`tools/make_stub_vocab.py`), the random numbers a fixed SplitMix64 sequence (`tools/stub_table.py` prints the same table). The screen says "stub: words are placeholders".

A trained reader replaces it with three files in `/sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader/`: `reader.tflite` (one input of T × 512 soft tokens, outputs T × d vectors and T blank logits), `word_table.npy` (float16 or float32, N × d) or `word_table.bin` (raw float16), and `vocab.txt` (N words, one per line, row order). `reader_config.json` may set `blank_logit_scale`. If anything does not match, the app keeps the stub and the Reader line says why. The word search runs on the CPU and takes tables up to 32 M values in this version.

## NPU

Qualcomm's license lets its runtime libraries ship only inside an application, so they are not in this repository. Before building an APK that can use the NPU:

```sh
tools/fetch_npu_libs.sh v81      # SM8850 (Galaxy S26); v79 SM8750, v75 SM8650, v73 SM8550
```

It downloads LiteRT's v2.2.0 dispatch library and JIT compiler plugin and QAIRT 2.47.0.260601 (about 2.4 GB) and puts 10 files in `app/src/main/jniLibs/arm64-v8a/` (ignored by git). `tools/fetch_npu_libs.sh --from <dir> v81` copies them from a directory that already has them. The app extracts them at install (`useLegacyPackaging`) and keeps them byte for byte (`keepDebugSymbols`). With LiteRT 2.3.0 the matching files come from the Maven artifact `com.google.ai.edge.litert:litert-npu-runtime-qualcomm` and QAIRT 2.50.0.260828 instead; this project has not been run that way.

The LiteRT 2.2.0 AAR declares FOREGROUND_SERVICE, WAKE_LOCK, ACCESS_NETWORK_STATE and RECEIVE_BOOT_COMPLETED for its Play model delivery, which this app does not use; they are in the merged manifest.

## What it did on a Galaxy S26 (2026-10-13)

One run of the device check below, with the encoder file above, the stub reader and LiteRT 2.2.0 (a debug build):

| input | backend | time | speed |
|---|---|---|---|
| 54 min 17 s (the LibriVox fixture, 125 windows) | GPU fp32 | 42.1 s | 77.4× real time |
| 30 s | CPU (4 threads) | 0.92 s | 32.6× |
| 30 s | GPU fp32 | 0.55 s | 54.7× |
| 30 s | GPU default | the GPU could not build the graph at its default precision; the app ran the CPU instead (0.84 s) and said so | |
| 30 s, the fp16-weight file (589 MB, `audio_30s` only) | GPU fp32 | 0.69 s | 43.5× |
| 30 s | NPU | not run: compiling the 30 s graph for the NPU ran this phone out of memory in an earlier test | |

The times cover everything from the log-mel of window 1 to the last word (log-mel, encoder, reader and word search), with the encoder at a median of 329 ms per 30 s window on the GPU. Reading the file came on top: the 54-minute FLAC took 616 s through the decoder loop of that build, which waited up to 10 ms for each decoded block; the loop is fixed in this version and has not been timed on the phone again. The words matched, token for token, the same pipeline run on a computer with the same file and the same placeholder table, and the soft tokens matched the encoder run on the whole recording at once (cosine 0.9999998 on average, 0.9977 at the lowest). Silence, a 0.5 s file, a missing model file and the microphone with and without its permission each ended with their sentence on the screen. The phone was warm from earlier tests (thermal status 1 to 2), the screen on. A trained reader with a larger word table will take more time than the 1,000-word placeholder.

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
