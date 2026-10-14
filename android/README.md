# Eg2 Transcribe for Android (LiteRT)

Record with the microphone or open an audio file, and the app writes down the English words with their times, on the phone. The audio encoder of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) runs on LiteRT (CPU, GPU or the Qualcomm NPU) and turns every 40 ms of audio into a vector; a small trained reader (reader v1, `reader/RELEASE.md`) maps each vector into the model's text space and the best of 200,000 words, or a blank, is read out. Without the reader's files the app runs a stub reader whose words are placeholders.

The app has no INTERNET permission; audio, model and transcript stay on the phone.

## Build and install

```sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Android 8.0 (API 26) or newer, arm64. LiteRT 2.2.0 (`com.google.ai.edge.litert:litert`, set in `gradle.properties`). `./gradlew -Plitert.version=2.3.0 :app:assembleDebug` builds against 2.3.0: from 2.3.0 the GPU accelerator is the separate `litert-gpu` artifact, which `app/build.gradle.kts` then adds, and the 2.3.0 classes carry Kotlin 2.4 metadata, which this build's Kotlin 2.2.21 reads with `-Xskip-metadata-version-check` (added for 2.3.0 only). The 2.3.0 build has not been run on a phone, and its NPU libraries come from another source (NPU below).

## The model files

The app needs five files, which are not in this repository. Download them from this project's Hugging Face repository (its id is fixed at release):

| file | bytes | SHA-256 | what |
|---|---:|---|---|
| `eg2_audio_f16safe_fp16w_w30.tflite` | 589,273,696 | `f0f6c37ee62b51372ab28013617aad53793a84aaddd96e1e844d0cea0eb2f11d` | the encoder for 30 s windows (signature `audio_30s`) |
| `eg2_audio_f16safe_fp16w_w10.tflite` | 589,015,376 | `1405986b353c3152e4a677d9446a5694a0f8fd0b13b794d26c892ef85295188c` | the encoder for 10 s windows (`audio_10s`; the NPU runs this one) |
| `reader_w30.tflite` | 361,494,720 | `76bdcce94e7311ee36529727e4bfd94675fe225c9d361c5628e98f5c432742e8` | reader v1 with the word table, 30 s windows (`reader_30s`; The reader, below) |
| `reader_w10.tflite` | 361,238,720 | `9419000ac89aa3f34478906dda81523f48bf0e8c802929df3345ba0850b72d3f` | the same for 10 s windows (`reader_10s`) |
| `vocab.txt` | 1,737,588 | `bb3b3c70379082333e1bbd90d07ca69fe5c0096bd1c3b2b3f83edd65f1062a68` | the reader's 200,000 words, one per line (`reader/vocab.txt` of this repository) |

The encoder files are the audio tower and `embed_audio` of `google/embeddinggemma-2` exported with litert-torch, one file per window length, with every RMSNorm rescaled so that its sums stay inside the float16 range (in float32 the rescaled graph gives the same numbers bit for bit; the GPU at float16 and the NPU need the rescaling) and the weights of the fully connected and convolution layers stored as float16. Inputs `mel` [1, F, 128] and `mask` [1, F] (F = 3000 or 1000 log-mel frames), outputs `soft` [1, F / 4, 512] and `mask` [1, F / 4]; masks are float32, 1 for a real frame or token. The CPU and the GPU run the 30 s file, the NPU the 10 s file (Windows, below).

The float16 weights halve the files; the graph turns them back into float32 before use, so the CPU and the GPU at fp32 precision compute in float32 with rounded weights. Measured against the float32 PyTorch encoder on test windows: on the GPU of a Pixel 8a at fp32 precision the 30 s file's tokens are within 5.1 × 10⁻⁴ (cosine 0.9999999955 at the lowest, 7 windows, 2,306 tokens); on the NPU of a Galaxy S26 the 10 s file's tokens reach a cosine of 0.9999936 on average and 0.9975 at the lowest (9 windows, 1,806 tokens). The float32-weight pair `eg2_audio_f16safe_w30.tflite` / `_w10.tflite` (1,175,047,892 / 1,174,789,572 bytes) is used the same way.

A file named `<name>_w30.tflite` and one named `<name>_w10.tflite` are one model: the app offers a window length when its file is there. Any other `.tflite` with one or both of the encoder signatures (such as an earlier export that holds both) is a model of its own; the NPU runs only a `_w10` file.

Put them on the phone in one of two ways:

- **adb**: start the app once (it creates its files directory and shows "MODEL MISSING"), then
  `adb push eg2_audio_f16safe_fp16w_w30.tflite /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/`
  `adb push eg2_audio_f16safe_fp16w_w10.tflite /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/`
  and the reader files into `files/reader/` (The reader, below), and come back to the app. A directory made by `adb shell` before the app has started once is not one the app can read.
- **On the phone**: tap **Change** next to Model, then **Choose a file on this phone…** and pick an encoder file; the app copies it into its files directory (589 MB each; copy both files of a pair). The reader files go into `files/reader/` with adb or a file manager.

Other `.tflite` files in the files directory show up under Model → Change.

## Using it

- **Record** starts the microphone (16 kHz mono); **Stop** ends the recording and the transcription starts. **Open audio file** takes WAV, FLAC, MP3, M4A/AAC, Ogg and whatever else the phone can decode; other sample rates are resampled to 16 kHz, stereo is mixed down. **Example → Try** transcribes the 8 s LibriSpeech utterance bundled in the app (test-clean 1284-1180-0000), to try the app without a file of your own. **Again** transcribes the last input again (for example on another backend), **Stop** ends a transcription after the current window.
- While it runs, the screen shows "Window k of N", the time since the start and the speed as a multiple of real time (audio seconds per second). The transcript is lower case without punctuation, a line with its time every 10 s of audio.
- Input shorter than 1 s is refused ("TOO SHORT"), and so is input quieter than RMS 0.005 (−46 dBFS, "NO SPEECH": the floor sits above two silent recordings made through a Galaxy S26 microphone). Without the microphone permission the screen says how to turn it on; a tap on that line opens the app's settings.

## Backends

| button | what runs the encoder |
|---|---|
| CPU | LiteRT's CPU accelerator (XNNPACK), 4 threads |
| GPU fp32 | LiteRT's GPU accelerator with fp32 precision |
| GPU default | LiteRT's GPU accelerator with its default precision (fp16 on most phones) |
| NPU | the Qualcomm Hexagon NPU with the 10 s file; the graph is compiled on the phone once per model file (needs the NPU libraries in the APK, below) |

GPU fp32 is the default. When the GPU cannot compile the graph, cannot run it, or gives non-finite values in a test run, the app runs the encoder on the CPU and says so in red under the backend buttons. When the NPU cannot be used (an unsupported chip, an APK without the libraries, no `_w10` file, a failed compile, or a graph LiteRT did not hand to the NPU), the app runs GPU fp32 instead and says why in the same place. The reader runs on the same accelerator as the encoder (GPU fp32 when the encoder is on the NPU), right after it on each window; the log-mel of the next window is computed on the CPU meanwhile.

## Windows

The encoder is called on fixed windows of 30 s (or 10 s), and a call knows nothing about the audio before its window, so the tokens at its start come out wrong. The windows follow the reader's window contract (`reader/reader_config.json`, `window`: 30 s, 96 tokens of left overlap, 1 token at the right), the same cut as the iOS port's:

- Window k starts at token 653 k (750 − 96 − 1; frame 2,612 k) and keeps its tokens from 96 on (3.84 s of overlap); window 0 keeps them from 0.
- Every window but the last stops one token short of its end: the next window keeps that token.
- The window that reaches the end of the recording keeps through its last token: a recording of F mel frames gives ceil(F / 4) tokens, the encoder's count (the last one is computed with the frames past the end masked, as in the reader's training data).

The log-mel is computed on the whole recording and each window is a slice of those frames (token j reads frames 4j − 3 to 4j + 3, so a window cut out of exactly 30 s of audio would lose the last frame of its last token). The reader reads each window's 750 tokens as one block, both ways in time, and the kept ranges of all windows make the transcript. Measured on a Mac with the opening 10 minutes of the 54-minute fixture against the encoder run on the whole stream: 30 s windows with a 96-token overlap give a minimum token cosine of 0.99994 (0.99999994 on average); the encoder does 750 / 653 = 1.149× the work of one pass. **Windows → Change** offers the lengths the model has a file for and a few overlaps. With a file that holds both signatures, a last window with at most 10 s of audio runs on `audio_10s`.

The NPU runs 10 s windows. Its compiler builds the graph on the phone, and for the 30 s graph that step needed more memory than a Galaxy S26 (12 GB) had free (6.8 GB, the run was stopped; the 10 s graph compiles). With the same contract a 10 s window keeps 153 new tokens of its 250 (1.634× the encoder work), and the reader reads 10 s at a time.

## The reader

The reader is reader v1 of this repository (`reader/RELEASE.md`: a 4-layer transformer of 13.3 M parameters trained on LibriSpeech train-clean-100; greedy decoding, no language model). It reads a whole window of tokens at once, both ways in time, turns each 40 ms token into a direction in EmbeddingGemma 2's text space and scores it against the 200,000 words of `vocab.txt` (91.218 × the cosine to each word's text embedding) and against a blank: a token is its best word unless the blank scores higher, a run of the same word is one word, and blanks are dropped. Its known weaknesses (short words, homophones, read speech only) are listed in `reader/RELEASE.md`.

On the phone the reader, the word table and the choice of the best word are one LiteRT graph per window length, so the 200,000 scores of a token never leave the accelerator; the graph gives each token its best word, that word's score and the blank's score:

| file | bytes | signature | inputs | outputs |
|---|---|---|---|---|
| `reader_w30.tflite` | 361,494,720 | `reader_30s` | `soft` [1, 750, 512], `mask` [1, 750] | `ids` [1, 750] int32, `best` [1, 750], `blank` [1, 750] |
| `reader_w10.tflite` | 361,238,720 | `reader_10s` | `soft` [1, 250, 512], `mask` [1, 250] | `ids` [1, 250] int32, `best` [1, 250], `blank` [1, 250] |

The word table sits in the graph as float16, the release's own values (`word_table.npy`, 307 MB of each file), and is used in float32; the reader itself is float32. A window with fewer real tokens (the end of a recording) is padded and masked. Put a file with the release's `vocab.txt` (1,737,588 bytes) in the app's `reader` directory:

```sh
adb shell mkdir -p /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader
adb push reader_w30.tflite /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader/
adb push vocab.txt /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader/
```

30 s windows need `reader_w30.tflite`, 10 s windows (the NPU) `reader_w10.tflite` or the 30 s file (it pads). The reader runs on the accelerator the encoder runs on (on GPU fp32 when the encoder is on the NPU); when the GPU cannot compile it, it runs on the CPU and the screen says so. The Reader line on the screen names the file and the accelerator.

Without these files the app runs a stub reader and says "Stub reader: the words are placeholders": identity (the 512-d soft token is its own vector), a table of 1,000 placeholder words with random rows and a random blank vector, searched on the CPU. Its words are the 1,000 most frequent words of `reader/fixtures/long-speckled-band/transcript.txt` (`tools/make_stub_vocab.py`), its random numbers a fixed SplitMix64 sequence (`tools/stub_table.py` prints the same table). It keeps the app usable for the encoder alone.

## NPU

Qualcomm's license lets its runtime libraries ship only inside an application, so they are not in this repository. Before building an APK that can use the NPU:

```sh
tools/fetch_npu_libs.sh v81      # SM8850 (Galaxy S26); v79 SM8750, v75 SM8650, v73 SM8550
```

It downloads LiteRT's v2.2.0 dispatch library and JIT compiler plugin and QAIRT 2.47.0.260601 (about 2.4 GB) and puts 10 files in `app/src/main/jniLibs/arm64-v8a/` (ignored by git). `tools/fetch_npu_libs.sh --from <dir> v81` copies them from a directory that already has them. The app extracts them at install (`useLegacyPackaging`) and keeps them byte for byte (`keepDebugSymbols`), and its manifest declares `libcdsprpc.so`, the library the NPU is reached through. With LiteRT 2.3.0 the matching files come from the Maven artifact `com.google.ai.edge.litert:litert-npu-runtime-qualcomm` and QAIRT 2.50.0.260828 instead; this project has not been run that way.

LiteRT 2.2.0 has an NPU path for the Qualcomm SM8550, SM8650, SM8750 and SM8850. When the NPU is picked for a model file that has not been compiled on this phone yet, LiteRT's Qualcomm compiler builds the graph on the phone (the screen says it compiles the encoder for the NPU, about 2–3 minutes, and counts the time), with the CPU accelerator for any operation the NPU would not take, and keeps the result in the app's cache: `cache/npu_jit/<file name>-<12 hex digits of the file's SHA-256>/`. Later starts read it from there. A file with other bytes under the same name gets a new entry, and the old one is removed; Android's "Clear cache" for the app removes them all. The app counts a run as NPU only when LiteRT's log says its dispatch delegate took the graph (a graph the compiler could not take would otherwise run on the CPU without an error). On a Galaxy S26 (SM8850) compiling `eg2_audio_f16safe_fp16w_w10.tflite` took 121.6 s and left a 620 MB entry; a later start read it in 0.6 s.

The LiteRT 2.2.0 AAR declares FOREGROUND_SERVICE, WAKE_LOCK, ACCESS_NETWORK_STATE and RECEIVE_BOOT_COMPLETED for its Play model delivery, which this app does not use; they are in the merged manifest.

## What it did with the reader (2026-10-14)

The device check below with this version (debug build, LiteRT 2.2.0), the fp16-weight 30 s encoder file and `reader_w30.tflite`. The word error rate (WER) is counted on the 54-minute LibriVox fixture (`reader/fixtures/long-speckled-band`, 9,889 reference words) with the normalization of `reader/RELEASE.md` (lower case, runs of spaces collapsed); the same windows through the reader in PyTorch on a computer give 5.784 %.

| phone | encoder and reader on | input | time | speed | words |
|---|---|---|---|---|---|
| Pixel 8a | GPU fp32 | 54 min 17 s | 276.8 s | 11.8× real time | WER 5.784 %: the PyTorch path's word or blank on all 81,436 tokens |
| Pixel 8a | GPU fp32 | the 20 test-clean fixtures | 46.0 s for the 20 | | 20 of 20 transcripts equal to the reader's outputs listed in `reader/RELEASE.md` |
| Pixel 8a | CPU, 4 threads | the 20 fixtures | 130.2 s for the 20 | | 20 of 20 |

Per 30 s window on the Pixel 8a, the encoder took 1.40 s and the reader 0.84 s on the GPU (median over the 54 minutes, thermal status 2), and 3.4–4.6 s and 2.1–2.8 s on the CPU. The times cover log-mel, encoder, reader and words; reading the 54-minute FLAC came on top (417.8 s on the Pixel 8a). With both models on the GPU the app peaked at 4,478,456 kB (VmHWM, 4.3 GiB) over the 54 minutes and at 5,000,744 kB (4.8 GiB) over the 20 fixtures. When the check ran the 20 fixtures on the GPU right after the CPU run, MemAvailable fell to 496,116 kB while the reader compiled and the check stopped the app at its own line of 800,000 kB; a fresh start passed with MemAvailable at 1,683,528 kB or more.

A missing encoder file and 3 s of near silence ended with their sentences on the screen ("MODEL MISSING", "NO SPEECH").

## Before the reader: a Galaxy S26 with the stub reader (2026-10-13)

One run of the device check with the build before the reader (its windows stepped 654 tokens): the fp16-weight pair above, the stub reader, LiteRT 2.2.0, a debug build:

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

A start with extras runs the same code as the buttons and writes a JSON report: `adb shell am start -n io.github.johnrocky.eg2transcribe/.MainActivity --es file audio/f30.wav --es backend gpu_fp32 --es report r_gpu.json` (`--es dir audio/fixtures20` transcribes every audio file of a directory in turn, by name, into one report with an item per file; `--ei mic_ms 3000` records from the microphone instead; `--ez dump true` also writes the soft tokens and, per token, the label, the best word and the two scores). The report (`files/<report>`) holds the backend asked for and the one used for the encoder and for the reader, the compile times, LiteRT's own log lines about the delegate, the window count, the times per window, ×RT, the transcript, or the error. On the computer, the transcripts of the 20 test-clean fixtures are compared with the reader's outputs listed in `reader/RELEASE.md`, and the 54-minute transcript's word error rate with that of the same pipeline in PyTorch. Such a start shows over the lock screen and keeps the display on.

## Code

`MelFrontEnd.kt` (log-mel, written from transformers' `Gemma4AudioFeatureExtractor`), `Windowing.kt` (the window plan), `Encoder.kt` (LiteRT `CompiledModel`), `Pipeline.kt` (the window loop: log-mel of the next window while the encoder and the reader's graph run, the labels and words on another thread), `TfliteReader.kt` (the reader graph), `Reader.kt` / `Ctc.kt` (scores to labels to words; the stub reader), `WordTable.kt` (the stub's word search), `AudioDecoder.kt` / `Resampler.kt` / `Mic.kt` (input), `Jobs.kt` / `Engine.kt` / `Session.kt` / `MainActivity.kt` (the app). `./gradlew :app:testDebugUnitTest` runs the JVM tests: the log-mel against the extractor's output for two LibriSpeech utterances (`app/src/test/resources/mel`), the window plan, the reader graph's outputs for three test-clean utterances (computed on a computer, `app/src/test/resources/reader`) read back as the `reader/RELEASE.md` transcripts, the word table, the stub's numbers, the collapse and the window loop with a fake encoder and a fake reader.

## Credits and licenses

- Code: Apache-2.0.
- EmbeddingGemma 2 (Google): Apache-2.0; use must follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- `app/src/test/resources/mel`: two LibriSpeech test-clean utterances (2300-131720-0000, 121-121726-0001) as 16-bit samples, and their log-mel rows. LibriSpeech ASR corpus, V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books", ICASSP 2015, https://www.openslr.org/12, CC BY 4.0.
- `app/src/main/assets/examples/1284-1180-0000.flac`: LibriSpeech test-clean utterance 1284-1180-0000, the original 16 kHz FLAC (CC BY 4.0 as above).
- `app/src/test/resources/reader`: the reader graph's outputs for three LibriSpeech test-clean utterances (1089-134686-0000, 1284-1180-0000, 2300-131720-0000), the words of `reader/vocab.txt` they name (the LibriSpeech lexicon, https://www.openslr.org/11, public domain) and the reader's transcripts from `reader/RELEASE.md`; LibriSpeech CC BY 4.0 as above.
- `app/src/main/assets/stub_vocab.txt`: words from Project Gutenberg eBook #1661, *The Adventures of Sherlock Holmes* (public domain in the USA).
- The idea comes from Shahir's post (@TslShahir, 2026-10-13) and the *Embedding Gemma Transcribe* Mac app. This is an independent project, not affiliated with Google.
