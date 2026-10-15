# Eg2 Transcribe for Android (LiteRT)

Record with the microphone or open an audio file, and the app writes down the English words with their times, on the phone. The audio encoder of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) runs on LiteRT (CPU, GPU or the Qualcomm NPU) and turns every 40 ms of audio into a vector; a small trained reader (reader v2, `reader/RELEASE.md`) maps each vector into the model's text space and the best of 200,000 words, or a blank, is read out. Without the reader's files the app runs a stub reader whose words are placeholders.

The app has no INTERNET permission; audio, model and transcript stay on the phone.

## Build and install

```sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Android 8.0 (API 26) or newer, arm64. LiteRT 2.2.0 (`com.google.ai.edge.litert:litert`, set in `gradle.properties`). `./gradlew -Plitert.version=2.3.0 :app:assembleDebug` builds against 2.3.0: from 2.3.0 the GPU accelerator is the separate `litert-gpu` artifact, which `app/build.gradle.kts` then adds, and the 2.3.0 classes carry Kotlin 2.4 metadata, which this build's Kotlin 2.2.21 reads with `-Xskip-metadata-version-check` (added for 2.3.0 only). The 2.3.0 build has not been run on a phone, and its NPU libraries come from another source (NPU below).

## The model files

The app needs five files, which are not in this repository. Download them from this project's Hugging Face repository, [mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT):

| file | bytes | SHA-256 | what |
|---|---:|---|---|
| `eg2_audio_f16safe_fp16w_w30.tflite` | 589,273,696 | `f0f6c37ee62b51372ab28013617aad53793a84aaddd96e1e844d0cea0eb2f11d` | the encoder for 30 s windows (signature `audio_30s`) |
| `eg2_audio_f16safe_fp16w_w10.tflite` | 589,015,376 | `1405986b353c3152e4a677d9446a5694a0f8fd0b13b794d26c892ef85295188c` | the encoder for 10 s windows (`audio_10s`; the NPU runs this one) |
| `reader_w30.tflite` | 386,726,384 | `129d6aa2d829dc4418675d01659ccc6fc42e33b84c2486f36faa9407d45c8716` | reader v2 with the word table, 30 s windows (`reader_30s`; The reader, below) |
| `reader_w10.tflite` | 386,470,384 | `128119442ca776deb5499f20189eac86c05060bd774e14fcb93d8bfd9e8c97fb` | the same for 10 s windows (`reader_10s`) |
| `vocab.txt` | 1,737,588 | `bb3b3c70379082333e1bbd90d07ca69fe5c0096bd1c3b2b3f83edd65f1062a68` | the reader's 200,000 words, one per line (`reader/vocab.txt` of this repository) |

The encoder files are the audio tower and `embed_audio` of `google/embeddinggemma-2` exported with litert-torch, one file per window length, with every RMSNorm rescaled so that its sums stay inside the float16 range (in float32 the rescaled graph gives the same numbers bit for bit; the GPU at float16 and the NPU need the rescaling) and the weights of the fully connected and convolution layers stored as float16. Inputs `mel` [1, F, 128] and `mask` [1, F] (F = 3000 or 1000 log-mel frames), outputs `soft` [1, F / 4, 512] and `mask` [1, F / 4]; masks are float32, 1 for a real frame or token. On the CPU and the GPU a recording of up to 9.96 s runs on the 10 s file and a longer one on the 30 s file; the NPU runs the 10 s file (Windows, below).

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

**Auto**, the default, picks the window length per recording. A recording of up to 9.96 s (249 tokens: one 10 s window with the contract's token at the right to spare) runs on the 10 s files, the encoder's `_w10` and `reader_w10.tflite`: one call of each on 250 tokens instead of 750. A longer recording runs in 30 s windows as below (one window up to 30 s, then steps of 653 tokens); it never gets 10 s windows, which would add window edges. Without both 10 s files a short recording runs in a 30 s window too. The app loads the 10 s files when it starts (a recording from the microphone and the bundled example are short) and compiles the other pair when a recording needs it; a folder of files in the device check runs the files that fit the loaded pair before the others, so it compiles each pair once. The detail line under the progress line says which window ran and why, for the example "1 window of 10 s (auto: 203 tokens fit one 10 s window (at most 249))", and the run report has it under `window.pick`. **Windows → Change** also offers fixed 30 s or 10 s windows.

The encoder is called on fixed windows of 30 s (or 10 s), and a call knows nothing about the audio before its window, so the tokens at its start come out wrong. The windows follow the reader's window contract (`reader/reader_config.json`, `window`: 30 s, 96 tokens of left overlap, 1 token at the right), the same cut as the iOS port's:

- Window k starts at token 653 k (750 − 96 − 1; frame 2,612 k) and keeps its tokens from 96 on (3.84 s of overlap); window 0 keeps them from 0.
- Every window but the last stops one token short of its end: the next window keeps that token.
- The window that reaches the end of the recording keeps through its last token: a recording of F mel frames gives ceil(F / 4) tokens, the encoder's count (the last one is computed with the frames past the end masked, as in the reader's training data).

The log-mel is computed on the whole recording and each window is a slice of those frames (token j reads frames 4j − 3 to 4j + 3, so a window cut out of exactly 30 s of audio would lose the last frame of its last token). The reader reads each window's 750 tokens as one block, both ways in time, and the kept ranges of all windows make the transcript. Measured on a Mac with the opening 10 minutes of the 54-minute fixture against the encoder run on the whole stream: 30 s windows with a 96-token overlap give a minimum token cosine of 0.99994 (0.99999994 on average); the encoder does 750 / 653 = 1.149× the work of one pass. **Windows → Change** offers the lengths the model has a file for and a few overlaps. With a file that holds both signatures, a last window with at most 10 s of audio runs on `audio_10s`.

The NPU runs 10 s windows. Its compiler builds the graph on the phone, and for the 30 s graph that step needed more memory than a Galaxy S26 (12 GB) had free (6.8 GB, the run was stopped; the 10 s graph compiles). With the same contract a 10 s window keeps 153 new tokens of its 250 (1.634× the encoder work), and the reader reads 10 s at a time.

## The reader

The reader is reader v2 of this repository (`reader/RELEASE.md`: a 6-layer transformer of 19.6 M parameters trained on LibriSpeech's three training sets, 960 h; greedy decoding, no language model). It reads a whole window of tokens at once, both ways in time, turns each 40 ms token into a direction in EmbeddingGemma 2's text space and scores it against the 200,000 words of `vocab.txt` (216.351 × the cosine to each word's text embedding) and against a blank: a token is its best word unless the blank scores higher, a run of the same word is one word, and blanks are dropped. Its known weaknesses (long words, homophones, read speech only) are listed in `reader/RELEASE.md`. Reader v1 (4 layers, train-clean-100) has the same inputs, outputs and file names, so the app runs either file pair; its files are revision `54d0e902` of the Hugging Face repository.

On the phone the reader, the word table and the choice of the best word are one LiteRT graph per window length, so the 200,000 scores of a token never leave the accelerator; the graph gives each token its best word, that word's score and the blank's score:

| file | bytes | signature | inputs | outputs |
|---|---|---|---|---|
| `reader_w30.tflite` | 386,726,384 | `reader_30s` | `soft` [1, 750, 512], `mask` [1, 750] | `ids` [1, 750] int32, `best` [1, 750], `blank` [1, 750] |
| `reader_w10.tflite` | 386,470,384 | `reader_10s` | `soft` [1, 250, 512], `mask` [1, 250] | `ids` [1, 250] int32, `best` [1, 250], `blank` [1, 250] |

The word table sits in the graph as float16, the release's own values (`word_table.npy`, 307 MB of each file), and is used in float32; the reader itself is float32. A window with fewer real tokens (the end of a recording) is padded and masked. Put a file with the release's `vocab.txt` (1,737,588 bytes) in the app's `reader` directory:

```sh
adb shell mkdir -p /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader
adb push reader_w30.tflite /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader/
adb push reader_w10.tflite /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader/
adb push vocab.txt /sdcard/Android/data/io.github.johnrocky.eg2transcribe/files/reader/
```

30 s windows need `reader_w30.tflite`; short recordings (auto) need `reader_w10.tflite`, and 10 s windows (the NPU) take it or the 30 s file (it pads). The reader runs on the accelerator the encoder runs on (on GPU fp32 when the encoder is on the NPU); when the GPU cannot compile it, it runs on the CPU and the screen says so. The Reader line on the screen names the accelerator and the file that runs, with its size and the leading 12 hex digits of its SHA-256: "Reader on GPU fp32 · reader_w30.tflite (387 MB, sha256 129d6aa2d829) · 200,000 words · compiled in 4.06 s". The Encoder line under the Windows row names the encoder file the same way: "Encoder on GPU fp32 · eg2_audio_f16safe_fp16w_w30.tflite (589 MB, sha256 f0f6c37ee62b) · signature audio_30s · compiled in 3.68 s". The digits are those of the table above; reader v1's files (revision `54d0e902`) show others. The app reads each file once for its SHA-256 when it loads it, after the compile: on a Pixel 8a that took 0.31–0.33 s for a reader file and 0.42–0.44 s for an encoder file (2026-10-15, right after `adb push`, in a run of the 20 fixtures that gave the same 20 transcripts and 5,036 tokens as below). On the NPU the encoder's hash is the one its cache keeps (NPU, below). The run report has the two lines without the compile time as `reader.note` and `model.note`, the whole SHA-256 as `reader.sha256` and `model.sha256`, and the time the hash took as `reader.load.sha256_ms` and `model.load.sha256_ms`; `reader_run` and `encoder_run` (in a folder run, each item's `reader` and `encoder`) name the files that transcribed that input, with their bytes and SHA-256.

The run report says how each window's calls were spent: `timing.calls` holds, per window, `encoder_write_ms` (the log-mel window and its mask into the input buffers), `encoder_run_ms` (the run call) and `encoder_read_ms` (the soft tokens and their mask back), and the same three for the reader, whose write includes padding the window and whose read is the word ids and the two scores; `timing.call_medians` has their medians, and the detail line under the progress line shows them as write, run, read. On the GPU the run call returns after a few milliseconds and the read waits for the GPU, so the GPU's time shows up in the read: for the example on a Pixel 8a with reader v2 (in the 20-fixture run below), encoder write 1.0, run 37.2, read 388.3 ms and reader write 0.9, run 9.1, read 208.8 ms. `encoder_ms` and `reader_ms` are the three added up. `timing` covers the transcription; loading is reported with each model: `backend.compile_ms` and `reader.compile_ms` for the compiles, `model.load.sha256_ms` and `reader.load.sha256_ms` for the SHA-256 reads.

Without these files the app runs a stub reader and says "Stub reader: the words are placeholders": identity (the 512-d soft token is its own vector), a table of 1,000 placeholder words with random rows and a random blank vector, searched on the CPU. Its words are the 1,000 most frequent words of `reader/fixtures/long-speckled-band/transcript.txt` (`tools/make_stub_vocab.py`), its random numbers a fixed SplitMix64 sequence (`tools/stub_table.py` prints the same table). It keeps the app usable for the encoder alone.

## NPU

Qualcomm's license lets its runtime libraries ship only inside an application, so they are not in this repository. Before building an APK that can use the NPU:

```sh
tools/fetch_npu_libs.sh v81      # SM8850 (Galaxy S26); v79 SM8750, v75 SM8650, v73 SM8550
```

It downloads LiteRT's v2.2.0 dispatch library and JIT compiler plugin and QAIRT 2.47.0.260601 (about 2.4 GB) and puts 10 files in `app/src/main/jniLibs/arm64-v8a/` (ignored by git). `tools/fetch_npu_libs.sh --from <dir> v81` copies them from a directory that already has them. The app extracts them at install (`useLegacyPackaging`) and keeps them byte for byte (`keepDebugSymbols`), and its manifest declares `libcdsprpc.so`, the library the NPU is reached through. With LiteRT 2.3.0 the matching files come from the Maven artifact `com.google.ai.edge.litert:litert-npu-runtime-qualcomm` and QAIRT 2.50.0.260828 instead; this project has not been run that way.

LiteRT 2.2.0 has an NPU path for the Qualcomm SM8550, SM8650, SM8750 and SM8850. When the NPU is picked for a model file that has not been compiled on this phone yet, LiteRT's Qualcomm compiler builds the graph on the phone (the screen says it compiles the encoder for the NPU, about 2–3 minutes, and counts the time), with the CPU accelerator for any operation the NPU would not take, and keeps the result in the app's cache: `cache/npu_jit/<file name>-<12 hex digits of the file's SHA-256>/`. Later starts read it from there. A file with other bytes under the same name gets a new entry, and the old one is removed; Android's "Clear cache" for the app removes them all. The app counts a run as NPU only when LiteRT's log says its dispatch delegate took the graph (a graph the compiler could not take would otherwise run on the CPU without an error). On a Galaxy S26 (SM8850) compiling `eg2_audio_f16safe_fp16w_w10.tflite` took 121.6 s and left a 620 MB entry; a later start read it in 0.6 s.

The LiteRT 2.2.0 AAR declares FOREGROUND_SERVICE, WAKE_LOCK, ACCESS_NETWORK_STATE and RECEIVE_BOOT_COMPLETED for its Play model delivery, which this app does not use; they are in the merged manifest.

## What it did with reader v2 (2026-10-15)

The device check with this version (debug build, LiteRT 2.2.0, windows on auto) on a Pixel 8a, the two encoder files and the reader v2 files, in two sessions (5.2 and 6.4 minutes). The word error rate (WER) is counted on the 54-minute LibriVox fixture (`reader/fixtures/long-speckled-band`, 9,889 reference words) with the normalization of `reader/RELEASE.md`; the same windows through reader v2 in PyTorch on a computer give 3.812 %.

| encoder and reader on | input | time | words |
|---|---|---|---|
| GPU fp32 | 54 min 17 s: auto picks 30 s windows (125) | 300.8 s (10.8× real time): encoder 1,510 ms, reader 989 ms per window | WER 3.812 % (377 errors): the PyTorch path's word or blank on all 81,436 tokens |
| GPU fp32 | the 20 test-clean fixtures: 11 on the 10 s files, then 9 on the 30 s files | 36.8 s for the 20 | 20 of 20 transcripts equal to `reader/RELEASE.md`, the PyTorch path's word or blank on all 5,036 tokens |
| CPU, 4 threads | the 20 fixtures | 158.0 s for the 20 | 20 of 20, all 5,036 tokens |

Per window on the GPU the reader took 218.3 ms on the 10 s files and 710.3 ms on the 30 s files (medians over the fixtures; reader v1 205.4 and 627.8 ms on 2026-10-14) and the encoder 419.9 and 1,122.6 ms; over the 54 minutes the reader took 989 ms per window against reader v1's 839 ms (2026-10-14, an earlier build; thermal status 2 at the end of both runs). On the CPU the encoder itself ran 1.5–1.7 times slower than on 2026-10-14 (2.30 s and 6.85 s per 10 s and 30 s window), and the reader took 1.37 s and 4.19 s. The 54-minute run reached thermal status 2 in its second minute; reading the FLAC took 5.4 s on top; the app peaked at 5,267,760 kB (VmHWM) and the phone's MemAvailable fell to 873,848 kB while the app loaded its models. In the earlier session the 54-minute run came right after the CPU run: MemAvailable fell under the device check's own line of 800,000 kB while the app loaded the 10 s files at its start (auto loads them before it reads the file), and the check stopped the app; the later session ran it alone, on a rested phone.

## What it did with auto windows, reader v1 (2026-10-14)

The device check with this version (debug build, LiteRT 2.2.0, windows on auto) on a Pixel 8a, all five model files, in one session of 18.6 minutes:

| encoder and reader on | input | time | words |
|---|---|---|---|
| GPU fp32 | the bundled example, 8.1 s: auto picks one 10 s window | 0.80 s (10.1× real time): encoder 471.5 ms, reader 205.8 ms | the reader's output listed in `reader/RELEASE.md` |
| GPU fp32 | the 20 test-clean fixtures: 11 of up to 9.96 s on the 10 s files, then 9 on the 30 s files | 43.2 s for the 20 | 20 of 20 transcripts equal to `reader/RELEASE.md` |
| CPU, 4 threads | the 20 fixtures | 115.2 s for the 20 | 20 of 20 |
| GPU, the reader at float16 with float32 accumulation (`--es reader_precision fp16acc32`) | the 20 fixtures | 38.9 s for the 20 | 19 of 20 |

Per window on the GPU, the encoder took 426–475 ms and the reader 202–206 ms on the 10 s files, and 1,077–1,109 ms and 624–635 ms on the 30 s files (the run before the 10 s files, with every file in a 30 s window, took 46.0 s for the 20 fixtures); on the CPU 1.25–1.54 s and 0.75–0.96 s, and 4.61–4.76 s and 2.77–2.82 s. Switching from the 10 s to the 30 s files compiled the 30 s encoder in 3.4 s and its reader in 3.8 s. The reader at float16 with float32 accumulation took 117–120 ms per 10 s window and 346–350 ms per 30 s window and gave the same word or blank as the CPU on 5,035 of the 5,036 tokens; the other one turned "tintoret" into "tiara" (1188-133604-0000, the one word whose two best scores are within 0.01 in PyTorch), so the reader stays at fp32. On GPU fp32 and on the CPU every token matched. The app peaked at 5,060,236 kB (VmHWM) over the 20 fixtures on the GPU.

## Audio files

A FLAC file is read by the app's own FLAC reader (`FlacDecoder.kt`), which checks every frame's CRCs. A FLAC file it cannot read (a frame whose CRC does not match, a file that ends early, a form it does not know) goes to Android's MediaExtractor and MediaCodec instead, and the detail line and the run report (`input.decode.path`, `input.decode.fallback_reason`) say so. WAV is read directly; MP3, M4A, Ogg and other formats go through MediaCodec.

On a Pixel 8a the FLAC reader reads the 54-minute LibriVox FLAC in 5.15 s (633× real time: 2.6 s decoding, 2.4 s writing the samples out). Through MediaCodec the same file took 409.4 s, because that path costs about 3 ms of the app's own MediaCodec calls for every compressed frame (129,651 frames of 402 samples: queueInputBuffer 146.4 s, dequeueInputBuffer 59.5 s, releaseOutputBuffer 55.9 s and dequeueOutputBuffer 46.8 s, the extractor 29.4 s, copying the samples out 55.4 s, waiting for the codec 0.05 s; the Pixel's extractor decodes the FLAC itself and hands raw PCM to a pass-through codec). Driving MediaCodec with callbacks halved that, to 203.9 s. Every way gave the same 52,119,092 samples, bit for bit those of ffmpeg (SHA-256 of the 16-bit samples). The bundled example cut after 60 % of its bytes stops the FLAC reader in the middle of a frame; MediaCodec then read 73,728 of its 129,920 declared samples, and the app transcribed those. On a Galaxy S26 the MediaCodec path read the 54-minute file in 44 s (an earlier build). `--es decode_mode sync|async|legacy` sends FLAC through MediaCodec too, for measurements, and `--ez decode_only true` measures a file's read without the model.

## What it did with reader v1 (2026-10-14)

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

A start with extras runs the same code as the buttons and writes a JSON report: `adb shell am start -n io.github.johnrocky.eg2transcribe/.MainActivity --es file audio/f30.wav --es backend gpu_fp32 --es report r_gpu.json` (`--es dir audio/fixtures20` transcribes every audio file of a directory into one report with an item per file, by name; `--ei mic_ms 3000` records from the microphone instead; `--ez dump true` also writes the soft tokens and, per token, the label, the best word and the two scores; `--ei window_s 0|30|10` sets the windows, 0 = auto). Three extras exist for measurements and are not on the screen: `--es reader_precision fp16acc32` runs the reader on the GPU at float16 with float32 accumulation (default fp32); `--ez decode_only true` with `--es file` reads the file, writes where the time went (`input.decode`: the extractor, the codec's input and output calls, the waits, the samples out, and a time line every 2 s) and stops without a model (`--ei decode_cap_s <s>` stops the read after that long); `--es decode_mode flac|sync|async|legacy` picks how a compressed file is read: flac, the default, reads FLAC with the app's own reader and the rest with MediaCodec; the other three send every compressed file through MediaCodec (Audio files, below). The report (`files/<report>`) holds the backend asked for and the one used for the encoder and for the reader, the files that ran (name, bytes, SHA-256), the compile times, LiteRT's own log lines about the delegate, the window count, the times per window, ×RT, the transcript, or the error. On the computer, the transcripts of the 20 test-clean fixtures are compared with the reader's outputs listed in `reader/RELEASE.md`, and the 54-minute transcript's word error rate with that of the same pipeline in PyTorch. Such a start shows over the lock screen and keeps the display on.

## Code

`MelFrontEnd.kt` (log-mel, written from transformers' `Gemma4AudioFeatureExtractor`), `Windowing.kt` (the window plan), `Encoder.kt` (LiteRT `CompiledModel`), `Pipeline.kt` (the window loop: log-mel of the next window while the encoder and the reader's graph run, the labels and words on another thread), `TfliteReader.kt` (the reader graph), `Reader.kt` / `Ctc.kt` (scores to labels to words; the stub reader), `WordTable.kt` (the stub's word search), `AudioDecoder.kt` / `FlacDecoder.kt` / `Resampler.kt` / `Mic.kt` (input), `Jobs.kt` / `Engine.kt` / `Session.kt` / `MainActivity.kt` (the app). `./gradlew :app:testDebugUnitTest` runs the JVM tests: the log-mel against the extractor's output for two LibriSpeech utterances (`app/src/test/resources/mel`), the window plan and the auto rule, the reader graph's outputs for three test-clean utterances (computed on a computer, `app/src/test/resources/reader`) read back as the `reader/RELEASE.md` transcripts, the word table, the stub's numbers, the collapse, the window loop with a fake encoder and a fake reader (and its per-call times), and the FLAC reader against ffmpeg's samples of the bundled example. With `-Peg2.flac=<list>` the FLAC check runs on a list of files ("path samples sha256" lines), with `-Peg2.replay=<dir>` the app's pipeline runs on encoder and reader outputs computed on a computer (`ReplayTest.kt`).

## Credits and licenses

- Code: Apache-2.0.
- EmbeddingGemma 2 (Google): Apache-2.0; use must follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- `app/src/test/resources/mel`: two LibriSpeech test-clean utterances (2300-131720-0000, 121-121726-0001) as 16-bit samples, and their log-mel rows. LibriSpeech ASR corpus, V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books", ICASSP 2015, https://www.openslr.org/12, CC BY 4.0.
- `app/src/main/assets/examples/1284-1180-0000.flac`: LibriSpeech test-clean utterance 1284-1180-0000, the original 16 kHz FLAC (CC BY 4.0 as above).
- `app/src/test/resources/reader`: the reader graph's outputs for three LibriSpeech test-clean utterances (1089-134686-0000, 1284-1180-0000, 2300-131720-0000), the words of `reader/vocab.txt` they name (the LibriSpeech lexicon, https://www.openslr.org/11, public domain) and the reader's transcripts from `reader/RELEASE.md`; LibriSpeech CC BY 4.0 as above.
- `app/src/main/assets/stub_vocab.txt`: words from Project Gutenberg eBook #1661, *The Adventures of Sherlock Holmes* (public domain in the USA).
- The idea comes from Shahir's post (@TslShahir, 2026-10-13) and the *Embedding Gemma Transcribe* Mac app. This is an independent project, not affiliated with Google.
