# EmbeddingGemma 2 Transcribe — Android (LiteRT) and iOS (Core AI)

Speech-to-text built on the audio encoder of Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).
The encoder stays frozen. A small trained *reader* maps each 40 ms audio token into the model's
text space, and the nearest English word is read out as the transcript.

The idea comes from Shahir's post (@TslShahir, 2026-10-13) and the *Embedding Gemma Transcribe*
Mac app (eg2.haku.sh), whose code and weights are not public. This repository is an independent
reproduction for phones: Android on LiteRT and iOS on Core AI. It is not affiliated with Google.

| Directory | What it holds |
|---|---|
| `reader/` | Training recipe for the reader, the word table and the test fixtures |
| `android/` | Android app on LiteRT (encoder + reader on CPU / GPU / NPU) |
| `ios/` | iOS and macOS app on Core AI (encoder + reader on GPU / Neural Engine) |

## Android

`android/` is an app for Android 8.0+ (arm64) on LiteRT: record with the microphone or open an audio file, and the words appear with their times, without leaving the phone. The encoder runs on the CPU, the GPU or a Qualcomm NPU, on 30 s windows that overlap by 3.84 s so that each window starts with the context it needs. The encoder file (`eg2_audio_fp32.tflite`, 1.18 GB) is not in this repository; `android/README.md` says how to put it on the phone and how the device check works. Until the trained reader is out, the app reads with a stub whose words are placeholders. On a Galaxy S26 it went through a 54-minute recording in 42.1 s on the GPU (77× real time, not counting reading the file; with the placeholder reader), and its words matched the same pipeline run on a computer token for token (`android/README.md` has the table).

Status: in progress (started 2026-10-13). Numbers appear here only after they are measured on a device.

License: Apache-2.0 for the code in this repository. EmbeddingGemma 2 weights are Apache-2.0 and
deployments must follow the Gemma Prohibited Use Policy.
