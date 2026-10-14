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

`android/` is an app for Android 8.0+ (arm64) on LiteRT: record with the microphone, open an audio file or try the bundled example, and the words appear with their times without leaving the phone; the encoder (30 s windows with a 3.84 s overlap) and reader v1 (one graph with its 200,000-word table) run on the GPU or the CPU, and the encoder also on a Qualcomm NPU.

On a Pixel 8a it transcribed the 54-minute fixture in 276.8 s with both on the GPU at fp32, with a word error rate of 5.784 %, the same word or blank as the PyTorch reader on every token, and the 20 test-clean fixtures as listed in `reader/RELEASE.md`; `android/README.md` has the model files, the table and the numbers.

Status: in progress (started 2026-10-13). Numbers appear here only after they are measured on a device.

License: Apache-2.0 for the code in this repository. EmbeddingGemma 2 weights are Apache-2.0 and
deployments must follow the Gemma Prohibited Use Policy.
