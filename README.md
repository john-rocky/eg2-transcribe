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

## Reader v2

The reader reads each 40 ms audio token as the closest of 200,000 English words (the words of the LibriSpeech
lexicon) or as blank. Reader v2 is a 6-layer transformer (19.6 M parameters) trained with CTC on the three
LibriSpeech training sets (960 h; 890 h after leaving out the utterances with a word outside the vocabulary) against
a fixed table of EmbeddingGemma 2 text embeddings of those words; only the reader is trained. Greedy decoding, no
language model:

| Set | v2 WER | v1 WER |
|---|---:|---:|
| test-clean | 4.21 % | 6.81 % |
| test-other | 10.42 % | 15.10 % |
| dev-clean (model selection) | 3.75 % | 6.57 % |
| dev-other | 10.26 % | 14.93 % |

Reader v1, a 4-layer transformer (13.3 M parameters) trained on train-clean-100 alone, is revision `1af9c5f5` of the
Hugging Face repository below and this repository's history.

`reader/RELEASE.md` lists the files with their SHA-256, the training recipe, the reader's output on the 20 fixture
utterances and the known weaknesses; `reader/reader_config.json` is the contract for ports. The weights (`reader.pt`,
`word_table.npy`), `vocab.txt` and `reader_config.json` are on Hugging Face:
[mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader).
The Core AI build
([mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI),
iOS / macOS) uses reader v2; the Android build
([mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT))
ships its own copy of the reader, and that repository's card says which version it carries.

License: Apache-2.0. The reader is trained on LibriSpeech (CC BY 4.0); the word table holds EmbeddingGemma 2 outputs,
so deployments follow the Gemma Prohibited Use Policy.

## iOS / macOS (Core AI)

`ios/` is a SwiftUI app for iPhone (iOS 27) and Mac (macOS 27): choose an audio file or record with the microphone,
and the transcript appears, computed on the device. The app computes the log-mel in Swift (with a float64 FFT, within
4.77e-7 of the transformers extractor) and runs 30 s windows through two Core AI bundles: the EmbeddingGemma 2 audio
encoder (fp16) and the reader with its word table (fp16). `ios/README.md` says how to build it and how the model
folder is laid out.

The model folder is
[mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI):
`pipeline.json`, `reader_config.json`, `vocab.txt`, `mel_filters.bin`, and the bundles in `macos/` (the Mac), `ios/`
(any iPhone; the phone specializes them when it loads them) and `ios-h19p/` (compiled ahead of time for the iPhone 18
Pro). The conversion, its checks and every number below are on the
[zoo card](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/embeddinggemma-2-transcribe).

For running the encoder alone on long audio on a Mac, the same repo also holds a second form of the encoder in
`macos-encoder-split7-30s/`: two bundles, a float32 front on the GPU and a float16 body on the Neural Engine, that run
a stream of 30 s windows at 30.96 ms per window on the M4 Max with two windows in flight. The app does not download it.

| Device | What | Time |
|---|---|---|
| Mac (M4 Max), macOS 27.0 | encoder, one 30 s window, GPU | 40.02 ms |
| Mac (M4 Max), macOS 27.0 | reader v2 + word table, one 30 s window, GPU | 23.85 ms |
| Mac (M4 Max), macOS 27.0 | a 54-minute recording (3,257.4 s): encoder on the GPU, reader v2 on the Neural Engine, two windows in flight | 6.33 s (515× real time), WER 3.80 % |
| iPhone 18 Pro, iOS 27.2 | encoder, one 30 s window, GPU (compiled ahead of time) | 59.60 ms |
| iPhone 18 Pro, iOS 27.2 | reader v2 + word table, one 30 s window, GPU (compiled ahead of time) | 34.76 ms |
| Mac (M4 Max), macOS 27.0 | the app on the 54-minute recording, its default: encoder on the GPU, reader v2 on the Neural Engine, two windows in flight | 5.95 s (548× real time), WER 3.82 % |
| iPhone 18 Pro, iOS 27.2 | the app on the 54-minute recording, its default: encoder and reader v2 on the GPU, one window at a time | 12.01 s (271× real time), WER 3.81 % |

The Mac app with reader v2 on an 8.6 s LibriVox clip, from the file to its 27 words in 105 ms:
[12-second screen recording](https://github.com/john-rocky/coreai-assets/blob/main/demos/eg2-transcribe-coreai-mac-20261015.mp4) (`demos/eg2-transcribe-coreai-mac-20261015.mp4` in coreai-assets; the recording with reader v1, 2026-10-14, is beside it).
The iPhone 18 Pro app with reader v2 on a 3.1 s sentence of the same chapter, from the file to its 10 words in 156 ms:
[12-second screen recording](https://github.com/john-rocky/coreai-assets/blob/main/demos/eg2-transcribe-coreai-iphone-20261016.mp4).

## Android

`android/` is an app for Android 8.0+ (arm64) on LiteRT: record with the microphone, open an audio file or try the bundled example, and the words appear with their times without leaving the phone; the encoder (30 s windows with a 3.84 s overlap) and reader v1 (one graph with its 200,000-word table) run on the GPU or the CPU, and the encoder also on a Qualcomm NPU.

On a Pixel 8a it transcribed the 54-minute fixture in 276.8 s with both on the GPU at fp32, with a word error rate of 5.784 %, the same word or blank as the PyTorch reader on every token, and the 20 test-clean fixtures as listed in `reader/RELEASE.md`; `android/README.md` has the model files, the table and the numbers.

Status: in progress (started 2026-10-13). Numbers appear here only after they are measured on a device.

License: Apache-2.0 for the code in this repository. EmbeddingGemma 2 weights are Apache-2.0 and
deployments must follow the Gemma Prohibited Use Policy.
