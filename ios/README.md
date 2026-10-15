# EG2 Transcribe for iPhone and Mac (Core AI)

An app that turns speech into text on the device: choose an audio file or record with the microphone, and the
transcript appears, computed by the EmbeddingGemma 2 audio encoder and the trained reader on Core AI. One SwiftUI code
base builds the iPhone app (iOS 27) and the Mac app (macOS 27).

## Run it

Needs Xcode 27 and [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```sh
cd ios
xcodegen generate
open EG2Transcribe.xcodeproj
```

Pick the `EG2Transcribe` scheme for an iPhone (set your team under Signing) or `EG2TranscribeMac` for the Mac, and
run. The app needs the model folder (next section) before it can transcribe.

## The model folder

The bundles are not in this repository. The app reads a folder laid out like the Hugging Face repo will be:

```
pipeline.json        the forms the app can run (which bundles, which compute unit), the default form, overlap
reader_config.json   the reader's window contract and vocabulary file (written by reader/train/eval_wer.py)
vocab.txt            one word per line; the line number is the reader's word index
mel_filters.bin      the log-mel filter bank, 257 × 128 float64 (tools/make_mel_filters.py)
SHA256SUMS           the SHA-256 of every file (optional; part of the folder's fingerprint)
macos/               eg2_audio_fp16_T3000.aimodel, eg2_reader_v2_fp16_T750.aimodel (specialized when the app loads them);
                     the split-encoder form also needs eg2_front_layer7_fp32_T3000.aimodel and
                     eg2_body_layer7_fp16_T3000.aimodel here (not in the published folder: Settings lists it only then)
ios/                 the same .aimodel bundles, for any iPhone
ios-<arch>/          the same bundles compiled ahead of time for one chip (.aimodelc), e.g. ios-h19p for iPhone 18 Pro
```

On the iPhone the app looks in `ios-<arch>/` for its own chip and falls back to `ios/`; on the Mac it uses
`macos/`. Where the folder can be:

- **iPhone**: `Documents/models/eg2-transcribe` of the app (copy it with the Files app, or
  `xcrun devicectl device copy to … --domain-type appDataContainer --domain-identifier com.coreai.eg2transcribe`).
- **Mac**: `~/Documents/models/eg2-transcribe`, or any folder picked with *Choose model folder…*.
- **Download**: `ModelFiles.repo` in `Sources/Models/ModelFiles.swift` names the Hugging Face repo
  (`mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI`); the app offers *Download* (the shared files and this device's
  bundle folder).

### When the model folder changes

Core AI keeps what it specializes from a bundle in the app's cache, keyed by the bundle's `main.hash`. For a bundle
compiled ahead of time that hash covers the program but not the weights, so a bundle replaced by one with the same
program and new weights would get the old weights' specialization back, and the old transcripts with it. The app
therefore keeps a fingerprint of the folder it loaded (`ModelCache.swift`: the SHA-256 of `SHA256SUMS` and of every
bundle file's path, size, modification time and file number) and clears its Core AI cache
(`AIModelCache.default.deleteAll()`) before it loads a folder with another fingerprint. A load right after a change
is therefore a cold one. On the Mac the cache is `~/Library/Caches/coreai-cache/<OS build>/com.coreai.eg2transcribe.mac/`,
on the iPhone `Library/Caches/coreai-cache/<OS build>/com.coreai.eg2transcribe/` in the app's container.

`mel_filters.bin` is the matrix `transformers.audio_utils.mel_filter_bank(257, 128, 0, 8000, 16000, norm=None,
mel_scale="htk")`, the one the feature extractor builds; to make it again:

```sh
python3 tools/make_mel_filters.py <model folder>/mel_filters.bin
```

### pipeline.json

Each form lists its stages in order. Values pass between stages by tensor name: the host provides `features`,
`half_mask` and `token_mask`; the encoder returns `soft_tokens`; the reader returns `top1_index`, `top1_logit` and
`blank_logit`. A form whose encoder is split in two bundles passes `hidden` from the front to the body (a float32
output becomes float16 when the next stage takes float16).

```json
{
  "schema": 1,
  "default_form": {"macos": "gpu-ane", "ios": "gpu"},
  "overlap": {"macos": true, "ios": false},
  "forms": [
    {"id": "gpu", "title": "Encoder GPU · reader GPU",
     "stages": [{"name": "encoder", "bundle": "eg2_audio_fp16_T3000", "compute": "gpu"},
                {"name": "reader", "bundle": "eg2_reader_v2_fp16_T750", "compute": "gpu"}]},
    {"id": "gpu-ane", "title": "Encoder GPU · reader Neural Engine", "platforms": ["macos"],
     "stages": [{"name": "encoder", "bundle": "eg2_audio_fp16_T3000", "compute": "gpu"},
                {"name": "reader", "bundle": "eg2_reader_v2_fp16_T750", "compute": "ane"}]},
    {"id": "split7-gpu", "title": "Encoder in two bundles (GPU, Neural Engine) · reader GPU", "platforms": ["macos"],
     "stages": [{"name": "front", "bundle": "eg2_front_layer7_fp32_T3000", "compute": "gpu"},
                {"name": "body", "bundle": "eg2_body_layer7_fp16_T3000", "compute": "ane"},
                {"name": "reader", "bundle": "eg2_reader_v2_fp16_T750", "compute": "gpu"}]}
  ]
}
```

`compute` is `gpu`, `ane` (Neural Engine), `cpu` or `default`. `platforms` limits a form to `macos` or `ios`.
`default_form` and `overlap` take one value or one per platform. On the Mac the default is the reader on the Neural
Engine with overlap on (5.5 s for the 54-min recording, against 7.3 s with both on the GPU). On the iPhone both
bundles run on the GPU (the Neural Engine takes none of them at 30 s), and overlap does not help there because the two
calls share the one GPU. Settings shows the forms whose bundles are all in the folder, and the overlap switch.

## How it works

1. **Audio**: `AVAudioFile` reads the file (WAV, FLAC, M4A, MP3, CAF …); `AVAudioConverter` makes it 16 kHz mono
   Float32 when it is not already. A 16 kHz FLAC gives the same samples as Python's `soundfile`.
2. **Log-mel** (`LogMelExtractor.swift`): `Gemma4AudioFeatureExtractor` of transformers, step for step — 160 zeros in
   front, 20 ms frames every 10 ms, periodic Hann window, |FFT| of 512, 128 HTK mel bins, `log(mel + 0.001)`, only the
   frames whose window is all real audio. The FFT runs in float64 and |X| is rounded to float32, which keeps the
   features within 5e-7 of the extractor's.
3. **Windows** (`Windowing.swift`): 30 s windows (3000 frames, 750 tokens) that start 96 tokens before the end of the
   previous one. Each window keeps only its new tokens and drops its last token (the encoder is causal: the start of a
   window lacks its left context, its last token one input frame); the last window keeps through the end.
4. **Encoder, then reader** (`Stage.swift`, `Transcriber.swift`): the Core AI bundles of the form, loaded once and
   warmed up with one call. With overlap on, the encoder of the next window runs while the reader reads the current
   one; a function never gets two calls at once (that corrupts its outputs).
5. **Words** (`Greedy.swift`): per token, blank where `blank_logit > top1_logit`, else the word `top1_index`; repeated
   labels merged, blanks dropped.

## Self-test

The same code runs headless from launch arguments, for checks and measurements. Every run writes
`<out>/<tag>/log.txt` (last line `DONE …` or `ERROR: …`) and `<out>/<tag>/result.json`.

```sh
# Mac
EG2Transcribe.app/Contents/MacOS/EG2Transcribe --selftest long --models <model folder> --audio speech.flac --overlap both
# iPhone (paths are inside the app's Documents)
xcrun devicectl device process launch --device <id> --terminate-existing com.coreai.eg2transcribe -- \
  --selftest transcribe --audio speech.flac --tag launch
```

| Command | What it does |
|---|---|
| `probe` | device, thermal state, power |
| `cache` | the model folder's fingerprint check alone (with `--model-cache clear`: clear the Core AI cache) |
| `transcribe --audio F` | load, warm up, transcribe; seconds from the process start to the transcript |
| `gate --fixtures DIR [--expect TSV]` | the reader fixtures: log-mel vs `features.npy`, soft tokens vs `soft_tokens.npy`, transcripts |
| `long --audio F [--overlap both]` | a long recording with and without overlap; wall clock, ms per window and stage, host time |
| `decode --audio F --dump D` | the 16 kHz samples as float32, to compare with Python |
| `thermal [--cap S]` | the thermal state every 10 s until nominal |

Every command takes `--model-cache check` (the default: clear the Core AI cache when the folder's fingerprint changed),
`clear` (always clear before loading: a cold start) or `off` (neither check nor clear).

`-autoplay <file>` instead runs the app's own button path once (for screen recordings); `-textSize 44` sets the
transcript's text size for that launch (Settings → Display → Transcript text).

## Numbers

Release builds on the model folder. Mac (macOS 27.0): reader v2, 2026-10-15. iPhone 18 Pro (iOS 27.2, on USB power):
reader v2, 2026-10-15, and reader v1, 2026-10-14. Long recording: *The Adventure of the Speckled
Band* (LibriVox, 54 min 17 s, 125 windows), reference text from Project Gutenberg; the reference (torch) path's WER on
it is 3.81 % with reader v2 and 5.78 % with reader v1.

| Device | Reader | Form | Overlap | 54-min recording (windows) | Times real time | Decode + log-mel | WER |
|---|---|---|---|---|---|---|---|
| Mac (M4 Max) | v2 | encoder GPU · reader Neural Engine (default) | on | 5.95 s | 548× | 0.66 s + 0.05 s | 3.82 % |
| Mac (M4 Max) | v2 | encoder GPU · reader Neural Engine | off | 10.91 s | 299× | | 3.82 % |
| Mac (M4 Max) | v2 | encoder GPU · reader GPU | on | 7.40 s | 440× | | 3.83 % |
| Mac (M4 Max) | v2 | encoder GPU · reader GPU | off | 8.06 s | 404× | | 3.83 % |
| iPhone 18 Pro | v2 | encoder GPU · reader GPU (default) | off | 12.01 s | 271× | 0.51 s + 0.16 s | 3.81 % |
| iPhone 18 Pro | v1 | encoder GPU · reader GPU (default) | off | 11.63 s | 280× | 0.49 s + 0.16 s | 5.77 % |
| iPhone 18 Pro | v1 | encoder GPU · reader GPU | on | 11.68 s | 279× | | 5.77 % |

The iPhone's passes slow down as the GPU stays busy, even while the thermal state reads nominal: with reader v1, four
alternating passes, each started at nominal, took 11.63 (off), 11.68 (on), 12.37 (off) and 14.34 s (on). With reader
v2, two passes with overlap off took 12.04 s (started at thermal state fair) and 12.01 s (started at nominal).

From launch to the transcript of a 10 s clip: right after a model folder change (the app clears Core AI's cache,
then loads) 8.49 s on the Mac with reader v2, where the reader is specialized for the Neural Engine (5.59 s of it),
and 1.40 s on the iPhone with reader v2, whose bundles are compiled ahead of time (that launch began at thermal state
serious; reader v1: 1.28 s); later launches 0.57 s and 0.55 s (reader v1 on the iPhone: 0.51 s). A
launch right after installing the app on the iPhone took longer in an earlier measurement with an interim reader:
11.5 s, the opening calls of the two bundles taking 9.3 s and 1.0 s.

Checks: the app's log-mel is within 4.8e-7 of the extractor's on the 20 fixture utterances (bit-identical on the Mac
and the iPhone); through the float32 encoder its soft tokens match the reference to a per-token cosine of 0.9999999995.
Fed the extractor's own features, the app's windows give the same labels as the Python pipeline on all 81,436 tokens of
the long recording, with the reader on the GPU and on the Neural Engine alike. The fixture transcripts match the
reference (torch) path on 20 of 20 with both bundles on the GPU (Mac and iPhone) and on 19 of 20 with the reader on the
Neural Engine (one word chosen over another that scored within 0.0001 in cosine).

## License

The code in this folder is Apache-2.0, like the rest of the repository. The audio encoder is EmbeddingGemma 2
(Apache-2.0; deployments must follow the Gemma Prohibited Use Policy). The reader and the word table are trained in
this repository (`reader/`) on LibriSpeech (CC BY 4.0).
