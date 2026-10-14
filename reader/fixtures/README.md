# Reader fixtures

Reference outputs of the EmbeddingGemma 2 audio encoder for testing the encoder ports (Core AI, LiteRT) and the
reader. `MANIFEST.json` lists every file with its sha256 and is written last: if it is there, the set is complete.
The `.npy` and `.flac` files are not in git (see `.gitignore`); `make_fixtures.py` rebuilds them.

| Path | Content |
|---|---|
| `test-clean-20/<id>/audio.flac` | LibriSpeech test-clean utterance, the original 16 kHz mono FLAC |
| `test-clean-20/<id>/features.npy` | float32 `[n_frames, 128]` log-mel from `Gemma4AudioFeatureExtractor` (valid rows) |
| `test-clean-20/<id>/soft_tokens.npy` | float32 `[n_tokens, 512]` audio soft tokens |
| `test-clean-20/<id>/embedding_768.npy` | float32 `[768]` sentence-transformers embedding of the audio (L2-normalized) |
| `test-clean-20/<id>/transcript.txt` | LibriSpeech transcript, lower case |
| `long-speckled-band/audio.flac` | 54.3 min LibriVox reading, 16 kHz mono FLAC (see `SOURCE.md`) |
| `long-speckled-band/features.npy` | float32 `[325744, 128]` log-mel of the whole clip |
| `long-speckled-band/soft_tokens_continuous.npy` | float32 `[81436, 512]`, the whole clip encoded as one stream |
| `long-speckled-band/soft_tokens_w30.npy` | float32 `[81436, 512]`, independent 30 s windows without overlap, concatenated |
| `long-speckled-band/transcript.txt` | Project Gutenberg text of the story, normalized |

## Token contract

- Reference: `transformers` 5.19.0 `EmbeddingGemma2Model`, `google/embeddinggemma-2` revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`, float32 on CPU, default attention implementation (sdpa).
  With transformers 5.19.0 do not load it with `attn_implementation="eager"`: the eager mask is a float mask,
  the Gemma 4 audio attention applies `masked_fill(mask.logical_not(), ...)`, and the attention then reads the
  positions it should ignore (768-d cosine to the default path about 0.75).
- Soft token = `model.get_audio_features(input_features, input_features_mask).pooler_output` at the valid rows,
  the output of `embed_audio`: 512 values per 40 ms, 25 tokens per second.
- `n_frames = floor((n_samples - 161) / 160) + 1` and `n_tokens = ceil(ceil(n_frames / 2) / 2)`
  (30 s = 480,000 samples = 2,999 frames = 750 tokens).
- The 768-d embedding is the text model run on `<bos> <|audio> n_tokens x <|audio|> <audio|> <eos>`
  (ids 2, 256000, 258881, 258883, 1) with the soft tokens in the `<|audio|>` slots, mean-pooled over all
  `n_tokens + 4` positions and L2-normalized; this equals `SentenceTransformer.encode` of the audio.
- The encoder is causal: each layer attends to the token itself and the 11 before it, then a causal depthwise
  convolution of width 5; over 12 layers a token sees at most 180 tokens (7.2 s) back. Ahead, token j reads mel
  frames up to 4j + 3, which need audio up to sample 640 (j + 1): one sample past its own 40 ms slot. A 30 s
  window of 480,000 samples therefore yields 2,999 frames, and its last token is computed without its last frame.

## Licenses

- LibriSpeech (test-clean utterances): V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR
  corpus based on public domain audio books", ICASSP 2015, https://www.openslr.org/12, CC BY 4.0.
- LibriVox recording (long clip): public domain, read by Kirsten Ferreri,
  https://archive.org/details/adventures_holmes.
- Project Gutenberg eBook #1661 (long clip transcript): public domain in the USA; only the normalized chapter
  text is included, without the Project Gutenberg header or license.

## Rebuild

```sh
curl -L -o test-clean.parquet https://huggingface.co/datasets/openslr/librispeech_asr/resolve/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/clean/test/0000.parquet
curl -L -o adventureholmes_08_doyle.mp3 https://archive.org/download/adventures_holmes/adventureholmes_08_doyle.mp3
curl -L -o pg1661.txt https://www.gutenberg.org/cache/epub/1661/pg1661.txt
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "transformers==5.19.0" "sentence-transformers==6.1.0" \
  "torch==2.11.0" "torchvision==0.26.0" numpy soundfile pyarrow pillow
.venv/bin/python make_fixtures.py --test-clean-parquet test-clean.parquet \
  --long-mp3 adventureholmes_08_doyle.mp3 --gutenberg pg1661.txt
```

`ffmpeg` must be on `PATH`. The model comes from the Hugging Face Hub at the pinned revision (`--model` also
takes a local snapshot directory). The 20 utterances are the first 20 of test-clean in id order with a length
of 4 to 20 s and at most two per speaker. The whole run takes a few minutes on a laptop CPU.
