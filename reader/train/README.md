# Reader training

The reader turns the audio soft tokens of EmbeddingGemma 2 (`embed_audio` output, float32 `[T, 512]`, 25 per
second) into one 768-d unit vector per 40 ms token plus a blank logit. Each token reads out the closest of the
200,000 words in `../vocab.txt` (cosine to `../word_table.npy`, times a learned scale) or blank; greedy CTC
decoding (merge repeats, drop blanks) gives the transcript. No language model. The encoder stays frozen; only the
reader is trained, with CTC, on the cached soft tokens of LibriSpeech train-clean-100.

| File | What it does |
|---|---|
| `cache_soft_tokens.py` | LibriSpeech parquet -> soft tokens per utterance, float16 `.npy` + `index.tsv` |
| `make_table.py` | lexicon -> `vocab.txt` + `word_table.npy` (EmbeddingGemma 2 text embeddings), with checks |
| `data.py` | cache index, word ids, length-bucketed batches |
| `model.py` | reader bodies (a / b / c), CTC over all N + 1 classes, greedy decoding; `--self-test`, `--check-eg2` |
| `train.py` | training loop, dev WER per epoch, early stopping, checkpoints |
| `decode.py` | transcribe soft token files with a released reader |
| `eval_wer.py` | export the best checkpoint as `reader.pt` + `reader_config.json`, WER on the LibriSpeech splits and the fixtures |

Every script runs with `python -I` and has `--help`.

## Objective

For token t with unit vector h_t and blank logit b_t, the logits over the N + 1 classes are
`scale * cos(h_t, table[w])` for the N words and `b_t` for blank (index N). `torch.nn.functional.ctc_loss` (float32,
`zero_infinity=True`, mean over the batch of loss / target length) is applied to the log-softmax over all N + 1
classes. The N-way normalizer is computed in chunks with its own backward; the CTC recursion only reads the blank
and the words of the utterance, so it runs on those gathered columns. `python -I model.py --self-test` checks that
this equals `F.ctc_loss` on the full log-softmax in value and gradient (float64, max |difference| below 1e-9), and
that the check fails without the gradient correction described in `model.py`.

## Reproduce

Environment (Python 3.12):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "transformers==5.19.0" "sentence-transformers==6.1.0" \
  "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0" numpy soundfile pyarrow safetensors jiwer==4.0.0
```

Data (LibriSpeech is CC BY 4.0; the lexicon comes from openslr.org/resources/11):

```sh
HF_HUB_DISABLE_XET=1 .venv/bin/hf download openslr/librispeech_asr --repo-type dataset \
  --revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1 --local-dir hf_librispeech \
  --include "clean/train.100/*" --include "clean/validation/*" --include "other/validation/*" \
  --include "clean/test/*" --include "other/test/*"
HF_HUB_DISABLE_XET=1 .venv/bin/hf download google/embeddinggemma-2 \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --local-dir embeddinggemma-2
curl -L -o librispeech-lexicon.txt https://www.openslr.org/resources/11/librispeech-lexicon.txt
```

Steps (paths relative to this directory; `cache/` and `runs/` can live anywhere):

```sh
# 1. soft tokens (float32 compute, float16 storage, default attention = sdpa)
.venv/bin/python -I cache_soft_tokens.py --data-root hf_librispeech --model-dir embeddinggemma-2 --out cache \
  --splits train-clean-100 dev-clean dev-other test-clean test-other
# 2. vocabulary and word table
.venv/bin/python -I make_table.py --lexicon librispeech-lexicon.txt --model-dir embeddinggemma-2 --out-dir .. \
  --index cache/test-clean/index.tsv cache/test-other/index.tsv --fixtures ../fixtures/test-clean-20 \
  --report make_table.json
# 3. one epoch per candidate (epoch 1 of a 10-epoch schedule), compare dev-clean WER
for a in a b c; do
  .venv/bin/python -I train.py --arch $a --cache cache --vocab ../vocab.txt --table ../word_table.npy \
    --model-dir embeddinggemma-2 --run-dir runs/cmp_$a --epochs 1 --schedule-epochs 10
done
# 4. continue the winner (v1: arch a) up to 10 epochs in total, early stopping on dev-clean (patience 2), at most
#    3 hours of training + dev evaluation counted from epoch 1; one process per epoch (see "Memory on Apple silicon")
mkdir -p runs/v1 && cp runs/cmp_a/ckpt_epoch1.pt runs/v1/
for e in $(seq 2 10); do
  .venv/bin/python -I train.py --arch a --cache cache --vocab ../vocab.txt --table ../word_table.npy \
    --run-dir runs/v1 --resume runs/v1/ckpt_epoch$((e - 1)).pt --epochs $e --schedule-epochs 10 \
    --max-hours 3 --patience 2 || break
  [ "$(jq -r .stop_reason runs/v1/metrics.json)" = "reached $e epochs" ] || break  # early stop or budget
done
# 5. export the best epoch and measure WER
.venv/bin/python -I eval_wer.py --from-run runs/v1 --reader-dir .. --cache cache --out-dir runs/v1 \
  --fixtures ../fixtures/test-clean-20
# transcribe soft tokens with the released reader
.venv/bin/python -I decode.py --reader-dir .. --fixtures ../fixtures/test-clean-20
```

Model selection and early stopping use dev-clean only; test-clean and test-other are decoded once, at the end.

## Compute used for v1

One Mac: Apple M4 Max (128 GB), macOS 27.0, torch 2.11.0 on MPS, MPS allocations capped at 40 GB (`--mps-max-gb`; the
epochs peaked at 34.7 to 38.6 GB). v1 trained 8 epochs (epoch 1 is the `cmp_a` candidate's) and stopped on the 3-hour
budget of step 4: 17.8 to 23.7 min of training per epoch, 157.6 min in total, plus 5.5 min of dev-clean decoding
(2.72 h). The released weights are those of epoch 8 (best dev-clean WER). Step 5 takes about 1.5 min (22 s per split).

## Memory on Apple silicon

The normalizer over 200,000 words works on `[512, 200000]` float32 blocks. Every block has the same size (the last
one is padded with zero rows): with blocks of varying size, the MPS caching allocator keeps a buffer of every size it
has seen, and one training process reached 99 GB of MPS allocations at step 200 and a 126 GB footprint before step
300, filling the swap. `train.py` also caps the process's MPS allocations (`--mps-max-gb`, default 40, through
`torch.mps.set_per_process_memory_fraction`; the torch default is 1.7 x the GPU's recommended working set) and
releases the allocator cache every 50 steps (`--empty-cache-every`). With these, one epoch on LibriSpeech batches of
16 utterances peaks at 29 to 39 GB of MPS allocations.

Each new input shape (batch size, length) also leaves Metal memory outside torch's allocator until the process ends;
`torch.mps.empty_cache()` does not release it. A dev-clean pass in batches of 32 had added 6.7 GiB after 80 of its 85 batches, and after one
training epoch plus a dev pass a process held 30 GiB of it, so the next training step hit the 40 GB cap. Hence one
process per epoch in step 4 (the checkpoint carries the model, optimizer, schedule, history and early-stopping state;
the shuffle of each epoch depends only on `--seed` and the epoch), and `eval_wer.py` pads the time axis of each batch
to a multiple of 64 tokens (`--pad-to`; padding is masked, and on dev-clean the transcripts are the same as without
it).
