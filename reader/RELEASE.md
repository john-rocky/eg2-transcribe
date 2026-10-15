# Reader v2

Released 2026-10-15. Reads EmbeddingGemma 2 audio soft tokens as words. Greedy decoding, no language model: **test-clean WER 4.21 %, test-other WER 10.42 %**. Trained on the three LibriSpeech training sets (960 h; 890 h after leaving out the utterances with a word outside the vocabulary); reader v1 was trained on train-clean-100 only.

Pre-registered v2 line: test-clean <= 5.5 % and test-other <= 12 % (greedy, no LM). test-clean meets it (4.21 %), test-other meets it (10.42 %).

## v2 and v1

Same encoder, vocabulary, word table, objective, optimizer settings and decoding; the training data and the machine differ (and the body depth, when the table says so).

| Set | v1 WER | v2 WER | Change (points) |
|---|---:|---:|---:|
| dev-clean | 6.57 % | 3.75 % | -2.81 |
| dev-other | 14.93 % | 10.26 % | -4.66 |
| test-clean | 6.81 % | 4.21 % | -2.60 |
| test-other | 15.10 % | 10.42 % | -4.68 |
| fixtures `test-clean-20` | 4.81 % | 3.70 % | -1.11 |

| | v1 | v2 |
|---|---|---|
| Training data | train-clean-100, 26,422 utterances, 92.8 h | 960 h set, 261,486 utterances, 890.2 h |
| Body | arch a, 4 layers, 13,268,994 parameters | arch a, 6 layers, 19,573,762 parameters |
| Epochs (released) | 8 (8) | 9 (7) |
| Steps per epoch | 1,652 | 16,343 |
| Machine | Apple M4 Max, torch 2.11.0 MPS | NVIDIA A100-SXM4-80GB, torch 2.11.0+cu130 CUDA 13.0 |
| Precision | float32 (MPS) | training matmuls in TF32 (A100), evaluation in fp32 |

## Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `vocab.txt` | 1,737,588 | `bb3b3c70379082333e1bbd90d07ca69fe5c0096bd1c3b2b3f83edd65f1062a68` |
| `word_table.npy` | 307,200,128 | `b2d0b13cc70c492f1444b8a42deab437d25254eac47d752d777d27b65377faa8` |
| `reader.pt` | 78,323,637 | `d605f6fba8687497be6b5fd717ea6c51beaee5d5f69c30a64225da96ebe0d8c2` |
| `reader_config.json` | 2,080 | `3aede371f7fc1b69bd948a3fce6f861e44eedb237d7b6dce360063ea007a90ed` |

`vocab.txt` and `word_table.npy` are the files of v1, unchanged. `reader.pt` and `word_table.npy` are ignored by git (`*.pt`, `*.npy` in `.gitignore`); the SHA-256 above identifies them. `reader_config.json` is the contract for ports: input, output, logits, decoding and the 30 s window policy of the encoder.

## What the reader is

- Input: EmbeddingGemma 2 audio soft tokens, float32 `[T, 512]`, 25 per second (`embed_audio` output of `google/embeddinggemma-2` revision 914f7f89; the encoder is frozen).
- Body: a bidirectional transformer encoder trained from scratch: LayerNorm -> Linear 512->512, 6 pre-LayerNorm layers (d 512, 8 heads, FFN 2048 GELU, rotary (base 10000) positions), LayerNorm -> Linear 512->768. Attention covers the whole input. 19,573,762 parameters in total.
- Output per token: a unit vector in the 768-d text space and a blank logit. Logits over the 200,000 words = scale x cosine to `word_table.npy`, scale = 216.351 (learned); blank is class N. Decoding: argmax per token, merge repeats, drop blanks.

## WER

Greedy, no language model, every utterance of each set. Normalization of reference and hypothesis: lower case, runs of spaces collapsed, nothing else (LibriSpeech transcripts have no punctuation and keep apostrophes). Words outside the vocabulary stay in the references and count as errors.

| Set | Utterances | Reference words | WER | Substitutions | Deletions | Insertions |
|---|---:|---:|---:|---:|---:|---:|
| dev-clean (model selection) | 2,703 | 54,402 | 3.75 % | 1,621 | 210 | 211 |
| dev-other | 2,864 | 50,948 | 10.26 % | 4,109 | 679 | 441 |
| test-clean | 2,620 | 52,576 | 4.21 % | 1,739 | 247 | 226 |
| test-other | 2,939 | 52,343 | 10.42 % | 4,249 | 655 | 551 |
| fixtures `test-clean-20` | 20 | 541 | 3.70 % | 18 | 1 | 1 |

The fixture row decodes `fixtures/test-clean-20/<id>/soft_tokens.npy` (the float32 CPU reference soft tokens). The reader's output on each of them, for comparing ports of the reader:

| Utterance | Word errors | Reader output |
|---|---:|---|
| `1089-134686-0000` | 2 / 28 | he hoped there would bestow stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fatten sauce |
| `1089-134686-0002` | 1 / 18 | after early nightfall the yellow lamps would light up here and there the squalid quarter of the broths |
| `1188-133604-0000` | 2 / 19 | you will find me continually speaking of four men titian holborn turner and tintoret a in almost the same terms |
| `1188-133604-0001` | 1 / 18 | they unite every quality and sometimes you will find me referring to them as colorists sometimes as kio |
| `121-121726-0000` | 0 / 17 | also a popular contrivance whereby love making may be suspended but not stopped during the picnic season |
| `121-121726-0001` | 0 / 8 | harangue the tiresome product of a tireless tongue |
| `1221-135766-0000` | 0 / 37 | how strange it seemed to the sad woman as she watched the growth and the beauty that became every day more brilliant and the intelligence that threw its quivering sunshine over the tiny features of this child |
| `1221-135766-0001` | 1 / 48 | god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bottom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven |
| `1284-1180-0000` | 1 / 24 | he wore blue silk stockings blue knee pans with gold buckles a blue ruffled waist and a jacket of bright blue braided with gold |
| `1284-1180-0001` | 1 / 26 | his hat had a peaked crown in a flat brim and around the brim was a row of tiny golden bells that tinkled when he moved |
| `1320-122612-0000` | 1 / 41 | since the period of our the active spirit of the country has surrounded it with a belt of rich and thriving settlements though none but the hunter or the savage is ever known even now to penetrate its wild recesses |
| `1320-122612-0001` | 2 / 30 | the dews were suffered to exhale and the sun had dispersed the mists and was shedding a strong and clear light in the forests when the travellers resumed their journey |
| `1580-141083-0000` | 1 / 30 | i will endeavor in my statement to avoid such terms as would serve to limit the events to any particular place or give a clue as to the people concerned |
| `1580-141083-0001` | 0 / 34 | i had always known him to be restless in his manner but on this particular occasion he was in such a state of uncontrollable agitation that it was clear something very unusual had occurred |
| `1995-1826-0000` | 0 / 27 | in the debate between the senior societies her defence of the fifteenth amendment had been not only a notable bit of reasoning but delivered with real enthusiasm |
| `1995-1826-0001` | 0 / 27 | the south she had not thought of seriously and yet knowing of its delightful hospitality and mild climate she was not averse to charleston or new orleans |
| `2094-142345-0001` | 1 / 21 | but the windows are patched with wooden panes and the door i think is like the gate it isn't never opened |
| `2094-142345-0002` | 3 / 42 | for it is a solid heavy handsome door and must once have been in the habit of shedding with a sonorous bang behind the liveried lackey who had just seen his master and mistress off the grounds in a carriage and pear |
| `2300-131720-0000` | 2 / 13 | but paris plant like that of the crystal palace was a temporary exhibit |
| `2300-131720-0001` | 1 / 33 | the london plant was last temporary but not permanent supplying before it was torn out no fewer than three thousand lamps in hotels churches stores and dwellings in the vicinity of holborn viaduct |

## Training

- Data: LibriSpeech train-clean-100, train-clean-360 and train-other-500 (CC BY 4.0), 281,241 utterances, 961.6 h, read as one set per epoch. Utterances with a word outside the vocabulary were left out: 261,486 utterances, 890.2 h, 80,121,094 tokens in training.

| Split | Utterances | Hours | Left out (word outside the vocabulary) | In training | Hours in training |
|---|---:|---:|---:|---:|---:|
| train-clean-100 | 28,539 | 100.65 | 2,117 | 26,422 | 92.79 |
| train-clean-360 | 104,014 | 363.82 | 7,532 | 96,482 | 336.45 |
| train-other-500 | 148,688 | 497.17 | 10,106 | 138,582 | 461.00 |

- Soft tokens: computed once on the training machine in float32 with TF32 off (stored float16), 16 utterances of similar length per encoder call (checked against one at a time: per-token cosine min 0.999999994, max |difference| 0.00051). Check before use: the 20 fixture utterances (5,036 tokens) against the float32 CPU reference soft tokens: per-token cosine min 0.9999999978, max |difference| 0.00036.
- The cache index (ids, samples, frames, tokens, transcripts) of dev-clean, dev-other, test-clean, test-other and train-clean-100 is identical to the v1 cache index.
- Dev: dev-clean, used for the choice of the reader body and for early stopping. test-clean and test-other were decoded once, after training.
- Objective: CTC over the 200,000 words + blank (`torch.nn.functional.ctc_loss`, float32, zero_infinity, mean of loss / target length).
- Optimizer: AdamW, peak learning rate 0.0003, weight decay 0.01 on matrices, linear warmup over 500 steps, then cosine to zero at step 163,430, batch 16 utterances bucketed by length, gradient clipping 1.0, float32 weights and activations; training matmuls in TF32 (A100), evaluation in fp32: the dev, test and fixture WER are decoded in full float32, as in v1.
- Length: 9 epochs, 147,087 steps; best dev-clean WER at epoch 7 (the released weights). Stop: early stop: no better dev-clean WER for 2 epochs.
- Time: 4.38 h of training plus 2 min of dev decoding on one NVIDIA A100-SXM4-80GB (Google Colab, torch 2.11.0+cu130, CUDA 13.0).
- Colab compute units: 95.7 in total (both runtimes: soft tokens, training, evaluation), 6.77 per hour per A100 runtime; about 22 of them were the two runtimes waiting idle after the work, before they were deleted.

| Epoch | Steps | Train loss (last 100) | dev-clean WER | Training (min) | Dev decoding (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 16,343 | 0.318 | 6.08 % | 29.2 | 15 |
| 2 | 32,686 | 0.242 | 5.16 % | 29.5 | 12 |
| 3 | 49,029 | 0.203 | 4.57 % | 29.3 | 12 |
| 4 | 65,372 | 0.170 | 4.38 % | 29.2 | 12 |
| 5 | 81,715 | 0.154 | 4.10 % | 29.2 | 12 |
| 6 | 98,058 | 0.103 | 3.83 % | 29.2 | 12 |
| 7 | 114,401 | 0.082 | 3.75 % | 29.1 | 12 |
| 8 | 130,744 | 0.052 | 3.77 % | 29.2 | 12 |
| 9 | 147,087 | 0.037 | 3.79 % | 29.2 | 12 |

Command of the released run (paths as on the training machine):

```sh
python -u /content/r7/code/train/train.py --arch a --layers 6 --cache /content/r7/cache --vocab /content/r7/assets/reader/vocab.txt --table /content/r7/assets/reader/word_table.npy --run-dir /content/r7/runs/v2a6 --device cuda --train-split train-clean-100,train-clean-360,train-other-500 --dev-split dev-clean --epochs 10 --schedule-epochs 10 --patience 2 --max-hours 4.727 --matmul tf32 --log-every 500 --stop-file /content/r7/control/stop_v2a6 --resume /content/r7/runs/v2a6/ckpt_epoch1.pt
```

### Candidates (epoch 1 of each, the same data and schedule)

| Body | Parameters | dev-clean WER after 1 epoch | Train loss (last 100) | Epoch (min) |
|---|---:|---:|---:|---:|
| a, 4 layers | 13,268,994 | 6.27 % | 0.322 | 27.5 |
| a, 6 layers | 19,573,762 | 6.08 % | 0.318 | 29.2 |

Both bodies, trained to the end (early stopping on dev-clean, patience 2) and decoded the same way:

| Body | Epochs | Best epoch | dev-clean | dev-other | test-clean | test-other | fixtures |
|---|---:|---:|---:|---:|---:|---:|---:|
| a, 4 layers | 10 | 8 | 3.88 % | 10.77 % | 4.32 % | 10.88 % | 3.14 % |
| a, 6 layers (released) | 9 | 7 | 3.75 % | 10.26 % | 4.21 % | 10.42 % | 3.70 % |

Chosen: arch a with 6 layers. After one epoch its dev-clean WER was the lower one (6.08 % against 6.27 %), so both runs went on; each stopped early on dev-clean, the 6-layer run at epoch 9 (best epoch 7, 3.75 %) and the 4-layer run at epoch 10 (best epoch 8, 3.88 %). The choice uses dev-clean only. The 6-layer run trained epoch 1 with --epochs 1 and then continued from that checkpoint with --epochs 10 (the command above); the cosine schedule spans 10 epochs in both.

## Vocabulary and word table

- `vocab.txt`: 200,000 words = every word of the LibriSpeech lexicon (openslr.org/resources/11, public domain), lower case, sorted by code point, apostrophes kept (as in v1).
- `word_table.npy`: EmbeddingGemma 2 text embedding of each word with the prompt `task: search result | query: ` (sentence-transformers, float32, then stored as float16, L2-normalized). Check: each of 100 random words, embedded again, comes back at rank 1 (top-1 1.000).
- Out of vocabulary: test-clean 220 of 52,576 words (0.42 %), test-other 283 of 52,343 (0.54 %); fixtures 1 word (`chiaroscurists`). These are always errors.

## Known weaknesses

- By word length: on the 2,518 aligned words of 100 test-clean utterances, the word error rate is 1.8 % for words of 1-2 tokens (<= 80 ms, n 824), 2.0 % for 3-5 tokens (n 700) and 6.1 % for 6+ tokens (n 994).
- Substitutions on test-clean: 1,739 in 1,420 distinct pairs; most frequent and -> in (27), in -> and (22), a -> the (16), the -> a (11), an -> and (9), this -> the (6), i've -> i (5), man -> men (5). Homophones (same lexicon pronunciation): 3 of the top 20 pairs, 12.9 % of all substitutions; a reader without a language model cannot tell them apart.
- No language model, no punctuation or casing; trained only on read English audiobook speech.
- Trained on whole LibriSpeech utterances of up to 29.7 s and measured on whole utterances of up to 35.2 s. Long audio goes through the encoder in 30 s windows (see `reader_config.json` window and "Long audio in 30 s windows" below).

## Long audio in 30 s windows

The long fixture `fixtures/long-speckled-band` (a 3,257.4 s LibriVox recording) was read in 125 windows of 30 s, each stepping 653 tokens with a 96-token left overlap and its last token dropped, through the Core AI conversion of the encoder (GPU) and of this reader (fp16, Neural Engine) on an Apple M4 Max: WER 3.80 % against the book text (9,889 words; the LibriVox intro and outro are in the audio only), next to 3.81 % for the torch float32 path on the same windows. The labels of the 81,436 tokens are identical whether the windows run one at a time or two in flight. Records: `exports/pipeline_v2/pipeline_long_fp16_ane_overlap.json` and `exports/pipeline_v2/pipeline_long_torch.json` of the Core AI conversion work (not in this repository).

## Reproduce

`train/README.md` lists the environment, the data downloads and every command; v2 ran the same scripts on CUDA with the three training splits joined by commas in `--train-split` (the command above). `train/model.py --self-test` checks the CTC objective.

## Licenses and credits

- LibriSpeech: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books", ICASSP 2015, https://www.openslr.org/12, CC BY 4.0. The reader weights are trained on train-clean-100, train-clean-360 and train-other-500.
- LibriSpeech lexicon: https://www.openslr.org/11, public domain.
- EmbeddingGemma 2 (`google/embeddinggemma-2`, Google DeepMind): Apache 2.0; it produces the soft tokens and the word table. Deployments must adhere to the Gemma Prohibited Use Policy (https://ai.google.dev/gemma/prohibited_use_policy).
- `reader.pt` and `word_table.npy` are this repository's own artifacts: the reader weights were trained with `train/train.py` on the LibriSpeech data above, and the table was computed with `train/make_table.py` from EmbeddingGemma 2 and the lexicon words.
