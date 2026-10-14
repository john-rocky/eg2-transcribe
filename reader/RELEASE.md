# Reader v1

Released 2026-10-14. Reads EmbeddingGemma 2 audio soft tokens as words. Greedy decoding, no language model: **test-clean WER 6.81 %, test-other WER 15.10 %**.

Pre-registered v1 line: test-clean <= 10 % and test-other <= 20 % (greedy, no LM). test-clean meets it (6.81 %), test-other meets it (15.10 %).

## Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `vocab.txt` | 1,737,588 | `bb3b3c70379082333e1bbd90d07ca69fe5c0096bd1c3b2b3f83edd65f1062a68` |
| `word_table.npy` | 307,200,128 | `b2d0b13cc70c492f1444b8a42deab437d25254eac47d752d777d27b65377faa8` |
| `reader.pt` | 53,095,605 | `67a06a857b9aeb066347f8947e4faeed11d2ec3e89e0c23974be0a2c20243d5b` |
| `reader_config.json` | 2,019 | `68efaabae4c2eaa8ae2a5abf8f85fab115791b13178ee3a22e57a0cec727b94e` |

`reader.pt` and `word_table.npy` are ignored by git (`*.pt`, `*.npy` in `.gitignore`); the SHA-256 above identifies them. `reader_config.json` is the contract for ports: input, output, logits, decoding and the 30 s window policy of the encoder.

## What the reader is

- Input: EmbeddingGemma 2 audio soft tokens, float32 `[T, 512]`, 25 per second (`embed_audio` output of `google/embeddinggemma-2` revision 914f7f89; the encoder is frozen).
- Body: a bidirectional transformer encoder trained from scratch: LayerNorm -> Linear 512->512, 4 pre-LayerNorm layers (d 512, 8 heads, FFN 2048 GELU, rotary (base 10000) positions), LayerNorm -> Linear 512->768. Attention covers the whole input. 13,268,994 parameters in total.
- Output per token: a unit vector in the 768-d text space and a blank logit. Logits over the 200,000 words = scale x cosine to `word_table.npy`, scale = 91.218 (learned); blank is class N. Decoding: argmax per token, merge repeats, drop blanks.

## WER

Greedy, no language model, every utterance of each set. Normalization of reference and hypothesis: lower case, runs of spaces collapsed, nothing else (LibriSpeech transcripts have no punctuation and keep apostrophes). Words outside the vocabulary stay in the references and count as errors.

| Set | Utterances | Reference words | WER | Substitutions | Deletions | Insertions |
|---|---:|---:|---:|---:|---:|---:|
| dev-clean (model selection) | 2,703 | 54,402 | 6.57 % | 2,823 | 375 | 374 |
| dev-other | 2,864 | 50,948 | 14.93 % | 5,885 | 1,089 | 631 |
| test-clean | 2,620 | 52,576 | 6.81 % | 2,789 | 412 | 378 |
| test-other | 2,939 | 52,343 | 15.10 % | 6,040 | 1,141 | 724 |
| fixtures `test-clean-20` | 20 | 541 | 4.81 % | 22 | 1 | 3 |

The fixture row decodes `fixtures/test-clean-20/<id>/soft_tokens.npy` (the float32 CPU reference soft tokens). The reader's output on each of them, for comparing ports of the reader:

| Utterance | Word errors | Reader output |
|---|---:|---|
| `1089-134686-0000` | 2 / 28 | he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat muttered pieces to be ladled out in thick peppered flour fatten sauce |
| `1089-134686-0002` | 3 / 18 | after early nightfall fall all the yellow lamps would light up here and there the squalid quarter of the brawls |
| `1188-133604-0000` | 2 / 19 | you will find me continually speaking of four men tion sublime turner and tintoret in almost the same terms |
| `1188-133604-0001` | 2 / 18 | the unite every quality and sometimes you will find me referring to them as colorists sometimes as curio |
| `121-121726-0000` | 2 / 17 | also a popular can drives whereby love making may be suspended but not stopped during the picnic season |
| `121-121726-0001` | 1 / 8 | hurrah the tiresome product of a tireless tongue |
| `1221-135766-0000` | 1 / 37 | how strange it seemed to the sad woman as she watched the growth and the beauty that became every day more brilliant and the intelligence that through its quivering sunshine over the tiny features of this child |
| `1221-135766-0001` | 2 / 48 | god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonored bottom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven |
| `1284-1180-0000` | 0 / 24 | he wore blue silk stockings blue knee pants with gold buckles a blue ruffled waist and a jacket of bright blue braided with gold |
| `1284-1180-0001` | 1 / 26 | his hat had a peaked crown at a flat brim and around the brim was a row of tiny golden bells that tinkled when he moved |
| `1320-122612-0000` | 1 / 41 | since the period of art tale the active spirit of the country has surrounded it with a belt of rich and thriving settlements though none but the hunter or the savage is ever known even now to penetrate its wild recesses |
| `1320-122612-0001` | 0 / 30 | the dews were suffered to exhale and the sun had dispersed the mists and was shedding a strong and clear light in the forest when the travelers resumed their journey |
| `1580-141083-0000` | 2 / 30 | i will endeavor in my statement to avoid such terms as would serve to limit the events to any particular place or give a clew as to the people concerned |
| `1580-141083-0001` | 0 / 34 | i had always known him to be restless in his manner but on this particular occasion he was in such a state of uncontrollable agitation that it was clear something very unusual had occurred |
| `1995-1826-0000` | 0 / 27 | in the debate between the senior societies her defence of the fifteenth amendment had been not only a notable bit of reasoning but delivered with real enthusiasm |
| `1995-1826-0001` | 1 / 27 | the south she had not thought of seriously and yet knowing of its delightful hospitality and mild climate she was not averse to charlestown or new orleans |
| `2094-142345-0001` | 0 / 21 | but the windows are patched with wooden panes and the door i think is like the gate it is never opened |
| `2094-142345-0002` | 2 / 42 | for it is a solid heavy handsome door and must once have been in the habit of shedding with a sonorous bang behind the liveried lackey who had just seen his master and mistress off the grounds in a carriage and pair |
| `2300-131720-0000` | 1 / 13 | the paris plant like that of the crystal palace was a temporary exhibit |
| `2300-131720-0001` | 3 / 33 | the london plat was last temporary but not permanent supplying before it was torn out no fewer than three thousand lamps in hotels churches stores and dwellings in the vicinity of holborn |

## Training

- Data: LibriSpeech train-clean-100 (CC BY 4.0), 28,539 utterances, 100.65 h. The 2,117 utterances with a word outside the vocabulary were left out: 26,422 utterances, 92.8 h, 8,351,376 tokens. Soft tokens computed once (float32, stored float16).
- Dev: dev-clean, used for the choice of the reader body and for early stopping. test-clean and test-other were decoded once, after training.
- Objective: CTC over the 200,000 words + blank (`torch.nn.functional.ctc_loss`, float32, zero_infinity, mean of loss / target length).
- Optimizer: AdamW, peak learning rate 0.0003, weight decay 0.01 on matrices, linear warmup over 500 steps, then cosine to zero at step 16,520, batch 16 utterances bucketed by length, gradient clipping 1.0, float32 throughout.
- Length: 8 epochs, 13,216 steps; best dev-clean WER at epoch 8 (the released weights). Stop: active-time budget: 2.72 h used over 8 epochs, one more (~0.34 h) would pass 3.0 h.
- Time: 3.64 h of training wall clock (1.02 h of it paused while other jobs measured on the same Mac) plus 6 min of dev decoding. Apple M4 Max (128 GB), torch 2.11.0 MPS, macOS 27.0 (26A428).

| Epoch | Steps | Train loss (last 100) | dev-clean WER | Wall (min) | Paused (min) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,652 | 1.103 | 13.48 % | 22.2 | 3.0 |
| 2 | 3,304 | 0.434 | 9.24 % | 22.2 | 2.0 |
| 3 | 4,956 | 0.281 | 7.99 % | 45.7 | 21.9 |
| 4 | 6,608 | 0.199 | 7.24 % | 22.0 | 3.4 |
| 5 | 8,260 | 0.136 | 7.07 % | 34.5 | 16.7 |
| 6 | 9,912 | 0.094 | 6.84 % | 20.8 | 2.8 |
| 7 | 11,564 | 0.057 | 6.74 % | 20.1 | 1.0 |
| 8 | 13,216 | 0.028 | 6.57 % | 31.1 | 10.2 |

Command that continued the chosen body (paths as in `train/README.md`):

```sh
mkdir -p runs/v1 && cp runs/cmp_a/ckpt_epoch1.pt runs/v1/
for e in $(seq 2 10); do  # one process per epoch, see train/README.md "Memory on Apple silicon"
  .venv/bin/python -I train.py --arch a --cache cache --vocab ../vocab.txt --table ../word_table.npy \
    --run-dir runs/v1 --resume runs/v1/ckpt_epoch$((e - 1)).pt --epochs $e --schedule-epochs 10 \
    --max-hours 3 --patience 2 || break
  [ "$(jq -r .stop_reason runs/v1/metrics.json)" = "reached $e epochs" ] || break  # early stop or budget
done
```

### Candidates (one epoch each, the same data and schedule)

| Body | Parameters | dev-clean WER after 1 epoch | Train loss (last 100) | Epoch, min (paused) | Peak MPS (GB) |
|---|---:|---:|---:|---:|---:|
| a: transformer from scratch (4 layers) | 13,268,994 | 13.48 % | 1.103 | 19.3 (3.0) | 38.6 |
| b: EmbeddingGemma 2 text layers 0..3 | 22,427,394 | 14.60 % | 2.104 | 23.6 (4.6) | 28.9 |
| c: per-token MLP | 1,314,306 | 25.21 % | 1.788 | 16.2 (26.6) | 10.0 |

Epoch minutes count training only (no dev decoding); the paused time, while other jobs measured on the same Mac, is shown in parentheses and not counted.

Chosen: `cmp_a` (lowest dev-clean WER (greedy, no LM) after one epoch; tie -> fewer parameters; test sets not used).

## Vocabulary and word table

- `vocab.txt`: 200,000 words = every word of the LibriSpeech lexicon (openslr.org/resources/11, public domain), lower case, sorted by code point, apostrophes kept.
- `word_table.npy`: EmbeddingGemma 2 text embedding of each word with the prompt `task: search result | query: ` (sentence-transformers, float32, then stored as float16, L2-normalized). Check: each of 100 random words, embedded again, comes back at rank 1 (top-1 1.000).
- Out of vocabulary: test-clean 220 of 52,576 words (0.42 %), test-other 283 of 52,343 (0.54 %); fixtures 1 word (`chiaroscurists`). These are always errors.

## Known weaknesses

- Long words err most: on the 2,518 aligned words of 100 test-clean utterances, the word error rate rises with word length: 2.8 % for words of 1-2 tokens (<= 80 ms, n 824), 3.4 % for 3-5 tokens (n 700) and 9.7 % for 6+ tokens (n 994). The frozen-embedding probe before training had the opposite profile (short words hardest); the trained reader reverses it.
- Substitutions on test-clean: 2,789 in 2,234 distinct pairs; most frequent in -> and (41), and -> in (39), the -> a (20), an -> and (20), a -> the (14), is -> as (11), of -> a (10), this -> the (8). Homophones (same lexicon pronunciation): 2 of the top 20 pairs, 12.4 % of all substitutions; a reader without a language model cannot tell them apart.
- No language model, no punctuation or casing; trained only on read English audiobook speech.
- Trained on whole LibriSpeech utterances of up to 24.5 s and measured on whole utterances of up to 35.2 s. Long audio goes through the encoder in 30 s windows (see `reader_config.json` window and "Long audio in 30 s windows" below).

## Long audio in 30 s windows

The long fixture `fixtures/long-speckled-band` (a 3,257.4 s LibriVox recording) was read in 125 windows of 30 s, each stepping 653 tokens with a 96-token left overlap and its last token dropped, through the Core AI conversion of the encoder (GPU) and of this reader (fp16, Neural Engine) on an Apple M4 Max: WER 5.80 % against the book text (9,889 words; the LibriVox intro and outro are in the audio only), next to 5.78 % for the torch float32 path on the same windows. The labels of the 81,436 tokens are identical whether the windows run one at a time or two in flight. Records: `exports/pipeline/pipeline_long_fp16_ane_overlap.json` and `exports/pipeline/pipeline_long_torch.json` of the Core AI conversion work (not in this repository).

## Reproduce

`train/README.md` lists the environment, the data downloads and every command; `train/model.py --self-test` checks the CTC objective.

## Licenses and credits

- LibriSpeech: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books", ICASSP 2015, https://www.openslr.org/12, CC BY 4.0. The reader weights are trained on train-clean-100.
- LibriSpeech lexicon: https://www.openslr.org/11, public domain.
- EmbeddingGemma 2 (`google/embeddinggemma-2`, Google DeepMind): Apache 2.0; it produces the soft tokens and the word table. Deployments must adhere to the Gemma Prohibited Use Policy (https://ai.google.dev/gemma/prohibited_use_policy).
- `reader.pt` and `word_table.npy` are this repository's own artifacts: the reader weights were trained with `train/train.py` on the LibriSpeech data above, and the table was computed with `train/make_table.py` from EmbeddingGemma 2 and the lexicon words.
