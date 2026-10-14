#!/usr/bin/env python3
"""Build the reader's word table: vocab.txt (one word per line) and word_table.npy (float16 [N, 768], L2-normalized).

Vocabulary = every word of the LibriSpeech lexicon (column 1, lower case, unique, sorted by code point; the
LibriSpeech spelling keeps apostrophes). Each word is embedded by EmbeddingGemma 2 through sentence-transformers
(float32, default attention) with the query prompt "task: search result | query: " in front of the word, as text;
the pooled 768-d output is L2-normalized and stored as float16.

Checks written to --report: text -> text top-1 for --check-words random words (the word, re-encoded, must find
itself), and the out-of-vocabulary words of each transcript set given with --index (cache index.tsv files) or
--fixtures (a directory searched for transcript.txt files).

Example:
  python -I make_table.py --lexicon librispeech-lexicon.txt --model-dir embeddinggemma-2 --out-dir .. \
      --index cache/test-clean/index.tsv cache/test-other/index.tsv --fixtures ../fixtures/test-clean-20 \
      --report make_table.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

PROMPT = "task: search result | query: "


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lexicon_words(path):
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                words.add(parts[0].lower())
    return sorted(words)


def transcript_sets(index_paths, fixture_dir):
    """-> {name: list of transcripts}."""
    sets = {}
    for p in index_paths or []:
        name = os.path.basename(os.path.dirname(os.path.abspath(p)))
        with open(p, encoding="utf-8") as f:
            sets[name] = [line.rstrip("\n").split("\t")[4] for line in f if line.strip()]
    if fixture_dir:
        texts = []
        for p in sorted(glob.glob(os.path.join(fixture_dir, "**", "transcript.txt"), recursive=True)):
            with open(p, encoding="utf-8") as f:
                texts.append(" ".join(f.read().lower().split()))
        sets["fixtures:" + os.path.basename(os.path.normpath(fixture_dir))] = texts
    return sets


def oov_stats(texts, vocab_set):
    tokens = [w for t in texts for w in t.split()]
    oov_tokens = [w for w in tokens if w not in vocab_set]
    oov_utts = sum(1 for t in texts if any(w not in vocab_set for w in t.split()))
    return {"utterances": len(texts), "tokens": len(tokens), "types": len(set(tokens)),
            "oov_tokens": len(oov_tokens), "oov_types": len(set(oov_tokens)),
            "oov_token_ratio": len(oov_tokens) / max(1, len(tokens)),
            "utterances_with_oov": oov_utts, "oov_examples": sorted(set(oov_tokens))[:20]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", required=True, help="librispeech-lexicon.txt (openslr.org/resources/11)")
    ap.add_argument("--model-dir", required=True, help="local copy of google/embeddinggemma-2 (revision 914f7f89)")
    ap.add_argument("--out-dir", required=True, help="writes vocab.txt and word_table.npy here")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--check-words", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", nargs="*", default=[], help="cache index.tsv files for the OOV report")
    ap.add_argument("--fixtures", default=None, help="directory with */transcript.txt for the OOV report")
    ap.add_argument("--report", default=None, help="JSON report path")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    vocab = lexicon_words(args.lexicon)
    log(f"vocabulary: {len(vocab)} words from {args.lexicon}")
    st = SentenceTransformer(args.model_dir, device=args.device, model_kwargs={"dtype": torch.float32},
                             config_kwargs={"vision_config": None})
    t_load = time.time() - t0
    t1 = time.time()
    emb = st.encode([args.prompt + w for w in vocab], batch_size=args.batch_size, convert_to_numpy=True,
                    show_progress_bar=False).astype(np.float32)
    t_encode = time.time() - t1
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    table = emb.astype(np.float16)
    log(f"encoded {len(vocab)} words in {t_encode:.1f} s on {args.device} ({len(vocab) / t_encode:.0f} words/s)")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    np.save(os.path.join(args.out_dir, "word_table.npy"), table)

    # text -> text: re-encode random words with the same prompt; each must come back at rank 1.
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(vocab), size=min(args.check_words, len(vocab)), replace=False)
    q = st.encode([args.prompt + vocab[i] for i in pick], batch_size=args.check_words, convert_to_numpy=True,
                  show_progress_bar=False).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sims = q @ table.astype(np.float32).T
    top1 = sims.argmax(1)
    hit = float((top1 == pick).mean())
    gap = np.sort(sims, axis=1)[:, -2]
    log(f"text -> text top-1 {hit:.3f} over {len(pick)} words "
        f"(self cos min {sims[np.arange(len(pick)), pick].min():.6f}, runner-up cos max {gap.max():.6f})")

    vocab_set = set(vocab)
    oov = {name: oov_stats(texts, vocab_set) for name, texts in transcript_sets(args.index, args.fixtures).items()}
    for name, s in oov.items():
        log(f"OOV {name}: {s['oov_tokens']} / {s['tokens']} tokens ({100 * s['oov_token_ratio']:.2f} %), "
            f"{s['oov_types']} types, {s['utterances_with_oov']} / {s['utterances']} utterances")

    norms = np.linalg.norm(table.astype(np.float32), axis=1)
    report = {
        "vocab_size": len(vocab), "prompt": args.prompt, "device": args.device,
        "model_load_s": round(t_load, 1), "encode_s": round(t_encode, 1),
        "words_per_s": round(len(vocab) / t_encode, 1),
        "table": {"dtype": "float16", "shape": list(table.shape),
                  "norm_min": float(norms.min()), "norm_max": float(norms.max())},
        "text_to_text": {"n": int(len(pick)), "top1": hit,
                         "self_cos_min": float(sims[np.arange(len(pick)), pick].min()),
                         "runner_up_cos_max": float(gap.max()), "seed": args.seed},
        "oov": oov,
        "versions": {"torch": torch.__version__,
                     "sentence_transformers": __import__("sentence_transformers").__version__,
                     "transformers": __import__("transformers").__version__},
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        log(f"-> {args.report}")


if __name__ == "__main__":
    main()
