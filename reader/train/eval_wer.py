#!/usr/bin/env python3
"""Export a trained reader and measure its WER (greedy, no language model).

--from-run RUN_DIR: copy RUN_DIR/best_model.pt to <reader-dir>/reader.pt and write <reader-dir>/reader_config.json.
Then decode every utterance of --splits (cached soft tokens) and of --fixtures (soft_tokens.npy + transcript.txt),
write <out-dir>/hyp_<split>.tsv (id, reference, hypothesis) and <out-dir>/wer.json, and put the WER into
reader_config.json. Normalization (reference and hypothesis alike): lower case, runs of spaces collapsed to one,
nothing else (LibriSpeech transcripts have no punctuation and keep apostrophes).

Analysis in wer.json: substitutions / deletions / insertions; word error rate by spoken word length on the 2,518
words of the r1 probe (100 test-clean utterances with forced-alignment spans in 40 ms tokens: 1-2 / 3-5 / 6+);
the 20 most frequent substitution pairs and the share of homophones (same LibriSpeech lexicon pronunciation,
stress marks ignored).

Example:
  python -I eval_wer.py --from-run runs/v1 --reader-dir .. --cache cache --out-dir runs/v1 \
      --fixtures ../fixtures/test-clean-20 --spans neighbor_probe_spans.npz --lexicon librispeech-lexicon.txt
"""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D  # noqa: E402
from decode import load_reader_dir, transcribe_batch, transcribe_split  # noqa: E402
from model import limit_mps_memory  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(text):
    return " ".join(text.lower().split())


def score(refs, hyps):
    import jiwer
    o = jiwer.process_words([norm(r) for r in refs], [norm(h) for h in hyps])
    n = o.substitutions + o.deletions + o.hits
    return o, {"wer": o.wer, "substitutions": o.substitutions, "deletions": o.deletions,
               "insertions": o.insertions, "hits": o.hits, "ref_words": n, "utterances": len(refs)}


def ref_word_outcomes(o):
    """-> per utterance, a list over reference words of 'ok' / 'sub' / 'del' from jiwer alignments."""
    out = []
    for chunks, ref in zip(o.alignments, o.references):
        tags = [None] * len(ref)
        for c in chunks:
            t = {"equal": "ok", "substitute": "sub", "delete": "del"}.get(c.type)
            if t:
                for i in range(c.ref_start_idx, c.ref_end_idx):
                    tags[i] = t
        out.append(tags)
    return out


def substitution_pairs(o):
    pairs = collections.Counter()
    for chunks, ref, hyp in zip(o.alignments, o.references, o.hypotheses):
        for c in chunks:
            if c.type == "substitute":
                for a, b in zip(ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]):
                    pairs[(a, b)] += 1
    return pairs


def pronunciations(lexicon):
    pron = collections.defaultdict(set)
    with open(lexicon, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                pron[parts[0].lower()].add(" ".join(re.sub(r"\d", "", p) for p in parts[1:]))
    return pron


def write_config(reader_dir, model_scale, n_vocab, table_shape, arch, params):
    cfg = {
        "version": "v1",
        "input": {"shape": "[T, 512]", "dtype": "float32",
                  "what": "EmbeddingGemma 2 audio soft tokens (embed_audio output), 25 per second"},
        "context": "bidirectional within the window",
        "architecture": dict(arch, params=params),
        "output": {"vector": "[T, 768] L2-normalized", "blank_logit": "[T]"},
        "logits": "scale * cos(vector, table[w]) for w in vocab; blank from blank_logit",
        "scale": model_scale,
        "vocab": "vocab.txt",
        "table": {"file": "word_table.npy", "dtype": "float16", "shape": f"[{table_shape[0]}, {table_shape[1]}]",
                  "prompt": "task: search result | query: ", "normalized": True},
        "decode": "greedy argmax over N+1, collapse repeats, drop blank",
        "window": {"seconds": 30, "left_overlap_tokens": 96, "right_extra_tokens": 1,
                   "note": "encoder is causal; the 83 tokens at the start of a window and its last token differ from a continuous "
                           "run (fixtures/long-speckled-band: soft_tokens_w30 vs soft_tokens_continuous)"},
        "weights": {"file": "reader.pt", "format": "torch state_dict, float32",
                    "code": "train/model.py Reader(arch, n_vocab, k)",
                    "keys": "body.* (reader body), blank.weight [1, 768], blank.bias [1], "
                            "log_scale (scale = exp(log_scale))"},
        "n_vocab": n_vocab,
        "wer": {},
    }
    with open(os.path.join(reader_dir, "reader_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader-dir", required=True, help="holds (or receives) reader.pt, reader_config.json, vocab, table")
    ap.add_argument("--from-run", default=None, help="export RUN_DIR/best_model.pt into --reader-dir before decoding")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other", "test-clean", "test-other"])
    ap.add_argument("--fixtures", default=None, help="directory with <id>/soft_tokens.npy and transcript.txt")
    ap.add_argument("--spans", default=None, help="r1 probe spans (npz: utt, word, s, e) for the length analysis")
    ap.add_argument("--lexicon", default=None, help="librispeech-lexicon.txt for the homophone share")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--pad-to", type=int, default=64, help="pad each batch's time axis to a multiple of this (0 = off)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device == "mps":
        limit_mps_memory(40.0)

    if args.from_run:
        with open(os.path.join(args.from_run, "metrics.json"), encoding="utf-8") as f:
            metrics = json.load(f)
        arch = metrics["arch"]
        shutil.copyfile(os.path.join(args.from_run, "best_model.pt"), os.path.join(args.reader_dir, "reader.pt"))
        sd = torch.load(os.path.join(args.reader_dir, "reader.pt"), map_location="cpu", weights_only=True)
        table_shape = np.load(os.path.join(args.reader_dir, "word_table.npy"), mmap_mode="r").shape
        write_config(args.reader_dir, float(sd["log_scale"].exp()), metrics["vocab"], table_shape,
                     arch, metrics["params"])
        log(f"exported {args.from_run}/best_model.pt (epoch {metrics['best_epoch']}) -> {args.reader_dir}")

    model, table, vocab, cfg = load_reader_dir(args.reader_dir, args.device)
    widx = {w: i for i, w in enumerate(vocab)}
    results, objects = {}, {}
    for name in args.splits:
        split = D.Split(args.cache, name, widx)
        t0 = time.time()
        hyps = transcribe_split(model, split, table, vocab, args.device, pad_to=args.pad_to)
        o, s = score(split.texts, [hyps[u] for u in split.ids])
        s["decode_s"] = round(time.time() - t0, 1)
        s["utterances_with_oov"] = len(split) - len(split.in_vocab())
        results[name], objects[name] = s, (o, split)
        with open(os.path.join(args.out_dir, f"hyp_{name}.tsv"), "w", encoding="utf-8") as f:
            for u, ref in zip(split.ids, split.texts):
                f.write(f"{u}\t{norm(ref)}\t{norm(hyps[u])}\n")
        log(f"{name}: WER {100 * s['wer']:.2f} % (S {s['substitutions']} D {s['deletions']} I {s['insertions']} "
            f"/ {s['ref_words']} words, {s['utterances']} utterances) in {s['decode_s']:.0f} s")

    if args.fixtures:
        ids, refs, hyps = [], [], []
        for p in sorted(glob.glob(os.path.join(args.fixtures, "*", "soft_tokens.npy"))):
            d = os.path.dirname(p)
            with open(os.path.join(d, "transcript.txt"), encoding="utf-8") as f:
                refs.append(norm(f.read()))
            hyps.append(norm(transcribe_batch(model, [np.load(p).astype(np.float32)], table, vocab, args.device,
                                             args.pad_to)[0]))
            ids.append(os.path.basename(d))
        _, s = score(refs, hyps)
        results["fixtures"] = dict(s, source=os.path.basename(os.path.normpath(args.fixtures)),
                                   soft_tokens="fixture soft_tokens.npy (float32, CPU reference)")
        with open(os.path.join(args.out_dir, "hyp_fixtures.tsv"), "w", encoding="utf-8") as f:
            for u, r, h in zip(ids, refs, hyps):
                f.write(f"{u}\t{r}\t{h}\n")
        log(f"fixtures: WER {100 * s['wer']:.2f} % over {len(ids)} utterances")

    analysis = {}
    if args.spans and "test-clean" in objects:
        sp = np.load(args.spans)
        o, split = objects["test-clean"]
        tags = dict(zip(split.ids, ref_word_outcomes(o)))
        refs = dict(zip(split.ids, [norm(t).split() for t in split.texts]))
        lengths = sp["e"] - sp["s"]
        pos = collections.defaultdict(int)
        outcome = []
        for u, w in zip(sp["utt"].tolist(), sp["word"].tolist()):
            k = pos[u]
            pos[u] += 1
            if refs[u][k] != w:
                raise AssertionError(f"span word {w!r} != reference word {refs[u][k]!r} in {u}")
            outcome.append(tags[u][k])
        outcome = np.array(outcome)
        bins = {"1-2": lengths <= 2, "3-5": (lengths >= 3) & (lengths <= 5), "6+": lengths >= 6}
        analysis["error_rate_by_span_tokens"] = {
            b: {"n": int(m.sum()), "error_rate": float((outcome[m] != "ok").mean()),
                "substituted": int((outcome[m] == "sub").sum()), "deleted": int((outcome[m] == "del").sum())}
            for b, m in bins.items()}
        analysis["error_rate_by_span_tokens"]["all"] = {"n": len(outcome),
                                                        "error_rate": float((outcome != "ok").mean())}
        analysis["span_source"] = "r1 probe: torchaudio WAV2VEC2_ASR_BASE_960H forced alignment, 100 test-clean utterances"
    if args.lexicon:
        pron = pronunciations(args.lexicon)
        homo = lambda a, b: bool(pron.get(a, set()) & pron.get(b, set()))  # noqa: E731
        for name in ("test-clean", "test-other"):
            if name not in objects:
                continue
            pairs = substitution_pairs(objects[name][0])
            total = sum(pairs.values())
            top = pairs.most_common(20)
            analysis[f"substitutions_{name}"] = {
                "total": total, "distinct_pairs": len(pairs),
                "top20": [{"ref": a, "hyp": b, "count": c, "homophone": homo(a, b)} for (a, b), c in top],
                "top20_homophone_pairs": sum(1 for (a, b), _ in top if homo(a, b)),
                "homophone_share_of_all": sum(c for (a, b), c in pairs.items() if homo(a, b)) / max(1, total)}

    out = {"decode": "greedy argmax over N+1, collapse repeats, drop blank; no language model",
           "normalization": "lower case, runs of spaces collapsed to one, nothing else",
           "reader": cfg["architecture"], "scale": cfg["scale"], "vocab": len(vocab),
           "results": results, "analysis": analysis}
    with open(os.path.join(args.out_dir, "wer.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    cfg["wer"] = {k: {"wer": round(v["wer"], 5), "utterances": v["utterances"], "ref_words": v["ref_words"]}
                  for k, v in results.items()}
    cfg["wer"]["decode"] = "greedy, no language model"
    with open(os.path.join(args.reader_dir, "reader_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    log(f"-> {os.path.join(args.out_dir, 'wer.json')}; WER written into reader_config.json")


if __name__ == "__main__":
    main()
