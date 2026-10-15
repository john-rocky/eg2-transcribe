#!/usr/bin/env python3
"""Data for the reader: cached soft tokens, transcripts as word ids, length-bucketed batches.

The cache is what cache_soft_tokens.py writes: <cache>/<split>/<id>.npy (float16 [n_tokens, 512]) and
<cache>/<split>/index.tsv (id, n_samples, n_frames, n_tokens, transcript). Soft tokens go back to float32 here.
Utterances with a word outside vocab.txt are left out of training (their number is reported).

CLI: print the statistics of the splits against a vocabulary.
"""
import argparse
import os
import queue
import threading

import numpy as np
import torch


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def read_index(cache_dir, split):
    """-> list of (id, n_tokens, transcript) in index order."""
    rows = []
    with open(os.path.join(cache_dir, split, "index.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                uid, _, _, n_tok, text = line.rstrip("\n").split("\t")
                rows.append((uid, int(n_tok), " ".join(text.split())))
    return rows


class Split:
    """One cached split, or several joined with commas ("train-clean-100,train-clean-360,train-other-500"): ids,
    lengths, transcripts and word ids (None when a word is out of vocabulary), in index order, split after split."""

    def __init__(self, cache_dir, split, word_index):
        self.cache_dir, self.name = cache_dir, split
        self.parts = split.split(",")
        rows, self.dirs = [], []
        for part in self.parts:
            part_rows = read_index(cache_dir, part)
            rows += part_rows
            self.dirs += [part] * len(part_rows)
        self.ids = [r[0] for r in rows]
        self.n_tokens = np.array([r[1] for r in rows], dtype=np.int64)
        self.texts = [r[2] for r in rows]
        self.word_ids = []
        for text in self.texts:
            ids = [word_index.get(w) for w in text.split()]
            self.word_ids.append(None if any(i is None for i in ids) else np.array(ids, dtype=np.int64))

    def __len__(self):
        return len(self.ids)

    def in_vocab(self):
        return [i for i, w in enumerate(self.word_ids) if w is not None]

    def tokens(self, i):
        return np.load(os.path.join(self.cache_dir, self.dirs[i], f"{self.ids[i]}.npy")).astype(np.float32)

    def part_stats(self):
        """-> {part: utterances, hours, in-vocabulary utterances and hours, utterances left out (OOV)}."""
        out = {}
        for part in self.parts:
            idx = [i for i, d in enumerate(self.dirs) if d == part]
            keep = [i for i in idx if self.word_ids[i] is not None]
            out[part] = {"utterances": len(idx), "hours": round(float(self.n_tokens[idx].sum()) / 25 / 3600, 3),
                         "tokens": int(self.n_tokens[idx].sum()), "in_vocab_utterances": len(keep),
                         "in_vocab_hours": round(float(self.n_tokens[keep].sum()) / 25 / 3600, 3),
                         "oov_left_out": len(idx) - len(keep)}
        return out


def ctc_feasible(n_tokens, word_ids):
    """CTC needs one frame per label plus one blank between equal neighbours."""
    repeats = int((word_ids[1:] == word_ids[:-1]).sum()) if len(word_ids) > 1 else 0
    return n_tokens >= len(word_ids) + repeats


def bucket_batches(lengths, items, batch_size, rng, jitter=0.05):
    """Sort `items` by length with a small random jitter, cut into batches, shuffle the batch order."""
    lengths = np.asarray(lengths, dtype=np.float64)[items]
    key = lengths * (1.0 + rng.uniform(-jitter, jitter, size=len(items)))
    order = np.asarray(items)[np.argsort(key, kind="stable")]
    batches = [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def sorted_batches(lengths, items, batch_size):
    """Deterministic length-sorted batches for evaluation."""
    order = np.asarray(items)[np.argsort(np.asarray(lengths)[items], kind="stable")]
    return [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]


def collate(split, idx, with_targets=True):
    """-> dict: x [B, T, 512] float32, lengths [B], and (for training) targets."""
    xs = [split.tokens(i) for i in idx]
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    x = torch.zeros(len(xs), int(lengths.max()), xs[0].shape[1], dtype=torch.float32)
    for b, t in enumerate(xs):
        x[b, :t.shape[0]] = torch.from_numpy(t)
    out = {"x": x, "lengths": lengths, "idx": list(idx)}
    if with_targets:
        out["words"] = [torch.from_numpy(split.word_ids[i]) for i in idx]
    return out


class Prefetcher:
    """Loads batches on a background thread so disk reads overlap the GPU work."""

    def __init__(self, split, batches, with_targets=True, depth=4):
        self.q = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._run, args=(split, batches, with_targets), daemon=True)
        self.t.start()

    def _run(self, split, batches, with_targets):
        try:
            for idx in batches:
                self.q.put(collate(split, idx, with_targets))
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
            self.q.put(e)
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--splits", nargs="+", default=["train-clean-100", "dev-clean", "dev-other", "test-clean",
                                                    "test-other"])
    args = ap.parse_args()
    vocab = load_vocab(args.vocab)
    word_index = {w: i for i, w in enumerate(vocab)}
    print(f"vocabulary {len(vocab)} words")
    for name in args.splits:
        s = Split(args.cache, name, word_index)
        keep = s.in_vocab()
        infeasible = sum(1 for i in keep if not ctc_feasible(s.n_tokens[i], s.word_ids[i]))
        print(f"{name}: {len(s)} utterances, {int(s.n_tokens.sum())} tokens ({s.n_tokens.sum() / 25 / 3600:.2f} h), "
              f"in vocabulary {len(keep)} ({int(s.n_tokens[keep].sum()) / 25 / 3600:.2f} h), "
              f"CTC-infeasible {infeasible}, max tokens {int(s.n_tokens.max())}")


if __name__ == "__main__":
    main()
