#!/usr/bin/env python3
"""Greedy decoding with a trained reader: soft tokens [T, 512] -> words.

Per frame: argmax over the N words (scale * cos(h_t, table[w])) and the blank logit; merge repeats; drop blanks.
No language model.

Examples:
  python -I decode.py --reader-dir .. soft_tokens.npy            # reader.pt, reader_config.json, vocab, table
  python -I decode.py --reader-dir .. --fixtures ../fixtures/test-clean-20
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Reader, collapse, greedy_labels, limit_mps_memory  # noqa: E402


def load_table(path, device):
    t = torch.from_numpy(np.load(path).astype(np.float32)).to(device)
    return F.normalize(t, dim=-1)


def load_reader_dir(reader_dir, device):
    """-> (model, table, vocab, config) from reader.pt, reader_config.json, vocab.txt, word_table.npy."""
    with open(os.path.join(reader_dir, "reader_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(reader_dir, cfg["vocab"]), encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f if line.strip()]
    arch = cfg["architecture"]
    model = Reader(arch["name"], len(vocab), k=arch.get("k", 4))
    model.load_state_dict(torch.load(os.path.join(reader_dir, "reader.pt"), map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    table = load_table(os.path.join(reader_dir, cfg["table"]["file"]), device)
    return model, table, vocab, cfg


@torch.no_grad()
def transcribe_batch(model, xs, table, vocab, device, pad_to=None):
    """xs: list of float32 numpy [T_i, 512] -> list of transcripts.

    pad_to: pad the time axis to a multiple of this many tokens (padding is masked). On MPS every new input shape
    keeps some Metal memory outside torch's allocator until the process ends, so long decoding jobs use few shapes.
    """
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    T = int(lengths.max())
    if pad_to:
        T = -(-T // pad_to) * pad_to
    x = torch.zeros(len(xs), T, xs[0].shape[1], dtype=torch.float32)
    for b, t in enumerate(xs):
        x[b, :t.shape[0]] = torch.from_numpy(np.asarray(t, dtype=np.float32))
    x, lengths = x.to(device), lengths.to(device)
    h, blank = model(x, lengths)
    valid = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
    labels = greedy_labels(h[valid], blank[valid], table, model.scale).cpu().numpy()
    out, a = [], 0
    for n in lengths.cpu().tolist():
        words = collapse(labels[a:a + n].tolist(), len(vocab))
        out.append(" ".join(vocab[i] for i in words))
        a += n
    return out


@torch.no_grad()
def transcribe_split(model, split, table, vocab, device, batch_size=32, items=None, pad_to=None):
    """-> {utterance id: transcript} for a data.Split (all utterances unless `items`)."""
    items = list(range(len(split))) if items is None else items
    order = sorted(items, key=lambda i: split.n_tokens[i])
    hyps = {}
    for a in range(0, len(order), batch_size):
        idx = order[a:a + batch_size]
        texts = transcribe_batch(model, [split.tokens(i) for i in idx], table, vocab, device, pad_to)
        for i, t in zip(idx, texts):
            hyps[split.ids[i]] = t
    return hyps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader-dir", required=True, help="directory with reader.pt, reader_config.json, vocab, table")
    ap.add_argument("--fixtures", default=None, help="directory searched for */soft_tokens.npy")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("npy", nargs="*", help="soft token files, float [T, 512]")
    args = ap.parse_args()
    if args.device == "mps":
        limit_mps_memory(40.0)
    model, table, vocab, _ = load_reader_dir(args.reader_dir, args.device)
    paths = list(args.npy)
    if args.fixtures:
        paths += sorted(glob.glob(os.path.join(args.fixtures, "**", "soft_tokens.npy"), recursive=True))
    for p in paths:
        text = transcribe_batch(model, [np.load(p).astype(np.float32)], table, vocab, args.device)[0]
        print(f"{p}\t{text}")


if __name__ == "__main__":
    main()
