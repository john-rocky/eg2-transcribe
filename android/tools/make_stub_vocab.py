#!/usr/bin/env python3
"""Writes app/src/main/assets/stub_vocab.txt, the stub reader's 1000 placeholder words: the most frequent words of
reader/fixtures/long-speckled-band/transcript.txt (Project Gutenberg eBook #1661, public domain in the USA), by count
and then alphabetically, one per line.

    python3 tools/make_stub_vocab.py ../reader/fixtures/long-speckled-band/transcript.txt app/src/main/assets/stub_vocab.txt
"""
import collections, sys


def main(src, dst, n=1000):
    words = open(src, encoding="utf-8").read().split()
    counts = collections.Counter(words)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    if len(top) < n:
        sys.exit(f"{src} has only {len(top)} distinct words")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(w for w, _ in top) + "\n")
    print(f"{dst}: {len(top)} words from {len(words)} in {src} ({len(counts)} distinct); last count {top[-1][1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
