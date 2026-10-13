#!/usr/bin/env python3
"""The stub reader's word table and blank vector, the same float32 values the app computes (`Stub` in
app/src/main/kotlin/io/github/johnrocky/eg2transcribe/Reader.kt): SplitMix64 from seed 20261013, the top 24 bits of
each output as bits / 2^23 - 1 (exact in float32), the table = the first 1000 x 512 values row by row, the blank
vector = the next 512; rows L2-normalized in float64 with left-to-right sums (np.cumsum, as the Kotlin loops add)
and stored as float32. Rows follow app/src/main/assets/stub_vocab.txt.

    python3 tools/stub_table.py            # prints the first values and the sha256 of both arrays
"""
import hashlib

import numpy as np

SEED = 20261013
WORDS = 1000
DIM = 512
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB


def uniform(count, skip=0, seed=SEED):
    k = np.arange(skip + 1, skip + count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + k * np.uint64(GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(M1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(M2)
        z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(40)).astype(np.float64) / 8388608.0 - 1.0).astype(np.float32)


def normalize_rows(raw):
    w = np.asarray(raw, np.float32).astype(np.float64)
    n = np.sqrt(np.cumsum(w * w, axis=-1)[..., -1:])
    return (w / np.where(n == 0, 1.0, n)).astype(np.float32)


def table():
    return normalize_rows(uniform(WORDS * DIM).reshape(WORDS, DIM))


def blank():
    return normalize_rows(uniform(DIM, skip=WORDS * DIM)[None])[0]


if __name__ == "__main__":
    u = uniform(4)
    print("uniform[0:4] =", [float(x) for x in u], "bits", [hex(np.float32(x).view(np.uint32)) for x in u])
    t, b = table(), blank()
    print("table[0][0:3] =", [float(x) for x in t[0, :3]], "table[999][511] =", float(t[999, 511]))
    print("blank[0:3] =", [float(x) for x in b[:3]])
    print("sha256 table", hashlib.sha256(t.astype("<f4").tobytes()).hexdigest(), "blank", hashlib.sha256(b.astype("<f4").tobytes()).hexdigest())
