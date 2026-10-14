#!/usr/bin/env python3
"""Write mel_filters.bin: the log-mel filter bank of the EmbeddingGemma 2 audio feature extractor, for the app.

The matrix is transformers' `audio_utils.mel_filter_bank(num_frequency_bins=257, num_mel_filters=128,
min_frequency=0, max_frequency=8000, sampling_rate=16000, norm=None, mel_scale="htk")` — what
`Gemma4AudioFeatureExtractor` builds with google/embeddinggemma-2's preprocessor_config.json (fft 512, 128 mels,
0–8 kHz). The same formula is written out below in numpy (Apache-2.0, after transformers), so only numpy is needed;
with transformers installed, the result is also checked against it.

Output: 257 × 128 float64 (the extractor's own precision: rounding the filters to float32 moves the log-mel by up to
5e-7), little-endian, row-major (frequency bin first), 263,168 bytes.

    python3 tools/make_mel_filters.py <model folder>/mel_filters.bin
"""
import argparse
import hashlib

import numpy as np

BINS, MELS, F_MIN, F_MAX, SR = 257, 128, 0.0, 8000.0, 16000


def hertz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + (freq / 700.0))


def mel_to_hertz(mels):
    return 700.0 * (np.power(10, mels / 2595.0) - 1.0)


def mel_filter_bank():
    """transformers.audio_utils.mel_filter_bank with the extractor's arguments (htk, norm None), in float64."""
    mel_freqs = np.linspace(hertz_to_mel(F_MIN), hertz_to_mel(F_MAX), MELS + 2)
    filter_freqs = mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, SR // 2, BINS)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="path of mel_filters.bin")
    args = ap.parse_args()
    filters = mel_filter_bank()
    assert filters.shape == (BINS, MELS), filters.shape
    try:
        from transformers.audio_utils import mel_filter_bank as hf_mel_filter_bank
    except ImportError:
        print("transformers not installed: the numpy formula is used unchecked")
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hf = hf_mel_filter_bank(num_frequency_bins=BINS, num_mel_filters=MELS, min_frequency=F_MIN,
                                    max_frequency=F_MAX, sampling_rate=SR, norm=None, mel_scale="htk")
        assert np.array_equal(hf, filters), float(np.abs(hf - filters).max())
        print("identical to transformers' mel_filter_bank")
    data = filters.astype("<f8").tobytes()
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"{args.out}: {filters.shape} float64, {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
