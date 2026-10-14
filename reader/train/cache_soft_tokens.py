#!/usr/bin/env python3
"""Cache EmbeddingGemma 2 audio soft tokens for LibriSpeech splits (the reader's training input).

Input: the LibriSpeech parquet copy on the Hugging Face Hub (openslr/librispeech_asr, revision 71cacbfb), whose rows
hold the original 16 kHz mono FLAC bytes and the transcript. Per utterance: soundfile float32 ->
Gemma4AudioFeatureExtractor with no 30 s cut (max_length=None, truncation=False) -> get_audio_features in float32
(default attention = sdpa), batch 1 -> the valid rows saved as float16 [n_tokens, 512] to <out>/<split>/<id>.npy,
and one line appended to <out>/<split>/index.tsv: id, n_samples, n_frames, n_tokens, transcript (lower case).
Files already listed in index.tsv are skipped, so an interrupted run resumes.

Do not load the model with attn_implementation="eager" under transformers 5.19.0: the eager mask is a float mask
and the Gemma 4 audio attention applies masked_fill(mask.logical_not(), ...), which reads the wrong positions.

Example:
  python -I cache_soft_tokens.py --data-root hf_librispeech --model-dir embeddinggemma-2 --out cache \
      --splits dev-clean dev-other
"""
import argparse
import glob
import io
import os
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

SR = 16000
SPLITS = {
    "train-clean-100": "clean/train.100",
    "dev-clean": "clean/validation",
    "dev-other": "other/validation",
    "test-clean": "clean/test",
    "test-other": "other/test",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def n_frames_for(n_samples):
    """Valid mel frames: frame i needs samples up to 160 * i + 160."""
    return max(0, (n_samples - 161) // 160 + 1)


def n_tokens_for(n_frames):
    """ceil(ceil(n_frames / 2) / 2): two stride-2 convolutions."""
    return ((n_frames + 1) // 2 + 1) // 2


def iter_split(data_root, split):
    """Yield (id, waveform float32, transcript lower case) in parquet order."""
    files = sorted(glob.glob(os.path.join(data_root, SPLITS[split], "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files for {split} under {os.path.join(data_root, SPLITS[split])}")
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["id", "text", "audio"]):
            for row in batch.to_pylist():
                wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
                if sr != SR or wav.ndim != 1:
                    raise ValueError(f"{row['id']}: expected 16 kHz mono, got sr={sr} shape={wav.shape}")
                yield row["id"], wav, row["text"].lower()


@torch.no_grad()
def soft_tokens(model, fe, wav, device):
    """-> soft tokens [n_tokens, 512] float32 numpy for one waveform (no 30 s cut)."""
    # A one-item list, as the processor passes it: a bare 1-D array fails in transformers 5.19.0.
    out = fe([np.asarray(wav, dtype=np.float32)], return_tensors="np", max_length=None, truncation=False)
    feats = torch.from_numpy(out["input_features"][0].astype(np.float32))[None].to(device)
    mask = torch.from_numpy(out["input_features_mask"][0].astype(bool))[None].to(device)
    n_frames = n_frames_for(len(wav))
    if int(mask.sum()) != n_frames:
        raise AssertionError(f"extractor mask {int(mask.sum())} != {n_frames} frames")
    res = model.get_audio_features(feats, mask, return_dict=True)
    soft = res.pooler_output[0][res.attention_mask[0]].float().cpu().numpy()
    if soft.shape[0] != n_tokens_for(n_frames):
        raise AssertionError(f"{soft.shape[0]} tokens != {n_tokens_for(n_frames)}")
    return soft, n_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="local copy of openslr/librispeech_asr (the parquet tree)")
    ap.add_argument("--model-dir", required=True, help="local copy of google/embeddinggemma-2 (revision 914f7f89)")
    ap.add_argument("--out", required=True, help="cache root; writes <out>/<split>/<id>.npy and index.tsv")
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other"], choices=list(SPLITS))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new utterances per split (0 = all)")
    ap.add_argument("--every", type=int, default=500, help="progress line every N utterances")
    args = ap.parse_args()

    from transformers import EmbeddingGemma2Model, Gemma4AudioFeatureExtractor
    fe = Gemma4AudioFeatureExtractor.from_pretrained(args.model_dir)
    model = EmbeddingGemma2Model.from_pretrained(args.model_dir, dtype=torch.float32, vision_config=None)
    model = model.to(args.device).eval()
    impl = getattr(model.config.audio_config, "_attn_implementation", None)
    if impl == "eager":
        raise SystemExit("audio attention is eager; transformers 5.19.0 eager reads the wrong positions")
    log(f"device={args.device} audio attention={impl} torch={torch.__version__}")
    for split in args.splits:
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        index_path = os.path.join(out_dir, "index.tsv")
        done = set()
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                done = {line.split("\t", 1)[0] for line in f if line.strip()}
        t0, n_new, audio_s = time.time(), 0, 0.0
        with open(index_path, "a", encoding="utf-8") as idx:
            for uid, wav, text in iter_split(args.data_root, split):
                path = os.path.join(out_dir, f"{uid}.npy")
                if uid in done and os.path.exists(path):
                    continue
                soft, n_frames = soft_tokens(model, fe, wav, args.device)
                if not np.isfinite(soft).all():
                    raise RuntimeError(f"non-finite soft tokens for {uid}")
                np.save(path, soft.astype(np.float16))
                idx.write(f"{uid}\t{len(wav)}\t{n_frames}\t{soft.shape[0]}\t{text}\n")
                idx.flush()
                n_new += 1
                audio_s += len(wav) / SR
                if n_new % args.every == 0:
                    el = time.time() - t0
                    log(f"{split}: {n_new} utterances, {audio_s / 3600:.2f} h in {el:.0f} s "
                        f"= x{audio_s / max(el, 1e-9):.0f} realtime")
                if args.limit and n_new >= args.limit:
                    break
        el = time.time() - t0
        log(f"{split} done: {n_new} new utterances, {audio_s / 3600:.3f} h audio in {el:.0f} s")


if __name__ == "__main__":
    main()
