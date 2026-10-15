#!/usr/bin/env python3
"""Cache EmbeddingGemma 2 audio soft tokens for LibriSpeech splits (the reader's training input).

Input: the LibriSpeech parquet copy on the Hugging Face Hub (openslr/librispeech_asr, revision 71cacbfb), whose rows
hold the original 16 kHz mono FLAC bytes and the transcript. Per utterance: soundfile float32 ->
Gemma4AudioFeatureExtractor with no 30 s cut (max_length=None, truncation=False) -> get_audio_features in float32
(default attention = sdpa; on CUDA without TF32) -> the valid rows saved as float16 [n_tokens, 512] to
<out>/<split>/<id>.npy, and one line appended to <out>/<split>/index.tsv: id, n_samples, n_frames, n_tokens,
transcript (lower case), in parquet order. Files already listed in index.tsv are skipped, so an interrupted run
resumes.

--batch-size 1 (default) runs one utterance per encoder call, as the reader v1 cache was made. With --batch-size B
the utterances of each chunk of --chunk rows are sorted by length and run B at a time, right-padded with zero
feature rows under input_features_mask: the encoder's attention and convolutions only read to the left of a valid
frame or masked rows, so a valid token does not depend on the padding, and a batch of one is the extractor's own
input (check the batches against --batch-size 1 before use).
--workers N decodes the FLAC and computes the log-mel features in N processes.

Do not load the model with attn_implementation="eager" under transformers 5.19.0: the eager mask is a float mask
and the Gemma 4 audio attention applies masked_fill(mask.logical_not(), ...), which reads the wrong positions.

Example:
  python -I cache_soft_tokens.py --data-root hf_librispeech --model-dir embeddinggemma-2 --out cache \
      --splits dev-clean dev-other
"""
import argparse
import glob
import io
import multiprocessing as mp
import os
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

SR = 16000
SPLITS = {
    "train-clean-100": "clean/train.100",
    "train-clean-360": "clean/train.360",
    "train-other-500": "other/train.500",
    "dev-clean": "clean/validation",
    "dev-other": "other/validation",
    "test-clean": "clean/test",
    "test-other": "other/test",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def fp32_math():
    """CUDA: float32 matmuls and convolutions without TF32 (cuDNN convolutions use TF32 by default)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def n_frames_for(n_samples):
    """Valid mel frames: frame i needs samples up to 160 * i + 160."""
    return max(0, (n_samples - 161) // 160 + 1)


def n_tokens_for(n_frames):
    """ceil(ceil(n_frames / 2) / 2): two stride-2 convolutions."""
    return ((n_frames + 1) // 2 + 1) // 2


def parquet_files(data_root, split):
    files = sorted(glob.glob(os.path.join(data_root, SPLITS[split], "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files for {split} under {os.path.join(data_root, SPLITS[split])}")
    return files


def iter_rows(path):
    """Yield (id, FLAC bytes, transcript lower case) of one parquet file, in file order."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["id", "text", "audio"]):
        for row in batch.to_pylist():
            yield row["id"], row["audio"]["bytes"], row["text"].lower()


def decode(flac, uid):
    wav, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=False)
    if sr != SR or wav.ndim != 1:
        raise ValueError(f"{uid}: expected 16 kHz mono, got sr={sr} shape={wav.shape}")
    return wav


def iter_split(data_root, split):
    """Yield (id, waveform float32, transcript lower case) in parquet order."""
    for f in parquet_files(data_root, split):
        for uid, flac, text in iter_rows(f):
            yield uid, decode(flac, uid), text


def load_encoder(model_dir, device):
    """-> (feature extractor, EmbeddingGemma 2 model in float32 on device, settings dict)."""
    from transformers import EmbeddingGemma2Model, Gemma4AudioFeatureExtractor
    fe = Gemma4AudioFeatureExtractor.from_pretrained(model_dir)
    info = {"device": device, "torch": torch.__version__}
    if device == "cuda":
        info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), math=fp32_math())
    model = EmbeddingGemma2Model.from_pretrained(model_dir, dtype=torch.float32, vision_config=None)
    model = model.to(device).eval()
    impl = getattr(model.config.audio_config, "_attn_implementation", None)
    if impl == "eager":
        raise SystemExit("audio attention is eager; transformers 5.19.0 eager reads the wrong positions")
    info["audio_attention"] = impl
    return fe, model, info


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


def features(fe, wav):
    """-> the valid log-mel rows [n_frames, 128] float32 of one waveform (the extractor pads with zero rows)."""
    out = fe([np.asarray(wav, dtype=np.float32)], return_tensors="np", max_length=None, truncation=False)
    f = out["input_features"][0].astype(np.float32)
    m = out["input_features_mask"][0].astype(bool)
    n = n_frames_for(len(wav))
    if int(m.sum()) != n or not m[:n].all():
        raise AssertionError(f"extractor mask {int(m.sum())} != {n} leading frames")
    if np.any(f[n:]):
        raise AssertionError("extractor padding rows are not zero")
    return f[:n]


def extractor_frames(n_samples, pad_to_multiple_of=128):
    """Frames the extractor returns for one waveform (valid + zero rows): it pads the audio to a multiple of 128
    samples, prepends 160 zeros and unfolds windows of 321 samples every 160."""
    m = -(-n_samples // pad_to_multiple_of) * pad_to_multiple_of
    return (m + 160 - 321) // 160 + 1


@torch.no_grad()
def soft_tokens_batch(model, feats, device, n_samples):
    """feats: list of valid log-mel rows [n_frames_i, 128], n_samples: the waveform lengths -> list of soft tokens
    float32 numpy [n_tokens_i, 512].

    The batch is right-padded with zero rows to the longest extractor output of the batch, with
    input_features_mask True on the valid rows; one utterance gets exactly the extractor's input."""
    n = [x.shape[0] for x in feats]
    T = max(extractor_frames(s) for s in n_samples)
    if T < max(n):
        raise AssertionError(f"{T} frames < {max(n)} valid frames")
    x = np.zeros((len(feats), T, feats[0].shape[1]), np.float32)
    m = np.zeros((len(feats), T), bool)
    for b, f in enumerate(feats):
        x[b, :n[b]] = f
        m[b, :n[b]] = True
    res = model.get_audio_features(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device), return_dict=True)
    pooled, am = res.pooler_output, res.attention_mask
    out = []
    for b in range(len(feats)):
        s = pooled[b][am[b]].float().cpu().numpy()
        if s.shape[0] != n_tokens_for(n[b]):
            raise AssertionError(f"{s.shape[0]} tokens != {n_tokens_for(n[b])}")
        out.append(s)
    return out


_FE = None


def _worker_init(model_dir):
    global _FE
    from transformers import Gemma4AudioFeatureExtractor
    _FE = Gemma4AudioFeatureExtractor.from_pretrained(model_dir)


def _worker_features(item):
    uid, flac, text = item
    wav = decode(flac, uid)
    return uid, len(wav), features(_FE, wav), text


def feature_pool(model_dir, workers):
    """Processes for FLAC decoding + log-mel (one BLAS / OpenMP thread each). Create it before CUDA starts."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    return mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(model_dir,))


class Writer:
    """<out>/<split>/<id>.npy (float16) and index.tsv lines (the .npy is written before its index line)."""

    def __init__(self, out, split):
        self.dir = os.path.join(out, split)
        os.makedirs(self.dir, exist_ok=True)
        self.index_path = os.path.join(self.dir, "index.tsv")
        self.done = set()
        if os.path.exists(self.index_path):
            with open(self.index_path, encoding="utf-8") as f:
                self.done = {line.split("\t", 1)[0] for line in f if line.strip()}
        self.fh = open(self.index_path, "a", encoding="utf-8")

    def is_done(self, uid):
        return uid in self.done and os.path.exists(os.path.join(self.dir, f"{uid}.npy"))

    def put(self, uid, n_samples, n_frames, soft, text):
        if not np.isfinite(soft).all():
            raise RuntimeError(f"non-finite soft tokens for {uid}")
        np.save(os.path.join(self.dir, f"{uid}.npy"), soft.astype(np.float16))
        self.fh.write(f"{uid}\t{n_samples}\t{n_frames}\t{soft.shape[0]}\t{text}\n")
        self.fh.flush()
        self.done.add(uid)

    def close(self):
        self.fh.close()


def cache_file(model, fe, path, writer, device, batch_size=1, pool=None, chunk=256):
    """Cache the utterances of one parquet file that are not in writer yet. -> (utterances, audio seconds)."""
    rows = (r for r in iter_rows(path) if not writer.is_done(r[0]))
    n_new, audio_s = 0, 0.0
    if batch_size <= 1 and pool is None:
        for uid, flac, text in rows:
            wav = decode(flac, uid)
            soft, n_frames = soft_tokens(model, fe, wav, device)
            writer.put(uid, len(wav), n_frames, soft, text)
            n_new, audio_s = n_new + 1, audio_s + len(wav) / SR
        return n_new, audio_s
    if pool is not None:
        items = pool.imap(_worker_features, rows, chunksize=4)
    else:
        items = ((uid, len(w), features(fe, w), text) for uid, flac, text in rows for w in [decode(flac, uid)])
    buf = []

    def flush():
        order = sorted(range(len(buf)), key=lambda i: buf[i][2].shape[0])
        soft = [None] * len(buf)
        for a in range(0, len(order), batch_size):
            idx = order[a:a + batch_size]
            for i, s in zip(idx, soft_tokens_batch(model, [buf[i][2] for i in idx], device, [buf[i][1] for i in idx])):
                soft[i] = s
        for (uid, n_samples, f, text), s in zip(buf, soft):  # parquet order
            writer.put(uid, n_samples, f.shape[0], s, text)
        buf.clear()

    for item in items:
        buf.append(item)
        n_new, audio_s = n_new + 1, audio_s + item[1] / SR
        if len(buf) >= chunk:
            flush()
    if buf:
        flush()
    return n_new, audio_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="local copy of openslr/librispeech_asr (the parquet tree)")
    ap.add_argument("--model-dir", required=True, help="local copy of google/embeddinggemma-2 (revision 914f7f89)")
    ap.add_argument("--out", required=True, help="cache root; writes <out>/<split>/<id>.npy and index.tsv")
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other"], choices=list(SPLITS))
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--batch-size", type=int, default=1, help="utterances per encoder call (1 = one at a time)")
    ap.add_argument("--chunk", type=int, default=256, help="with --batch-size > 1: rows sorted by length together")
    ap.add_argument("--workers", type=int, default=0, help="processes for FLAC decoding and log-mel (0 = none)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new utterances per split (0 = all)")
    ap.add_argument("--every", type=int, default=500, help="progress line every N utterances")
    args = ap.parse_args()

    pool = feature_pool(args.model_dir, args.workers) if args.workers else None
    fe, model, info = load_encoder(args.model_dir, args.device)
    log(f"device={args.device} audio attention={info['audio_attention']} torch={torch.__version__}"
        + (f" cuda={info['cuda']} gpu={info['gpu']} {info['math']}" if args.device == "cuda" else "")
        + f" batch={args.batch_size} workers={args.workers}")
    for split in args.splits:
        writer = Writer(args.out, split)
        t0, n_new, audio_s, last = time.time(), 0, 0.0, 0
        if args.limit or (args.batch_size <= 1 and pool is None):
            # one utterance at a time (reader v1); --limit counts single utterances
            for uid, wav, text in iter_split(args.data_root, split):
                if writer.is_done(uid):
                    continue
                soft, n_frames = soft_tokens(model, fe, wav, args.device)
                writer.put(uid, len(wav), n_frames, soft, text)
                n_new += 1
                audio_s += len(wav) / SR
                if n_new % args.every == 0:
                    el = time.time() - t0
                    log(f"{split}: {n_new} utterances, {audio_s / 3600:.2f} h in {el:.0f} s "
                        f"= x{audio_s / max(el, 1e-9):.0f} realtime")
                if args.limit and n_new >= args.limit:
                    break
        else:
            for path in parquet_files(args.data_root, split):
                n, s = cache_file(model, fe, path, writer, args.device, args.batch_size, pool, args.chunk)
                n_new, audio_s = n_new + n, audio_s + s
                if n_new - last >= args.every:
                    last = n_new
                    el = time.time() - t0
                    log(f"{split}: {n_new} utterances, {audio_s / 3600:.2f} h in {el:.0f} s "
                        f"= x{audio_s / max(el, 1e-9):.0f} realtime")
        writer.close()
        el = time.time() - t0
        log(f"{split} done: {n_new} new utterances, {audio_s / 3600:.3f} h audio in {el:.0f} s")
    if pool is not None:
        pool.close()


if __name__ == "__main__":
    main()
