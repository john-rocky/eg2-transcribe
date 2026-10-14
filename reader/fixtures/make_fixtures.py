#!/usr/bin/env python3
"""Build the reader test fixtures: EmbeddingGemma 2 audio soft tokens from the transformers reference.

Outputs (next to this script unless --out is given):
  test-clean-20/<id>/{audio.flac, features.npy, soft_tokens.npy, embedding_768.npy, transcript.txt}
  long-speckled-band/{audio.flac, features.npy, soft_tokens_continuous.npy, soft_tokens_w30.npy, transcript.txt}
  MANIFEST.json  (written last; its presence means the set is complete)

Reference = transformers EmbeddingGemma2Model, float32, CPU, default attention implementation (sdpa).
Do not pass attn_implementation="eager" with transformers 5.19.0: the eager mask is a float mask while the
Gemma 4 audio attention applies masked_fill(mask.logical_not(), ...), which masks the allowed keys instead.

Soft token = model.get_audio_features(input_features, input_features_mask).pooler_output at the valid rows
(the embed_audio output), float32 [n_tokens, 512], 25 tokens per second.
n_frames = floor((n_samples - 161) / 160) + 1, n_tokens = ceil(ceil(n_frames / 2) / 2).

Inputs:
  --test-clean-parquet  openslr/librispeech_asr (HF dataset) clean/test/0000.parquet
  --long-mp3            archive.org adventures_holmes/adventureholmes_08_doyle.mp3 (LibriVox)
  --gutenberg           Project Gutenberg #1661 plain text (pg1661.txt)
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F

SR = 16000
WINDOW_SAMPLES = 480_000                # 30 s, the extractor's default max_length
BOS, EOS, PAD = 2, 1, 0
BOA, AUDIO, EOA = 256000, 258881, 258883
DEFAULT_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LIBRISPEECH_CREDIT = ("LibriSpeech ASR corpus, V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "
                      "\"Librispeech: an ASR corpus based on public domain audio books\", ICASSP 2015; "
                      "https://www.openslr.org/12; CC BY 4.0")


def n_frames_for(n_samples):
    return max(0, (n_samples - 161) // 160 + 1)


def n_tokens_for(n_frames):
    return ((n_frames + 1) // 2 + 1) // 2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(fe, wav, long=False):
    """-> (features incl. right padding rows, mask). The waveform goes in as a one-item list (as the processor
    passes it); a bare 1-D array fails inside Gemma4AudioFeatureExtractor in transformers 5.19.0."""
    kw = {"return_tensors": "np"}
    if long:
        kw.update(max_length=None, truncation=False)
    out = fe([np.asarray(wav, dtype=np.float32)], **kw)
    return out["input_features"][0].astype(np.float32), out["input_features_mask"][0].astype(bool)


@torch.no_grad()
def soft_tokens(model, feats, mask):
    out = model.get_audio_features(torch.from_numpy(np.ascontiguousarray(feats))[None],
                                   torch.from_numpy(np.ascontiguousarray(mask))[None], return_dict=True)
    om = out.attention_mask[0]
    return out.pooler_output[0][om].float().numpy()


def soft_tokens_stitched(model, feats_valid, chunk_tokens, ctx_tokens):
    """Continuous soft tokens for audio too long for one call: chunks of chunk_tokens kept tokens, each run with
    ctx_tokens of features on both sides. The encoder is causal with a 180-token reach (12 layers x (11 attention
    + 4 depthwise conv)) plus 3 frames of conv lookahead, so ctx_tokens >= 181 reproduces the one-call result."""
    n_frames = feats_valid.shape[0]
    n_tokens = n_tokens_for(n_frames)
    out = np.empty((n_tokens, 512), np.float32)
    chunks = []
    for a in range(0, n_tokens, chunk_tokens):
        b = min(a + chunk_tokens, n_tokens)
        lo, hi = max(0, a - ctx_tokens), min(n_tokens, b + ctx_tokens)
        f0, f1 = 4 * lo, min(n_frames, 4 * hi)
        s = soft_tokens(model, feats_valid[f0:f1], np.ones(f1 - f0, bool))
        out[a:b] = s[a - lo:b - lo]
        chunks.append([a, b, lo, hi])
    return out, chunks


@torch.no_grad()
def sequence_embedding(model, soft):
    """<bos> <|audio> soft.. <audio|> <eos> -> text model -> mean over all positions -> L2 (= the ST pipeline)."""
    n = soft.shape[0]
    ids = torch.tensor([[BOS, BOA] + [PAD] * n + [EOA, EOS]])
    emb = model.language_model.embed_tokens(ids)
    emb[0, 2:2 + n] = torch.from_numpy(soft)
    h = model.language_model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), return_dict=True)
    return F.normalize(h.last_hidden_state[0].float().mean(0), dim=-1).numpy()


def select_utterances(parquet, n, min_s, max_s, per_speaker):
    pf = pq.ParquetFile(parquet)
    rows = []
    for batch in pf.iter_batches(batch_size=64, columns=["id", "speaker_id", "chapter_id", "text", "audio"]):
        for r in batch.to_pylist():
            rows.append({"id": r["id"], "speaker_id": int(r["speaker_id"]), "chapter_id": int(r["chapter_id"]),
                         "text": r["text"], "flac": r["audio"]["bytes"],
                         "n_samples": sf.info(io.BytesIO(r["audio"]["bytes"])).frames})
    rows.sort(key=lambda r: r["id"])
    picked, per = [], {}
    for r in rows:
        if not (min_s * SR <= r["n_samples"] <= max_s * SR):
            continue
        if per.get(r["speaker_id"], 0) >= per_speaker:
            continue
        per[r["speaker_id"]] = per.get(r["speaker_id"], 0) + 1
        picked.append(r)
        if len(picked) == n:
            break
    return picked, len(rows)


def normalize_text(text):
    """Lower case, no punctuation, digits kept, word-internal apostrophes kept (LibriSpeech style)."""
    t = text.replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^a-z0-9']+", " ", t.lower())
    return " ".join(w for w in (x.strip("'") for x in t.split()) if w)


def gutenberg_chapter(path):
    lines = open(path, encoding="utf-8-sig").read().splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r"^VIII\. THE ADVENTURE OF THE SPECKLED BAND\s*$", l))
    end = next(i for i, l in enumerate(lines) if i > start and re.match(r"^IX\. THE ADVENTURE OF THE ENGINEER", l))
    body = "\n".join(lines[start + 1:end])
    non_ascii = sorted({c for c in body if ord(c) > 127})
    return normalize_text(body), {"start_line": start + 1, "end_line": end, "non_ascii_chars": non_ascii}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/embeddinggemma-2", help="HF repo id or local snapshot dir")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--test-clean-parquet", required=True)
    ap.add_argument("--long-mp3", required=True)
    ap.add_argument("--gutenberg", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-s", type=float, default=4.0)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--per-speaker", type=int, default=2)
    ap.add_argument("--chunk-tokens", type=int, default=15000, help="kept tokens per call for the continuous run")
    ap.add_argument("--ctx-tokens", type=int, default=200, help="context tokens on each side of a chunk")
    args = ap.parse_args()

    import sentence_transformers
    import transformers
    from sentence_transformers import SentenceTransformer
    from transformers import EmbeddingGemma2Model, Gemma4AudioFeatureExtractor

    t_start = time.time()
    out_dir = args.out
    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)           # the manifest marks a complete set; it is rewritten last
    local = os.path.isdir(args.model)
    load_kw = {} if local else {"revision": args.revision}
    fe = Gemma4AudioFeatureExtractor.from_pretrained(args.model, **load_kw)
    model = EmbeddingGemma2Model.from_pretrained(args.model, dtype=torch.float32, vision_config=None,
                                                 **load_kw).eval()
    st = SentenceTransformer(args.model, device="cpu", model_kwargs={"dtype": torch.float32},
                             config_kwargs={"vision_config": None}, **load_kw)
    attn = {"audio": model.config.audio_config._attn_implementation,
            "text": model.config.text_config._attn_implementation}
    if "eager" in attn.values():
        sys.exit(f"refusing to run with eager attention: {attn}")
    if local:
        weights = os.path.join(args.model, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(args.model, "model.safetensors", revision=args.revision)
    model_sha = sha256_file(weights)

    # ---- 20 test-clean utterances
    picked, n_rows = select_utterances(args.test_clean_parquet, args.n, args.min_s, args.max_s, args.per_speaker)
    assert len(picked) == args.n, len(picked)
    entries = []
    for r in picked:
        d = os.path.join(out_dir, "test-clean-20", r["id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "audio.flac"), "wb") as f:
            f.write(r["flac"])
        wav, sr = sf.read(os.path.join(d, "audio.flac"), dtype="float32")
        assert sr == SR and wav.ndim == 1 and len(wav) == r["n_samples"]
        feats, mask = extract(fe, wav)
        n_frames = n_frames_for(len(wav))
        assert int(mask.sum()) == n_frames, (r["id"], int(mask.sum()), n_frames)
        soft = soft_tokens(model, feats, mask)
        n_tokens = n_tokens_for(n_frames)
        assert soft.shape == (n_tokens, 512), (r["id"], soft.shape, n_tokens)
        emb = st.encode([{"array": wav, "sampling_rate": SR}], batch_size=1, convert_to_numpy=True)[0]
        emb = emb.astype(np.float32)
        ours = sequence_embedding(model, soft)
        np.save(os.path.join(d, "features.npy"), feats[mask])
        np.save(os.path.join(d, "soft_tokens.npy"), soft)
        np.save(os.path.join(d, "embedding_768.npy"), emb)
        text = r["text"].lower()
        with open(os.path.join(d, "transcript.txt"), "w") as f:
            f.write(text + "\n")
        entries.append({
            "id": r["id"], "path": f"test-clean-20/{r['id']}",
            "source": f"LibriSpeech test-clean {r['id']}", "license": "CC BY 4.0",
            "speaker_id": r["speaker_id"], "chapter_id": r["chapter_id"],
            "duration_s": round(len(wav) / SR, 6), "n_samples": len(wav),
            "n_frames": n_frames, "n_tokens": n_tokens,
            "sha256": {k: sha256_file(os.path.join(d, k)) for k in
                       ("audio.flac", "features.npy", "soft_tokens.npy", "embedding_768.npy", "transcript.txt")},
            "cos_sequence_path_vs_st_encode": float(np.dot(ours, emb) / (np.linalg.norm(ours) * np.linalg.norm(emb))),
            "transcript": text,
        })
        print(f"{r['id']} {len(wav) / SR:6.2f} s tokens={n_tokens} cos={entries[-1]['cos_sequence_path_vs_st_encode']:.9f}",
              flush=True)

    # ---- long clip
    ld = os.path.join(out_dir, "long-speckled-band")
    os.makedirs(ld, exist_ok=True)
    flac = os.path.join(ld, "audio.flac")
    ffmpeg_cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", args.long_mp3,
                  "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
                  "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", "-c:a", "flac", flac]
    subprocess.run(ffmpeg_cmd, check=True)
    ffmpeg_version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    wav, sr = sf.read(flac, dtype="float32")
    assert sr == SR and wav.ndim == 1
    feats, mask = extract(fe, wav, long=True)
    n_frames = n_frames_for(len(wav))
    assert int(mask.sum()) == n_frames, (int(mask.sum()), n_frames)
    feats_valid = np.ascontiguousarray(feats[mask])
    del feats
    n_tokens = n_tokens_for(n_frames)
    t0 = time.time()
    cont, chunks = soft_tokens_stitched(model, feats_valid, args.chunk_tokens, args.ctx_tokens)
    t_cont = time.time() - t0
    # one-call check across the first stitch boundary: tokens [chunk - 3000, chunk + 3000) in a single call,
    # compared after its own 200-token warm-up (no left context in that call) and before its last token
    stitch_check = None
    if n_tokens > args.chunk_tokens + 3000:
        lo, hi = args.chunk_tokens - 3000, args.chunk_tokens + 3000
        ref = soft_tokens(model, feats_valid[4 * lo:4 * hi], np.ones(4 * (hi - lo), bool))
        x, y = cont[lo + 200:hi - 1], ref[200:-1]
        c = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        stitch_check = {"tokens_compared": [lo + 200, hi - 1], "boundary_at": args.chunk_tokens,
                        "max_abs_diff": float(np.abs(x - y).max()), "cos_min": float(c.min()),
                        "cos_mean": float(c.mean())}
        print(f"stitch check {stitch_check}", flush=True)
    t0 = time.time()
    w30 = []
    for k, a in enumerate(range(0, len(wav), WINDOW_SAMPLES)):
        seg = wav[a:a + WINDOW_SAMPLES]
        f_w, m_w = extract(fe, seg)
        s_w = soft_tokens(model, f_w, m_w)
        assert s_w.shape[0] == n_tokens_for(n_frames_for(len(seg))), (k, s_w.shape)
        w30.append(s_w)
    w30 = np.concatenate(w30, 0)
    t_w30 = time.time() - t0
    assert cont.shape == w30.shape == (n_tokens, 512), (cont.shape, w30.shape, n_tokens)
    np.save(os.path.join(ld, "features.npy"), feats_valid)
    np.save(os.path.join(ld, "soft_tokens_continuous.npy"), cont)
    np.save(os.path.join(ld, "soft_tokens_w30.npy"), w30)
    transcript, gmeta = gutenberg_chapter(args.gutenberg)
    with open(os.path.join(ld, "transcript.txt"), "w") as f:
        f.write(transcript + "\n")
    cosw = (cont * w30).sum(1) / (np.linalg.norm(cont, axis=1) * np.linalg.norm(w30, axis=1))
    long_entry = {
        "id": "long-speckled-band", "path": "long-speckled-band",
        "source": {"work": "The Adventure of the Speckled Band (Arthur Conan Doyle), chapter 8 of the LibriVox "
                           "recording 'The Adventures of Sherlock Holmes'",
                   "reader": "Kirsten Ferreri", "archive_org_identifier": "adventures_holmes",
                   "url": "https://archive.org/download/adventures_holmes/adventureholmes_08_doyle.mp3",
                   "catalog": "https://librivox.org/the-adventures-of-sherlock-holmes/",
                   "mp3_sha256": sha256_file(args.long_mp3)},
        "license": "public domain (LibriVox recording); transcript from Project Gutenberg eBook #1661 (public domain in the USA)",
        "conversion": {"ffmpeg": ffmpeg_version, "command": "ffmpeg " + " ".join(
            "<mp3>" if x == args.long_mp3 else ("audio.flac" if x == flac else x) for x in ffmpeg_cmd[1:])},
        "duration_s": round(len(wav) / SR, 6), "n_samples": len(wav), "n_frames": n_frames, "n_tokens": n_tokens,
        "sha256": {k: sha256_file(os.path.join(ld, k)) for k in
                   ("audio.flac", "features.npy", "soft_tokens_continuous.npy", "soft_tokens_w30.npy",
                    "transcript.txt")},
        "trim": None,
        "trim_note": "Full length kept. The LibriVox intro at the start and the outro at the end are not in the "
                     "Gutenberg text; locate the first and last sentences of the body to trim.",
        "transcript": {"file": "transcript.txt", "from": "Project Gutenberg #1661, section 'VIII. THE ADVENTURE OF "
                       "THE SPECKLED BAND' (body after the heading, up to 'IX.')", **gmeta,
                       "normalization": "lower case, punctuation removed, digits kept, word-internal apostrophes "
                                        "kept, curly apostrophes mapped to '",
                       "n_words": len(transcript.split()),
                       "note": "WER on this clip is indicative only (reading may differ from the printed text)"},
        "window_policy": {
            "soft_tokens_continuous.npy": {
                "what": "continuous encoding of the whole clip",
                "how": f"one get_audio_features call cannot fit in memory here (dense [T, T] attention mask, "
                       f"T = {n_tokens}); chunks of {args.chunk_tokens} kept tokens, each run with "
                       f"{args.ctx_tokens} tokens of features on both sides, stitched",
                "chunks_[keep_from, keep_to, run_from, run_to]": chunks,
                "exactness": "causal encoder, receptive field 180 tokens + 3 frames lookahead",
                "one_call_check_across_boundary": stitch_check},
            "soft_tokens_w30.npy": {
                "what": "independent 30 s windows, no overlap, tokens concatenated",
                "how": "window k = samples [480000 k, 480000 (k + 1)); each window through the extractor "
                       "(default max_length 480000) and get_audio_features on its own",
                "windows": int(np.ceil(len(wav) / WINDOW_SAMPLES)), "tokens_per_full_window": 750,
                "index": "global token g = 750 k + p, p = 0..749 within window k"},
            "continuous_vs_w30_token_cos": {"mean": float(cosw.mean()), "min": float(cosw.min())},
        },
        "timing_s": {"continuous": round(t_cont, 1), "w30": round(t_w30, 1)},
    }

    pc = json.load(open(os.path.join(args.model, "preprocessor_config.json"))) if local else fe.to_dict()
    manifest = {
        "name": "eg2-transcribe reader fixtures",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {"repo": "google/embeddinggemma-2", "revision": args.revision,
                  "model_safetensors_sha256": model_sha, "dtype": "float32", "device": "cpu",
                  "attn_implementation": attn,
                  "note": "with transformers 5.19.0 do not use attn_implementation='eager' (inverted audio "
                          "attention mask); the default (sdpa) path matches a brute-force sliding-window "
                          "attention to 5e-5"},
        "software": {"python": sys.version.split()[0], "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__,
                     "numpy": np.__version__, "ffmpeg": ffmpeg_version, "torch_threads": torch.get_num_threads()},
        "extractor": {k: pc.get(k) for k in ("feature_extractor_type", "sampling_rate", "feature_size",
                                             "frame_length", "hop_length", "fft_length", "min_frequency",
                                             "max_frequency", "mel_floor", "preemphasis", "dither",
                                             "input_scale_factor", "per_bin_mean", "per_bin_stddev",
                                             "padding_side", "padding_value")},
        "token_contract": {
            "tokens_per_second": 25, "ms_per_token": 40, "samples_per_token": 640,
            "n_frames": "floor((n_samples - 161) / 160) + 1 (= input_features_mask sum)",
            "n_tokens": "ceil(ceil(n_frames / 2) / 2) (= encoder attention_mask sum)",
            "soft_token": "get_audio_features(input_features, input_features_mask).pooler_output valid rows "
                          "(embed_audio output), float32 [n_tokens, 512]",
            "audio_sequence": "<bos>=2 <|audio>=256000 n_tokens x <|audio|>=258881 <audio|>=258883 <eos>=1",
            "embedding_768": "SentenceTransformer.encode({'array', 'sampling_rate': 16000}) float32, "
                             "= mean over the n_tokens + 4 positions of the text model output, L2-normalized",
            "waveform": "float32 in [-1, 1) as decoded by soundfile from audio.flac"},
        "files": {
            "audio.flac": "16 kHz mono 16-bit FLAC (LibriSpeech: the original file bytes)",
            "features.npy": "float32 [n_frames, 128] log-mel, the extractor rows with input_features_mask True",
            "soft_tokens.npy": "float32 [n_tokens, 512]",
            "embedding_768.npy": "float32 [768], L2-normalized",
            "transcript.txt": "lower case, no punctuation (LibriSpeech text lower-cased)"},
        "selection": {
            "split": "LibriSpeech test-clean, HF dataset openslr/librispeech_asr rev "
                     "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1, clean/test/0000.parquet",
            "parquet_sha256": sha256_file(args.test_clean_parquet), "utterances_in_split": n_rows,
            "rule": f"id string ascending; {args.min_s:g} s <= duration <= {args.max_s:g} s; at most "
                    f"{args.per_speaker} per speaker; first {args.n}"},
        "licenses": {"librispeech": LIBRISPEECH_CREDIT,
                     "librivox": "public domain; reader Kirsten Ferreri; "
                                 "https://archive.org/details/adventures_holmes",
                     "gutenberg": "Project Gutenberg eBook #1661, public domain in the USA; normalized chapter "
                                  "text only, without the Project Gutenberg header or license"},
        "entries": entries,
        "long": long_entry,
        "wall_s": round(time.time() - t_start, 1),
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, manifest_path)
    print(f"wrote {manifest_path} ({len(entries)} utterances + long {len(wav) / SR / 60:.1f} min) "
          f"in {manifest['wall_s']} s")


if __name__ == "__main__":
    main()
