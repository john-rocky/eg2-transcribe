#!/usr/bin/env python3
"""Train the reader: CTC over the word table, dev WER (greedy) after every epoch, early stopping, checkpoints.

Data: the cached soft tokens of --train-split (one split, or several joined with commas, read as one set;
utterances with an out-of-vocabulary word are left out), batches of --batch-size utterances bucketed by length and
shuffled. Optimizer: AdamW (weight decay on matrices only), linear warmup for --warmup steps, then cosine to zero at
--schedule-epochs; gradient clipping at --clip; float32 (on CUDA, --matmul ieee keeps the training matmuls in full
float32, tf32 uses the TF32 tensor cores; the dev WER is always decoded in full float32). The blank logit starts at
the median word log-normalizer of one median-length batch (blank probability about 0.5).

Each epoch writes <run-dir>/ckpt_epoch<N>.pt (model, optimizer, schedule) and <run-dir>/metrics.json; the model of
the best dev epoch is also saved as <run-dir>/best_model.pt. Training stops at --epochs epochs in total, after
--patience epochs without a better dev WER, when the active time (pauses by quiet_yield.py excluded) would pass
--max-hours, or at the start of an epoch when --stop-file exists. --resume continues a run from a checkpoint (same
schedule unless --schedule-epochs changes it).

Example:
  python -I train.py --arch b --cache cache --vocab ../vocab.txt --table ../word_table.npy \
      --model-dir embeddinggemma-2 --run-dir runs/b4 --epochs 1
"""
import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D  # noqa: E402
from decode import load_table, transcribe_split  # noqa: E402
from model import (Reader, _LogSumExpCos, cuda_matmul, default_device, describe, limit_mps_memory,  # noqa: E402
                   word_ctc_loss)

DEFAULT_LR = {"a": 3e-4, "b": 1e-4, "c": 3e-4}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score(refs, hyps):
    import jiwer
    o = jiwer.process_words(refs, hyps)
    n_ref = o.substitutions + o.deletions + o.hits
    return {"wer": o.wer, "substitutions": o.substitutions, "deletions": o.deletions, "insertions": o.insertions,
            "ref_words": n_ref}


class Clock:
    """Wall and active time. A gap between two steps longer than max(30 s, 20 x the median step) is counted as a
    pause (quiet_yield.py stops the process while another lane's measurement window is open)."""

    def __init__(self):
        self.t0 = self.last = time.time()
        self.paused = 0.0
        self.steps = []

    def tick(self):
        now = time.time()
        dt, self.last = now - self.last, now
        if len(self.steps) >= 20:
            med = float(np.median(self.steps[-200:]))
            if dt > max(30.0, 20 * med):
                self.paused += dt - med
                dt = med
        self.steps.append(dt)

    def resync(self):
        self.last = time.time()

    def wall(self):
        return time.time() - self.t0

    def active(self):
        return self.wall() - self.paused


def peak_gb():
    """MPS: memory allocated by the Metal driver; CUDA: memory reserved by torch's caching allocator (peak)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved() / 1e9
    return torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0


def max_rss_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, kilobytes on Linux
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


@torch.no_grad()
def evaluate(model, split, table, vocab, device):
    model.eval()
    t0 = time.time()
    hyps = transcribe_split(model, split, table, vocab, device)
    s = score(split.texts, [hyps[u] for u in split.ids])
    s["eval_s"] = round(time.time() - t0, 1)
    model.train()
    return s, hyps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["a", "b", "c"], required=True)
    ap.add_argument("--k", type=int, default=4, help="arch b: number of EmbeddingGemma 2 text layers kept")
    ap.add_argument("--layers", type=int, default=4, help="arch a: number of transformer layers")
    ap.add_argument("--cache", required=True, help="soft token cache root (cache_soft_tokens.py --out)")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--model-dir", default=None, help="EmbeddingGemma 2 checkpoint directory (arch b init)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10, help="train until this many epochs in total")
    ap.add_argument("--schedule-epochs", type=float, default=10.0, help="warmup + cosine length in epochs")
    ap.add_argument("--lr", type=float, default=None, help="peak learning rate (default: a 3e-4, b 1e-4, c 3e-4)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=512, help="frames per chunk of the N-way normalizer")
    ap.add_argument("--mps-max-gb", type=float, default=40.0, help="cap on this process's MPS allocations")
    ap.add_argument("--empty-cache-every", type=int, default=50, help="release cached MPS buffers every N steps")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--max-hours", type=float, default=0.0, help="active-time budget (0 = none)")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--stop-file", default=None, help="stop at the start of an epoch when this file exists")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--train-split", default="train-clean-100",
                    help="cached split, or several joined with commas (one epoch = all of them)")
    ap.add_argument("--dev-split", default="dev-clean")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--matmul", choices=["ieee", "tf32"], default="ieee",
                    help="CUDA only: float32 matmuls in full precision (ieee) or on TF32 tensor cores")
    ap.add_argument("--time-steps", type=int, default=0, help="only time this many steps after 3 warm-up steps")
    ap.add_argument("--limit-train", type=int, default=0, help="smoke test: use only this many training utterances")
    ap.add_argument("--limit-dev", type=int, default=0, help="smoke test: use only this many dev utterances")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = {"device": args.device, "torch": torch.__version__}
    if args.device == "mps":
        frac, rec = limit_mps_memory(args.mps_max_gb)
        log(f"MPS allocations capped at {args.mps_max_gb:g} GB (fraction {frac:.3f} of recommended {rec:.1f} GB)")
    if args.device == "cuda":
        env.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), matmul=cuda_matmul(args.matmul),
                   eval_matmul="ieee")
        log(f"CUDA {env['cuda']} on {env['gpu']}, torch {env['torch']}, float32 matmul in training: {env['matmul']}; "
            f"dev decoding in full float32 (ieee)")
    dev = torch.device(args.device)
    os.makedirs(args.run_dir, exist_ok=True)
    vocab = D.load_vocab(args.vocab)
    widx = {w: i for i, w in enumerate(vocab)}
    table = load_table(args.table, dev)
    if table.shape[0] != len(vocab):
        raise SystemExit(f"table rows {table.shape[0]} != vocabulary {len(vocab)}")

    train = D.Split(args.cache, args.train_split, widx)
    in_vocab = train.in_vocab()
    items = [i for i in in_vocab if D.ctc_feasible(train.n_tokens[i], train.word_ids[i])]
    n_infeasible = len(in_vocab) - len(items)
    if args.limit_train:
        items = items[:args.limit_train]
    log(f"{args.train_split}: {len(train)} utterances ({train.n_tokens.sum() / 25 / 3600:.2f} h); "
        f"left out {len(train) - len(in_vocab)} with an out-of-vocabulary word and "
        f"{n_infeasible} CTC-infeasible; training on {len(items)} "
        f"({train.n_tokens[items].sum() / 25 / 3600:.2f} h, {int(train.n_tokens[items].sum())} tokens)")
    split_stats = train.part_stats()
    if len(train.parts) > 1:
        for part, s in split_stats.items():
            log(f"  {part}: {s['utterances']} utterances ({s['hours']:.2f} h), left out {s['oov_left_out']} with an "
                f"out-of-vocabulary word, in vocabulary {s['in_vocab_utterances']} ({s['in_vocab_hours']:.2f} h)")
    dev_split = D.Split(args.cache, args.dev_split, widx)
    if args.limit_dev:
        dev_split.ids, dev_split.texts = dev_split.ids[:args.limit_dev], dev_split.texts[:args.limit_dev]
        dev_split.n_tokens, dev_split.word_ids = dev_split.n_tokens[:args.limit_dev], dev_split.word_ids[:args.limit_dev]
    log(f"{args.dev_split}: {len(dev_split)} utterances ({dev_split.n_tokens.sum() / 25 / 3600:.2f} h), "
        f"{len(dev_split) - len(dev_split.in_vocab())} with an out-of-vocabulary word (kept, they count as errors)")

    model = Reader(args.arch, len(vocab), k=args.k, layers=args.layers)
    if args.arch == "b" and not args.resume:
        if not args.model_dir:
            raise SystemExit("--model-dir is needed to initialize arch b")
        model.body.load_eg2(os.path.join(args.model_dir, "model.safetensors"))
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    lr = args.lr or DEFAULT_LR[args.arch]
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": args.weight_decay},
              {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps_per_epoch = math.ceil(len(items) / args.batch_size)
    total = max(args.warmup + 1, int(round(args.schedule_epochs * steps_per_epoch)))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    step, start_epoch, history = 0, 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck.get("layers", 4) != args.layers:
            raise SystemExit(f"{args.resume} has {ck.get('layers', 4)} layers, --layers is {args.layers}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, start_epoch, history = ck["step"], ck["epoch"], ck["history"]
        log(f"resumed {args.resume}: epoch {start_epoch}, step {step}, lr now {opt.param_groups[0]['lr']:.3e}")
    log(f"arch {args.arch} ({describe(args.arch, args.k, args.layers)['kind']}"
        f"{f', {args.layers} layers' if args.arch == 'a' else ''}), {n_params:,} parameters, lr {lr:g}, "
        f"{steps_per_epoch} steps/epoch, schedule {total} steps, vocabulary {len(vocab)}, device {args.device}")

    if not args.resume:
        # blank logit starts at the median word log-normalizer of one median-length batch: blank probability about 0.5
        with torch.no_grad():
            b0 = D.collate(train, D.sorted_batches(train.n_tokens, items, args.batch_size)[len(items) // 32])
            h, _ = model(b0["x"].to(dev), b0["lengths"].to(dev))
            valid = torch.arange(h.shape[1], device=dev)[None, :] < b0["lengths"].to(dev)[:, None]
            lse = _LogSumExpCos.apply(h[valid], table, model.scale, args.chunk)
            model.blank.weight.zero_()
            model.blank.bias.fill_(float(lse.median()))
        log(f"blank bias initialized to {model.blank.bias.detach().item():.3f} (log N = {math.log(len(vocab)):.3f})")

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "step": step, "epoch": epoch, "history": history, "args": vars(args), "arch": args.arch,
                    "k": args.k, "layers": args.layers, "n_vocab": len(vocab)}, path)

    def write_metrics(extra=None):
        best = min(history, key=lambda h: h["dev"]["wer"]) if history else None
        m = {"arch": describe(args.arch, args.k, args.layers), "params": n_params, "lr": lr,
             "batch_size": args.batch_size,
             "warmup": args.warmup, "weight_decay": args.weight_decay, "clip": args.clip,
             "schedule_steps": total, "steps_per_epoch": steps_per_epoch, "train_utterances": len(items),
             "train_tokens": int(train.n_tokens[items].sum()), "vocab": len(vocab), "dev_split": args.dev_split,
             "train_split": args.train_split, "train_splits": split_stats, "ctc_infeasible_left_out": n_infeasible,
             "env": env, "history": history, "best_epoch": best["epoch"] if best else None,
             "best_dev_wer": best["dev"]["wer"] if best else None}
        m.update(extra or {})
        with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1)

    clock = Clock()
    best_wer = min((h["dev"]["wer"] for h in history), default=float("inf"))
    best_i = min(range(len(history)), key=lambda i: history[i]["dev"]["wer"]) if history else -1
    bad = len(history) - 1 - best_i if history else 0
    if history and bad == 0 and not args.time_steps:
        # the resumed checkpoint is the best so far: keep it as this run's best model too
        torch.save(model.state_dict(), os.path.join(args.run_dir, "best_model.pt"))
    stop_reason = f"reached {args.epochs} epochs"
    model.train()
    for epoch in range(start_epoch, args.epochs):
        if args.stop_file and os.path.exists(args.stop_file) and not args.time_steps:
            stop_reason = f"stop file {os.path.basename(args.stop_file)} before epoch {epoch + 1}"
            break
        if args.max_hours and history and not args.time_steps:
            # budget over all epochs of the run, resumed ones included: training (pauses excluded) + dev eval
            costs = [h["active_s"] + h["dev"]["eval_s"] for h in history]
            if sum(costs) + float(np.mean(costs)) > args.max_hours * 3600:
                stop_reason = (f"active-time budget: {sum(costs) / 3600:.2f} h used over {len(history)} epochs, one "
                               f"more (~{np.mean(costs) / 3600:.2f} h) would pass {args.max_hours} h")
                break
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        batches = D.bucket_batches(train.n_tokens, items, args.batch_size, rng)
        t_ep, paused_ep, losses, window, tok_window = time.time(), clock.paused, [], [], 0
        t_win, peak, n_bad_total = time.time(), 0.0, 0
        timed = []
        for bi, batch in enumerate(D.Prefetcher(train, batches)):
            t_step = time.time()
            x = batch["x"].to(dev)
            lengths = batch["lengths"].to(dev)
            h, blank = model(x, lengths)
            loss, _, n_bad = word_ctc_loss(h, blank, lengths, batch["words"], table, model.scale, args.chunk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            step += 1
            lv = loss.item()
            if not math.isfinite(lv):
                save(os.path.join(args.run_dir, f"nonfinite_step{step}.pt"), epoch)
                raise SystemExit(f"non-finite loss {lv} at step {step} (epoch {epoch + 1}); lr {sched.get_last_lr()[0]:.3e}")
            losses.append(lv)
            window.append(lv)
            tok_window += int(batch["lengths"].sum())
            n_bad_total += n_bad
            clock.tick()
            peak = max(peak, peak_gb())
            if args.time_steps:
                if bi >= 3:
                    timed.append((time.time() - t_step, int(batch["lengths"].sum())))
                if len(timed) >= args.time_steps:
                    sec = np.array([t for t, _ in timed])
                    tok = np.array([n for _, n in timed])
                    per_tok = sec.sum() / tok.sum()
                    est_med = float(np.median(sec)) * steps_per_epoch
                    est_tok = per_tok * float(train.n_tokens[items].sum())
                    res = {"arch": args.arch, "k": args.k, "layers": args.layers, "env": env,
                           "train_split": args.train_split, "vocab": len(vocab), "batch_size": args.batch_size,
                           "steps_timed": len(timed), "median_step_s": float(np.median(sec)),
                           "step_s": sec.round(4).tolist(), "batch_tokens": tok.tolist(),
                           "s_per_1k_tokens": per_tok * 1000, "steps_per_epoch": steps_per_epoch,
                           "epoch_estimate_min_median": est_med / 60, "epoch_estimate_min_tokens": est_tok / 60,
                           "peak_driver_gb": peak, "params": n_params}
                    with open(os.path.join(args.run_dir, "timing.json"), "w", encoding="utf-8") as f:
                        json.dump(res, f, indent=1)
                    log(f"timing arch {args.arch}: median {np.median(sec):.3f} s/step over {len(timed)} steps "
                        f"(batch tokens median {int(np.median(tok))}), {per_tok * 1000:.3f} s per 1k tokens -> "
                        f"epoch ~{est_med / 60:.1f} min (median x {steps_per_epoch} steps) / "
                        f"~{est_tok / 60:.1f} min (per token x {int(train.n_tokens[items].sum())} tokens); "
                        f"peak driver memory {peak:.1f} GB")
                    log("DONE timing")
                    return
            if args.device == "mps" and args.empty_cache_every and step % args.empty_cache_every == 0:
                torch.mps.empty_cache()
            if step % args.log_every == 0:
                el = time.time() - t_win
                log(f"step {step} epoch {epoch + 1} batch {bi + 1}/{len(batches)} loss {np.mean(window):.4f} "
                    f"lr {sched.get_last_lr()[0]:.3e} {len(window) / el:.2f} step/s {tok_window / el:.0f} tok/s "
                    f"scale {model.scale.detach().item():.2f} blank_bias {model.blank.bias.detach().item():.2f} "
                    f"wall {clock.wall() / 60:.1f} min active {clock.active() / 60:.1f} min "
                    f"paused {clock.paused / 60:.1f} min mem {peak_gb():.1f} GB peak {peak:.1f} GB")
                window, tok_window, t_win = [], 0, time.time()
        ep_wall = time.time() - t_ep
        ep_paused = clock.paused - paused_ep
        ck = os.path.join(args.run_dir, f"ckpt_epoch{epoch + 1}.pt")
        save(ck, epoch + 1)  # before the dev pass; saved again below with the dev result in the history
        if args.device == "cuda":
            cuda_matmul("ieee")  # the dev WER is decoded in full float32 whatever --matmul trains with
        dev_score, hyps = evaluate(model, dev_split, table, vocab, dev)
        if args.device == "cuda":
            cuda_matmul(args.matmul)
        if args.device == "mps":
            torch.mps.empty_cache()
        clock.resync()
        rec = {"epoch": epoch + 1, "step": step, "train_loss_mean": float(np.mean(losses)),
               "train_loss_last100": float(np.mean(losses[-100:])), "dev": dev_score,
               "wall_s": round(ep_wall, 1), "paused_s": round(ep_paused, 1), "active_s": round(ep_wall - ep_paused, 1),
               "peak_driver_gb": round(peak, 2), "max_rss_gb": round(max_rss_gb(), 2),
               "lr_end": sched.get_last_lr()[0], "scale": model.scale.detach().item(), "ctc_infeasible": n_bad_total}
        history.append(rec)
        log(f"epoch {epoch + 1} done: train loss {rec['train_loss_mean']:.4f} (last 100 {rec['train_loss_last100']:.4f}), "
            f"{args.dev_split} WER {100 * dev_score['wer']:.2f} % (S {dev_score['substitutions']} D {dev_score['deletions']} "
            f"I {dev_score['insertions']} / {dev_score['ref_words']}), wall {ep_wall / 60:.1f} min, paused "
            f"{ep_paused / 60:.1f} min, eval {dev_score['eval_s']:.0f} s, peak {peak:.1f} GB, scale {rec['scale']:.2f}")
        save(ck, epoch + 1)
        with open(os.path.join(args.run_dir, f"hyp_{args.dev_split}_epoch{epoch + 1}.tsv"), "w", encoding="utf-8") as f:
            for u, ref in zip(dev_split.ids, dev_split.texts):
                f.write(f"{u}\t{ref}\t{hyps[u]}\n")
        if dev_score["wer"] < best_wer:
            best_wer, bad = dev_score["wer"], 0
            torch.save(model.state_dict(), os.path.join(args.run_dir, "best_model.pt"))
            log(f"new best {args.dev_split} WER {100 * best_wer:.2f} % at epoch {epoch + 1} -> best_model.pt")
        else:
            bad += 1
        write_metrics()
        if bad >= args.patience:
            stop_reason = f"early stop: no better {args.dev_split} WER for {bad} epochs"
            break
    write_metrics({"stop_reason": stop_reason, "wall_s_total": round(clock.wall(), 1),
                   "paused_s_total": round(clock.paused, 1)})
    log(f"stop: {stop_reason}; best {args.dev_split} WER {100 * best_wer:.2f} %; wall {clock.wall() / 60:.1f} min, "
        f"paused {clock.paused / 60:.1f} min")
    log("DONE")


if __name__ == "__main__":
    main()
