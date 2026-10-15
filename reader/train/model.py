#!/usr/bin/env python3
"""The reader and its CTC objective.

reader(soft tokens [B, T, 512]) -> v [B, T, 768]; h = v / |v|; blank logit = Linear(768 -> 1)(v).
Logits over the N + 1 classes = scale * cos(h_t, table[w]) for the N words, and the blank logit last (index N).
scale = exp(log_scale), a learned scalar that starts at 20.

Reader bodies (all output v [B, T, 768]):
  a  LayerNorm -> Linear 512->512 -> 4 pre-LN bidirectional transformer layers (d 512, 8 heads, FFN 2048, GELU,
     rotary positions, base 10000) -> LayerNorm -> Linear 512->768; `layers` sets the number of layers (v1: 4)
  b  EmbeddingGemma 2 text model cut to k layers: projection-only per-layer embeddings (PLE), layers
     0..k-1, final RMSNorm and embedding_projection 512->768, initialized from the checkpoint; the soft tokens go
     in as inputs_embeds at positions 0..T-1, attention is bidirectional over the whole utterance (no sliding
     window), padding keys are masked. Written here in plain torch; check_eg2() compares it with transformers.
  c  per-token MLP: LayerNorm -> Linear 512->1024 -> GELU -> Linear 1024->768 (no context)

CTC: the normalizer log sum_c exp(logit_c) runs over all N + 1 classes (chunked, custom backward); the CTC
recursion itself only needs the blank and the words of the utterance, so torch's CTC loss runs on those gathered
columns. torch.nn.functional.ctc_loss returns exp(lp) - posterior as the gradient, which is the gradient with
respect to softmax inputs; on a subset of the columns the exp(lp) term is wrong, so _GatheredCTC removes it.
The result equals F.ctc_loss over the full N + 1 log-softmax in value and gradient (python model.py --self-test).
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_IN, D_OUT = 512, 768


# ---------------------------------------------------------------- shared pieces

def key_mask(lengths, T):
    """[B, 1, 1, T] bool, True = a real token (attend), for scaled_dot_product_attention."""
    return (torch.arange(T, device=lengths.device)[None, :] < lengths[:, None])[:, None, None, :]


def rope(T, dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    f = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------- (a) transformer from scratch

class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ffn)
        self.ff2 = nn.Linear(ffn, d)

    def forward(self, x, mask, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class ReaderA(nn.Module):
    def __init__(self, d=512, layers=4, heads=8, ffn=2048, rope_base=10000.0):
        super().__init__()
        self.heads, self.rope_base = heads, rope_base
        self.ln_in = nn.LayerNorm(D_IN)
        self.inp = nn.Linear(D_IN, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, D_OUT)

    def forward(self, x, lengths):
        T = x.shape[1]
        mask = key_mask(lengths, T)
        cos, sin = rope(T, self.inp.out_features // self.heads, self.rope_base, x.device)
        h = self.inp(self.ln_in(x))
        for blk in self.blocks:
            h = blk(h, mask, cos, sin)
        return self.out(self.ln_out(h))


# ---------------------------------------------------------------- (b) EmbeddingGemma 2 text layers 0..k-1

class RMSNorm(nn.Module):
    """x * (mean(x^2) + eps)^-0.5 [* weight], computed in float32 (as EmbeddingGemma2RMSNorm)."""

    def __init__(self, dim, eps=1e-6, scale=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x):
        y = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.type_as(x)


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


class GemmaLayer(nn.Module):
    """One EmbeddingGemma 2 sliding-attention encoder layer (hidden 512, 4 query heads, 2 kv heads, head 256)."""

    def __init__(self, d=512, heads=4, kv_heads=2, head_dim=256, ffn=2048, d_ple=512, eps=1e-6):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.input_layernorm = RMSNorm(d, eps)
        self.q_proj = nn.Linear(d, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, d, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.v_norm = RMSNorm(head_dim, eps, scale=False)
        self.post_attention_layernorm = RMSNorm(d, eps)
        self.pre_feedforward_layernorm = RMSNorm(d, eps)
        self.gate_proj = nn.Linear(d, ffn, bias=False)
        self.up_proj = nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)
        self.post_feedforward_layernorm = RMSNorm(d, eps)
        self.per_layer_input_gate = nn.Linear(d, d_ple, bias=False)
        self.per_layer_projection = nn.Linear(d_ple, d, bias=False)
        self.post_per_layer_input_norm = RMSNorm(d, eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, ple, mask, cos, sin):
        B, T, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(h).view(B, T, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(B, T, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_norm(self.v_proj(h).view(B, T, self.kv_heads, self.head_dim)).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        a = self.o_proj(a.transpose(1, 2).reshape(B, T, -1))
        x = x + self.post_attention_layernorm(a)
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(gelu_tanh(self.gate_proj(h)) * self.up_proj(h))
        x = x + self.post_feedforward_layernorm(h)
        g = self.per_layer_projection(gelu_tanh(self.per_layer_input_gate(x)) * ple)
        x = x + self.post_per_layer_input_norm(g)
        return x * self.layer_scalar


class ReaderB(nn.Module):
    def __init__(self, k=4, d=512, d_ple=512, eps=1e-6, rope_base=10000.0, head_dim=256):
        super().__init__()
        self.k, self.d, self.d_ple, self.rope_base, self.head_dim = k, d, d_ple, rope_base, head_dim
        self.per_layer_model_projection = nn.Linear(d, k * d_ple, bias=False)
        self.per_layer_projection_norm = RMSNorm(d_ple, eps)
        self.layers = nn.ModuleList([GemmaLayer(d=d, d_ple=d_ple, eps=eps, head_dim=head_dim) for _ in range(k)])
        self.norm = RMSNorm(d, eps)
        self.embedding_projection = nn.Linear(d, D_OUT, bias=False)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        ple = self.per_layer_model_projection(x) * (self.d ** -0.5)
        ple = self.per_layer_projection_norm(ple.view(B, T, self.k, self.d_ple))
        mask = key_mask(lengths, T)
        cos, sin = rope(T, self.head_dim, self.rope_base, x.device)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, ple[:, :, i], mask, cos, sin)
        return self.embedding_projection(self.norm(h))

    @torch.no_grad()
    def load_eg2(self, safetensors_path):
        """Copy layers 0..k-1, the PLE projection rows they use, norm and embedding_projection (bf16 -> fp32)."""
        from safetensors import safe_open
        p = "language_model."
        with safe_open(safetensors_path, "pt") as f:
            get = lambda name: f.get_tensor(p + name).float()  # noqa: E731
            self.per_layer_model_projection.weight.copy_(get("ple.per_layer_model_projection.weight")[:self.k * self.d_ple])
            self.per_layer_projection_norm.weight.copy_(get("ple.per_layer_projection_norm.weight"))
            self.norm.weight.copy_(get("norm.weight"))
            self.embedding_projection.weight.copy_(get("embedding_projection.weight"))
            names = {"input_layernorm": "input_layernorm", "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
                     "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj", "q_norm": "self_attn.q_norm",
                     "k_norm": "self_attn.k_norm", "post_attention_layernorm": "post_attention_layernorm",
                     "pre_feedforward_layernorm": "pre_feedforward_layernorm", "gate_proj": "mlp.gate_proj",
                     "up_proj": "mlp.up_proj", "down_proj": "mlp.down_proj",
                     "post_feedforward_layernorm": "post_feedforward_layernorm",
                     "per_layer_input_gate": "ple_block.per_layer_input_gate",
                     "per_layer_projection": "ple_block.per_layer_projection",
                     "post_per_layer_input_norm": "ple_block.post_per_layer_input_norm"}
            for i, layer in enumerate(self.layers):
                for mine, theirs in names.items():
                    getattr(layer, mine).weight.copy_(get(f"layers.{i}.{theirs}.weight"))
                layer.layer_scalar.copy_(get(f"layers.{i}.layer_scalar"))


# ---------------------------------------------------------------- (c) per-token MLP

class ReaderC(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(D_IN), nn.Linear(D_IN, hidden), nn.GELU(), nn.Linear(hidden, D_OUT))

    def forward(self, x, lengths):
        return self.net(x)


# ---------------------------------------------------------------- reader = body + blank head + scale

class Reader(nn.Module):
    def __init__(self, arch, n_vocab, k=4, layers=4):
        super().__init__()
        self.arch = arch
        self.body = {"a": lambda: ReaderA(layers=layers), "b": lambda: ReaderB(k=k), "c": lambda: ReaderC()}[arch]()
        self.blank = nn.Linear(D_OUT, 1)
        nn.init.constant_(self.blank.bias, math.log(n_vocab))
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    @property
    def scale(self):
        return self.log_scale.exp()

    def forward(self, x, lengths):
        """-> unit vectors [B, T, 768], blank logits [B, T]."""
        v = self.body(x, lengths)
        return F.normalize(v, dim=-1), self.blank(v).squeeze(-1)


def describe(arch, k=4, layers=4):
    if arch == "a":
        return {"name": "a", "kind": "transformer encoder from scratch", "layers": layers, "d_model": 512, "heads": 8,
                "ffn": 2048, "activation": "gelu", "norm": "pre-LayerNorm", "positions": "rotary (base 10000)",
                "input": "LayerNorm -> Linear 512->512", "output": "LayerNorm -> Linear 512->768"}
    if arch == "b":
        return {"name": "b", "kind": "EmbeddingGemma 2 text model cut to layers 0..k-1, fine-tuned", "k": k,
                "from": "google/embeddinggemma-2 914f7f89 language_model (PLE projection rows 0..k*512, layers "
                        "0..k-1, norm, embedding_projection)",
                "hidden": 512, "heads": 4, "kv_heads": 2, "head_dim": 256, "ffn": 2048,
                "positions": "rotary (base 10000), 0..T-1", "attention": "bidirectional, no sliding window",
                "layer_scalar": "fixed buffers from the checkpoint"}
    return {"name": "c", "kind": "per-token MLP (no context)", "layers": "LayerNorm -> Linear 512->1024 -> GELU -> "
                                                                  "Linear 1024->768"}


# ---------------------------------------------------------------- CTC over N + 1 classes

def pad_rows(x, multiple):
    """Zero rows appended so that every [chunk, N] buffer has the same size: with variable sizes the MPS caching
    allocator keeps buffers of every size it has seen and grows without bound (126 GB in 700 steps)."""
    extra = (-x.shape[0]) % multiple
    return torch.cat([x, x.new_zeros(extra, *x.shape[1:])]) if extra else x


class _LogSumExpCos(torch.autograd.Function):
    """lse_t = log sum_w exp(scale * h_t . table_w) over all rows of the table, in fixed chunks of t.

    h: [M, D] unit rows, table: [N, D] unit rows (constant), scale: 0-dim tensor."""

    @staticmethod
    def forward(ctx, h, table, scale, chunk):
        M = h.shape[0]
        hp = pad_rows(h, chunk)
        lse = hp.new_empty(hp.shape[0])
        ecos = hp.new_empty(hp.shape[0])
        probs = []
        for a in range(0, hp.shape[0], chunk):
            logit = (hp[a:a + chunk] @ table.T).mul_(scale)
            z = torch.logsumexp(logit, dim=1)
            p = (logit - z[:, None]).exp_()
            lse[a:a + chunk] = z
            ecos[a:a + chunk] = (p * logit).sum(1) / scale
            probs.append(p)
            del logit
        ctx.probs, ctx.M = probs, M
        ctx.save_for_backward(table, scale, ecos[:M])
        return lse[:M]

    @staticmethod
    def backward(ctx, g):
        table, scale, ecos = ctx.saved_tensors
        grad_h = torch.cat([p @ table for p in ctx.probs])[:ctx.M] * (scale * g)[:, None]
        ctx.probs = None
        return grad_h, None, (g * ecos).sum(), None


class _GatheredCTC(torch.autograd.Function):
    """Per-utterance CTC negative log-likelihood on gathered log-probabilities lp [B, T, C] (blank = column 0).

    Runs F.ctc_loss on the CPU. Its gradient exp(lp) - posterior is turned into the gradient with respect to lp
    (-posterior) by removing exp(lp) on the frames t < input length of the utterances it applies to."""

    @staticmethod
    def forward(ctx, lp, targets, in_lens, tgt_lens, feasible):
        x = lp.detach().to("cpu").transpose(0, 1).contiguous().requires_grad_(True)
        with torch.enable_grad():
            nll = F.ctc_loss(x, targets, in_lens, tgt_lens, blank=0, reduction="none", zero_infinity=True)
            nll.sum().backward()
        valid = (torch.arange(x.shape[0])[:, None] < in_lens[None, :]) & feasible[None, :]
        grad = x.grad - x.detach().exp() * valid[..., None]
        ctx.save_for_backward(grad.transpose(0, 1).to(lp.device))
        return nll.to(lp.device)

    @staticmethod
    def backward(ctx, gout):
        (grad,) = ctx.saved_tensors
        return grad * gout[:, None, None], None, None, None, None


def word_ctc_loss(h, blank, lengths, words, table, scale, chunk=512):
    """Mean over the batch of nll / target length (torch CTCLoss reduction="mean"), blank = class N.

    h [B, T, D] unit vectors, blank [B, T], lengths [B] (long, on h.device), words: list of 1-D long tensors (word
    ids into table), table [N, D] unit rows on h.device. -> (loss, per-utterance nll, number of infeasible)."""
    B, T, _ = h.shape
    dev = h.device
    valid = torch.arange(T, device=dev)[None, :] < lengths[:, None]
    lse_words = _LogSumExpCos.apply(h[valid], table, scale, chunk)
    logz = torch.logaddexp(lse_words, blank[valid])
    logz_bt = h.new_zeros(B, T).masked_scatter(valid, logz)

    uniq, inv = zip(*[torch.unique(w, return_inverse=True) for w in words])
    C = max(len(u) for u in uniq)
    L = max(len(w) for w in words)
    ids = torch.zeros(B, C, dtype=torch.long)
    cls_ok = torch.zeros(B, C + 1, dtype=torch.bool)
    targets = torch.zeros(B, L, dtype=torch.long)
    tgt_lens = torch.tensor([len(w) for w in words], dtype=torch.long)
    feasible = torch.zeros(B, dtype=torch.bool)
    lengths_cpu = lengths.cpu()
    for b in range(B):
        ids[b, :len(uniq[b])] = uniq[b]
        cls_ok[b, :len(uniq[b]) + 1] = True
        targets[b, :len(words[b])] = inv[b] + 1
        w = words[b]
        rep = int((w[1:] == w[:-1]).sum()) if len(w) > 1 else 0
        feasible[b] = int(lengths_cpu[b]) >= len(w) + rep
    z = torch.bmm(h, table[ids.to(dev)].transpose(1, 2)) * scale
    lp = torch.cat([blank[..., None], z], dim=-1) - logz_bt[..., None]
    lp = lp.masked_fill(~cls_ok.to(dev)[:, None, :], -1e4)
    nll = _GatheredCTC.apply(lp, targets, lengths_cpu, tgt_lens, feasible)
    loss = (nll / tgt_lens.to(dev, nll.dtype)).mean()
    return loss, nll, int((~feasible).sum())


@torch.no_grad()
def greedy_labels(h, blank, table, scale, chunk=1024):
    """h [M, D] unit, blank [M] -> class per frame [M] (N = blank). Fixed-size chunks (see pad_rows)."""
    M = h.shape[0]
    if M == 0:
        return torch.zeros(0, dtype=torch.long, device=h.device)
    hp, bp = pad_rows(h, chunk), pad_rows(blank, chunk)
    out = []
    for a in range(0, hp.shape[0], chunk):
        v, i = (hp[a:a + chunk] @ table.T).max(dim=1)
        out.append(torch.where(bp[a:a + chunk] > v * scale, torch.full_like(i, table.shape[0]), i))
    return torch.cat(out)[:M]


def limit_mps_memory(max_gb):
    """Cap this process's MPS allocations at max_gb (the allocator frees its cache before refusing).

    torch's default high watermark is 1.7 x recommendedMaxWorkingSetSize, which lets one process push the machine
    into swap. Returns (fraction, recommended GB)."""
    if not torch.backends.mps.is_available() or not max_gb:
        return None, None
    rec = torch.mps.recommended_max_memory()
    frac = min(1.0, max_gb * 1e9 / rec)
    torch.mps.set_per_process_memory_fraction(frac)
    return frac, rec / 1e9


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cuda_matmul(mode="ieee"):
    """On CUDA: float32 matmuls and convolutions in full precision ("ieee"; torch's default for matmuls, while
    cuDNN convolutions default to TF32) or on TF32 tensor cores ("tf32"). -> the settings in effect."""
    tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return {"mode": mode, "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def collapse(labels, blank_id):
    """Greedy CTC read-out: merge repeats, drop blanks. labels: 1-D sequence of ints."""
    out, prev = [], None
    for c in labels:
        if c != prev and c != blank_id:
            out.append(c)
        prev = c
    return out


# ---------------------------------------------------------------- checks

def self_test():
    """Gathered CTC vs F.ctc_loss over the full log-softmax (float64, CPU): loss and gradients."""
    torch.manual_seed(0)
    N, D, B = 300, 16, 4
    table = F.normalize(torch.randn(N, D, dtype=torch.float64), dim=-1)
    lengths = torch.tensor([30, 25, 18, 3])
    T = int(lengths.max())
    words = [torch.tensor([5, 7, 7, 9, 5, 12]), torch.tensor([1, 2, 3, 2, 1]), torch.tensor([42, 42, 42, 8]),
             torch.tensor([10, 11, 12, 13])]  # the last one cannot fit in 3 frames
    v0 = torch.randn(B, T, D, dtype=torch.float64)
    b0 = torch.randn(B, T, dtype=torch.float64) + 3.0
    s0 = torch.tensor(5.0, dtype=torch.float64)
    results = {}
    for name in ("gathered", "full"):
        v = v0.clone().requires_grad_(True)
        bl = b0.clone().requires_grad_(True)
        s = s0.clone().requires_grad_(True)
        h = F.normalize(v, dim=-1)
        if name == "gathered":
            loss, _, n_bad = word_ctc_loss(h, bl, lengths, words, table, s, chunk=7)
        else:
            logits = torch.cat([s * h @ table.T, bl[..., None]], dim=-1)
            lp = logits.log_softmax(-1).transpose(0, 1)
            tg = torch.zeros(B, max(len(w) for w in words), dtype=torch.long)
            for i, w in enumerate(words):
                tg[i, :len(w)] = w
            loss = F.ctc_loss(lp, tg, lengths, torch.tensor([len(w) for w in words]), blank=N, reduction="mean",
                              zero_infinity=True)
        loss.backward()
        results[name] = (loss.detach(), v.grad, bl.grad, s.grad)
    g, f = results["gathered"], results["full"]
    diffs = {"loss": abs(float(g[0] - f[0])), "grad_v": float((g[1] - f[1]).abs().max()),
             "grad_blank": float((g[2] - f[2]).abs().max()), "grad_scale": abs(float(g[3] - f[3]))}
    print(f"loss gathered {float(g[0]):.10f} full {float(f[0]):.10f}; max|d| {diffs}; infeasible {n_bad}")
    ok = all(d < 1e-9 for d in diffs.values())
    print("self-test", "PASS" if ok else "FAIL")
    return ok


@torch.no_grad()
def check_eg2(model_dir, k=4, device="cpu"):
    """ReaderB (plain torch) vs transformers' EmbeddingGemma 2 text layers 0..k-1 on random soft tokens."""
    import os
    from transformers import EmbeddingGemma2Model
    hf = EmbeddingGemma2Model.from_pretrained(model_dir, dtype=torch.float32, vision_config=None).to(device).eval()
    lm = hf.language_model
    mine = ReaderB(k=k)
    mine.load_eg2(os.path.join(model_dir, "model.safetensors"))
    mine = mine.to(device).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 57, 512, device=device) * 0.05
    ple = lm.ple(x)
    pos = torch.arange(x.shape[1], device=device)[None]
    pe = lm.rotary_emb(x, pos, "sliding_attention")
    h = x
    for i in range(k):
        h = lm.layers[i](h, ple[:, :, i, :], attention_mask=None, position_embeddings=pe)
    ref = lm.embedding_projection(lm.norm(h))
    out = mine(x, torch.tensor([57, 57], device=device))
    d_full = float((out - ref).abs().max())
    # padding: utterance 1 cut to 40 tokens, alone vs in a padded batch
    alone = mine(x[1:, :40], torch.tensor([40], device=device))
    padded = mine(x, torch.tensor([57, 40], device=device))[1:, :40]
    d_pad = float((alone - padded).abs().max())
    print(f"ReaderB k={k} vs transformers: max|d| {d_full:.3e} (output max {float(ref.abs().max()):.3f}); "
          f"padded vs alone max|d| {d_pad:.3e}")
    return d_full, d_pad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="gathered CTC == full CTC (loss and gradients)")
    ap.add_argument("--check-eg2", default=None, metavar="MODEL_DIR", help="ReaderB vs transformers")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4, help="arch a: number of transformer layers")
    args = ap.parse_args()
    if args.self_test:
        if not self_test():
            raise SystemExit(1)
    if args.check_eg2:
        check_eg2(args.check_eg2, k=args.k)
    for arch in ("a", "b", "c"):
        n = sum(p.numel() for p in Reader(arch, 200000, k=args.k, layers=args.layers).parameters())
        print(f"arch {arch}{f' ({args.layers} layers)' if arch == 'a' else ''}: {n:,} parameters")


if __name__ == "__main__":
    main()
