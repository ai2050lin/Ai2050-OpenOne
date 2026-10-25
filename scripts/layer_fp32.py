# -*- coding: utf-8 -*-
"""单层干预的 fp32 重算工具（L30-v2 起的口径）：bf16 前向第 0–(layer−1) 层并缓存残差输入，
其后各层、最终 RMSNorm 与 lm_head 用 fp32 从缓存重算。供 L30-v3 等脚本复用。"""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import numpy as np

import run_layer_attribution as V1


class Corpus:
    def __init__(self):
        self.man = json.loads((V1.RUN / 'manifest.json').read_text(encoding='utf-8'))
        cdir = V1.NATIVE / '_census' / 'corpus' / self.man['corpus_fp']
        cman = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
        self.sep, self.seq, self.skip = int(cman['separator_id']), int(self.man['seq_len']), int(self.man['skip_head'])
        self.streams = [np.load(cdir / s['file'])['tokens'] for s in cman['sources']]
        self.n_win = [len(t) // self.seq for t in self.streams]

    def windows(self, rem, count, mod=47):
        return V1.pick_windows(self.n_win, mod, rem, count)

    def ids(self, ws):
        return np.stack([self.streams[s][w * self.seq:(w + 1) * self.seq] for s, w in ws]).astype(np.int64)

    def mask(self, ids):
        """(B, T−1) 布尔：位置 t 计入损失（预测 t+1）。"""
        pos = np.arange(self.seq)[None, :]
        nxt = np.roll(ids, -1, 1)
        return ((pos >= self.skip) & (pos <= self.seq - 2) & (ids != self.sep) & (nxt != self.sep))[:, :-1]


class Fp32Stack:
    """持有 fp32 的第 layer..末层、norm、lm_head；down_proj 前钩子按 self.mode 改写第 layer 层的 MLP 中间激活。"""

    def __init__(self, layer: int, dev: str = 'cuda'):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch, self.dev, self.layer = torch, dev, layer
        self.model = AutoModelForCausalLM.from_pretrained(str(V1.MODEL), dtype=torch.bfloat16, device_map=dev).eval()
        self.model.requires_grad_(False)
        self.pos_ids = torch.arange(V1_SEQ, device=dev)[None]
        self.mode = None
        self.cap = None

    def cache_residuals(self, ids_np, bs=8):
        torch = self.torch
        box = {}

        class _Stop(Exception):
            pass

        def grab(mod, args, kwargs):
            box['h'] = (args[0] if args else kwargs['hidden_states']).detach()
            raise _Stop()
        hk = self.model.model.layers[self.layer].register_forward_pre_hook(grab, with_kwargs=True)
        out = []
        with torch.inference_mode():
            for i in range(0, len(ids_np), bs):
                try:
                    self.model(input_ids=torch.tensor(ids_np[i:i + bs], device=self.dev))
                except _Stop:
                    pass
                out.append(box['h'].cpu())
        hk.remove()
        return torch.cat(out)

    def full_bf16_loss(self, ids_np, m_np, bs=8):
        torch = self.torch
        s = k = 0.0
        with torch.inference_mode():
            for i in range(0, len(ids_np), bs):
                ids = torch.tensor(ids_np[i:i + bs], device=self.dev)
                tl, _ = token_losses(torch, self.model(input_ids=ids).logits, ids)
                m = torch.tensor(m_np[i:i + bs], device=self.dev)
                s += float((tl * m).sum()); k += float(m.sum())
        return s / k

    def manual_loss(self, res, ids_np, m_np, mods, dtype, bs=4):
        torch = self.torch
        s = k = 0.0
        with torch.inference_mode():
            for i in range(0, len(ids_np), bs):
                ids = torch.tensor(ids_np[i:i + bs], device=self.dev)
                tl, _ = token_losses(torch, self.run_from(res[i:i + bs], mods, dtype), ids)
                m = torch.tensor(m_np[i:i + bs], device=self.dev)
                s += float((tl * m).sum()); k += float(m.sum())
        return s / k

    def bf16_mods(self):
        return (list(self.model.model.layers[self.layer:]), self.model.model.norm, self.model.lm_head)

    def to_fp32(self):
        """复制 fp32 副本并释放 bf16 模型；之后 self.mods 为 fp32 栈。"""
        torch = self.torch
        m = self.model
        L32 = [copy.deepcopy(L).float() for L in m.model.layers[self.layer:]]
        norm32 = copy.deepcopy(m.model.norm).float()
        head32 = torch.nn.Linear(m.lm_head.in_features, m.lm_head.out_features, bias=False, device=self.dev, dtype=torch.float32)
        head32.weight.data.copy_(m.lm_head.weight.data.float())
        self.rotary = copy.deepcopy(m.model.rotary_emb)
        self.model = None
        del m
        gc.collect(); torch.cuda.empty_cache()
        for mod in L32 + [norm32, head32]:
            mod.requires_grad_(False)
        self.mods = (L32, norm32, head32)
        L32[0].mlp.down_proj.register_forward_pre_hook(self._pre_down)

    def _pre_down(self, mod, args):
        A = args[0]
        m = self.mode
        if m is None:
            return None
        kind = m[0]
        if kind == 'capture':
            self.cap = A[..., m[1]].detach().clone()
            return None
        y = A.clone()
        if kind == 'clone_same':
            y[..., m[1]] = A[..., m[1]]
        elif kind == 'masked_set':                 # m = (kind, unit, mask(B,T), value)
            y[..., m[1]] = self.torch.where(m[2], m[3], A[..., m[1]])
        return (y,)

    def run_from(self, res, mods, dtype):
        torch = self.torch
        h = res.to(self.dev, dtype)
        rot = self.rotary if hasattr(self, 'rotary') else self.model.model.rotary_emb
        pe = rot(h, self.pos_ids)
        for L in mods[0]:
            h = L(h, attention_mask=None, position_embeddings=pe, position_ids=self.pos_ids)
        return mods[2](mods[1](h))


V1_SEQ = 512


def token_losses(torch, logits, ids):
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    return -lp.gather(-1, ids[:, 1:, None])[..., 0], lp


def q_from_hist(hist_row: np.ndarray, q: float) -> float:
    """普查直方图（65 箱 × 4 符号区）上的分位数：累计比例首次 ≥ q 的箱的下边界。"""
    from shared.dist_protocol import census as C
    counts = hist_row.sum(1).astype(np.float64)
    cum = np.cumsum(counts) / counts.sum()
    b = int(np.searchsorted(cum, q))
    edges = C.bin_edges()
    if b > C.N_SIDE:
        return float(edges[b - C.N_SIDE - 1])
    if b == C.N_SIDE:
        return -C.LO
    return float(-edges[C.N_SIDE - b])


def sign_flip_p(x: np.ndarray, n_perm: int, rng) -> float:
    """均值的单侧符号翻转置换检验：p = (#{置换均值 ≥ 观测均值} + 1) / (n_perm + 1)。"""
    x = np.asarray(x, dtype=np.float64)
    obs = x.mean()
    cnt = 0
    for i in range(0, n_perm, 500):
        k = min(500, n_perm - i)
        s = rng.integers(0, 2, size=(k, len(x)), dtype=np.int8) * 2 - 1
        cnt += int(((s * x[None, :]).mean(1) >= obs).sum())
    return (cnt + 1) / (n_perm + 1)
