# -*- coding: utf-8 -*-
"""一次性调试：为什么自算的 attention 与模型给出的差 1e-2。

判定路径：
  ① 自己算 RoPE 的 cos/sin，与模型 rotary_emb 的输出比 → 定位是不是 RoPE。
  ② 比较 s_ours 与 log(A_model)：若每行只差一个常数，则 softmax 应当一致；
     若不一致，差异必然来自 s 本身，而不是归一化。
  ③ 逐位置看差异是否集中在被 mask 的位置或某几个 token（位置指纹）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pat_q1_unbind import PERSONS, SYS

MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
TMPL = '{A}提到了{B}的{W}。'


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation='eager').to(dev).eval()
    cfg = model.config

    text = TMPL.format(A=PERSONS[0], B=PERSONS[3], W='苹果')
    full = tok.apply_chat_template(
        [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': text}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(full, add_special_tokens=False)['input_ids']
    t = torch.tensor([ids], device=dev)

    seen = {}

    def pre(mod, args, kwargs):
        if 'pe' not in seen:
            seen['pe'] = kwargs.get('position_embeddings') or (args[1] if len(args) > 1 else None)
            seen['hs'] = args[0] if args else kwargs.get('hidden_states')

    h = model.model.layers[0].self_attn.register_forward_pre_hook(pre, with_kwargs=True)
    cap = {}

    def hook(mod, inp, out):
        cap['q'] = mod.q_norm(mod.q_proj(inp[0])).transpose(1, 2).detach() if False else None
    model.model.layers[0].self_attn.q_norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__('q', o.detach().transpose(1, 2)))
    model.model.layers[0].self_attn.k_norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__('k', o.detach().transpose(1, 2)))

    with torch.no_grad():
        out = model(input_ids=t, use_cache=False, output_attentions=True)
    h.remove()

    pe = seen['pe']
    cos_m, sin_m = pe
    T = t.shape[1]
    print('T=%d  cos 形状=%s dtype=%s' % (T, tuple(cos_m.shape), cos_m.dtype))

    # ① 自查 RoPE
    theta = getattr(cfg, 'rope_theta', None) or 1000000.0
    hd = getattr(cfg, 'head_dim', 128)
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    pos = torch.arange(T, device=dev).float()
    f = torch.outer(pos, inv)
    cos_s = torch.cat([f.cos(), f.cos()], dim=-1)
    sin_s = torch.cat([f.sin(), f.sin()], dim=-1)
    print('① cos 自查 max|Δ|=%.3e（theta=%.0f）' % (
        float((cos_s - cos_m.float()).abs().max()), theta))

    # ② s 与 log A 的关系
    q, k = cap['q'], cap['k']
    qr, kr = apply_rotary_pos_emb(q, k, cos_m, sin_m)
    kr = kr.repeat_interleave(cfg.num_attention_heads // cfg.num_key_value_heads, dim=1)
    scale = model.model.layers[0].self_attn.scaling
    s = (torch.einsum('bhtd,bhsd->bhts', qr, kr) * scale).float()

    A = out.attentions[0].float()
    lg = torch.log(A[0, :, -1].clamp_min(1e-30))
    d = s[0, :, -1] - lg
    dmax = d.max(dim=-1).values
    resid = (d - dmax[:, None]).abs()
    print('② 末位：s−logA 每行最大值范围 [%.3f, %.3f]' % (float(dmax.min()), float(dmax.max())))
    print('   去行常数后残差 max=%.3e（若接近 0 则 s 正确、问题在别处）' % float(resid.max()))

    # ③ 差异的位置指纹
    w = torch.softmax(s.masked_fill(
        torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)[None, None],
        torch.finfo(torch.float32).min), dim=-1)
    diff = (w[0, :, -1] - A[0, :, -1]).abs().max(dim=0).values
    top = torch.topk(diff, k=6)
    print('③ 逐位置差异 top6（位置, Δ）：',
          [(int(i), round(float(v), 4)) for v, i in zip(top.values, top.indices)])
    print('   位置 0..8 的 Δ：', [round(float(x), 4) for x in diff[:9]])
    print('   k 与 q 的 dtype: %s %s' % (k.dtype, q.dtype))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
