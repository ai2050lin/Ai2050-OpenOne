# -*- coding: utf-8 -*-
"""Q3-T 结构探针：在写正式脚本前，把三件必须先确认的事钉死。

为什么必须先跑这个（而不是直接写 T1/T2）
----------------------------------------
1. key 是 GQA 的：32 个 query 头共享 8 套 K/V。预注册写的「36 × 32 头」对 key 侧是错的。
   如果按 32 算，tensor 形状会对不上，或者更糟——自动 broadcast 出一个假的 32 头结果。
2. 要算 s = q·k，q/k 必须是 **加了 RoPE 之后** 的那一份。hook 在 k_norm 拿到的是 raw，
   位置不同就不可比（T1 要跨上下文比较 key，不处理 RoPE 全是伪信号）。
3. 自己的 RoPE 实现对不对，不能靠"看起来像"，只能靠一个数字闸：用重算的 logits 做 softmax，
   与模型 output_attentions 给出的权重比，最大偏差必须极小。

本探针只做测量，不产出实验结论。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance
from pat_q1_unbind import PERSONS, SYS

MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
TMPL = '{A}提到了{B}的{W}。'
NOUNS = ['苹果', '香蕉', '橘子', '桃子', '面包', '蛋糕', '杯子', '碗', '毛巾', '雨伞',
         '铅笔', '书包', '钥匙', '手表', '眼镜', '帽子', '围巾', '手套', '鞋子', '镜子',
         '想法', '计划', '建议', '问题', '答案', '理由', '办法', '习惯', '故事', '消息',
         '照片', '地图', '票', '日记', '礼物', '玩具', '篮球', '吉他', '花瓶', '台灯']


def render(tok, user_text):
    return tok.apply_chat_template(
        [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_text}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation='eager',
        trust_remote_code=True).to(dev).eval()
    cfg = model.config
    print('L=%d  q_heads=%d  kv_heads=%d  head_dim=%d' % (
        cfg.num_hidden_layers, cfg.num_attention_heads,
        cfg.num_key_value_heads, getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)))

    # ---- 1) 单 token 名词筛选 + 位置恒定检验 ----
    def w_span(text, w):
        full = render(tok, text)
        base = full.find(text)
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        c0 = text.index(w) + base
        sel = [i for i, (a, b) in enumerate(enc['offset_mapping'])
               if a < c0 + len(w) and b > c0]
        return tuple(sel), len(enc['input_ids']), enc['input_ids']

    single = []
    for w in NOUNS:
        sel, _, _ = w_span(TMPL.format(A=PERSONS[0], B=PERSONS[3], W=w), w)
        if len(sel) == 1:
            single.append(w)
    print('单 token 名词 %d / %d' % (len(single), len(NOUNS)))

    pos_set, len_set = {}, set()
    for a in range(3):
        for b in range(3):
            for w in single[:8]:
                sel, n, _ = w_span(TMPL.format(A=PERSONS[a], B=PERSONS[b + 4], W=w), w)
                pos_set[w] = pos_set.get(w, set()) | {sel}
                len_set.add(n)
    uniform = all(len(v) == 1 for v in pos_set.values())
    print('位置恒定=%s  序列长集合=%s' % (uniform, sorted(len_set)))
    for w in single[:3]:
        print('  %s -> %s' % (w, sorted(pos_set[w])))

    # ---- 2) hook 拿 q/k + rotary，复现注意力 ----
    cap, rot = {}, {}

    def mk(li, name):
        def hook(mod, inp, out):
            cap.setdefault(li, {})[name] = out.detach()
        return hook

    def mk2(store, li, name):
        def hook(mod, inp, out):
            store.setdefault(li, {})[name] = out.detach()
        return hook

    for li in range(cfg.num_hidden_layers):
        att = model.model.layers[li].self_attn
        for nm in ('q_norm', 'k_norm'):
            if hasattr(att, nm):
                getattr(att, nm).register_forward_hook(mk(li, nm))

    def rot_hook(mod, inp, out):
        rot['cos'], rot['sin'] = out

    model.model.rotary_emb.register_forward_hook(rot_hook)

    text = TMPL.format(A=PERSONS[0], B=PERSONS[3], W=single[0])
    full = render(tok, text)
    ids = tok(full, add_special_tokens=False)['input_ids']
    t = torch.tensor([ids], device=dev)
    with torch.no_grad():
        out = model(input_ids=t, use_cache=False, output_attentions=True)

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


    scale = getattr(model.model.layers[0].self_attn, 'scaling', 128 ** -0.5)
    T = t.shape[1]
    causal = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), diagonal=1)

    errs = []
    for li in (0, 17, 35):
        q = cap[li]['q_norm'].transpose(1, 2).float()
        k = cap[li]['k_norm'].transpose(1, 2).float()
        qr, kr = apply_rotary_pos_emb(q, k, rot['cos'].float(), rot['sin'].float())
        rep = cfg.num_attention_heads // cfg.num_key_value_heads
        kr = kr.repeat_interleave(rep, dim=1)
        s = torch.einsum('bhtd,bhsd->bhts', qr, kr) * scale
        s = s.masked_fill(causal[None, None], torch.finfo(torch.float32).min)
        w = torch.softmax(s, dim=-1)
        ref = out.attentions[li].float()
        e_last = float((w[0, :, -1] - ref[0, :, -1]).abs().max())
        e_all = float((w - ref).abs().max())
        errs.append((li, e_last, e_all))
        print('L%-2d  末位 max|Δ|=%.3e   全矩阵 max|Δ|=%.3e' % (li, e_last, e_all))

    gate_bf16 = max(e for _, e, _ in errs)

    # ---- 3) 精度归因：1e-2 的差是实现错，还是 bf16 逐步精度？
    # 判法：全程保持 bf16（RoPE 也不 upcast，与模型内部路径一致），若误差大幅下降即是精度问题。
    errs_bf = []
    for li in (0, 17, 35):
        q = cap[li]['q_norm'].transpose(1, 2)
        k = cap[li]['k_norm'].transpose(1, 2)
        qr, kr = apply_rotary_pos_emb(q, k, rot['cos'], rot['sin'])
        kr = kr.repeat_interleave(cfg.num_attention_heads // cfg.num_key_value_heads, dim=1)
        s = torch.einsum('bhtd,bhsd->bhts', qr, kr).float() * scale
        s = s.masked_fill(causal[None, None], torch.finfo(torch.float32).min)
        w = torch.softmax(s, dim=-1)
        e = float((w[0, :, -1] - out.attentions[li].float()[0, :, -1]).abs().max())
        errs_bf.append(e)
        print('L%-2d  全程bf16 复现 末位 max|Δ|=%.3e' % (li, e))
    gate_bf16_native = max(errs_bf)

    # ---- 4) 可选：fp32 全模型交叉验证（慢，默认跳过） ----
    gate_fp32 = None
    if os.environ.get('FP32_CHECK') == '1':
        m32 = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, attn_implementation='eager',
            trust_remote_code=True).to(dev).eval()
        cap2, rot2 = {}, {}
        for li in range(cfg.num_hidden_layers):
            att = m32.model.layers[li].self_attn
            for nm in ('q_norm', 'k_norm'):
                if hasattr(att, nm):
                    getattr(att, nm).register_forward_hook(mk2(cap2, li, nm))
        m32.model.rotary_emb.register_forward_hook(lambda mod, i, o: rot2.update(cos=o[0], sin=o[1]))
        with torch.no_grad():
            out2 = m32(input_ids=t, use_cache=False, output_attentions=True)
        worst = 0.0
        for li in (0, 17, 35):
            q = cap2[li]['q_norm'].transpose(1, 2)
            k = cap2[li]['k_norm'].transpose(1, 2)
            qr, kr = apply_rotary_pos_emb(q, k, rot2['cos'], rot2['sin'])
            kr = kr.repeat_interleave(cfg.num_attention_heads // cfg.num_key_value_heads, dim=1)
            s = torch.einsum('bhtd,bhsd->bhts', qr, kr) * scale
            s = s.masked_fill(causal[None, None], torch.finfo(torch.float32).min)
            w = torch.softmax(s, dim=-1)
            worst = max(worst, float((w - out2.attentions[li].float()).abs().max()))
        gate_fp32 = worst
        print('fp32 重算 max|Δ|=%.3e' % worst)

    gate = gate_bf16_native < 5e-3

    rec = {
        'layers': cfg.num_hidden_layers, 'q_heads': cfg.num_attention_heads,
        'kv_heads': cfg.num_key_value_heads,
        'head_dim': getattr(cfg, 'head_dim', 128),
        'n_single_token_nouns': len(single), 'pos_uniform': uniform,
        'seq_len': int(t.shape[1]),
        'repro_err_fp32compute': gate_bf16,
        'repro_err_bf16matmul': gate_bf16_native,
        'repro_err_fp32_model': gate_fp32,
        'gate_pass': gate,
        'nouns_single': single,
    }
    provenance.write_result('pat_q3t_probe', rec, model_path=MODEL_PATH,
                            seed=20261101, tool='pat_q3t_probe.py')
    print(json.dumps({k: v for k, v in rec.items() if k != 'nouns_single'}, ensure_ascii=False))
    return 0 if gate else 1


if __name__ == '__main__':
    raise SystemExit(main())
