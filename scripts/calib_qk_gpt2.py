# -*- coding: utf-8 -*-
"""方法校准 C2：注意力「为什么看这里」（QK 归因）——在 GPT-2 small 的两个已知 QK 组合上检验路径修补。

已知答案：
  A. IOI 的查询侧组合（Wang et al. 2022）：S 抑制头 7.3 / 7.9 / 8.6 / 8.10 的输出进入名称搬运头 9.9 / 9.6 / 10.0 的查询，
     让它们在末位看向间接宾语（IO）。
  B. 归纳头的键侧组合（Olsson et al. 2022）：前一 token 头（GPT-2 small 中主要是 4.11）的输出进入归纳头
     5.1 / 5.5 / 6.9 / 7.2 / 7.10 的键，让它们看向「上一次出现之后的那个 token」。

做法（直接路径修补，只改一条输入路径）：目标头 (L, H) 的注意力由 ln_1(x_L) 算出 q、k。对上游头 (l, h)，l < L，
把它在残差中的输出 o 换成破坏输入上的输出，只用于 q（A）或只用于 k（B）：
    x_q = x_L − o_clean + o_corrupt，   q = ln_1(x_q) W_Q，  k = ln_1(x_L) W_K（B 反过来）
效应 = 目标注意力概率的平均下降（A：末位 → IO；B：第二段位置 i → i − n + 1）。
判据在运行前固定于 CRITERIA，随结果写出。

用法：python scripts/calib_qk_gpt2.py --model D:/develop/model/gpt2-local --out research/calibration/C2-qk-gpt2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_ioi_gpt2 as IOI  # noqa: E402

CRITERIA = {
    'schema': 'calib_qk.v1',
    'ioi_targets': ['9.9', '9.6', '10.0'], 's_inhibition': ['7.3', '7.9', '8.6', '8.10'],
    'ioi_top4_s_inhibition_min': 3,                 # 查询路径效应前 4 的上游头中至少 3 个是 S 抑制头
    'induction_targets': ['5.1', '5.5', '6.9', '7.2', '7.10'], 'prev_token_head': '4.11',
    'induction_prev_rank_max': 2,                   # 键路径效应中 4.11 排在前 2
    'ioi_prompts': 240, 'induction_seqs': 64, 'induction_len': 40, 'seed': 0,
}


class Capture:
    """记录每层残差输入 x_l 与每个头写入残差的输出 o_{l,h}（不含 c_proj 偏置，偏置在差值中抵消）。"""

    def __init__(self, model):
        self.m, self.L, self.H = model, model.config.n_layer, model.config.n_head
        self.dh = model.config.n_embd // self.H
        self.x, self.z = {}, {}
        self.hooks = []
        for l, blk in enumerate(model.transformer.h):
            self.hooks.append(blk.register_forward_pre_hook(lambda m, a, k, l=l: self.x.__setitem__(l, (a[0] if a else k['hidden_states']).detach()), with_kwargs=True))
            self.hooks.append(blk.attn.c_proj.register_forward_pre_hook(lambda m, a, l=l: self.z.__setitem__(l, a[0].detach())))

    def run(self, ids):
        with torch.no_grad():
            self.m(ids)
        return {l: v.clone() for l, v in self.x.items()}, {l: v.clone() for l, v in self.z.items()}

    def head_out(self, z, l, h):
        W = self.m.transformer.h[l].attn.c_proj.weight
        return z[l][..., h * self.dh:(h + 1) * self.dh] @ W[h * self.dh:(h + 1) * self.dh]

    def pattern(self, l, h, xq, xk):
        blk = self.m.transformer.h[l]
        d = self.m.config.n_embd
        Wa, ba = blk.attn.c_attn.weight, blk.attn.c_attn.bias
        sl = slice(h * self.dh, (h + 1) * self.dh)
        q = blk.ln_1(xq) @ Wa[:, :d][:, sl] + ba[:d][sl]
        k = blk.ln_1(xk) @ Wa[:, d:2 * d][:, sl] + ba[d:2 * d][sl]
        s = q @ k.transpose(-1, -2) / self.dh ** 0.5
        T = s.shape[-1]
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1), float('-inf'))
        return s.softmax(-1)


def parse(name):
    l, h = name.split('.')
    return int(l), int(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='D:/develop/model/gpt2-local')
    ap.add_argument('--out', default='research/calibration/C2-qk-gpt2')
    args = ap.parse_args()
    from transformers import AutoTokenizer, GPT2LMHeadModel
    t0 = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, dtype=torch.float32, attn_implementation='eager').to(dev).eval()
    cap = Capture(model)
    L, H = cap.L, cap.H
    names = [f'{l}.{h}' for l in range(L) for h in range(H)]

    # ---- A. IOI 查询侧 ----
    rows = IOI.build(tok, CRITERIA['ioi_prompts'], CRITERIA['seed'])
    groups = {}
    for r in rows:
        groups.setdefault(r['template'], []).append(r)
    effA = np.zeros((L, H)); base_attn = []; n = 0
    with torch.no_grad():
        for g in groups.values():
            cl = tok([r['clean'] for r in g], return_tensors='pt').input_ids.to(dev)
            co = tok([r['corrupt'] for r in g], return_tensors='pt').input_ids.to(dev)
            io = torch.tensor([tok.encode(r['io'])[0] for r in g], device=dev)
            io_pos = (cl == io[:, None]).float().argmax(1)
            xc, zc = cap.run(cl)
            _, zx = cap.run(co)
            b = torch.arange(len(g), device=dev)
            for tgt in CRITERIA['ioi_targets']:
                Lt, Ht = parse(tgt)
                a0 = cap.pattern(Lt, Ht, xc[Lt], xc[Lt])[b, -1, io_pos]
                base_attn.append(float(a0.mean()))
                for l in range(Lt):
                    for h in range(H):
                        xq = xc[Lt] - cap.head_out(zc, l, h) + cap.head_out(zx, l, h)
                        a1 = cap.pattern(Lt, Ht, xq, xc[Lt])[b, -1, io_pos]
                        effA[l, h] += float((a0 - a1).sum()) / len(CRITERIA['ioi_targets'])
            n += len(g)
    effA /= n

    # ---- B. 归纳头键侧 ----
    rng = np.random.default_rng(CRITERIA['seed'])
    S, T = CRITERIA['induction_seqs'], CRITERIA['induction_len']
    def rep_batch():
        r = rng.integers(1000, 30000, size=(S, T))
        return torch.as_tensor(np.concatenate([np.full((S, 1), tok.bos_token_id), r, r], 1), device=dev)
    clean, corrupt = rep_batch(), rep_batch()
    effB = np.zeros((L, H)); effBq = np.zeros((L, H)); ind_base = {}
    qpos = torch.arange(T + 1, 2 * T + 1, device=dev)          # 第二段的位置
    kpos = qpos - T + 1                                          # 上一次出现之后的 token
    with torch.no_grad():
        xc, zc = cap.run(clean)
        _, zx = cap.run(corrupt)
        for tgt in CRITERIA['induction_targets']:
            Lt, Ht = parse(tgt)
            p0 = cap.pattern(Lt, Ht, xc[Lt], xc[Lt])[:, qpos, kpos]
            ind_base[tgt] = float(p0.mean())
            for l in range(Lt):
                for h in range(H):
                    o_c, o_x = cap.head_out(zc, l, h), cap.head_out(zx, l, h)
                    xk = xc[Lt] - o_c + o_x
                    effB[l, h] += float((p0 - cap.pattern(Lt, Ht, xc[Lt], xk)[:, qpos, kpos]).mean()) / len(CRITERIA['induction_targets'])
                    effBq[l, h] += float((p0 - cap.pattern(Lt, Ht, xk, xc[Lt])[:, qpos, kpos]).mean()) / len(CRITERIA['induction_targets'])
    for hk in cap.hooks:
        hk.remove()

    oA = np.argsort(-effA.reshape(-1)); oB = np.argsort(-effB.reshape(-1))
    topA = [names[i] for i in oA[:10]]; topB = [names[i] for i in oB[:10]]
    li, hi = parse(CRITERIA['prev_token_head'])
    checks = {
        'ioi_query_s_inhibition': sum(h in CRITERIA['s_inhibition'] for h in topA[:4]) >= CRITERIA['ioi_top4_s_inhibition_min'],
        'induction_key_prev_token': CRITERIA['prev_token_head'] in topB[:CRITERIA['induction_prev_rank_max']],
    }
    res = {
        'ioi': {'baseline_attn_to_io': base_attn, 'top10_query_path': [{'head': names[i], 'effect': float(effA.reshape(-1)[i])} for i in oA[:10]]},
        'induction': {'baseline_attn': ind_base, 'top10_key_path': [{'head': names[i], 'effect': float(effB.reshape(-1)[i])} for i in oB[:10]],
                      'prev_token_head_key_vs_query': [float(effB[li, hi]), float(effBq[li, hi])]},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'qk_effects.npz', ioi_query=effA, induction_key=effB, induction_query=effBq)
    snap = hashlib.sha256((Path(args.model) / 'model.safetensors').read_bytes()).hexdigest()
    report = {'criteria': CRITERIA, 'checks': checks, 'passed': all(checks.values()), 'result': res, 'model': args.model,
              'model_sha256': snap, 'dtype': 'float32', 'attn': 'eager', 'torch': torch.__version__, 'seconds': round(time.time() - t0, 1)}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'checks': checks, 'topA': topA[:6], 'topB': topB[:6], 'ind_base': ind_base,
                      'prev_k_vs_q': res['induction']['prev_token_head_key_vs_query']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
