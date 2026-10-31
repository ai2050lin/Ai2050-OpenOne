# -*- coding: utf-8 -*-
"""方法校准 C1：GPT-2 small 上的 IOI（间接宾语识别）——在「答案已知」的模型与任务上检验本项目的因果归因做法。

已知答案（Wang et al. 2022, "Interpretability in the Wild"）：名称搬运头 9.9 / 9.6 / 10.0 把间接宾语名字写进输出；
负名称搬运头 10.7 / 11.10 起反作用；S 抑制头 7.3 / 7.9 / 8.6 / 8.10；归纳头 5.5 / 6.9。

做法：
  1. 干净句（BABA 模板）与 ABC 破坏句（三个名字全换成别的名字，长度相同）；指标 = 末位 logit(IO) − logit(S)。
  2. 逐头激活修补（精确）：把破坏句中某头的输出（c_proj 输入里该头的切片，全部位置）换进干净句，
     效应 = (LD_patched − LD_clean) / (LD_corrupt − LD_clean)，1 表示该头单独解释了全部差异。
  3. 逐头 attribution patching（一阶近似，AtP）：效应 ≈ (z_corrupt − z_clean) · ∂LD/∂z，同样归一化。
判据在运行前固定于 CRITERIA，结果随判据一起写出；不在看到结果后修改。

  4. 事后诊断（看到第一次运行结果后加入，判据 POST_HOC 不属于预注册）：逐头直接 logit 归因（DLA）——
     末位上该头输出经 ln_f（用实际末层残差的尺度）投到 W_U[IO] − W_U[S]。它不经过下游层，因此不受自我修复影响。

用法：python scripts/calib_ioi_gpt2.py --model D:/develop/model/gpt2-local --out research/calibration/C1-ioi-gpt2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

CRITERIA = {
    'schema': 'calib_ioi.v1',
    'clean_logit_diff_min': 2.0,                    # 论文在其数据上约 3.5；低于 2 说明材料或指标没复现任务
    'top3_name_movers_min': 2,                      # 精确修补效应前 3 的头中至少 2 个属于 {9.9, 9.6, 10.0}
    'name_movers': ['9.9', '9.6', '10.0'],
    'negative_movers': ['10.7', '11.10'],           # 至少一个效应为负（换入破坏后 LD 反而变大）
    's_inhibition': ['7.3', '7.9', '8.6', '8.10'],  # 记录：至少 2 个进入前 15
    'atp_spearman_min': 0.7,                        # AtP 与精确修补在 144 个头上的秩相关
    'n_prompts': 240, 'seed': 0,
}

POST_HOC = {'dla_top3_name_movers_min': 2, 'dla_bottom2_negative_movers': True}

NAMES = ['Mary', 'John', 'Tom', 'James', 'Dan', 'Sid', 'Martin', 'Amy', 'Kate', 'Paul', 'Mark', 'Lisa', 'Anna',
         'Jack', 'Sam', 'Ben', 'Alex', 'David', 'Laura', 'Emma', 'Mike', 'Sarah', 'Peter', 'Kevin']
PLACES = ['store', 'garden', 'restaurant', 'school', 'hospital', 'office', 'house', 'station']
OBJECTS = ['ring', 'kiss', 'bone', 'basketball', 'computer', 'necklace', 'drink', 'snack']
TEMPLATES = [
    'Then, {B} and {A} went to the {P}. {B} gave a {O} to',
    'When {B} and {A} got a {O} at the {P}, {B} decided to give it to',
    'After {B} and {A} went to the {P}, {B} gave a {O} to',
    'While {B} and {A} were working at the {P}, {B} gave a {O} to',
    'Friends {B} and {A} found a {O} at the {P}. {B} gave it to',
    'Then, {B} and {A} had a long argument, and afterwards {B} said to',
]


def single_token(tok, words):
    return [w for w in words if len(tok.encode(' ' + w)) == 1]


def build(tok, n, seed):
    rng = random.Random(seed)
    names, places, objs = single_token(tok, NAMES), single_token(tok, PLACES), single_token(tok, OBJECTS)
    rows = []
    for i in range(n):
        t = TEMPLATES[i % len(TEMPLATES)]
        a, b, c, d, e = rng.sample(names, 5)
        p, o = rng.choice(places), rng.choice(objs)
        clean = t.format(A=a, B=b, P=p, O=o)
        corrupt = t.format(A=c, B=d, P=p, O=o).replace(f' {d} gave', f' {e} gave').replace(f' {d} decided', f' {e} decided') \
            .replace(f'afterwards {d}', f'afterwards {e}')
        rows.append({'template': i % len(TEMPLATES), 'clean': clean, 'corrupt': corrupt, 'io': ' ' + a, 's': ' ' + b})
    return rows


def dla_heads(model, tok, rows, dev):
    L, H = model.config.n_layer, model.config.n_head
    dh = model.config.n_embd // H
    store = {}
    hooks = [model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m, i, l=l: store.__setitem__(l, i[0]))
             for l in range(L)]
    hooks.append(model.transformer.ln_f.register_forward_pre_hook(lambda m, i: store.__setitem__('f', i[0])))
    WU, g = model.lm_head.weight, model.transformer.ln_f.weight
    dla, groups = np.zeros((L, H)), {}
    for r in rows:
        groups.setdefault(r['template'], []).append(r)
    with torch.no_grad():
        for gr in groups.values():
            model(tok([r['clean'] for r in gr], return_tensors='pt').input_ids.to(dev))
            io = torch.tensor([tok.encode(r['io'])[0] for r in gr], device=dev)
            s = torch.tensor([tok.encode(r['s'])[0] for r in gr], device=dev)
            xf = store['f'][:, -1]
            sd = (xf - xf.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).add(model.config.layer_norm_epsilon).sqrt()
            d = (WU[io] - WU[s]) * g
            for l in range(L):
                W, z = model.transformer.h[l].attn.c_proj.weight, store[l][:, -1]
                for h in range(H):
                    o = z[:, h * dh:(h + 1) * dh] @ W[h * dh:(h + 1) * dh]
                    dla[l, h] += (((o - o.mean(-1, keepdim=True)) / sd) * d).sum().item()
    for hk in hooks:
        hk.remove()
    return dla / len(rows)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='D:/develop/model/gpt2-local')
    ap.add_argument('--out', default='research/calibration/C1-ioi-gpt2')
    args = ap.parse_args()
    from transformers import AutoTokenizer, GPT2LMHeadModel
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation='eager').to(dev).eval()
    L, H = model.config.n_layer, model.config.n_head
    dh = model.config.n_embd // H
    rows = build(tok, CRITERIA['n_prompts'], CRITERIA['seed'])

    z_store = {}                      # 层 → c_proj 输入 [b, t, n_embd]
    patch = {}                        # 层 → (head, 替换值)
    def make_hook(l):
        def pre(mod, inp):
            x = inp[0]
            if l in patch:
                h, val = patch[l]
                x = x.clone()
                x[..., h * dh:(h + 1) * dh] = val[..., h * dh:(h + 1) * dh]
            if x.requires_grad:
                x.retain_grad()
            z_store[l] = x
            return (x,)
        return pre
    hooks = [model.transformer.h[l].attn.c_proj.register_forward_pre_hook(make_hook(l)) for l in range(L)]

    def ld(logits, io, s):
        return logits[:, -1].gather(1, io[:, None])[:, 0] - logits[:, -1].gather(1, s[:, None])[:, 0]

    exact = np.zeros((L, H)); atp = np.zeros((L, H)); ld_clean_all, ld_corrupt_all = [], []
    groups = {}
    for r in rows:
        groups.setdefault(r['template'], []).append(r)
    t0 = time.time()
    for g in groups.values():
        cl = tok([r['clean'] for r in g], return_tensors='pt').input_ids.to(dev)
        co = tok([r['corrupt'] for r in g], return_tensors='pt').input_ids.to(dev)
        assert cl.shape == co.shape, '干净句与破坏句的 token 长度必须相同'
        io = torch.tensor([tok.encode(r['io'])[0] for r in g], device=dev)
        s = torch.tensor([tok.encode(r['s'])[0] for r in g], device=dev)
        with torch.no_grad():
            model(co)
            z_co = {l: z_store[l].detach().clone() for l in range(L)}
            ld_co = ld(model(co).logits, io, s)
        model.zero_grad()
        out = model(cl)
        ld_cl = ld(out.logits, io, s)
        ld_cl.sum().backward()
        z_cl = {l: z_store[l].detach() for l in range(L)}
        grads = {l: z_store[l].grad.detach() for l in range(L)}
        denom = (ld_co - ld_cl).detach()
        for l in range(L):
            contrib = ((z_co[l] - z_cl[l]) * grads[l]).view(*z_cl[l].shape[:2], H, dh).sum(dim=(1, 3))   # [b, H]
            atp[l] += (contrib / denom[:, None]).sum(0).cpu().numpy()
        with torch.no_grad():
            for l in range(L):
                for h in range(H):
                    patch.clear(); patch[l] = (h, z_co[l])
                    exact[l, h] += ((ld(model(cl).logits, io, s) - ld_cl.detach()) / denom).sum().item()
            patch.clear()
        ld_clean_all += ld_cl.detach().cpu().tolist(); ld_corrupt_all += ld_co.cpu().tolist()
    for hk in hooks:
        hk.remove()
    n = len(rows)
    exact /= n; atp /= n
    names = [f'{l}.{h}' for l in range(L) for h in range(H)]
    flat = exact.reshape(-1)
    order = np.argsort(-flat)
    top = [names[i] for i in order[:15]]
    neg = {k: float(exact[int(k.split('.')[0]), int(k.split('.')[1])]) for k in CRITERIA['negative_movers']}
    res = {
        'clean_logit_diff': float(np.mean(ld_clean_all)), 'corrupt_logit_diff': float(np.mean(ld_corrupt_all)),
        'top15_exact': [{'head': names[i], 'effect': float(flat[i])} for i in order[:15]],
        'top15_atp': [{'head': names[i], 'effect': float(atp.reshape(-1)[i])} for i in np.argsort(-atp.reshape(-1))[:15]],
        'negative_movers_effect': neg,
        'atp_vs_exact_spearman': spearman(atp.reshape(-1), flat),
        'atp_vs_exact_pearson': float(np.corrcoef(atp.reshape(-1), flat)[0, 1]),
    }
    checks = {
        'clean_logit_diff': res['clean_logit_diff'] >= CRITERIA['clean_logit_diff_min'],
        'top3_name_movers': sum(h in CRITERIA['name_movers'] for h in top[:3]) >= CRITERIA['top3_name_movers_min'],
        'negative_mover_sign': any(v < 0 for v in neg.values()),
        'atp_spearman': res['atp_vs_exact_spearman'] >= CRITERIA['atp_spearman_min'],
    }
    res['s_inhibition_in_top15'] = [h for h in CRITERIA['s_inhibition'] if h in top]
    dla = dla_heads(model, tok, rows, dev).reshape(-1)
    d_order = np.argsort(-dla)
    res['post_hoc_dla'] = {'top8': [{'head': names[i], 'dla': float(dla[i])} for i in d_order[:8]],
                           'bottom4': [{'head': names[i], 'dla': float(dla[i])} for i in d_order[-4:]]}
    post = {'dla_top3_name_movers': sum(names[i] in CRITERIA['name_movers'] for i in d_order[:3]) >= POST_HOC['dla_top3_name_movers_min'],
            'dla_bottom2_negative_movers': set(names[i] for i in d_order[-2:]) == set(CRITERIA['negative_movers'])}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'head_effects.npz', exact=exact, atp=atp, dla=dla.reshape(L, H))
    snap = hashlib.sha256((Path(args.model) / 'model.safetensors').read_bytes()).hexdigest()
    report = {'criteria': CRITERIA, 'checks': checks, 'passed': all(checks.values()), 'post_hoc_criteria': POST_HOC,
              'post_hoc_checks': post, 'result': res,
              'n_prompts': n, 'examples': rows[:4], 'model': args.model, 'model_sha256': snap,
              'torch': torch.__version__, 'device': dev, 'dtype': 'float32', 'attn': 'eager', 'seconds': round(time.time() - t0, 1)}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'checks': checks, 'clean_ld': res['clean_logit_diff'], 'top8': top[:8],
                      'neg': neg, 'spearman': res['atp_vs_exact_spearman'], 'post_hoc': post}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
