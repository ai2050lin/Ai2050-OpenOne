# -*- coding: utf-8 -*-
"""GQ-ATOM-v1：GR-Q（疑问 / 陈述 → 说话人「不确定 / 确定」）在 Qwen3-4B 第 L 块上，字典特征与原生神经元谁更像被使用的原子。

预注册（写于运行之前，判据在 CRITERIA，随结果写出）：
  行为量：probe 提示后完整答案对数概率差 m = log P(不确定) − log P(确定)；有符号间隔 s = m（疑问）或 −m（陈述）。
  1. 选择（只用 discovery）：在 probe 提示最后一个 token（读出位置 probe_answer）上，计算每个 transcoder 特征激活
     与每个 MLP 中间单元 A = silu(G)⊙V 的疑问 − 陈述标准化均值差 d（Cohen d）；按 |d| 各取前 N 个。
  2. 复现：在留出划分（confirm + lockbox：未见填充词；lockbox_frame：未见句式）上 d 同号的比例。
  3. 必要性（只在留出划分上）：在提示的全部位置把所选单元换成自然语料均值——
       特征：y' = y − Σ_f (a_f − ā_f)·d_f（保留 transcoder 误差项，只去掉这些特征偏离均值的部分）；
       神经元：A_i → Ā_i 后重算 down_proj。
     效应 = 留出项平均 s 的下降比例；随机对照 = 同数量随机单元（从自然语料激活频率不为零的单元中抽），R 次。
  判据：某原子类型「被使用」当且仅当 top-N 消融的下降比例 ≥ min_drop 且大于全部 R 次随机对照；
       「特征原子胜出」当且仅当两者都算被使用时特征的下降更大，或只有特征被使用。
用法：python scripts/grq_atom_test.py --collect <dir> --tc tc-m16384-k64-s0 --out research/transcoder/GQ-ATOM-v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CRITERIA = {'schema': 'gq_atom.v1', 'N': 10, 'R': 5, 'min_drop': 0.10, 'seed': 0, 'readout': 'probe_answer (last prompt token)',
            'ablation_positions': 'all prompt positions', 'mean_source': 'first 200k natural-corpus tokens of x_train'}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', required=True)
    ap.add_argument('--tc', required=True)
    ap.add_argument('--grammar', default='shared/dist_protocol/grammars/GR-Q.json')
    ap.add_argument('--out', default='research/transcoder/GQ-ATOM-v1')
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from shared.dist_protocol import grammar as G
    from shared.dist_protocol import transcoder as T
    t0 = time.time()
    cdir = Path(args.collect)
    man = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    L = man['layer']
    tc = T.TopKTranscoder.from_state(np.load(cdir / args.tc / 'transcoder.npz'), device='cuda')
    spec = json.loads(Path(args.grammar).read_text(encoding='utf-8'))
    items = [it for it in G.generate(spec)['items'] if it.get('split') != 'canary']
    tok = AutoTokenizer.from_pretrained(man['model'])
    model = AutoModelForCausalLM.from_pretrained(man['model'], dtype=torch.bfloat16, device_map='cuda').eval()
    mlp = model.model.layers[L].mlp
    Wd_T = mlp.down_proj.weight.T.float()

    # 自然语料均值（特征激活与神经元 A）
    Xt = np.load(cdir / 'x_train.npy', mmap_mode='r')
    with torch.no_grad():
        a_sum = torch.zeros(tc.m, device='cuda'); A_sum = torch.zeros(Wd_T.shape[0], device='cuda'); n = 0
        a_nz = torch.zeros(tc.m, device='cuda')
        for s in range(0, 200_000, 4096):
            x = torch.as_tensor(np.asarray(Xt[s:s + 4096]), device='cuda')
            a = tc.encode(x.float()); a_sum += a.sum(0); a_nz += (a > 0).float().sum(0)
            xb = x.to(torch.bfloat16)
            A_sum += (torch.nn.functional.silu(mlp.gate_proj(xb)) * mlp.up_proj(xb)).float().sum(0); n += len(x)
        a_mean, A_mean = a_sum / n, A_sum / n
    alive_feat = torch.nonzero(a_nz > 0).flatten().cpu().numpy()

    ctl = {'mode': 'clean', 'upto': None, 'feat': None, 'neur': None, 'cap': None}
    def hook(mod, inp, out):
        x = inp[0]
        if ctl['cap'] is not None:
            xf = x[0, -1].float()
            ctl['cap']['a'] = tc.encode(xf[None])[0]
            ctl['cap']['A'] = (torch.nn.functional.silu(mod.gate_proj(x[0, -1])) * mod.up_proj(x[0, -1])).float()
        if ctl['mode'] == 'clean':
            return out
        u = ctl['upto']
        xf = x[0, :u].float()
        if ctl['mode'] == 'feat':
            f = torch.as_tensor(ctl['feat'], device='cuda')
            a = tc.encode(xf)[:, f]
            delta = ((a - a_mean[f]) @ tc.W_dec[f]) * tc.y_scale
            new = out[0, :u].float() - delta
        else:
            i = torch.as_tensor(ctl['neur'], device='cuda')
            A = (torch.nn.functional.silu(mod.gate_proj(x[0, :u])) * mod.up_proj(x[0, :u])).float()
            new = out[0, :u].float() - (A[:, i] - A_mean[i]) @ Wd_T[i]
        out = out.clone()
        out[0, :u] = new.to(out.dtype)
        return out
    hk = mlp.register_forward_hook(hook)

    def margin(it):
        p = tok(it['probe']['prompt'], add_special_tokens=False).input_ids
        lp = {}
        for ans in ('不确定', '确定'):
            a = tok(ans, add_special_tokens=False).input_ids
            ids = torch.as_tensor([p + a], device='cuda')
            ctl['upto'] = len(p)
            logits = model(input_ids=ids).logits[0].float()
            logp = torch.log_softmax(logits[len(p) - 1:-1], -1)
            lp[ans] = float(logp.gather(-1, torch.as_tensor(a, device='cuda')[:, None]).sum())
        m = lp['不确定'] - lp['确定']
        return m if it['cond']['mood'] == 'interrog' else -m

    # 1. 选择与复现：读出位置上的激活（clean）
    acts_f, acts_n, mood, split = [], [], [], []
    with torch.no_grad():
        for it in items:
            cap = {}
            ctl.update(mode='clean', cap=cap)
            model(input_ids=torch.as_tensor([tok(it['probe']['prompt'], add_special_tokens=False).input_ids], device='cuda'))
            acts_f.append(cap['a'].cpu().numpy()); acts_n.append(cap['A'].cpu().numpy())
            mood.append(it['cond']['mood'] == 'interrog'); split.append(it['split'])
    ctl['cap'] = None
    acts_f, acts_n, mood, split = np.array(acts_f), np.array(acts_n), np.array(mood), np.array(split)

    def cohen(X, sel):
        a, b = X[sel & mood], X[sel & ~mood]
        sd = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-6
        return (a.mean(0) - b.mean(0)) / sd
    disc = split == 'discovery'
    N, R = CRITERIA['N'], CRITERIA['R']
    res = {}
    rng = np.random.default_rng(CRITERIA['seed'])
    for kind, X, pool in (('feature', acts_f, alive_feat), ('neuron', acts_n, np.arange(acts_n.shape[1]))):
        d = cohen(X, disc)
        d[~np.isin(np.arange(len(d)), pool)] = 0
        top = np.argsort(-np.abs(d))[:N]
        rep = {}
        for nm, sel in (('unseen_fillers', np.isin(split, ['confirm', 'lockbox'])), ('unseen_frames', split == 'lockbox_frame')):
            dh = cohen(X, sel)
            rep[nm] = float(np.mean(np.sign(dh[top]) == np.sign(d[top])))
        res[kind] = {'top': [{'unit': int(u), 'd_discovery': float(d[u])} for u in top], 'sign_replication': rep,
                     'random_sets': [rng.choice(pool, N, replace=False).tolist() for _ in range(R)]}
    # 2. 必要性（留出划分）
    held = [it for it in items if it['split'] in ('confirm', 'lockbox', 'lockbox_frame')]
    def mean_s(mode, units=None):
        ctl.update(mode=mode, feat=units if mode == 'feat' else None, neur=units if mode == 'neur' else None)
        with torch.no_grad():
            return float(np.mean([margin(it) for it in held]))
    s_clean = mean_s('clean')
    for kind, mode in (('feature', 'feat'), ('neuron', 'neur')):
        r = res[kind]
        s_top = mean_s(mode, [u['unit'] for u in r['top']])
        s_rand = [mean_s(mode, rs) for rs in r['random_sets']]
        drop = lambda s: (s_clean - s) / s_clean
        r['drop_top'] = drop(s_top)
        r['drop_random'] = [drop(s) for s in s_rand]
        r['used'] = bool(r['drop_top'] >= CRITERIA['min_drop'] and r['drop_top'] > max(r['drop_random']))
    hk.remove()
    f, g = res['feature'], res['neuron']
    verdict = {'feature_used': f['used'], 'neuron_used': g['used'],
               'feature_wins': bool((f['used'] and g['used'] and f['drop_top'] > g['drop_top']) or (f['used'] and not g['used']))}
    report = {'criteria': CRITERIA, 'verdict': verdict, 'clean_mean_signed_margin': s_clean, 'n_heldout': len(held),
              'n_discovery': int(disc.sum()), 'layer': L, 'transcoder': args.tc, 'model_snapshot': man['model_snapshot'],
              'grammar': spec['family_id'], 'grammar_version': spec.get('grammar_version'), 'result': res,
              'seconds': round(time.time() - t0, 1), 'torch': torch.__version__}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'verdict': verdict, 's_clean': s_clean,
                      'feature': {k: f[k] for k in ('drop_top', 'drop_random', 'sign_replication')},
                      'neuron': {k: g[k] for k in ('drop_top', 'drop_random', 'sign_replication')}}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
