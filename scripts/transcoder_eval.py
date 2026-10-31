# -*- coding: utf-8 -*-
"""transcoder.v1 评估：同一批留出 token 上，比较「字典特征作原子」与「原生神经元作原子」。

1. 重建：y = MLP_L(x)；transcoder 在其训练 k 下的 FVU；神经元在多个 k 下的 FVU（只保留贡献最大的 k 个中间单元）。
   报告「神经元需要多少个才能达到 transcoder 的 FVU」。
2. 模型内损失恢复：整模型前向，把第 L 块 MLP 输出替换为 clean / transcoder / 神经元 top-k / 零 / 训练均值，
   在留出窗口（各来源末尾，与训练窗口不重叠）的 skip_head 之后位置计算下一 token 交叉熵与相对 clean 的 KL。
   损失恢复分别相对零消融与均值消融报告。
3. 单义性代理：各特征 / 神经元正向最强 100 次激活处的 token 集中度、激活频率与死特征比例（描述，不是判据）。
判据见 transcoder.CRITERIA（同一 k 下 FVU 更低且损失恢复更高才算特征原子胜出），随结果写出。

用法：python scripts/transcoder_eval.py --collect <dir> --tc tc-m16384-k64-s0 [--tc ...] --out research/transcoder/TC-v1
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
sys.path.insert(0, str(ROOT / 'scripts'))


def running_topk(vals_iter, n_units, top, device):
    """逐块合并每个单元的最强 top 次激活（值与全局位置）。vals_iter 产出 (offset, [n, n_units] 张量)。"""
    import torch
    best_v = torch.full((top, n_units), -float('inf'), device=device)
    best_i = torch.zeros((top, n_units), dtype=torch.long, device=device)
    for off, v in vals_iter:
        k = min(top, v.shape[0])
        tv, ti = v.topk(k, dim=0)
        cv = torch.cat([best_v, tv]); ci = torch.cat([best_i, ti + off])
        sv, si = cv.topk(top, dim=0)
        best_v, best_i = sv, ci.gather(0, si)
    return best_v, best_i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', required=True)
    ap.add_argument('--tc', action='append', required=True)
    ap.add_argument('--neuron-k', default='16,32,64,128,256,512,1024,2048')
    ap.add_argument('--eval-windows', type=int, default=128, help='损失恢复用的留出窗口数（各来源均分）')
    ap.add_argument('--out', default='research/transcoder/TC-v1-qwen3-4b-L30')
    args = ap.parse_args()
    import torch
    from shared.dist_protocol import transcoder as T
    from transcoder_train import load_mlp
    t0 = time.time()
    cdir = Path(args.collect)
    man = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    L = man['layer']
    mlp, acts, W = load_mlp(man['model'], L, 'cuda')
    Wd_T = W['down_proj'].T.float()                                  # [d_ff, d]
    tcs = {name: T.TopKTranscoder.from_state(np.load(cdir / name / 'transcoder.npz'), device='cuda') for name in args.tc}
    Xe = np.load(cdir / 'x_eval.npy', mmap_mode='r')
    toks = np.load(cdir / 'tok_eval.npy')
    N = len(Xe)
    ks = [int(v) for v in args.neuron_k.split(',')]
    B = 4096
    # ---- 1. 重建 ----
    sse = {**{n: 0.0 for n in tcs}, **{f'neuron_k{k}': 0.0 for k in ks}}
    sy, syy = None, 0.0
    freq = {n: torch.zeros(t.m, device='cuda') for n, t in tcs.items()}
    nfreq = torch.zeros(Wd_T.shape[0], device='cuda')
    with torch.no_grad():
        for s in range(0, N, B):
            x = torch.as_tensor(np.asarray(Xe[s:s + B]), device='cuda').float()
            A = acts(x).float()
            y = A @ Wd_T
            sy = y.sum(0) if sy is None else sy + y.sum(0)
            syy += float(y.pow(2).sum())
            for n, t in tcs.items():
                a = t.encode(x)
                freq[n] += (a > 0).float().sum(0)
                sse[n] += float((y - t.decode(a)).pow(2).sum())
            nfreq += (A > 0).float().sum(0)
            for k in ks:
                sse[f'neuron_k{k}'] += float((y - T.neuron_topk_recon(A, Wd_T, k)).pow(2).sum())
    sst = syy - float(sy.pow(2).sum()) / N
    recon = {n: v / sst for n, v in sse.items()}
    # ---- 3. 集中度（正极最强 100 次）----
    top = T.CRITERIA['concentration_top']
    def it_feat(t):
        for s in range(0, N, B):
            x = torch.as_tensor(np.asarray(Xe[s:s + B]), device='cuda').float()
            yield s, t.encode(x)
    def it_neur():
        for s in range(0, N, B):
            x = torch.as_tensor(np.asarray(Xe[s:s + B]), device='cuda').float()
            yield s, acts(x).float()
    tok_t = torch.as_tensor(toks.astype(np.int64), device='cuda')
    def conc(best_v, best_i, alive):
        tk = tok_t[best_i]                                           # [top, units]
        mode = torch.mode(tk, dim=0).values
        c = (tk == mode[None]).float().mean(0)
        return c[alive].cpu().numpy()
    monos = {}
    with torch.no_grad():
        for n, t in tcs.items():
            bv, bi = running_topk(it_feat(t), t.m, top, 'cuda')
            alive = (bv[-1] > 0)                                     # 至少 top 次正激活
            c = conc(bv, bi, alive)
            monos[n] = {'units_with_100_acts': int(alive.sum()), 'concentration_median': float(np.median(c)),
                        'concentration_ge_0.5': float((c >= 0.5).mean()), 'concentration_ge_0.9': float((c >= 0.9).mean()),
                        'dead_in_eval': float((freq[n] == 0).float().mean()), 'freq_median': float(freq[n].median() / N)}
        bv, bi = running_topk(it_neur(), Wd_T.shape[0], top, 'cuda')
        alive = (bv[-1] > 0)
        c = conc(bv, bi, alive)
        monos['neurons'] = {'units_with_100_acts': int(alive.sum()), 'concentration_median': float(np.median(c)),
                            'concentration_ge_0.5': float((c >= 0.5).mean()), 'concentration_ge_0.9': float((c >= 0.9).mean()),
                            'positive_freq_median': float(nfreq.median() / N)}
    del W, mlp
    torch.cuda.empty_cache()
    # ---- 2. 模型内损失恢复 ----
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(man['model'], dtype=torch.bfloat16, device_map='cuda').eval()
    wins = np.load(cdir / 'eval_windows.npy')
    per_src = len(wins) // len(man['sources'])
    take = args.eval_windows // len(man['sources'])
    sel = np.concatenate([np.arange(i * per_src, i * per_src + take) for i in range(len(man['sources']))])
    wins = wins[sel]
    sk = man['skip_head']
    # 均值消融用训练集 y 的均值（前 20 万 token 估计）
    Xt = np.load(cdir / 'x_train.npy', mmap_mode='r')
    with torch.no_grad():
        ym = sum(mlp_y.sum(0) for mlp_y in (model.model.layers[L].mlp(torch.as_tensor(np.asarray(Xt[s:s + B]), device='cuda').to(torch.bfloat16)).float()
                                            for s in range(0, 200_000, B))) / 200_000
    mode = {'name': 'clean'}
    def hook(mod, inp, out):
        x = inp[0]
        nm = mode['name']
        if nm == 'clean':
            return out
        sh = out.shape
        xf = x.reshape(-1, sh[-1]).float()
        if nm == 'zero':
            r = torch.zeros_like(xf)
        elif nm == 'mean':
            r = ym[None].expand_as(xf)
        elif nm.startswith('neuron_k'):
            A = (torch.nn.functional.silu(mod.gate_proj(x)) * mod.up_proj(x)).reshape(-1, mod.down_proj.in_features).float()
            r = T.neuron_topk_recon(A, mod.down_proj.weight.T.float(), int(nm[8:]))
        else:
            r = tcs[nm](xf)
        return r.reshape(sh).to(out.dtype)
    hk = model.model.layers[L].mlp.register_forward_hook(hook)
    conds = ['clean', 'zero', 'mean'] + list(tcs) + [f'neuron_k{k}' for k in ks]
    ce = {c: 0.0 for c in conds}; kl = {c: 0.0 for c in conds}; ntok = 0
    with torch.no_grad():
        for b in range(0, len(wins), 2):
            ids = torch.as_tensor(wins[b:b + 2], device='cuda')
            tgt = ids[:, sk + 1:]
            logp_clean = None
            for c in conds:
                mode['name'] = c
                lp = torch.log_softmax(model(input_ids=ids).logits[:, sk:-1].float(), -1)
                ce[c] += float(-lp.gather(-1, tgt[..., None]).sum())
                if c == 'clean':
                    logp_clean = lp
                else:
                    kl[c] += float((logp_clean.exp() * (logp_clean - lp)).sum())
            ntok += tgt.numel()
    hk.remove()
    ce = {c: v / ntok for c, v in ce.items()}; kl = {c: v / ntok for c, v in kl.items()}
    rows = {}
    for c in conds[3:]:
        rows[c] = {'fvu': recon.get(c), 'ce': ce[c], 'kl': kl[c],
                   'loss_recovered_vs_zero': T.loss_recovered(ce['clean'], ce[c], ce['zero']),
                   'loss_recovered_vs_mean': T.loss_recovered(ce['clean'], ce[c], ce['mean'])}
    # 原子比较：每个 transcoder 与同 k 的神经元
    compare = {}
    for n, t in tcs.items():
        nk = f'neuron_k{t.k}'
        if nk in rows:
            f, g = rows[n], rows[nk]
            compare[n] = {'k': t.k, 'feature_wins': bool(f['fvu'] < g['fvu'] and f['loss_recovered_vs_zero'] > g['loss_recovered_vs_zero']),
                          'fvu': [f['fvu'], g['fvu']], 'loss_recovered_vs_zero': [f['loss_recovered_vs_zero'], g['loss_recovered_vs_zero']],
                          'neurons_needed_for_same_fvu': next((k for k in ks if recon[f'neuron_k{k}'] <= f['fvu']), None)}
    report = {'schema': T.SCHEMA, 'criteria': T.CRITERIA, 'collect': str(cdir), 'model_snapshot': man['model_snapshot'],
              'layer': L, 'n_eval_tokens_recon': int(N), 'n_eval_tokens_ce': int(ntok), 'eval_windows': int(len(wins)),
              'ce_clean': ce['clean'], 'ce_zero': ce['zero'], 'ce_mean': ce['mean'], 'conditions': rows, 'atom_compare': compare,
              'monosemanticity_proxy': monos,
              'transcoders': {n: json.loads((cdir / n / 'train.json').read_text(encoding='utf-8')) | {'curve': None} for n in tcs},
              'seconds': round(time.time() - t0, 1), 'torch': torch.__version__}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'ce': {k: round(v, 4) for k, v in ce.items()}, 'compare': compare,
                      'recon': {k: round(v, 4) for k, v in recon.items()}}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
