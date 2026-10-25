# -*- coding: utf-8 -*-
"""单层 MLP 单元的因果归因与真实消融验证（L30-v1 第 3 部分，预注册 research/neurons/L30-v1.prereg.json）。

归因：第 0–29 层不建计算图（参数不求梯度、第 30 层的残差输入换成 requires_grad 的叶子），对 30–35 层与 lm_head
做一次反向。损失为计入位置的下一 token 交叉熵之和；对第 30 层 down_proj 输入 A 的梯度因此包含该位置对之后所有位置
损失的影响，与「在全部位置做均值消融」的一阶近似一致：
    ΔL̂_j = Σ_s (μ_j − A_j(s)) · ∂(Σ_t L_t)/∂A_j(s) / N_计入
直接通路用最终 RMSNorm 之前残差流的梯度投影到 W_down 列：D̂_j = Σ_s (μ_j − A_j(s)) · ⟨W_down[:, j], ∂(ΣL)/∂x^final_s⟩ / N。
验证：前 50（按 |ΔL̂|）+ 随机 50 个单元在不重叠的验证窗口上逐个做真实均值消融（全部位置），以及前 50 同时消融。

用法：python scripts/run_layer_attribution.py --layer 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NATIVE = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))
RUN = NATIVE / '_census' / '6be1034f0b55a6df-bf16' / '2ca5f0a756d36c8a-20000000'
MODEL = ROOT / 'models' / 'hf' / 'qwen3-4b'
PREREG = ROOT / 'research' / 'neurons' / 'L30-v1.prereg.json'


def pick_windows(n_windows, mod, rem, count):
    """按来源轮流取窗口序号 w ≡ rem (mod)，直到 count 个。返回 [(source_id, w)]。"""
    cand = [[w for w in range(n) if w % mod == rem] for n in n_windows]
    out, i = [], 0
    while len(out) < count and any(cand):
        s = i % len(cand)
        if cand[s]:
            out.append((s, cand[s].pop(0)))
        i += 1
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> int:
    import torch
    from transformers import AutoModelForCausalLM
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--layer', type=int, default=30)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--val-batch', type=int, default=8)
    ap.add_argument('--out', default=str(ROOT / 'research' / 'neurons' / 'L30-v1-attribution.json'))
    a = ap.parse_args(argv)
    t0 = time.time()
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    P3 = pre['part3_attribution']
    man = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    cdir = NATIVE / '_census' / 'corpus' / man['corpus_fp']
    cman = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    sep = int(cman['separator_id'])
    seq, skip = int(man['seq_len']), int(man['skip_head'])
    streams = [np.load(cdir / s['file'])['tokens'] for s in cman['sources']]
    n_win = [len(t) // seq for t in streams]
    assert sum(n_win) == 38424, sum(n_win)
    att_w = pick_windows(n_win, 47, 0, 400)
    val_w = pick_windows(n_win, 47, 23, 40)
    assert not set(att_w) & set(val_w)
    z = np.load(RUN / f'L{a.layer}.npz')
    F = z['s1'].shape[0]
    mu_np = (z['s1'] / float(z['n'])).astype(np.float32)

    dev = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.bfloat16, device_map=dev)
    model.eval()
    model.requires_grad_(False)
    dec = model.model.layers
    L = dec[a.layer]
    Wd = L.mlp.down_proj.weight                                    # (d, F) bf16
    mu = torch.tensor(mu_np, device=dev)

    def batch_ids(ws):
        return torch.tensor(np.stack([streams[s][w * seq:(w + 1) * seq] for s, w in ws]).astype(np.int64), device=dev)

    def counted_mask(ids):
        pos = torch.arange(seq, device=dev)[None, :]
        nxt = torch.roll(ids, -1, 1)
        return (pos >= skip) & (pos <= seq - 2) & (ids != sep) & (nxt != sep)

    def token_losses(logits, ids):
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        return -lp.gather(-1, ids[:, 1:, None])[..., 0]          # (B, T-1)：位置 t 预测 t+1

    # ---------- 归因 ----------
    st = {}

    def pre_layer(mod, args, kwargs):
        h = (args[0] if args else kwargs['hidden_states']).detach().requires_grad_(True)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = h
        return args, kwargs

    def pre_down(mod, args):
        A = args[0]
        st['A'] = A
        A.retain_grad()

    def pre_norm(mod, args):
        x = args[0]
        st['x'] = x
        x.retain_grad()

    def gate_hook(mod, i, o):
        st['G'] = o.detach()

    def up_hook(mod, i, o):
        st['V'] = o.detach()
    hs = [L.register_forward_pre_hook(pre_layer, with_kwargs=True), L.mlp.down_proj.register_forward_pre_hook(pre_down),
          model.model.norm.register_forward_pre_hook(pre_norm), L.mlp.gate_proj.register_forward_hook(gate_hook),
          L.mlp.up_proj.register_forward_hook(up_hook)]
    acc = {k: torch.zeros(F, dtype=torch.float64, device=dev) for k in ('total', 'direct', 'magnitude', 'zero_total', 'A_sum')}
    n_counted, n_positions, a_err, base_loss_sum = 0, 0, None, 0.0
    for i in range(0, len(att_w), a.batch):
        ids = batch_ids(att_w[i:i + a.batch])
        m = counted_mask(ids)
        with torch.enable_grad():
            out = model(input_ids=ids)
            tl = token_losses(out.logits, ids)
            mm = m[:, :-1]
            loss = (tl * mm).sum()
            loss.backward()
        A = st['A'].detach().float()                               # (B, T, F)
        gA = st['A'].grad.float()
        gx = st['x'].grad                                          # (B, T, d)
        gD = (gx.to(Wd.dtype) @ Wd).float()                        # (B, T, F)：⟨W_down[:, j], ∂L/∂x^final⟩
        dmu = mu[None, None, :] - A
        c = dmu * gA
        acc['total'] += c.sum((0, 1)).double()
        acc['magnitude'] += c.abs().sum((0, 1)).double()
        acc['direct'] += (dmu * gD).sum((0, 1)).double()
        acc['zero_total'] += (-A * gA).sum((0, 1)).double()
        acc['A_sum'] += (A * m[..., None]).sum((0, 1)).double()
        n_counted += int(mm.sum()); n_positions += int(m.sum()); base_loss_sum += float(loss.detach())
        if a_err is None:
            ref = torch.nn.functional.silu(st['G'].float()) * st['V'].float()
            a_err = float((A - ref).norm() / ref.norm())
        st.clear()
        if (i // a.batch) % 10 == 0:
            print(f'[attr] {i + a.batch}/{len(att_w)} 窗口 · {time.time() - t0:.0f}s · 显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)
    for h in hs:
        h.remove()
    N = float(n_counted)
    total = (acc['total'] / N).cpu().numpy(); direct = (acc['direct'] / N).cpu().numpy()
    mag = (acc['magnitude'] / N).cpu().numpy(); zero_total = (acc['zero_total'] / N).cpu().numpy()
    sample_mean = (acc['A_sum'] / max(n_positions, 1)).cpu().numpy()
    t_attr = time.time() - t0

    # ---------- 真实消融验证 ----------
    order = np.argsort(-np.abs(total))
    top50 = [int(u) for u in order[:50]]
    rest = np.setdiff1d(np.arange(F), top50)
    rand50 = [int(u) for u in np.random.default_rng(0).choice(rest, 50, replace=False)]
    abl = {'units': None}

    def pre_down_abl(mod, args):
        if abl['units'] is None:
            return None
        x = args[0].clone()
        idx = abl['units']
        x[..., idx] = mu[idx].to(x.dtype)
        return (x,)
    hh = L.mlp.down_proj.register_forward_pre_hook(pre_down_abl)
    val_batches = [batch_ids(val_w[i:i + a.val_batch]) for i in range(0, len(val_w), a.val_batch)]

    @torch.inference_mode()
    def val_loss(units):
        abl['units'] = None if units is None else torch.tensor(units, device=dev)
        s, k = 0.0, 0
        for ids in val_batches:
            m = counted_mask(ids)[:, :-1]
            tl = token_losses(model(input_ids=ids).logits, ids)
            s += float((tl * m).sum()); k += int(m.sum())
        return s / k, k
    base, n_val = val_loss(None)
    base2, _ = val_loss(None)
    rows = []
    for j, u in enumerate(top50 + rand50):
        lv, _ = val_loss([u])
        rows.append({'unit': u, 'set': 'top50' if j < 50 else 'random50', 'est_total': float(total[u]), 'est_direct': float(direct[u]),
                     'est_indirect': float(total[u] - direct[u]), 'true_delta': lv - base})
        if j % 10 == 0:
            print(f'[val] {j + 1}/100 · {time.time() - t0:.0f}s', flush=True)
    group, _ = val_loss(top50)
    hh.remove()
    est = np.array([r['est_total'] for r in rows]); tru = np.array([r['true_delta'] for r in rows])
    rho = spearman(est, tru)
    sign = float(np.mean(np.sign(est[:50]) == np.sign(tru[:50])))
    crit = P3['validation']['pass_if']
    passed = rho >= 0.6 and sign >= 0.8
    d_top = np.abs(direct[top50]); i_top = np.abs(total[top50] - direct[top50])
    out = {
        'schema': 'layer_attribution.v1', 'prereg': 'L30-v1',
        'prereg_sha256': PREREG.with_suffix('.sha256').read_text().split()[0],
        'model_key': man['model_key'], 'layer': a.layer,
        'samples': {'attribution_windows': len(att_w), 'attribution_counted_positions': n_counted,
                    'validation_windows': len(val_w), 'validation_counted_positions': n_val},
        'baseline_loss': {'attribution_set': base_loss_sum / N, 'validation_set': base, 'validation_repeat': base2},
        'self_checks': {
            'A_vs_silu_G_V_rel_err': a_err, 'A_pass': a_err < 0.02,
            'cached_residual_recompute': '不适用：验证直接用完整前向（每次消融都重算第 0–35 层），不经缓存残差',
            'validation_repeat_abs_diff': abs(base - base2),
            'mu_vs_sample_mean': {'median_abs_diff': float(np.median(np.abs(mu_np - sample_mean))),
                                  'p90_abs_diff': float(np.quantile(np.abs(mu_np - sample_mean), 0.9)),
                                  'median_abs_mu': float(np.median(np.abs(mu_np)))}},
        'validation': {'criteria': crit, 'spearman_100': rho, 'sign_agreement_top50': sign, 'passed': passed,
                       'verdict': '一阶归因在 L30 可用于排序' if passed else '归因估计未通过验证：只公开 100 个真实消融结果',
                       'group_top50_true_delta': group - base, 'sum_top50_true_delta': float(tru[:50].sum()),
                       'rows': rows},
        'direct_share_top50': {'median': float(np.median(d_top / (d_top + i_top + 1e-12))),
                               'quartiles': [float(np.quantile(d_top / (d_top + i_top + 1e-12), q)) for q in (0.25, 0.75)]},
        'distribution': {k: {q: float(np.quantile(v, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)}
                         for k, v in (('total', total), ('direct', direct), ('magnitude', mag), ('zero_total', zero_total))},
        'units': [{'unit': j, 'total': float(total[j]), 'direct': float(direct[j]), 'indirect': float(total[j] - direct[j]),
                   'magnitude': float(mag[j]), 'zero_total': float(zero_total[j])} for j in range(F)],
        'publish_ranking': passed,
        'seconds': {'attribution': round(t_attr, 1), 'total': round(time.time() - t0, 1)},
        'gpu_peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2),
        'note': '梯度取计入位置损失之和对 A 的梯度（含该位置对之后位置损失的影响），与全部位置均值消融的一阶近似一致；真实消融在全部位置替换为 μ_j。',
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('samples', 'baseline_loss', 'self_checks', 'direct_share_top50', 'gpu_peak_gib', 'seconds')}, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out['validation'].items() if k != 'rows'}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
