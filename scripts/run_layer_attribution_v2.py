# -*- coding: utf-8 -*-
"""L30-v2（预注册 research/neurons/L30-v2.prereg.json）：fp32 重算第 30–35 层的归因、验证（含噪声对照）与条件化消融。

流程：
  1. bf16 原模型前向第 0–29 层，缓存第 30 层的残差输入（CPU，bf16）；同时算完整 bf16 前向的验证集损失。
  2. 等价性自检：用原 bf16 层走手动重算路径，平均损失应与完整前向一致（< 1e-3 nats）。
  3. 把第 30–35 层、最终 RMSNorm、lm_head 复制为 fp32，释放 bf16 模型；之后所有比较都从缓存残差开始用 fp32 重算。
  4. A 部分：fp32 一阶归因（down_proj 输入换成叶子张量，对 30–35 层与 lm_head 反向）。
  5. B 部分：前 50 + 随机 50 个单元的全部位置均值消融；20 个单元 × ±0.001 的噪声对照；前 50 组消融；判定。
  6. C 部分：331 个 T1/T2 单元只在强激活位置（A > 10^-0.5）做均值消融；伪触发对照；Wilcoxon 单侧 + BH。

用法：python scripts/run_layer_attribution_v2.py
"""
from __future__ import annotations

import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_layer_attribution as V1  # noqa: E402  复用窗口选择与路径

PREREG = ROOT / 'research' / 'neurons' / 'L30-v2.prereg.json'
LAYER = 30
STRONG = 10 ** -0.5
OUT_DIR = ROOT / 'research' / 'neurons'


def log(msg, t0):
    print(f'[v2 {time.time() - t0:7.0f}s] {msg}', flush=True)


def bh(p, q=0.05):
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    ok = p[order] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    sig = np.zeros(n, bool)
    sig[order[:k]] = True
    return sig


def main() -> int:
    import torch
    from scipy.stats import wilcoxon
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    dev = 'cuda'
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    man = json.loads((V1.RUN / 'manifest.json').read_text(encoding='utf-8'))
    cdir = V1.NATIVE / '_census' / 'corpus' / man['corpus_fp']
    cman = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    sep, seq, skip = int(cman['separator_id']), int(man['seq_len']), int(man['skip_head'])
    streams = [np.load(cdir / s['file'])['tokens'] for s in cman['sources']]
    n_win = [len(t) // seq for t in streams]
    att_w = V1.pick_windows(n_win, 47, 0, 400)
    val_w = V1.pick_windows(n_win, 47, 23, 204)
    assert val_w[:40] == V1.pick_windows(n_win, 47, 23, 40) and not set(att_w) & set(val_w)
    z = np.load(V1.RUN / f'L{LAYER}.npz')
    F = z['s1'].shape[0]
    mu = torch.tensor(z['s1'] / float(z['n']), dtype=torch.float32, device=dev)
    readout_top = z['readout_top']
    pano = json.loads((OUT_DIR / 'L30-v1-panorama.json').read_text(encoding='utf-8'))
    tt_units = [u['unit'] for u in pano['units'] if u['type'].startswith(('T1', 'T2'))]
    assert len(tt_units) == 331, len(tt_units)

    def ids_of(ws):
        return np.stack([streams[s][w * seq:(w + 1) * seq] for s, w in ws]).astype(np.int64)
    att_ids, val_ids = ids_of(att_w), ids_of(val_w)

    def mask_np(ids):
        pos = np.arange(seq)[None, :]
        nxt = np.roll(ids, -1, 1)
        return ((pos >= skip) & (pos <= seq - 2) & (ids != sep) & (nxt != sep))[:, :-1]   # (B, T-1)：位置 t 预测 t+1
    att_mask, val_mask = mask_np(att_ids), mask_np(val_ids)

    # ---------- 1. bf16：缓存第 30 层残差输入，完整前向损失 ----------
    model = AutoModelForCausalLM.from_pretrained(str(V1.MODEL), dtype=torch.bfloat16, device_map=dev).eval()
    model.requires_grad_(False)
    layers = model.model.layers
    cap = {}

    class _Stop(Exception):
        pass

    def grab(mod, args, kwargs):
        cap['h'] = (args[0] if args else kwargs['hidden_states']).detach()
        if cap.get('stop'):
            raise _Stop()
    hk = layers[LAYER].register_forward_pre_hook(grab, with_kwargs=True)

    def token_losses(logits, ids):
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        return -lp.gather(-1, ids[:, 1:, None])[..., 0], lp

    @torch.inference_mode()
    def cache_residuals(ids_np, bs=8):
        out = []
        cap['stop'] = True
        for i in range(0, len(ids_np), bs):
            try:
                model(input_ids=torch.tensor(ids_np[i:i + bs], device=dev))
            except _Stop:
                pass
            out.append(cap['h'].cpu())
        cap['stop'] = False
        return torch.cat(out)
    res_att = cache_residuals(att_ids)
    res_val = cache_residuals(val_ids)
    log(f'残差缓存 att {tuple(res_att.shape)} val {tuple(res_val.shape)}', t0)

    @torch.inference_mode()
    def full_bf16_loss(ids_np, m_np, bs=8):
        s = k = 0.0
        for i in range(0, len(ids_np), bs):
            ids = torch.tensor(ids_np[i:i + bs], device=dev)
            tl, _ = token_losses(model(input_ids=ids).logits, ids)
            m = torch.tensor(m_np[i:i + bs], device=dev)
            s += float((tl * m).sum()); k += float(m.sum())
        return s / k
    base_full_bf16 = full_bf16_loss(val_ids, val_mask)
    hk.remove()
    rotary = model.model.rotary_emb
    pos_ids = torch.arange(seq, device=dev)[None]

    def run_from(res, mods, dtype):
        """从第 30 层残差输入重算到 logits（mods = (layers, norm, lm_head)）。"""
        h = res.to(dev, dtype)
        pe = rotary(h, pos_ids)
        for L in mods[0]:
            h = L(h, attention_mask=None, position_embeddings=pe, position_ids=pos_ids)
        x = h
        return mods[2](mods[1](x)), x

    @torch.inference_mode()
    def path_loss(res_all, ids_np, m_np, mods, dtype, bs=4):
        s = k = 0.0
        for i in range(0, len(ids_np), bs):
            ids = torch.tensor(ids_np[i:i + bs], device=dev)
            logits, _ = run_from(res_all[i:i + bs], mods, dtype)
            tl, _ = token_losses(logits, ids)
            m = torch.tensor(m_np[i:i + bs], device=dev)
            s += float((tl * m).sum()); k += float(m.sum())
        return s / k
    bf16_mods = (list(layers[LAYER:]), model.model.norm, model.lm_head)
    base_manual_bf16 = path_loss(res_val, val_ids, val_mask, bf16_mods, torch.bfloat16)
    log(f'等价性：完整 bf16 {base_full_bf16:.6f} · 手动 bf16 {base_manual_bf16:.6f}', t0)

    # ---------- 3. fp32 副本，释放 bf16 ----------
    L32 = [copy.deepcopy(L).float() for L in layers[LAYER:]]
    norm32 = copy.deepcopy(model.model.norm).float()
    head32 = torch.nn.Linear(model.lm_head.in_features, model.lm_head.out_features, bias=False, device=dev, dtype=torch.float32)
    head32.weight.data.copy_(model.lm_head.weight.data.float())
    rotary = copy.deepcopy(rotary)
    del model, layers, bf16_mods
    gc.collect(); torch.cuda.empty_cache()
    for mdl in L32 + [norm32, head32]:
        mdl.requires_grad_(False)
    mods32 = (L32, norm32, head32)
    Wd = L32[0].mlp.down_proj.weight                                # (d, F) fp32
    log(f'fp32 副本就绪 · 显存 {torch.cuda.memory_allocated() / 2**30:.1f} GiB', t0)

    # down_proj 钩子：mode 决定行为
    st = {'mode': None}

    def pre_down(mod, args):
        A = args[0]
        m = st['mode']
        if m is None:
            return None
        kind = m[0]
        if kind == 'leaf':                          # 归因：叶子张量
            leaf = A.detach().requires_grad_(True)
            st['A'] = leaf
            return (leaf,)
        if kind == 'capture':                       # 记录指定单元的 A
            st['A_cap'] = A[..., m[1]].detach().clone()
            return None
        y = A.clone()
        if kind == 'clone_same':
            y[..., m[1]] = A[..., m[1]]
        elif kind == 'mean':
            y[..., m[1]] = mu[m[1]]
        elif kind == 'tiny':
            y[..., m[1]] = y[..., m[1]] + m[2]
        elif kind == 'masked_mean':                 # m[1] 单元、m[2] (B,T) 布尔位置
            col = y[..., m[1]]
            y[..., m[1]] = torch.where(m[2], mu[m[1]], col)
        return (y,)

    def gate_up(mod, i, o):
        st.setdefault('GV', []).append(o.detach())
    L32[0].mlp.down_proj.register_forward_pre_hook(pre_down)
    h_g = L32[0].mlp.gate_proj.register_forward_hook(gate_up)
    h_u = L32[0].mlp.up_proj.register_forward_hook(gate_up)

    def pre_norm(mod, args):
        if st['mode'] is not None and st['mode'][0] == 'leaf':
            args[0].retain_grad()
            st['x'] = args[0]
    norm32.register_forward_pre_hook(pre_norm)

    # 自检：fp32 基线、A 误差、空操作
    st['mode'] = ('capture', slice(None))
    with torch.inference_mode():
        run_from(res_val[:2], mods32, torch.float32)
    g, v = st['GV'][0].float(), st['GV'][1].float()
    a_err = float((st['A_cap'] - torch.nn.functional.silu(g) * v).norm() / st['A_cap'].norm())
    h_g.remove(); h_u.remove(); st.pop('GV', None); st.pop('A_cap', None)
    st['mode'] = None
    base32 = path_loss(res_val, val_ids, val_mask, mods32, torch.float32)
    st['mode'] = ('clone_same', 5000)
    noop = path_loss(res_val, val_ids, val_mask, mods32, torch.float32) - base32
    st['mode'] = None
    log(f'fp32 基线 {base32:.6f} · A 相对误差 {a_err:.2e} · 空操作 {noop:.2e}', t0)

    # ---------- 4. A 部分：fp32 归因 ----------
    acc = {k: torch.zeros(F, dtype=torch.float64, device=dev) for k in ('total', 'direct', 'magnitude', 'zero_total')}
    n_att = 0
    for i in range(0, len(att_ids), 2):
        ids = torch.tensor(att_ids[i:i + 2], device=dev)
        m = torch.tensor(att_mask[i:i + 2], device=dev)
        st['mode'] = ('leaf',)
        with torch.enable_grad():
            logits, _ = run_from(res_att[i:i + 2], mods32, torch.float32)
            tl, _ = token_losses(logits, ids)
            (tl * m).sum().backward()
        A = st['A'].detach(); gA = st['A'].grad
        gD = st['x'].grad @ Wd
        dmu = mu[None, None, :] - A
        c = dmu * gA
        acc['total'] += c.sum((0, 1)).double(); acc['magnitude'] += c.abs().sum((0, 1)).double()
        acc['direct'] += (dmu * gD).sum((0, 1)).double(); acc['zero_total'] += (-A * gA).sum((0, 1)).double()
        n_att += int(m.sum())
        st['mode'] = None; st.pop('A', None); st.pop('x', None)
        del logits, tl, A, gA, gD, c, dmu
        if i % 80 == 0:
            log(f'归因 {i + 2}/{len(att_ids)} · 峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', t0)
    est = {k: (v / n_att).cpu().numpy() for k, v in acc.items()}
    total = est['total']
    v1 = json.loads((OUT_DIR / 'L30-v1-attribution.json').read_text(encoding='utf-8'))
    v1_total = np.array([u['total'] for u in v1['units']])
    top_v1 = set(np.argsort(-np.abs(v1_total))[:50].tolist())

    # ---------- 5. B 部分：验证 ----------
    order = np.argsort(-np.abs(total))
    top50 = [int(u) for u in order[:50]]
    rest = np.setdiff1d(np.arange(F), top50)
    rand50 = [int(u) for u in np.random.default_rng(0).choice(rest, 50, replace=False)]

    def val_delta(mode):
        st['mode'] = mode
        d = path_loss(res_val, val_ids, val_mask, mods32, torch.float32) - base32
        st['mode'] = None
        return d
    rows = []
    for j, u in enumerate(top50 + rand50):
        rows.append({'unit': u, 'set': 'top50' if j < 50 else 'random50', 'est_total': float(total[u]),
                     'est_direct': float(est['direct'][u]), 'true_delta': val_delta(('mean', u))})
        if j % 10 == 0:
            log(f'验证 {j + 1}/100', t0)
    pool = np.setdiff1d(np.arange(F), top50 + rand50)
    ctrl_units = [int(u) for u in np.random.default_rng(1).choice(pool, 20, replace=False)]
    noise = [{'unit': u, 'eps': e, 'delta': val_delta(('tiny', u, e))} for u in ctrl_units for e in (1e-3, -1e-3)]
    nv = np.array([x['delta'] for x in noise])
    m0, s0 = float(nv.mean()), float(nv.std(ddof=1))
    group = val_delta(('mean', torch.tensor(top50, device=dev)))
    log(f'噪声底 m0 {m0:.2e} s0 {s0:.2e} · 组消融 {group:.2e}', t0)
    e_ = np.array([r['est_total'] for r in rows]); t_ = np.array([r['true_delta'] for r in rows])
    zv = (t_ - m0) / s0
    for r, zz in zip(rows, zv):
        r['z'] = float(zz)
    gate_ok = s0 <= 0.25 * float(np.median(np.abs(t_[:50])))
    rho = V1.spearman(e_, t_)
    sel = np.abs(zv[:50]) > 2
    sign_basis = 'top50_|z|>2' if sel.sum() >= 20 else 'top50_all'
    idx = np.where(sel)[0] if sel.sum() >= 20 else np.arange(50)
    sign = float(np.mean(np.sign(e_[idx]) == np.sign(t_[idx])))
    passed = bool(gate_ok and rho >= 0.6 and sign >= 0.8)
    verdict = ('不判定（测量分辨力不足）' if not gate_ok else
               '一阶归因在 L30 可用于排序' if passed else '未通过：只公开 100 个真实消融结果')
    d_top = np.abs(est['direct'][top50]); i_top = np.abs(total[top50] - est['direct'][top50])
    out_a = {
        'schema': 'layer_attribution.v2', 'prereg': 'L30-v2', 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().split()[0],
        'model_key': man['model_key'], 'layer': LAYER,
        'samples': {'attribution_windows': len(att_w), 'attribution_counted_positions': n_att,
                    'validation_windows': len(val_w), 'validation_counted_positions': int(val_mask.sum())},
        'self_checks': {'full_bf16_loss': base_full_bf16, 'manual_bf16_loss': base_manual_bf16,
                        'equivalence_abs_diff': abs(base_full_bf16 - base_manual_bf16),
                        'equivalence_pass': abs(base_full_bf16 - base_manual_bf16) < 1e-3,
                        'fp32_loss': base32, 'fp32_vs_bf16_abs_diff': abs(base32 - base_full_bf16),
                        'fp32_vs_bf16_pass': abs(base32 - base_full_bf16) < 0.01,
                        'A_rel_err': a_err, 'A_pass': a_err < 1e-4, 'noop_delta': noop, 'noop_pass': noop == 0.0},
        'noise_control': {'units': ctrl_units, 'm0': m0, 's0': s0, 'rows': noise},
        'validation': {'criteria': pre['part_b_validation']['pass_if'], 'validity_gate': pre['part_b_validation']['validity_gate'],
                       'gate_ok': bool(gate_ok), 'median_abs_true_top50': float(np.median(np.abs(t_[:50]))),
                       'spearman_100': rho, 'sign_basis': sign_basis, 'n_top50_z_gt2': int(sel.sum()), 'sign_agreement': sign,
                       'passed': passed, 'verdict': verdict, 'group_top50_true_delta': group,
                       'sum_top50_true_delta': float(t_[:50].sum()), 'rows': rows},
        'comparison_with_v1': {'spearman_all_units': V1.spearman(total, v1_total),
                               'top50_overlap': len(top_v1 & set(top50))},
        'direct_share_top50': {'median': float(np.median(d_top / (d_top + i_top + 1e-12)))},
        'publish_ranking': passed,
        'units': [{'unit': j, 'total': float(total[j]), 'direct': float(est['direct'][j]), 'indirect': float(total[j] - est['direct'][j]),
                   'magnitude': float(est['magnitude'][j]), 'zero_total': float(est['zero_total'][j])} for j in range(F)],
    }
    (OUT_DIR / 'L30-v2-attribution.json').write_text(json.dumps(out_a, ensure_ascii=False, indent=1), encoding='utf-8')
    log(f'B 判定：{verdict} · Spearman {rho:.3f} · 符号 {sign:.2f}（{sign_basis}）· 门 {gate_ok}', t0)

    # ---------- 6. C 部分：条件化消融 ----------
    tt = torch.tensor(tt_units, device=dev)
    capA = []
    for i in range(0, len(val_ids), 4):
        st['mode'] = ('capture', tt)
        with torch.inference_mode():
            run_from(res_val[i:i + 4], mods32, torch.float32)
        capA.append(st['A_cap'].cpu())
    st['mode'] = None
    capA = torch.cat(capA).numpy()                                     # (W, T, 331)
    union = np.unique(readout_top[tt_units].ravel())
    upos = {int(t): k for k, t in enumerate(union)}
    base_tl, base_lp = [], []
    with torch.inference_mode():
        for i in range(0, len(val_ids), 4):
            ids = torch.tensor(val_ids[i:i + 4], device=dev)
            logits, _ = run_from(res_val[i:i + 4], mods32, torch.float32)
            tl, lp = token_losses(logits, ids)
            base_tl.append(tl.cpu()); base_lp.append(lp[..., torch.tensor(union, device=dev)].half().cpu())
            del logits, lp
    base_tl = torch.cat(base_tl).numpy(); base_lp = torch.cat(base_lp)  # (W, T-1) / (W, T-1, |U|)
    N_val = float(val_mask.sum())
    rng2 = np.random.default_rng(2)

    def cond_run(u_col, fire):
        """fire：(W, T) 布尔，消融位置。返回触发计入位置的 δ、全局平均差、读出 Δlogp。"""
        wins = np.where(fire.any(1))[0]
        deltas, gsum, dlp = [], 0.0, []
        unit = tt_units[u_col]
        prom = torch.tensor([upos[int(t)] for t in readout_top[unit]])
        for k0 in range(0, len(wins), 4):
            wb = wins[k0:k0 + 4]
            ids = torch.tensor(val_ids[wb], device=dev)
            st['mode'] = ('masked_mean', unit, torch.tensor(fire[wb], device=dev))
            with torch.inference_mode():
                logits, _ = run_from(res_val[wb], mods32, torch.float32)
                tl, lp = token_losses(logits, ids)
            st['mode'] = None
            tl = tl.cpu().numpy()
            m = val_mask[wb]
            gsum += float(((tl - base_tl[wb]) * m).sum())
            f = fire[wb][:, :-1] & m                                    # 触发且计入（位置 s 预测 s+1）
            deltas.append((tl - base_tl[wb])[f])
            lp_sel = lp[..., torch.tensor(union[prom.numpy()], device=dev)].cpu()
            diff = (lp_sel.float() - base_lp[wb][..., prom].float()).numpy()
            dlp.append(diff[f].mean(1) if f.any() else np.zeros(0))
            del logits, lp
        d = np.concatenate(deltas) if deltas else np.zeros(0)
        dl = np.concatenate(dlp) if dlp else np.zeros(0)
        return d, gsum / N_val, dl
    rows_c = []
    for c, unit in enumerate(tt_units):
        a = capA[..., c]                                                # (W, T)
        fire = a > STRONG
        n_fire = int((fire[:, :-1] & val_mask).sum())
        rec = {'unit': unit, 'type': pano['units'][unit]['type'], 'n_fire': n_fire, 'evaluable': n_fire >= 20}
        if rec['evaluable']:
            d, g, dl = cond_run(c, fire)
            p = float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0
            cand = np.argwhere((a[:, :-1] <= 0.1) & val_mask)
            pick = cand[rng2.choice(len(cand), min(n_fire, len(cand)), replace=False)]
            pf = np.zeros_like(fire)
            pf[pick[:, 0], pick[:, 1]] = True
            dp, gp, _ = cond_run(c, pf)
            pp = float(wilcoxon(dp, alternative='greater').pvalue) if np.any(dp != 0) else 1.0
            rec.update({'local_mean': float(d.mean()), 'local_median': float(np.median(d)), 'p_local': p, 'global': g,
                        'dlogp_promote': float(dl.mean()) if len(dl) else None,
                        'pseudo_local_mean': float(dp.mean()), 'p_pseudo': pp, 'pseudo_global': gp})
        rows_c.append(rec)
        if c % 20 == 0:
            log(f'条件化 {c + 1}/331', t0)
    ev = [r for r in rows_c if r['evaluable']]
    sig = bh([r['p_local'] for r in ev]); sigp = bh([r['p_pseudo'] for r in ev])
    for r, s1, s2 in zip(ev, sig, sigp):
        r['sig_local'] = bool(s1); r['sig_pseudo'] = bool(s2)
    frac = float(sig.mean()) if len(ev) else 0.0
    frac_p = float(sigp.mean()) if len(ev) else 0.0
    ok_c = bool(len(ev) and frac >= 0.5 and frac > 2 * frac_p)
    dl_all = np.array([r['dlogp_promote'] for r in ev if r['dlogp_promote'] is not None])
    out_c = {'schema': 'layer_conditional.v1', 'prereg': 'L30-v2', 'layer': LAYER, 'n_units': len(tt_units), 'n_evaluable': len(ev),
             'frac_sig_local': frac, 'frac_sig_pseudo': frac_p, 'verdict': 'T1/T2 单元在其触发位置被因果使用' if ok_c else '未支持',
             'readout_consistency': {'frac_dlogp_negative': float((dl_all < 0).mean()) if len(dl_all) else None,
                                     'median_dlogp': float(np.median(dl_all)) if len(dl_all) else None},
             'criteria': pre['part_c_conditional']['verdict'], 'rows': rows_c,
             'seconds_total': round(time.time() - t0, 1), 'gpu_peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (OUT_DIR / 'L30-v2-conditional.json').write_text(json.dumps(out_c, ensure_ascii=False, indent=1), encoding='utf-8')
    log(f'C 判定：{out_c["verdict"]} · 可评估 {len(ev)} · 显著 {frac:.2f} · 伪触发 {frac_p:.2f}', t0)
    print(json.dumps({'self_checks': out_a['self_checks'], 'noise': {'m0': m0, 's0': s0},
                      'validation': {k: v for k, v in out_a['validation'].items() if k != 'rows'},
                      'comparison_with_v1': out_a['comparison_with_v1'], 'direct_share_top50': out_a['direct_share_top50'],
                      'conditional': {k: v for k, v in out_c.items() if k != 'rows'}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
