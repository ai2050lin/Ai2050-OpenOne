# -*- coding: utf-8 -*-
"""L30-v3（预注册 research/neurons/L30-v3.prereg.json）：T1/T2 单元在短语触发点的均值消融（fp32），
主检验只看下一 token 属于读出的触发位置，符号翻转置换检验 + BH，替换单元对照。

用法：python scripts/run_layer_trigger_v3.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import layer_fp32 as LF  # noqa: E402
import run_layer_attribution as V1  # noqa: E402
from shared.dist_protocol import census as C  # noqa: E402
from run_layer_attribution_v2 import bh  # noqa: E402

PREREG = ROOT / 'research' / 'neurons' / 'L30-v3.prereg.json'
OUT = ROOT / 'research' / 'neurons' / 'L30-v3-trigger.json'
LAYER = 30
N_PERM = 10_000


def main() -> int:
    import torch
    t0 = time.time()

    def log(msg):
        print(f'[v3 {time.time() - t0:7.0f}s] {msg}', flush=True)
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    cor = LF.Corpus()
    ws = cor.windows(35, 800)
    assert not set(ws) & set(cor.windows(0, 400)) and not set(ws) & set(cor.windows(23, 204))
    ids = cor.ids(ws)
    mask = cor.mask(ids)                                           # (W, T−1)
    z = np.load(V1.RUN / f'L{LAYER}.npz')
    F = z['s1'].shape[0]
    mu_np = (z['s1'] / float(z['n'])).astype(np.float32)
    pano = json.loads((ROOT / 'research' / 'neurons' / 'L30-v1-panorama.json').read_text(encoding='utf-8'))
    units = [u['unit'] for u in pano['units'] if u['type'].startswith(('T1', 'T2'))]
    assert len(units) == 331

    def mode_id(idx, prev):
        src, off = C.split_index(idx)
        toks = [int(cor.streams[s][o - 1 if prev else o]) for s, o in zip(src, off) if o > 0]
        return Counter(toks).most_common(1)[0][0]
    meta = {}
    for u in units:
        t1 = pano['units'][u]['type'].startswith('T1')
        meta[u] = {'type': 'T1' if t1 else 'T2', 'mode_token_id': mode_id(z['top_idx'][u], prev=not t1),
                   'q99': LF.q_from_hist(z['hist'][u], 0.99), 'readout': [int(t) for t in z['readout_top'][u]]}
    non_tt = np.setdiff1d(np.arange(F), units)
    ctrl = {u: int(c) for u, c in zip(units, np.random.default_rng(4).choice(non_tt, len(units), replace=False))}

    S = LF.Fp32Stack(LAYER)
    res = S.cache_residuals(ids)
    log(f'残差缓存 {tuple(res.shape)}')
    chk = slice(0, 100)
    full_bf16 = S.full_bf16_loss(ids[chk], mask[chk])
    man_bf16 = S.manual_loss(res[chk], ids[chk], mask[chk], S.bf16_mods(), torch.bfloat16)
    S.to_fp32()
    base32_chk = S.manual_loss(res[chk], ids[chk], mask[chk], S.mods, torch.float32)
    S.mode = ('clone_same', 5000)
    noop = S.manual_loss(res[chk], ids[chk], mask[chk], S.mods, torch.float32) - base32_chk
    S.mode = None
    log(f'自检（前 100 窗口）：完整 bf16 {full_bf16:.6f} · 手动 bf16 {man_bf16:.6f} · fp32 {base32_chk:.6f} · 空操作 {noop:.1e}')

    # 激活与基线（fp32）
    ut = torch.tensor(units, device='cuda')
    capA, base_tl = [], []
    union = sorted({t for u in units for t in meta[u]['readout']})
    upos = {t: k for k, t in enumerate(union)}
    union_t = torch.tensor(union, device='cuda')
    base_lp = []
    for i in range(0, len(ids), 4):
        S.mode = ('capture', ut)
        with torch.inference_mode():
            logits = S.run_from(res[i:i + 4], S.mods, torch.float32)
            tl, lp = LF.token_losses(torch, logits, torch.tensor(ids[i:i + 4], device='cuda'))
        capA.append(S.cap.cpu()); base_tl.append(tl.cpu()); base_lp.append(lp[..., union_t].half().cpu())
        del logits, lp
    S.mode = None
    capA = torch.cat(capA).numpy(); base_tl = torch.cat(base_tl).numpy(); base_lp = torch.cat(base_lp)
    log(f'激活与基线完成 · 显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')

    def ablate(unit, pos_mask):
        """只在 pos_mask(W,T) 的位置把 unit 换成均值；返回计入位置 δ (W,T−1) 稀疏结果与读出 Δlogp。"""
        wins = np.where(pos_mask.any(1))[0]
        delta = np.zeros((len(ids), cor.seq - 1), dtype=np.float64)
        dlp = np.full((len(ids), cor.seq - 1), np.nan)
        val = torch.tensor(float(mu_np[unit]), device='cuda')
        for k in range(0, len(wins), 4):
            wb = wins[k:k + 4]
            S.mode = ('masked_set', unit, torch.tensor(pos_mask[wb], device='cuda'), val)
            with torch.inference_mode():
                logits = S.run_from(res[wb], S.mods, torch.float32)
                tl, lp = LF.token_losses(torch, logits, torch.tensor(ids[wb], device='cuda'))
            S.mode = None
            delta[wb] = tl.cpu().numpy() - base_tl[wb]
            if unit in meta:
                pr = torch.tensor([upos[t] for t in meta[unit]['readout']])
                cur = lp[..., union_t[pr]].cpu().float()
                dlp[wb] = (cur - base_lp[wb][..., pr].float()).mean(-1).numpy()
            del logits, lp
        return delta, dlp

    rng = np.random.default_rng(3)
    rows = []
    for c, u in enumerate(units):
        m = meta[u]
        a = capA[..., c]                                            # (W, T)
        tok_hit = (ids == m['mode_token_id']) if m['type'] == 'T1' else np.roll(ids == m['mode_token_id'], 1, 1)
        if m['type'] == 'T2':
            tok_hit[:, 0] = False
        trig = tok_hit & (a >= m['q99'])                            # (W, T)
        trig[:, :-1] &= mask                                        # 只取计入位置（预注册 triggers.definition）
        trig[:, -1] = False
        trig_c = trig[:, :-1]
        nxt = ids[:, 1:]
        R = trig_c & np.isin(nxt, m['readout'])
        rec = {'unit': u, **{k: m[k] for k in ('type', 'mode_token_id', 'q99')}, 'n_trigger': int(trig_c.sum()), 'n_R': int(R.sum()),
               'eval_primary': bool(R.sum() >= 10), 'eval_secondary': bool(trig_c.sum() >= 20), 'control_unit': ctrl[u]}
        if rec['eval_primary'] or rec['eval_secondary']:
            d, dl = ablate(u, trig)
            dc, _ = ablate(ctrl[u], trig)
            rec['dlogp_promote'] = float(np.nanmean(dl[trig_c])) if trig_c.any() else None
            if rec['eval_secondary']:
                x, xc = d[trig_c], dc[trig_c]
                rec.update({'d_T': float(x.mean()), 'p_T': LF.sign_flip_p(x, N_PERM, rng),
                            'ctrl_d_T': float(xc.mean()), 'ctrl_p_T': LF.sign_flip_p(xc, N_PERM, rng)})
            if rec['eval_primary']:
                x, xc = d[R], dc[R]
                rec.update({'d_R': float(x.mean()), 'p_R': LF.sign_flip_p(x, N_PERM, rng),
                            'ctrl_d_R': float(xc.mean()), 'ctrl_p_R': LF.sign_flip_p(xc, N_PERM, rng)})
        rows.append(rec)
        if c % 25 == 0:
            log(f'单元 {c + 1}/331 · 触发 {rec["n_trigger"]} · R {rec["n_R"]}')

    def verdict(key_p, key_cp, ev_key):
        ev = [r for r in rows if r[ev_key]]
        if not ev:
            return {'n_evaluable': 0, 'frac_sig': None, 'frac_sig_control': None}, ev
        sig = bh([r[key_p] for r in ev]); sigc = bh([r[key_cp] for r in ev])
        for r, s1, s2 in zip(ev, sig, sigc):
            r['sig_' + key_p] = bool(s1); r['sig_' + key_cp] = bool(s2)
        return {'n_evaluable': len(ev), 'frac_sig': float(sig.mean()), 'frac_sig_control': float(sigc.mean())}, ev
    prim, evp = verdict('p_R', 'ctrl_p_R', 'eval_primary')
    sec, evs = verdict('p_T', 'ctrl_p_T', 'eval_secondary')

    def judge(v, min_n):
        if v['n_evaluable'] < min_n:
            return '不判定（样本不足）'
        return '支持' if (v['frac_sig'] >= 0.5 and v['frac_sig'] > 2 * v['frac_sig_control']) else '未支持'
    prim['verdict'] = judge(prim, 20)
    prim['statement'] = 'T1/T2 单元在短语触发点、当读出就是正确答案时被因果使用'
    sec['verdict'] = judge(sec, 1)
    sec['statement'] = 'T1/T2 单元在其触发点被因果使用'
    dl = np.array([r['dlogp_promote'] for r in rows if r.get('dlogp_promote') is not None])
    out = {'schema': 'layer_trigger.v1', 'prereg': 'L30-v3', 'prereg_sha256': PREREG.with_suffix('.sha256').read_text().split()[0],
           'layer': LAYER, 'windows': len(ws), 'counted_positions': int(mask.sum()),
           'self_checks': {'full_bf16_loss_100w': full_bf16, 'manual_bf16_loss_100w': man_bf16,
                           'equivalence_abs_diff': abs(full_bf16 - man_bf16), 'equivalence_pass': abs(full_bf16 - man_bf16) < 1e-3,
                           'fp32_loss_100w': base32_chk, 'fp32_vs_bf16_abs_diff': abs(base32_chk - full_bf16),
                           'fp32_vs_bf16_pass': abs(base32_chk - full_bf16) < 0.01, 'noop_delta': noop, 'noop_pass': noop == 0.0,
                           'note': '自检在前 100 个窗口上计算'},
           'primary': prim, 'secondary': sec,
           'descriptive': {'n_units_with_triggers': int(sum(r['n_trigger'] > 0 for r in rows)),
                           'n_trigger_median': float(np.median([r['n_trigger'] for r in rows])),
                           'n_R_median': float(np.median([r['n_R'] for r in rows])),
                           'frac_dlogp_negative': float((dl < 0).mean()) if len(dl) else None,
                           'median_dlogp': float(np.median(dl)) if len(dl) else None},
           'criteria': pre['verdicts'], 'rows': rows, 'seconds': round(time.time() - t0, 1),
           'gpu_peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    log(f'主检验：{prim} · 第二检验：{sec}')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
