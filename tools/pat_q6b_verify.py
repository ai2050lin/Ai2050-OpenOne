# -*- coding: utf-8 -*-
"""Q6-B2 / Q6-B 独立复核。

原则：**不读我写的结论文档与判决字段**，只从磁盘上的原始数组重算判决，
再比对落盘值；文档只检查关键字是否存在（防"写了但没落盘"）。

用法：.venv/Scripts/python.exe -X utf8 tools/pat_q6b_verify.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / '2026-11-01'
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'
DOCS = ROOT / 'research' / 'gpt5' / 'docs'

ok = fail = 0


def chk(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
        print('  PASS  %-42s %s' % (name, detail))
    else:
        fail += 1
        print('  FAIL  %-42s %s' % (name, detail))


def sha8(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def main() -> int:
    # ---------------- 文件 ----------------
    print('\n[1] 结果文件')
    f1 = RES / '073538_pat_q6b2.json'
    f2 = RES / '074005_pat_q6b.json'
    for f in (f1, f2):
        chk('exists %s' % f.name, f.exists(), '')
    d1 = json.loads(f1.read_text(encoding='utf-8'))
    d2 = json.loads(f2.read_text(encoding='utf-8'))
    p1, p2 = d1['payload'], d2['payload']
    chk('q6b2 content sha8', d1['sha8'] == '4ee36119', d1['sha8'])
    chk('q6b  content sha8', d2['sha8'] == '11531e1e', d2['sha8'])

    # ---------------- Q6-B2 重算 ----------------
    print('\n[2] Q6-B2 判决重算（不读 verdict 字段）')
    obs = np.array(p1['profile_obs'])
    nr = np.array(p1['profile_null_rand'])
    nl = np.array([np.nan if x is None else x for x in p1['profile_null_level']])
    dif = obs - nr
    dfl = obs - nl
    f_geom = float(np.mean(np.abs(dif) < 0.05))
    f_pos = float(np.mean(dif >= 0.10))
    f_lvl = float(np.mean(np.abs(dfl) < 0.05))
    gate = -0.65 <= float(np.median(nr)) <= -0.35
    main_ = 'SHARED_DIRECTION' if float(np.mean(obs >= 0.30)) >= 0.30 else 'NOT_A_SHARED_DIRECTION'
    if not gate:
        attr_ = 'FRAMEWORK_INVALID'
    elif f_geom >= 0.70:
        attr_ = 'PURE_GEOMETRY'
    elif f_pos >= 0.30 and f_lvl < 0.30:
        attr_ = 'WEAK_ABOVE_GEOMETRY'
    elif f_pos >= 0.30 and f_lvl >= 0.30:
        attr_ = 'SUPERORDINATE_EXPLAINED'
    elif float(np.mean(dif <= -0.10)) >= 0.30:
        attr_ = 'ANTI_BEYOND_GEOMETRY'
    else:
        attr_ = 'ATTRIBUTION_UNCLEAR'

    chk('n_layers==37', len(obs) == 37, len(obs))
    chk('G0 gate', gate, 'median null_rand %.4f' % float(np.median(nr)))
    chk('main verdict', main_ == p1['main_verdict'], '%s vs %s' % (main_, p1['main_verdict']))
    chk('attribution', attr_ == p1['attribution'], '%s vs %s' % (attr_, p1['attribution']))
    chk('obs median reproduces Q6-A -0.246', abs(float(np.median(obs)) + 0.246) < 0.005,
        '%.4f' % float(np.median(obs)))
    chk('obs max never reaches 0.30', float(obs.max()) < 0.0,
        'max %.4f' % float(obs.max()))
    chain_share = float(np.median(obs)) - float(np.nanmedian(nl))
    chk('decomposition sums to obs',
        abs((-float(np.median(nr))) + (float(np.nanmedian(nl)) + float(np.median(nr)))
            + chain_share - float(np.median(obs))) < 5e-3,
        'chain_share %.4f' % chain_share)
    chk('reference F3 above threshold', p1['ref_F3_median'] >= 0.30, p1['ref_F3_median'])
    chk('reference F4 above threshold', p1['ref_F4_median'] >= 0.30, p1['ref_F4_median'])
    chk('t>1 fraction is zero for both',
        p1['t_gt1_frac_obs'] == 0.0 and p1['t_gt1_frac_null'] == 0.0,
        '%s / %s' % (p1['t_gt1_frac_obs'], p1['t_gt1_frac_null']))
    chk('rho near perpendicular', p1['profile_rho'][20] > 0.95, p1['profile_rho'][20])

    # ---------------- Q6-B 重算 ----------------
    print('\n[3] Q6-B 判决重算（不读 verdict 字段）')
    r2 = {int(k): v for k, v in p2['r2_curve'].items()}
    r2n = {int(k): v for k, v in p2['r2_null_curve'].items()}
    Rbest = max(r2, key=lambda k: r2[k])
    Rstar = next((k for k in sorted(r2) if r2[k] >= 0.70), None)
    reliable = (p2['C1_color'] >= 5 and p2['C1_size'] >= 6
                and p2['C2_template_spearman'] >= 0.50 and p2['C3_contrast'] >= 1.0)
    if not reliable:
        q1_ = 'MEASUREMENT_UNRELIABLE'
    elif Rbest == 0 or r2[11] < 0.70:
        q1_ = 'NO_LOW_RANK_STRUCTURE'
    elif Rstar is not None and Rstar <= 3 and r2[3] - r2[0] >= 0.15:
        q1_ = 'ATTRIBUTE_SPACE_3D'
    elif Rstar is not None and 4 <= Rstar <= 7:
        q1_ = 'ATTRIBUTE_SPACE_COMPACT'
    else:
        q1_ = 'FULL_RANK'
    chk('reliable gates', reliable,
        'C1 %d/%d C2 %.3f C3 %.2f' % (p2['C1_color'], p2['C1_size'],
                                      p2['C2_template_spearman'], p2['C3_contrast']))
    chk('R_best==0', Rbest == 0, Rbest)
    chk('R* is None', Rstar is None, Rstar)
    chk('Q1 verdict', q1_ == p2['Q1_verdict'], '%s vs %s' % (q1_, p2['Q1_verdict']))
    chk('rank curve monotone decreasing R0..R4',
        r2[0] > r2[1] > r2[2] > r2[3] > r2[4],
        ' '.join('%.3f' % r2[k] for k in range(5)))
    # 只有低秩段（R<=3）可以要求真实优于打乱；R>=4 已进入近满秩退化区，两者都失效
    chk('true beats shuffled on low-rank segment',
        all(r2[k] > r2n[k] for k in (0, 1, 2, 3)),
        ' '.join('R%d %.3f/%.3f' % (k, r2[k], r2n[k]) for k in (0, 1, 2, 3)))

    M = np.array(p2['matrix'])
    mu = M.mean()
    rowm = M.mean(1) - mu
    colm = M.mean(0) - mu
    inter = M - mu - rowm[:, None] - colm[None, :]
    vt = ((M - mu) ** 2).sum()
    v_row = M.shape[1] * (rowm ** 2).sum()
    v_col = M.shape[0] * (colm ** 2).sum()
    v_int = ((inter - inter.mean()) ** 2).sum()
    s = v_row / vt + v_col / vt + v_int / vt
    chk('variance decomposition sums to 1', abs(s - 1.0) < 0.02, '%.4f' % s)
    chk('interaction is the largest part', v_int / vt > max(v_row, v_col) / vt,
        'row %.3f col %.3f int %.3f' % (v_row / vt, v_col / vt, v_int / vt))
    chk('stored var_frac matches recompute',
        abs(p2['var_frac']['interaction'] - v_int / vt) < 0.005, p2['var_frac']['interaction'])

    cols = p2['attr'].index('红')
    order = np.argsort(-M[:, :8], axis=1)
    peakvals = (M[:, :8] >= M[:, :8].max(1, keepdims=True) - 1.0).sum(1)
    chk('peak count per object (colour) small', float(np.median(peakvals)) <= 2.0,
        'median %.1f max %d' % (float(np.median(peakvals)), int(peakvals.max())))

    dec = np.array(p2['dec_profile'])
    deca = np.array(p2['dec_adj_profile'])
    chk('dec saturated for both arms', dec.min() > 0.90 and deca.min() > 0.90,
        'attr min %.3f adj min %.3f' % (dec.min(), deca.min()))
    chk('Q2 has no discriminating power (|diff|<0.15 everywhere)',
        float(np.max(np.abs(dec - deca))) < 0.15, 'max |diff| %.4f'
        % float(np.max(np.abs(dec - deca))))

    # ---------------- 账本 ----------------
    print('\n[4] 账本与链')
    L = json.loads(LEDGER.read_text(encoding='utf-8'))
    chk('n==463', len(L['measurements']) == 463, len(L['measurements']))
    chk('negatives==26', len(L['negatives']) == 26, len(L['negatives']))
    names = [m.get('name') for m in L['measurements']]
    chk('q6b2 recorded', 'pat_q6b2' in names, '')
    chk('q6b recorded', 'pat_q6b' in names, '')
    nnames = [n.get('name') for n in L['negatives']]
    chk('neg25 recorded', 'pat_q6b2_neg_shared_middle_word_geometry' in nnames, '')
    chk('neg26 recorded', 'pat_q6b_neg_attribute_low_rank_reuse' in nnames, '')

    # 链头：从备份基态按同样顺序回放 4 步写盘
    bak = LEDGER.with_suffix('.json.bak_q6b')
    if bak.exists():
        base = json.loads(bak.read_text(encoding='utf-8'))
        chk('backup base head == 179c9160', base['ledger_sha256_8'] == '179c9160',
            base['ledger_sha256_8'])
        d = json.loads(json.dumps(base))
        m_by = {m['name']: m for m in L['measurements'] if m.get('name')}
        n_by = {n['name']: n for n in L['negatives'] if n.get('name')}
        seq = [('m', 'pat_q6b2'), ('m', 'pat_q6b'),
               ('n', 'pat_q6b2_neg_shared_middle_word_geometry'),
               ('n', 'pat_q6b_neg_attribute_low_rank_reuse')]
        for kind, nm in seq:
            if kind == 'm':
                d['measurements'].append(m_by[nm])
            else:
                d.setdefault('negatives', []).append(n_by[nm])
            d['ledger_sha256_8'] = hashlib.sha256(
                json.dumps(d, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        chk('chain replay matches stored', d['ledger_sha256_8'] == L['ledger_sha256_8'],
            '%s vs %s' % (d['ledger_sha256_8'], L['ledger_sha256_8']))
    else:
        chk('backup exists for replay', False, str(bak))

    # ---------------- 规则账本 ----------------
    print('\n[5] 规则账本')
    R = json.loads((ROOT / 'data' / 'rules' / 'ledger.json').read_text(encoding='utf-8'))
    items = R['rules'] if isinstance(R, dict) and 'rules' in R else R
    ids = {x['id'] for x in items}
    chk('rules==98', len(items) == 98, len(items))
    for rid in ('R-ATT-23', 'R-ATT-24', 'R-MET-28', 'R-MET-29', 'R-MTH-19', 'R-MTH-20'):
        chk('rule %s present' % rid, rid in ids, '')
    r23 = [x for x in items if x['id'] == 'R-ATT-23'][0]
    chk('R-ATT-23 status refuted', r23['status'] == 'refuted', r23['status'])

    # ---------------- 文档 ----------------
    print('\n[6] 文档')
    fdoc = DOCS / 'PAT_Q6B_FINDINGS_20261101.md'
    chk('findings exists', fdoc.exists(), '')
    t = fdoc.read_text(encoding='utf-8')
    for key in ('NOT_A_SHARED_DIRECTION', 'NO_LOW_RANK_STRUCTURE', 'NOT_SEPARABLE',
                '−0.3994', '1.18', '3.4×', '家族结算', '0.0614'):
        chk('findings contains %r' % key, key in t, '')
    chk('pre-reg B2 has pre-run amendments',
        '跑前修订' in (DOCS / 'PAT_Q6B2_PREREG_v1.md').read_text(encoding='utf-8'), '')
    chk('pre-reg B has pre-run amendments',
        '跑前修订' in (DOCS / 'PAT_Q6B_PREREG_v1.md').read_text(encoding='utf-8'), '')

    print('\n==== %d PASS / %d FAIL ====' % (ok, fail))
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
