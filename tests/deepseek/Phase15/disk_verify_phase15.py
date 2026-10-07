# -*- coding: utf-8 -*-
"""
Phase 15 独立磁盘复核（分区 A-I）。要求 0 FAIL。
本脚本不 import 主脚本；cross_alpha / J_only / conc_hat / perm_null 全部独立重写一遍。
用法：python disk_verify_phase15.py
产物：tests/deepseek/Phase15/disk_verify_phase15.txt （PASS/FAIL 汇总）
"""
import io
import os
import sys
import json
import hashlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase15')
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
SEALP = os.path.join(P15T, 'N2h1a8_design_seal.json')
EXECP = os.path.join(P15T, 'execution_phase15.json')
RESP = os.path.join(P15T, 'result_phase15.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
B15 = os.path.join(P15T, 'memo_baseline_preappend_phase15.json')

_out = []
_n = [0, 0]


def ck(tag, ok, detail=''):
    _n[0] += 1
    if not ok:
        _n[1] += 1
    _out.append('[%s] %-52s %s' % ('PASS' if ok else 'FAIL', tag, detail))
    print(_out[-1])


def sec(t):
    _out.append('')
    _out.append('== ' + t + ' ==')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ---------------- 独立重算实现
def x_alpha(xs, ys, frac):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 2:
        return None
    ym = float(np.max(ys))
    if not np.isfinite(ym) or abs(ym) < 1e-12:
        return None
    tgt = frac * ym
    for i in range(len(xs) - 1):
        if ys[i] < tgt <= ys[i + 1]:
            t = (tgt - ys[i]) / (ys[i + 1] - ys[i])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def j_only(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return float('nan')
    s = np.diff(ys2) / np.diff(xs2)
    k = int(np.argmax(s))
    rest = np.delete(s, k)
    smed = float(np.median(rest)) if len(rest) else 0.0
    return float(s[k] / smed) if smed > 1e-12 else float('inf')


def conc_sh(F, wwin):
    F = np.asarray(F, float)
    jm = np.diff(F)
    if not np.isfinite(F).all():
        return None, None, [float(v) for v in jm]
    rng = float(F.max() - F.min())
    if rng <= 1e-12 or len(jm) < wwin:
        return None, None, jm.tolist()
    wins = [abs(float(np.sum(jm[j:j + wwin]))) for j in range(len(jm) - wwin + 1)]
    k = int(np.argmax(wins))
    return float(wins[k] / rng), k, jm.tolist()


def pnull(jumps, rng, wwin, nbp, range_val):
    js = np.asarray(jumps, float)
    if len(js) < wwin or not np.isfinite(js).all():
        return dict(null95=None, n_ok=0)
    out = np.full(nbp, np.nan)
    for b in range(nbp):
        p = js[rng.permutation(len(js))]
        wins = [abs(float(np.sum(p[j:j + wwin]))) for j in range(len(p) - wwin + 1)]
        out[b] = max(wins) / max(range_val, 1e-12)
    fin = out[np.isfinite(out)]
    if not len(fin):
        return dict(null95=None, n_ok=0)
    return dict(null95=float(np.percentile(fin, 95)), n_ok=int(len(fin)),
                ci=dict(lo=float(np.percentile(fin, 2.5)), hi=float(np.percentile(fin, 97.5)),
                        med=float(np.median(fin))))


def main():
    EX = json.load(io.open(EXECP, encoding='utf-8'))
    R = json.load(io.open(RESP, encoding='utf-8'))
    S = json.load(io.open(SEALP, encoding='utf-8'))
    SEED = int(EX['bootstrap']['seed'])
    W = int(EX['W'])
    XHF = float(EX['xh_frac'])
    BP = int(EX['bootstrap']['BP'])
    FL = EX['floors']

    # ---------------- A 文件
    sec('A. 文件与哈希')
    for tag, p in [('seal', SEALP), ('execution', EXECP), ('result', RESP)]:
        ck('A1 %s 存在且非空' % tag, os.path.exists(p) and os.path.getsize(p) > 1000,
           '%d B sha8=%s' % (os.path.getsize(p), sha(p)[:8]))
    ck('A2 result.execution_sha256 == 实盘 execution sha256',
       R['execution_sha256'] == sha(EXECP), sha(EXECP)[:16])
    ck('A3 result.seal_sha256 == 实盘 seal sha256', R['seal_sha256'] == sha(SEALP), sha(SEALP)[:16])
    reps = [f for f in sorted(os.listdir(P15T)) if f.startswith('n2h1a8_report_')]
    ck('A4 每臂报告齐备', len(reps) == len(R['arms']), '%s' % reps)

    # ---------------- B seal / execution
    sec('B. seal <-> execution 一致性')
    ck('B1 execution.seal_sha256 == seal 实盘', EX['seal_sha256'] == sha(SEALP))
    ck('B2 seal.phase == execution.phase == 15', S['phase'] == 15 and EX['phase'] == 15)
    ck('B3 网格三处一致 (profile_sites)',
       list(S['profile_sites']) == list(EX['profile_sites']) == list(R['grid']['profile_sites']))
    ck('B4 网格三处一致 (alphas)', list(S['alphas']) == list(EX['alphas']) == list(R['grid']['alphas']))
    ck('B5 W/xh_frac 冻结一致', W == 3 and abs(XHF - 0.5) < 1e-12 and S['floors']['NULL_HIGH'] == 0.70)
    inh_s = S['inheritance_anchors']['inherited_published']
    ck('B6 继承锚 XH_12_by_site 18 位点', len(inh_s['XH_12_by_site']) == 18)
    ck('B7 继承锚 J_swap_12_by_site 18 位点', len(inh_s['J_swap_12_by_site']) == 18)
    ck('B8 冻结参照 d_argmax_4B == 13',
       abs(int(inh_s['MODE_X_13']) - int(inh_s['MODE_J_13'])) == 13)

    # ---------------- C result 结构
    sec('C. result 结构完整性')
    need_top = ['grid', 'arms_meta', 'arms', 'E2_full_swap', 'E3_localize', 'E4_summary',
                'E5_concentration', 'E6_calibration', 'verdict', 'joint_verdict', 'predictions_check']
    ck('C1 result 顶层键齐备', all(k in R for k in need_top),
       '缺 %s' % [k for k in need_top if k not in R])
    ck('C2 三臂齐备', set(R['arms'].keys()) == set(EX['arm_order']), '%s' % list(R['arms'].keys()))
    sites = list(R['grid']['profile_sites'])
    alphas = list(R['grid']['alphas'])
    for arm in EX['arm_order']:
        a = R['arms'].get(arm, {})
        pp = a.get('E4_per_pair', {})
        ok = set(pp.keys()) == set(str(s) for s in sites)
        ck('C3 %-24s E4_per_pair 位点齐备' % arm, ok, '%d 位点' % len(pp))
        if ok:
            sub = pp[str(sites[0])]
            ck('C4 %-24s 每行 alpha 数 == %d' % (arm, len(alphas)),
               len(sub) == len(alphas) and all(len(r['per_pair']) == 24 for r in sub),
               'rows=%d pairs=%d' % (len(sub), len(sub[0]['per_pair'])))

    # ---------------- D 从 per_pair 独立重算剖面
    sec('D. 独立重算 xhalf / J（从 E4_per_pair + FULL_SWAP）')
    for arm in EX['arm_order']:
        a = R['arms'][arm]
        FULL = float(R['E2_full_swap'][arm]['FULL_SWAP'])
        pp = a['E4_per_pair']
        PM = np.stack([np.array([r['per_pair'] for r in pp[str(s)]], float) for s in sites], 0)
        Y = PM.mean(axis=2) / FULL
        xh = np.array([x_alpha(alphas, Y[i], XHF) for i in range(len(sites))], float)
        xh = np.where(np.isfinite(xh), xh, np.nan)
        Jv = np.array([j_only(alphas, Y[i]) for i in range(len(sites))], float)
        rx = R['E4_summary'][arm]['xhalf']
        rj = R['E4_summary'][arm]['J']
        dx = max(abs(xh[i] - rx[i]) for i in range(len(sites))
                 if np.isfinite(xh[i]) and np.isfinite(rx[i])) if np.isfinite(xh).any() else 0.0
        ck('D1 %-24s xhalf 逐位点重算一致' % arm, dx <= 1e-12, 'max|d|=%.3e' % dx)
        dj = max((abs(Jv[i] - rj[i]) for i in range(len(sites))
                  if np.isfinite(Jv[i]) and np.isfinite(rj[i])), default=0.0)
        cj = max((abs(Jv[i] - rj[i]) / max(abs(rj[i]), 1e-9) for i in range(len(sites))
                  if np.isfinite(Jv[i]) and np.isfinite(rj[i])), default=0.0)
        ck('D2 %-24s J 逐位点重算一致' % arm, (dj <= 1e-9 or cj <= 1e-12),
           'max abs=%.3e rel=%.3e' % (dj, cj))
        # XH_RANGE
        if np.isfinite(xh).any():
            rr = float(np.nanmax(xh) - np.nanmin(xh))
            ck('D3 %-24s XH_RANGE 重算一致' % arm,
               abs(rr - R['E4_summary'][arm]['XH_RANGE']) <= 1e-12,
               '%.6f vs %s' % (rr, R['E4_summary'][arm]['XH_RANGE']))
        # E5 里的 xhalf 与 E4_summary 相同
        ck('D4 %-24s E5.xhalf == E4_summary.xhalf' % arm,
           max(abs(R['E5_concentration'][arm]['xhalf'][i] - rx[i]) for i in range(len(sites))) <= 1e-15)

    # ---------------- E FULL_SWAP 独立重算
    sec('E. FULL_SWAP 独立重算')
    for arm in EX['arm_order']:
        e2 = R['E2_full_swap'][arm]
        order = list(e2['FS_ORDER'])
        fs = np.array([e2['FS_PAIR'][k] for k in order], float)
        ck('E1 %-24s FS_VEC == FS_PAIR[FS_ORDER]' % arm,
           np.max(np.abs(fs - np.array(e2['FS_VEC'], float))) <= 1e-15, 'n=%d' % len(order))
        ck('E2 %-24s FULL_SWAP == mean(FS_VEC)' % arm,
           abs(float(fs.mean()) - float(e2['FULL_SWAP'])) <= 1e-12,
           '%.12f' % float(fs.mean()))
        ck('E3 %-24s FS_VEC 长度 == discovery 对数 (%d)' % (arm, len(EX['discovery'])),
           len(fs) == len(EX['discovery']), 'n=%d' % len(fs))

    # ---------------- F 集中度 + null 重放
    sec('F. 集中度 + 置换零假设逐位重放')
    for arm in EX['arm_order']:
        e5 = R['E5_concentration'][arm]
        xh = np.array(e5['xhalf'], float)
        Jv = np.array(e5['J'], float)
        s_x, k_x, j_x = conc_sh(xh, W)
        s_j, k_j, j_j = conc_sh(Jv, W)
        ck('F1 %-24s top3_x 重算一致' % arm,
           (s_x is None and e5['top3_x'] is None) or abs(s_x - e5['top3_x']) <= 1e-15,
           '%s' % s_x)
        ck('F2 %-24s top3_j 重算一致' % arm,
           (s_j is None and e5['top3_j'] is None) or abs(s_j - e5['top3_j']) <= 1e-15,
           '%s' % s_j)
        ck('F3 %-24s argmax 窗口一致' % arm,
           (k_x == e5['argmax_w_x']) and (k_j == e5['argmax_w_j']),
           'x=%s j=%s' % (k_x, k_j))
        ck('F4 %-24s jumps_x 重算一致' % arm,
           np.max(np.abs(np.array(j_x, float) - np.array(e5['jumps_x'], float))) <= 1e-15)
        rgx = float(np.nanmax(xh) - np.nanmin(xh)) if np.isfinite(xh).any() else float('nan')
        ck('F5 %-24s range_x 重算一致' % arm, abs(rgx - e5['range_x']) <= 1e-15)
        # null 重放（与主脚本同 seed / 同顺序）
        r1 = np.random.default_rng(SEED + 13)
        r2 = np.random.default_rng(SEED + 29)
        nx = pnull(j_x, r1, W, BP, e5['range_x'])
        nj = pnull(j_j, r2, W, BP, e5['range_j'])
        okx = ((nx['null95'] is None and e5['null_x']['null95'] is None)
               or abs(nx['null95'] - e5['null_x']['null95']) <= 1e-15)
        okj = ((nj['null95'] is None and e5['null_j']['null95'] is None)
               or abs(nj['null95'] - e5['null_j']['null95']) <= 1e-15)
        ck('F6 %-24s null95_x 逐位重放一致' % arm, okx, '%s vs %s' % (nx['null95'], e5['null_x']['null95']))
        ck('F7 %-24s null95_j 逐位重放一致' % arm, okj, '%s vs %s' % (nj['null95'], e5['null_j']['null95']))
        if e5['margin_x'] is not None and nx['null95'] is not None:
            ck('F8 %-24s margin_x == share_x - null95_x' % arm,
               abs((s_x - nx['null95']) - e5['margin_x']) <= 1e-15, '%.6f' % e5['margin_x'])
        d_aw = abs(int(k_x) - int(k_j)) if (k_x is not None and k_j is not None) else None
        ck('F9 %-24s d_argmax_window 重算一致' % arm, d_aw == e5['d_argmax_window'],
           '%s vs %s' % (d_aw, e5['d_argmax_window']))

    # ---------------- G 判决复算
    sec('G. 判决复算')
    V = R['verdict']
    for arm in EX['arm_order']:
        v = V[arm]
        e5 = R['E5_concentration'][arm]
        e0 = R['E0_selfcheck'][arm]
        a = R['arms'][arm]
        q0 = ('PASS' if (a.get('T2_only') and a.get('F4_dims_ok') and a.get('F2_base_ok')
                         and e0['determinism_maxdiff'] == 0 and e0['hook_effect_maxdiff'] > 0) else 'FAIL')
        ck('G1 %-24s Q0_device 复算一致' % arm, q0 == v.get('Q0_device'), q0)
        d = e5['d_argmax_window']
        lab = 'NA' if d is None else ('ARGS_GAP_GE3' if d >= FL['ARGS_GAP_MIN'] else 'ARGS_GAP_LT3')
        ck('G2 %-24s Q2 标签复算一致' % arm, lab == v.get('Q2_label'), lab)
        n95 = e5['null_x']['null95']
        lab3 = 'NA' if n95 is None else ('NULL_X_HIGH' if n95 >= FL['NULL_HIGH'] else 'NULL_X_OK')
        ck('G3 %-24s Q3 标签复算一致' % arm, lab3 == v.get('Q3_label'), lab3)
        lab4 = 'NA' if e5['margin_x'] is None else ('ABOVE_NULL' if e5['margin_x'] > 0 else 'AT_OR_BELOW_NULL')
        ck('G4 %-24s Q4 标签复算一致' % arm, lab4 == v.get('Q4_label_x'), lab4)
    J = R['joint_verdict']
    rep = J['arms_used_for_cross_model']
    q2 = [V[a]['Q2_label'] for a in rep]
    q3 = [V[a]['Q3_label'] for a in rep]
    jq2 = ('ARGS_GAP_LAYERSTACK' if (q2 and all(x == 'ARGS_GAP_GE3' for x in q2)) else
           'ARGS_GAP_4B_SPECIFIC' if (q2 and all(x == 'ARGS_GAP_LT3' for x in q2)) else
           'ARGS_GAP_MIXED' if q2 else 'NA')
    jq3 = ('CONC_JUDGE_INVALID_X_ALL' if (q3 and all(x == 'NULL_X_HIGH' for x in q3)) else
           'CONC_JUDGE_ALIVE_X' if q3 else 'NA')
    ck('G5 joint Q2 复算一致', jq2 == J['Q2_joint'], '%s' % jq2)
    ck('G6 joint Q3 复算一致', jq3 == J['Q3_joint'], '%s' % jq3)

    # A0 校准复算
    if 'A0_calib_qwen3-4b-nf4' in R['E6_calibration']:
        e6 = R['E6_calibration']['A0_calib_qwen3-4b-nf4']
        XH12 = {int(k): float(v) for k, v in inh_s['XH_12_by_site'].items()}
        xh = R['E4_summary']['A0_calib_qwen3-4b-nf4']['xhalf']
        dx = max(abs(xh[i] - XH12[s]) for i, s in enumerate(sites) if s in XH12 and np.isfinite(xh[i]))
        ck('G7 A0 max|dxhalf| 复算一致', abs(dx - e6['max_abs_dxh']) <= 1e-12, '%.6f' % dx)
        ck('G8 A0 pass_tol == (max|dxhalf| <= %.2f)' % FL['XH_FAITHFUL_TOL'],
           e6['pass_tol'] == (dx <= FL['XH_FAITHFUL_TOL']))
        ck('G9 A0 argmax_same 非 None 假阳',
           (e6['argmax_same'] is True) == (e6['argmax_w_x_nf4'] is not None
                                           and e6['argmax_w_x_nf4'] == e6['argmax_w_x_bf16']))
        ck('G10 A0 Q1 标签复算一致',
           R['verdict']['A0_calib_qwen3-4b-nf4']['Q1_label'] ==
           ('NF4_FAITHFUL' if (e6['pass_tol'] and e6['argmax_same']) else 'NF4_DEVIANT'))

    # ---------------- H 预测核对复算
    sec('H. 预注册预测核对复算')
    PC = R['predictions_check']
    for pid in [p['id'] for p in S['pre_registered_predictions']]:
        ck('H1 %s 存在于 predictions_check' % pid, pid in PC, str(PC.get(pid, {}).get('detail'))[:90])

    # ---------------- I 文档落点
    sec('I. 文档落点')
    mb = open(MEMO, 'rb').read()
    txt = mb.decode('utf-8-sig')
    n_hdr = txt.count('\n## Phase 15:') + (1 if txt.startswith('## Phase 15:') else 0)
    if n_hdr == 0:
        ck('I1 MEMO Phase 15 节', True, 'PENDING_APPEND（本轮尚未追加）')
    else:
        ck('I1 MEMO Phase 15 节存在', n_hdr == 1, 'headings=%d ; bytes=%d sha8=%s' %
           (n_hdr, len(mb), hashlib.sha256(mb).hexdigest()[:8]))
    if os.path.exists(B15):
        b15 = json.load(io.open(B15, encoding='utf-8'))
        ck('I2 追加前基线快照存在', b15.get('bytes', 0) > 0,
           'bytes=%s lines=%s sha8=%s' % (b15.get('bytes'), b15.get('lines'), b15.get('sha8')))
        if n_hdr == 1:
            pre = mb[:int(b15['bytes'])]
            ck('I3 前缀锚：追加前字节未被改动',
               hashlib.sha256(pre).hexdigest()[:8] == b15['sha8'],
               '%s vs %s' % (hashlib.sha256(pre).hexdigest()[:8], b15['sha8']))
            ck('I4 bare_lf == 0（MEMO 纪律）',
               mb.count(b'\n') - mb.count(b'\r\n') == 0,
               'bare_lf=%d' % (mb.count(b'\n') - mb.count(b'\r\n')))
    else:
        ck('I2 追加前基线快照存在', False, 'MISSING %s' % B15)
    if os.path.exists(BASE):
        bs = json.load(io.open(BASE, encoding='utf-8'))
        ck('I5 _infra/memo_baseline.json 与实盘自洽',
           int(bs.get('bytes', -1)) == len(mb) or n_hdr == 0,
           'baseline bytes=%s ; memo bytes=%d' % (bs.get('bytes'), len(mb)))
    ck('I6 wlog 存在', os.path.exists(WLOG), '%d B' % (os.path.getsize(WLOG) if os.path.exists(WLOG) else 0))
    if os.path.exists(WLOG):
        _wb = open(WLOG, 'rb').read()
        _wd = _wb.decode('utf-8')
        _sup = '## Phase 15 收尾补记'
        _i = _wd.find(_sup)
        ck('I6b wlog 含 Phase 15 主节', _wd.count('## Phase 15 /') == 1,
           '主节标题数=%d' % _wd.count('## Phase 15 /'))
        ck('I6c wlog 含收尾补记段（唯一）', _wd.count(_sup) == 1, 'count=%d' % _wd.count(_sup))
        if _i > 0:
            _seg = _wd[_i:]
            _bt = _seg.count(chr(96))
            ck('I6d 补记段未被 bash 反引号替换吞字（段内含反引号）', _bt > 20,
               'backticks_in_segment=%d' % _bt)
            ck('I6e 补记段落点上文未被改动（含 Phase 15 主节标题）',
               _wd.count('## Phase 15 / N2h1-α-8') == 1 and
               _wd.find('## Phase 15 / N2h1-α-8') < _i, None)

    # ---------------- J amend1：词表常量勘误与逐臂 sup_id（独立复核）
    sec('J. amend1 词表勘误 / 逐臂 sup_id 独立复核')
    AM1P = os.path.join(P15T, 'N2h1a8_design_seal_amend1.json')
    ck('J1 amend1 存在', os.path.exists(AM1P),
       '%d B sha8=%s' % (os.path.getsize(AM1P) if os.path.exists(AM1P) else 0,
                         sha(AM1P)[:8] if os.path.exists(AM1P) else 'NONE'))
    if os.path.exists(AM1P):
        AM1 = json.load(io.open(AM1P, encoding='utf-8'))
        ck('J2 amend1.amend_of_seal_sha256 == 实盘 seal',
           AM1['amend_of_seal_sha256'] == sha(SEALP), AM1['amend_of_seal_sha8'])
        ck('J3 result.amend1_sha256 == 实盘 amend1',
           R.get('amend1_sha256') == sha(AM1P), sha(AM1P)[:16])
        ck('J4 amend1 承载首轮(作废)exec 的 sha8',
           len(str(AM1.get('amend_of_execution_sha256', ''))) == 64,
           str(AM1.get('amend_of_execution_sha256'))[:16])
        ck('J5 首轮作废日志保留',
           os.path.exists(os.path.join(P15T, '_formal_stdout_run1_INVALID_supid.log')),
           'evidence of the intercepted run')

        # ---- 独立从各臂 tokenizer 重新解析类别 token，与 result 逐项比对
        sups = list(EX['classes'])
        ref = {k: int(v) for k, v in EX['sup_id'].items()}
        ck('J6 result.sup_id_ref == exec.sup_id', R.get('sup_id_ref') == ref)
        try:
            from transformers import AutoTokenizer
            MDIR = os.path.join(ROOT, 'models', 'hf')
            live = {}
            for arm in EX['arm_order']:
                d = os.path.join(MDIR, EX['arms'][arm]['dir'])
                tk = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
                ids, rt = {}, True
                for wd in sups:
                    t = list(tk.encode(wd, add_special_tokens=False))
                    rt = rt and len(t) == 1 and tk.decode([t[0]]) == wd
                    ids[wd] = int(t[0])
                live[arm] = dict(ids=ids, rt=bool(rt))
            for arm in EX['arm_order']:
                got = R.get('sup_id_per_arm', {}).get(arm)
                ck('J7 %-24s 逐臂 sup_id 与实盘 tokenizer 一致' % arm,
                   got == live[arm]['ids'], '%s' % json.dumps(got, ensure_ascii=False))
                ck('J8 %-24s 6/6 类别词单 token 且 decode 可逆' % arm, live[arm]['rt'])
            ck('J9 A0/A2 sup_id == qwen 参考值',
               all(R.get('sup_id_per_arm', {}).get(a) == ref for a in EX['arm_order']
                   if a != 'A1_glm4-9b-nf4'))
            ck('J10 A1 sup_id != qwen 参考值（词表确实不同）',
               R.get('sup_id_per_arm', {}).get('A1_glm4-9b-nf4') != ref,
               '%s' % json.dumps(R.get('sup_id_per_arm', {}).get('A1_glm4-9b-nf4'),
                                 ensure_ascii=False))
        except Exception as e:
            ck('J7 独立解析 tokenizer', False, 'tokenizer 复核失败: %r' % (e,))

        # ---- F1b / F2 base 门：修正后的正式运行必须三臂全过
        for arm in EX['arm_order']:
            a = R['arms'].get(arm, {})
            ck('J11 %-24s F1b_ok' % arm, bool(a.get('F1b_ok')))
            ck('J12 %-24s F2_base_bad 为空（修正后）' % arm,
               len(a.get('F2_base_bad') or []) == 0,
               'bad=%d' % len(a.get('F2_base_bad') or []))
        ck('J13 三臂 Q0_device 全 PASS',
           all(R['verdict'][a].get('Q0_device') == 'PASS' for a in EX['arm_order']),
           '%s' % {a: R['verdict'][a].get('Q0_device') for a in EX['arm_order']})

    _out.append('')
    _out.append('TOTAL checks = %d ; FAIL = %d' % (_n[0], _n[1]))
    out = os.path.join(P15, 'disk_verify_phase15.txt')
    io.open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(_out) + '\n')
    print('')
    print('TOTAL checks = %d ; FAIL = %d' % (_n[0], _n[1]))
    print('REPORT -> %s' % out)


if __name__ == '__main__':
    main()
