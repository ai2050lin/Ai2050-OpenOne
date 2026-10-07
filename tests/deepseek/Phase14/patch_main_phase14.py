# -*- coding: utf-8 -*-
"""Phase 14 主脚本补丁：A8 臂 + amend2 锚 + 双族集中度泛化（逐处 assert + 回读）。"""
import io, hashlib

P = r'tests/deepseek/Phase14/n2h1a7_prefix_swap.py'
t = io.open(P, encoding='utf-8', newline='').read()
orig = t


def rep(old, new, expect=1):
    global t
    n = t.count(old)
    assert n == expect, 'count=%d != %d for %r' % (n, expect, old[:70])
    t = t.replace(old, new, 1)


# ---------- (1) F29/F30 作用域 + 逐对恒等式 ----------
rep("""F29_ok = bool(F29_dev <= ANCHOR_TOL)
F30_ok = bool(F30_dev <= ANCHOR_TOL)""",
    """FULL_PANEL = (len(_A1_PAIRS) == 24)
# F30(a) 与子集无关的逐对恒等式：alpha=1 时 per_pair == Phase12 FULL_SWAP_pairs[受体词]
F30a_dev = 0.0
_n_f30a = 0
for s in _A1_SITES:
    for row in A1[str(s)]:
        if abs(row['alpha'] - 1.0) > 1e-12:
            continue
        for k_, rw_ in enumerate(row['order']):
            ref = float(R12['FULL_SWAP_pairs'][rw_])
            F30a_dev = max(F30a_dev, abs(row['per_pair'][k_] - ref) / max(abs(ref), 1e-9))
            _n_f30a += 1
F30a_ok = bool(F30a_dev <= 1e-4)
# F30(b) 面板级常数性（仅在臂的 pair 集 == 24 个 discovery 对时断言）
_y01_sorted = np.array([_y01[s] for s in sorted(_y01)], float)
F30b_range = float(_y01_sorted.max() - _y01_sorted.min()) if len(_y01_sorted) else None
F30b_dev = (abs(float(_y01_sorted[0]) - 1.0) if len(_y01_sorted) else None)
if FULL_PANEL and F30b_range is not None:
    F30b_ok = bool(F30b_range <= 1e-12 and F30b_dev <= ANCHOR_TOL)
else:
    F30b_ok = None
F29_ok = bool(F29_dev <= ANCHOR_TOL) if FULL_PANEL else None
F30_ok = bool(F30a_ok and (F30b_ok is not False))
w('  F29 full_panel=%s ok=%s ; F30a(n=%d) dev=%.3e ok=%s ; F30b ok=%s (range=%s)' %
  (FULL_PANEL, F29_ok, _n_f30a, F30a_dev, F30a_ok, F30b_ok,
   ('%.3e' % F30b_range) if F30b_range is not None else 'NA'))""")

# ---------- (2) A6 段整体替换 ----------
A6_START = "w('--- A6 双坐标集中度（铁律 t）---')\n"
A7_MARK = "# ================================================================ A7 range / steepness"
i0 = t.index(A6_START)
i1 = t.index(A7_MARK)
assert i0 < i1

NEW_A6 = '''w('--- A6 双坐标集中度（铁律 t）：A1（位置前缀）与 A8（逐层累积）各一份 ---')
n_xh_ok = int(np.sum([A1_XH[str(s)] is not None for s in _A1_SITES]))
F32_ok = (n_xh_ok == len(_A1_SITES))
F33_ok = all(np.isfinite(A1_J[str(s)]) for s in _A1_SITES) and (not SMOKE)


def _hist(v):
    v = np.asarray(v)[np.asarray(v) >= 0]
    if len(v) == 0:
        return {}
    u, c = np.unique(v, return_counts=True)
    return {str(int(k)): int(n) for k, n in zip(u, c)}


def full_concentration(per_list, alpha_grid, sites, tag, do_boot=True):
    """点估计 + bootstrap 双坐标集中度 + 置换零假设 + 位点间配对 Δ（口径逐字沿用 Phase 13）。"""
    PM = np.stack([np.array(p, float) for p in per_list], 0)
    nS = PM.shape[0]; nP = PM.shape[2]
    xs = np.array(alpha_grid, float)
    Y = PM.mean(axis=2) / FULL_SWAP
    xh = np.array([cross_alpha(xs, Y[i], XHF) for i in range(nS)], float)
    Jv = np.array([J_only(xs, Y[i]) for i in range(nS)], float)
    share_x, axw_x, jx = conc_hat(xh)
    share_j, axw_j, jj = conc_hat(Jv)
    o = dict(tag=tag, sites=list(sites), alpha_grid=list(alpha_grid),
             xhalf=[float(v) for v in xh], J=[float(v) for v in Jv],
             top3_x=share_x, argmax_w_x=axw_x, jumps_x=jx,
             top3_j=share_j, argmax_w_j=axw_j, jumps_j=jj,
             range_x=float(xh.max() - xh.min()), range_j=float(Jv.max() - Jv.min()),
             win_sem_x=(None if axw_x is None else dict(w=axw_x, a=sites[axw_x], b=sites[axw_x + 3])),
             win_sem_j=(None if axw_j is None else dict(w=axw_j, a=sites[axw_j], b=sites[axw_j + 3])))
    if (not do_boot) or nS < 4 or nP < 8:
        o['bootstrap'] = dict(skipped=True)
        o['null'] = dict(skipped=True)
        o['paired'] = dict(skipped=True, N_dec_J=0, N_dec_X=0, rows=[], n_pairs=max(nS - 1, 0))
        return o
    BRNG = np.random.default_rng(SEED)
    xh_b = np.full((BS, nS), np.nan); J_b = np.full((BS, nS), np.nan)
    t3x_b = np.full(BS, np.nan); t3j_b = np.full(BS, np.nan)
    ax_b = np.full(BS, -1, int); aj_b = np.full(BS, -1, int)
    dJ_b = np.full((BS, nS - 1), np.nan); dX_b = np.full((BS, nS - 1), np.nan)
    t_b = time.time()
    for b in range(BS):
        idx = BRNG.integers(0, nP, nP)
        fs_b = float(FS_VEC[idx].mean())
        if abs(fs_b) < 1e-9:
            continue
        Yb = PM[:, :, idx].mean(axis=2) / fs_b
        for i in range(nS):
            xv = cross_alpha(xs, Yb[i], XHF)
            xh_b[b, i] = xv if xv is not None else np.nan
            J_b[b, i] = J_only(xs, Yb[i])
        dX_b[b] = xh_b[b, :-1] - xh_b[b, 1:]
        dJ_b[b] = J_b[b, :-1] - J_b[b, 1:]
        xr = xh_b[b]
        if np.all(np.isfinite(xr)):
            rr = float(xr.max() - xr.min()); jm = np.diff(xr)
            if rr > 1e-12:
                ws = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
                t3x_b[b] = max(ws) / rr; ax_b[b] = int(np.argmax(ws))
        jr = J_b[b]
        if np.all(np.isfinite(jr)):
            rr = float(jr.max() - jr.min()); jm = np.diff(jr)
            if rr > 1e-12:
                ws = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
                t3j_b[b] = max(ws) / rr; aj_b[b] = int(np.argmax(ws))
    fnx = t3x_b[np.isfinite(t3x_b)]; fnj = t3j_b[np.isfinite(t3j_b)]
    hx = _hist(ax_b); hj = _hist(aj_b)
    mx = int(max(hx, key=lambda k: hx[k])) if hx else -1
    mj = int(max(hj, key=lambda k: hj[k])) if hj else -1
    o['bootstrap'] = dict(
        BS=BS, secs=round(time.time() - t_b, 1),
        P_ge_060_x=(float(np.mean(fnx >= 0.60)) if len(fnx) else None),
        P_le_040_x=(float(np.mean(fnx <= 0.40)) if len(fnx) else None),
        P_ge_060_j=(float(np.mean(fnj >= 0.60)) if len(fnj) else None),
        P_le_040_j=(float(np.mean(fnj <= 0.40)) if len(fnj) else None),
        mode_x=mx, mode_j=mj,
        freq_x=((hx[str(mx)] / max(sum(hx.values()), 1)) if hx else None),
        freq_j=((hj[str(mj)] / max(sum(hj.values()), 1)) if hj else None),
        hist_x=hx, hist_j=hj, n_ok_x=int(len(fnx)), n_ok_j=int(len(fnj)),
        ci_top3_x=_ci(t3x_b), ci_top3_j=_ci(t3j_b))
    NBRNG = np.random.default_rng(SEED + 13)
    t3n_x = np.full(BP, np.nan); t3n_j = np.full(BP, np.nan)
    jxv = np.array(jx, float); jjv = np.array(jj, float)
    for b in range(BP):
        px = jxv[NBRNG.permutation(len(jxv))]; pj = jjv[NBRNG.permutation(len(jjv))]
        wx = [abs(float(np.sum(px[j:j + W]))) for j in range(len(px) - W + 1)]
        wj = [abs(float(np.sum(pj[j:j + W]))) for j in range(len(pj) - W + 1)]
        t3n_x[b] = max(wx) / o['range_x']; t3n_j[b] = max(wj) / o['range_j']
    nx95 = float(np.percentile(t3n_x, 95)); nj95 = float(np.percentile(t3n_j, 95))
    o['null'] = dict(BP=BP, null_x_95=nx95, null_j_95=nj95,
                     x_above_null=bool(share_x is not None and share_x >= nx95),
                     j_above_null=bool(share_j is not None and share_j >= nj95),
                     ci_x=_ci(t3n_x), ci_j=_ci(t3n_j))
    rows = []; ndJ = 0; ndX = 0
    for i in range(nS - 1):
        rj_ = _ci(dJ_b[:, i]); rx_ = _ci(dX_b[:, i])
        lj = ('DECISIVE_DOWN' if (rj_['lo'] is not None and rj_['lo'] > 0)
              else 'DECISIVE_UP' if (rj_['hi'] is not None and rj_['hi'] < 0) else 'TIE')
        lx = ('DECISIVE_DOWN' if (rx_['lo'] is not None and rx_['lo'] > 0)
              else 'DECISIVE_UP' if (rx_['hi'] is not None and rx_['hi'] < 0) else 'TIE')
        ndJ += int(lj != 'TIE'); ndX += int(lx != 'TIE')
        rows.append(dict(a=sites[i], b=sites[i + 1],
                         dJ=dict(obs=float(Jv[i] - Jv[i + 1]), **rj_), label_j=lj,
                         dX=dict(obs=float(xh[i] - xh[i + 1]), **rx_), label_x=lx))
    o['paired'] = dict(rows=rows, N_dec_J=int(ndJ), N_dec_X=int(ndX), n_pairs=int(nS - 1))
    return o


A1C = full_concentration([A1_PER[str(s)] for s in _A1_SITES], _A1_AL, _A1_SITES,
                         'A1_position_prefix', do_boot=(not SMOKE))
_A8i = list(range(len(_A8_SITES)))
A8C = full_concentration([A8_PER[str(i)] for i in _A8i], _A8_AL, _A8i,
                         'A8_cumulative_layer', do_boot=(not SMOKE))

for _C in (A1C, A8C):
    _bt = _C['bootstrap']
    w('')
    w('  [%s] top3_x = %s (argmax_w %s) ; top3_j = %s (argmax_w %s)' %
      (_C['tag'], _C['top3_x'], _C['argmax_w_x'], _C['top3_j'], _C['argmax_w_j']))
    if _C['win_sem_x']:
        w('    X 窗口: w=%d <=> %s -> %s' % (_C['win_sem_x']['w'], _C['win_sem_x']['a'], _C['win_sem_x']['b']))
    if _C['win_sem_j']:
        w('    J 窗口: w=%d <=> %s -> %s' % (_C['win_sem_j']['w'], _C['win_sem_j']['a'], _C['win_sem_j']['b']))
    w('    range_x = %.6f ; range_j = %.6f' % (_C['range_x'], _C['range_j']))
    if _bt.get('skipped'):
        w('    bootstrap: skipped')
    else:
        w('    P(>=.60) X=%.4f J=%.4f ; P(<=.40) X=%.4f J=%.4f ; n_ok %d/%d (%.1fs)' %
          (_bt['P_ge_060_x'], _bt['P_ge_060_j'], _bt['P_le_040_x'], _bt['P_le_040_j'],
           _bt['n_ok_x'], _bt['n_ok_j'], _bt['secs']))
        w('    mode X=%d (freq %.4f) hist %s' % (_bt['mode_x'], _bt['freq_x'], _bt['hist_x']))
        w('    mode J=%d (freq %.4f) hist %s' % (_bt['mode_j'], _bt['freq_j'], _bt['hist_j']))
        w('    null 95th: X=%.4f (above=%s) J=%.4f (above=%s)' %
          (_C['null']['null_x_95'], _C['null']['x_above_null'],
           _C['null']['null_j_95'], _C['null']['j_above_null']))
        w('    paired Δ: N_dec_J=%d/%d N_dec_X=%d/%d' %
          (_C['paired']['N_dec_J'], _C['paired']['n_pairs'],
           _C['paired']['N_dec_X'], _C['paired']['n_pairs']))

xh_arr = np.array(A1C['xhalf'], float)
J_arr = np.array(A1C['J'], float)
share_x = A1C['top3_x']; axw_x = A1C['argmax_w_x']; jumps_x = A1C['jumps_x']
share_j = A1C['top3_j']; axw_j = A1C['argmax_w_j']; jumps_j = A1C['jumps_j']
_b1 = A1C['bootstrap']
mode_x = _b1.get('mode_x', -1); mode_j = _b1.get('mode_j', -1)
freq_x = _b1.get('freq_x'); freq_j = _b1.get('freq_j')
P_ge_x = _b1.get('P_ge_060_x'); P_le_x = _b1.get('P_le_040_x')
P_ge_j = _b1.get('P_ge_060_j'); P_le_j = _b1.get('P_le_040_j')
null_x_95 = (A1C['null']['null_x_95'] if not A1C['null'].get('skipped') else float('nan'))
null_j_95 = (A1C['null']['null_j_95'] if not A1C['null'].get('skipped') else float('nan'))
A6 = dict(A1=A1C, A8=A8C)
A6b = A1C['bootstrap']
A6n = A1C['null']
A7p = A1C['paired']
sys.stdout.flush()

'''
t = t[:i0] + NEW_A6 + t[i1:]

# ---------- (3) 判决段替换 ----------
V_START = "G0p = bool(F24_ok and F25_ok and F26_ok and F27_ok and F28_ok and F29_ok and F30_ok and F31_ok)"
V_END = "# 位置判决"
j0 = t.index(V_START)
j1 = t.index(V_END)
assert j0 < j1

NEW_V = '''G0p = bool(F24_ok and F25_ok and F26_ok and F27_ok and F28_ok
           and (F29_ok is not False) and (F30_ok is not False) and F31_ok)
endpoint_dev = F30b_dev if F30b_dev is not None else float('nan')


def verdict_of(C):
    """同坐标判决表（6 行）+ 跨族迁移判据（4 行），逐条对齐 seal/amend2 的冻结文本。"""
    if not isinstance(C, dict) or C.get('skipped'):
        return dict(verdict_same_coordinate='SKIPPED', verdict_cross_family='SKIPPED')
    bts = C.get('bootstrap') or {}
    s_x = C['top3_x']; s_j = C['top3_j']; w_x = C['argmax_w_x']; w_j = C['argmax_w_j']
    pg_x = bts.get('P_ge_060_x'); pl_x = bts.get('P_le_040_x')
    pg_j = bts.get('P_ge_060_j'); pl_j = bts.get('P_le_040_j')
    cd = bool(w_x is not None and w_j is not None and w_x != w_j and abs(w_x - w_j) >= 3
              and s_x is not None and s_x >= 0.40 and s_j is not None and s_j >= 0.40)
    if pg_x is not None and pg_j is not None and pg_x >= 0.95 and pg_j >= 0.95:
        vs = 'CONCENTRATION_FEW_LAYER_ROBUST'
    elif pl_x is not None and pl_j is not None and pl_x >= 0.95 and pl_j >= 0.95:
        vs = 'CONCENTRATION_ACCUMULATE_ROBUST'
    elif cd:
        vs = 'CONCENTRATION_COORDINATE_DEPENDENT'
    else:
        vs = 'CONCENTRATION_UNDECIDED'
    if w_x is None or w_j is None:
        vf = 'TRANSFER_UNDECIDED'
    else:
        dx = abs(w_x - MODE_X_13); dj = abs(w_j - MODE_J_13)
        if dx <= 2 and dj <= 2:
            vf = 'FAMILY_TRANSFER_BOTH'
        elif dx <= 2:
            vf = 'FAMILY_TRANSFER_XHALF_ONLY'
        elif dj <= 2:
            vf = 'FAMILY_TRANSFER_J_ONLY'
        else:
            vf = 'FAMILY_NO_TRANSFER'
    return dict(verdict_same_coordinate=vs, verdict_cross_family=vf, coord_dep=bool(cd),
                share_x=s_x, share_j=s_j, argmax_w_x=w_x, argmax_w_j=w_j,
                P_ge_060_x=pg_x, P_le_040_x=pl_x, P_ge_060_j=pg_j, P_le_040_j=pl_j,
                d_x=(abs(w_x - MODE_X_13) if w_x is not None else None),
                d_j=(abs(w_j - MODE_J_13) if w_j is not None else None),
                mode_x=bts.get('mode_x'), mode_j=bts.get('mode_j'),
                freq_x=bts.get('freq_x'), freq_j=bts.get('freq_j'))


V_A8 = verdict_of(A8C)
V_A1 = verdict_of(A1C)
if not G0p:
    verdict_same = verdict_fam = 'DEVICE_ANCHOR_FAILED'
    verdict_same_A1 = verdict_fam_A1 = 'DEVICE_ANCHOR_FAILED'
else:
    verdict_same = V_A8['verdict_same_coordinate']      # 主第三口径 = A8 逐层累积
    verdict_fam = V_A8['verdict_cross_family']
    verdict_same_A1 = ('PREFIX_ENDPOINT_NOT_DEGENERATE' if (endpoint_dev >= 1e-9)
                       else V_A1['verdict_same_coordinate'])
    verdict_fam_A1 = V_A1['verdict_cross_family']
w('  G0p = %s' % G0p)
w('  端点/恒等: A1 面板级 max|y01-1| = %s ; A1 逐对恒等 max rel dev = %.3e (n=%d)' %
  (('%.3e' % endpoint_dev) if endpoint_dev == endpoint_dev else 'NA', F30a_dev, _n_f30a))
w('  [A8 逐层累积 · 主第三口径] %s | %s' % (verdict_same, verdict_fam))
w('     share_x=%s w_x=%s | share_j=%s w_j=%s | P(>=.60) X=%s J=%s' %
  (V_A8['share_x'], V_A8['argmax_w_x'], V_A8['share_j'], V_A8['argmax_w_j'],
   V_A8['P_ge_060_x'], V_A8['P_ge_060_j']))
w('     d_x = %s (靶 %d) ; d_j = %s (靶 %d)' % (V_A8['d_x'], MODE_X_13, V_A8['d_j'], MODE_J_13))
w('  [A1 位置前缀 · 阴性对照] %s | %s' % (verdict_same_A1, verdict_fam_A1))
w('     share_x=%s w_x=%s | share_j=%s w_j=%s' % (V_A1['share_x'], V_A1['argmax_w_x'],
                                                  V_A1['share_j'], V_A1['argmax_w_j']))

'''
t = t[:j0] + NEW_V + t[j1:]

# ---------- (4) 预测 P6/P7 ----------
rep("""    w('')
    w('--- 预注册预测 ---')""",
    """    _a8y = [A8_CURVES[str(i)]['y'][-1] for i in range(len(_A8_SITES))]
    _a8mono = all(_a8y[i + 1] >= _a8y[i] - 1e-12 for i in range(len(_a8y) - 1))
    pc['P6'] = dict(desc=AM2['fix_3_new_arm']['predictions_for_A8']['P6'],
                    got=[float(v) for v in _a8y], monotone=bool(_a8mono), y_at_i0=float(_a8y[0]),
                    pass_=bool(_a8mono and _a8y[0] >= 0.95))
    _b8 = A8C.get('bootstrap') or {}
    pc['P7'] = dict(desc=AM2['fix_3_new_arm']['predictions_for_A8']['P7'],
                    got_mode=_b8.get('mode_x'), got_freq=_b8.get('freq_x'),
                    pass_=bool(_b8.get('mode_x') in (13, 14) and (_b8.get('freq_x') or 0) >= 0.50))
    w('')
    w('--- 预注册预测 ---')""")

# ---------- (5) result 落盘补 A8 ----------
rep("""    'A5_floor': A5,""",
    """    'A5_floor': A5,
    'A8_curves': A8_CURVES, 'A8_xhalf': A8_XH, 'A8_J': A8_J, 'A8_perpair': A8_PER,
    'A8_verdict': dict(V_A8=V_A8, V_A1=V_A1, endpoint_dev=(float(endpoint_dev) if endpoint_dev == endpoint_dev else None),
                       F30a_dev=float(F30a_dev), G0p=bool(G0p)),
    'A8_predictions': {k: pc[k] for k in ('P6', 'P7') if k in pc},""")

rep("""    'verdict': dict(G0p=bool(G0p), endpoint_dev=float(endpoint_dev),
                    verdict_same_coordinate=verdict_same,
                    mode_x=mode_x, mode_j=mode_j, freq_x=freq_x, freq_j=freq_j,
                    share_x=share_x, share_j=share_j, argmax_w_x=axw_x, argmax_w_j=axw_j,
                    P_ge_060_x=P_ge_x, P_le_040_x=P_le_x, P_ge_060_j=P_ge_j, P_le_040_j=P_le_j,
                    d_x=(abs(axw_x - MODE_X_13) if axw_x is not None else None),
                    d_j=(abs(axw_j - MODE_J_13) if axw_j is not None else None),
                    verdict_cross_family=verdict_fam,
                    verdict_position=pos_verdict, verdict_additivity=add_verdict),""",
    """    'verdict': dict(G0p=bool(G0p),
                    endpoint_dev=(float(endpoint_dev) if endpoint_dev == endpoint_dev else None),
                    verdict_same_coordinate=verdict_same,
                    verdict_cross_family=verdict_fam,
                    verdict_same_coordinate_A1=verdict_same_A1,
                    verdict_cross_family_A1=verdict_fam_A1,
                    primary_third_family='A8_cumulative_layer',
                    V_A8=V_A8, V_A1=V_A1,
                    verdict_position=pos_verdict, verdict_additivity=add_verdict),""")

rep("""                  XH_RANGE_12=XH_RANGE_12, pertlim=PERT_LIM, phaselabel='N2h1-alpha-7'),""",
    """                  XH_RANGE_12=XH_RANGE_12, pertlim=PERT_LIM, phaselabel='N2h1-alpha-7',
                  amend2_sha8=sha(AMEND2)[:8],
                  A8_endpoint_curve_present=True,
                  F30a_pairs_checked=int(_n_f30a), FULL_PANEL=bool(FULL_PANEL)),""")

rep("""                    MODE_X_13=MODE_X_13, MODE_J_13=MODE_J_13, XH_RANGE_12=XH_RANGE_12),""",
    """                    MODE_X_13=MODE_X_13, MODE_J_13=MODE_J_13, XH_RANGE_12=XH_RANGE_12),
    'amend2': dict(sha8=sha(AMEND2)[:8], sha256=sha(AMEND2), kind=AM2['kind'],
                   trigger=AM2['trigger']),""")

assert t != orig
rb = t.encode('utf-8')
io.open(P, 'wb').write(rb)
t2 = io.open(P, encoding='utf-8', newline='').read()
for k in ["F30a_dev", "def full_concentration", "A8C = full_concentration", "def verdict_of",
          "V_A8 = verdict_of(A8C)", "pc['P6']", "'A8_curves': A8_CURVES", "amend2_sha8"]:
    assert k in t2, 'MISS %r' % k
print('PATCH OK ; bytes %d -> %d ; sha8 %s' % (len(orig.encode('utf-8')), len(rb),
                                               hashlib.sha256(rb).hexdigest()[:8]))
