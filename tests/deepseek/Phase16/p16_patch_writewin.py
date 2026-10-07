# -*- coding: utf-8 -*-
"""Phase 16 主脚本补丁生成器：从 Phase 15 主脚本生成 N2h1a9 脚本。

纪律（模板 §1.1）：
  - 先做全局 P15T -> P16T 重命名（避免新旧临时目录互写）；
  - 每处替换都断言语料片段出现次数，防止子串误伤；
  - 冻结片段（装置 hooks / BASE / FULL_SWAP / E3 localize 与 B_cat）**逐字节不出现**在替换列表中；
  - 生成后 py_compile + 关键字残留扫描。
"""
import io
import os
import re
import sys
import py_compile

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'deepseek', 'Phase15', 'n2h1a8_cross_model_profile.py')
DST = os.path.join(ROOT, 'tests', 'deepseek', 'Phase16', 'n2h1a9_writewin_origin_profile.py')

s = io.open(SRC, encoding='utf-8', newline='').read()
n_p15t = s.count('P15T')
s = s.replace('P15T', 'P16T')
print('global P15T -> P16T : %d 处' % n_p15t)

REPS = []


def rep(old, new, cnt=1, tag=''):
    REPS.append((old, new, cnt, tag))


# ---------------- R1 路径块
rep("""ROOT = r'D:\\AI2050\\Ai2050-OpenOne'
P15 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase15')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
EXECP = os.path.join(P16T, 'execution_phase15.json')
SEALP = os.path.join(P16T, 'N2h1a8_design_seal.json')""",
"""ROOT = r'D:\\AI2050\\Ai2050-OpenOne'
P16 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase16')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
EXECP = os.path.join(P16T, 'execution_phase16.json')
SEALP = os.path.join(P16T, 'N2h1a9_design_seal.json')
ANCHP = os.path.join(P15T, 'result_phase15.json')
os.makedirs(P16T, exist_ok=True)""", 1, 'paths')

# ---------------- R2 amend1 -> 冻结锚
rep("""# amend1（词表常量勘误）：sup_id 必须逐臂由该臂 tokenizer 解析，禁止沿用源模型硬编码 id
AM1P = os.path.join(P16T, 'N2h1a8_design_seal_amend1.json')
AM1 = json.load(io.open(AM1P, encoding='utf-8'))
assert AM1['amend_of_seal_sha256'] == EX['seal_sha256'], \\
    'DRIFT: amend1 指向的 seal 与本 exec 不一致'""",
"""# Phase 15 result 作为**冻结锚**（P2 逐位复现对象）；其 sha 冻结在 seal/exec 中
assert os.path.exists(ANCHP), 'ANCHOR_MISSING: %s' % ANCHP
_ab = open(ANCHP, 'rb').read()
assert hashlib.sha256(_ab).hexdigest() == EX['anchor_result_sha256'], \\
    'DRIFT: Phase 15 result 锚漂移'
ANCH_ALL = json.loads(_ab.decode('utf-8'))
ANCH = EX['anchor_values']

# amend1（锚判据分层，见 tests/deepseek_temp/Phase16/N2h1a9_design_seal_amend1.json）
AM1P = os.path.join(P16T, 'N2h1a9_design_seal_amend1.json')
assert os.path.exists(AM1P), 'AMEND1_MISSING: %s' % AM1P
AM1 = json.load(io.open(AM1P, encoding='utf-8'))
assert AM1['amend_of_seal_sha256'] == EX['seal_sha256'], 'DRIFT: amend1 指向的 seal 与本 exec 不一致'""", 1, 'anchor')

# ---------------- R3 SMOKE 网格
rep("""if SMOKE:
    ALPHAS = [0.0, 0.5, 1.0]
    PROFILE = PROFILE[:3]
    CANDS = [4, 6, 20]
    BP = 200""",
"""if SMOKE:
    # SMOKE 必须覆盖三类位点：浅端 (1,2,3，检验写入窗下探)、写入窗本身 (6)、深端 (30,34)
    ALPHAS = [0.0, 0.5, 1.0]
    PROFILE = [1, 2, 3, 6, 30, 34]
    CANDS = [1, 4, 6, 20]
    BP = 200""", 1, 'smoke')

# ---------------- R4 新增统计量函数（插在 "- 单臂" 分界之前）
NEW_FUNCS = '''# ---------------------------------------------------------------- 集中度重设计（Phase 16）
# 结构性说明：置换零假设**保留 jump 的多重集**，故任何只依赖多重集的量（谱熵、max|d|/mean|d|、
# 参与比 sum|d|^2/(sum|d|)^2）在零假设下**恒等于观测值** -> 双边检验必得 p=1，属于**结构性退化族**。
# 可检验的量必须**顺序敏感**。本 Phase 用两个：
#   com_layer : |delta| 在**物理层号**轴上的质心（单位：层）。尺度无关；定义在物理轴上 => 对网格
#               加密/下探不变（这是 Phase 15 用 jump 序号归一化的版本所缺的性质）。
#   span_k    : 最大的 k 个 |delta| 所在步的跨度 / (n-1)。尺度无关；小 = 主变挤在少数相邻步。

def stat_com_layer(jumps, sites):
    j = np.asarray(jumps, float)
    if len(j) == 0 or len(sites) != len(j) + 1:
        return None, None
    mid = (np.asarray(sites, float)[:-1] + np.asarray(sites, float)[1:]) / 2.0
    a = np.abs(j)
    den = float(a.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None, None
    com_abs = float((a * mid).sum() / den)
    ds = float(j.sum())
    com_sgn = float((j * mid).sum() / ds) if abs(ds) > 1e-12 else None
    return com_abs, com_sgn


def stat_span_k(jumps, k=3):
    j = np.asarray(jumps, float)
    n = len(j)
    if n < k or not np.isfinite(j).all():
        return None
    idx = np.argsort(-np.abs(j))[:k]
    return float(idx.max() - idx.min()) / max(n - 1, 1)


def _ns_empty(n_bp, reason):
    """降级分支也必须返回**同一个 schema**（SMOKE 实测教训：缺键会在叙述行 KeyError）。"""
    return dict(BP=n_bp, n_ok=0, reason=reason,
                obs_com=None, com_p5=None, com_p95=None, com_tail=None,
                obs_com_signed=None, obs_span=None, span_p5=None, span_p95=None,
                span_tail=None, span_degenerate=None, com_ci=None, span_ci=None)


def perm_null_new(jumps, sites, rng_obj, n_bp, k=3):
    """同一置换协议（多重集随机排列）下 com_layer 与 span_k 的**双边**分位。"""
    j = np.asarray(jumps, float)
    n = len(j)
    if n < k + 1 or not np.isfinite(j).all():
        return _ns_empty(n_bp, 'bad_input(n=%d)' % n)
    obs_com, obs_com_sgn = stat_com_layer(j, sites)
    obs_span = stat_span_k(j, k)
    cs = np.full(n_bp, np.nan)
    ss = np.full(n_bp, np.nan)
    for b in range(n_bp):
        p = j[rng_obj.permutation(n)]
        c, _ = stat_com_layer(p, sites)
        cs[b] = c if c is not None else np.nan
        vv = stat_span_k(p, k)
        ss[b] = vv if vv is not None else np.nan
    fc = cs[np.isfinite(cs)]
    fs = ss[np.isfinite(ss)]
    if len(fc) == 0 or len(fs) == 0:
        return _ns_empty(n_bp, 'all_nan')
    p5c, p95c = float(np.percentile(fc, 5)), float(np.percentile(fc, 95))
    p5s, p95s = float(np.percentile(fs, 5)), float(np.percentile(fs, 95))
    tc = ('low' if (obs_com is not None and obs_com <= p5c) else
          'high' if (obs_com is not None and obs_com >= p95c) else 'none')
    ts = ('low' if (obs_span is not None and obs_span <= p5s) else
          'high' if (obs_span is not None and obs_span >= p95s) else 'none')
    return dict(BP=n_bp, n_ok=int(len(fc)),
                obs_com=obs_com, com_p5=p5c, com_p95=p95c, com_tail=tc,
                obs_com_signed=obs_com_sgn,
                obs_span=obs_span, span_p5=p5s, span_p95=p95s, span_tail=ts,
                span_degenerate=bool(p95s - p5s < 1e-12),
                com_ci=dict(lo=p5c, hi=p95c, med=float(np.median(fc))),
                span_ci=dict(lo=p5s, hi=p95s, med=float(np.median(fs))))


def slice_sites(F, sites, keep):
    """按位点集合取子向量（保持原顺序）。"""
    idx = [i for i, sv in enumerate(sites) if int(sv) in keep]
    return np.array([F[i] for i in idx], float), [int(sites[i]) for i in idx]


'''
rep("# ---------------------------------------------------------------- 单臂\ndef run_arm(arm_id, acfg):",
    NEW_FUNCS + "# ---------------------------------------------------------------- 单臂\ndef run_arm(arm_id, acfg):",
    1, 'new_funcs')

rep("""def flush_log(name):
    io.open(os.path.join(P16T, name), 'w', encoding='utf-8', newline='\\n').write('\\n'.join(_log) + '\\n')""",
    """def flush_log(name):
    io.open(os.path.join(P16T, name), 'w', encoding='utf-8', newline='\\n').write('\\n'.join(_log) + '\\n')


def F3(v, nd=3):
    \"\"\"None-安全的数值格式化（叙述行一律走它，避免降级分支 KeyError/TypeError）。\"\"\"
    try:
        return ('%.*f' % (nd, float(v))) if (v is not None and np.isfinite(float(v))) else 'None'
    except Exception:
        return 'None'""", 1, 'F3_helper')

# ---------------- R5 E4_summary 增加 legacy XH_RANGE（在 dict 之前先切子域）
rep("""    xh_sites = [PROFILE[i] for i in range(len(PROFILE)) if np.isfinite(xh[i])]""",
    """    xh_sites = [PROFILE[i] for i in range(len(PROFILE)) if np.isfinite(xh[i])]
    LEGACY_SET = set(int(x) for x in EX['profile_sites_legacy'])
    _li = [i for i, sv in enumerate(PROFILE) if int(sv) in LEGACY_SET]
    xh_leg = np.array([xh[i] for i in _li], float)
    Jv_leg = np.array([Jv[i] for i in _li], float)
    sites_leg = [int(PROFILE[i]) for i in _li]
    w('  E4b legacy 子域 n=%d (sites %s..%s)' % (len(sites_leg), sites_leg[0], sites_leg[-1]))""",
    1, 'legacy_slice')

rep("""                             XH_RANGE=(float(np.nanmax(xh) - np.nanmin(xh)) if np.isfinite(xh).any() else None),""",
    """                             XH_RANGE=(float(np.nanmax(xh) - np.nanmin(xh)) if np.isfinite(xh).any() else None),
                             XH_RANGE_legacy=(float(np.nanmax(xh_leg) - np.nanmin(xh_leg)) if np.isfinite(xh_leg).any() else None),""",
    1, 'xh_range_legacy')

# ---------------- R6 E5 整块替换（三口径 + E7 可达性）
OLD_E5_START = "    # --- E5 集中度 + 置换零假设（零额外前向）"
_i0 = s.find(OLD_E5_START)
assert _i0 > 0, '找不到 E5 块起点'
_i1 = s.find("    # α=1 的 y 值（recover）", _i0)
assert _i1 > _i0, '找不到 E5 块终点'
OLD_E5 = s[_i0:_i1]
# 逐字节断言：旧块必须包含这 4 个特征串（防止拿到别的段落）
for _frag in ['share_x, axw_x, jx = conc_hat(xh, W)',
              "nx = perm_null(jx, rng_n, W, BP, range_x)",
              "rec['E5_concentration']['d_argmax_window']",
              "null95_x=%s margin_x=%s"]:
    assert _frag in OLD_E5, 'E5 块特征串缺失: %s' % _frag
rep(OLD_E5, '''    # --- E7 可达性剖面（rho(ell) = Y(ell, alpha=1)）与可达域 REACH
    a1i = list(ALPHAS).index(1.0) if 1.0 in ALPHAS else None
    rho = (Y[:, a1i] if a1i is not None else np.full(len(PROFILE), np.nan))
    UNR = float(FL['UNREACH_y'])
    REACH = [int(sv) for i, sv in enumerate(PROFILE)
             if np.isfinite(rho[i]) and float(rho[i]) >= UNR]
    EXCL = [int(sv) for sv in PROFILE if int(sv) not in REACH]
    _cross = [int(sv) for i, sv in enumerate(PROFILE)
              if np.isfinite(rho[i]) and float(rho[i]) >= 0.5]
    ell_reach = (min(_cross) if _cross else None)
    rec['E7_reach'] = dict(sites=[int(sv) for sv in PROFILE], rho=[float(v) for v in rho],
                           unreach_y=UNR, reach=REACH, excluded=EXCL,
                           ell_reach=ell_reach, full_swap=FULL_SWAP)
    w('  E7 可达性: REACH n=%d ; EXCL(%d)=%s ; ell_reach(rho>=0.5)=%s' %
      (len(REACH), len(EXCL), EXCL, ell_reach))
    w('     rho: %s' % ' '.join('L%d:%.4f' % (PROFILE[i], rho[i]) for i in range(len(PROFILE))))

    # --- E5 集中度（三口径）
    #   legacy_domain : 与 Phase 15 完全相同的 6..34 子域 + 旧量 top3_share + 冻结种子 -> P2 锚复现
    #   main_domain   : 可达域 REACH 上的旧量（对照）
    #   new_stat      : 可达域 REACH 上的新量 (com_layer, span_k)，双边分位
    RSET = set(REACH)
    xh_m, sites_m = slice_sites(xh, PROFILE, RSET)
    Jv_m, _ = slice_sites(Jv, PROFILE, RSET)

    def _conc(F, sites_v, rng_obj):
        share, axw, jm = conc_hat(F, W)
        rng_v = (float(np.nanmax(F) - np.nanmin(F)) if (len(F) and np.isfinite(F).any())
                 else float('nan'))
        nn = (perm_null(jm, rng_obj, W, BP, rng_v)
              if (np.isfinite(jm).all() and np.isfinite(rng_v)) else dict(BP=BP, null95=None))
        mg = ((share - nn['null95']) if (share is not None and nn.get('null95') is not None)
              else None)
        return dict(sites=[int(v) for v in sites_v], F=[float(v) for v in F],
                    jumps=[float(v) for v in jm], top3=share, argmax_w=axw, range=rng_v,
                    null=nn, margin=mg,
                    win_sem=(None if axw is None else
                             dict(w=axw, a=int(sites_v[axw]),
                                  b=int(sites_v[min(axw + W, len(sites_v) - 1)]))),
                    spearman_depth=spearman(list(F), [int(v) for v in sites_v]))

    rngA = np.random.default_rng(SEED + 13)   # 与 Phase 15 A0 旧量完全同一协议
    rngB = np.random.default_rng(SEED + 29)
    rngC = np.random.default_rng(SEED + 41)
    rngD = np.random.default_rng(SEED + 53)
    leg_x = _conc(xh_leg, sites_leg, rngA)
    leg_j = _conc(Jv_leg, sites_leg, rngB)
    main_x = _conc(xh_m, sites_m, rngC)
    main_j = _conc(Jv_m, sites_m, rngD)
    new_x = perm_null_new(np.diff(xh_m), sites_m, np.random.default_rng(SEED + 61), BP, W)
    new_j = perm_null_new(np.diff(Jv_m), sites_m, np.random.default_rng(SEED + 67), BP, W)
    rec['E5_concentration'] = dict(
        W=W, alpha_grid=ALPHAS,
        legacy_domain=dict(
            x=leg_x, j=leg_j,
            d_argmax_window=(abs(int(leg_x['argmax_w']) - int(leg_j['argmax_w']))
                             if (leg_x['argmax_w'] is not None and leg_j['argmax_w'] is not None)
                             else None)),
        main_domain=dict(
            x=main_x, j=main_j,
            d_argmax_window=(abs(int(main_x['argmax_w']) - int(main_j['argmax_w']))
                             if (main_x['argmax_w'] is not None and main_j['argmax_w'] is not None)
                             else None)),
        new_stat=dict(x=new_x, j=new_j, k=W))
    w('  E5 legacy 域(%d 步) top3_x=%s (w=%s) top3_j=%s (w=%s)' %
      (len(leg_x['jumps']), leg_x['top3'], leg_x['argmax_w'], leg_j['top3'], leg_j['argmax_w']))
    w('     legacy null95_x=%s margin_x=%s ; null95_j=%s margin_j=%s' %
      (leg_x['null'].get('null95'), leg_x['margin'], leg_j['null'].get('null95'), leg_j['margin']))
    w('  E5 主域(%d 步) 旧量 top3_x=%s(w=%s) top3_j=%s(w=%s) -> sig=%s' %
      (len(main_x['jumps']), main_x['top3'], main_x['argmax_w'], main_j['top3'], main_j['argmax_w'],
       [k for k in ('x', 'j') if (dict(x=main_x, j=main_j)[k]['margin'] or -1) > 0]))
    w('  E5 新量 com_x=%s [%s, %s] tail=%s ; com_j=%s [%s, %s] tail=%s' %
      (F3(new_x.get('obs_com')), F3(new_x.get('com_p5')), F3(new_x.get('com_p95')),
       new_x.get('com_tail'), F3(new_j.get('obs_com')), F3(new_j.get('com_p5')),
       F3(new_j.get('com_p95')), new_j.get('com_tail')))
    w('     span_x=%s tail=%s ; span_j=%s tail=%s ; com_sep=%s 层' %
      (F3(new_x.get('obs_span'), 4), new_x.get('span_tail'),
       F3(new_j.get('obs_span'), 4), new_j.get('span_tail'),
       F3((None if (new_x.get('obs_com') is None or new_j.get('obs_com') is None)
           else new_x['obs_com'] - new_j['obs_com']))))
    w('     新量降级原因: x=%s / j=%s' % (new_x.get('reason'), new_j.get('reason')))

''', 1, 'E5E7')

# ---------------- R7 E6 校准：旧量改走 legacy 域 + bf16 只比 6..34
rep("""        share_x_12, axw_x_12v, _ = conc_hat(np.array([XH12[s] for s in PROFILE], float), W)""",
    """        share_x = leg_x['top3']; axw_x = leg_x['argmax_w']
        _L6 = [int(s) for s in PROFILE if int(s) in XH12]
        share_x_12, axw_x_12v, _ = conc_hat(np.array([XH12[s] for s in _L6], float), W)""",
    1, 'e6_share')
rep("""            XH_RANGE_nf4=rec['E4_summary']['XH_RANGE'], XH_RANGE_bf16=float(INH['XH_RANGE_12']))""",
    """            XH_RANGE_nf4=rec['E4_summary']['XH_RANGE_legacy'], XH_RANGE_bf16=float(INH['XH_RANGE_12']))""",
    1, 'e6_range')
rep("""           rec['E4_summary']['XH_RANGE'] or float('nan'), rec['E6_calibration']['XH_RANGE_bf16']))""",
    """           rec['E4_summary']['XH_RANGE_legacy'] or float('nan'), rec['E6_calibration']['XH_RANGE_bf16']))""",
    1, 'e6_range2')

# ---------------- R8 partial / 报告 / 结果 文件名
rep("""            pp = os.path.join(P16T, '_armrec_%s.json' % arm_id)""",
    """            pp = os.path.join(P16T, '_armrec16_%s.json' % arm_id)""", 1, 'merge_partial')
rep("""        pp = os.path.join(P16T, '_armrec_%s.json' % arm_id)
        io.open(pp, 'w', encoding='utf-8', newline='\\n').write(json.dumps(rec, ensure_ascii=False))""",
    """        pp = os.path.join(P16T, '_armrec16_%s.json' % arm_id)
        io.open(pp, 'w', encoding='utf-8', newline='\\n').write(json.dumps(rec, ensure_ascii=False))""",
    1, 'partial')
rep("""    OUT = os.path.join(P16T, 'result_phase15.json' if not SMOKE else 'result_phase15_smoke.json')""",
    """    OUT = os.path.join(P16T, 'result_phase16.json' if not SMOKE else 'result_phase16_smoke.json')""",
    1, 'out')
rep("""        io.open(os.path.join(P16T, 'n2h1a8_report_%s.txt' % arm_id), 'w',""",
    """        io.open(os.path.join(P16T, 'n2h1a9_report_%s.txt' % arm_id), 'w',""", 1, 'report')

# ---------------- R9 per_arm_verdict 整块替换
_i0 = s.find("def per_arm_verdict(rec):")
_i1 = s.find("def main():", _i0)
assert _i0 > 0 and _i1 > _i0
OLD_PAV = s[_i0:_i1]
for _frag in ["v['Q2_d_argmax'] = d", "v['Q3_label'] = ('NA' if n95x is None", "v['Q1_label'] = ('NF4_FAITHFUL'"]:
    assert _frag in OLD_PAV, 'per_arm_verdict 特征串缺失: %s' % _frag
rep(OLD_PAV, '''def per_arm_verdict(rec, anch):
    c = rec.get('E5_concentration') or {}
    e6 = rec.get('E6_calibration')
    e3 = rec.get('E3_localize') or {}
    e7 = rec.get('E7_reach') or {}
    ld = c.get('legacy_domain') or {}
    md = c.get('main_domain') or {}
    ns = c.get('new_stat') or {}
    v = {}
    v['Q0_device'] = ('PASS' if (rec.get('F1b_ok') and rec.get('T2_only') and rec.get('F4_dims_ok') and
                                 rec.get('F2_base_ok') and
                                 rec['E0_selfcheck']['determinism_maxdiff'] == 0 and
                                 rec['E0_selfcheck']['hook_effect_maxdiff'] > 0) else 'FAIL')
    # ---- Q1 冻结锚复现（legacy 6..34 域）
    lx = ((ld.get('x') or {}).get('F') or [])
    lj = ((ld.get('j') or {}).get('F') or [])
    ls = ((ld.get('x') or {}).get('sites') or [])
    axh = {str(k): float(val) for k, val in (anch.get('xhalf_by_site') or {}).items()}
    aJ = {str(k): float(val) for k, val in (anch.get('J_by_site') or {}).items()}
    dx = [abs(lx[i] - axh[str(ls[i])]) for i in range(len(ls))
          if i < len(lx) and np.isfinite(lx[i]) and str(ls[i]) in axh]
    dJ = [abs(lj[i] - aJ[str(ls[i])]) / max(abs(aJ[str(ls[i])]), 1e-9) for i in range(len(ls))
          if i < len(lj) and np.isfinite(lj[i]) and str(ls[i]) in aJ]
    v['Q1_n_sites'] = len(ls)
    v['Q1_max_abs_dxh_leg'] = (float(max(dx)) if dx else None)
    v['Q1_max_rel_dJ_leg'] = (float(max(dJ)) if dJ else None)
    v['Q1_argmax_same_leg'] = bool(ls and (ld.get('x') or {}).get('argmax_w') is not None and
                                   (ld.get('x') or {}).get('argmax_w') == anch.get('legacy_argmax_w_x') and
                                   (ld.get('j') or {}).get('argmax_w') is not None and
                                   (ld.get('j') or {}).get('argmax_w') == anch.get('legacy_argmax_w_j'))
    v['Q1_label'] = 'RECON_DRIFT'
    _t3x = (ld.get('x') or {}).get('top3')
    _t3j = (ld.get('j') or {}).get('top3')
    _dt3x = (abs(_t3x - float(anch['legacy_top3_x']))
             if (_t3x is not None and anch.get('legacy_top3_x') is not None) else None)
    _dt3j = (abs(_t3j - float(anch['legacy_top3_j']))
             if (_t3j is not None and anch.get('legacy_top3_j') is not None) else None)
    v['Q1_dtop3_x'] = _dt3x
    v['Q1_dtop3_j'] = _dt3j
    _fn = AM1['floors']
    v['Q1_dxh_by_site'] = {str(ls[i]): float(dx[i]) for i in range(min(len(dx), len(ls)))}
    _base_ok = bool(dx and dJ and _dt3x is not None and _dt3j is not None and
                    max(dJ) <= _fn['RECON_TOL_J_REL'] and _dt3x <= _fn['RECON_TOL_TOP3'] and
                    _dt3j <= _fn['RECON_TOL_TOP3'] and v['Q1_argmax_same_leg'])
    if _base_ok and max(dx) <= _fn['RECON_TOL_XH_STRICT']:
        v['Q1_label'] = 'RECON_OK'
    elif _base_ok and max(dx) <= _fn['RECON_TOL_XH_LOOSE']:
        v['Q1_label'] = 'RECON_OK_LOOSE'
    else:
        v['Q1_label'] = 'RECON_DRIFT'
    # ---- Q2 可达域左端点 == 写入窗
    v['Q2_L_star_own'] = e3.get('L_star_own')
    v['Q2_ell_reach'] = e7.get('ell_reach')
    v['Q2_excluded'] = e7.get('excluded')
    v['Q2_label'] = ('NA' if (v['Q2_L_star_own'] is None or v['Q2_ell_reach'] is None) else
                     ('REACH_EQ_WRITEWIN' if int(v['Q2_ell_reach']) == int(v['Q2_L_star_own'])
                      else 'REACH_OFFSET(d=%+d)' % (int(v['Q2_ell_reach']) - int(v['Q2_L_star_own']))))
    # ---- Q3 写入窗入域
    _sites_all = e7.get('sites') or []
    v['Q3_label'] = ('NA' if (v['Q2_L_star_own'] is None or not _sites_all) else
                     ('WIN_IN_DOMAIN' if min(_sites_all) <= int(v['Q2_L_star_own']) <= max(_sites_all)
                      else 'WIN_OUT_OF_DOMAIN'))
    # ---- Q4 物理深度质心分离
    cx = (ns.get('x') or {}).get('obs_com')
    cj = (ns.get('j') or {}).get('obs_com')
    v['Q4_com_x'] = cx
    v['Q4_com_j'] = cj
    v['Q4_sep'] = (float(cx) - float(cj)) if (cx is not None and cj is not None) else None
    v['Q4_label'] = ('NA' if v['Q4_sep'] is None else
                     ('CENTROID_SEPARATED' if v['Q4_sep'] >= FL['CENTROID_SEP_MIN']
                      else 'CENTROID_OVERLAP'))
    v['Q4_com_after_win_x'] = (float(cx) - int(v['Q2_L_star_own'])
                               if (cx is not None and v['Q2_L_star_own'] is not None) else None)
    # ---- Q5 重设计有效性（主域上：旧量单边 vs 新量双边）
    _md = dict(x=md.get('x') or {}, j=md.get('j') or {})
    _ns = dict(x=ns.get('x') or {}, j=ns.get('j') or {})
    v['Q5_old_sig'] = [k for k in ('x', 'j')
                       if _md[k].get('margin') is not None and _md[k]['margin'] > 0]
    v['Q5_new_sig'] = [k for k in ('x', 'j')
                       if _ns[k].get('com_tail') not in (None, 'none')]
    v['Q5_new_sig_any'] = [k for k in ('x', 'j')
                           if _ns[k].get('com_tail') not in (None, 'none')
                           or _ns[k].get('span_tail') not in (None, 'none')]
    v['Q5_old_margins'] = {k: _md[k].get('margin') for k in ('x', 'j')}
    v['Q5_new_tails'] = {k: _ns[k].get('com_tail') for k in ('x', 'j')}
    v['Q5_new_spans'] = {k: _ns[k].get('obs_span') for k in ('x', 'j')}
    if len(v['Q5_new_sig']) >= len(v['Q5_old_sig']) and len(v['Q5_new_sig']) >= FL['NEW_NONDEG_MIN']:
        v['Q5_label'] = 'STAT_REDESIGN_EFFECTIVE'
    elif len(v['Q5_new_sig']) >= len(v['Q5_old_sig']):
        v['Q5_label'] = 'STAT_REDESIGN_PARTIAL'
    else:
        v['Q5_label'] = 'STAT_REDESIGN_EQUIVALENT'
    if e6 is not None:
        v['Q6_e6_label'] = ('NF4_FAITHFUL' if e6.get('pass_tol') else 'NF4_DEVIANT')
        v['Q6_e6_max_abs_dxh'] = e6.get('max_abs_dxh')
    return v


''', 1, 'per_arm_verdict')

# ---------------- R10 joint 判决块
rep("""    rep = [a for a in ['A1_glm4-9b-nf4', 'A2_qwen3-14b-nf4'] if a in verdict]
    labels_ok = [a for a in rep if verdict[a].get('Q0_device') == 'PASS']
    q2 = [verdict[a].get('Q2_label') for a in labels_ok]
    q3 = [verdict[a].get('Q3_label') for a in labels_ok]
    joint = dict(
        arms_present=list(results.keys()), arms_used_for_cross_model=labels_ok,
        Q2_joint=('ARGS_GAP_LAYERSTACK' if (q2 and all(x == 'ARGS_GAP_GE3' for x in q2)) else
                  'ARGS_GAP_4B_SPECIFIC' if (q2 and all(x == 'ARGS_GAP_LT3' for x in q2)) else
                  'ARGS_GAP_MIXED' if q2 else 'NA'),
        Q3_joint=('CONC_JUDGE_INVALID_X_ALL' if (q3 and all(x == 'NULL_X_HIGH' for x in q3)) else
                  'CONC_JUDGE_ALIVE_X' if q3 else 'NA'),
    )""",
"""    rep = [a for a in ['A1_glm4-9b-nf4', 'A2_qwen3-14b-nf4'] if a in verdict]
    labels_ok = [a for a in verdict if verdict[a].get('Q0_device') == 'PASS']
    q1 = [verdict[a].get('Q1_label') for a in labels_ok]
    q2 = [verdict[a].get('Q2_label') for a in labels_ok]
    q3 = [verdict[a].get('Q3_label') for a in labels_ok]
    q4 = [verdict[a].get('Q4_label') for a in labels_ok]
    q5o = [len(verdict[a].get('Q5_old_sig') or []) for a in labels_ok]
    q5n = [len(verdict[a].get('Q5_new_sig') or []) for a in labels_ok]
    joint = dict(
        arms_present=list(results.keys()), arms_used_for_cross_model=labels_ok,
        Q1_joint=('ANCHOR_ROBUST' if (q1 and all(x == 'RECON_OK' for x in q1)) else
                  'ANCHOR_ROBUST_TIERED' if (q1 and all(x in ('RECON_OK', 'RECON_OK_LOOSE') for x in q1)
                                             and sum(1 for x in q1 if x == 'RECON_OK') >= 2) else
                  'ANCHOR_PARTIAL' if (q1 and any(x in ('RECON_OK', 'RECON_OK_LOOSE') for x in q1)) else
                  'ANCHOR_FAIL' if q1 else 'NA'),
        Q2_joint=('REACH_IDENTITY_ROBUST' if (q2 and all(x == 'REACH_EQ_WRITEWIN' for x in q2)) else
                  'REACH_IDENTITY_PARTIAL' if (q2 and any(x == 'REACH_EQ_WRITEWIN' for x in q2)) else
                  'REACH_IDENTITY_FAIL' if q2 else 'NA'),
        Q3_joint=('WIN_IN_DOMAIN_ALL' if (q3 and all(x == 'WIN_IN_DOMAIN' for x in q3)) else
                  'WIN_DOMAIN_MIXED' if q3 else 'NA'),
        Q4_joint=('CENTROID_SEPARATED_ALL' if (q4 and all(x == 'CENTROID_SEPARATED' for x in q4)) else
                  'CENTROID_PARTIAL' if (q4 and any(x == 'CENTROID_SEPARATED' for x in q4)) else
                  'CENTROID_OVERLAP' if q4 else 'NA'),
        Q5_joint=('STAT_REDESIGN_EFFECTIVE' if (q5n and all(n >= o for n, o in zip(q5n, q5o))
                                                and max(q5n) >= FL['NEW_NONDEG_MIN']) else
                  'STAT_REDESIGN_EQUIVALENT' if q5n else 'NA'),
        Q5_counts=dict(old_sig=q5o, new_sig=q5n),
    )""", 1, 'joint')

# ---------------- R11 verdict 调用签名（传到 per_arm_verdict 的锚）
rep("""            verdict[arm_id] = per_arm_verdict(rec)""",
    """            verdict[arm_id] = per_arm_verdict(rec, ANCH[arm_id])""", 1, 'verdict_call')

# ---------------- R12 RESULT dict
rep("""    RESULT = dict(phase=15, smoke=SMOKE, execution_sha256=sha(EXECP), seal_sha256=EX['seal_sha256'],
                  grid=dict(profile_sites=PROFILE, alphas=ALPHAS, W=W, BP=BP, xh_frac=XHF,
                            cands=list(EX['localize']['cands'])),""",
    """    RESULT = dict(phase=16, smoke=SMOKE, execution_sha256=sha(EXECP), seal_sha256=EX['seal_sha256'],
                  grid=dict(profile_sites=PROFILE, profile_sites_legacy=EX['profile_sites_legacy'],
                            alphas=ALPHAS, W=W, BP=BP, xh_frac=XHF,
                            cands=list(EX['localize']['cands'])),""", 1, 'result_head')
rep("""                  E5_concentration={k: v.get('E5_concentration') for k, v in results.items() if 'error' not in v},""",
    """                  E5_concentration={k: v.get('E5_concentration') for k, v in results.items() if 'error' not in v},
                  E7_reach={k: v.get('E7_reach') for k, v in results.items() if 'error' not in v},""",
    1, 'result_e7')
rep("""                  amend1_sha256=sha(AM1P), amend1_sha8=sha(AM1P)[:8],
                  amend1_kind=AM1['kind'],""",
    """                  anchor_result_sha256=EX['anchor_result_sha256'],
                  anchor_result_sha8=EX['anchor_result_sha256'][:8],
                  anchor_phase=15,
                  amend1_sha256=hashlib.sha256(open(AM1P, 'rb').read()).hexdigest(),
                  amend1_sha8=hashlib.sha256(open(AM1P, 'rb').read()).hexdigest()[:8],
                  amend1_kind=AM1['kind'],""", 1, 'result_anchor')

# ---------------- R13 预测块
_i0 = s.find("    pc['P1'] = dict(")
_i1 = s.find("    RESULT['predictions_check'] = pc", _i0)
assert _i0 > 0 and _i1 > _i0
OLD_PC = s[_i0:_i1]
for _frag in ["pc['P3'] = dict(pass_=all((x is not None and x >= 0.60)",
              "pc['P7'] = dict(pass_=all((x is not None and band[0] <= x <= band[1])"]:
    assert _frag in OLD_PC, '预测块特征串缺失: %s' % _frag
rep(OLD_PC, """    pc['P1'] = dict(pass_=bool(dev_ok), detail='装置自检 ok_arms=%s err_arms=%s' % (ok_arms, err_arms))
    # P2 冻结锚逐位复现（三臂 legacy 6..34；amend1 分层判据）
    _r1 = {k: verdict[k].get('Q1_label') for k in ok_arms}
    _n_ok = sum(1 for k in ok_arms if verdict[k].get('Q1_label') == 'RECON_OK')
    _n_loose = sum(1 for k in ok_arms if verdict[k].get('Q1_label') == 'RECON_OK_LOOSE')
    pc['P2'] = dict(pass_=bool(len(ok_arms) == 3 and
                               all(v in ('RECON_OK', 'RECON_OK_LOOSE') for v in _r1.values()) and
                               _n_ok >= 2),
                    detail=dict(labels=_r1, n_strict_ok=_n_ok, n_loose=_n_loose,
                                max_abs_dxh={k: verdict[k].get('Q1_max_abs_dxh_leg') for k in ok_arms},
                                max_rel_dJ={k: verdict[k].get('Q1_max_rel_dJ_leg') for k in ok_arms},
                                dtop3={k: [verdict[k].get('Q1_dtop3_x'), verdict[k].get('Q1_dtop3_j')] for k in ok_arms},
                                argmax_same={k: verdict[k].get('Q1_argmax_same_leg') for k in ok_arms},
                                dxh_by_site={k: verdict[k].get('Q1_dxh_by_site') for k in ok_arms},
                                tiers=AM1['floors'], p2_criterion=AM1['p2_criterion']))
    # P3 写入窗入域且重算与 Phase 15 一致
    _ls_now = {k: verdict[k].get('Q2_L_star_own') for k in ok_arms}
    _ls_15 = {k: int(ANCH[k]['L_star_own']) for k in ok_arms}
    _dom = {k: verdict[k].get('Q3_label') for k in ok_arms}
    pc['P3'] = dict(pass_=bool(ok_arms and all(_ls_now[k] == _ls_15[k] for k in ok_arms)
                               and all(_dom[k] == 'WIN_IN_DOMAIN' for k in ok_arms)),
                    detail=dict(L_star_now=_ls_now, L_star_15=_ls_15, domain=_dom,
                                profile_lo=int(min(PROFILE)), profile_hi=int(max(PROFILE))))
    # P4 可达域左端点 == 写入窗（严格相等）
    _er = {k: verdict[k].get('Q2_ell_reach') for k in ok_arms}
    pc['P4'] = dict(pass_=bool(ok_arms and all(verdict[k].get('Q2_label') == 'REACH_EQ_WRITEWIN'
                                               for k in ok_arms)),
                    detail=dict(ell_reach=_er, L_star=_ls_now, excluded={k: verdict[k].get('Q2_excluded') for k in ok_arms},
                                labels={k: verdict[k].get('Q2_label') for k in ok_arms}))
    # P5 新量不劣于旧量
    _n_old = sum(len(verdict[k].get('Q5_old_sig') or []) for k in ok_arms)
    _n_new = sum(len(verdict[k].get('Q5_new_sig') or []) for k in ok_arms)
    pc['P5'] = dict(pass_=bool(_n_new >= _n_old and _n_new >= FL['NEW_NONDEG_MIN']),
                    detail=dict(n_old_sig=_n_old, n_new_sig=_n_new, min_required=FL['NEW_NONDEG_MIN'],
                                old_margins={k: verdict[k].get('Q5_old_margins') for k in ok_arms},
                                new_tails={k: verdict[k].get('Q5_new_tails') for k in ok_arms}))
    # P6 物理深度质心分离 >= 4.0 层（3/3）
    _sep = {k: verdict[k].get('Q4_sep') for k in ok_arms}
    pc['P6'] = dict(pass_=bool(ok_arms and all((v is not None and v >= FL['CENTROID_SEP_MIN'])
                                               for v in _sep.values())),
                    detail=dict(sep_layers=_sep, com_x={k: verdict[k].get('Q4_com_x') for k in ok_arms},
                                com_j={k: verdict[k].get('Q4_com_j') for k in ok_arms},
                                threshold=FL['CENTROID_SEP_MIN']))
    # P7 xhalf 质心位于写入窗之后 >= 5 层（>= 2/3）
    _aft = {k: verdict[k].get('Q4_com_after_win_x') for k in ok_arms}
    _naft = sum(1 for v in _aft.values() if v is not None and v >= FL['CENTROID_AFTER_WIN_MIN'])
    pc['P7'] = dict(pass_=bool(_naft >= 2 and len(_aft) == 3),
                    detail=dict(after_win_layers=_aft, n_pass=_naft, n_arms=len(_aft),
                                threshold=FL['CENTROID_AFTER_WIN_MIN']))
""", 1, 'predictions')

# ---------------- R14 文件头 + main 标题 + joint 打印
rep('''"""
Phase 15 (N2h1-alpha-8) 主脚本：跨模型复算「统一剖面」。''',
    '''"""
Phase 16 (N2h1-alpha-9) 主脚本：写入窗原点化剖面 + 集中度统计量重设计。''', 1, 'docstring')
rep("""          -> E5 集中度 + 置换零假设 -> (A0) E6 量化保真校准""",
    """          -> E5 集中度（legacy 域 / 主域旧量 / 主域新量）+ E7 可达性剖面 -> (A0) E6 量化保真校准

本 Phase 相对 Phase 15 的**唯一**改动：
  1) profile_sites 下探到 1..5（使 A1 的 L*=3 / A2 的 L*=4 落入剖面域）；
  2) 以可达性掩膜 REACH = {ell : rho(ell) >= UNREACH_y} 定义主域，使写入窗成为主域左端点；
  3) 集中度重设计：以顺序敏感的 (com_layer, span_k) 替换极值型 top3_share，并做**双边**置换检验；
     同时保留 legacy 6..34 域上的旧量用于 P2 冻结锚逐位复现。
装置（hooks / BASE / FULL_SWAP / E3 localize / B_cat）与 Phase 15 逐字节相同。""", 1, 'docstring2')
rep("""    w('Phase 15 (N2h1-alpha-8) 跨模型复算「统一剖面」')""",
    """    w('Phase 16 (N2h1-alpha-9) 写入窗原点化剖面 + 集中度重设计')""", 1, 'main_title')
rep("""    w('in 继承参照: d_argmax_4B=%d ; null95 由本 Phase 现场重算' %
      abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13'])))""",
    """    w('冻结锚: Phase 15 result sha8 = %s ; 三臂 L*_own=%s' %
      (EX['anchor_result_sha256'][:8], {k: ANCH[k]['L_star_own'] for k in EX['arm_order']}))
    w('新增网格: 浅端 %s ; 主域 = REACH(rho>=%.2f) ; 新量 = com_layer / span_k' %
      ([s for s in PROFILE if s < 6], FL['UNREACH_y']))""", 1, 'main_anchor_log')
rep("""    w('joint_verdict: Q2=%s ; Q3=%s' % (joint['Q2_joint'], joint['Q3_joint']))""",
    """    w('joint_verdict: Q1=%s ; Q2=%s ; Q3=%s ; Q4=%s ; Q5=%s' %
      (joint['Q1_joint'], joint['Q2_joint'], joint['Q3_joint'], joint['Q4_joint'], joint['Q5_joint']))""",
    1, 'joint_log')
rep("""                     'E5: %s' % rec['E5_concentration'],""",
    """                     'E5: %s' % rec['E5_concentration'],
                     'E7: %s' % rec.get('E7_reach'),""", 1, 'report_e7')

rep("""        rep_lines = ['# Phase 15 report %s (%s)' % (arm_id, rec['model']),""",
    """        rep_lines = ['# Phase 16 report %s (%s)' % (arm_id, rec['model']),""", 1, 'report_hdr')
rep("""用法：SMOKE=1 python n2h1a8_cross_model_profile.py    (仅 A0，缩小网格)
      python n2h1a8_cross_model_profile.py            (三臂正式)""",
    """用法：SMOKE=1 python n2h1a9_writewin_origin_profile.py   (仅 A0，浅+深混合小网格)
      python n2h1a9_writewin_origin_profile.py           (三臂正式；建议 SPLIT_PARTIAL=1 逐臂进程隔离)""",
    1, 'usage')

# ---------------- 应用
for old, new, cnt, tag in REPS:
    c = s.count(old)
    assert c == cnt, '[%s] 期望 %d 次，实测 %d 次 | head=%r' % (tag, cnt, c, old[:90])
    s = s.replace(old, new)
    print('  [ok] %-16s x%d' % (tag, cnt))

io.open(DST, 'w', encoding='utf-8', newline='\n').write(s)

# ---------------- 残留扫描
bad = []
for pat in ['P15T', 'execution_phase15.json', 'result_phase15.json', 'n2h1a8_report',
            'N2h1a8_design_seal.json', "phase=15", "AM1P", "amend1_sha256",
            'share_x, axw_x, jx = conc_hat', "per_arm_verdict(rec)"]:
    if pat in s and pat not in ('P15T', 'execution_phase15.json', 'result_phase15.json'):
        bad.append(pat)
for pat in ['P15T', 'execution_phase15.json', 'result_phase15.json']:
    if pat in s and s.count(pat) and pat == 'P15T':
        # P15T 只允许出现在 ANCHP 定义里
        if s.count('P15T') != 2:
            bad.append('P15T x%d' % s.count('P15T'))
print('残留可疑:', bad if bad else 'NONE')

# 冻结片段必须逐字节未变
KEEP = ['def fwd_patch(text, site, vec):', 'def capture(text):',
        'FS_PAIR[rw] = float(score_of(CAP[dw][1], ds, ids_of(dw)[0]) - BASE[rw][\'sd0\'])',
        'Ub, sv, ncls = est_U(l + 1)', 'vec = hr + proj((hd - hr).astype(np.float32), Ub)',
        'lg = fwd_patch(TMPL % rw, site, torch.tensor(h0 + a * dv, device=\'cuda\'))']
for frag in KEEP:
    assert frag in s, '冻结片段被改: %s' % frag
print('冻结片段全部在位 (%d/%d)' % (len(KEEP), len(KEEP)))

py_compile.compile(DST, doraise=True)
print('py_compile OK -> %s (%d B)' % (DST, os.path.getsize(DST)))
