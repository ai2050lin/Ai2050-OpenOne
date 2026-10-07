# -*- coding: utf-8 -*-
"""补丁 12：disk_verify 增 G12（P9 域分解独立重算）+ closeout rev_note 诚实改写。"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D20 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase20')
LOG = []


def rep(fname, old, new, tag, count=1):
    p = os.path.join(D20, fname)
    s = io.open(p, encoding='utf-8').read()
    n = s.count(old)
    assert n == count, '[%s] 期望 %d 处，实为 %d：%s' % (fname, count, n, tag)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
    LOG.append('  OK  %-26s %s' % (fname, tag))


# =====================================================  disk_verify: G12
DV = 'disk_verify_phase20.py'
OLD = ("    chk('G11f', '探针落盘可读 %s' % nm, ok)\n"
       "\n"
       "# ================================================================ 汇总\n")
NEW = ("    chk('G11f', '探针落盘可读 %s' % nm, ok)\n"
       "\n"
       "# ================================================================ G12 P9 域分解（独立重算）\n"
       "w('')\n"
       "w('=== G12 P9 xhalf 域分解（独立重算；as-coded 判决不改） ===')\n"
       "PH = load(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'))\n"
       "PHD = {r['pair']: r for r in PH['pairs']}\n"
       "R16 = load(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16', 'result_phase16.json'))\n"
       "_XH12 = sorted(int(k) for k in R16['inheritance_used']['XH_12_by_site'].keys())\n"
       "_p16max = None\n"
       "for _a, _rec in R16['arms'].items():\n"
       "    _e6 = _rec.get('E6_calibration') or {}\n"
       "    if _e6.get('max_abs_dxh') is not None:\n"
       "        _p16max = float(_e6['max_abs_dxh'])\n"
       "        break\n"
       "chk('G12a', 'P16 标定域 == 冻结 REACH(ell>=6), n=18',\n"
       "    _XH12 == [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34], _XH12)\n"
       "chk('G12b', 'posthoc 记录的 P16 标定域与实盘一致',\n"
       "    sorted(int(s) for s in PH['p16_calib_domain']) == _XH12)\n"
       "chk('G12c', 'posthoc 记录的 P16 标定最大值与 P16 result 一致',\n"
       "    abs(PH['p16_calib_max_abs_dxh'] - _p16max) <= 1e-12,\n"
       "    PH['p16_calib_max_abs_dxh'], round(_p16max, 12))\n"
       "_TO = float(FL['QUANT_TOL_XHALF'])\n"
       "for _p in R['quant_pairs']:\n"
       "    _k = _p['arm_nf4'] + '|' + _p['arm_bf16']\n"
       "    _x = _p['xhalf']\n"
       "    _sv = {int(s): abs(float(a) - float(b)) for s, a, b in zip(_x['sites'], _x['nf4'], _x['bf16'])}\n"
       "    _full = max(_sv.values())\n"
       "    _dom = [s for s in _XH12 if s in _sv]\n"
       "    _dommax = max(_sv[s] for s in _dom) if _dom else None\n"
       "    _shy = [s for s in _sv if s not in _XH12]\n"
       "    _shymax = max(_sv[s] for s in _shy) if _shy else None\n"
       "    chk('G12d', '%s as-coded max|dxhalf| 独立重算' % _k,\n"
       "        abs(_full - _x['max_abs_dxh']) <= 1e-12, round(_full, 12), round(_x['max_abs_dxh'], 12))\n"
       "    chk('G12e', '%s 冻结 REACH 域 max|dxhalf| == posthoc' % _k,\n"
       "        _dommax is not None and abs(_dommax - PHD[_k]['reach_domain_max_abs_dxh']) <= 1e-12,\n"
       "        round(_dommax, 12) if _dommax else None, PHD[_k]['reach_domain_max_abs_dxh'])\n"
       "    chk('G12f', '%s 浅端 max|dxhalf| == posthoc' % _k,\n"
       "        _shymax is not None and abs(_shymax - PHD[_k]['shallow_max_abs_dxh']) <= 1e-12,\n"
       "        round(_shymax, 12) if _shymax else None, PHD[_k]['shallow_max_abs_dxh'])\n"
       "    chk('G12g', '%s 冻结域内 max <= 容差（两对皆 PASS）' % _k, _dommax is not None and _dommax <= _TO,\n"
       "        round(_dommax, 12) if _dommax else None, _TO)\n"
       "    chk('G12h', '%s as-coded 判决与 result 一致' % _k,\n"
       "        (_x['max_abs_dxh'] <= _TO) == PHD[_k]['as_coded_pass'],\n"
       "        _x['max_abs_dxh'] <= _TO, PHD[_k]['as_coded_pass'])\n"
       "_A0K = 'A0_nf4|A0_bf16'\n"
       "chk('G12i', 'A0 as-coded FAIL（冻结判决不改）', PHD[_A0K]['as_coded_pass'] is False)\n"
       "chk('G12j', 'A0 超差 100%% 由浅端承担（as-coded == shallow）',\n"
       "    abs(PHD[_A0K]['shallow_max_abs_dxh'] - PHD[_A0K]['as_coded_max_abs_dxh']) <= 1e-12)\n"
       "chk('G12k', 'result 的 P9 判决 == FAIL（未改判）',\n"
       "    R['predictions_check']['P9']['pass_'] is False,\n"
       "    R['predictions_check']['P9']['pass_'])\n"
       "chk('G12l', 'A0 冻结域值复现 P16 标定值（<=1e-12 相对）',\n"
       "    abs(PHD[_A0K]['reach_domain_max_abs_dxh'] - _p16max) <= 1e-12,\n"
       "    round(PHD[_A0K]['reach_domain_max_abs_dxh'], 15), round(_p16max, 15))\n"
       "\n"
       "# ================================================================ 汇总\n")
rep(DV, OLD, NEW, 'G12 块')

# =====================================================  closeout: 载入 posthoc
CO = 'closeout_phase20.py'
rep(CO, "PB = {}\nfor nm in ('A0_nf4', 'A0_bf16'):\n",
    "PH21 = json.load(io.open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), encoding='utf-8'))\n"
    "PHD21 = {r['pair']: r for r in PH21['pairs']}\n"
    "PB = {}\nfor nm in ('A0_nf4', 'A0_bf16'):\n",
    'P 载入 posthoc')

# closeout: rev_note P9 子句诚实改写
OLD = ("    'P9 %(p9)s: the half-saturation dose is precision-stable -- max|delta xhalf| = %(dxh0)s (A0) / '\n"
       "    '%(dxh1)s (A1), against the 0.05 tolerance already published by Phase 16. '\n")
NEW = ("    'P9 %(p9)s: the half-saturation dose AS SEALED (max|delta xhalf| over ALL common profile sites; '\n"
       "    'the criterion text did not pin the domain) FAILS -- %(dxh0)s (A0) / %(dxh1)s (A1) against the '\n"
       "    '0.05 tolerance. POST-HOC domain decomposition shows the excess is carried ENTIRELY by the '\n"
       "    'shallow site ell=1, which lies OUTSIDE the ell>=6 domain on which Phase 16 calibrated '\n"
       "    'XH_FAITHFUL_TOL; restricted to that frozen REACH domain the same two pairs read %(dxh0r)s (A0) / '\n"
       "    '%(dxh1r)s (A1), BOTH PASS with an ~8x margin, and the A0 value reproduces the Phase-16 '\n"
       "    'calibration figure to ~1e-16 relative (the A0 nf4<->bf16 pair is the same comparison Phase 16 ran). '\n"
       "    'So the single FAIL is a PRE-REGISTRATION DOMAIN AMBIGUITY, not a physical quantisation instability. '\n")
rep(CO, OLD, NEW, 'rev_note P9 子句')

OLD = ("    'READING: Phases 8-18s behavioural-side conclusions obtain the cross-precision support that Phase 19 '\n"
       "    'gave the vector side -- neither the shallower behavioural centroid, nor the MLP-led behavioural '\n"
       "    'attribution, nor the write-window concentration profile is an nf4 quantisation-floor artefact. '\n")
NEW = ("    'READING: 8 of the 9 directional predictions pass; Phases 8-18s behavioural-side conclusions obtain '\n"
       "    'the cross-precision support that Phase 19 gave the vector side -- neither the shallower behavioural '\n"
       "    'centroid, nor the MLP-led behavioural attribution, nor the write-window concentration profile is an '\n"
       "    'nf4 quantisation-floor artefact. The single FAIL (P9) is the domain ambiguity above. '\n")
rep(CO, OLD, NEW, 'rev_note READING 子句')

# closeout: dict 里补 dxh0r / dxh1r
OLD = ("    dxh1=(('%.4f' % QPM[PID2]['xhalf']['max_abs_dxh'])\n"
       "          if (QPM.get(PID2) and QPM[PID2]['xhalf'].get('max_abs_dxh') is not None) else 'NA'),\n")
NEW = (OLD +
       "    dxh0r=('%.6f' % PHD21[PID]['reach_domain_max_abs_dxh']),\n"
       "    dxh1r=('%.6f' % PHD21[PID2]['reach_domain_max_abs_dxh']),\n")
rep(CO, OLD, NEW, 'dict 补 dxh*r')

# closeout: ENTRY 增 posthoc sha（可追溯）
rep(CO, "    anchor_result_sha8=RES['anchor_result_p18_sha256'][:8],\n)\n",
    "    anchor_result_sha8=RES['anchor_result_p18_sha256'][:8],\n"
    "    posthoc_p9_sha8=sha8b(open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), 'rb').read()),\n)\n",
    'ENTRY 补 posthoc_sha8')
rep(CO, "    CHK = ('result_sha8', 'rev_note', 'verdict', 'n_rows', 'n_forwards_per_arm',\n"
        "           'seal_sha8', 'exec_sha8', 'probe_sha8', 'anchor_result_sha8')\n",
    "    CHK = ('result_sha8', 'rev_note', 'verdict', 'n_rows', 'n_forwards_per_arm',\n"
    "           'seal_sha8', 'exec_sha8', 'probe_sha8', 'anchor_result_sha8', 'posthoc_p9_sha8')\n",
    'CHK 补 posthoc_sha8')

io.open(os.path.join(D20, '_patch_p20_12_verify_ledger.log'), 'w', encoding='utf-8', newline='\n')\
    .write('\n'.join(LOG) + '\n')
print('\n'.join(LOG))
print('patched:', len(LOG))
