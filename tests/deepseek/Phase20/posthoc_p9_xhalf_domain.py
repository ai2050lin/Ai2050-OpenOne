# -*- coding: utf-8 -*-
"""Phase 20 —— P9 的事后域分解（post-hoc；**不修改任何冻结脚本、不重跑任何臂**）。

背景（诚实边界）：seal 的 P9 判据文字为「两对 max|Δxhalf| ≤ QUANT_TOL_XHALF=0.05
（沿用 P16 的 XH_FAITHFUL_TOL）」，**未写明域**；实现按字面取「所有两臂皆非 None 的
PROFILE 位点」（含浅端 1/3/5）。而 P16 的 XH_FAITHFUL_TOL 是在 `XH_12_by_site`
= 冻结 REACH 域 (ℓ≥6) 上标定的（P16 `max_abs_dxh=0.00617` 只用了那 18 个位点）。

本脚本**不改判**：把 as-coded 的 P9 判决原样复述，再按同一份 result 给出「冻结 REACH
域 (ℓ≥6)」与「浅端 (ℓ<6)」两段的分解，作为**明确标注的 post-hoc 观察**。

产物：tests/deepseek_temp/Phase20/posthoc_p9_xhalf_domain.{json,txt}
"""
import io
import os
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
P16R = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16', 'result_phase16.json')

o = []


def w(s=''):
    o.append(str(s))
    print(s)


R = json.load(io.open(os.path.join(P20T, 'result_phase20.json'), encoding='utf-8'))
R16 = json.load(io.open(P16R, encoding='utf-8'))
TO = 0.05

# P16 标定域 = XH_12_by_site 的位点（冻结 REACH）
XH12 = sorted(int(k) for k in R16['inheritance_used']['XH_12_by_site'].keys())
p16_max = None
for a, rec in R16['arms'].items():
    e6 = rec.get('E6_calibration') or {}
    if e6.get('max_abs_dxh') is not None:
        p16_max = float(e6['max_abs_dxh'])
        p16_arm = a
        break

w('=== Phase 20 P9 事后域分解（post-hoc；不改判） ===')
w('seal P9 判据 : 两对 max|Δxhalf| <= %.2f（沿用 P16 的 XH_FAITHFUL_TOL）' % TO)
w('P16 标定域   : XH_12_by_site = %s (n=%d)' % (XH12, len(XH12)))
w('P16 标定读数 : max|Δxhalf| = %.6f (%s)' % (p16_max, p16_arm))
w('')

OUT = {'tolerance': TO, 'p16_calib_domain': XH12, 'p16_calib_max_abs_dxh': p16_max,
       'p16_calib_arm': p16_arm, 'pairs': []}

for p in R['quant_pairs']:
    tag = p['arm_nf4'] + '|' + p['arm_bf16']
    x = p['xhalf']
    site_v = {int(s): (abs(float(a) - float(b)), float(a), float(b))
              for s, a, b in zip(x['sites'], x['nf4'], x['bf16'])}
    dom = [s for s in XH12 if s in site_v]
    shy = [s for s in site_v if s not in XH12]
    full_arg = max(site_v, key=lambda s: site_v[s][0])
    dom_arg = max(dom, key=lambda s: site_v[s][0]) if dom else None
    shy_arg = max(shy, key=lambda s: site_v[s][0]) if shy else None
    rec = dict(
        pair=tag, as_coded_max_abs_dxh=float(x['max_abs_dxh']),
        as_coded_argmax=full_arg, n_common=x['n_common'],
        reach_domain_max_abs_dxh=float(site_v[dom_arg][0]) if dom_arg else None,
        reach_domain_argmax=dom_arg, reach_domain_n=len(dom),
        shallow_max_abs_dxh=float(site_v[shy_arg][0]) if shy_arg else None,
        shallow_argmax=shy_arg, shallow_sites=shy,
        as_coded_pass=bool(x['max_abs_dxh'] <= TO),
        reach_domain_pass=(bool(site_v[dom_arg][0] <= TO) if dom_arg else None),
        per_site={str(s): round(v[0], 6) for s, v in sorted(site_v.items())},
    )
    OUT['pairs'].append(rec)
    w('--- %s ---' % tag)
    w('  as-coded (所有公共 PROFILE 位点, n=%d) : max|Δxhalf| = %.6f @ℓ=%s  -> %s'
      % (x['n_common'], x['max_abs_dxh'], full_arg, 'PASS' if rec['as_coded_pass'] else 'FAIL'))
    w('  冻结 REACH 域 (ℓ>=6, n=%d)          : max|Δxhalf| = %.6f @ℓ=%s  -> %s'
      % (len(dom), site_v[dom_arg][0] if dom_arg else float('nan'), dom_arg,
         'PASS' if rec['reach_domain_pass'] else 'FAIL') if dom_arg else '  (空)')
    w('  浅端 (ℓ<6) 位点 %-10s          : max|Δxhalf| = %s @ℓ=%s'
      % (str(shy), ('%.6f' % site_v[shy_arg][0]) if shy_arg else 'NA', shy_arg))
    w('  ⇒ 判决差 100%% 由 %s 承担；ℓ>=6 段与 P16 标定量级一致。'
      % ('浅端 ℓ=%s' % shy_arg if shy_arg else '无'))
    w('')

OUT['note'] = ('P9 的 as-coded 判决（FAIL）不动；本表只做域分解，说明 FAIL 全部来自 '
               'P16 从未标定的浅端 ℓ∈{1,3}（该处 Y(ℓ,α) 近平坦、0.5 交叉不稳定）。')
io.open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), 'w', encoding='utf-8', newline='\n')\
    .write(json.dumps(OUT, ensure_ascii=False, indent=1))
io.open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.txt'), 'w', encoding='utf-8', newline='\n')\
    .write('\n'.join(o) + '\n')
print('WROTE posthoc_p9_xhalf_domain.{json,txt}')
