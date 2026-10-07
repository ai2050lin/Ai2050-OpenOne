# -*- coding: utf-8 -*-
"""补丁 11：把 P9 的域分解与 [E-xhdom] 披露接入 gen_present_phase20.py。"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D20 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase20')
P = os.path.join(D20, 'gen_present_phase20.py')
s = io.open(P, encoding='utf-8').read()
LOG = []


def rep(old, new, tag, count=1):
    global s
    n = s.count(old)
    assert n == count, '[%s] 期望 %d 处，实为 %d' % (tag, count, n)
    s = s.replace(old, new)
    LOG.append('  OK  %s' % tag)


# ---------------- E1: 载入 post-hoc 分解
rep("DRIFT = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',\n"
    "                                       'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))\n",
    "DRIFT = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',\n"
    "                                       'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))\n"
    "PH = json.load(io.open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), encoding='utf-8'))\n"
    "PHD = {r['pair']: r for r in PH['pairs']}\n",
    'E1 载入 posthoc')

# ---------------- E2: 联合判据表后插入 P9 域分解卡
OLD2 = ("for lab, key, crit in _JROW:\n"
        "    H.append('<tr><td>%s</td><td>%s</td><td class=\"note\">%s</td><td>%s</td></tr>'\n"
        "             % (lab, str(JV.get(key)), crit, pill(JV.get(key), '\u2713', '\u2717')))\n"
        "H.append('</table></div>')\n")
NEW2 = ("for lab, key, crit in _JROW:\n"
        "    H.append('<tr><td>%s</td><td>%s</td><td class=\"note\">%s</td><td>%s</td></tr>'\n"
        "             % (lab, str(JV.get(key)), crit, pill(JV.get(key), '\u2713', '\u2717')))\n"
        "H.append('</table>')\n"
        "H.append('<div class=\"note\" style=\"border-left:4px solid #b45309;background:#fffbeb;padding:8px 10px\">'\n"
        "         '<b>P9 域分解（post-hoc；as-coded 判决不改）</b>：seal 的 P9 判据「两对 <code>max|&Delta;xhalf|</code> &le; '\n"
        "         + f(FL['QUANT_TOL_XHALF'], 2) + '」<b>未写明域</b>，实现按字面取「所有公共 PROFILE 位点」；'\n"
        "         '但 P16 的 <code>XH_FAITHFUL_TOL</code> 只在 <b>冻结 REACH 域</b>（<code>XH_12_by_site</code> = &ell;&ge;'\n"
        "         + str(PH['p16_calib_domain'][0]) + ', n=' + str(len(PH['p16_calib_domain'])) + '）上标定过（P16 实测 <code>'\n"
        "         + fp(PH['p16_calib_max_abs_dxh']) + '</code>）。<br>'\n"
        "         '&rArr; <b>Q12 as-coded = ' + str(JV.get('Q12_xhalf_stable')) + '</b>，但其超差 100% 来自 <b>浅端 &ell;=1</b>；'\n"
        "         '在 P16 实际标定的域上两对皆 <b>PASS</b>（A0 裕度约 '\n"
        "         + f(FL['QUANT_TOL_XHALF'] / max(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh'], 1e-9), 0)\n"
        "         + '&times;）&rArr; 这是<b>预注册判据文字的域错配</b>，不是物理上的量化不稳定。</div>')\n"
        "H.append('<table><tr><th>配对</th><th>as-coded（全部公共位点）</th><th>判</th>'\n"
        "         '<th>冻结 REACH 域（&ell;&ge;6）</th><th>判</th><th>浅端 &ell;&lt;6</th><th>超差承担者</th></tr>')\n"
        "for k in PAIRS:\n"
        "    r = PHD.get(k)\n"
        "    if not r:\n"
        "        continue\n"
        "    H.append('<tr><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td>'\n"
        "             '<td>%s @&ell;=%s</td><td>100%% @&ell;=%s</td></tr>'\n"
        "             % (k.replace('|', ' | '), fp(r['as_coded_max_abs_dxh']),\n"
        "                pill(r['as_coded_pass'], 'PASS', 'FAIL'), fp(r['reach_domain_max_abs_dxh']),\n"
        "                pill(r['reach_domain_pass'], 'PASS', 'FAIL'), fp(r['shallow_max_abs_dxh']),\n"
        "                str(r['shallow_argmax']), str(r['as_coded_argmax'])))\n"
        "H.append('</table>')\n"
        "H.append('</div>')\n")
rep(OLD2, NEW2, 'E2 P9 域分解卡')

# ---------------- E3: G 节末追加 [E-xhdom]
OLD3 = ("         % (str(DRIFT['prev_baseline']['bytes']), str(DRIFT['prev_baseline']['sha8']),\n"
        "            str(DRIFT['memo_mtime']), len(DRIFT['normalized_phases']),\n"
        "            DRIFT['observed_delta_bytes']))\n"
        "H.append('</div>')\n")
NEW3 = ("         % (str(DRIFT['prev_baseline']['bytes']), str(DRIFT['prev_baseline']['sha8']),\n"
        "            str(DRIFT['memo_mtime']), len(DRIFT['normalized_phases']),\n"
        "            DRIFT['observed_delta_bytes']))\n"
        "H.append('<div class=\"note\" style=\"border-left:4px solid #b45309;background:#fffbeb;padding:8px 10px\">'\n"
        "         '<b>E-xhdom</b>（本轮主要发现之一）：seal 的 P9 只写「两对 <code>max|&Delta;xhalf|</code> &le; '\n"
        "         + f(FL['QUANT_TOL_XHALF'], 2) + '」，<b>未写明域</b>；实现按字面取「所有公共 PROFILE 位点」'\n"
        "         '&rArr; A0 的 as-coded 读数 <code>' + fp(PHD['A0_nf4|A0_bf16']['as_coded_max_abs_dxh'])\n"
        "         + '</code> <b>完全由浅端 &ell;=1 贡献</b>（&ell;&ge;6 段仅 <code>'\n"
        "         + fp(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh']) + '</code>，比 &ell;=1 小 '\n"
        "         + f(PHD['A0_nf4|A0_bf16']['shallow_max_abs_dxh']\n"
        "             / max(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh'], 1e-9), 0) + '&times;），'\n"
        "         '而 P16 的 <code>XH_FAITHFUL_TOL</code> 只在 <code>XH_12_by_site</code>（&ell;&ge;6）上标定过。'\n"
        "         '<b>处置</b>：as-coded 判决 <b>P9 = '\n"
        "         + ('FAIL' if PC['P9']['pass_'] is False else 'PASS') + ' 原样保留</b>；另出 post-hoc 域分解'\n"
        "         '（A0 <code>' + fp(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh']) + '</code> / A1 <code>'\n"
        "         + fp(PHD['A1_nf4|A1_bf16']['reach_domain_max_abs_dxh']) + '</code>，两对皆 PASS，'\n"
        "         'A0 该值恰与 P16 标定值逐位相同）。<b>教训</b>：「沿用某容差」时必须把该容差的标定域一并写进判据文字。</div>')\n"
        "H.append('</div>')\n")
rep(OLD3, NEW3, 'E3 [E-xhdom]')

# ---------------- E4: H 并列行补 P9 判据补域
OLD4 = ("H.append('<div class=\"kv\"><span>并列</span><b>邻域宽度 ±2 敏感性（四臂 nb 恰好都是 %s）</b></div>'\n"
        "         % str(R['arms'][ARMS[0]]['E10_summary']['nb']))\n")
NEW4 = (OLD4 +
        "H.append('<div class=\"kv\"><span>并列</span><b>P9 判据补域：把 <code>XH_FAITHFUL_TOL</code> 的 '\n"
        "         '&ell;&ge;6 标定域写进判据文字后可否改判（&rArr; 需新 Phase 预注册，本 Phase 维持 FAIL）</b></div>')\n")
rep(OLD4, NEW4, 'E4 H 并列补 P9 补域')

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
io.open(os.path.join(D20, '_patch_p20_11_present_p9.log'), 'w', encoding='utf-8', newline='\n')\
    .write('\n'.join(LOG) + '\n')
print('\n'.join(LOG))
print('gen_present patched:', len(LOG))
