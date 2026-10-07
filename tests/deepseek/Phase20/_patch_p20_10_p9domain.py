# -*- coding: utf-8 -*-
"""补丁 10：把 P9 的「域歧义」诚实披露接入 Phase 20 收尾链生成器。

背景：MERGE 实测 P9 = FAIL（A0 `max|Δxhalf|=0.2645`），但该 FAIL 100% 由浅端 ℓ=1 承担，
而 P16 的 `XH_FAITHFUL_TOL` 只在冻结 REACH 域（`XH_12_by_site` = ℓ≥6, n=18）上标定过
（P16 实测 0.006171）。本补丁**不改判**（P9 保持 FAIL），只把域分解作为**明确标注的 post-hoc**
披露写进 gen_memo / gen_present / disk_verify / closeout rev_note。

约定：只做「字符串断言唯一 → 替换」；每处 assert count==1；改完 py_compile。
"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D20 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase20')

LOG = []


def patch(fname, old, new, tag, count=1):
    p = os.path.join(D20, fname)
    s = io.open(p, encoding='utf-8').read()
    n = s.count(old)
    assert n == count, '[%s] 期望 %d 处，实为 %d：%s' % (fname, count, n, tag)
    s2 = s.replace(old, new)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s2)
    LOG.append('  OK  %-28s %s' % (fname, tag))


# =====================================================================  gen_memo
GM = 'gen_memo_phase20.py'

# --- P1: 载入 post-hoc 分解
old = ("DRIFT = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',\n"
       "                                       'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))\n")
new = old + ("PH = json.load(io.open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), encoding='utf-8'))\n"
             "PHD = {r['pair']: r for r in PH['pairs']}\n")
patch(GM, old, new, 'P1 载入 posthoc json')

# --- P1b: 辅助函数（放在 ps 定义之后）
old = ("def E(a, k):\n    return ARMS[a]['E10_summary'].get(k)\n")
new = ("def ph_(pair, key, nd=6):\n"
       "    r = PHD.get(pair)\n"
       "    return f(r[key], nd) if r and isinstance(r.get(key), (int, float)) else 'NA'\n"
       "\n\n"
       "def E(a, k):\n    return ARMS[a]['E10_summary'].get(k)\n")
patch(GM, old, new, 'P1b ph_ 辅助')

# --- P2: §0 末尾补一句
old = "A('⇒ P19 给**向量侧**的跨精度支撑，本 Phase 给**行为侧**补齐。')\n"
new = (old +
       "A('9 条定向预测中 8 条 PASS；**P9 按判据字面（全部公共 PROFILE 位点）判 FAIL** —— '\n"
       "  '但其超差 100% 来自 P16 从未标定的浅端 ℓ=1，在 P16 实际标定的冻结 REACH 域（ℓ≥6）上两对皆 PASS'\n"
       "  '（见 §5 的域分解与 §11 `[E-xhdom]`）。')\n")
patch(GM, old, new, 'P2 §0 补 P9 说明')

# --- P3: §5 容差注 → 域分解
old = ("A('`xhalf` 容差 ' + f(FL['QUANT_TOL_XHALF'], 2) + ' 是 **P16 已发表的跨精度容差**（P16 E6 对 Phase 12 bf16 '\n"
       "  '实测 0.006：`XH_FAITHFUL_TOL`），本 Phase 沿用以保持口径一致。')\n"
       "A('')\n")
new = ("A('`xhalf` 容差 ' + f(FL['QUANT_TOL_XHALF'], 2) + ' 是 **P16 已发表的跨精度容差**（`XH_FAITHFUL_TOL`）；'\n"
       "  '但 P16 只在 **冻结 REACH 域**（`XH_12_by_site` = ℓ≥' + str(PH['p16_calib_domain'][0]) + '，n='\n"
       "  + str(len(PH['p16_calib_domain'])) + '）上标定过该容差（P16 实测 max|Δxhalf| = '\n"
       "  + f(PH['p16_calib_max_abs_dxh'], 6) + '，臂 `' + str(PH['p16_calib_arm']) + '`）。')\n"
       "A('')\n"
       "A('**P9 的域分解（post-hoc；as-coded 判决不改）**：')\n"
       "A('')\n"
       "A('| 配对 | as-coded（全部公共 PROFILE 位点） | 判 | 冻结 REACH 域（ℓ≥6, n='\n"
       "  + str(len(PH['p16_calib_domain'])) + '） | 判 | 浅端 ℓ<6 | 超差承担者 |')\n"
       "A('|---|---|---|---|---|---|---|')\n"
       "for _k in (PID, PID2):\n"
       "    _r = PHD.get(_k)\n"
       "    if not _r:\n"
       "        continue\n"
       "    A('| `' + _k + '` | ' + f(_r['as_coded_max_abs_dxh'], 6) + ' | '\n"
       "      + ('PASS' if _r['as_coded_pass'] else '**FAIL**') + ' | '\n"
       "      + f(_r['reach_domain_max_abs_dxh'], 6) + ' | '\n"
       "      + ('PASS' if _r['reach_domain_pass'] else 'FAIL') + ' | '\n"
       "      + f(_r['shallow_max_abs_dxh'], 6) + ' @ℓ=' + str(_r['shallow_argmax']) + ' | '\n"
       "      + '100% @ℓ=' + str(_r['as_coded_argmax']) + ' |')\n"
       "A('')\n"
       "A('⇒ 在 P16 **实际标定的那个域**上，两对都是 **PASS**，A0 裕度 ≈ '\n"
       "  + f(FL['QUANT_TOL_XHALF'] / max(PHD[PID]['reach_domain_max_abs_dxh'], 1e-9), 0) + '×（'\n"
       "  + f(PHD[PID]['reach_domain_max_abs_dxh'], 6) + ' vs ' + f(FL['QUANT_TOL_XHALF'], 2) + '）；'\n"
       "  'as-coded 的 FAIL 是**预注册判据文字未写明域**造成的**域错配**（承 P18 教训「判据域须与 rationale '\n"
       "  '标定域一致」），**不是**物理上的量化不稳定。原始判决与分解数据分别在 `result_phase20.json` 与 '\n"
       "  '`posthoc_p9_xhalf_domain.{json,txt}`；**不改判**。')\n"
       "A('')\n")
patch(GM, old, new, 'P3 §5 域分解表')

# --- P4: §11 加 [E-xhdom]
old = ("  + '`drift_events` 登记本事件）。审计件：`tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json`。')\n"
       "A('')\n")
new = (old +
       "A('- **[E-xhdom] 预注册判据的域歧义（本轮主要发现之一）**：seal 的 P9 只写「两对 max|Δxhalf| ≤ '\n"
       "  + f(FL['QUANT_TOL_XHALF'], 2) + '」，**未写明域**；实现按字面取「所有公共 PROFILE 位点」，'\n"
       "  '而 P16 的 `XH_FAITHFUL_TOL` 是在 `XH_12_by_site`（ℓ≥6）上标定的 ⇒ A0 的 as-coded 读数被浅端 ℓ=1 '\n"
       "  '（' + ph_(PID, 'shallow_max_abs_dxh') + '）抬高到 ' + ph_(PID, 'as_coded_max_abs_dxh') + '。'\n"
       "  '**处置**：as-coded 判决 **P9=' + ps('P9') + ' 原样保留**；另出 post-hoc 域分解（冻结 REACH 域 A0 '\n"
       "  + ph_(PID, 'reach_domain_max_abs_dxh') + ' / A1 ' + ph_(PID2, 'reach_domain_max_abs_dxh')\n"
       "  + '，两对皆 PASS，A0 该值恰与 P16 标定值逐位相同）并标注为 post-hoc。'\n"
       "  '教训：**「沿用某容差」时必须把该容差的标定域一并写进判据文字**。')\n"
       "A('')\n")
patch(GM, old, new, 'P4 §11 [E-xhdom]')

# --- P5: §12 下一步补一条
old = ("A('- **并列**：邻域宽度 ±2 敏感性（四臂 `nb` 恰都 ' + str(EX['arms'][AO[0]]['nb'])\n"
       "  + '）；P17 `P6` 的 MEMO 改判（承 P18 `P5`）。')\n")
new = ("A('- **并列**：邻域宽度 ±2 敏感性（四臂 `nb` 恰都 ' + str(EX['arms'][AO[0]]['nb'])\n"
       "  + '）；P17 `P6` 的 MEMO 改判（承 P18 `P5`）；**P9 判据补域**（把 `XH_FAITHFUL_TOL` 的 ℓ≥6 标定域'\n"
       "  '写进判据文字后可否改判 ⇒ 需新 Phase 预注册，本 Phase 维持 FAIL）。')\n")
patch(GM, old, new, 'P5 §12 下一步')

# --- P6: 附录资产补 posthoc 件
old = ("'`_probe20_A0_{nf4,bf16}.json`、`_armrec20_*.json`、`result_phase20.json`、`verify_ledger_phase20.txt`、'\n"
       "  '`disk_verify_phase20.txt`、`present_phase20.html`')\n")
new = ("'`_probe20_A0_{nf4,bf16}.json`、`_armrec20_*.json`、`result_phase20.json`、`verify_ledger_phase20.txt`、'\n"
       "  '`disk_verify_phase20.txt`、`present_phase20.html`、`posthoc_p9_xhalf_domain.{json,txt}`')\n")
patch(GM, old, new, 'P6 附录资产')

io.open(os.path.join(D20, '_patch_p20_10_p9domain.log'), 'w', encoding='utf-8', newline='\n')\
    .write('\n'.join(LOG) + '\n')
print('\n'.join(LOG))
print('gen_memo patched:', len(LOG))
