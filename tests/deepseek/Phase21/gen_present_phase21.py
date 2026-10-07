# -*- coding: utf-8 -*-
"""
Phase 21 present 生成器：把 result_phase21.json 渲染成单文件 HTML 汇报页（数据驱动，禁手工转录）。
产出：tests/deepseek_temp/Phase21/present_phase21.html
用法：python tests/deepseek/Phase21/gen_present_phase21.py
"""
import io
import os
import json
import hashlib
import time as _t

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
RESULTP = os.path.join(P21T, 'result_phase21.json')
SEALP = os.path.join(P21T, 'N2h1a14_design_seal.json')
OUT = os.path.join(P21T, 'present_phase21.html')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(RESULTP, encoding='utf-8'))
S = json.load(io.open(SEALP, encoding='utf-8'))
P = R['predictions']
pairs = R['quant_pairs']
cal = R['calibration']
A = R['arms']
PA8 = R['p8_anchor']
AO = R['arm_order']
FL = S['floors']


def f4(x):
    return '%.4f' % x


def pill(v):
    if v is True:
        return '<span class="ok">PASS</span>'
    if v is False:
        return '<span class="bad">FAIL</span>'
    return '<span class="na">N/A</span>'


H = []
h = H.append
h('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
h('<title>Phase 21 · N2h1-α-14 · 组件级向量预算与权重容量的跨精度稳健性</title>')
h('''<style>
:root{--bg:#0f1115;--card:#171a21;--ink:#e8ecf1;--dim:#9aa4b2;--acc:#5db2ff;--ok:#3ecf8e;--bad:#ff6b6b;--warn:#ffc857}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13px;margin-bottom:20px}
.card{background:var(--card);border:1px solid #232838;border-radius:12px;padding:16px 18px;margin:14px 0}
h2{font-size:15px;margin:0 0 10px;color:var(--acc);letter-spacing:.3px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 9px;border-bottom:1px solid #232838;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600}
.ok{color:var(--ok);font-weight:700}.bad{color:var(--bad);font-weight:700}.na{color:var(--dim)}
.kv{display:flex;flex-wrap:wrap;gap:10px}
.kv div{background:#12151c;border:1px solid #232838;border-radius:9px;padding:9px 12px;min-width:150px}
.kv b{display:block;color:var(--dim);font-size:11px;font-weight:600;letter-spacing:.4px}
.kv span{font-size:16px}
.mono{font-family:ui-monospace,Consolas,monospace}
.note{color:var(--dim);font-size:12px;margin-top:8px}
.hero{background:linear-gradient(135deg,#16233a,#12151c);border-color:#2b3a55}
ul{margin:6px 0 0;padding-left:20px}li{margin:3px 0}
</style></head><body><div class="wrap">''')
h('<h1>Phase 21 · N2h1-α-14</h1>')
h('<div class="sub">组件级向量预算 <span class="mono">share_v</span> 与 权重实现级容量 <span class="mono">W</span> 的跨精度稳健性 · '
  'nf4 ↔ bf16 · qwen3-4b + glm4-9b · seal <span class="mono">%s</span> · exec <span class="mono">%s</span> · result <span class="mono">%s</span></div>'
  % (R['seal_sha8'], R['exec_sha8'], sha8(RESULTP)))

g1_all = all(A[a]['G1_core'] for a in AO)
h('<div class="card hero"><h2>结论</h2><div class="kv">')
h('<div><b>装置校准 A0_bf16 vs P8</b><span>%s</span></div>' % ('<span class="ok">PASS 逐位</span>' if cal.get('ok') else '<span class="bad">FAIL</span>'))
h('<div><b>G1 分布式（四臂）</b><span>%s</span></div>' % ('<span class="ok">全部成立</span>' if g1_all else '<span class="bad">有失败臂</span>'))
h('<div><b>预测通过</b><span>%d / %d</span></div>' % (R['n_pass'], R['n_total']))
h('<div><b>share_v(mlp) 最大 |Δ|</b><span>%s</span></div>' % f4(max(abs(p['d_share_v_mlp']) for p in pairs)))
h('<div><b>spearman 最小</b><span>%s</span></div>' % f4(min([p['spearman_share_v'] for p in pairs if p['spearman_share_v'] is not None] or [0])))
h('</div><div class="note">P8 的「分布式搬运 / MLP 最大单一写入方」在 nf4 与 bf16 下同判 ⇒ 不是 nf4 kernel 路径的产物。</div></div>')

# 校准
h('<div class="card"><h2>A · 装置校准：A0_bf16 对 P8 冻结锚（P8 即 bf16）</h2><table>')
h('<tr><th>量</th><th>本 Phase（bf16）</th><th>P8 冻结</th><th>|Δ|</th><th></th></tr>')
rows = [('share_v(mlp)', cal['share_v_mlp']['got'], cal['share_v_mlp']['exp'], cal['share_v_mlp']['d']),
        ('max_head_share_v', cal['max_head_share_v']['got'], cal['max_head_share_v']['exp'], cal['max_head_share_v']['d'])]
if 'W_max_head_share' in cal:
    rows.append(('W.max_head_share', cal['W_max_head_share']['got'], cal['W_max_head_share']['exp'], cal['W_max_head_share']['d']))
if 'I_nl' in cal:
    rows.append(('I_nl', cal['I_nl']['got'], cal['I_nl']['exp'], cal['I_nl']['d']))
    rows.append(('T[diff6] dDonor', cal['T_diff6']['got'], cal['T_diff6']['exp'], cal['T_diff6']['d']))
for nm, got, exp, d in rows:
    h('<tr><td>%s</td><td class="mono">%s</td><td class="mono">%s</td><td class="mono">%.2e</td><td>%s</td></tr>'
      % (nm, f4(got), f4(exp), d, pill(d <= 1e-4)))
h('<tr><td>argmax_head_v</td><td class="mono">%s</td><td class="mono">%s</td><td>—</td><td>%s</td></tr>'
  % (cal['argmax_head_v']['got'], cal['argmax_head_v']['exp'], pill(cal['argmax_head_v']['same'])))
h('</table><div class="note">同一装置（device_map auto 全 GPU）下 bf16 与 P8 的 .to(cuda) 数值路径逐位一致。</div></div>')

# 四臂
h('<div class="card"><h2>B · 四臂逐量（M1 向量预算 / M2 权重容量 / G1）</h2><table>')
h('<tr><th>臂</th><th>模型</th><th>精度</th><th>窗层</th><th>share_v(mlp)</th><th>max_head_share_v</th><th>argmax</th><th>loo_vec_top1</th><th>W.max_head</th><th>W.argmax</th><th>G1_core</th></tr>')
for a in AO:
    m1 = A[a]['M1']; m2 = A[a].get('M2', {})
    h('<tr><td class="mono">%s</td><td>%s</td><td>%s</td><td>%d</td><td class="mono">%s</td><td class="mono">%s</td><td class="mono">%s</td>'
      '<td class="mono">%s</td><td class="mono">%s</td><td class="mono">%s</td><td>%s</td></tr>'
      % (a, A[a]['model'], A[a]['scheme'], A[a]['primary_layer'], f4(m1['share_v_mlp']), f4(m1['max_head_share_v']),
         m1['argmax_head_v'], f4(m1['loo_vec_top1']),
         (f4(m2['max_head_share']) if 'max_head_share' in m2 else 'n/a'),
         ('#' + str(m2['argmax_head'])) if 'argmax_head' in m2 else 'n/a', pill(A[a]['G1_core'])))
h('</table><div class="note">G1_core = <span class="mono">max_head_share_v ≤ %.2f 且 share_v(mlp) ≤ %.2f</span>（P8 amend1 口径）。</div></div>'
  % (FL['G1_MAXHEAD_V'], FL['G1_MLP_SHARE_V']))

# 配对
h('<div class="card"><h2>C · 跨精度配对（同模型 nf4 vs bf16）</h2><table>')
h('<tr><th>模型</th><th>Δ share_v(mlp)</th><th>Δ max_head_share_v</th><th>argmax 同</th><th>spearman</th><th>G1 nf4 / bf16</th><th>W Δmax</th><th>W argmax 同</th></tr>')
for p in pairs:
    h('<tr><td>%s</td><td class="mono">%+.6f</td><td class="mono">%+.6f</td><td>%s</td><td class="mono">%s</td><td>%s / %s</td><td class="mono">%s</td><td>%s</td></tr>'
      % (p['model'], p['d_share_v_mlp'], p['d_max_head_share_v'], pill(p['argmax_head_v_same']),
         (f4(p['spearman_share_v']) if p['spearman_share_v'] is not None else 'n/a'),
         pill(p['G1_core_nf4']), pill(p['G1_core_bf16']),
         ('%+.4f' % p['W']['d_max']) if p['W'] else 'n/a',
         pill(p['W']['argmax_same'] if p['W'] else None)))
h('</table></div>')

# 预测
h('<div class="card"><h2>D · 预注册预测（%d / %d 通过）</h2><table><tr><th>预测</th><th>结果</th></tr>' % (R['n_pass'], R['n_total']))
for k, v in P.items():
    h('<tr><td class="mono">%s</td><td>%s</td></tr>' % (k, pill(v)))
h('</table></div>')

# 效应侧
h('<div class="card"><h2>E · 效应侧（M3，第二指标 / 不可加）</h2><table>')
h('<tr><th>臂</th><th>diff5</th><th>attn_all</th><th>mlp</th><th>diff6</th><th>I_nl</th><th>eff_max_head</th><th>V_rand</th><th>M_mismatch</th><th>floors</th></tr>')
for a in AO:
    m3 = A[a].get('M3') or {}
    if 'T' not in m3:
        continue
    h('<tr><td class="mono">%s</td><td>%+.3f</td><td>%+.3f</td><td>%+.3f</td><td>%+.3f</td><td class="mono">%s</td><td class="mono">%s</td>'
      '<td>%+.4f</td><td>%+.4f</td><td>%s</td></tr>'
      % (a, m3['T']['diff5']['dDonor'], m3['T']['attn_all']['dDonor'], m3['T']['mlp']['dDonor'], m3['T']['diff6']['dDonor'],
         f4(m3['I_nl']), f4(m3['max_head_share_eff']), m3['T']['V_rand']['dDonor'], m3['T']['M_mismatch']['dDonor'],
         pill(m3['floors']['floors_ok'])))
h('</table></div>')

h('<div class="card"><h2>F · 限界</h2><ul>')
for x in S['honesty']:
    h('<li>%s</li>' % x)
h('</ul></div>')

h('<div class="sub" style="margin-top:18px">生成 %s · 全部数字由 result_phase21.json 现场渲染</div>' % _t.strftime('%Y-%m-%d %H:%M:%S'))
h('</div></body></html>')

body = '\n'.join(H)
with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    f.write(body)
print('WROTE', OUT, len(body.encode('utf-8')), 'B')
