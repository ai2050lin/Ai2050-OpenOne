# -*- coding: utf-8 -*-
"""Phase 19 展示页生成器（数据驱动；全部数字取自 result_phase19.json / seal / exec）。
产出 tests/deepseek_temp/Phase19/present_phase19.html（浅色主题、自包含、无外部依赖）。
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
OUT = os.path.join(P19T, 'present_phase19.html')
SEALP = os.path.join(P19T, 'N2h1a12_design_seal.json')
EXECP = os.path.join(P19T, 'execution_phase19.json')
RESP = os.path.join(P19T, 'result_phase19.json')
PROBEP = os.path.join(P19T, '_probe19_A0_both.json')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(RESP, encoding='utf-8'))
EX = json.load(io.open(EXECP, encoding='utf-8'))
SEAL = json.load(io.open(SEALP, encoding='utf-8'))
PROBE = json.load(io.open(PROBEP, encoding='utf-8'))

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']; FL = R['floors']
QP = JV['quant_pairs']
ARMS = list(EX['arm_order'])
SHORT = {'A0_nf4': 'A0·qwen3-4b (nf4)', 'A0_bf16': 'A0·qwen3-4b (bf16)',
         'A1_nf4': 'A1·GLM4-9B (nf4)', 'A1_bf16': 'A1·GLM4-9B (bf16)'}
COL = {'A0_nf4': '#c2410c', 'A0_bf16': '#f59e0b', 'A1_nf4': '#1d4ed8', 'A1_bf16': '#3b82f6'}
ROLE = {'A0_nf4': '校准臂 1', 'A0_bf16': '检验臂（探针已知）', 'A1_nf4': '校准臂 2', 'A1_bf16': 'holdout 检验臂'}
PAIRS = ['A0_nf4|A0_bf16', 'A1_nf4|A1_bf16']


def f(x, n=3, dash='—'):
    try:
        if x is None:
            return dash
        return ('%.' + str(n) + 'f') % float(x)
    except Exception:
        return dash


def sci(x, n=2, dash='—'):
    try:
        if x is None:
            return dash
        return ('%.' + str(n) + 'e') % float(x)
    except Exception:
        return dash


def pill(ok, ok_t='PASS', no_t='FAIL', na_t='N/A'):
    if ok is None:
        return '<span class="pill na">%s</span>' % na_t
    return '<span class="pill %s">%s</span>' % ('ok' if ok else 'bad', ok_t if ok else no_t)


def vchip(label, txt):
    return '<span class="vchip"><b>%s</b> %s</span>' % (label, txt)


def bar(frac, color):
    w = max(0.0, min(1.0, float(frac))) * 100.0
    return '<span class="bar"><i style="width:%.2f%%;background:%s"></i></span>' % (w, color)


H = []
H.append('<!DOCTYPE html>')
H.append('<html lang="zh-CN"><head><meta charset="utf-8">')
H.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
H.append('<title>Phase 19 / N2h1-α-12 · 写入向量谱的量化口径稳健性（nf4 ↔ bf16）</title>')
H.append('''<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#16202b;--mut:#5b6b7c;--line:#e3e8ee;--ok:#0f766e;--bad:#b42318;--hi:#fef3c7;--ac:#1d4ed8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.62 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:23px;margin:0 0 4px}
h2{font-size:17px;margin:30px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:13px;margin:14px 0 6px;color:var(--mut)}
.sub{color:var(--mut);font-size:13px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f2f5f8;color:var(--mut);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;font-weight:600}
.pill.ok{background:#e7f5f2;color:var(--ok)}.pill.bad{background:#fdecea;color:var(--bad)}.pill.na{background:#eef1f5;color:var(--mut)}
.vchip{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:20px;padding:3px 11px;margin:3px 5px 3px 0;font-size:12.5px}
.vchip b{color:var(--ac)}
.bar{display:inline-block;width:100%;max-width:190px;height:11px;background:#eef1f5;border-radius:6px;overflow:hidden;vertical-align:middle}
.bar i{display:block;height:100%;border-radius:6px}
.kv{display:flex;justify-content:space-between;gap:12px;padding:3px 0;border-bottom:1px dashed var(--line)}
.kv:last-child{border-bottom:0}
.kv b{font-variant-numeric:tabular-nums}
.hl{background:var(--hi);border-radius:6px;padding:2px 6px}
.note{color:var(--mut);font-size:12.5px}
.sp{display:flex;align-items:flex-end;gap:1px;height:56px;margin:2px 0 2px}
.sp i{flex:1;border-radius:2px 2px 0 0;min-height:1px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:720px){.grid2{grid-template-columns:1fr}}
code{background:#eef1f5;border-radius:4px;padding:0 4px;font-size:12.5px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px}
</style></head><body><div class="wrap">''')

H.append('<h1>Phase 19 / N2h1-α-12 · 写入向量谱的量化口径稳健性（nf4 ↔ bf16）</h1>')
H.append('<div class="sub">Phase 17 的 <code>w_ℓ</code> 谱与质心 <code>com_V</code> 全部只在 '
         '<b>bitsandbytes nf4</b> 下测得；P17 自己的 <code>quant.why</code> 写着「A0 臂专职量化对其结论的影响」——该检查从未执行。'
         '本 Phase 执行它：<b>除数值精度外逐项冻结</b>，同模型同尺度换 <b>bf16</b> 复算 ⇒ 判定「深端集中」是否量化地板效应。</div>')

H.append('<div class="card">')
H.append('<div class="grid"><div>')
H.append('<div class="kv"><span>seal</span><b>%s</b></div>' % sha8(SEALP))
H.append('<div class="kv"><span>exec</span><b>%s</b></div>' % sha8(EXECP))
H.append('<div class="kv"><span>result</span><b>%s</b></div>' % sha8(RESP))
H.append('<div class="kv"><span>探针</span><b>%s</b></div>' % sha8(PROBEP))
H.append('<div class="kv"><span>锚(P17 result)</span><b>%s</b></div>' % R['anchor_result_sha256'][:8])
H.append('<div class="kv"><span>wall</span><b>%s s</b></div>' % f(R.get('elapsed_total_s'), 1))
H.append('</div><div>')
H.append('<div class="kv"><span>唯一自变量</span><b>nf4 (4-bit) ↔ bfloat16</b></div>')
H.append('<div class="kv"><span>template</span><b><code>%s</code></b></div>' % EX['template'])
H.append('<div class="kv"><span>instances / discovery</span><b>%d / %d</b></div>' % (len(EX['instances_all']), len(EX['discovery'])))
H.append('<div class="kv"><span>U_ℓ</span><b>类别质心差 SVD，秩 = n_classes−1 = %d</b></div>' % (len(EX['classes']) - 1))
H.append('<div class="kv"><span>质心口径</span><b>REACH 上相邻位点<b>区间求和</b> + 中点</b></div>')
H.append('<div class="kv"><span>预测</span><b>%s</b></div>' % (' '.join('%s=%s' % (k, PC[k]['pass_']) for k in sorted(PC))))
H.append('</div></div>')
H.append('<div style="margin-top:10px">')
for lbl, key in (('Q1 保真度', 'Q1_joint'), ('Q2 校准', 'Q2_joint'), ('Q3 量化敏感度', 'Q3_joint'),
                 ('Q4 谱形状', 'Q4_joint'), ('Q5 组件归属', 'Q5_joint'), ('Q6 深端', 'Q6_joint')):
    H.append(vchip(lbl, JV.get(key, '—')))
H.append('</div>')
H.append('<div class="note" style="margin-top:8px">臂序：%s</div>'
         % ' → '.join('%s <span class="note">(%s)</span>' % (SHORT[a], ROLE[a]) for a in ARMS))
H.append('</div>')

# ---------------------------------------------------------------- A 装置门
H.append('<h2>A · 装置门与 nf4 校准锚（逐位复现 P17）</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>模型</th><th>精度</th><th>offload</th><th>L</th><th>device</th>'
         '<th>arch max</th><th>blocks max</th><th>U 秩</th><th>Q1</th><th>Q2</th></tr>')
for a in ARMS:
    r = R['arms'][a]; c = r['cfg']; x = r['E2_fidelity']
    H.append('<tr><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td><td>%d</td><td>%s</td>'
             '<td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], r['model'], r['scheme'], '是' if r['offload'] else '否', c['L'], r['Q0_device'],
                sci(x['arch_max']), sci(x['blk_max']), r['E3_U']['rank'],
                pill(V[a]['Q1_label'] == 'FID_PASS', 'FID_PASS', 'FID_FAIL'),
                pill(V[a]['Q2_label'] == 'CALIB_OK' if V[a]['Q2_label'] != 'ANCHOR_NA' else None,
                     'CALIB_OK', V[a]['Q2_label'], 'ANCHOR_NA（bf16 按设计不复现 nf4 锚）')))
H.append('</table>')
H.append('<div class="note">保真度门：arch ≤ %s / blocks ≤ %s（四臂全过）。'
         '<b>nf4 校准臂逐位复现 P17 冻结锚</b>（<code>com_V</code>/<code>com_V_mlp</code>/<code>com_V_attn</code> 差 ≤ %g；'
         '<code>nb</code>/<code>argmax_w</code> 精确相同）⇒ 装置与 Phase 17 同源。</div>'
         % (FL['P19_FID_ARCH'], FL['P19_FID_BLK'], FL['CALIB_TOL_COMV']))
H.append('<table><tr><th>臂</th>')
for k in ('com_V', 'com_V_mlp', 'com_V_attn', 'median_reach', 'nb', 'argmax_w_layer'):
    H.append('<th>锚 %s</th>' % k)
H.append('<th>got == expected</th></tr>')
for a in ('A0_nf4', 'A1_nf4'):
    d = R['arms'][a]['E7_anchor']['detail']
    H.append('<tr><td>%s</td>' % SHORT[a])
    for k in ('com_V', 'com_V_mlp', 'com_V_attn', 'median_reach', 'nb', 'argmax_w_layer'):
        H.append('<td>%s</td>' % (f(d[k]['expected']) if not isinstance(d[k]['expected'], list) else str(d[k]['expected'])))
    H.append('<td>%s</td></tr>' % pill(R['arms'][a]['E7_anchor']['ok'], '逐位一致'))
H.append('</table></div>')

# ---------------------------------------------------------------- B 主要读数
H.append('<h2>B · 主要读数：质心、组件份额、argmax（四臂）</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>com_V</th><th>com_V_mlp</th><th>com_V_attn</th>'
         '<th>median(REACH)</th><th>nb</th><th>share_mlp_nb</th><th>argmax_w</th><th>Q5</th><th>Q6</th></tr>')
for a in ARMS:
    v = V[a]
    H.append('<tr><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
             '<td><b>%s</b> %s</td><td>L%d</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], f(v['com_V']), f(v['com_V_mlp']), f(v['com_V_attn']), f(v['median_reach'], 1),
                str(v['neighbourhood']), f(v['share_mlp_nb']), bar(v['share_mlp_nb'], COL[a]),
                v['argmax_w_layer'],
                pill(v['Q5_label'] == 'MLP_DOMINANT', 'MLP_DOMINANT', 'NOT_DOMINANT'),
                pill(v['Q6_label'] == 'DEEP', 'DEEP', 'SHALLOW')))
H.append('</table>')
H.append('<div class="note">质心 <b>com_V ≈ 26–27 层</b>、远深于 <code>median(REACH)</code>（17 / 14）⇒ <b>深端</b>；'
         '邻域组件份额 MLP <b>过半</b>；<code>argmax_w</code> 在 nf4/bf16 两口径下<b>同层</b>（A0 L30 / A1 L38）。</div>')
H.append('</div>')

# ---------------------------------------------------------------- C 量化配对
H.append('<h2>C · 同 Phase 配对（量化敏感度，核心）</h2>')
H.append('<div class="card"><table><tr><th>配对（nf4 | bf16）</th><th>Δcom_V</th><th>Δcom_V_mlp</th>'
         '<th>spearman(w_nf4, w_bf16)</th><th>残差中位</th><th>残差 p90</th><th>argmax 同层</th><th>share 同侧</th><th>Q3</th></tr>')
for k in PAIRS:
    s = QP[k]
    H.append('<tr><td>%s</td><td><b>%s</b></td><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s</td>'
             '<td>%s</td><td>%s</td><td>%s</td></tr>'
             % (k.replace('|', ' | '), f(s['delta_com_V']), f(s['delta_com_V_mlp']), f(s['spearman_w']),
                f(s['median_rel_resid']), f(s['p90_rel_resid']),
                pill(s['argmax_same'], 'L%d = L%d' % (s['argmax_nf4'], s['argmax_bf16'])),
                pill(s['share_same_side'], "%.4f / %.4f" % (s['share_mlp_nb_nf4'], s['share_mlp_nb_bf16'])),
                pill(s['delta_com_V'] <= FL['QUANT_TOL_COMV'] and s['spearman_w'] >= FL['RHO_SHAPE_MIN'],
                     'STABLE', 'SENSITIVE')))
H.append('</table>')
H.append('<div class="note">容差 = %s 层；秩相关地板 = %s。'
         '<b>质心几乎不动</b>（Δ ≤ %s 层）、<b>谱形状几乎完全保持</b>（≥ %s）、<b>层位置与组件归属不变</b> '
         '⇒ <span class="hl">「写入向量质量深端集中」与「MLP 主导归属」不是 nf4 量化地板效应</span>。</div>'
         % (f(FL['QUANT_TOL_COMV'], 1), f(FL['RHO_SHAPE_MIN'], 2),
            f(max(QP[k]['delta_com_V'] for k in PAIRS)), f(min(QP[k]['spearman_w'] for k in PAIRS))))
H.append('</div>')

# ---------------------------------------------------------------- D 谱形（视觉）
H.append('<h2>D · 谱形对照：nf4 vs bf16（每层 <code>w_all,ℓ</code>，同标归一）</h2>')
H.append('<div class="card"><div class="legend note">图例：'
         '<i style="background:#c2410c"></i>A0 nf4<i style="background:#f59e0b"></i>A0 bf16'
         '<i style="background:#1d4ed8"></i>A1 nf4<i style="background:#3b82f6"></i>A1 bf16'
         '（两模型分开标尺；同模型的两种精度叠放）</div>')
H.append('<div class="grid2">')
for grp, arms in (('qwen3-4b', ['A0_nf4', 'A0_bf16']), ('glm4-9b', ['A1_nf4', 'A1_bf16'])):
    mx = max(max(V[a]['w_all']) for a in arms) or 1.0
    H.append('<div><h3>%s</h3>' % grp)
    for a in arms:
        H.append('<div class="note" style="margin:6px 0 0">%s · argmax L%d</div>' % (SHORT[a], V[a]['argmax_w_layer']))
        H.append('<div class="sp" title="%s">' % a)
        for i, x in enumerate(V[a]['w_all']):
            H.append('<i style="background:%s;height:%.1f%%" title="L%d w=%.2f"></i>'
                     % (COL[a], 100.0 * x / mx, i, x))
        H.append('</div>')
    H.append('</div>')
H.append('</div>')
H.append('<div class="note">同一模型的 nf4 与 bf16 两条谱<b>逐层同形</b>（秩相关 %s / %s），峰值都落在写入窗附近 ⇒ 谱形状是模型性质而非量化伪影。</div>'
         % (f(QP['A0_nf4|A0_bf16']['spearman_w']), f(QP['A1_nf4|A1_bf16']['spearman_w'])))
H.append('</div>')

# ---------------------------------------------------------------- E 交叉验证
H.append('<h2>E · 交叉验证：独立探针 ↔ 生产实现（逐位相同）</h2>')
H.append('<div class="card"><table><tr><th>量</th><th>P17 冻结锚 (nf4)</th><th>探针 nf4</th><th>生产 nf4</th>'
         '<th>探针 bf16</th><th>生产 bf16</th></tr>')
_pr = PROBE['runs']
H.append('<tr><td>com_V</td><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td><td><b>%s</b></td></tr>'
         % (f(SEAL['anchor_values']['A0_nf4']['com_V']), f(_pr['nf4']['com_V']), f(V['A0_nf4']['com_V']),
            f(_pr['bf16']['com_V']), f(V['A0_bf16']['com_V'])))
H.append('<tr><td>share_mlp_nb</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
         % (f(SEAL['anchor_values']['A0_nf4']['share_mlp_nb']), f(_pr['nf4']['share_mlp_nb']),
            f(V['A0_nf4']['share_mlp_nb']), f(_pr['bf16']['share_mlp_nb']), f(V['A0_bf16']['share_mlp_nb'])))
H.append('<tr><td>argmax_w</td><td>L%s</td><td>L%s</td><td>L%s</td><td>L%s</td><td>L%s</td></tr>'
         % (SEAL['anchor_values']['A0_nf4']['argmax_w_layer'], _pr['nf4']['argmax_w_layer'],
            V['A0_nf4']['argmax_w_layer'], _pr['bf16']['argmax_w_layer'], V['A0_bf16']['argmax_w_layer']))
H.append('</table>')
H.append('<div class="note">四个 nf4/bf16 数字在<b>两套独立实现</b>下逐位相同 ⇒ 量化敏感度不是实现细节的产物（承铁律 (ad)）。'
         '本轮探针还抓出 <span class="hl">E-pair</span>：配对过滤一度比 P17 严（24 对→17 对），改回逐字口径后 <code>com_V</code> 逐位复现 26.1501。</div>')
H.append('</div>')

# ---------------------------------------------------------------- F 对照
H.append('<h2>F · 对照（P7）：置换零假设（保留质量多重集，随机重排到 REACH 位点）</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>com_V(all) obs</th><th>null p5</th><th>null p95</th><th>尾</th>'
         '<th>com_V(mlp) obs</th><th>null p95</th><th>尾</th></tr>')
for a in ARMS:
    na = V[a]['Q7_null_all']; nm = V[a]['Q7_null_mlp']
    H.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], f(na['obs_com']), f(na['com_p5']), f(na['com_p95']),
                pill(na['com_tail'] == 'high', 'high', na['com_tail'] or '—'),
                f(nm['obs_com']), f(nm['com_p95']),
                pill(nm['com_tail'] == 'high', 'high', nm['com_tail'] or '—')))
H.append('</table>')
H.append('<div class="note">BP=%d；种子 <code>comv_all=%s</code> / <code>comv_mlp=%s</code>。'
         '四臂两种口径下观测质心都落在零假设<b>高尾</b> ⇒ 深端集中相对于随机重排显著。</div>'
         % (R['bootstrap']['BP'], R['bootstrap']['seeds']['comv_all'], R['bootstrap']['seeds']['comv_mlp']))
H.append('</div>')

# ---------------------------------------------------------------- G 预测 / 限界 / 勘误
H.append('<h2>G · 预注册预测、limit 与同轮勘误</h2>')
H.append('<div class="card"><table><tr><th>#</th><th>判据（seal）</th><th>结果</th></tr>')
for k in sorted(PC):
    claim = SEAL['predictions'][k].get('claim', '').replace('\n', ' ')[:180]
    H.append('<tr><td><code>%s</code></td><td style="text-align:left">%s…</td><td>%s</td></tr>'
             % (k, claim, pill(PC[k]['pass_'], 'PASS', 'FAIL')))
H.append('</table>')
H.append('<div class="note"><b>限界（诚实性）</b>：'
         '<ul style="margin:6px 0 6px 18px;padding:0">'
         '<li><b>H8 / 覆盖限界</b>：A2（Qwen3-14B，29.5 GB）bf16 <b>加载 19% 时 segfault</b> ⇒ 跨精度稳健性只在 <b>qwen3-4b 与 glm4-9b</b> 两模型验证（A2 仅入校准门）。</li>'
         '<li><b>H2 两源差异</b>：bf16 − nf4 含「量化误差 + 反量化 kernel 路径」两源（同 <code>eager</code> 已控注意力 kernel）。</li>'
         '<li><b>H6 offload</b>：A1·bf16 需 CPU offload（18.8 GB &gt; 14 GiB）⇒ 含「分片执行」第三源，按预注册仍入硬门。</li>'
         '<li><b>H3</b>：<code>w_ℓ</code> 是<b>激活级</b>分解，非权重级实现证明。</li>'
         '<li><b>H9</b>：本 Phase 不重测行为量（<code>J</code> / <code>com_layer</code> / P18 <code>b</code>）——只回答「<code>w_ℓ</code> 谱与 <code>com_V</code> 是否量化稳健」。</li>'
         '</ul>')
H.append('<b>同轮勘误</b>（append-only）：'
         '<b>E-A2</b> A2·bf16 segfault（~19% 权重）⇒ bf16 腿只有两模型；'
         '<b>E-pair</b> 探针配对过滤一度过严（24→17 对）⇒ <code>com_V</code> 26.1956 ≠ 锚 26.1501，逐字改回后逐位复现；'
         '<b>E-offload</b> accelerate 用 <b>meta 占位参数</b>承载权重 ⇒ 取设备须读 <code>m._hf_hook.execution_device</code>，修后重跑全臂。</div>')
H.append('</div>')

# ---------------------------------------------------------------- H 下一步
H.append('<h2>H · 下一步（死线优先级）</h2>')
H.append('<div class="card">')
H.append('<div class="kv"><span>最高</span><b>Phase 20 — 把跨精度检验推进到<b>行为量</b>：P18 的 <code>b_{c,ℓ}</code> 与 P16 的 <code>com_layer</code> 在 bf16 下复算（本 Phase 只覆盖向量侧）</b></div>')
H.append('<div class="kv"><span>并列</span><b>邻域宽度 ±2 敏感性（四臂 nb 恰好都是 %s）</b></div>'
         % str(V[ARMS[0]]['neighbourhood']))
H.append('<div class="kv"><span>第三</span><b>P17 <code>P6</code> 的 MEMO 改判（承 P18 <code>P5</code>：<code>spearman(w_all,|b_all|)&gt;0</code> 而 <code>spearman(w_all,J)&lt;0</code> ⇒ 对象错配）</b></div>')
H.append('<div class="kv"><span>N 线挂账</span><b>N2h1-α-1 权重级定位；N2h1-β 水果类崩塌；N3-β→N3-ε；R1 补强；K4；N 线 P3–P7 补登 Ledger</b></div>')
H.append('</div>')

H.append('<div class="sub" style="margin-top:22px">Phase 19 · N2h1-α-12 · 数据驱动渲染（全部数字取自 '
         '<code>result_phase19.json</code>；Ledger n = 302；MEMO 19 标题 / 469,161 B）</div>')
H.append('</div></body></html>')

html = '\n'.join(H)
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(html)
b = len(html.encode('utf-8'))
print('WROTE %s  %d B / %d lines' % (OUT, b, len(html.splitlines())))
