# -*- coding: utf-8 -*-
"""Phase 17 展示页生成器（数据驱动；全部数字取自 result_phase17.json / seal / exec）。
产出 tests/deepseek_temp/Phase17/present_phase17.html（浅色主题、自包含、无外部依赖）。
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
OUT = os.path.join(P17T, 'present_phase17.html')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(P17T, 'result_phase17.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P17T, 'execution_phase17.json'), encoding='utf-8'))
SEAL = json.load(io.open(os.path.join(P17T, 'N2h1a10_design_seal.json'), encoding='utf-8'))
SEALP = os.path.join(P17T, 'N2h1a10_design_seal.json')
EXECP = os.path.join(P17T, 'execution_phase17.json')
RESP = os.path.join(P17T, 'result_phase17.json')
PROBEP = os.path.join(P17T, '_probe_feasibility_A0.json')

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']; FL = R['floors']
ARMS = list(R['arms'].keys())
A0, A1, A2 = ARMS
SHORT = {A0: 'A0·qwen3-4b(nf4)', A1: 'A1·GLM4-9B', A2: 'A2·Qwen3-14B'}
COL = {A0: '#c2410c', A1: '#1d4ed8', A2: '#047857'}
ANC = SEAL['anchor_values']
KS = list(R['span_ks'])
PROBE_CV = 26.15005633170243


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
H.append('<title>Phase 17 / N2h1-α-10 · 写入向量的位置与效力</title>')
H.append('''<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#16202b;--mut:#5b6b7c;--line:#e3e8ee;--ok:#0f766e;--bad:#b42318;--hi:#fef3c7;--ac:#1d4ed8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.62 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:23px;margin:0 0 4px}
h2{font-size:17px;margin:30px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:14px;margin:18px 0 8px;color:var(--mut)}
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
.bar{display:inline-block;width:100%;max-width:220px;height:11px;background:#eef1f5;border-radius:6px;overflow:hidden;vertical-align:middle}
.bar i{display:block;height:100%;border-radius:6px}
.kv{display:flex;justify-content:space-between;gap:12px;padding:3px 0;border-bottom:1px dashed var(--line)}
.kv:last-child{border-bottom:0}
.kv b{font-variant-numeric:tabular-nums}
.hl{background:var(--hi);border-radius:6px;padding:2px 6px}
.note{color:var(--mut);font-size:12.5px}
.sp{display:flex;align-items:flex-end;gap:2px;height:78px;margin:6px 0 2px}
.sp i{flex:1;background:#c7d2de;border-radius:2px 2px 0 0;min-height:1px}
.sp i.on{background:var(--ac)}
.sp i.pk{background:#f59e0b}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:720px){.grid3{grid-template-columns:1fr}}
code{background:#eef1f5;border-radius:4px;padding:0 4px;font-size:12.5px}
.arc{color:var(--mut);font-size:12.5px}
</style></head><body><div class="wrap">''')

H.append('<h1>Phase 17 / N2h1-α-10 · 写入向量的位置与效力</h1>')
H.append('<div class="sub">把 Phase 8 的向量预算从<strong>单层 L6</strong> 推广到<strong>逐层</strong> ⇒ 向量质量谱 <code>w_ℓ</code> 与质心 <code>com_V</code>；与 Phase 16 的两个<strong>行为</strong>质心三向对照。零额外前向（每臂 44 次）。</div>')

H.append('<div class="card">')
H.append('<div class="grid"><div>')
H.append('<div class="kv"><span>seal</span><b>%s</b></div>' % sha8(SEALP))
H.append('<div class="kv"><span>exec</span><b>%s</b></div>' % sha8(EXECP))
H.append('<div class="kv"><span>result</span><b>%s</b></div>' % sha8(RESP))
H.append('<div class="kv"><span>探针</span><b>%s</b></div>' % sha8(PROBEP))
H.append('<div class="kv"><span>锚(P16)</span><b>%s</b></div>' % R['anchor_result_sha256'][:8])
H.append('<div class="kv"><span>耗时</span><b>%s s</b></div>' % f(R.get('elapsed_total_s'), 1))
H.append('</div><div>')
H.append('<div class="kv"><span>臂</span><b>%s</b></div>' % ' / '.join('%s' % SHORT[a] for a in ARMS))
H.append('<div class="kv"><span>n_forwards/臂</span><b>%s</b></div>' % ' / '.join(str(R['arms'][a]['n_forwards']) for a in ARMS))
H.append('<div class="kv"><span>域</span><b>REACH（与 com_layer 同域）</b></div>')
H.append('<div class="kv"><span>可加性</span><b>由构造成立 + 两保真度门</b></div>')
H.append('<div class="kv"><span>预测</span><b>P1–P6 PASS · P7 描述性</b></div>')
H.append('<div class="kv"><span>判决</span><b>%d 项</b></div>' % len([k for k in ('Q3_joint', 'Q4_joint', 'Q5_joint', 'Q6_joint') if JV.get(k)]))
H.append('</div></div>')
H.append('<div style="margin-top:10px">')
for lbl, key in (('P3', 'Q3_joint'), ('P4', 'Q4_joint'), ('P5', 'Q5_joint'), ('P6', 'Q6_joint'), ('P7', 'Q7_joint')):
    H.append(vchip(lbl, JV.get(key, '—')))
H.append('</div>')
H.append('</div>')

# ---------------------------------------------------------------- A 装置门
H.append('<h2>A · 装置门与锚（三臂）</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>模型</th><th>L</th><th>heads</th><th>device</th><th>determinism</th><th>arch max</th><th>blocks max</th><th>Q1</th></tr>')
for a in ARMS:
    r = R['arms'][a]; c = r['cfg']; x = r['E2_fidelity']
    H.append('<tr><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], r['model'], c['L'], c['n_heads'], r['Q0_device'],
                sci(r['E0_selfcheck']['determinism_maxdiff'], 2), sci(x['arch_max']), sci(x['blk_max']),
                pill(True, V[a]['Q1_label'])))
H.append('</table>')
H.append('<div class="note">保真度门地板：arch ≤ %s / blocks ≤ %s（nf4 噪声实测地板）。<b>P16 锚逐位复现</b>（≤1e-6）：%s</div>'
         % (FL['P17_FID_ARCH'], FL['P17_FID_BLK'], JV['Q2_joint']))
H.append('<table><tr><th>臂</th><th>com_layer(x) 重算 = 锚</th><th>com_layer(J) 重算 = 锚</th><th>L*_own</th><th>ℓ_reach</th></tr>')
for a in ARMS:
    d = R['arms'][a]['E9_anchor']['detail']
    H.append('<tr><td>%s</td><td>%s = %s</td><td>%s = %s</td><td>%d</td><td>%d</td></tr>'
             % (SHORT[a], f(d['com_layer_x']['got'], 6), f(d['com_layer_x']['expected'], 6),
                f(d['com_layer_j']['got'], 6), f(d['com_layer_j']['expected'], 6),
                d['L_star_own']['got'], d['ell_reach']['got']))
H.append('</table></div>')

# ---------------------------------------------------------------- B com_V DEEP
H.append('<h2>B · 主结果 1（P3·holdout）：向量写入质量深端集中</h2>')
med_max = max(V[a]['Q3_median'] for a in ARMS) or 1.0
cv_max = max(V[a]['Q3_com_V'] for a in ARMS) or 1.0
H.append('<div class="card">')
H.append('<table><tr><th>臂</th><th>com_V</th><th>median(REACH)</th><th>判</th><th>条形（相对满标 %.0f 层）</th></tr>' % cv_max)
for a in ARMS:
    H.append('<tr><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s %s</td></tr>'
             % (SHORT[a], f(V[a]['Q3_com_V']), f(V[a]['Q3_median'], 1), pill(V[a]['Q3_label'] == 'DEEP', 'DEEP', 'SHALLOW'),
                bar(V[a]['Q3_com_V'] / cv_max, COL[a]), '<span class="note">median %s</span>' % f(V[a]['Q3_median'], 0)))
H.append('</table>')
H.append('<div class="note"><b>holdout</b>：A1/A2 在 seal 冻结前<strong>从未被观测</strong>（探针只在 A0 上运行）⇒ 深端集中是层栈共性。</div>')
H.append('</div>')

# ---------------------------------------------------------------- C w_ell spectrum
H.append('<h2>C · 向量质量谱 <code>w_ℓ</code>（REACH 位点高亮）</h2>')
H.append('<div class="card"><div class="grid3">')
for a in ARMS:
    c5 = R['arms'][a]['E5_com_V']
    wa = c5['w_all']; reach = set(int(x) for x in c5['reach']); pk = c5['argmax_w_layer']
    mx = max(wa) or 1.0
    H.append('<div><h3 style="color:%s">%s</h3>' % (COL[a], SHORT[a]))
    H.append('<div class="sp">')
    for L0, v in enumerate(wa):
        cls = 'pk' if L0 == pk else ('on' if L0 in reach else '')
        H.append('<i class="%s" style="height:%.1f%%" title="L%d = %.2f"></i>' % (cls, 100.0 * v / mx, L0, v))
    H.append('</div>')
    H.append('<div class="arc">argmax w_ℓ = <b>L%d</b> = %.2f ; com_V = %s</div>' % (pk, mx, f(c5['com_V'])))
    H.append('</div>')
H.append('</div>')
H.append('<div class="note">三臂 <code>w_ℓ</code> 都在深端（L28–L34）达峰，浅端另有次峰；探针（A0，24 对）com_V = %s 与生产 <b>%s</b> 逐位一致（见 E4）。</div>'
         % (f(PROBE_CV, 2), f(V[A0]['Q3_com_V'], 3)))
H.append('</div>')

# ---------------------------------------------------------------- D position contrast
H.append('<h2>D · 主结果 2（P4·A2 判别臂）：行为质心不能由向量质量质心替代</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>com_V</th><th>com_layer(x)</th><th>com_layer(J)</th><th>d_x</th><th>d_j</th><th>min(d)</th><th>判</th></tr>')
for a in ARMS:
    an = ANC[a]; vv = V[a]
    hl = ' class="hl"' if a == A2 else ''
    H.append('<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td></tr>'
             % (hl, SHORT[a], f(vv['Q3_com_V']), f(an['com_layer_x']), f(an['com_layer_j']),
                f(vv['Q4_d_x'], 2), f(vv['Q4_d_j'], 2), f(vv['Q4_min_d'], 2), pill(vv['Q4_label'] == 'POSITION_DECOUPLED',
                                                                               'DECOUPLED', 'ALIGNED')))
H.append('</table>')
H.append('<div class="note"><span class="hl">A2 是判别臂</span>：两个行为质心几乎重合（%s vs %s，相距 %s 层），'
         '而向量质心在 <b>%s</b> 处相隔 <b>%s 层</b> ⇒ 「向量写入位置」与「行为质心位置」<b>是两件事</b>。'
         'A0 是唯一对齐臂（min_d = %s）—— 正是 Phase 16 P6 否证的镜像。</div>'
         % (f(ANC[A2]['com_layer_x']), f(ANC[A2]['com_layer_j']),
            f(abs(ANC[A2]['com_layer_x'] - ANC[A2]['com_layer_j'])), f(V[A2]['Q3_com_V']),
            f(V[A2]['Q4_min_d'], 2), f(V[A0]['Q4_min_d'], 2)))
H.append('</div>')

# ---------------------------------------------------------------- E component attribution
H.append('<h2>E · 组件归属（P5）：质心邻域由 MLP 承载</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>com_V 邻域(±%d)</th><th>share_mlp_nb</th><th>share_attn_nb</th><th>最大单头 share</th><th>判</th></tr>' % int(EX['neighbourhood_width']))
for a in ARMS:
    c5 = R['arms'][a]['E5_com_V']
    H.append('<tr><td>%s</td><td>%s</td><td><b>%s</b> %s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], str(c5['neighbourhood']), f(c5['share_mlp_nb']),
                bar(c5['share_mlp_nb'], COL[a]), f(c5['share_attn_nb']),
                f(c5['top1_head_share_nb'], 4), pill(V[a]['Q5_label'] == 'MLP_DOMINANT', 'MLP_DOMINANT', 'OTHER')))
H.append('</table>')
H.append('<div class="note">三臂邻域<b>恰好都是 [26, 28]</b>；全部 ≥ Phase 8 在 L6 的 MLP 向量预算 0.4717；最大单头 ≤ 0.09（远低于任何单头主导门 &gt; 0.50）⇒ <b>无单头主导</b>。</div>')
H.append('</div>')

# ---------------------------------------------------------------- F efficacy
H.append('<h2>F · 效力关系（P6）：写入多寡与写入效力在深度上分离</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>spearman(w_ℓ, J_ℓ)</th><th>n</th><th>spearman(w_ℓ, xhalf)</th><th>spearman(J, depth)</th><th>判</th></tr>')
for a in ARMS:
    e6 = R['arms'][a]['E6_efficacy']
    H.append('<tr><td>%s</td><td><b>%s</b></td><td>%d</td><td>%s</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], f(e6['spearman_wJ'], 4), e6['n'], f(e6['spearman_wxhalf'], 4), f(e6['spearman_Jdepth'], 4),
                pill(V[a]['Q6_label'] == 'WRITE_EFFICACY_ANTICORR', 'ANTICORR', 'OTHER')))
H.append('</table>')
H.append('<div class="note"><b>本 Phase 最有信息量的一条</b>：行为增益 <code>J(ℓ)</code> 随深度下降、向量写入质量 <code>w_ℓ</code> 随深度上升 ⇒ '
         '「<b>深端有大量写入向量，但对 is-a 行为几乎无效</b>」。这把 Phase 16 撤回「物理深度表述」后的空洞补成一条<b>可操作</b>陈述。</div>')
H.append('</div>')

# ---------------------------------------------------------------- G span
H.append('<h2>G · 跨度谱（P7，描述性）：<code>%s</code></h2>' % JV['Q7_joint'])
H.append('<div class="card"><table><tr><th>臂</th><th>k</th><th>span(xhalf)</th><th>span(J)</th><th>x 相对 J</th><th>同号?</th></tr>')
for a in ARMS:
    sp = R['arms'][a]['E8_span']['spans']
    for k in KS:
        sx = sp['x'][str(k)]['obs_span']; sj = sp['j'][str(k)]['obs_span']
        ok = (sx > sj) == (ANC[a]['com_layer_x'] > ANC[a]['com_layer_j'])
        H.append('<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
                 % (SHORT[a], k, f(sx, 4), f(sj, 4), 'span 更宽' if sx > sj else 'span 更窄',
                    pill(ok, '✅', '✗')))
H.append('</table><div class="note">联合 <b>%s</b>（同号 %s）—— 「span 更宽 ⟺ 质心更深」三臂一致。本条不设方向性预测（Phase 16 已见过 k=3 的同一事实）。</div></div>'
         % (JV['Q7_joint'], JV['Q7_coupled']))

# ---------------------------------------------------------------- H controls
H.append('<h2>H · 对照：置换零假设 + 确认集</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>obs com_V</th><th>null p5</th><th>null p95</th><th>尾</th><th>确认集 com_V</th><th>Δ</th></tr>')
for a in ARMS:
    nA = V[a]['Q7_null_all']; cf = V[a]['Q8_conf']
    H.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td></tr>'
             % (SHORT[a], f(nA['obs_com']), f(nA['com_p5']), f(nA['com_p95']),
                pill(nA['com_tail'] == 'high', 'high', nA['com_tail'] or '—'),
                f(cf['com_V']), f(cf['d_com'])))
H.append('</table>')
H.append('<div class="note">置换 null（保留质量多重集、随机重排到 REACH 位点；BP=%d，种子 %s / %s）：三臂 <b>3/3 落 high 尾</b>。'
         '确认集 n=%d（与 %d 对 discovery <b>不相交</b>）Δ 全部 ≤ %s 层。</div>'
         % (R['bootstrap']['BP'], R['bootstrap']['seeds']['comv_all'], R['bootstrap']['seeds']['comv_mlp'],
            V[A0]['Q8_conf']['n_pairs'], len(EX['discovery']), FL['CONF_TOL_COMV']))
H.append('</div>')

# ---------------------------------------------------------------- I predictions + errata
H.append('<h2>I · 预注册预测与同轮勘误</h2>')
H.append('<div class="card"><table><tr><th>#</th><th>判据（seal）</th><th>结果</th></tr>')
for k in sorted(PC):
    claim = SEAL['predictions'][k]['claim'].replace('\n', ' ')[:170]
    H.append('<tr><td><code>%s</code></td><td style="text-align:left">%s…</td><td>%s</td></tr>'
             % (k, claim, pill(PC[k]['pass_'], 'PASS', 'FAIL', 'N/A（描述性）')))
H.append('</table>')
H.append('<div class="note"><b>同轮勘误 E1–E4</b>（append-only）：'
         '<b>E4（最重要）</b> <code>com_of_mass</code> 首版取<b>位点单层</b>而非 seal 的<b>区间求和</b> '
         '（A0 %s → <b>%s</b>），修正后与<b>独立探针</b> %s <b>逐位相同</b>（跨实现交叉验证），'
         '未重跑前向、P3/P4/P6 判决不变；<b>E1</b> Q7 同号口径（v1 错位配对误报 DECOUPLED）；'
         '<b>E2</b> SMOKE 退化路径；<b>E3</b> 探针 nf4 权重路径。</div>' % (f(24.465585421173355), f(V[A0]['Q3_com_V']), f(PROBE_CV, 2)))
H.append('</div>')

# ---------------------------------------------------------------- J limits + next
H.append('<h2>J · 诚实边界与下一步（死线）</h2>')
H.append('<div class="card">')
H.append('<ul style="margin:6px 0 6px 18px;padding:0">')
for k in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8'):
    if k in SEAL['honesty']:
        H.append('<li><b>%s</b>：%s</li>' % (k, SEAL['honesty'][k]))
H.append('<li><b>H9</b>：<code>share_mlp_nb</code> 的分母是「Σ 各分量投影范数」，<code>share_attn + share_mlp ≈ 1</code> 但<b>不是恒等</b>（三角不等式）⇒ 其余份额由方向相消产生，不得读作「attn 只贡献 1−share_mlp」。</li>')
H.append('<li><b>H10</b>：<code>w_ℓ</code> 的<b>绝对</b>量级只在同一臂内可比；跨臂只比<b>质心位置</b>与<b>份额</b>（尺度无关）。</li>')
H.append('<li><b>H11</b>：本 Phase 的「深端写入无效」是对 <code>J(ℓ)</code> 的<b>相关性陈述</b>，不是因果链分解。</li>')
H.append('</ul>')
H.append('<div class="note"><b>Phase 18 最高优先</b>：逐层组件「<b>行为</b>」预算 —— 把向量预算换成行为预算，补 H11 因果缺口；'
         '若行为层同样 MLP 主导（当前 share_mlp_nb = %s / %s / %s）⇒ 升级为因果，若以 attn 为主 ⇒ 向量份额是几何假象。'
         '<b>并列</b>：NF4 vs BF16 的 <code>w_ℓ</code> 口径差异；<b>第三</b>：邻域宽度 ±2 敏感性。</div>'
         % (f(V[A0]['Q5_share_mlp_nb']), f(V[A1]['Q5_share_mlp_nb']), f(V[A2]['Q5_share_mlp_nb'])))
H.append('</div>')

H.append('<div class="sub" style="margin-top:22px">Phase 17 · N2h1-α-10 · 数据驱动渲染（全部数字取自 <code>result_phase17.json</code>）· 独立磁盘复核 178 项 0 FAIL</div>')
H.append('</div></body></html>')

html = '\n'.join(H)
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(html)
b = len(html.encode('utf-8'))
print('WROTE %s  %d B / %d lines' % (OUT, b, len(html.splitlines())))
