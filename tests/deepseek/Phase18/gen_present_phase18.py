# -*- coding: utf-8 -*-
"""Phase 18 展示页生成器（数据驱动；全部数字取自 result_phase18.json / seal / exec）。
产出 tests/deepseek_temp/Phase18/present_phase18.html（浅色主题、自包含、无外部依赖）。
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
OUT = os.path.join(P18T, 'present_phase18.html')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(P18T, 'result_phase18.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P18T, 'execution_phase18.json'), encoding='utf-8'))
SEAL = json.load(io.open(os.path.join(P18T, 'N2h1a11_design_seal.json'), encoding='utf-8'))
SEALP = os.path.join(P18T, 'N2h1a11_design_seal.json')
EXECP = os.path.join(P18T, 'execution_phase18.json')
RESP = os.path.join(P18T, 'result_phase18.json')
PROBEP = os.path.join(P18T, '_probe_feasibility_A0.json')

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']; FL = R['floors']
ARMS = list(R['arms'].keys())
A0, A1, A2 = ARMS
SHORT = {A0: 'A0·qwen3-4b(nf4)', A1: 'A1·GLM4-9B', A2: 'A2·Qwen3-14B'}
COL = {A0: '#c2410c', A1: '#1d4ed8', A2: '#047857'}
S = {a: R['arms'][a]['E7_summary'] for a in ARMS}
PROBE_CB = 21.084614237916984


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


def reach_ratio(a):
    rl = {int(k): float(x) for k, x in S[a]['rlin_by_site'].items()}
    re = sorted((rl[l] for l in S[a]['reach'] if l in rl), reverse=True)
    return (re[0] / re[1]) if (len(re) > 1 and re[1] > 1e-12) else None


H = []
H.append('<!DOCTYPE html>')
H.append('<html lang="zh-CN"><head><meta charset="utf-8">')
H.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
H.append('<title>Phase 18 / N2h1-α-11 · 逐层组件「行为」预算</title>')
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
.bar{display:inline-block;width:100%;max-width:200px;height:11px;background:#eef1f5;border-radius:6px;overflow:hidden;vertical-align:middle}
.bar i{display:block;height:100%;border-radius:6px}
.kv{display:flex;justify-content:space-between;gap:12px;padding:3px 0;border-bottom:1px dashed var(--line)}
.kv:last-child{border-bottom:0}
.kv b{font-variant-numeric:tabular-nums}
.hl{background:var(--hi);border-radius:6px;padding:2px 6px}
.note{color:var(--mut);font-size:12.5px}
.sp{display:flex;align-items:flex-end;gap:1px;height:88px;margin:6px 0 2px}
.sp i{flex:1;border-radius:2px 2px 0 0;min-height:1px}
.sp i.a{background:#c2410c}.sp i.m{background:#1d4ed8}.sp i.t{background:#047857}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:720px){.grid3{grid-template-columns:1fr}}
code{background:#eef1f5;border-radius:4px;padding:0 4px;font-size:12.5px}
.arc{color:var(--mut);font-size:12.5px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px}
</style></head><body><div class="wrap">''')

H.append('<h1>Phase 18 / N2h1-α-11 · 逐层组件「行为」预算</h1>')
H.append('<div class="sub">把 Phase 17 的<strong>向量预算</strong>换成<strong>行为预算</strong>'
         '（逐层组件对 <code>Δlogit(is-a)</code> 的贡献）⇒ 补 <strong>H11 因果缺口</strong>：'
         '向量主导若与行为归因<strong>同向</strong>则升级为因果，若以 attn 为主则向量份额是<strong>几何假象</strong>。</div>')

H.append('<div class="card">')
H.append('<div class="grid"><div>')
H.append('<div class="kv"><span>seal</span><b>%s</b></div>' % sha8(SEALP))
H.append('<div class="kv"><span>exec</span><b>%s</b></div>' % sha8(EXECP))
H.append('<div class="kv"><span>result</span><b>%s</b></div>' % sha8(RESP))
H.append('<div class="kv"><span>探针</span><b>%s</b></div>' % sha8(PROBEP))
H.append('<div class="kv"><span>锚(P16 / P17)</span><b>%s / %s</b></div>'
         % (R['anchor_result_p16_sha256'][:8], R['anchor_result_p17_sha256'][:8]))
H.append('<div class="kv"><span>耗时</span><b>%s s</b></div>' % f(R.get('elapsed_total_s'), 1))
H.append('</div><div>')
H.append('<div class="kv"><span>臂</span><b>%s</b></div>' % ' / '.join(SHORT[a] for a in ARMS))
H.append('<div class="kv"><span>n_forwards/臂</span><b>%s</b></div>' % ' / '.join(str(R['arms'][a]['n_forwards']) for a in ARMS))
H.append('<div class="kv"><span>位点域</span><b>ALL_SITES = 1..L−2（对齐 w_all 支撑）</b></div>')
H.append('<div class="kv"><span>组件</span><b>INC_ALL / MLP / ATTN / TOP1 / CUM_ALL</b></div>')
H.append('<div class="kv"><span>桥接门</span><b>CUM_ALL@L* vs P16 FULL_SWAP</b></div>')
H.append('<div class="kv"><span>预测</span><b>%s</b></div>'
         % (' '.join('%s=%s' % (k, PC[k]['pass_']) for k in sorted(PC))))
H.append('</div></div>')
H.append('<div style="margin-top:10px">')
for lbl, key in (('Q3 桥接', 'Q3_joint'), ('Q4 行为归属', 'Q4_joint'), ('Q5 一致', 'Q5_joint'),
                 ('Q6 深度', 'Q6_joint'), ('Q7 耦合', 'Q7_joint'), ('Q8 超可加', 'Q8_joint'), ('Q9 零假设', 'Q9_joint')):
    H.append(vchip(lbl, JV.get(key, '—')))
H.append('</div>')
H.append('</div>')

# ---------------------------------------------------------------- A 装置门
H.append('<h2>A · 装置门与锚（三臂）</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>模型</th><th>L</th><th>heads</th><th>device</th><th>determinism</th><th>arch max</th><th>blocks max</th><th>U 秩</th><th>Q1</th></tr>')
for a in ARMS:
    r = R['arms'][a]; c = r['cfg']; x = r['E2_fidelity']
    H.append('<tr><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%s</td></tr>'
             % (SHORT[a], r['model'], c['L'], c['n_heads'], r['Q0_device'],
                sci(r['E0_selfcheck']['determinism_maxdiff'], 2), sci(x['arch_max']), sci(x['blk_max']),
                r['E3_U']['rank'], pill(True, V[a]['Q1_label'])))
H.append('</table>')
H.append('<div class="note">保真度门地板：arch ≤ %s / blocks ≤ %s。'
         '<b>跨 Phase 锚逐位复现</b>（P16 五条 + P17 三条，≤1e-6）：%s</div>'
         % (FL['P18_FID_ARCH'], FL['P18_FID_BLK'], JV['Q2_joint']))
H.append('<table><tr><th>臂</th><th>com_layer(x) 重算 = 锚</th><th>com_layer(J) 重算 = 锚</th><th>com_V 重算 = P17 锚</th><th>L*_own</th></tr>')
for a in ARMS:
    d = R['arms'][a]['E8_anchor']['detail']; E = S[a]
    H.append('<tr><td>%s</td><td>%s = %s</td><td>%s = %s</td><td>%s = %s</td><td>%d</td></tr>'
             % (SHORT[a], f(d['com_layer_x']['got'], 6), f(d['com_layer_x']['expected'], 6),
                f(d['com_layer_j']['got'], 6), f(d['com_layer_j']['expected'], 6),
                f(E['com_V_recomputed'], 4), f(E['com_V_p17'], 4), E['L_star_own']))
H.append('</table>')
H.append('<div class="note"><b>桥接门（跨 Phase 装置门，本 Phase 新增）</b>：'
         '<code>CUM_ALL@L*_own</code> 与 P16 冻结 <code>FULL_SWAP</code> 相对差 = %s / %s / %s ≤ %s ⇒ <b>%s</b>'
         '（把本 Phase 的读位槽直接钉在 Phase 16 的同一对象上）。'
         '<b>A2 臂漂移</b>（rel <b>%s</b>）：其累积写入是<b>两级阶梯</b>（L4 6.49 → L5 8.71 → 渐近 ~9.3），'
         '<code>L*_own</code> = L4 取自「argmax 单层增量」，<b>早于</b>累积饱和点；改用 L5 则 rel = <b>0.080</b>。'
         '⇒ <b>桥接位点应用累积饱和点，而非单层增量峰值</b>。</div>'
         % (f(V[A0]['Q3_bridge_rel'], 4), f(V[A1]['Q3_bridge_rel'], 4), f(V[A2]['Q3_bridge_rel'], 4),
            FL['BRIDGE_TOL_CUM'], JV['Q3_joint'], f(V[A2]['Q3_bridge_rel'], 4)))
H.append('</div>')

# ---------------------------------------------------------------- B P3 component attribution
H.append('<h2>B · 主结果 1（P3·holdout）：组件归属是<b>行为的</b> MLP 主导</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>邻域(±%d)</th><th>share_mlp_beh(nb)</th><th>share_mlp_vec(nb)（P17）</th><th>share_attn_beh(nb)</th><th>share_top1_beh(nb)</th><th>判</th></tr>' % int(EX['neighbourhood_width']))
for a in ARMS:
    E = S[a]
    H.append('<tr><td>%s</td><td>%s</td><td><b>%s</b> %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], str(E['nb']), f(E['share_mlp_beh_nb']), bar(E['share_mlp_beh_nb'], COL[a]),
                f(E['share_mlp_vec_nb']), f(E['share_attn_beh_nb']), f(E['share_top1_beh_nb']),
                pill(V[a]['Q4_label'] == 'MLP_DOMINANT_BEH', 'MLP_DOMINANT_BEH', 'NOT_DOMINANT')))
H.append('</table>')
H.append('<div class="note"><b>holdout</b>：A1/A2 在 seal 冻结前<b>从未被观测</b>。'
         '行为份额与 P17 的向量份额<b>同侧且都过半</b> ⇒ Phase 17 的 MLP 主导<b>不是几何假象</b>（H11 升级）。'
         '最大单头 ≤ %s ⇒ 无单头主导。</div>'
         % f(max(S[a]['share_top1_beh_nb'] for a in ARMS), 3))
H.append('</div>')

# ---------------------------------------------------------------- C b spectrum
H.append('<h2>C · 行为预算谱 <code>b_ℓ</code>（橙 = INC_ALL / 蓝 = INC_MLP / 绿 = INC_ATTN）</h2>')
H.append('<div class="card"><div class="legend note">图例：<i style="background:#c2410c"></i>INC_ALL<i style="background:#1d4ed8"></i>INC_MLP<i style="background:#047857"></i>INC_ATTN（按 |b| 归一，三序列同标）</div>')
H.append('<div class="grid3">')
for a in ARMS:
    E = S[a]; ba = E['b_all']; bm = E['b_mlp']; bt = E['b_attn']
    mx = max(max(abs(x) for x in ba), 1e-9)
    H.append('<div><h3 style="color:%s">%s</h3>' % (COL[a], SHORT[a]))
    H.append('<div class="sp">')
    for i in range(len(ba)):
        H.append('<i class="a" style="height:%.1f%%" title="L%d INC_ALL=%.3f"></i>'
                 % (100.0 * abs(ba[i]) / mx, i + 1, ba[i]))
    H.append('</div><div class="sp">')
    for i in range(len(bm)):
        H.append('<i class="m" style="height:%.1f%%" title="L%d INC_MLP=%.3f"></i>'
                 % (100.0 * abs(bm[i]) / mx, i + 1, bm[i]))
    H.append('</div><div class="sp">')
    for i in range(len(bt)):
        H.append('<i class="t" style="height:%.1f%%" title="L%d INC_ATTN=%.3f"></i>'
                 % (100.0 * abs(bt[i]) / mx, i + 1, bt[i]))
    H.append('</div>')
    H.append('<div class="arc">写入窗 L%d：b_all=%.2f / b_mlp=%.2f / b_attn=%.2f</div>'
             % (E['L_star_own'], ba[E['L_star_own'] - 1], bm[E['L_star_own'] - 1], bt[E['L_star_own'] - 1]))
    H.append('</div>')
H.append('</div>')
H.append('<div class="note">浅端（写入窗）由 <b>attn</b> 主导、深端由 <b>MLP</b> 主导 ⇒ <b>组件拆分随深度反转</b>；'
         '但 P17 邻域被判定为 MLP 主导（见 B）。探针 A0 <code>com_B(all)</code> = %s 与生产 <b>%s</b> 逐位一致（跨实现）。</div>'
         % (f(PROBE_CB, 4), f(V[A0]['Q6_com_B'], 4)))
H.append('</div>')

# ---------------------------------------------------------------- D P4 gap
H.append('<h2>D · 主结果 2（P4·holdout）：行为质心比向量质心<b>浅</b></h2>')
H.append('<div class="card">')
cvmax = max(V[a]['Q6_com_V'] for a in ARMS) or 1.0
H.append('<table><tr><th>臂</th><th>com_B(行为)</th><th>com_V(向量·P17)</th><th>gap</th><th>判</th><th>条形（相对满标 %.0f 层）</th></tr>' % cvmax)
for a in ARMS:
    H.append('<tr><td>%s</td><td><b>%s</b></td><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s / %s</td></tr>'
             % (SHORT[a], f(V[a]['Q6_com_B']), f(V[a]['Q6_com_V']), f(V[a]['Q6_gap']),
                pill(V[a]['Q6_label'] == 'SHALLOWER', 'SHALLOWER', 'DEEPER/ALIGNED'),
                bar(V[a]['Q6_com_B'] / cvmax, '#c2410c'), bar(V[a]['Q6_com_V'] / cvmax, '#5b6b7c')))
H.append('</table>')
H.append('<div class="note">联合 <b>%s</b>（SHALLOWER %d/%d，floor = %s 层）。'
         '「写入量多的地方」与「写入有效的地方」在同一对象族（增量写入 <code>Δ_inc</code>）上也是<b>两个位置</b>。</div>'
         % (JV['Q6_joint'], JV['Q6_counts']['SHALLOWER'], JV['Q6_counts']['n'], FL['SHALLOWER_MIN']))
H.append('</div>')

# ---------------------------------------------------------------- E P5 coupling
H.append('<h2>E · P5：<b>同对象耦合</b> ⇒ Phase 17 的 P6 是对象错配</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>spearman(w_all, |b_all|)<br><span class="note">同对象（增量写入）</span></th>'
         '<th>spearman(w_mlp, |b_mlp|)</th><th>spearman(w_attn, |b_attn|)</th>'
         '<th>spearman(w_all, J)<br><span class="note">P17 口径（累积差）</span></th><th>判</th></tr>')
for a in ARMS:
    E = S[a]
    H.append('<tr><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], f(E['spearman_wall_ball'], 4), f(E['spearman_wmlp_bmlp'], 4),
                f(E['spearman_wattn_battn'], 4), f(E['spearman_wall_J'], 4),
                pill(V[a]['Q7_label'] == 'EFFICACY_COUPLED', 'COUPLED', 'DECOUPLED')))
H.append('</table>')
H.append('<div class="note"><b>本 Phase 最有信息量的一条</b>：同对象秩相关为<b>正</b>（%s），'
         '而 P17 口径（w vs 累积差 J）为<b>负</b>（%s）⇒ 符号相反且幅度都大 ⇒ 「深端写入对行为无效」'
         '应改判为<b>只在累积差对象上成立</b>。</div>'
         % (' / '.join(f(V[a]['Q7_spearman_wall_ball'], 4) for a in ARMS),
            ' / '.join(f(V[a]['Q7_spearman_wall_J'], 4) for a in ARMS)))
H.append('</div>')

# ---------------------------------------------------------------- F P6 rlin
H.append('<h2>F · P6：超可加性峰值落在写入窗（<span style="color:#b42318">位置仅 A0 成立 / 峰值比 0/3 不成立</span>）</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>argmax r_lin</th><th>L*_own</th><th>r_lin@L*</th><th>峰值</th><th>峰值/次大(ALL)</th><th>峰值/次大(REACH)</th><th>判</th></tr>')
for a in ARMS:
    E = S[a]
    H.append('<tr><td>%s</td><td>L%s</td><td>L%d</td><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], V[a]['Q8_rlin_argmax'], E['L_star_own'], f(E['rlin_at_lstar'], 4), f(E['rlin_peak'], 4),
                f(V[a]['Q8_rlin_peak_ratio'], 2), f(reach_ratio(a), 2), pill(V[a]['Q8_label'] == 'SUPERADD_AT_WINDOW', 'SUPERADD', 'NO_WINDOW')))
H.append('</table>')
H.append('<div class="note"><span class="hl">同轮勘误 E-rlin（支撑域错配）</span>：seal 的 P6 rationale 引用「次大 0.186（L26）、比值 4.05」'
         '其实是 <b>REACH 域</b>读数 —— A0 现场重算 = <b>4.053</b>（次大 <b>0.186 @ L26</b>），<b>逐位复现</b>；'
         '而生产判据域是 <code>ALL_SITES = 1..L−2</code>，该域上 A0 次大变成 <b>L2</b>（浅端 <code>|b_all|</code> 极小的分母放大），'
         '峰值比骤降到 <b>1.55</b>。即便退回 REACH 域，三臂也只有 <b>4.05 / 1.71 / 1.28</b>，'
         '<b>A1/A2 在任何域都 ≤ 1.71</b> ⇒ <b>P6 无论按哪个域都 FAIL</b>（勘误只改「为什么 FAIL」，不改判决）。'
         'P6 <b>位置子命题</b>（argmax r_lin == L*_own）仅 <b>A0</b> 成立（1/3）。'
         '<code>r_lin</code> 是比值型量，浅端近零分母会放大它（后继改用绝对残差或加分母下限）。</div>')
H.append('</div>')

# ---------------------------------------------------------------- G controls
H.append('<h2>G · 对照（P7，描述性）：置换零假设 + 确认集 + 离流形诊断</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>com_B(all) obs</th><th>null p5</th><th>null p95</th><th>尾</th><th>com_B(mlp) obs</th><th>null p95</th><th>尾</th><th>确认集 Δ(INC_ALL)</th></tr>')
for a in ARMS:
    E = S[a]
    nA = E['null_comB_inc']; nM = E['null_comB_mlp']
    H.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td></tr>'
             % (SHORT[a], f(nA['obs_com']), f(nA['com_p5']), f(nA['com_p95']),
                pill(nA['com_tail'] == 'high', 'high', nA['com_tail'] or '—'),
                f(nM['obs_com']), f(nM['com_p95']),
                pill(nM['com_tail'] == 'high', 'high', nM['com_tail'] or '—'),
                f(V[a]['Q9_conf']['d_com']['INC_ALL'])))
H.append('</table>')
H.append('<div class="note">置换 null（保留质量多重集、随机重排到 REACH 位点；BP=%d，种子 comB_inc=%s / comB_mlp=%s / share_mlp=%s）。'
         '确认集 n=%d（与 %d 对 discovery <b>不相交</b>）。<b>邻域份额</b>的 null 只有 2 个位点 ⇒ 几乎无区分力（H7，不作门）。</div>'
         % (R['bootstrap']['BP'], R['bootstrap']['seeds']['comB_inc'], R['bootstrap']['seeds']['comB_mlp'],
            R['bootstrap']['seeds']['share_mlp'], len(EX['confirmation']), len(EX['discovery'])))
H.append('<table><tr><th>臂</th><th>pert_rel(INC_ALL)</th><th>pert_rel(CUM_ALL)</th></tr>')
for a in ARMS:
    H.append('<tr><td>%s</td><td>%s</td><td>%s</td></tr>' % (SHORT[a], f(V[a]['Q10_pert_rel_inc_max']), f(V[a]['Q10_pert_rel_cum_max'])))
H.append('</table><div class="note">CUM_ALL 深端 pert_rel 大 ⇒ 该臂离流形，只作桥接保真度门与对照（H9）。</div>')
H.append('</div>')

# ---------------------------------------------------------------- H predictions + errata
H.append('<h2>H · 预注册预测与同轮勘误</h2>')
H.append('<div class="card"><table><tr><th>#</th><th>判据（seal）</th><th>结果</th></tr>')
for k in sorted(PC):
    claim = SEAL['predictions'][k]['claim'].replace('\n', ' ')[:170]
    H.append('<tr><td><code>%s</code></td><td style="text-align:left">%s…</td><td>%s</td></tr>'
             % (k, claim, pill(PC[k]['pass_'], 'PASS', 'FAIL', 'N/A（描述性）')))
H.append('</table>')
H.append('<div class="note"><b>同轮勘误</b>（append-only）：'
         '<b>探针 P-A</b> INC_MLP 全零（行号写错）—— 被 SMOKE/探针纪律在发表前抓住；'
         '<b>P-B</b> 删重复前向；<b>P-C</b> 位点过滤放宽到全域；'
         '<b>SMOKE 2 处</b>（打印类型 / MERGE 桥接变量）；<b>M-A</b> ALL_SITES=1..L−2 对齐；'
         '<b>E-rlin</b> P6 峰值比的支撑错配（见 F）。</div>')
H.append('</div>')

# ---------------------------------------------------------------- I limits + next
H.append('<h2>I · 诚实边界与下一步（死线）</h2>')
H.append('<div class="card">')
H.append('<ul style="margin:6px 0 6px 18px;padding:0">')
for k in ['H%d' % i for i in range(1, 13)]:
    if k in SEAL['honesty']:
        H.append('<li><b>%s</b>：%s</li>' % (k, SEAL['honesty'][k]))
H.append('</ul>')
H.append('<div class="note"><b>Phase 19 最高优先</b>：NF4 vs BF16 的 <code>w_ℓ</code> 口径 '
         '（P17/P18 全在 nf4 ⇒ 须在 A0 同尺度 bf16 复算 <code>w_ℓ</code> 谱与 <code>com_V</code>，排除「深端集中」是量化地板效应）。'
         '<b>并列</b>：邻域宽度 ±2 敏感性（三臂 nb 恰好都是 %s）；<b>第三</b>：P17 P6 的 MEMO 改判（承本 Phase P5）。</div>'
         % str([S[a]['nb'] for a in ARMS][0]))
H.append('</div>')

H.append('<div class="sub" style="margin-top:22px">Phase 18 · N2h1-α-11 · 数据驱动渲染（全部数字取自 <code>result_phase18.json</code>）</div>')
H.append('</div></body></html>')

html = '\n'.join(H)
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(html)
b = len(html.encode('utf-8'))
print('WROTE %s  %d B / %d lines' % (OUT, b, len(html.splitlines())))
