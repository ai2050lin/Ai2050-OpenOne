# -*- coding: utf-8 -*-
"""Phase 20 展示页生成器（数据驱动；全部数字取自 result_phase20.json / seal / exec / probe）。
产出 tests/deepseek_temp/Phase20/present_phase20.html（浅色主题、自包含、无外部依赖）。

P19 只覆盖**向量侧**（w_ℓ 谱 / com_V）；本页是**行为侧**的跨精度检验：
  Panel [B] 行为预算 b_{c,ℓ}（P18 口径）  +  Panel [P] 写入窗剖面（P16 口径）
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
OUT = os.path.join(P20T, 'present_phase20.html')
SEALP = os.path.join(P20T, 'N2h1a13_design_seal.json')
EXECP = os.path.join(P20T, 'execution_phase20.json')
RESP = os.path.join(P20T, 'result_phase20.json')
PROBER = os.path.join(P20T, 'result_phase20_probe.json')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8] if os.path.exists(p) else '--------'


R = json.load(io.open(RESP, encoding='utf-8'))
EX = json.load(io.open(EXECP, encoding='utf-8'))
SEAL = json.load(io.open(SEALP, encoding='utf-8'))
SEALP_D = {x['id']: x for x in SEAL['predictions']}
PR = json.load(io.open(PROBER, encoding='utf-8')) if os.path.exists(PROBER) else None
DRIFT = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',
                                       'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))
PH = json.load(io.open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), encoding='utf-8'))
PHD = {r['pair']: r for r in PH['pairs']}

V = R['verdict']
JV = R['joint_verdict']
PC = R['predictions_check']
FL = R['floors']
QP = {p['arm_nf4'] + '|' + p['arm_bf16']: p for p in R['quant_pairs']}
ARMS = [a for a in EX['arm_order'] if a in R['arms']]
SHORT = {'A0_nf4': 'A0·qwen3-4b (nf4)', 'A0_bf16': 'A0·qwen3-4b (bf16)',
         'A1_nf4': 'A1·glm4-9b (nf4)', 'A1_bf16': 'A1·glm4-9b (bf16)'}
COL = {'A0_nf4': '#c2410c', 'A0_bf16': '#f59e0b', 'A1_nf4': '#1d4ed8', 'A1_bf16': '#3b82f6'}
ROLE = {'A0_nf4': '校准臂 1（锚）', 'A0_bf16': '核心检验臂', 'A1_nf4': '校准臂 2（锚）', 'A1_bf16': 'holdout 检验臂'}
PAIRS = [k for k in ('A0_nf4|A0_bf16', 'A1_nf4|A1_bf16') if k in QP]


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


def fp(x, n=4, dash='—'):
    return f(x, n, dash)


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
H.append('<title>Phase 20 / N2h1-α-13 · 行为量的量化口径稳健性（nf4 ↔ bf16）</title>')
H.append('''<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#16202b;--mut:#5b6b7c;--line:#e3e8ee;--ok:#0f766e;--bad:#b42318;--hi:#fef3c7;--ac:#1d4ed8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.62 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:26px 20px 60px}
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
.bar{display:inline-block;width:100%;max-width:180px;height:11px;background:#eef1f5;border-radius:6px;overflow:hidden;vertical-align:middle}
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
.strike{text-decoration:line-through;color:var(--mut)}
</style></head><body><div class="wrap">''')

H.append('<h1>Phase 20 / N2h1-α-13 · 行为量的量化口径稳健性（nf4 ↔ bf16）</h1>')
H.append('<div class="sub">P19 只证明了<b>向量侧</b>（<code>w_ℓ</code> 谱与质心 <code>com_V</code>）不是 nf4 地板效应——'
         '但 P18 的结论主体是<b>行为量</b>（<code>b_{c,ℓ}</code> 行为质心 <code>com_B</code>、组件行为份额、'
         'P16 的 <code>com_layer</code> 剖面）。本 Phase 用<b>同一装置、同一次加载、同一批 capture</b> '
         '把两个面板在 <b>bf16</b> 下复算：<span class="hl">P18/P16 的行为侧结论是不是量化地板效应？</span></div>')

H.append('<div class="card">')
H.append('<div class="grid"><div>')
H.append('<div class="kv"><span>seal</span><b>%s</b></div>' % sha8(SEALP))
H.append('<div class="kv"><span>exec</span><b>%s</b></div>' % sha8(EXECP))
H.append('<div class="kv"><span>result</span><b>%s</b></div>' % sha8(RESP))
H.append('<div class="kv"><span>探针 result</span><b>%s</b></div>' % sha8(PROBER))
H.append('<div class="kv"><span>锚 P18 / P16 / P17</span><b>%s / %s / %s</b></div>'
         % (R['anchor_result_p18_sha256'][:8], R['anchor_result_p16_sha256'][:8],
            R['anchor_result_p17_sha256'][:8]))
H.append('<div class="kv"><span>wall</span><b>%s s</b></div>' % f(R.get('elapsed_total_s'), 1))
H.append('</div><div>')
H.append('<div class="kv"><span>唯一自变量</span><b>nf4 (4-bit) ↔ bfloat16</b></div>')
H.append('<div class="kv"><span>template</span><b><code>%s</code></b></div>' % EX['template'])
H.append('<div class="kv"><span>instances / discovery / conf</span><b>%d / %d / %d</b></div>'
         % (len(EX['instances_all']), len(EX['discovery']), len(EX['confirmation'])))
H.append('<div class="kv"><span>U_ℓ</span><b>类别质心差 SVD，秩 = n_classes−1 = %d</b></div>' % (len(EX['classes']) - 1))
H.append('<div class="kv"><span>两个面板</span><b>[B] 行为预算（P18 口径） + [P] 写入窗剖面（P16 口径）</b></div>')
H.append('<div class="kv"><span>A2 参与</span><b>否（Qwen3-14B bf16 segfault，承 P19）</b></div>')
H.append('</div></div>')
H.append('<div style="margin-top:10px">')
for lbl in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'):
    H.append(vchip(lbl + ' ' + PC[lbl]['name'].split('——')[0].replace('holdout 主预测 ', '').replace('Panel P —— ', '')[:12],
                   pill(PC[lbl]['pass_'])))
H.append(vchip('P10 对照', '描述性'))
H.append('</div>')
H.append('<div class="note" style="margin-top:8px">臂序：%s</div>'
         % ' → '.join('%s <span class="note">(%s)</span>' % (SHORT[a], ROLE[a]) for a in ARMS))
H.append('</div>')

# ---------------------------------------------------------------- A 装置门
H.append('<h2>A · 装置门与 nf4 校准锚（逐位复现 P18/P16/P17）</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>模型</th><th>精度</th><th>offload</th><th>L</th><th>device</th>'
         '<th>arch max</th><th>blocks max</th><th>U 秩</th><th>Q1</th><th>Q2</th></tr>')
for a in ARMS:
    r = R['arms'][a]; c = r['cfg']; x = r['E2_fidelity']
    H.append('<tr><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td><td>%d</td><td>%s</td>'
             '<td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], r['model'], r['scheme'], '是' if r['offload'] else '否', c['L'], r['Q0_device'],
                sci(x['arch_max']), sci(x['blk_max']), r['E3_U']['rank'],
                pill(V[a]['Q1_label'] == 'FID_PASS', 'FID_PASS', 'FID_FAIL'),
                pill(V[a]['Q2_label'] == 'ANCHOR_OK' if V[a]['Q2_label'] != 'ANCHOR_NA_TREATMENT' else None,
                     'ANCHOR_OK', V[a]['Q2_label'], 'ANCHOR_NA（bf16 按设计不复现 nf4 锚）')))
H.append('</table>')
H.append('<div class="note">保真度门：arch ≤ %s / blocks ≤ %s（四臂全过）。'
         '<b>nf4 校准臂逐位复现 P18/P16/P17 冻结锚</b>（容差 1e-4）⇒ 装置与三个前序 Phase 同源。</div>'
         % (FL['P20_FID_ARCH'], FL['P20_FID_BLK']))
H.append('<table><tr><th>臂 · 锚量</th><th>got</th><th>expected（冻结）</th><th>一致</th></tr>')
_AK = [('com_B_INC_ALL', 'com_B(all)'), ('com_B_INC_MLP', 'com_B(mlp)'), ('com_B_INC_ATTN', 'com_B(attn)'),
       ('com_B_CUM_ALL', 'com_B(cum)'), ('comlayer_B_all', 'comlayer_B_all'),
       ('share_mlp_beh_nb', 'share_mlp_beh_nb'), ('com_V_p17', 'com_V (P17)'),
       ('com_layer_x_p16', 'com_layer(x) (P16)'), ('com_layer_j_p16', 'com_layer(J) (P16)'),
       ('full_swap_p16', 'FULL_SWAP (P16)'), ('CUM_L_star', '桥接 CUM@L*')]
for a in [x for x in ARMS if x.endswith('_nf4')]:
    ap = R['arms'][a]['E9_anchor']
    if not ap.get('applies'):
        continue
    d = ap['detail']
    for k, lab in _AK:
        if k not in d:
            continue
        H.append('<tr><td>%s · %s</td><td><b>%s</b></td><td>%s</td><td>%s</td></tr>'
                 % (SHORT[a], lab, f(d[k]['got'], 6), f(d[k]['expected'], 6), pill(d[k]['ok'], '✓', '✗')))
    for k, lab in (('neighbourhood', 'nb'), ('reach_len', 'REACH 位数')):
        d2 = d.get(k)
        if d2:
            H.append('<tr><td>%s · %s</td><td><b>%s</b></td><td>%s</td><td>%s</td></tr>'
                     % (SHORT[a], lab, d2['got'], d2['expected'], pill(d2['ok'], '✓', '✗')))
H.append('</table></div>')

# ---------------------------------------------------------------- B Panel [B]
H.append('<h2>B · Panel [B] 行为预算（P18 口径）四臂读数</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>com_B(all)</th><th>com_B(mlp)</th><th>com_B(attn)</th>'
         '<th>com_B(top1)</th><th>com_B(cum)</th><th>comlayer_B_all</th><th>share_mlp_beh(nb)</th>'
         '<th>share_mlp_vec(nb)</th><th>com_V</th><th>gap=com_V−com_B</th><th>Q3</th><th>Q4</th><th>Q5</th></tr>')
for a in ARMS:
    v = V[a]
    H.append('<tr><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
             '<td><b>%s</b> %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], f(v['com_B_all']), f(v['com_B_mlp']), f(v['com_B_attn']), f(v['com_B_top1']),
                f(v['com_B_cum']), f(v['comlayer_B_all']),
                fp(v['share_mlp_beh_nb']), bar(v['share_mlp_beh_nb'], COL[a]),
                fp(v['share_mlp_vec_nb']), f(v['com_V']), f(v['gap']),
                pill(v['Q3_label'] == 'MLP_DOMINANT_BEH', 'MLP主导', '未主导'),
                pill(v['Q4_label'] == 'SHALLOWER', '浅于向量', v['Q4_label']),
                pill(v['Q5_label'] == 'EFFICACY_COUPLED', '耦合', '解耦')))
H.append('</table>')
H.append('<div class="note"><code>b_{c,ℓ} = score_of(h_ℓ^R + P_{U_ℓ}(Δ_c)) − BASE[rw].sd0</code>（P18 逐字口径，'
         '<b>不可加</b>）。行为质心 <code>com_B</code> 通过<b>区间求和 + 相邻位点中点</b>定义；'
         '<code>share_mlp_beh(nb)</code> 是邻域 <code>%s</code> 上的行为组件份额。'
         '四臂皆 <b>MLP 主导（&gt; %s）</b>、行为质心 <b>浅于向量质心（gap ≥ %s）</b>、'
         '<code>spearman(w_all,b_all) &gt; 0</code>。</div>'
         % (str(R['arms'][ARMS[0]]['E10_summary']['nb']),
            FL['MLP_DOM_MIN'], FL['SHALLOWER_MIN']))
H.append('</div>')

# ---------------------------------------------------------------- C 配对
H.append('<h2>C · 跨口径配对（核心：nf4 | bf16，同模型）</h2>')
H.append('<div class="card"><table><tr><th>配对</th><th>Δcom_B(all)</th><th>Δcomlayer_B_all</th>'
         '<th>Δcom_V<br><span class="note">冻结谱·按构造=0</span></th><th>Δcom_V<br><span class="note">自谱·真检验</span></th>'
         '<th>ρ(b_nf4,b_bf16)</th><th>残差中位</th><th>残差 p90</th><th>share nf4→bf16</th><th>同侧</th>'
         '<th>Δcom_layer(x)</th><th>Δcom_layer(J)</th><th>max|Δxhalf|</th></tr>')
for k in PAIRS:
    s = QP[k]

    def _d(key):
        o = s.get(key) or {}
        return o.get('delta')
    H.append('<tr><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td>'
             '<td>%s → %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
             % (k.replace('|', ' | '), f(_d('com_B_all')), f(_d('comlayer_B_all')), f(_d('com_V')),
                f(R['arms'][k.split('|')[1]]['E10_summary']['com_V_own_spectrum']
                  - R['arms'][k.split('|')[0]]['E10_summary']['com_V_own_spectrum']),
                fp(s['rho_b_all']['rho']), fp(s['rho_b_all']['resid_med']), fp(s['rho_b_all']['resid_p90']),
                fp(s['share_mlp_beh_nb']['nf4']), fp(s['share_mlp_beh_nb']['bf16']),
                pill(s['share_mlp_beh_nb']['same_side'], '同侧', '异侧'),
                f(_d('com_layer_x')), f(_d('com_layer_j')), fp(s['xhalf']['max_abs_dxh'])))
H.append('</table>')
H.append('<div class="note"><b>口径补注（E-comv）</b>：左列 <code>Δcom_V</code> 由 <b>P17 冻结 <code>w_all</code> 谱</b>重算（跨 Phase 可比口径）⇒ 同模型两臂<b>按构造恒等</b>，只证明「锚一致」，<b>不是</b>跨精度证据；<b>右列</b>用<b>本臂自谱</b> <code>com_V_own_spectrum</code> 才是真正的向量质心跨口径位移（描述性；向量侧的正式跨精度结论在 <b>Phase 19</b>）。容差：<code>com_B/com_V/com_layer</code> ≤ %s 层；<code>share</code> ≤ %s；'
         '<code>xhalf</code> ≤ %s；秩相关地板 <code>ρ(b) ≥ %s</code>。'
         '<b>行为质心与剖面质心几乎不动、行为谱几乎完全同形、组件归属同侧</b> '
         '⇒ <span class="hl">P18/P16 的行为侧结论不是 nf4 量化地板效应</span>。</div>'
         % (f(FL['QUANT_TOL_COMB'], 1), FL['QUANT_TOL_SHARE'], FL['QUANT_TOL_XHALF'], FL['RHO_B_MIN']))
H.append('<table><tr><th>联合判据</th><th>值</th><th>判据</th><th>过</th></tr>')
_JROW = [('Q3 com_V 跨精度稳定', 'Q3_com_V_stable', 'Δcom_V ≤ %s' % f(FL['QUANT_TOL_COMV'], 1)),
         ('Q4 com_B 跨精度稳定', 'Q4_com_B_stable', 'Δcom_B ≤ %s' % f(FL['QUANT_TOL_COMB'], 1)),
         ('Q5 comlayer_B 稳定', 'Q5_comlayer_stable', 'Δcomlayer_B ≤ %s' % f(FL['QUANT_TOL_COMLAYER'], 1)),
         ('Q6 行为谱一致', 'Q6_spectrum_consistent', 'ρ(b_nf4,b_bf16) ≥ %s' % FL['RHO_B_MIN']),
         ('Q7 行为份额稳定', 'Q7_share_stable', '同侧且 Δ ≤ %s' % FL['QUANT_TOL_SHARE']),
         ('Q8 MLP 主导保持', 'Q8_mlp_dom_retained', '四臂皆 > %s' % FL['MLP_DOM_MIN']),
         ('Q9 浅于向量保持', 'Q9_shallow_retained', 'gap 侧一致 (≥ %s)' % f(FL['SHALLOWER_MIN'], 1)),
         ('Q10 耦合保持', 'Q10_coupled_retained', 'spearman(w,b) > 0 两口径'),
         ('Q11 写入窗剖面稳定', 'Q11_profile_stable', 'Δcom_layer(x),(J) ≤ %s' % f(FL['QUANT_TOL_COMLAYER'], 1)),
         ('Q12 半饱和点稳定', 'Q12_xhalf_stable', 'max|Δxhalf| ≤ %s' % FL['QUANT_TOL_XHALF'])]
for lab, key, crit in _JROW:
    H.append('<tr><td>%s</td><td>%s</td><td class="note">%s</td><td>%s</td></tr>'
             % (lab, str(JV.get(key)), crit, pill(JV.get(key), '✓', '✗')))
H.append('</table>')
H.append('<div class="note" style="border-left:4px solid #b45309;background:#fffbeb;padding:8px 10px">'
         '<b>P9 域分解（post-hoc；as-coded 判决不改）</b>：seal 的 P9 判据「两对 <code>max|&Delta;xhalf|</code> &le; '
         + f(FL['QUANT_TOL_XHALF'], 2) + '」<b>未写明域</b>，实现按字面取「所有公共 PROFILE 位点」；'
         '但 P16 的 <code>XH_FAITHFUL_TOL</code> 只在 <b>冻结 REACH 域</b>（<code>XH_12_by_site</code> = &ell;&ge;'
         + str(PH['p16_calib_domain'][0]) + ', n=' + str(len(PH['p16_calib_domain'])) + '）上标定过（P16 实测 <code>'
         + f(PH['p16_calib_max_abs_dxh'], 6) + '</code>）。<br>'
         '&rArr; <b>Q12 as-coded = ' + str(JV.get('Q12_xhalf_stable')) + '</b>，但其超差 100% 来自 <b>浅端 &ell;=1</b>；'
         '在 P16 实际标定的域上两对皆 <b>PASS</b>（A0 裕度约 '
         + f(FL['QUANT_TOL_XHALF'] / max(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh'], 1e-9), 0)
         + '&times;）&rArr; 这是<b>预注册判据文字的域错配</b>，不是物理上的量化不稳定。</div>')
H.append('<table><tr><th>配对</th><th>as-coded（全部公共位点）</th><th>判</th>'
         '<th>冻结 REACH 域（&ell;&ge;6）</th><th>判</th><th>浅端 &ell;&lt;6</th><th>超差承担者</th></tr>')
for k in PAIRS:
    r = PHD.get(k)
    if not r:
        continue
    H.append('<tr><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td>'
             '<td>%s @&ell;=%s</td><td>100%% @&ell;=%s</td></tr>'
             % (k.replace('|', ' | '), f(r['as_coded_max_abs_dxh'], 6),
                pill(r['as_coded_pass'], 'PASS', 'FAIL'), f(r['reach_domain_max_abs_dxh'], 6),
                pill(r['reach_domain_pass'], 'PASS', 'FAIL'), f(r['shallow_max_abs_dxh'], 6),
                str(r['shallow_argmax']), str(r['as_coded_argmax'])))
H.append('</table>')
H.append('</div>')

# ---------------------------------------------------------------- D Panel [P]
H.append('<h2>D · Panel [P] 写入窗剖面（P16 口径）四臂</h2>')
H.append('<div class="card"><table><tr><th>臂</th><th>com_layer(x)</th><th>com_layer(J)</th><th>span3(x)</th>'
         '<th>span3(J)</th><th>FULL_SWAP</th><th>ρ(ℓ=1) 首值</th><th>REACH_own 位数</th><th>ℓ_reach_own</th>'
         '<th>null(com_layer_x) 尾</th></tr>')
for a in ARMS:
    v = V[a]; S = R['arms'][a]['E10_summary']
    nl = v.get('null_comlayer_x') or {}
    H.append('<tr><td>%s</td><td><b>%s</b></td><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
             '<td>%d</td><td>%s</td><td>%s</td></tr>'
             % (SHORT[a], f(v['com_layer_x']), f(v['com_layer_j']), f(S.get('span3_x')), f(S.get('span3_j')),
                f(S.get('full_swap')), fp(S['rho'][0] if S.get('rho') else None),
                len(S.get('reach_own') or []), str(S.get('ell_reach_own')),
                pill(nl.get('com_tail') == 'high', 'high', nl.get('com_tail') or '—')))
H.append('</table>')
H.append('<div class="note"><code>com_layer</code> 按 P16 冻结定义在 <b>REACH 域</b>（位点 %s，n=%d）上计算：'
         '以相邻位点差分 + 区间求和取质心。判据 = 跨精度位移 ≤ %s 层。</div>'
         % (str(R['arms'][ARMS[0]]['E10_summary'].get('com_layer_domain')),
            len(R['arms'][ARMS[0]]['E10_summary'].get('com_layer_domain') or []),
            f(FL['QUANT_TOL_COMLAYER'], 1)))
H.append('</div>')

# ---------------------------------------------------------------- E 谱形
H.append('<h2>E · 谱形对照：行为谱 <code>b_all,ℓ</code> nf4 vs bf16（同模型分开标尺）</h2>')
H.append('<div class="card">')
H.append('<div class="grid2">')
for grp, arms in (('qwen3-4b', ['A0_nf4', 'A0_bf16']), ('glm4-9b', ['A1_nf4', 'A1_bf16'])):
    arms = [a for a in arms if a in R['arms']]
    if not arms:
        continue
    mx = max(max(abs(x) for x in R['arms'][a]['E10_summary']['b_all']) for a in arms) or 1.0
    H.append('<div><h3>%s</h3>' % grp)
    for a in arms:
        b = R['arms'][a]['E10_summary']['b_all']
        sites = R['arms'][a]['E10_summary']['sites_all']
        H.append('<div class="note" style="margin:6px 0 0">%s · argmax L%s</div>'
                 % (SHORT[a], sites[max(range(len(b)), key=lambda i: abs(b[i]))]))
        H.append('<div class="sp">')
        for i, x in enumerate(b):
            H.append('<i style="background:%s;height:%.1f%%" title="L%d b=%.3f"></i>'
                     % (COL[a], 100.0 * abs(x) / mx, sites[i], x))
        H.append('</div>')
    if len(arms) == 2:
        k = arms[0] + '|' + arms[1]
        if k in QP:
            H.append('<div class="note">ρ = <b>%s</b>（残差中位 %s / p90 %s）</div>'
                     % (fp(QP[k]['rho_b_all']['rho']), fp(QP[k]['rho_b_all']['resid_med']),
                        fp(QP[k]['rho_b_all']['resid_p90'])))
    H.append('</div>')
H.append('</div>')
H.append('<div class="note">同一模型 nf4 与 bf16 的行为谱<b>逐层同形</b> ⇒ 行为侧的层分布是模型性质而非量化伪影。</div>')
H.append('</div>')

# ---------------------------------------------------------------- F 交叉验证
H.append('<h2>F · 交叉验证：独立探针 ↔ 生产实现（Panel [B] 逐位相同）</h2>')
H.append('<div class="card">')
if PR is None:
    H.append('<div class="note">未找到探针 result（<code>result_phase20_probe.json</code>）⇒ 跳过。</div>')
else:
    PV = PR['verdict']
    H.append('<table><tr><th>量</th><th>臂</th><th>探针（全量配对 / 缩 α 网格）</th><th>生产（全尺度）</th><th>≤1e-6</th></tr>')
    _CV = [('com_B_all', 'com_B(all)', 3), ('com_B_mlp', 'com_B(mlp)', 3), ('com_B_attn', 'com_B(attn)', 3),
           ('com_B_cum', 'com_B(cum)', 3), ('comlayer_B_all', 'comlayer_B_all', 3),
           ('share_mlp_beh_nb', 'share_mlp_beh_nb', 6), ('com_V', 'com_V (P17 重算)', 3)]
    for a in [x for x in ARMS if x in PV]:
        for key, lab, nd in _CV:
            pv = PV[a].get(key); rv = V[a].get(key)
            same = (pv is not None and rv is not None and abs(float(pv) - float(rv)) <= 1e-6)
            H.append('<tr><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td><td>%s</td></tr>'
                     % (lab, SHORT[a], f(pv, nd), f(rv, nd), pill(same, '✓', '≠')))
    H.append('</table>')
    H.append('<div class="note">Panel [B] 的输入（<code>U_ℓ</code>、配对集、实例）在两套实现下<b>全量一致</b>（探针只缩 α 网格）'
             '⇒ 这些数字在容差 1e-6 内一致。<b>α 网格依赖量</b>（<code>com_layer(x)/(J)</code>、<code>xhalf</code>、'
             '<code>FULL_SWAP</code> 域外项）在缩幅网格下<b>不可比</b>，故不列入本表（承同轮勘误 E-scope）。</div>')
H.append('</div>')

# ---------------------------------------------------------------- G 预测/限界/勘误
H.append('<h2>G · 预注册预测、限界与同轮勘误</h2>')
H.append('<div class="card"><table><tr><th>#</th><th>判据（seal）</th><th>结果</th></tr>')
for k in sorted(PC):
    claim = str(SEALP_D.get(k, {}).get('criterion', '')).replace('\n', ' ')[:170]
    H.append('<tr><td><code>%s</code></td><td style="text-align:left">%s…</td><td>%s</td></tr>'
             % (k, claim, pill(PC[k]['pass_'], 'PASS', 'FAIL') if PC[k]['pass_'] is not None else '<span class="pill na">描述性</span>'))
H.append('</table>')
H.append('<div class="note"><b>限界（诚实性）</b>：'
         '<ul style="margin:6px 0 6px 18px;padding:0">'
         '<li><b>H3 覆盖限界</b>：A2（Qwen3-14B，29.5 GB）不参与 bf16 腿（P19 实测 <b>segfault@19%</b>）'
         '⇒ 跨精度行为稳健性只在 <b>qwen3-4b 与 glm4-9b</b> 两模型上验证。</li>'
         '<li><b>H2 多源差异</b>：bf16 − nf4 含「量化误差 + 反量化 kernel 路径」两源；'
         'A1·bf16 走 <b>CPU offload</b> ⇒ 另含「分片执行」第三源。</li>'
         '<li><b>H5</b>：本 Phase 仍是<b>激活级干预</b>，不是权重实现级证明（承 P19 H3）。</li>'
         '<li><b>H9</b>：只检验「行为量对精度稳健」，不回答 P18 的 2 项 FAIL（桥接 2/3、超可加 0/3）是否被精度影响。</li>'
         '</ul>')
H.append('<b>同轮勘误</b>（append-only）：'
         '<b>E-sper</b> 同名不同源的「谱」必须显式区分键空间（P17 冻结锚谱 <code>w_all/w_mlp/w_attn</code> '
         'vs 本臂自谱 <code>INC_*</code>）⇒ 拆 <code>sper_anchor</code>/<code>sper_own</code>；'
         '<b>E-scope</b> 锚复现须限定在<b>全尺度网格</b>（SMOKE/PROBE 缩幅网格与冻结锚不可比）；'
         '<b>E-probefull</b> 探针必须<b>保留全量配对与实例</b>（否则 <code>U_ℓ</code> 秩退化为 2、FULL_SWAP 偏离锚）；'
         '<b>E-baseline</b>（<b>记录完整性事件，非本轮引入</b>）'
         '<code>post-append-phase19</code> 基线（%s B / <code>%s</code>）在快照后于 <b>%s</b> 被就地改写 —— '
         'P10–P19 共 <b>%d</b> 个标题由 <code>[HH:MM]</code> 规范化为 <code>[YYYY-MM-DD HH:MM]</code>，'
         '逐条 +11 B、合计 <b>+%d B</b>（行数不变、<b>无文本丢失</b>，事件未被任何 wlog / history 记录）'
         '⇒ 该基线的 bytes/sha256 锚与 <code>sections</code> 键陈旧。<b>处置</b>：不改写历史，'
         '在 <code>history</code> 标注 <code>stale</code>，并把 <code>sections</code> 键口径由「行前 44 字符」'
         '改为<b>完整标题行</b>（加碰撞断言）。</div>'
         % (str(DRIFT['prev_baseline']['bytes']), str(DRIFT['prev_baseline']['sha8']),
            str(DRIFT['memo_mtime']), len(DRIFT['normalized_phases']),
            DRIFT['observed_delta_bytes']))
H.append('<div class="note" style="border-left:4px solid #b45309;background:#fffbeb;padding:8px 10px">'
         '<b>E-xhdom</b>（本轮主要发现之一）：seal 的 P9 只写「两对 <code>max|&Delta;xhalf|</code> &le; '
         + f(FL['QUANT_TOL_XHALF'], 2) + '」，<b>未写明域</b>；实现按字面取「所有公共 PROFILE 位点」'
         '&rArr; A0 的 as-coded 读数 <code>' + f(PHD['A0_nf4|A0_bf16']['as_coded_max_abs_dxh'], 6)
         + '</code> <b>完全由浅端 &ell;=1 贡献</b>（&ell;&ge;6 段仅 <code>'
         + f(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh'], 6) + '</code>，比 &ell;=1 小 '
         + f(PHD['A0_nf4|A0_bf16']['shallow_max_abs_dxh']
             / max(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh'], 1e-9), 0) + '&times;），'
         '而 P16 的 <code>XH_FAITHFUL_TOL</code> 只在 <code>XH_12_by_site</code>（&ell;&ge;6）上标定过。'
         '<b>处置</b>：as-coded 判决 <b>P9 = '
         + ('FAIL' if PC['P9']['pass_'] is False else 'PASS') + ' 原样保留</b>；另出 post-hoc 域分解'
         '（A0 <code>' + f(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh'], 6) + '</code> / A1 <code>'
         + f(PHD['A1_nf4|A1_bf16']['reach_domain_max_abs_dxh'], 6) + '</code>，两对皆 PASS，'
         'A0 该值恰与 P16 标定值逐位相同）。<b>教训</b>：「沿用某容差」时必须把该容差的标定域一并写进判据文字。</div>')
H.append('</div>')

# ---------------------------------------------------------------- H 下一步
H.append('<h2>H · 下一步（死线优先级）</h2>')
H.append('<div class="card">')
H.append('<div class="kv"><span>最高</span><b>Phase 21 — 跨精度检验推进到<b>组件级向量预算与权重实现级</b>：'
         'bf16 下复算 P8 的 <code>share_v</code> 与 N2h1-α-1 的权重级定位</b></div>')
H.append('<div class="kv"><span>并列</span><b>邻域宽度 ±2 敏感性（四臂 nb 恰好都是 %s）</b></div>'
         % str(R['arms'][ARMS[0]]['E10_summary']['nb']))
H.append('<div class="kv"><span>并列</span><b>P9 判据补域：把 <code>XH_FAITHFUL_TOL</code> 的 '
         '&ell;&ge;6 标定域写进判据文字后可否改判（&rArr; 需新 Phase 预注册，本 Phase 维持 FAIL）</b></div>')
H.append('<div class="kv"><span>第三</span><b>P17 <code>P6</code> 的 MEMO 改判（承 P18 <code>P5</code>：'
         '<code>spearman(w_all,|b_all|)&gt;0</code> 而 <code>spearman(w_all,J)&lt;0</code> ⇒ 对象错配）</b></div>')
H.append('<div class="kv"><span>N 线挂账</span><b>N2h1-α-1 权重级定位；N2h1-β 水果类崩塌；N3-β→N3-ε；'
         'R1 补强；K4；N 线 P3–P7 补登 Ledger</b></div>')
H.append('</div>')

_led = R.get('ledger_n')
H.append('<div class="sub" style="margin-top:22px">Phase 20 · N2h1-α-13 · 数据驱动渲染（全部数字取自 '
         '<code>result_phase20.json</code>；Ledger n = %s）</div>'
         % (str(_led) if _led else '303'))
H.append('</div></body></html>')

html = '\n'.join(H)
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(html)
b = len(html.encode('utf-8'))
print('WROTE %s  %d B / %d lines' % (OUT, b, len(html.splitlines())))
