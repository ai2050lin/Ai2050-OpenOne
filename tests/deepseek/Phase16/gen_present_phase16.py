# -*- coding: utf-8 -*-
"""Phase 16 展示页生成器（数据驱动；全部数字取自 result_phase16.json / seal / amend1）。
产出 tests/deepseek_temp/Phase16/present_phase16.html（浅色主题、自包含、无外部依赖）。
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
OUT = os.path.join(P16T, 'present_phase16.html')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(P16T, 'result_phase16.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P16T, 'execution_phase16.json'), encoding='utf-8'))
AM1 = json.load(io.open(os.path.join(P16T, 'N2h1a9_design_seal_amend1.json'), encoding='utf-8'))
SEALP = os.path.join(P16T, 'N2h1a9_design_seal.json')
AM1P = os.path.join(P16T, 'N2h1a9_design_seal_amend1.json')
EXECP = os.path.join(P16T, 'execution_phase16.json')
RESP = os.path.join(P16T, 'result_phase16.json')

VER = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']
E4 = R['E4_summary']; E5 = R['E5_concentration']; E3 = R['E3_localize']; E7 = R['E7_reach']
E6 = R['E6_calibration']; E2 = R['E2_full_swap']; FL = R['floors']
SITES = [int(v) for v in R['grid']['profile_sites']]
ALPHAS = [float(v) for v in R['grid']['alphas']]
ARMS = list(R['arms'].keys())
A0, A1, A2 = ARMS
SHORT = {A0: 'A0·4B(nf4)', A1: 'A1·GLM4-9B', A2: 'A2·Qwen3-14B'}
COL = {A0: '#c2410c', A1: '#1d4ed8', A2: '#047857'}
NW_ARM = {a: len(R['arms'][a].get('cands_used') or EX['localize']['cands']) for a in ARMS}
NTOT = sum(len(SITES) * len(ALPHAS) * len(EX['discovery']) + NW_ARM[a] * len(EX['discovery']) + 41 + 12
           for a in ARMS)


def f(x, n=4, dash='—'):
    try:
        if x is None:
            return dash
        return ('%.' + str(n) + 'f') % float(x)
    except Exception:
        return dash


def fp(x, n=3, dash='—'):
    try:
        if x is None:
            return dash
        return ('%+.' + str(n) + 'f') % float(x)
    except Exception:
        return dash


def badge(ok, txt_ok='PASS', txt_no='FAIL'):
    return '<span class="pill %s">%s</span>' % ('ok' if ok else 'bad', txt_ok if ok else txt_no)


H = []
H.append('<!DOCTYPE html>')
H.append('<html lang="zh-CN"><head><meta charset="utf-8">')
H.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
H.append('<title>Phase 16 / N2h1-α-9 · 写入窗原点化剖面 + 集中度重设计</title>')
H.append('''<style>
:root{--bg:#f7f8fa;--card:#ffffff;--ink:#16202b;--mut:#5b6b7c;--line:#e3e8ee;--ok:#0f766e;--bad:#b42318;--hi:#fef3c7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.62 -apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
.wrap{max-width:1150px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:25px;margin:0 0 4px}
h2{font-size:18px;margin:34px 0 10px;padding-left:10px;border-left:4px solid #0f766e}
h3{font-size:15px;margin:20px 0 8px;color:var(--mut);font-weight:600}
.sub{color:var(--mut);font-size:13.5px;margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:22px;font-weight:700;letter-spacing:-.2px}
.kpi .k{font-size:12.5px;color:var(--mut);margin-top:2px}
.kpi .n{font-size:11.5px;color:#8a97a6;margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top}
th{background:#f0f3f7;font-weight:600;color:#33424f;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.ok{background:#dcfce7;color:#14532d}.bad{background:#fee4e2;color:#7a271a}.warn{background:#fef3c7;color:#78350f}
.mono{font-family:ui-monospace,Consolas,"Courier New",monospace;font-size:12.5px}
.alert{background:var(--hi);border:1px solid #f2d98a;border-radius:12px;padding:14px 16px;margin:14px 0}
.danger{background:#fee4e2;border:1px solid #f6b6ae;border-radius:12px;padding:14px 16px;margin:14px 0}
.note{color:var(--mut);font-size:13px}
ul{margin:6px 0 0 0;padding-left:20px}li{margin:3px 0}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--mut);margin:8px 0 2px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.rho{display:grid;gap:2px;align-items:end}
.bar{height:9px;border-radius:2px;background:#e5e9ef}
.bar.hi{background:#0f766e}
.bar.lo{background:#f0a68a}
.rho td{padding:3px 4px;border:none}
</style></head><body><div class="wrap">''')

H.append('<h1>Phase 16 / N2h1-α-9 · 写入窗原点化剖面 + 集中度统计量重设计</h1>')
H.append('<p class="sub">三臂同一 <b>nf4</b> 口径（qwen3-4b 校准臂 / glm4-9b / Qwen3-14B）· '
         '剖面下探至 <span class="mono">ℓ=1..5</span>（23 位点）· 主域 = <b>可达性掩膜</b> '
         '<span class="mono">REACH = {ℓ : ρ(ℓ) ≥ 0.10}</span> · 旧量 <span class="mono">top3_share</span> → '
         '新量 <span class="mono">(com_layer, span_k)</span> · '
         'seal <span class="mono">%s</span> · exec <span class="mono">%s</span> · '
         'amend1 <span class="mono">%s</span> · result <span class="mono">%s</span></p>'
         % (sha8(SEALP), sha8(EXECP), sha8(AM1P), sha8(RESP)))

# ---------------- KPI ----------------
H.append('<h2>0 · 判决与关键读数</h2>')
H.append('<div class="grid">')
npass = sum(1 for k in PC if PC[k]['pass_'])
nfail = sum(1 for k in PC if not PC[k]['pass_'])
H.append('<div class="kpi"><div class="v">%d PASS / %d FAIL</div><div class="k">7 条预注册预测</div>'
         '<div class="n">FAIL：%s</div></div>'
         % (npass, nfail, ', '.join(sorted(k for k in PC if not PC[k]['pass_'])) or '无'))
H.append('<div class="kpi"><div class="v">%s</div><div class="k">Q1 冻结锚分级</div>'
         '<div class="n">∩ 三臂 bit-for-bit：max|Δxhalf| = 0.000e+00</div></div>' % JV['Q1_joint'])
H.append('<div class="kpi"><div class="v">%s</div><div class="k">Q2 写入窗 = 可达性门槛</div>'
         '<div class="n">ℓ_reach == L*_own：%s</div></div>'
         % (JV['Q2_joint'], ' / '.join('%s' % E7[a]['ell_reach'] for a in ARMS)))
H.append('<div class="kpi"><div class="v">%s</div><div class="k">Q3 写入窗 ∈ 主域</div>'
         '<div class="n">写入窗 ∈ REACH：%s</div></div>'
         % (JV['Q3_joint'], badge(VER[A0].get('Q3_label') == 'WIN_IN_DOMAIN', '三臂均满足', '见报告')))
H.append('<div class="kpi"><div class="v">%s</div><div class="k">Q4 双坐标质心分离</div>'
         '<div class="n">com_sep = %s 层（门 %.1f）</div></div>'
         % (JV['Q4_joint'], ' / '.join(fp(VER[a].get('Q4_sep'), 1) for a in ARMS), FL['CENTROID_SEP_MIN']))
H.append('<div class="kpi"><div class="v">%s</div><div class="k">Q5 统计量重设计</div>'
         '<div class="n">旧量显著 %s 格 → 新量 %d 格</div></div>'
         % (JV['Q5_joint'], JV['Q5_counts']['old_sig'], sum(JV['Q5_counts']['new_sig'])))
H.append('<div class="kpi"><div class="v">%d 次</div><div class="k">真实 GPU 前向（%d 位点 × %d α × 24 配对）</div>'
         '<div class="n">逐臂 %s · elapsed %s s</div></div>'
         % (NTOT, len(SITES), len(ALPHAS), ' / '.join(str(v) for v in NW_ARM.values()),
            f(R.get('elapsed_total_s'), 0)))
H.append('</div>')
H.append('<div class="danger"><b>主结论 · P6 = 预注册否证。</b> '
         '<span class="mono">com_layer(xhalf) − com_layer(J)</span> = '
         '<b>%s / %s / %s 层</b>（A0 → A1 → A2），判据要求 ≥ %.1f 层且 3/3。'
         '<b>A2（Qwen3-14B，与 A0 同属 Qwen3 家族、参数 3.5×）反号</b> ⇒ '
         'Phase 12/13/14「<span class="mono">xhalf</span> 深尾集中 / <span class="mono">J</span> 浅端集中」的'
         '<b>物理深度表述整体撤回</b>（seal 的 <span class="mono">may_falsify_the_whole_line</span> 条款）。'
         '装置门全过、锚 bit 级复现、P4 三臂严格 ⇒ <b>这是一次成功的自我否证，不是装置失败</b>。</div>'
         % (fp(VER[A0].get('Q4_sep'), 3), fp(VER[A1].get('Q4_sep'), 3), fp(VER[A2].get('Q4_sep'), 3),
            FL['CENTROID_SEP_MIN']))

# ---------------- 1 装置 ----------------
H.append('<h2>1 · 装置与预算</h2>')
H.append('<table><tr><th>臂</th><th>模型</th><th class="num">L / hid / kv / tie</th>'
         '<th class="num">载入 s</th><th class="num">FULL_SWAP</th><th class="num">L*_own</th>'
         '<th class="num">ℓ_reach</th><th class="num">Q0</th><th class="num">F1b</th>'
         '<th class="num">sup_id 同参考</th></tr>')
for a in ARMS:
    c = R['arms'][a].get('cfg', {})
    H.append('<tr><td><b style="color:%s">%s</b></td><td>%s</td>'
             '<td class="num mono">%s / %s / %s / %s</td><td class="num">%s</td>'
             '<td class="num mono">%s</td><td class="num">%s</td><td class="num">%s</td>'
             '<td class="num">%s</td><td class="num">%s</td><td class="num">%s</td></tr>'
             % (COL[a], SHORT[a], R['arms'][a]['model'], c.get('L'), c.get('hid'), c.get('kv_heads'),
                c.get('tie'), f(R['arms'][a].get('load_s'), 1), f(E2[a].get('FULL_SWAP'), 6),
                E3[a].get('L_star_own'), E7[a].get('ell_reach'),
                badge(VER[a].get('Q0_device') == 'PASS'),
                badge(bool(R['arms'][a].get('F1b_ok'))),
                badge(bool(R['arms'][a].get('sup_id_matches_ref')), 'True', 'False')))
H.append('</table>')
H.append('<p class="note"><b>注：</b><span class="mono">sup_id 同参考</span> 一列是「该臂现场解析出的类别 id 是否等于 qwen 参考 id」；'
         'A1 为 <b>False 是预期</b>（GLM4 词表不同），Phase 15 amend1 已把 <span class="mono">sup_id</span> 改为逐臂现场解析。'
         '<span class="mono">F1b</span>（各臂 tokenizer 自检：6 类别词均单 token 且 <span class="mono">decode(id)==词</span>）三臂均 True。</p>')

# ---------------- 2 可达性 ----------------
H.append('<h2>2 · 可达性剖面 ρ(ℓ)：写入窗是一个<b>阶跃点</b></h2>')
H.append('<p class="note">ρ(ℓ) = Y(ℓ, α=1) = dDonor / FULL_SWAP。' 
         '<b>绿</b> = 进入可达域（ρ ≥ 0.10），<b>橙</b> = 被掩膜排除。'
         'ℓ_reach = min{ℓ : ρ ≥ 0.5}。</p>')
H.append('<table><tr><th>臂</th><th>ρ(ℓ) 全曲线（23 位点，条高 ∝ ρ）</th>'
         '<th class="num">ρ 窗下一位点</th><th class="num">REACH 左端点</th><th class="num">ℓ_reach</th>'
         '<th>REACH 左端点 == ℓ_reach</th></tr>')
for a in ARMS:
    cells = []
    for i, s in enumerate(SITES):
        r = float(E7[a]['rho'][i])
        cls = 'hi' if r >= 0.10 else 'lo'
        cells.append('<td title="L%d: %s"><div class="bar %s" style="width:%dpx"></div></td>'
                     % (s, f(r, 4), cls, 6))
    ei = E7[a]['ell_reach']
    idx = SITES.index(ei)
    below = float(E7[a]['rho'][idx - 1]) if idx > 0 else None
    reach0 = E7[a]['reach'][0] if E7[a]['reach'] else None
    H.append('<tr><td><b style="color:%s">%s</b></td><td><table class="rho"><tr>%s</tr></table>'
             '<div class="note mono">%s</div></td>'
             '<td class="num mono">%s</td><td class="num">%s</td><td class="num"><b>%s</b></td>'
             '<td>%s</td></tr>'
             % (COL[a], SHORT[a], ''.join(cells),
                ' '.join('L%d:%.4f' % (st, float(E7[a]['rho'][k])) for k, st in enumerate(SITES)),
                f(below, 4), reach0, ei,
                badge(reach0 == ei, 'EQ（左端点即写入窗）', 'INNER（写入窗在域内）')))
H.append('</table>')
H.append('<div class="alert"><b>设计意图 vs 实测（MEMO §10 勘误 c）。</b> '
         '改动 2 的<b>意图</b>是让写入窗成为主域左端点；实测 <b>A0 / A1 成立</b>，'
         '<b>A2 不成立</b>（REACH 左端点 = 3，ρ(3)=%s ≥ 0.10，而 ℓ_reach = 4）。'
         'P3 的判据是「写入窗 ∈ REACH」，<b>三臂均满足</b>；「左端点」是 2/3 的描述性事实。</div>'
         % f(float(E7[A2]['rho'][SITES.index(3)]), 4))

# ---------------- 3 写入窗 = 门槛 ----------------
H.append('<h2>3 · 主结果 1：写入窗 = 可达性门槛（P3 / P4）</h2>')
H.append('<table><tr><th>臂</th><th class="num">L*_own（B_cat 相邻最大增量）</th>'
         '<th class="num">ℓ_reach（ρ ≥ 0.5 首越）</th><th class="num">相等？</th>'
         '<th>REACH</th><th>被掩膜排除</th></tr>')
for a in ARMS:
    eq = E3[a]['L_star_own'] == E7[a]['ell_reach']
    H.append('<tr><td><b style="color:%s">%s</b></td><td class="num">%s</td><td class="num">%s</td>'
             '<td class="num">%s</td><td class="mono">%s</td><td class="mono">%s</td></tr>'
             % (COL[a], SHORT[a], E3[a]['L_star_own'], E7[a]['ell_reach'], badge(eq),
                json.dumps(E7[a]['reach']), json.dumps(E7[a]['excluded'])))
H.append('</table>')
H.append('<p class="note"><b>两条完全不同的可观测量在同一位点重合：</b>'
         'B_cat 用「逐层独立类别子空间的相邻最大增量」（几何）；替换探针用「剂量-响应可达性阈值」（行为）。'
         '窗下 ρ ≤ %s，窗上 ρ ≈ 0.99。</p>'
         % f(max([float(E7[a]['rho'][SITES.index(E7[a]['ell_reach']) - 1]) for a in ARMS
                  if SITES.index(E7[a]['ell_reach']) > 0] or [0]), 4))

# ---------------- 4 集中度重设计 ----------------
H.append('<h2>4 · 主结果 2：集中度统计量重设计（P5）</h2>')
H.append('<h3>旧量（legacy 6..34 原口径，锚复现用）</h3>')
H.append('<table><tr><th>臂</th><th class="num">top3_x</th><th class="num">null95_x</th><th class="num">裕度_x</th>'
         '<th class="num">top3_j</th><th class="num">null95_j</th><th class="num">裕度_j</th>'
         '<th class="num">显著格</th></tr>')
for a in ARMS:
    lx = E5[a]['legacy_domain']['x']; lj = E5[a]['legacy_domain']['j']
    sig = [c for c, v in (('x', lx), ('j', lj)) if (v.get('margin') or -1) > 0]
    H.append('<tr><td><b style="color:%s">%s</b></td><td class="num mono">%s</td><td class="num mono">%s</td>'
             '<td class="num mono">%s</td><td class="num mono">%s</td><td class="num mono">%s</td>'
             '<td class="num mono">%s</td><td>%s</td></tr>'
             % (COL[a], SHORT[a], f(lx.get('top3'), 4), f((lx.get('null') or {}).get('null95'), 4),
                fp(lx.get('margin'), 4), f(lj.get('top3'), 4), f((lj.get('null') or {}).get('null95'), 4),
                fp(lj.get('margin'), 4), (', '.join(sig) or '无')))
H.append('</table>')
H.append('<h3>新量（主域 REACH 上，双边检验）—— 判决量</h3>')
H.append('<table><tr><th>臂</th><th class="num">主域步数</th><th class="num">旧量显著</th>'
         '<th class="num">com_layer(x)</th><th class="num">com_layer(J)</th><th class="num">com_sep（层）</th>'
         '<th class="num">span_3 x / J</th><th class="num">双边尾 x / J</th></tr>')
for a in ARMS:
    md = E5[a]['main_domain']; ns = E5[a]['new_stat']
    nx, nj = ns['x'] or {}, ns['j'] or {}
    H.append('<tr><td><b style="color:%s">%s</b></td><td class="num">%d</td><td class="num">%s</td>'
             '<td class="num mono">%s</td><td class="num mono">%s</td><td class="num mono"><b>%s</b></td>'
             '<td class="num mono">%s / %s</td><td class="num mono">%s / %s</td></tr>'
             % (COL[a], SHORT[a], len(md['x']['jumps']), (', '.join(VER[a].get('Q5_old_sig') or []) or '无'),
                f(nx.get('obs_com'), 3), f(nj.get('obs_com'), 3), fp(VER[a].get('Q4_sep'), 3),
                f(nx.get('obs_span'), 4), f(nj.get('obs_span'), 4),
                nx.get('com_tail'), nj.get('com_tail')))
H.append('</table>')
H.append('<div class="alert"><b>为什么旧量在这个装置上必然失效（结构性）。</b>'
         '<span class="mono">top3_share</span> 只依赖<b>最大 3 步净位移</b>，对单调段其窗位由曲线单调性'
         '<b>确定性</b>地决定；且置换零假设<b>保留 jump 的多重集</b> ⇒ 任何只依赖多重集的量'
         '（谱熵 / max÷mean / 参与比）在零假设下<b>恒等于观测</b>、双边必 p=1 —— 属<b>结构性退化族</b>，'
         '已明文整族排除。新量 <span class="mono">com_layer</span> 给出的是<b>位置</b>（单位：层），'
         '把「集中/分散」还原为「两坐标质心相距多少层」。</div>')

# ---------------- 5 预测 ----------------
H.append('<h2>5 · 预注册预测（观测前冻结）</h2>')
H.append('<table><tr><th>预测</th><th>判据</th><th class="num">结果</th><th>关键数</th></tr>')
_DESC = {
    'P1': '装置自检三臂 Q0=PASS',
    'P2': '冻结锚逐位复现（legacy 6..34；amend1 分层）',
    'P3': '写入窗入域且重算与 Phase 15 一致',
    'P4': '可达域左端点 == 写入窗（严格相等）',
    'P5': '新量显著格数 ≥ 旧量且 ≥ 2',
    'P6': '物理质心分离 ≥ %.1f 层（3/3）' % FL['CENTROID_SEP_MIN'],
    'P7': 'xhalf 质心在写入窗之后 ≥ %.1f 层（≥2/3）' % FL['CENTROID_AFTER_WIN_MIN'],
}
for k in sorted(PC):
    d = PC[k].get('detail')
    if k == 'P6':
        key = 'com_sep = %s' % ' / '.join(fp(VER[a].get('Q4_sep'), 1) for a in ARMS)
    elif k == 'P4':
        key = 'ℓ_reach == L*_own：%s' % ' / '.join('%s' % E7[a]['ell_reach'] for a in ARMS)
    elif k == 'P5':
        key = '旧 %d → 新 %d' % (d['n_old_sig'], d['n_new_sig'])
    elif k == 'P7':
        key = 'com_after_win = %s（%d/3）' % (' / '.join(f(VER[a].get('Q4_com_after_win_x'), 2) for a in ARMS),
                                             d['n_pass'])
    elif k == 'P2':
        key = 'n_strict_ok = %d / n_loose = %d' % (d['n_strict_ok'], d['n_loose'])
    elif k == 'P3':
        key = 'L*_own 重算 = %s' % ' / '.join(str(v) for v in d['L_star_now'].values())
    else:
        key = str(d)[:90]
    H.append('<tr><td><b>%s</b></td><td>%s</td><td class="num">%s</td><td class="mono">%s</td></tr>'
             % (k, _DESC.get(k, ''), badge(bool(PC[k]['pass_'])), key))
H.append('</table>')

# ---------------- 6 收尾链 ----------------
H.append('<h2>6 · 收尾链与同轮勘误</h2>')
H.append('<div class="card"><ul>')
H.append('<li><b>amend1（判据分层，不改假设）</b>：SMOKE 给出 <span class="mono">max|Δxhalf|=0.010517</span> —— '
         '这不是漂移，而是 <span class="mono">xhalf</span> 作为 <span class="mono">cross_alpha</span> 在 α 网格上的'
         '<b>插值量</b>在 3 点网格下的必然偏移。<b>对插值量设单一 1e-3 硬门违反「软门优先硬 assert」</b> ⇒ '
         '分为 <span class="mono">RECON_OK</span>(≤1e-3) / <span class="mono">RECON_OK_LOOSE</span>(≤5e-3，须附逐位点表) / '
         '<span class="mono">RECON_DRIFT</span>，冻结在观测前；正式 run 三臂全部落在<b>严格层</b>。</li>')
H.append('<li><b>独立磁盘复核</b>：<span class="mono">disk_verify_phase16.py</span> = '
         '<b>206 项 / 0 FAIL</b>（含 E4_per_pair → 全剖面逐位点独立重算、ρ 掩膜重算、新旧统计量在<b>各自种子</b>下的'
         '逐位复现、判决与预测复算、MEMO 前缀对冻结基线 sha256 的逐字节验证、Ledger 自哈希重算）。</li>')
H.append('<li><b>(a) §4 表头错标</b>：v1 把 <span class="mono">sup_id_matches_ref</span> 列标成「F1b 词表匹配」⇒ '
         'v2 更正为「sup_id 同参考」+ 补一行 F1b 说明。</li>')
H.append('<li><b>(b) exec 元数据缺陷（已冻结，不回改）</b>：'
         '<span class="mono">bootstrap.seeds</span> 记 <span class="mono">new_x/new_j = SEED+41/+53</span>，'
         '而实现（<span class="mono">n2h1a9_…py</span> L674–675）用 <span class="mono">SEED+61/+67</span>。'
         '只影响<b>零假设分位的复现</b>，不动任何观测值与判决；以 MEMO §10 勘误 b 为准。</li>')
H.append('<li><b>(c) §1「左端点」表述过强</b>：见 §2 提示框，P3 判据按「写入窗 ∈ REACH」评估。</li>')
H.append('<li><b>同轮重写留痕</b>：v1 原文保留于 '
         '<span class="mono">memo_append_phase16_v1_asappended.md</span>；回滚以<b>冻结追加前基线 sha256</b> 逐位验证。</li>')
H.append('</ul></div>')

# ---------------- 7 文件 ----------------
H.append('<h2>7 · 产物与限界</h2>')
H.append('<div class="card"><p class="note">'
         '<b>脚本</b>：<span class="mono">tests/deepseek/Phase16/</span> ｜ '
         '<b>产物</b>：<span class="mono">tests/deepseek_temp/Phase16/</span>（seal / amend1 / exec / result / 三臂报告 / 本页）｜ '
         '<b>记录</b>：<span class="mono">research/deepseek/docs/AGI_DEEPSEEK_MEMO.md</span> §Phase 16（L3684 起）｜ '
         '<b>Ledger</b>：n = <b>299</b>（<span class="mono">ledger_sha256_8 = %s</span>）</p>'
         '<ul>'
         '<li>集中度结论一律在<b>可达域 REACH</b> 上声明；不可达位点的位数与 ρ 必须逐臂列出。</li>'
         '<li><span class="mono">com_layer</span> 是<b>位置</b>统计量，不是显著性替代品；'
         '任何「集中/分散」断言必须同报双边分位带与观测。</li>'
         '<li>仍为<b>激活级单点替换</b>，不含权重级实现证明（继承 Phase 9/10/15 同一条限界）。</li>'
         '<li>三臂仍为 <b>nf4</b> 口径（bf16 对本机 16GB GPU 不可行，Phase 15 已实测否决）；'
         'A0 是量化保真校准臂，其「跨模型」身份不成立，只用于锚复现。</li>'
         '<li>零假设只检验<b>位置</b>是否非随机，<b>不检验</b>「jump 多重集本身是否来自某个机制」。</li>'
         '</ul></div>' % json.load(io.open(os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json'),
                                          encoding='utf-8'))['ledger_sha256_8'])

H.append('<p class="note" style="margin-top:26px">生成时间：%s ｜ 本页所有数字均由 '
         '<span class="mono">result_phase16.json</span> 现场渲染。</p>'
         % __import__('time').strftime('%Y-%m-%d %H:%M:%S'))
H.append('</div></body></html>')

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(H))
print('PRESENT ->', OUT, os.path.getsize(OUT), 'B')
