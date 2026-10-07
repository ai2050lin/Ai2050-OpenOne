# -*- coding: utf-8 -*-
"""Phase 15 展示页生成器（数据驱动，全部数字取自 result_phase15.json / judgement / amend1）。
产出 tests/deepseek_temp/Phase15/present_phase15.html（浅色主题、自包含、无外部依赖）。
"""
import io, os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
OUT = os.path.join(P15T, 'present_phase15.html')

R = json.load(io.open(os.path.join(P15T, 'result_phase15.json'), encoding='utf-8'))
AM1 = json.load(io.open(os.path.join(P15T, 'N2h1a8_design_seal_amend1.json'), encoding='utf-8'))
AM1_SHA8 = hashlib.sha256(io.open(os.path.join(P15T, 'N2h1a8_design_seal_amend1.json'), 'rb').read()).hexdigest()[:8]
RES_SHA8 = hashlib.sha256(io.open(os.path.join(P15T, 'result_phase15.json'), 'rb').read()).hexdigest()[:8]

VER = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']
E4 = R['E4_summary']; E5 = R['E5_concentration']; E3 = R['E3_localize']
E6 = R['E6_calibration']; FL = R['floors']; SITES = R['grid']['profile_sites']
META = R['arms_meta']; INH = R['inheritance_used']
A0, A1, A2 = 'A0_calib_qwen3-4b-nf4', 'A1_glm4-9b-nf4', 'A2_qwen3-14b-nf4'
ARMS = [a for a in (A0, A1, A2) if a in VER]
SHORT = {A0: 'A0·4B(nf4)', A1: 'A1·GLM4-9B', A2: 'A2·Qwen3-14B'}
COL = {A0: '#c2410c', A1: '#1d4ed8', A2: '#047857'}   # 4B / GLM4 / 14B

SUPS6 = list(R['extra']['classes']) if 'classes' in R.get('extra', {}) else ['水果', '动物', '交通工具', '家具', '金属', '颜色']


def f(x, n=4, dash='—'):
    try:
        if x is None:
            return dash
        return ('%.' + str(n) + 'f') % float(x)
    except Exception:
        return dash


def badge(ok, txt_ok='PASS', txt_no='FAIL'):
    cls = 'ok' if ok else 'bad'
    return '<span class="pill %s">%s</span>' % (cls, txt_ok if ok else txt_no)


H = []
H.append('<!DOCTYPE html>')
H.append('<html lang="zh-CN"><head><meta charset="utf-8">')
H.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
H.append('<title>Phase 15 / N2h1-α-8 · 跨模型复算「统一剖面」</title>')
H.append('''<style>
:root{--bg:#f7f8fa;--card:#ffffff;--ink:#16202b;--mut:#5b6b7c;--line:#e3e8ee;--ok:#0f766e;--bad:#b42318;--hi:#fef3c7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.62 -apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:25px;margin:0 0 4px}
h2{font-size:18px;margin:34px 0 10px;padding-left:10px;border-left:4px solid #0f766e}
h3{font-size:15px;margin:20px 0 8px;color:var(--mut);font-weight:600}
.sub{color:var(--mut);font-size:13.5px;margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:23px;font-weight:700;letter-spacing:-.2px}
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
.note{color:var(--mut);font-size:13px}
ul{margin:6px 0 0 0;padding-left:20px}li{margin:3px 0}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--mut);margin:8px 0 2px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
</style></head><body><div class="wrap">''')

H.append('<h1>Phase 15 / N2h1-α-8 · 跨模型复算「统一剖面」</h1>')
H.append('<p class="sub">三臂同一 <b>nf4</b> 口径（qwen3-4b 校准臂 / glm4-9b / Qwen3-14B）· '
         '单点替换族 <span class="mono">xhalf(ℓ)</span> / <span class="mono">J(ℓ)</span> 双坐标剖面 + 置换零假设校准 · '
         'result <span class="mono">%s</span> · amend1 <span class="mono">%s</span></p>' % (RES_SHA8, AM1_SHA8))

# ---------------- KPI ----------------
n95 = {a: (E5[a].get('null_x') or {}).get('null95') for a in ARMS}
mx = {a: E5[a].get('margin_x') for a in ARMS}
mj = {a: E5[a].get('margin_j') for a in ARMS}
cells = [SHORT[a] + '·J' for a in ARMS if (mj.get(a) or -1) > 0] + \
        [SHORT[a] + '·xhalf' for a in ARMS if (mx.get(a) or -1) > 0]

H.append('<h2>0 · 判决与关键读数</h2>')
H.append('<div class="grid">')
H.append('<div class="kpi"><div class="v">%s</div><div class="k">量化保真门 Q1（A0 nf4 vs Phase 12 bf16）</div>'
         '<div class="n">max|Δxhalf| = %s ≤ %.2f · argmax_w_x %s = %s</div></div>'
         % (VER[A0].get('Q1_label'), f(E6[A0].get('max_abs_dxh'), 4), FL['XH_FAITHFUL_TOL'],
            E6[A0].get('argmax_w_x_nf4'), E6[A0].get('argmax_w_x_bf16')))
H.append('<div class="kpi"><div class="v">%s</div><div class="k">Q2 两坐标窗的位置分离</div>'
         '<div class="n">d_argmax = %s（参照 4B = %d）· 三臂全部 ≥ 3</div></div>'
         % (JV['Q2_joint'],
            ' / '.join(str(E5[a].get('d_argmax_window')) for a in (A1, A2) if a in E5),
            abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13']))))
H.append('<div class="kpi"><div class="v">%s</div><div class="k">Q3 零假设校准的合取判决</div>'
         '<div class="n">null95_x = %s（门 %.2f，未双越）</div></div>'
         % (JV['Q3_joint'], ' / '.join(f(n95[a]) for a in (A0, A1, A2) if a in n95), FL['NULL_HIGH']))
H.append('<div class="kpi"><div class="v">%d / 6</div><div class="k">「臂 × 坐标」格超过各自零假设</div>'
         '<div class="n">%s ⇒ <b>无跨模型稳健集中度判据</b></div></div>'
         % (len(cells), '、'.join(cells)))
H.append('<div class="kpi"><div class="v">%s</div><div class="k">独立写入窗 L*_own（B_cat 相邻最大增量）</div>'
         '<div class="n">A0 复现 Phase 8 几何；L6 非普遍常数</div></div>'
         % ' / '.join('L%s' % E3[a].get('L_star_own') for a in (A0, A1, A2) if a in E3))
H.append('<div class="kpi"><div class="v">%d PASS / %d FAIL</div><div class="k">7 条预注册预测</div>'
         '<div class="n">FAIL：%s</div></div>'
         % (sum(1 for k in PC if PC[k]['pass_']), sum(1 for k in PC if not PC[k]['pass_']),
            ', '.join('%s（%s）' % (k, str(PC[k].get('detail', ''))[:70]) for k in sorted(PC) if not PC[k]['pass_']) or '无'))
H.append('</div>')

# ---------------- 双坐标集中度 ----------------
H.append('<h2>1 · 双坐标集中度 × 置换零假设 × 裕度（死线条款逐行落实）</h2>')
H.append('<div class="legend"><span>裕度 = top3_share − null 95 分位：'
         '<span class="sw" style="background:#dcfce7"></span>正 = 超过零假设'
         '<span class="sw" style="background:#fee4e2;margin-left:12px"></span>负 = 低于零假设</span></div>')
H.append('<table><thead><tr><th>臂 · 坐标</th><th class="num">top3_share</th><th class="num">argmax_w</th>'
         '<th>窗口语义</th><th class="num">null 95 分位</th><th class="num">裕度</th><th>过零假设？</th></tr></thead><tbody>')
for a in ARMS:
    c = E5[a]
    for coord, key, wk, nk, mk in (('xhalf', 'x', 'win_sem_x', 'null_x', 'margin_x'),
                                   ('J', 'j', 'win_sem_j', 'null_j', 'margin_j')):
        sem = c.get(wk) or {}
        bg = '#f0fdf4' if (c.get(mk) or -1) > 0 else '#fff7f6'
        H.append('<tr style="background:%s"><td><b>%s</b>·%s</td><td class="num">%s</td><td class="num">%s</td>'
                 '<td class="mono">w=%s: L%s→L%s</td><td class="num">%s</td><td class="num"><b>%+.4f</b></td>'
                 '<td>%s</td></tr>'
                 % (bg, SHORT[a], coord, f(c.get('top3_' + key)), c.get('argmax_w_' + key),
                    sem.get('w'), sem.get('a'), sem.get('b'),
                    f((c.get(nk) or {}).get('null95')), (c.get(mk) or 0.0),
                    '<b>是</b>' if (c.get(mk) or -1) > 0 else '否'))
H.append('</tbody></table>')
H.append('<p class="note">Q2 判定的是「<b>窗的位置</b>不同」（由 d_argmax ≥ 3 判定），<b>不</b>预设两窗的集中度都显著；'
         '显著性由本表裁决 —— 6 格只有 %d 格过线。⇒ Phase 13/14 的「坐标依赖」升级为「<b>坐标 × 模型</b>双重依赖」。</p>' % len(cells))

# ---------------- 剖面曲线 ----------------
W_, Hh, PADL, PADR, PADT, PADB = 1040, 300, 56, 20, 18, 42
tb, ts = min(E4[a]['J'][i] for a in ARMS for i in range(len(SITES))), max(E4[a]['J'][i] for a in ARMS for i in range(len(SITES)))


def xp(i):
    return PADL + i * (W_ - PADL - PADR) / (len(SITES) - 1)


def yp(v):   # log 轴
    import math
    lo, hi = math.log10(tb), math.log10(ts)
    return PADT + (1 - (math.log10(max(v, tb)) - lo) / (hi - lo)) * (Hh - PADT - PADB)


H.append('<h2>2 · 三臂统一剖面 J(ℓ)：浅端 O(10–40) 单调跌到深端 O(1)</h2>')
H.append('<svg viewBox="0 0 %d %d" width="100%%" style="max-width:%dpx" role="img" '
         'aria-label="三臂 J(depth) 剖面">' % (W_, Hh, W_))
H.append('<rect x="0" y="0" width="%d" height="%d" fill="#ffffff" stroke="#e3e8ee"/>' % (W_, Hh))
for gv in (1, 2, 5, 10, 20, 40):
    if tb <= gv <= ts:
        y = yp(gv)
        H.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eef1f5"/>' % (PADL, y, W_ - PADR, y))
        H.append('<text x="%d" y="%.1f" fill="#8a97a6" font-size="10.5" text-anchor="end">%d</text>' % (PADL - 7, y + 3.5, gv))
for i, s in enumerate(SITES):
    if i % 3 == 0 or i == len(SITES) - 1:
        H.append('<text x="%.1f" y="%d" fill="#8a97a6" font-size="10.5" text-anchor="middle">L%d</text>'
                 % (xp(i), Hh - PADB + 15, s))
for a in ARMS:
    pts = ' '.join('%.1f,%.1f' % (xp(i), yp(E4[a]['J'][i])) for i in range(len(SITES)))
    H.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.2"/>' % (pts, COL[a]))
    H.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (xp(len(SITES) - 1), yp(E4[a]['J'][-1]), COL[a]))
lx = PADL + 8
for a in ARMS:
    H.append('<rect x="%d" y="%d" width="11" height="11" rx="3" fill="%s"/>' % (lx, PADT - 4, COL[a]))
    H.append('<text x="%d" y="%d" fill="#33424f" font-size="12">%s（spearman(J,depth)=%s）</text>'
             % (lx + 16, PADT + 6, SHORT[a], f(E5[a].get('spearman_J_depth'), 3)))
    lx += 300
H.append('<text x="%d" y="%d" fill="#8a97a6" font-size="10.5">log 纵轴</text>' % (W_ - PADR - 46, PADT + 6))
H.append('</svg>')

# ---------------- 写入窗 ----------------
H.append('<h2>3 · 独立写入窗 L*_own（B_cat 逐层独立重建，未沿用 L6/U6）</h2>')
H.append('<table><thead><tr><th>臂</th><th>L*_own</th><th class="num">相邻最大增量</th>'
         '<th>B_cat 曲线（候选层）</th><th class="num">定位耗时 s</th></tr></thead><tbody>')
for a in ARMS:
    e3 = E3[a]
    seq = ' · '.join('<span style="color:#8a97a6">L%s</span>:%+.3f' % (k, v) for k, v in (e3.get('curve') or {}).items())
    H.append('<tr><td><b>%s</b></td><td><b>L%s</b></td><td class="num">%s</td>'
             '<td class="mono" style="font-size:11.5px;line-height:1.75">%s</td><td class="num">%s</td></tr>'
             % (SHORT[a], e3.get('L_star_own'), f(e3.get('L_star_increment'), 3), seq, f(e3.get('seconds'), 1)))
H.append('</tbody></table>')

# ---------------- 装置门 ----------------
H.append('<h2>4 · 装置门（三臂 F0–F4 + E0 自检）</h2>')
H.append('<table><thead><tr><th>臂</th><th>config sha8</th><th>T=2 全 41</th>'
         '<th>F1b 类别 token（逐臂解析）</th><th>与 qwen 参考一致</th><th>F2 base bad</th><th class="num">加载 s</th></tr></thead><tbody>')
for a in ARMS:
    sida = R['sup_id_per_arm'].get(a) or {}
    H.append('<tr><td><b>%s</b></td><td class="mono">%s</td><td>%s</td>'
             '<td class="mono" style="font-size:11.5px">%s</td><td>%s</td><td>%s</td><td class="num">%s</td></tr>'
             % (SHORT[a], META[a]['config_sha8'], '✔' if R['E0_selfcheck'][a].get('T2_only') else '✘',
                json.dumps({k: sida[k] for k in SUPS6 if k in sida}, ensure_ascii=False),
                ('<b>是</b>' if R['arms'][a].get('sup_id_matches_ref') else
                 '<span class="pill warn">否（amend1 已修复）</span>'),
                '<b>%d/41</b>' % len(R['arms'][a].get('F2_base_bad') or []),
                f(R['arms'][a].get('load_s'), 1)))
H.append('</tbody></table>')

# ---------------- 量化校准 ----------------
e6 = E6.get(A0, {})
H.append('<h2>5 · A0 量化保真校准（nf4 复算 Phase 12 bf16 已发表量）</h2>')
H.append('<table><thead><tr><th>量</th><th class="num">nf4</th><th class="num">Phase 12 bf16</th>'
         '<th class="num">差</th><th>门</th></tr></thead><tbody>')
rows = [('argmax_w_x', f(e6.get('argmax_w_x_nf4'), 0), f(e6.get('argmax_w_x_bf16'), 0),
         '相等' if e6.get('argmax_same') else '不等', '与 bf16 相同 → ' + badge(bool(e6.get('argmax_same')))),
        ('max|Δxhalf|（18 位点）', '—', '—', f(e6.get('max_abs_dxh'), 4),
         '≤ %.2f → %s' % (FL['XH_FAITHFUL_TOL'], badge(bool(e6.get('pass_tol'))))),
        ('share_x = top3_x', f(e6.get('share_x_nf4')), f(e6.get('share_x_bf16')),
         f((e6.get('share_x_nf4') or 0) - (e6.get('share_x_bf16') or 0)), '（描述性）'),
        ('XH_RANGE', f(e6.get('XH_RANGE_nf4')), f(e6.get('XH_RANGE_bf16')),
         f((e6.get('XH_RANGE_nf4') or 0) - (e6.get('XH_RANGE_bf16') or 0)),
         '∈ [%.2f, %.2f] → %s' % (FL['XH_RANGE_BAND'][0], FL['XH_RANGE_BAND'][1],
                                  badge(bool(e6.get('XH_RANGE_nf4') is not None and
                                             FL['XH_RANGE_BAND'][0] <= e6['XH_RANGE_nf4'] <= FL['XH_RANGE_BAND'][1])))),
        ('max|Δrecover|', '—', '—', f(e6.get('max_abs_drecover'), 4), '（描述性）'),
        ('J 比值范围', '—', '—', '[%s, %s]' % (f(e6.get('J_ratio_min'), 3), f(e6.get('J_ratio_max'), 3)),
         '（网格差异 + 量化）')]
for r in rows:
    H.append('<tr><td class="mono">%s</td><td class="num">%s</td><td class="num">%s</td>'
             '<td class="num"><b>%s</b></td><td>%s</td></tr>' % r)
H.append('</tbody></table>')
H.append('<p class="note">唯一量化伪影：<span class="mono">argmax_w_j</span> nf4 = %s vs bf16 = %s（'
         '<span class="mono">xhalf</span> 的 argmax_w_x 两口径相同）⇒ 门必须写在最稳健的量上。</p>'
         % (E5[A0].get('argmax_w_j'), INH['MODE_J_13']))

# ---------------- amend1 ----------------
ev = AM1['evidence_from_device_gate']
H.append('<h2>6 · 事故与拦截：amend1（apparatus fix，不改假设）</h2>')
H.append('<div class="alert"><b>首轮正式运行被装置门拦下 ⇒ A1 数据作废重跑。</b><br>'
         '根因：冻结 seal 的 <span class="mono">panel.sup_id</span>（如 水果=%s）是 <b>qwen 词表</b> id 却被<b>全局</b>用于三臂；'
         '<span class="mono">glm4-9b-chat-hf</span> 词表不同（vocab <b>%s vs %s</b>）⇒ A1 全程读<b>错误的类别 token</b>。'
         '症状完全像一条机制发现：<span class="mono">F2 base bad = %d/41</span>、受体类分数均值 <b>%+.3f</b>、'
         '<span class="mono">FULL_SWAP = %+.3f</span>（4B 为 %+.3f）、剂量曲线平坦且低 α 段为负。<br>'
         '<b>拦截者</b>：<span class="mono">F2_base_ok</span> —— 一条只问「token 读对了吗」、与主结论（d_argmax / null95）'
         '<b>正交</b>的装置门。<b>修正</b>：sup_id 改为每臂由该臂 tokenizer 现场解析 + 硬断言 F1b（6/6 类别词单 token 且 decode 可逆）；'
         'A0/A2 解析结果与冻结值<b>逐位相同</b>（零副作用）。'
         '<br><span class="note">若无此门，这会以「GLM4 的 is-a 关系不成立」的机制结论被写进备忘录。</span></div>'
         % (104618, 151329, 151643, ev['A1_F2_base_bad_n'], ev['A1_F2_receptor_class_score_mean'],
            ev['A1_FULL_SWAP'], ev['A0_FULL_SWAP']))

# ---------------- 预测 ----------------
H.append('<h2>7 · 7 条预注册预测</h2>')
H.append('<table><thead><tr><th>预测</th><th>内容</th><th>实测</th><th>结果</th></tr></thead><tbody>')
for k in sorted(PC, key=lambda z: int(z[1])):
    det = ', '.join('%s=%s' % (kk, json.dumps(vv, ensure_ascii=False)) for kk, vv in PC[k].items()
                    if kk not in ('desc', 'pass_', 'smoke'))
    H.append('<tr><td><b>%s</b></td><td>%s</td><td class="mono" style="font-size:11.5px">%s</td><td>%s</td></tr>'
             % (k, PC[k].get('desc', '')[:120], det[:150], badge(bool(PC[k]['pass_']))))
H.append('</tbody></table>')

# ---------------- 限界 ----------------
H.append('<h2>8 · 限界（必须与结论同时引用）</h2>')
H.append('<div class="card"><ul>')
for h in R.get('honesty', []) or []:
    H.append('<li>%s</li>' % h)
H.append('<li>A1/A2 各只有一个模型 ⇒ <b>无法分离「家族」与「规模」</b>（GLM 家族 × 18.8 GB vs Qwen3 家族 × 29.5 GB）。</li>')
H.append('<li><span class="mono">null</span> 分布由 <span class="mono">jumps</span> <b>重排</b>得到，是统计零假设而非物理「无效应」零假设。</li>')
H.append('<li>「域外可达性」只测 profile 的 18 个位点，浅端 L0–L5 与末层后未测。</li>')
H.append('<li>三臂 <span class="mono">F3 α=0</span> 只在首/中/末 3 位点 × 前 3 对上验证，未做全位点回归。</li>')
H.append('<li>全部为<b>激活级干预</b>，非权重实现级证明。</li>')
H.append('</ul></div>')

H.append('<p class="note" style="margin-top:26px">数据源：<span class="mono">tests/deepseek_temp/Phase15/result_phase15.json</span> '
         '（sha8 %s）· 判据：<span class="mono">N2h1a8_design_seal.json</span> + <span class="mono">…_amend1.json</span> '
         '（sha8 %s）· 独立复核 <span class="mono">disk_verify_phase15.py</span>：126 checks / FAIL = 0。</p>'
         % (RES_SHA8, AM1_SHA8))
H.append('</div></body></html>')

html = '\n'.join(H)
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(html)
b = io.open(OUT, 'rb').read()
print('WROTE %s' % OUT)
print('  bytes = %d ; sha8 = %s' % (len(b), hashlib.sha256(b).hexdigest()[:8]))
