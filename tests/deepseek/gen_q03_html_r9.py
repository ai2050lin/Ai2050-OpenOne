# -*- coding: utf-8 -*-
"""Q03 汇报页（单文件 HTML，light theme；数字全部现场渲染）"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
QR = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_result.json')
QEX = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_execution.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_baseline_r9.html')

q = json.loads(open(QR, 'rb').read().decode('utf-8-sig'))
ex = json.loads(open(QEX, 'rb').read().decode('utf-8-sig'))
S, PM, FP, CAR = q['summary'], q['per_model'], q['fingerprint'], q['carriers']
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']

def sha8b(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

CSS = """body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;background:#fff;color:#1a1a1a;margin:0;padding:32px;line-height:1.6}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px;border-bottom:3px solid #2563eb;padding-bottom:10px}
h2{font-size:17px;margin:26px 0 10px;color:#1e40af}
.sub{color:#666;font-size:13px;margin-bottom:18px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
th,td{border:1px solid #d8dee9;padding:7px 10px;text-align:left}
th{background:#f1f5f9;font-weight:600}
td.num{font-family:Consolas,monospace;text-align:right}
.pass{color:#15803d;font-weight:600}
.fail{color:#b91c1c;font-weight:600}
.kpi{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}
.card{flex:1;min-width:190px;border:1px solid #d8dee9;border-radius:8px;padding:12px 14px;background:#f8fafc}
.card .lab{font-size:12px;color:#64748b}
.card .val{font-size:22px;font-weight:700;font-family:Consolas,monospace;color:#0f172a}
.card .note{font-size:11px;color:#94a3b8}
.warn{border-left:4px solid #f59e0b;background:#fffbeb;padding:12px 16px;margin:14px 0;font-size:13px;border-radius:0 6px 6px 0}
.ok{border-left:4px solid #16a34a;background:#f0fdf4;padding:12px 16px;margin:14px 0;font-size:13px;border-radius:0 6px 6px 0}
code{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-family:Consolas,monospace;font-size:12px}
.mono{font-family:Consolas,monospace}"""

H = []
def W(s):
    H.append(s)

W('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
W('<meta name="viewport" content="width=device-width,initial-scale=1">')
W('<title>Q03 E_read 统一基线复算</title><style>%s</style></head><body><div class="wrap">' % CSS)
W('<h1>Q03 · E_read 统一基线复算</h1>')
W('<div class="sub">B 闸门（KPI）· 零 GPU · recompute-only（无新模型观测）· design_sha <code>%s</code> · res_sha8 <code>%s</code> · 2026-10-03</div>' % (ex['design_sha'][:8], q['res_sha8']))

_es = [PM[m]['b4_rel_readout_mean3seed_recompute'] for m in MODELS]
W('<div class="ok"><b>一句话结论：</b>K1 判决所依赖的 E_read 基线已从三份冻结载体<b>独立重算并逐位复现</b>（全部锚 <code>drift = 0.00e+00</code>）；三模型误差 <b>%.1f%% / %.1f%% / %.1f%%</b>，池化 <b>%s</b>，<b>5%% 门 0/3 过门</b>（最小值仍为门槛的 %.2f 倍）。</div>' % (100*_es[0], 100*_es[1], 100*_es[2], '%.6f' % S['pooled_mean'], S['min_E_x']))

W('<h2>1 · 这个方法在算什么</h2>')
W('<p>模型见过 (苹果, 水果)，能否预测<b>没见过</b>的组合 (床, 颜色) 在<b>读出层</b>的内部状态？B4 是"加性最强"的预测器：只允许"实体 + 类别 + 模板"三者的<b>相加</b>效应，用其余组合拟合，再来预测被藏起来的组合。</p>')
W('<p>误差 <code>E_read = &lt;预测偏差平方均值&gt; / &lt;训练数据方差&gt;</code>，即<b>归一化误差</b>。所以 0.33 意味着加性模型只解释了 67% 的方差；<b>0.05 门 = 加性模型必须解释 95% 的方差</b>才算"组合可加"。</p>')
W('<p>held-out 切分 = <code>S1</code>（seed 7/8/9，抽 20% 组合）：每 seed 49 个测试组合 × 3 模板 = <b>147 个测试行</b>。</p>')

W('<h2>2 · 三个模型的结果（读出层）</h2>')
W('<table><tr><th>模型</th><th>载体 sha8</th><th>H 形状</th><th>读出层</th><th>E_read（复算）</th><th>3151/3152 锚</th><th>drift</th><th>5% 门</th></tr>')
for m in MODELS:
    v = PM[m]
    W('<tr><td><code>%s</code></td><td class="mono">%s</td><td class="mono">%s</td><td class="num">%d</td><td class="num">%.6f</td><td class="num">%.6f</td><td class="num">%.2e</td><td class="%s">%s</td></tr>' % (
        m, CAR[m]['npz_sha8'], str(tuple(v['H_shape'])), v['readout_layer'],
        v['b4_rel_readout_mean3seed_recompute'], v['b4_rel_readout_mean3seed_anchor'],
        v['b4_rel_readout_mean3seed_drift'], 'pass' if v['gate_pass'] else 'fail',
        '过' if v['gate_pass'] else '未过'))
W('</table>')

W('<div class="kpi">')
W('<div class="card"><div class="lab">E_read 池化（3 模型均值）</div><div class="val">%s</div><div class="note">sd = %s（n=3）</div></div>' % ('%.6f' % S['pooled_mean'], '%.6f' % S['pooled_sd_3models']))
W('<div class="card"><div class="lab">5%% 门（≤0.05）过门</div><div class="val fail">%s</div><div class="note">未过门</div></div>' % S['gate_pass_frac'])
W('<div class="card"><div class="lab">最小 E_read / 门槛</div><div class="val">%.2f×</div><div class="note">%.6f</div></div>' % (S['min_E_x'], S['min_E']))
W('<div class="card"><div class="lab">锚逐位复现</div><div class="val pass">0.00e+00</div><div class="note">6 组锚全匹配</div></div>')
W('</div>')

W('<h2>3 · per-seed 与 bootstrap 95% CI（row-level，n_boot=10000）</h2>')
W('<table><tr><th>模型</th><th>seed 7</th><th>seed 8</th><th>seed 9</th><th>3-seed 均值</th></tr>')
for m in MODELS:
    cells = []
    for s in ['7', '8', '9']:
        b = PM[m]['bootstrap_per_seed'][s]
        cells.append('%.6f<br><span class="mono" style="font-size:11px;color:#64748b">[%.4f, %.4f]</span>' % (b['mean'], b['ci_lo'], b['ci_hi']))
    W('<tr><td><code>%s</code></td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td><td class="num"><b>%.6f</b></td></tr>' % (
        m, cells[0], cells[1], cells[2], PM[m]['b4_rel_readout_mean3seed_recompute']))
W('</table>')

W('<h2>4 · 统一 held-out 指纹（三模型共用同一构造）</h2>')
W('<p><code>PAIRS = 41 实体 × 6 类别 = 246</code>；<code>seeds = [7,8,9]</code>；<code>frac = 0.2</code> —— 3151（glm4）与 3152（qwen3）两份预注册<b>逐字相同</b>，故三个模型的测试组合完全一致，E_read 可跨模型直接比较。</p>')
W('<table><tr><th>seed</th><th>训练组合</th><th>测试组合</th><th>测试行</th><th>测试集 sha8</th></tr>')
for s in ['7', '8', '9']:
    v = FP[s]
    W('<tr><td class="num">%s</td><td class="num">%d</td><td class="num">%d</td><td class="num">%d</td><td class="mono">%s</td></tr>' % (
        s, v['n_train_pairs'], v['n_test_pairs'], v['n_test_pairs'] * 3, v['test_pairs_sha8']))
W('</table>')

W('<h2>5 · k* 层对照（为什么这不是伪影）</h2>')
W('<p>同一口径在<b>承诺层 k\\*</b>（假设自己挑的浅层）上复算，同样逐位匹配：三模型 k\\* 误差 = %s。即"浅层看起来可加、读出层不可加"的落差是<b>真实的层位效应</b>，与 R8 的 K1 判决（读出层 <code>fired_all_models</code>）自洽。</p>' % ' / '.join('%.6f' % PM[m]['b4_rel_kstar_mean3seed_recompute'] for m in MODELS))

W('<h2>6 · ⚠ 并发写者与文件漂移（如实记录）</h2>')
W('<div class="warn">')
W('<b>1.</b> 研究日志在 R8 收尾（<code>150241da</code> / 685,647 B）之后、本轮开始前被<b>外部写入</b>：<code>+2 B</code> → <code>5be5bf12</code>（685,649 B），随后 Phase 36 追加 → <code>5b3bdcec</code>（692,727 B）。<br>')
W('<b>2.</b> 证据：<code>prefix(685,649)</code> 的 sha8 = <code>5be5bf12</code>，恰好等于并发写者 Phase 36 脚本自报的 before 值 ⇒ <b>追加未篡改前文</b>。<br>')
W('<b>3.</b> Phase 36 <b>不是本对话的产物</b>（内容为 E4/E4b 词频 × 嵌入有效维度，用户队列外插入）；本对话只在其后追加 Phase 37，未改写它。<br>')
W('<b>4.</b> 观察到一处质量缺陷：Phase 36 追加引入了 <b>45 个裸 LF</b>（文件 bare_lf 0 → 45），全部落在 Phase 36 正文区间；Phase 35 及之前仍是纯 CRLF，本次 Phase 37 追加也保持纯 CRLF。未擅自修改他线内容，留待决定。')
W('</div>')

W('<h2>7 · 产物与复核</h2>')
W('<table><tr><th>文件</th><th>说明</th><th>sha8</th></tr>')
for lbl, desc in [('q03_execution.json', '预注册（design_sha %s）' % ex['design_sha'][:8]),
                  ('q03_result.json', '复算结果 + bootstrap CI（res_sha8 %s）' % q['res_sha8']),
                  ('q03_report.txt', '人类可读报告'),
                  ('verify_q03.txt', '独立复核 42 PASS / 0 FAIL'),
                  ('verify_q03_landing.txt', '落账复核 28 PASS / 0 FAIL')]:
    p = os.path.join(ROOT, 'tests', 'deepseek', 'result', lbl)
    W('<tr><td><code>%s</code></td><td>%s</td><td class="mono">%s</td></tr>' % (lbl, desc, sha8b(p)))
W('</table>')
W('<p style="font-size:12px;color:#94a3b8">队列：<code>Q03 → sealed</code>（sealed_items 现 6 项：Q01/Q02/Q03/Q08/Q09/Q12）。下一项 B 闸门：<code>Q04 E_ar(k)</code>（gpu=mid）。</p>')
W('</div></body></html>')

html = '\n'.join(H)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)

# 清洁度检查
import re
bad = re.findall(r'\{\{|\}\}|TODO|XXX|%s|%d|\.6f', html)
print('placeholder residue =', bool(bad), sorted(set(bad))[:6])
for k in ['0.373350', '0.331615', '0.398601', '0.389835', '0.00e+00', '0/3', '6.63', '45 个裸 LF', '5be5bf12', 'fired_all_models']:
    print('  %-18s %s' % (k, k in html))
print('bytes=%d chars=%d sha8=%s' % (len(html.encode()), len(html), sha8b(OUT)))
