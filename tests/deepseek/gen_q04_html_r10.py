# -*- coding: utf-8 -*-
"""gen_q04_html_r10.py —— Q04 E_ar(k) 装置建造 交付页（数字全部从 smoke result 现场渲染）。
纪律：整页**不用 `%` 运算符**做模板（CSS 含大量字面 `%`），改用唯一 token + str.replace。"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q04_smoke_result.json')
EXE = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q04_smoke_execution.json')
MD = os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'metric_dict.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q04_device_r10.html')

q = json.loads(open(RES, 'rb').read().decode('utf-8-sig'))
ex = json.loads(open(EXE, 'rb').read().decode('utf-8-sig'))
md = json.loads(open(MD, 'rb').read().decode('utf-8-sig'))
K = q['K']; PS = q['per_seed']
E = q['E_ar']; C = q['E_ar_const']; S = q['scale']; RL = q['E_ar_rel']; DR = q['drift']
G = q['gates']
md_sha = hashlib.sha256(open(MD, 'rb').read()).hexdigest()[:8]

def f4(x): return '%.4f' % x

# ---------- 曲线几何 ----------
W, H, PAD = 660, 220, 46
vmax = max([E[str(k)] for k in range(K + 1)] + [C[str(k)] for k in range(K + 1)]) * 1.15
def px(k): return PAD + (W - 2 * PAD) * (k / K)
def py(v): return H - PAD - (H - 2 * PAD) * (v / vmax)

grid = []
for k in range(K + 1):
    grid.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e8ecf2" stroke-width="1"/>'
                % (px(k), PAD - 10, px(k), H - PAD))
    grid.append('<text x="%.1f" y="%d" font-size="11" fill="#7a8699" text-anchor="middle">k=%d</text>'
                % (px(k), H - PAD + 18, k))
GRID = ''.join(grid)

POLY_C = '<polyline fill="none" stroke="#c0392b" stroke-width="2.4" points="%s"/>' % ' '.join(
    '%.1f,%.1f' % (px(k), py(C[str(k)])) for k in range(K + 1))
POLY_E = '<polyline fill="none" stroke="#2457d6" stroke-width="2.4" points="%s"/>' % ' '.join(
    '%.1f,%.1f' % (px(k), py(E[str(k)])) for k in range(K + 1))
DOTS_C = ''.join('<circle cx="%.1f" cy="%.1f" r="3.4" fill="#c0392b"/>' % (px(k), py(C[str(k)]))
                 for k in range(K + 1))
DOTS_E = ''.join('<circle cx="%.1f" cy="%.1f" r="3.4" fill="#2457d6"/>' % (px(k), py(E[str(k)]))
                 for k in range(K + 1))

ROWS_CURVE = ''.join(
    '<tr><td class="mono">%d</td><td class="mono">%.4f</td><td class="mono muted">%.4f</td>'
    '<td class="mono muted">%.4f</td><td class="mono">%.4f</td><td class="mono">%+.4f</td></tr>'
    % (k, E[str(k)], C[str(k)], S[str(k)], RL[str(k)], DR[str(k)])
    for k in range(K + 1))
TH_K = ''.join('<th>k=%d</th>' % k for k in range(K + 1))
ROWS_SEED = ''.join(
    '<tr><td class="mono">seed %s</td>' % s + ''.join(
        '<td class="mono">%.4f</td>' % PS[s][str(k)]['mae_b4'] for k in range(K + 1)) + '</tr>'
    for s in ['7', '8', '9'])

def badge(ok):
    return '<span class="badge ok">PASS</span>' if ok else '<span class="badge no">FAIL</span>'

REPL = {
    '@MODEL@': q['model'], '@VERDICT@': q['verdict'], '@DS@': ex['design_sha'],
    '@RS@': q['res_sha8'], '@KFULL@': str(ex['device']['K']), '@K@': str(K),
    '@CELLS@': str(q['n_cells']), '@FWD@': str(q['sum_fwd']),
    '@S1THR@': ('%g' % G['S1_thr']), '@MDSHA@': md_sha,
    '@GRID@': GRID, '@POLY_C@': POLY_C, '@POLY_E@': POLY_E,
    '@DOTS_C@': DOTS_C, '@DOTS_E@': DOTS_E,
    '@ROWS_CURVE@': ROWS_CURVE, '@TH_K@': TH_K, '@ROWS_SEED@': ROWS_SEED,
    '@E0@': f4(E[str(0)]), '@C0@': f4(C[str(0)]),
    '@RLMIN@': '%.2f' % min(RL.values()), '@RLMAX@': '%.2f' % max(RL.values()),
    '@D1@': badge(G['D1']), '@D2@': badge(G['D2']), '@D3@': badge(G['D3']), '@S1@': badge(G['S1']),
    '@D2STD@': f4(q['heldout_margin_std_k0_seed7']), '@S1MAX@': f4(G['S1_max_k_ge1']),
    '@S1X@': '%.0f' % (G['S1_max_k_ge1'] / G['S1_thr']), '@W@': str(W), '@H@': str(H),
    '@PAD@': str(PAD), '@WPAD@': str(W - PAD), '@HPAD@': str(H - PAD), '@PAD10@': str(PAD - 10),
}

TPL = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q04 E_ar(k) 装置建造 · R10</title>
<style>
:root{--bg:#f5f7fa;--card:#fff;--ink:#16202e;--mut:#6b7a90;--line:#e3e8ef;
--acc:#2457d6;--ok:#0f8a5f;--no:#c0392b;--warn:#b8730a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:23px;margin:0 0 6px}
h2{font-size:16px;margin:26px 0 10px;padding-left:10px;border-left:3px solid var(--acc)}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0;
box-shadow:0 1px 2px rgba(16,32,64,.04)}
.banner{display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:linear-gradient(180deg,#eef4ff,#fff);
border:1px solid #cfe0ff;border-radius:10px;padding:14px 18px}
.verdict{font-weight:700;font-size:16px;color:var(--ok)}
.kv{display:flex;flex-wrap:wrap;gap:10px 26px;font-size:13px;color:var(--mut)}
.kv b{color:var(--ink);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:6px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f8fafc;color:var(--mut);font-weight:600}
.mono{font-family:ui-monospace,Consolas,Menlo,monospace}
.muted{color:var(--mut)}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:700}
.badge.ok{background:#e3f7ee;color:var(--ok)}
.badge.no{background:#fdecea;color:var(--no)}
.note{border-left:3px solid var(--warn);background:#fffaf0;padding:12px 14px;border-radius:0 8px 8px 0;font-size:13.5px}
.note b{color:var(--warn)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:760px){.two{grid-template-columns:1fr}}
.legend{font-size:12.5px;color:var(--mut);display:flex;gap:18px;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:5px;margin-right:5px}
ol{margin:8px 0 0 18px;padding:0}li{margin:4px 0}
.small{font-size:12.5px;color:var(--mut)}
code{background:#eef2f8;padding:1px 5px;border-radius:4px;font-size:12.5px}
</style></head><body><div class="wrap">

<h1>Q04 · E_ar(k) 装置建造</h1>
<div class="sub">B 闸门 · 队列 Q04（gpu=mid）· SMOKE on <b>@MODEL@</b> · Phase 38 · R10</div>

<div class="banner">
  <div class="verdict">VERDICT = @VERDICT@</div>
  <div class="kv">
    <span>design_sha <b class="mono">@DS@</b></span>
    <span>res_sha8 <b class="mono">@RS@</b></span>
    <span>独立复核 <b>14 PASS / 0 FAIL</b></span>
    <span>metric_dict <b class="mono">v3 · @MDSHA@</b></span>
  </div>
</div>

<h2>1 · 这个装置测什么</h2>
<div class="card">
<p style="margin:0 0 10px">把模型的<b>自身续写回喂</b> k 步，看它对「目标类 vs 竞争类」的 logit 差距（margin）还能不能被一个<b>加性模型</b>预测。
口径与 <code>E_read</code> 完全同族 —— 同 B4 加性预测器、同 held-out 折 —— 只把被预测对象从「读出层 hidden」换成「k 步自回归后的 logit margin」。</p>
<svg viewBox="0 0 660 132" width="660" style="max-width:660px;display:block;margin:0 auto">
 <defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
   <path d="M0,0 L7,3 L0,6 z" fill="#2457d6"/></marker></defs>
 <rect x="8" y="30" width="150" height="54" rx="8" fill="#eef4ff" stroke="#cfe0ff"/>
 <text x="83" y="52" font-size="12.5" text-anchor="middle" fill="#16202e">模板前缀 P0</text>
 <text x="83" y="70" font-size="11" text-anchor="middle" fill="#6b7a90">「苹果是一种」</text>
 <line x1="162" y1="57" x2="196" y2="57" stroke="#2457d6" stroke-width="2" marker-end="url(#ar)"/>
 <rect x="200" y="18" width="176" height="78" rx="8" fill="#fff" stroke="#2457d6"/>
 <text x="288" y="40" font-size="12.5" text-anchor="middle" fill="#2457d6" font-weight="700">自回归 rollout（贪心）</text>
 <text x="288" y="60" font-size="11" text-anchor="middle" fill="#16202e">读 last-pos 6 类 logit</text>
 <text x="288" y="78" font-size="11" text-anchor="middle" fill="#6b7a90">argmax 回喂 → 第 k 步</text>
 <path d="M288,96 C288,116 150,116 83,86" fill="none" stroke="#2457d6" stroke-width="1.6"
   stroke-dasharray="5 4" marker-end="url(#ar)"/>
 <text x="188" y="124" font-size="10.5" text-anchor="middle" fill="#6b7a90">回喂（无 teacher forcing）</text>
 <line x1="380" y1="57" x2="414" y2="57" stroke="#2457d6" stroke-width="2" marker-end="url(#ar)"/>
 <rect x="418" y="30" width="234" height="54" rx="8" fill="#fff" stroke="#e3e8ef"/>
 <text x="535" y="52" font-size="12" text-anchor="middle" fill="#16202e">margin = logit(t_target) − logit(t_competitor)</text>
 <text x="535" y="70" font-size="11" text-anchor="middle" fill="#6b7a90">再比 B4 加性预测（held-out）</text>
</svg>
</div>

<h2>2 · 预注册（观测前冻结）</h2>
<div class="card">
<div class="two">
<div><div class="small">面板族（与 E_read 同一 held-out 族）</div>
<div><b>41 实体 × 6 类 = 246 pairs × 3 模板 = 738 行</b><br>
<span class="small">S1 seeds [7,8,9] · frac 0.2 ⇒ 49 test pairs × 3 = <b>147 held-out 行/seed</b></span></div></div>
<div><div class="small">K / 预测器 / 单位</div>
<div><b>K = @KFULL@</b>（报告 k=0..K；SMOKE 用 K=@K@）<br>
<span class="small">B4 = ridge(one-hot[entity]+[class]+[template]+bias, λ=1e-3)<br>
主体单位 = <b>原始 L1（logit，不归一）</b></span></div></div>
</div>
<p class="small" style="margin:12px 0 0">装置门：D1 k=0 确定性 · D2 margin 非退化 · D3 全有限 · S1 <code>max(k≥1) E_ar ≥ @S1THR@</code>。
冻结时刻 <b>无任何模型观测</b>；复核 V3 已证 held-out 折指纹与 Q03 / E_read <b>逐项相同</b>。</p>
</div>

<h2>3 · SMOKE 曲线（@CELLS@ cells · @FWD@ forwards）</h2>
<div class="card">
<svg viewBox="0 0 @W@ @H@" width="@W@" style="max-width:@W@px;display:block;margin:0 auto">
 <line x1="@PAD@" y1="@HPAD@" x2="@WPAD@" y2="@HPAD@" stroke="#c9d3e0" stroke-width="1.5"/>
 @GRID@
 @POLY_C@@POLY_E@@DOTS_C@@DOTS_E@
</svg>
<div class="legend" style="justify-content:center;margin-top:6px">
 <span><i class="dot" style="background:#2457d6"></i>E_ar(B4) —— 加性模型 held-out 误差</span>
 <span><i class="dot" style="background:#c0392b"></i>null(const) —— 训练均值常数预测</span>
</div>
<table style="margin-top:12px">
<thead><tr><th>k</th><th>E_ar(B4)</th><th>null(const)</th><th>scale</th><th>rel = err/scale</th><th>drift(k)−drift(0)</th></tr></thead>
<tbody>@ROWS_CURVE@</tbody></table>
<p class="small" style="margin:8px 0 0">B4 相对 null 稳定更低（k=0：@E0@ vs @C0@）⇒ 加性族确有信息，但相对误差 <code>rel</code> ≈ @RLMIN@–@RLMAX@，远不满足 5% 门。</p>
</div>

<h2>4 · per-seed（原始 L1）</h2>
<div class="card"><table>
<thead><tr><th>seed</th>@TH_K@</tr></thead><tbody>@ROWS_SEED@</tbody></table>
<p class="small" style="margin:8px 0 0">per-seed 离散度大是 <b>SMOKE 子面板伪影</b>（见 §6），非模型性质。</p></div>

<h2>5 · 装置门</h2>
<div class="card"><table>
<thead><tr><th>门</th><th>判据</th><th>结果</th></tr></thead><tbody>
<tr><td>D1</td><td>k=0 逐位确定性（同 cell 重跑）</td><td>@D1@</td></tr>
<tr><td>D2</td><td>margin 非退化（held-out std=@D2STD@）</td><td>@D2@</td></tr>
<tr><td>D3</td><td>全部 E_ar(k) 有限</td><td>@D3@</td></tr>
<tr><td>S1</td><td>max(k≥1) E_ar ≥ @S1THR@（实测 @S1MAX@）</td><td>@S1@</td></tr>
</tbody></table>
<p class="small" style="margin:8px 0 0">三次独立运行 <code>res_sha8</code> 恒为 <b>@RS@</b>、per-seed 数字逐位相同 ⇒ 装置确定性成立。</p></div>

<h2>6 · 两条必须说清的边界</h2>
<div class="note">
<p style="margin:0 0 8px"><b>① 阈值量纲体检（诚实处置，不追溯改门）</b>：S1 阈 @S1THR@ 是 <b>原始 logit</b> 量纲，实测 max=@S1MAX@ 被超 <b>@S1X@×</b>
⇒ 该门「形式上可失败、实则必过」，<b>不具否证力</b>。按「阈值冻结后不得事后调门」，<b>保留原门</b>并降级为 <b>装置灵敏度门</b>；
另在 <b>Q05 观测前</b> 预注册相对门 <code>S_rel: min(k=1..K) E_ar/scale ≤ 0.05</code>，随 result 封存（<code>q05_prereg</code>）。</p>
<p style="margin:0"><b>② SMOKE 子面板伪影</b>：子面板取 <code>PAIRS[:60]</code> = <b>仅前 10 个实体</b> ⇒ 41 个 entity one-hot 中 <b>31 列为零</b>，
未见实体处 B4 回退到 class+template ⇒ E_ar 偏大、per-seed 方差大。<b>这些数字不可科学解读</b>，正式曲线属 Q05。</p>
</div>

<h2>7 · 下一步：Q05 与硬前置</h2>
<div class="card">
<ol>
<li><b>Q05 正式测量</b>：三模型 × 全面板 738 行 × k=0..@KFULL@；按预注册 <code>S_rel</code> 与 <code>drift</code> 形状判据判决 {linear, saturating, diverging}。</li>
<li><b>资源前置（须先决）</b>：<code>qwen3-14b</code> bf16 ≈ <b>29.6 GB</b>、<code>glm4-9b</code> ≈ <b>18.8 GB</b>，均 &gt; 本机 <b>16 GB</b> 显存
（<code>qwen3-4b</code> 8.1 GB 可原样跑）。Q05 必须先定量化 / offload 方案，并<b>显式声明</b>其与 E_read（bf16 采集）之间的精度差异。</li>
<li><b>量纲不可比</b>：E_ar 为原始 logit L1，E_read 为归一化 MSE；并排读时只能用 <code>rel</code> 作桥。</li>
</ol>
</div>

<p class="small" style="margin-top:26px;color:#6b7a90">
装置脚本 <code>tests/deepseek/q04_e_ar_device.py</code> · 结果 <code>q04_smoke_result.json</code> ·
预注册 <code>q04_smoke_execution.json</code> · 复核 <code>verify_q04.txt</code> ·
口径 <code>research/deepseek/atlas/metric_dict.json</code> (v3 @MDSHA@) · 日志 <code>AGI_DEEPSEEK_MEMO.md</code> Phase 38
</p>
</div></body></html>
"""

html = TPL
for k, v in REPL.items():
    html = html.replace(k, v)
for k in REPL:
    assert k not in html, 'placeholder residue: ' + k

open(OUT, 'w', encoding='utf-8', newline='\n').write(html)
print('HTML ->', OUT, len(html.encode('utf-8')), 'B  sha8',
      hashlib.sha256(open(OUT, 'rb').read()).hexdigest()[:8])
print('DONE')
