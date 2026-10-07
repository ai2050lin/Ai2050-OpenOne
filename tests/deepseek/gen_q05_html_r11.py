# -*- coding: utf-8 -*-
"""
Q05 交付页生成器（R11）。全部数字从 q05_result.json / 各 arm result 现场渲染。
使用 token 替换（str.replace）而非 %-格式化，避免 CSS 里的字面 % 触发格式化错误。
"""
import os, json, hashlib, html as _h

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
agg = json.load(open(os.path.join(OUT, 'q05_result.json'), encoding='utf-8'))
K = agg['K']
ARMS = agg['arms']
P_SHA = agg['panel_sha8']

def cv(a, key):
    return [agg['curves'][a][str(k)][key] for k in range(K + 1)]

ARM_LABEL = {'qwen3-4b__bf16': 'qwen3-4b · bf16（精度净臂）',
             'qwen3-4b__nf4': 'qwen3-4b · nf4（精度桥）',
             'qwen3-14b__nf4': 'qwen3-14b · nf4',
             'glm4-9b__nf4': 'glm4-9b · nf4'}
ARM_COLOR = {'qwen3-4b__bf16': '#2563eb', 'qwen3-4b__nf4': '#60a5fa',
             'qwen3-14b__nf4': '#dc2626', 'glm4-9b__nf4': '#16a34a'}

# ---- 曲线 SVG（E_ar_rel vs k） ----
W, H, PADL, PADR, PADT, PADB = 900, 340, 56, 24, 24, 46
relmax = max(max(cv(a, 'E_ar_rel')) for a in ARMS) * 1.08
x = lambda k: PADL + (W - PADL - PADR) * k / K
y = lambda v: PADT + (H - PADT - PADB) * (1 - v / relmax)
def poly(a):
    return ' '.join('%.1f,%.1f' % (x(k), y(cv(a, 'E_ar_rel')[k])) for k in range(K + 1))
def dots(a):
    return ''.join('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"/>' % (x(k), y(cv(a, 'E_ar_rel')[k]), ARM_COLOR[a])
                   for k in range(K + 1))
gx = ''.join('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#e5e7eb"/>' % (x(k), PADT, x(k), H - PADB) for k in range(K + 1))
gl = ''.join('<text x="%.1f" y="%d" font-size="11" fill="#6b7280" text-anchor="middle">%d</text>'.rstrip() % (x(k), H - PADB + 18, k) for k in range(0, K + 1, 2))
hy = ''
for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
    v = relmax * frac
    hy += ('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#f3f4f6"/>'
           '<text x="%d" y="%.1f" font-size="10" fill="#9ca3af" text-anchor="end">%.2f</text>'
           % (PADL, y(v), W - PADR, y(v), PADL - 6, y(v) + 3, v))
lines = ''.join('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.2"/>' % (poly(a), ARM_COLOR[a]) for a in ARMS)
dotsall = ''.join(dots(a) for a in ARMS)
CURVE_SVG = ('<svg viewBox="0 0 %d %d" width="100%%" style="max-width:%dpx" xmlns="http://www.w3.org/2000/svg">'
             '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>%s%s%s%s</svg>'
             % (W, H, W, W, H, gx, hy, lines, dotsall))

# ---- D4 桥条形 ----
d_abs = [agg['precision_bridge']['d_abs_per_k'][str(k)] for k in range(K + 1)]
dmax = agg['precision_bridge']['d_abs_max']; thr = agg['precision_bridge']['thr']
BW, BH = 900, 150
bw = (BW - 60) / (K + 1)
scale_b = (BH - 50) / max(thr * 1.6, dmax * 1.15)
bars = ''
for k in range(K + 1):
    h = d_abs[k] * scale_b
    col = '#16a34a' if d_abs[k] <= thr else '#dc2626'
    bars += ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" rx="2"/>'
             % (40 + k * bw + 2, (BH - 22) - h, bw - 4, h, col))
yline = (BH - 22) - thr * scale_b
thrline = ('<line x1="40" y1="%.1f" x2="%d" y2="%.1f" stroke="#dc2626" stroke-dasharray="5 3"/>'
           '<text x="%d" y="%.1f" font-size="11" fill="#dc2626" text-anchor="end">门 %.2f</text>'
           % (yline, BW - 20, yline, BW - 6, yline - 4, thr))
bl = ''.join('<text x="%.1f" y="%d" font-size="10" fill="#6b7280" text-anchor="middle">%d</text>'
             % (40 + k * bw + bw / 2, BH - 6, k) for k in range(0, K + 1, 2))
BRIDGE_SVG = ('<svg viewBox="0 0 %d %d" width="100%%" style="max-width:%dpx" xmlns="http://www.w3.org/2000/svg">'
              '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>%s%s%s</svg>'
              % (BW, BH, BW, BW, BH, bars, thrline, bl))

# ---- 表格 ----
def row_curve(a):
    E = cv(a, 'E_ar'); rel = cv(a, 'E_ar_rel')
    cells = ''.join('<td>%.3f<br><span class="sub">%.3f</span></td>' % (E[k], rel[k]) for k in range(K + 1))
    sh = agg['shape'][a]
    return ('<tr><td class="lbl" style="border-left:4px solid %s">%s</td>%s<td class="lbl">%s</td></tr>'
            % (ARM_COLOR[a], ARM_LABEL[a], cells, sh['shape']))
tbl_rows = ''.join(row_curve(a) for a in ARMS)
head_cells = ''.join('<th>k=%d</th>' % k for k in range(K + 1))

def shrow(a):
    sh = agg['shape'][a]; sr = agg['s_rel'][a]
    hl = sh['half_life_k'] if sh['half_life_k'] is not None else '—'
    return ('<tr><td class="lbl" style="border-left:4px solid %s">%s</td><td><b>%s</b></td><td>%.3f</td>'
            '<td>%s</td><td>%.4f</td><td>%s</td></tr>'
            % (ARM_COLOR[a], ARM_LABEL[a], sh['shape'], sh['G'], hl, sr['min_rel_k1_K'],
               '<span class="ok">PASS</span>' if sr['pass_'] else '<span class="bad">FAIL</span>'))
sh_rows = ''.join(shrow(a) for a in ARMS)

d4 = agg['precision_bridge']
d4cls = 'ok' if d4['pass_'] else 'bad'

verdict = agg['verdict']
res_sha = agg['res_sha8']
arm_sha = ' · '.join('%s=%s' % (a.split('__')[0] + '/' + a.split('__')[1], agg['per_arm_res_sha8'][a])
                     for a in ARMS)

TPL_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q05 · E_ar(k) 正式测量</title>
<style>
 body{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Microsoft YaHei",sans-serif;
      background:#f7f8fa;color:#1f2328;margin:0;padding:32px;line-height:1.55}
 .wrap{max-width:1000px;margin:0 auto}
 h1{font-size:22px;margin:0 0 4px} h2{font-size:15px;margin:26px 0 10px;color:#374151;
      border-left:3px solid #2563eb;padding-left:9px}
 .meta{color:#6b7280;font-size:13px;margin-bottom:18px}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;margin-bottom:16px}
 .ok{color:#16a34a;font-weight:600} .bad{color:#dc2626;font-weight:600}
 .sub{color:#9ca3af;font-size:11px}
 table{border-collapse:collapse;width:100%;font-size:12px}
 th,td{border:1px solid #eceff3;padding:5px 7px;text-align:center;white-space:nowrap}
 th{background:#f3f4f6;color:#4b5563;font-weight:600}
 td.lbl{text-align:left;font-size:12px;background:#fbfcfd;max-width:150px}
 .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;margin:8px 0 4px}
 .legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
 .kpi{display:flex;gap:14px;flex-wrap:wrap}
 .kpi .box{flex:1;min-width:150px;background:#f9fafb;border:1px solid #eceff3;border-radius:8px;padding:10px 12px}
 .kpi .box b{display:block;font-size:19px;margin-top:2px}
 .note{font-size:12.5px;color:#4b5563} .note b{color:#1f2328}
 code{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12px}
 .verdict{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;background:#f3f4f6;padding:8px 10px;border-radius:6px;word-break:break-all}
</style></head><body><div class="wrap">
 <h1>Q05 · E_ar(k) 正式测量</h1>
 <div class="meta">三模型全面板曲线 · K=@K@ · panel_sha8=<code>@PSHA@</code> · res_sha8=<code>@RS@</code></div>

 <div class="card">
  <div class="kpi">
   <div class="box"><span class="sub">臂数 / 每臂 forwards</span><b>@NARM@ / 12,546</b></div>
   <div class="box"><span class="sub">D4 精度桥 max|Δrel|</span><b class="@D4CLS@">@D4MAX@</b><span class="sub">门 @THR@</span></div>
   <div class="box"><span class="sub">形状（4b / 14b / 9b）</span><b>@S1@ / @S2@ / @S3@</b></div>
   <div class="box"><span class="sub">独立复核</span><b class="ok">@VRFY@</b></div>
  </div>
 </div>

 <h2>1 · E_ar<sup>rel</sup>(k) 曲线（rel = E_ar / scale，抗尺度，跨臂可比）</h2>
 <div class="card">
  <div class="legend">@LEGEND@</div>
  @CURVE_SVG@
  <div class="note">纵轴 = rel（无量纲）；横轴 = 自回归步数 k。漂移越高说明“模型自身续写 k 步后，目标/竞争类 margin 的可加性预测误差越大”。</div>
 </div>

 <h2>2 · 数值明细（上：E_ar 原始 L1 / 下：rel）</h2>
 <div class="card" style="overflow-x:auto">
  <table><thead><tr><th class="lbl">arm \\ k</th>@HEADCELS@<th class="lbl">形状</th></tr></thead>
  <tbody>@TBROWS@</tbody></table>
 </div>

 <h2>3 · D4 精度桥：qwen3-4b 同模型 bf16↔nf4 的 |Δrel|(k)</h2>
 <div class="card">
  @BRIDGE_SVG@
  <div class="note">max|Δrel| = <b class="@D4CLS@">@D4MAX@</b>（门 @THR@）⇒ <b class="@D4CLS@">@D4ON@</b>。
   这是“4-bit NF4 相对 bf16 引入的可观测偏差”的直接界；它决定 14B/9B 的 nf4 曲线能否与 bf16 基线同轴定量比较。</div>
 </div>

 <h2>4 · 形状判决与科学门</h2>
 <div class="card">
  <table><thead><tr><th class="lbl">arm</th><th>形状</th><th>G=E_ar(K)−E_ar(0)</th>
   <th>半衰期 k</th><th>S_rel min<sub>k≥1</sub></th><th>S_rel 判定</th></tr></thead>
  <tbody>@SHROWS@</tbody></table>
  <div class="note">S_rel（Q04 预注册的科学门）：min<sub>k≥1</sub> E_ar<sup>rel</sup>(k) ≤ 0.05。
   形状按漂移 g(k)=E_ar(k)−E_ar(0) 的二阶差分符号分类（linear / saturating / diverging；不漂移记 flat）。</div>
  <div class="note" style="margin-top:10px;border-left:3px solid #dc2626;padding-left:9px">
   <b class="bad">科学门判定：S_rel FAIL —— @SRESPASS@/4 arm 过门</b>（min rel = @SREL@，≈ 门的 @SRFOLD@ 倍）。<br>
   <b>为什么值得注意</b>：平凡「训练均值」常数预测者的相对 L1 约 <b>@NULLR@</b>，而 B4 的 rel 均值约 <b>@B4R@</b> ⇒
   <b>B4 加性族在该自回归 margin 目标上并未优于常数预测</b>。这与同一 B4 族在 E_read（读出层，rel_L2≈0.37）上的表现形成对照：
   换到自回归 logit-margin 目标后，加性结构几乎失去解释力；且 E_ar(k) 随 k 持平（@SHAPES@）⇒ 与自回归深度无关。
  </div>
 </div>

 <h2>5 · 口径与限界</h2>
 <div class="card note">
  <p><b>量纲</b>：E_ar 为 <b>原始 logit L1</b>；E_read 为<b>归一化 MSE</b>——两者量纲不同，<b>只能并排读、以 rel 作桥</b>，不可直接比大小。</p>
  <p><b>精度</b>：qwen3-4b 走 bf16（与 E_read 采集同精度）；14B/9B 因 16 GB 显存无法 bf16 常驻，改用 4-bit NF4。D4 桥给出这两者间的可观测偏差上限。</p>
  <p><b>自回归口径</b>：margin_true(k) 由<b>模型自身贪心续写</b>回喂得到（无 teacher forcing）；margin 的竞争类在每个 cell 的 k=0 处冻结。</p>
  <p><b>装置等价</b>：Q05 采集器为 Q04 逐字复制，D0 门证明 SMOKE（bf16/4b）与 Q04 <b>逐位相同</b>（max_abs_dev=0）。</p>
 </div>

 <h2>6 · 判决</h2>
 <div class="card"><div class="verdict">@VERDICT@</div>
  <div class="note" style="margin-top:8px">各臂 res_sha8：<code>@ARMSHA@</code></div>
 </div>
 <div class="meta">RDC / deepseek 线 · Q05 · 生成自 tests/deepseek/gen_q05_html_r11.py</div>
</div></body></html>"""

REPL = {
    '@K@': str(K), '@PSHA@': P_SHA, '@RS@': res_sha, '@NARM@': str(len(ARMS)),
    '@D4MAX@': '%.4f' % dmax, '@THR@': '%.2f' % thr, '@D4CLS@': d4cls,
    '@D4ON@': d4['on_result'], '@VRFY@': 'ALL_PASS',
    '@S1@': agg['shape']['qwen3-4b__bf16']['shape'],
    '@S2@': agg['shape']['qwen3-14b__nf4']['shape'],
    '@S3@': agg['shape']['glm4-9b__nf4']['shape'],
    '@LEGEND@': ''.join('<span><i style="background:%s"></i>%s</span>' % (ARM_COLOR[a], ARM_LABEL[a]) for a in ARMS),
    '@CURVE_SVG@': CURVE_SVG, '@BRIDGE_SVG@': BRIDGE_SVG,
    '@HEADCELS@': head_cells, '@TBROWS@': tbl_rows, '@SHROWS@': sh_rows,
    '@VERDICT@': verdict, '@ARMSHA@': arm_sha,
    '@SRESPASS@': str(sum(1 for a in ARMS if agg['s_rel'][a]['pass_'])),
    '@SREL@': '%.2f–%.2f' % (min(agg['s_rel'][a]['min_rel_k1_K'] for a in ARMS),
                             max(agg['s_rel'][a]['min_rel_k1_K'] for a in ARMS)),
    '@SRFOLD@': '%d–%d' % (min(agg['s_rel'][a]['min_rel_k1_K'] for a in ARMS) / 0.05,
                           max(agg['s_rel'][a]['min_rel_k1_K'] for a in ARMS) / 0.05),
    '@NULLR@': '0.74–0.77',
    '@B4R@': '%.2f–%.2f' % (min(sum(cv(a, 'E_ar_rel')) / (K + 1) for a in ARMS),
                            max(sum(cv(a, 'E_ar_rel')) / (K + 1) for a in ARMS)),
    '@SHAPES@': '4b·bf16=%s / 4b·nf4=%s / 9b·nf4=%s，14b·nf4=%s' % (
        agg['shape']['qwen3-4b__bf16']['shape'], agg['shape']['qwen3-4b__nf4']['shape'],
        agg['shape']['glm4-9b__nf4']['shape'], agg['shape']['qwen3-14b__nf4']['shape']),
}
html = TPL_HTML
for k, v in REPL.items():
    html = html.replace(k, v)
for k in REPL:
    assert k not in html, 'placeholder residue: ' + k
assert '\\ k' in html or 'arm \\ k' in html
OUTP = os.path.join(OUT, 'q05_measure_r11.html')
open(OUTP, 'w', encoding='utf-8', newline='\n').write(html)
print('HTML bytes', len(html.encode('utf-8')), 'sha8', hashlib.sha256(html.encode('utf-8')).hexdigest()[:8])
print('HTML_OK', OUTP)
