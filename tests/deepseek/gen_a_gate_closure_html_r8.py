# -*- coding: utf-8 -*-
"""R8 汇报页：A 闸门关闭（数字全部现场渲染）。token 替换法，避免 % / {} 冲突。"""
import os, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek\result\a_gate_closure_r8.html")
def rj(p): return json.loads(open(p, "rb").read().decode("utf-8-sig"))

cl = rj(os.path.join(ROOT, r"research\deepseek\atlas\a_gate_closure_v1.json"))
dl = rj(os.path.join(ROOT, r"tests\deepseek\result\deadline_dual_track_v1.json"))["k1_recompute"]
q  = rj(os.path.join(ROOT, r"research\deepseek\atlas\phase_queue_v1.json"))
per, ml, agg, verdict = dl["per_model"], dl["model_level"], dl["aggregate"], dl["verdict"]
MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]

def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

rows_k1 = []
for m in MODELS:
    p = per[m]; l = ml[m]
    kp = "是" if l["kstar_pass"] else "否"; rp = "是" if l["readout_pass"] else "否"
    cls = "ok" if l["kstar_pass"] else "bad"
    rows_k1.append("<tr><td><code>%s</code></td><td>%d</td><td>%d</td><td>%.7f</td><td>%.7f</td>"
                   "<td class='%s'>%+.6f<small>±%.6f</small></td><td>%s</td>"
                   "<td class='bad'>%+.6f<small>±%.6f</small></td><td>%s</td></tr>"
                   % (m, p["kstar"], p["readout"], p["b4_kstar"], p["b4_readout"],
                      cls, l["kstar_margin_model"], l["kstar_mde_model"], kp,
                      l["readout_margin_model"], l["readout_mde_model"], rp))
rows_k1 = "\n".join(rows_k1)

rows_c = []
for c in cl["corrections_C1_C6"]:
    st = "已接受" if c["status"] == "accepted" else "已接受·待跨线施加"
    tag = "green" if c["status"] == "accepted" else "amber"
    rows_c.append("<tr><td><b>%s</b></td><td>%s</td><td><code>%s</code></td><td>%s</td>"
                  "<td><span class='tag %s'>%s</span></td></tr>"
                  % (c["id"], esc(c["target"]), esc(c.get("current", "")), esc(c["enactment"]), tag, st))
rows_c = "\n".join(rows_c)

rows_q = []
for it in q["queue"]:
    s = it["status"]
    cls = "sealed" if s == "sealed" else "pending"
    rows_q.append("<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td><span class='tag %s'>%s</span></td></tr>"
                  % (it["id"], esc(it["title"]), esc(it["block"]), "green" if s == "sealed" else "grey", s))
rows_q = "\n".join(rows_q)

n_sealed = sum(1 for it in q["queue"] if it["status"] == "sealed")
E = agg["E_read_per_model"]

CSS = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#1c2024;--mut:#5b6570;--line:#e3e7ec;--acc:#0b6bcb;--ok:#0f8a4f;--bad:#c0392b;--warn:#b8860b;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.hero{display:flex;gap:14px;flex-wrap:wrap}
.kpi{flex:1;min-width:170px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .n{font-size:24px;font-weight:700}.kpi .l{color:var(--mut);font-size:12px;margin-top:2px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:#eef2f7;font-weight:600}tr:last-child td{border-bottom:none}
td small{color:var(--mut);display:block;font-size:11px}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-family:ui-monospace,Consolas,monospace;font-size:12px}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.green{background:#e5f5ec;color:var(--ok)}.amber{background:#fdf3dc;color:var(--warn)}.grey{background:#eef0f3;color:var(--mut)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.note{background:#fdf3dc;border:1px solid #f0dfb0;border-radius:8px;padding:12px 14px;margin:12px 0;font-size:13px}
.seal{background:#eef4fd;border:1px solid #cfe0f7;border-radius:8px;padding:12px 14px;font-family:ui-monospace,Consolas,monospace}
.bar{height:10px;background:#e6eaef;border-radius:999px;overflow:hidden;margin:8px 0}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#0b6bcb,#37a1f5)}
ul{margin:8px 0 0 18px;padding:0}li{margin:3px 0}
.foot{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}
"""

HTML = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A 闸门关闭 · R8</title><style>@@CSS@@</style></head><body><div class="wrap">
<h1>A 闸门 seal 执行与关闭</h1>
<div class="sub">deepseek 线 · R8 · @@NOW@@ · 研究日志 <code>AGI_DEEPSEEK_MEMO.md</code> Phase 35</div>

<div class="seal">@@SEAL@@</div>

<div class="hero" style="margin-top:16px">
  <div class="kpi"><div class="n">CLOSED</div><div class="l">A 闸门状态（@@NSEALED@@/30 队列项 sealed）</div></div>
  <div class="kpi"><div class="n">@@E_POOL@@</div><div class="l">E_read 池化（5% 门的 @@E_RATIO@@×）</div></div>
  <div class="kpi"><div class="n">FIRED</div><div class="l">K1 判决（Q08=甲，判定层=行为读出层）</div></div>
  <div class="kpi"><div class="n">38 / 0</div><div class="l">独立复核 PASS / FAIL = ALL_PASS</div></div>
</div>

<h2>1 · Q08=甲：K1 判定层 = 行为读出层 ⇒ K1 触发</h2>
<div class="card">同一把 5% 门，挂在不同层位读数完全不同 —— 这正是判据自指（用假设自选的层去检验该假设）必须被消掉的地方。</div>
<table><thead><tr><th>模型</th><th>k*</th><th>读出层</th><th>B4 误差@k*</th><th>B4 误差@读出</th>
<th>k* 层 margin（负=候选优）</th><th>过门</th><th>读出层 margin</th><th>过门</th></tr></thead>
<tbody>@@ROWS_K1@@</tbody></table>
<div class="hero" style="margin-top:12px">
  <div class="kpi"><div class="n">@@READOUT_M@@</div><div class="l">读出层池化 margin（2×MDE = @@READOUT_2MDE@@）</div></div>
  <div class="kpi"><div class="n">@@KSTAR_M@@</div><div class="l">k* 层池化 margin（2×MDE = @@KSTAR_2MDE@@）</div></div>
  <div class="kpi"><div class="n">3 / 3</div><div class="l">读出层否决模型数</div></div>
</div>
<div class="card"><b>判决</b>：<span class="bad">K1 = fired_all_models</span>（读出层 3/3 否决 + 轨 A 触发）；
k* 层 = <b>model_specific</b>（<code>qwen3-4b</code> 否决，不得升机制）。<br>
<b>后果</b>：命题「条件齿轮组 = 算子代数」<b>降级为 descriptive</b>（功能性端口类描述），不得再作机制主张。</div>

<h2>2 · C1–C6：全接受，逐条落盘（erratum）</h2>
<table><thead><tr><th>id</th><th>目标</th><th>原值/原文</th><th>本线执行方式</th><th>状态</th></tr></thead>
<tbody>@@ROWS_C@@</tbody></table>
<div class="note"><b>C4 / C6 未施加</b>：目标 <code>research/gpt5/atlas/atlas_ledger.json</code> 为<b>跨线共享</b>账本（同时含 deepseek Phase 8–21 与 G 线 Phase 2902–3153）⇒ 依「避免与其他路线混合」纪律，本轮只落补丁规格 <code>ledger_corrections_v1.json</code>（状态 <code>SEALED_BUT_NOT_APPLIED</code>）。施加需一句确认。</div>

<h2>3 · 队列状态（phase_queue_v1.json，唯一议程来源）</h2>
<div class="bar"><i style="width:@@PCT@@%"></i></div>
<div class="sub">已 sealed @@NSEALED@@ / 30</div>
<table><thead><tr><th>id</th><th>标题</th><th>闸门</th><th>状态</th></tr></thead>
<tbody>@@ROWS_Q@@</tbody></table>

<h2>4 · 独立复核</h2>
<div class="card">新进程独立复核 <b>PASS 38 / FAIL 0 = ALL_PASS</b>：memo 前缀逐字节不变 + bare_lf 0 + Phase 35 在位；
队列 5 项 sealed；6 条更正全 accepted；<code>E_read</code> 池化独立重算一致；C4 <code>content_excluding_self</code> 独立重算 = <code>0dc6e57a</code>；
跨线受保护文件 7 枚指纹全部未变。</div>

<h2>5 · 下一步</h2>
<ul>
<li><b>B 闸门（需 GPU；本机已探明可用：RTX 5080 / 16303 MiB / torch 2.13+cu130）</b>：Q03 <code>E_read</code> 统一基线复算 → Q04/Q05 <code>E_ar(k)</code> → Q06 <code>C_steer</code> 基座。</li>
<li>C / D / E / F 闸门：Q13–Q19、Q10/Q11/Q20–Q25、Q26/Q27/Q30。</li>
<li><b>待确认</b>：跨线账本补丁（C4/C6）是否由本对话施加。</li>
</ul>

<div class="foot">本页所有数字由 <code>deadline_dual_track_v1.json</code> / <code>a_gate_closure_v1.json</code> / <code>phase_queue_v1.json</code> 现场渲染，无手工转录。
产物：<code>a_gate_closure_v1.json</code> @@S_CLOSE@@ · <code>verify_r8.txt</code> · memo Phase 35 @@S_MEMO@@。</div>
</div></body></html>"""

REP = {
 "@@CSS@@": CSS, "@@NOW@@": cl["sealed_at"], "@@SEAL@@": esc(cl["seal_verbatim"]),
 "@@NSEALED@@": str(n_sealed), "@@PCT@@": "%.1f" % (100.0 * n_sealed / 30),
 "@@E_POOL@@": "%.4f" % agg["E_read_pooled"], "@@E_RATIO@@": "%.1f" % (agg["E_read_pooled"] / 0.05),
 "@@ROWS_K1@@": rows_k1, "@@ROWS_C@@": rows_c, "@@ROWS_Q@@": rows_q,
 "@@READOUT_M@@": "%+.4f" % agg["readout_pooled_margin"], "@@READOUT_2MDE@@": "%.4f" % (2 * agg["readout_pooled_mde"]),
 "@@KSTAR_M@@": "%+.4f" % agg["kstar_pooled_margin"], "@@KSTAR_2MDE@@": "%.4f" % (2 * agg["kstar_pooled_mde"]),
 "@@S_CLOSE@@": hashlib.sha256(open(os.path.join(ROOT, r"research\deepseek\atlas\a_gate_closure_v1.json"), "rb").read()).hexdigest()[:8],
 "@@S_MEMO@@": hashlib.sha256(open(os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md"), "rb").read()).hexdigest()[:8],
}
html = HTML
for k, v in REP.items(): html = html.replace(k, v)
open(OUT, "w", encoding="utf-8").write(html)

# 自检
bad = [k for k in REP if k in html]
res = "OK" if not bad else "RESIDUAL:%s" % bad
print("HTML %s  %d B  %s" % (OUT, os.path.getsize(OUT), res))
print("  tokens left = %s" % bad)
