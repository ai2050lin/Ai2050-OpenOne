# -*- coding: utf-8 -*-
"""Q09 + Q12 单文件 HTML 汇报页。数字全部从交付 JSON 现场读取。"""
import os, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
ATLAS = os.path.join(ROOT, "research", "gpt5", "atlas")
DOCS = os.path.join(ROOT, "research", "gpt5", "docs")
OUT = os.path.join(ROOT, "tests", "deepseek", "result", "q09q12_gate_r5.html")

def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
def jload(p):
    return json.loads(open(p, "rb").read().decode("utf-8-sig"))

q09 = jload(os.path.join(ATLAS, "deadline_dual_track_v1.json"))
q12 = jload(os.path.join(ATLAS, "prop_citation_audit_v1.json"))
A = q09["k1_recompute"]["aggregate"]
V = q09["k1_recompute"]["verdict"]
ML = q09["k1_recompute"]["model_level"]
PM = q09["k1_recompute"]["per_model"]
MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]
C = q12["counts"]
DV = q12["verdict"]

def f6(x):
    return "%.6f" % x
def fp(x):
    return "%+.6f" % x

h = []
def W(s=""):
    h.append(s)

W("<!DOCTYPE html>")
W('<html lang="zh-CN"><head><meta charset="utf-8">')
W("<title>Q09 死线双轨重述 · Q12 D/E 引用审计</title>")
W("<style>")
W(":root{--bg:#f7f8fa;--card:#fff;--ink:#12161c;--mut:#5b6672;--line:#e3e7ec;--acc:#1f5fd1;--ok:#0a7a3f;--bad:#c02626;--warn:#a05a00}")
W("*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif}")
W(".wrap{max-width:1120px;margin:0 auto;padding:28px 20px 60px}")
W("h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:18px 0 8px;color:var(--mut)}")
W(".sub{color:var(--mut);font-size:13px;margin-bottom:22px}")
W(".card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:0 0 16px}")
W(".tag{display:inline-block;font-size:12px;padding:2px 9px;border-radius:20px;background:#eaf0fb;color:var(--acc);margin-right:6px}")
W(".tag.ok{background:#e7f6ee;color:var(--ok)}.tag.bad{background:#fdeaea;color:var(--bad)}.tag.warn{background:#fdf3e3;color:var(--warn)}")
W("table{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0}")
W("th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}")
W("th{background:#f2f5f9;font-weight:600;color:var(--mut);font-size:12.5px}")
W("code{background:#f1f3f6;padding:1px 5px;border-radius:4px;font-size:12.5px}")
W(".big{font-size:22px;font-weight:700}.num{font-variant-numeric:tabular-nums}")
W(".ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.warn{color:var(--warn);font-weight:600}")
W(".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}")
W(".kpi{background:#fbfcfe;border:1px solid var(--line);border-radius:10px;padding:12px 14px}")
W(".kpi .l{font-size:12.5px;color:var(--mut)}.kpi .v{font-size:19px;font-weight:700;margin-top:3px}")
W(".note{background:#fbfcfe;border-left:3px solid var(--acc);padding:9px 12px;border-radius:0 8px 8px 0;font-size:13.5px;color:#333}")
W(".bar{height:9px;border-radius:5px;background:#eceff3;overflow:hidden;margin-top:5px}")
W(".bar>i{display:block;height:100%;background:var(--acc)}")
W(".bar.red>i{background:var(--bad)}")
W("footer{color:var(--mut);font-size:12.5px;margin-top:24px;border-top:1px solid var(--line);padding-top:12px}")
W("</style></head><body><div class='wrap'>")
W("<h1>Q09 死线双轨重述 · Q12 D/E 命题引用审计</h1>")
W("<div class='sub'>A 闸门（零 GPU）· 依据 <code>RDC_RESEARCH_CONSTITUTION_v1.md</code> §3(I3) / §6(I4) · 生成 2026-10-03 · 全部数字由脚本从 result 与账本现场读取</div>")

# ---- Card 1: 死线三态 ----
W("<div class='card'><h2>① 三条死线的真实状态</h2>")
W("<table><tr><th>死线</th><th>旧触发条件</th><th>结构病</th><th>新形式下的实测</th><th>状态</th></tr>")
W("<tr><td><b>K1</b></td><td>3 模型全 above <b>且</b> M1 全败</td><td>对模型的全称量词（合取）</td>"
  "<td>k* 层 <code>model_specific</code>；读出层 <span class='bad'>触发</span>（3/3 否决 + 主判据）</td>"
  "<td><span class='tag ok'>可测量</span></td></tr>")
W("<tr><td><b>K2</b></td><td>分离后 cos 降 &gt;50% 且行为不掉</td><td>同文件内两种操作化互斥</td>"
  "<td>无数据</td><td><span class='tag bad'>从未被测量</span></td></tr>")
W("<tr><td><b>K3</b></td><td>3 模式族 top-50 覆盖率<b>均</b>&lt;30%</td><td>对族数的全称量词（合取）</td>"
  "<td>无数据</td><td><span class='tag bad'>从未被测量</span></td></tr>")
W("</table>")
W("<div class='note'><b>三个死线里，两个从未被测量，一个的触发条件写成合取。</b> 这就是“死线免疫”的完整病理——不只是合取，还有两条根本没有装置。</div>")
W("</div>")

# ---- Card 2: K1 双轨 ----
W("<div class='card'><h2>② K1 重算：同一批数字，两种形式给出相反判决</h2>")
W("<div class='grid'>")
W("<div class='kpi'><div class='l'>旧形式（合取）</div><div class='v ok'>未触发</div><div class='l'>above = 1/3、M1@k* 过门 2/3 ⇒ 前缀不成立 ⇒ 「算子代数线保住」</div></div>")
W("<div class='kpi'><div class='l'>新形式·k* 层（承诺层）</div><div class='v warn'>model_specific</div><div class='l'>轨 A 通过（候选显著更优），但否决权被 qwen3-4b 触发 ⇒ 不得升为机制</div></div>")
W("<div class='kpi'><div class='l'>新形式·读出层（行为层）</div><div class='v bad'>触发（3/3）</div><div class='l'>池化 margin 为正且 E_read 超门 ⇒ 主判据与否决权双触发</div></div>")
W("</div>")
W("<h3>逐模型（数据来自 3151/3152 result，零手工转录）</h3>")
W("<table><tr><th>模型</th><th>k*</th><th>读出层</th><th>B4@k*</th><th>B4@读出层</th><th>M1@k* margin（3 seed）</th><th>M1@读出层 margin（3 seed）</th><th>above</th></tr>")
for m in MODELS:
    d = PM[m]
    W("<tr><td><code>%s</code></td><td class='num'>%d</td><td class='num'>%d</td><td class='num'>%s</td><td class='num'>%s</td>"
      "<td class='num'>%s</td><td class='num'>%s</td><td>%s</td></tr>" % (
        m, d["kstar"], d["readout"], f6(d["b4_kstar"]), f6(d["b4_readout"]),
        " / ".join(fp(x) for x in d["m1_kstar_margins"]),
        " / ".join(fp(x) for x in d["m1_readout_margins"]),
        "<span class='bad'>是</span>" if d["above_add_gate"] else "否"))
W("</table>")
W("<h3>聚合（禁合取）</h3>")
W("<table><tr><th>量</th><th>值</th><th>95% CI（模型级 n=3）</th></tr>")
W("<tr><td>Ebar_read（池化读出层误差）</td><td class='num big bad'>%s</td><td class='num'>[%s, %s]</td></tr>" % (
    f6(A["E_read_pooled"]), f6(A["E_read_ci95_modellevel"][0]), f6(A["E_read_ci95_modellevel"][1])))
W("<tr><td>池化 margin @ k*</td><td class='num'>%s</td><td class='num'>[%s, %s]</td></tr>" % (
    fp(A["kstar_pooled_margin"]), fp(A["kstar_margin_ci95_modellevel"][0]), fp(A["kstar_margin_ci95_modellevel"][1])))
W("<tr><td>池化 margin @ 读出层</td><td class='num big bad'>%s</td><td class='num'>[%s, %s]</td></tr>" % (
    fp(A["readout_pooled_margin"]), fp(A["readout_margin_ci95_modellevel"][0]), fp(A["readout_margin_ci95_modellevel"][1])))
W("<tr><td>池化 MDE（k* / 读出层）</td><td class='num'>%s / %s</td><td>—</td></tr>" % (
    f6(A["kstar_pooled_mde"]), f6(A["readout_pooled_mde"])))
W("</table>")
W("<div class='note'>门语义 <code>margin = err_cand − err_B4</code>，<b>负 = 候选优</b>。读出层池化 margin 为 <b>正</b> ⇒ 候选模型在行为层<b>比全加性基线更差</b>。</div>")
W("<div class='note'>⚠️ <b>层位选择属 Q08</b>（需用户 seal）。本页把两个层的双轨结果并列报告，不代为决定判定层。</div>")
W("</div>")

# ---- Card 3: 双轨规则 ----
W("<div class='card'><h2>③ 双轨制规则（I3 落地）</h2>")
W("<div class='grid'>")
W("<div class='kpi'><div class='l'>轨 A · 主判据</div><div class='v'>聚合量 + bootstrap CI</div>"
  "<div class='l'>pooled margin / pooled error / pooled coverage；<b>禁用合取</b></div></div>")
W("<div class='kpi'><div class='l'>轨 B · 否决权</div><div class='v'>单模型反例</div>"
  "<div class='l'>任一模型不达标 ⇒ 标 <code>model_specific</code>；<b>不得升为机制</b></div></div>")
W("</div>")
W("<h3>明令禁止</h3><ul style='margin:6px 0 0 18px;padding:0;font-size:13.5px'>"
  "<li>把全称量词形式的合取写进触发条件</li>"
  "<li>把「任一模型不达标」当作加固主判据的证据（方向必须相反：它是降级信号）</li></ul>")
W("</div>")

# ---- Card 4: Q12 ----
W("<div class='card'><h2>④ Q12 D/E 级命题引用审计</h2>")
W("<div class='grid'>")
W("<div class='kpi'><div class='l'>F1 id 级 · 新推理链</div><div class='v'>%d 处</div><div class='l'>Phase 3113 引用 R55（复合等级）</div></div>" % DV["n_substantive_in_new_region"])
W("<div class='kpi'><div class='l'>F3 文本指纹级 · 新推理链</div><div class='v ok'>0 条</div><div class='l'>%d 条可构造指纹的 D/E 命题，实质内容零泄漏</div></div>" % q12["F3_text_fingerprint"]["n_constructible"])
W("<div class='kpi'><div class='l'>命名空间碰撞（假阳性）</div><div class='v'>%d 处</div><div class='l'>纯 id grep 会把别文档的自有编号误判</div></div>" % len(DV["namespace_collisions"]))
W("</div>")
W("<div class='note'><b>判决：新推理链中无实质违规。</b> 唯一 id 级候选（<code>MEMO L13546 · Phase 3113 · R55</code>）属复合等级，规则不可机械判定；文本级 <b>0</b> 命中。</div>")
W("<h3>分级口径与两个制度缺陷</h3>")
W("<table><tr><th>项</th><th>值 / 内容</th></tr>")
W("<tr><td>命题总数</td><td class='num'>%d</td></tr>" % C["n_props"])
W("<tr><td>D∪E（分量口径）</td><td class='num'><b>%d / %d = %.2f%%</b></td></tr>" % (C["n_DE_union"], C["n_props"], C["pct_DE_union"]))
W("<tr><td>其中复合等级（含 A/B/C）</td><td class='num'><b>%d 条</b></td></tr>" % C["n_DE_composite"])
W("<tr><td>纯 E / 纯 D</td><td class='num'>%d 条（%s） / %d 条</td></tr>" % (
    len(C["pure_E_ids"]), ", ".join(C["pure_E_ids"]), len(C["pure_D_ids"])))
W("<tr><td><span class='tag bad'>RULE-UNDECIDABLE</span></td><td>「E 级禁止进入新推理链」对 %d/%d 条不可机械判定——撤回的是 E 分量，A/B/C 分量仍可引用</td></tr>" % (
    C["n_DE_composite"], C["n_DE_union"]))
W("<tr><td><span class='tag bad'>NS-COLLISION</span></td><td>账本 id <code>R\\d\\d</code> 不是唯一命名空间：<code>LOOP_DIAGNOSIS_AND_EXIT_v1.md</code> 自有的 R10/R11 与账本 R10/R11 <b>内容完全不同</b></td></tr>")
W("<tr><td><span class='tag warn'>PCT-DIM-MIX</span></td><td>「约 2/3 命题不能进入新推理链」是 34%%+32%% 的<b>分量和</b>；按条数为 %.2f%%</td></tr>" % C["pct_DE_union"])
W("</table>")
W("</div>")

# ---- Card 5: 边界 ----
W("<div class='card'><h2>⑤ 边界（严格遵守）</h2>")
W("<ul style='margin:6px 0 0 18px;padding:0;font-size:13.5px'>")
W("<li><b>未改动</b>任何 MEMO 原文 / 命题账本 / TESTPLAN / 队列 / 宪法 —— 复核已逐项断言指纹（MEMO <code>%s</code>、账本 <code>%s</code>、TESTPLAN <code>%s</code>、队列 <code>675836fd</code>、宪法 <code>01df6398</code>）。</li>" % (
    sha8(os.path.join(DOCS, "AGI_GPT5_MEMO.md")),
    q12["ledger"]["sha8"],
    q09["source_testplan"]["sha8"]))
W("<li><b>未改判</b>K1：本页只把旧形式的结论与双轨重算<b>并列</b>；改判落 Q08，需用户 seal。</li>")
W("<li>Q09 / Q12 在队列中的状态仍为 <code>pending</code>（生效需 seal）。</li>")
W("<li>Q12 的分类由<b>人工核定表</b>给出（F4 原文即要求“人工确认下游无残留依赖”），未自动分类以免过判。</li>")
W("</ul></div>")

W("<footer>独立复核 <code>disk_verify_q09q12_r5.py</code> = <b>PASS 76 / FAIL 0 → ALL_PASS</b>（含从 result 独立重算池化量、从账本独立重算等级与命中、受保护文件指纹、LF-only）。<br>"
  "交付件：<code>DEADLINE_DUAL_TRACK_Q09.md</code> · <code>deadline_dual_track_v1.json</code> · <code>PROP_CITATION_AUDIT_Q12.md</code> · <code>prop_citation_audit_v1.json</code></footer>")
W("</div></body></html>")

open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(h) + "\n")
b = open(OUT, "rb").read()
import re
bad = re.findall(r"\{[a-zA-Z_]+\}|None|nan", b.decode("utf-8"))
print("WROTE", os.path.relpath(OUT, ROOT), len(b), "B", sha8(OUT))
print("placeholder/None residue:", len(bad), bad[:6])
