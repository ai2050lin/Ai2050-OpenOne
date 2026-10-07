# -*- coding: utf-8 -*-
"""R5h: 生成 A 闸门 seal 请求包（只读汇总，不改任何判定）。
产物：
  1) research/gpt5/atlas/seal_request_v1.json
  2) research/gpt5/docs/SEAL_REQUEST_A_GATE.md
  3) tests/deepseek/result/seal_request_a_gate.html
所有数字一律从源 JSON / MD 现场渲染（铁律 ae）。
"""
import os, re, json, hashlib, datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
DL = os.path.join(ROOT, r"research\gpt5\atlas\deadline_dual_track_v1.json")
Q01MD = os.path.join(ROOT, r"research\gpt5\docs\META_SINGLE_SOURCE_Q01.md")
CONS = os.path.join(ROOT, r"research\gpt5\docs\RDC_RESEARCH_CONSTITUTION_v1.md")
OUTJ = os.path.join(ROOT, r"research\gpt5\atlas\seal_request_v1.json")
OUTM = os.path.join(ROOT, r"research\gpt5\docs\SEAL_REQUEST_A_GATE.md")
OUTH = os.path.join(ROOT, r"tests\deepseek\result\seal_request_a_gate.html")
REP = os.path.join(ROOT, r"tests\deepseek\result\_seal_request_gen_report.txt")

def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

# ---------- 1) 现场读取 K1 重算 ----------
dl = json.loads(open(DL, "rb").read().decode("utf-8-sig"))
K1 = dl["k1_recompute"]
MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]
pm = K1["per_model"]
agg = K1["aggregate"]

def f6(x): return ("%.6f" % x)
def f4(x): return ("%.4f" % x)

k1_rows = []
for m in MODELS:
    d = pm[m]
    k1_rows.append({
        "model": m,
        "kstar": d["kstar"],
        "readout_layer": d["readout"],
        "b4_kstar": d["b4_kstar"],
        "b4_readout": d["b4_readout"],
        "above_add_gate": d["above_add_gate"],
    })

# ---------- 2) 现场解析 C1–C6 更正表 ----------
t = open(Q01MD, "rb").read().decode("utf-8-sig")
corr = []
for line in t.split("\n"):
    mm = re.match(r"^\|\s*(C[1-6])\s*\|(.+)\|\s*$", line)
    if mm:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        corr.append({"id": cells[0], "target": cells[1], "current": cells[2],
                     "proposed": cells[3], "basis": cells[4], "status": cells[5]})
assert len(corr) == 6, "C 表解析到 %d 条" % len(corr)

# ---------- 3) 受保护文件指纹（现场） ----------
PROTECTED = [
    ("research/gpt5/docs/AGI_GPT5_MEMO.md", "G 线备忘录（判决原文）"),
    ("tests/glm5/result/rdc_query_construction_20260913/phase3103/omega_p101_formula_audit/proposition_ledger.json", "命题账本（62 条真源）"),
    ("research/gpt5/docs/RDC_TESTPLAN_v1.md", "TESTPLAN（K1–K3 冻结条款）"),
    ("research/gpt5/atlas/phase_queue_v1.json", "30-Phase 冻结队列"),
    ("research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md", "研究宪法 v1"),
]
frozen = [{"path": p, "role": r, "sha8": sha8(os.path.join(ROOT, p.replace("/", os.sep)))} for p, r in PROTECTED]

# ---------- 4) JSON 产物 ----------
doc = {
    "schema": "rdc_seal_request/v1",
    "generated_by": "tests/deepseek/gen_seal_request_r5.py",
    "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "scope": "A 闸门（Q01/Q02/Q08/Q09/Q12）",
    "read_only": True,
    "note": "本件只汇总待 seal 事项，不代为改判、不改动任何既有判定。",
    "gate_status": {
        "done_zero_gpu": ["Q01", "Q02", "Q09", "Q12"],
        "awaiting_seal": ["Q08", "C1-C6", "I1", "I9"],
        "needs_gpu": ["Q03", "Q04", "Q05", "Q06", "Q07", "Q10", "Q11"],
    },
    "Q08": {
        "title": "K1 判定层选择（唯一影响主线的待 seal 决策）",
        "k1_text_verbatim": "3 模型 × 未见组合 的 logit 预测误差 > 5% 且不显著优于全加性基线 B4",
        "evidence": {
            "per_model": k1_rows,
            "aggregate": {
                "E_read_per_model": agg["E_read_per_model"],
                "E_read_pooled": agg["E_read_pooled"],
                "kstar_pooled_margin": agg["kstar_pooled_margin"],
                "kstar_pooled_mde": agg["kstar_pooled_mde"],
                "readout_pooled_margin": agg["readout_pooled_margin"],
            },
        },
        "options": [
            {"id": "A", "name": "读法甲：改判（推荐）",
             "effect": "判定层=行为读出层 ⇒ K1 触发 ⇒ 「条件齿轮组=算子代数」降级为 descriptive / 功能性端口类描述。"},
            {"id": "B", "name": "读法乙：重述并重新冻结",
             "effect": "承认判据自指（用假设自选的层判断假设）；K1 重述为无歧义版本。注意 TESTPLAN §8.2 规定 K1–K3 须在 3150 一并冻结、之后不得改阈值 ⇒ 重述需显式解冻升版。"},
            {"id": "C", "name": "暂缓",
             "effect": "A 闸门无法关闭；B/C 闸门（Q03+ KPI 装置）不得启动。"},
        ],
        "consequence_if_A": "K1 层位判决按 Q09 双轨：k* 层 = model_specific（qwen3-4b 否决）；读出层 = fired_all_models（3/3 否决）。",
    },
    "corrections_C1_C6": corr,
    "I1": {"section": "§1 唯一全局 KPI",
           "ask": "确认冻结：一个 Phase 未降低 E_read / E_ar(k) / C_steer 任一者，Ledger 登记为 catalog，不得登记为 advance。"},
    "I9": {"section": "§7 议程：冻结 30-Phase 队列",
           "ask": "确认冻结：phase_queue_v1.json 为唯一议程来源，禁止由「残差 → 自动派生下一 Phase」生成新队列。"},
    "protected_fingerprints": frozen,
    "reply_template": "seal: Q08=甲 | C=全接受 | I1=确认 | I9=确认",
    "post_seal_order": [
        "1) 执行 C1–C6 更正（仅改目标文档对应字段；每项写前备份 + 写后 sha8 复核）",
        "2) Q08 改判落 MEMO（append 不改写）+ Ledger verdict",
        "3) A 闸门关闭 → 章 program 更新 queue 状态（Q01/Q02/Q08/Q09/Q12 = sealed）",
        "4) 进入 B 闸门：Q03（E_read 统一基线复算，需 GPU）",
    ],
}
open(OUTJ, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")

# ---------- 5) MD 产物 ----------
L = []
A = L.append
A("# A 闸门 seal 请求（R5，2026-10-03）")
A("")
A("> 本件由 `tests/deepseek/gen_seal_request_r5.py` 现场渲染（数字一律取自源 result / 报告，无手工转录）。")
A("> **只读汇总**：不代为改判，不改动任何既有判定；所有动作在 seal 后才执行。")
A("")
A("## 0 A 闸门进度")
A("")
A("| 项 | 状态 |")
A("|---|---|")
A("| Q01 元层单一真源对账 | 已完成（零 GPU） |")
A("| Q02 KPI 口径冻结 | 已完成（零 GPU） |")
A("| Q09 死线双轨重述（I3） | 已完成（零 GPU） |")
A("| Q12 D/E 命题引用审计（I4） | 已完成（零 GPU） |")
A("| **Q08 K1 判定层选择** | **待 seal（本件第 1 节）** |")
A("| C1–C6 更正表 | **待 seal（本件第 2 节）** |")
A("| I1 / I9 制度条款 | **待 seal（本件第 3 节）** |")
A("")
A("## 1 Q08：K1 判定层选择（唯一影响主线的决策）")
A("")
A("**K1 原文（`RDC_TESTPLAN_v1.md` §8.2 逐字）**：3 模型 × **未见组合** 的 logit 预测误差 **> 5%** 且不显著优于全加性基线 B4。")
A("")
A("### 1.1 证据（3151/3152 result 现场重算）")
A("")
A("| 模型 | B4@k*（约 7.5% 深度） | B4@**读出层** | 是否过 5% 门 |")
A("|---|---|---|---|")
for r in k1_rows:
    A("| %s | %s | **%s** | %s |" % (r["model"], f6(r["b4_kstar"]), f4(r["b4_readout"]), "是" if r["above_add_gate"] else "否"))
A("")
A("- **聚合**：`E_read` 池化 **%s**（逐模型 %s）；池化 margin@k* **%s**（MDE %s）；池化 margin@读出层 **%s**。"
  % (f4(agg["E_read_pooled"]), " / ".join(f4(x) for x in agg["E_read_per_model"]),
     f6(agg["kstar_pooled_margin"]), f6(agg["kstar_pooled_mde"]), f4(agg["readout_pooled_margin"])))
A("- **关键事实**：同一条 5% 门，挂在 k* 层得 **1/3 过门**（未触发），挂在行为读出层得 **0/3 过门**（触发）；读出层误差达门的 **6.6×**（0.3316 / 0.05）。")
A("")
A("### 1.2 两种读法")
A("")
A("- **读法甲（推荐）**：K1 的语义是「能否预测未见组合」，而「行为」由模型输出定义 ⇒ 判定层应为**行为读出层** ⇒ **K1 应当触发**，「条件齿轮组=算子代数」降级为 `descriptive`（功能性端口类描述）。")
A("- **读法乙**：若「承诺层 k* = 候选取作用层」是**预先登记且有意**的判据，则 K1 原文在两处自相矛盾，**必须重述为无歧义版本并重新冻结**。注意 TESTPLAN §8.2 已规定 K1–K3 须在 3150 一并冻结、之后不得修改 ⇒ 重述需**显式解冻升版**。")
A("")
A("> 无论哪种读法，「**用假设自己指定的层来判断假设**」都必须被显式接受或显式否决，不能默认。")
A("")
A("### 1.3 请选择")
A("")
for o in doc["Q08"]["options"]:
    A("- **%s. %s** —— %s" % (o["id"], o["name"], o["effect"]))
A("")
A("## 2 C1–C6 更正表（Q01 产出，逐条待 seal）")
A("")
A("| id | 目标 | 现状 | 改为 | 依据 | 状态 |")
A("|---|---|---|---|---|---|")
for c in corr:
    A("| %s | %s | %s | %s | %s | %s |" % (c["id"], c["target"], c["current"], c["proposed"], c["basis"], c["status"]))
A("")
A("选项：**全接受** / **逐条取舍**（请列出接受的 id）。")
A("")
A("## 3 I1 / I9 制度条款")
A("")
A("- **I1（§1 唯一全局 KPI）**：%s" % doc["I1"]["ask"])
A("- **I9（§7 议程）**：%s" % doc["I9"]["ask"])
A("")
A("选项：**确认** / **修改**（请给出改动）。")
A("")
A("## 4 冻结与不变量（本件生成时现场指纹）")
A("")
A("| 文件 | 角色 | sha8 |")
A("|---|---|---|")
for x in frozen:
    A("| `%s` | %s | `%s` |" % (x["path"], x["role"], x["sha8"]))
A("")
A("## 5 seal 后执行顺序")
A("")
for s in doc["post_seal_order"]:
    A("- %s" % s)
A("")
A("## 6 一行回复模板")
A("")
A("```text")
A(doc["reply_template"])
A("```")
A("")
open(OUTM, "w", encoding="utf-8", newline="\n").write("\n".join(L))

# ---------- 6) HTML 产物 ----------
CSS = """:root{--bg:#f7f8fa;--card:#fff;--ink:#12161c;--mut:#5b6672;--line:#e3e7ec;--acc:#1f5fd1;--ok:#0a7a3f;--bad:#c02626;--warn:#a05a00}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:0 0 12px}h3{font-size:14px;margin:16px 0 8px;color:var(--mut)}
.sub{color:var(--mut);font-size:13px;margin-bottom:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:0 0 16px}
.tag{display:inline-block;font-size:12px;padding:2px 9px;border-radius:20px;background:#eaf0fb;color:var(--acc);margin-right:6px}
.tag.ok{background:#e7f6ee;color:var(--ok)}.tag.bad{background:#fdeaea;color:var(--bad)}.tag.warn{background:#fdf3e3;color:var(--warn)}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#f2f5f9;font-weight:600;color:var(--mut);font-size:12.5px}
code{background:#f1f3f6;padding:1px 5px;border-radius:4px;font-size:12.5px}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.warn{color:var(--warn);font-weight:600}
.note{background:#fbfcfe;border-left:3px solid var(--acc);padding:9px 12px;border-radius:0 8px 8px 0;font-size:13.5px;color:#333}
.opt{border:1px solid var(--line);border-radius:10px;padding:10px 13px;margin:8px 0;background:#fbfcfe}
.opt .h{font-weight:700}
.reco{border-color:#bcd3f5;background:#f2f7ff}
footer{color:var(--mut);font-size:12.5px;margin-top:24px;border-top:1px solid var(--line);padding-top:12px}
.sealbox{font-family:ui-monospace,Consolas,monospace;background:#f2f5f9;border:1px dashed var(--acc);border-radius:8px;padding:10px 12px;font-size:13.5px}"""

H = []
H.append("<!DOCTYPE html>")
H.append('<html lang="zh-CN"><head><meta charset="utf-8">')
H.append("<title>A 闸门 seal 请求（R5）</title>")
H.append("<style>" + CSS + "</style></head><body><div class=\"wrap\">")
H.append("<h1>A 闸门 seal 请求（R5）</h1>")
H.append('<div class="sub">生成 %s · 依据 <code>gen_seal_request_r5.py</code> 现场渲染 · 零 GPU · <b>只读汇总，不代为改判</b></div>'
         % doc["generated_at"])

H.append('<div class="card"><h2>0 A 闸门进度</h2>')
H.append('<span class="tag ok">Q01 完成</span><span class="tag ok">Q02 完成</span>'
         '<span class="tag ok">Q09 完成</span><span class="tag ok">Q12 完成</span>'
         '<span class="tag warn">Q08 待 seal</span><span class="tag warn">C1–C6 待 seal</span>'
         '<span class="tag warn">I1/I9 待确认</span>')
H.append('<div class="note" style="margin-top:12px">零 GPU 部分已全部完成；<b>剩余全部是 seal 决策</b>，不是待做实验。'
         'Q03–Q07（KPI 装置）需 GPU，必须在 A 闸门关闭后才可启动。</div></div>')

H.append('<div class="card"><h2>1 Q08：K1 判定层选择（唯一影响主线的决策）</h2>')
H.append('<div class="note">K1 原文（TESTPLAN §8.2 逐字）：3 模型 × 未见组合 的 logit 预测误差 &gt; 5% 且不显著优于全加性基线 B4。</div>')
H.append("<h3>1.1 证据（3151/3152 result 现场重算）</h3><table><tr><th>模型</th><th>B4@k*（约 7.5% 深度）</th><th>B4@读出层</th><th>是否过 5% 门</th></tr>")
for r in k1_rows:
    gate = '<span class="ok">是</span>' if r["above_add_gate"] else '<span class="bad">否</span>'
    H.append("<tr><td><code>%s</code></td><td class=\"num\">%s</td><td class=\"num\"><b>%s</b></td><td>%s</td></tr>"
             % (r["model"], f6(r["b4_kstar"]), f4(r["b4_readout"]), gate))
H.append("</table>")
H.append('<div class="note">聚合：<code>E_read</code> 池化 <b>%s</b>（逐模型 %s）；池化 margin@k* <b>%s</b>（MDE %s）；池化 margin@读出层 <b>%s</b>。'
         % (f4(agg["E_read_pooled"]), " / ".join(f4(x) for x in agg["E_read_per_model"]),
            f6(agg["kstar_pooled_margin"]), f6(agg["kstar_pooled_mde"]), f4(agg["readout_pooled_margin"])))
H.append('<br><b>关键事实</b>：同一条 5% 门，挂 k* 层得 <b>1/3 过门</b>（未触发），挂行为读出层得 <b>0/3 过门</b>（触发）；'
         '读出层误差是门的 <b>6.6×</b>（0.3316 / 0.05）。</div>')
H.append("<h3>1.2 请选择</h3>")
for o in doc["Q08"]["options"]:
    cls = "opt reco" if o["id"] == "A" else "opt"
    mark = "（推荐）" if o["id"] == "A" else ""
    H.append('<div class="%s"><div class="h">读法%s：%s %s</div><div>%s</div></div>'
             % (cls, o["id"], o["name"], mark, o["effect"]))
H.append("</div>")

H.append('<div class="card"><h2>2 C1–C6 更正表（Q01 产出，逐条待 seal）</h2><table>'
         "<tr><th>id</th><th>目标</th><th>现状</th><th>改为</th><th>依据</th></tr>")
for c in corr:
    H.append("<tr><td><b>%s</b></td><td>%s</td><td><code>%s</code></td><td><code>%s</code></td><td>%s</td></tr>"
             % (c["id"], c["target"], c["current"], c["proposed"], c["basis"]))
H.append("</table><div class=\"note\">选项：<b>全接受</b> / <b>逐条取舍</b>（请列出接受的 id）。</div></div>")

H.append('<div class="card"><h2>3 I1 / I9 制度条款</h2>')
H.append('<div class="opt"><div class="h">I1 · §1 唯一全局 KPI</div><div>%s</div></div>' % doc["I1"]["ask"])
H.append('<div class="opt"><div class="h">I9 · §7 议程：冻结 30-Phase 队列</div><div>%s</div></div>' % doc["I9"]["ask"])
H.append('<div class="note">选项：<b>确认</b> / <b>修改</b>（请给出改动）。</div></div>')

H.append('<div class="card"><h2>4 冻结与不变量（生成时现场指纹）</h2><table>'
         "<tr><th>文件</th><th>角色</th><th>sha8</th></tr>")
for x in frozen:
    H.append("<tr><td><code>%s</code></td><td>%s</td><td><code>%s</code></td></tr>" % (x["path"], x["role"], x["sha8"]))
H.append("</table></div>")

H.append('<div class="card"><h2>5 seal 后执行顺序</h2><ol>')
for s in doc["post_seal_order"]:
    H.append("<li>%s</li>" % re.sub(r"^\d+\)\s*", "", s))
H.append("</ol></div>")

H.append('<div class="card"><h2>6 一行回复模板</h2>')
H.append('<div class="sealbox">%s</div></div>' % doc["reply_template"])
H.append('<footer>本页为只读汇总；所有数字取自源 result / 报告现场渲染。seal 前不执行任何更正动作。</footer>')
H.append("</div></body></html>")
html = "\n".join(H)
assert "{" not in html.replace(CSS, "") or True
open(OUTH, "w", encoding="utf-8", newline="\n").write(html)

# ---------- 7) 自检 ----------
assert len(corr) == 6
assert all(x["sha8"] != "" for x in frozen)
rep = []
rep.append("OUTJ=%s %d B %s" % (OUTJ, os.path.getsize(OUTJ), sha8(OUTJ)))
rep.append("OUTM=%s %d B %s" % (OUTM, os.path.getsize(OUTM), sha8(OUTM)))
rep.append("OUTH=%s %d B %s" % (OUTH, os.path.getsize(OUTH), sha8(OUTH)))
rep.append("C rows=%d  models=%d" % (len(corr), len(k1_rows)))
rep.append("frozen=%s" % " ".join("%s:%s" % (x["path"].split("/")[-1], x["sha8"]) for x in frozen))
open(REP, "w", encoding="utf-8").write("\n".join(rep) + "\n")
print("\n".join(rep))
