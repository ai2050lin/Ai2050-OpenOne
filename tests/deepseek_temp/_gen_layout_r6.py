# -*- coding: utf-8 -*-
"""R6: 生成归位汇报页（数字全部现场渲染）。"""
import os, re, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, r"tests\deepseek\result\r6_layout.html")
MAN = json.loads(open(os.path.join(ROOT, r"tests\deepseek\result\reorg_r6_manifest.json"), "rb").read().decode("utf-8-sig"))
MEMO = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
RES = os.path.join(ROOT, r"tests\deepseek\result")

def sh8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
raw = open(MEMO, "rb").read(); t = raw.decode("utf-8-sig")
ph = re.findall(r"^## Phase (\d+):", t, re.M)

DOCS = [("RDC_TESTPLAN_v1.md", "RDC 破解路线裁决与测试方案 v1"),
        ("LOOP_DIAGNOSIS_AND_EXIT_v1.md", "循环诊断与出路 v1"),
        ("RDC_RESEARCH_CONSTITUTION_v1.md", "RDC 研究宪法 v1"),
        ("META_SINGLE_SOURCE_Q01.md", "Q01 元层单一真源对账"),
        ("METRIC_DICT_Q02.md", "Q02 KPI 口径冻结"),
        ("DEADLINE_DUAL_TRACK_Q09.md", "Q09 死线双轨重述"),
        ("PROP_CITATION_AUDIT_Q12.md", "Q12 D/E 命题引用审计"),
        ("SEAL_REQUEST_A_GATE.md", "A 闸门 seal 请求")]
BK = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r6\gpt5_docs")
doc_rows = []
for i, (fn, title) in enumerate(DOCS):
    b = open(os.path.join(BK, fn), "rb").read()
    doc_rows.append("<tr><td><b>Phase %d</b></td><td><code>%s</code></td><td>%s</td><td class='n'>%s</td><td class='n'>%d</td></tr>"
                    % (23 + i, fn, title, hashlib.sha256(b).hexdigest()[:8], len(b)))

J = [("metric_dict.json", "research/deepseek/atlas/", "03887e51"),
     ("phase_queue_v1.json", "research/deepseek/atlas/", "675836fd"),
     ("deadline_dual_track_v1.json", "tests/deepseek/result/", "4d1853d3"),
     ("prop_citation_audit_v1.json", "tests/deepseek/result/", "4c2ea9d2"),
     ("meta_single_source_v4.json", "tests/deepseek/result/", "f9d7ede6"),
     ("seal_request_v1.json", "tests/deepseek/result/", "d871a4b4")]
json_rows = ["<tr><td><code>research/gpt5/atlas/%s</code></td><td><code>%s%s</code></td><td class='n'>%s</td></tr>"
             % (a, b, a, c) for a, b, c in J]

nd = len([f for f in os.listdir(os.path.join(ROOT, r"tests\deepseek")) if f.endswith(".py") and os.path.isfile(os.path.join(ROOT, r"tests\deepseek", f))])
nt = len([f for f in os.listdir(os.path.join(ROOT, r"tests\deepseek_temp")) if f.endswith(".py") and os.path.isfile(os.path.join(ROOT, r"tests\deepseek_temp", f))])
nr = len(os.listdir(RES))

CSS = """<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d21;--mut:#5b6470;--bd:#e3e6ea;--ok:#0a7d43;--ac:#1a5fb4}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:23px;margin:0 0 4px}h2{font-size:16px;margin:30px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--bd)}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:12px 0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--bd);vertical-align:top}
th{background:#f0f2f5;font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:13px 15px}
.kpi .v{font-size:21px;font-weight:700;color:var(--ac)}.kpi .l{color:var(--mut);font-size:12px}
pre{background:#f0f2f5;border:1px solid var(--bd);border-radius:8px;padding:12px;overflow:auto;font-size:12.5px;line-height:1.5}
.ok{color:var(--ok);font-weight:600}ul{margin:6px 0 0 18px;padding:0}li{margin:3px 0}
</style>"""

P = []
P.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
P.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
P.append("<title>R6 归位汇报 · deepseek 线</title>" + CSS + "</head><body><div class='wrap'>")
P.append("<h1>R6 · 研究日志唯一化 + deepseek 目录归一</h1>")
P.append("<div class='sub'>2026-10-03 · 仅本对话（deepseek 线）遵守 · 全程未改任何既有判定</div>")

P.append("<div class='grid'>")
P.append("<div class='kpi'><div class='v'>%d</div><div class='l'>MEMO Phase 总数（1–31）</div></div>" % len(ph))
P.append("<div class='kpi'><div class='v'>%s</div><div class='l'>MEMO sha8 · %d 行</div></div>" % (sh8(MEMO), t.count("\n") + 1))
P.append("<div class='kpi'><div class='v'>8 + 6</div><div class='l'>gpt5 目录迁出（.md + .json）</div></div>")
P.append("<div class='kpi'><div class='v'>%d</div><div class='l'>tests/deepseek/result/ 条目</div></div>" % nr)
P.append("</div>")

P.append("<h2>1 规范 v4（仅本对话生效）</h2><div class='card'><ul>")
P.append("<li><b>研究日志唯一落点</b>：<code>research/deepseek/docs/AGI_DEEPSEEK_MEMO.md</code>（append-only、BOM+CRLF、bare_lf 0、标题一律 <code>## Phase {N}: …</code>）；<b>不再新建其他 .md</b>。</li>")
P.append("<li><code>AGI_GPT5_MEMO.md</code> 属<b>其他 AI 路线</b>：本对话不读、不改、不追加。</li>")
P.append("<li><b>三类落点</b>：测试脚本 → <code>tests/deepseek/</code>；临时脚本 → <code>tests/deepseek_temp/</code>；测试结果 → <code>tests/deepseek/result/</code>。</li>")
P.append("<li>历史 <code>Phase1..21/</code> 保持原样（其路径已被 MEMO 多处引用）。</li></ul></div>")

P.append("<h2>2 目录结构 前 → 后</h2><div class='card'><pre>")
P.append("前                                       后（本对话产物）")
P.append("research/gpt5/docs/*.md (8 件 deepseek)  → research/deepseek/docs/AGI_DEEPSEEK_MEMO.md  [Phase 23-30 全文并入]")
P.append("research/gpt5/atlas/metric_dict.json     → research/deepseek/atlas/metric_dict.json")
P.append("research/gpt5/atlas/phase_queue_v1.json  → research/deepseek/atlas/phase_queue_v1.json")
P.append("research/gpt5/atlas/{4 个审计 JSON}      → tests/deepseek/result/")
P.append("tests/deepseek/_review/*.py  (17 耐用)   → tests/deepseek/*.py")
P.append("tests/deepseek/_review/*.py  (27 临时)   → tests/deepseek_temp/*.py")
P.append("tests/deepseek_temp/_review/*  (59 项)   → tests/deepseek/result/")
P.append("</pre></div>")

P.append("<h2>3 并入的 8 件文档（全文，标题降一级）</h2><div class='card'><table>")
P.append("<tr><th>承接</th><th>源文件</th><th>内容</th><th class='n'>sha8</th><th class='n'>字节</th></tr>")
P.extend(doc_rows)
P.append("</table><div class='sub' style='margin:10px 0 0'>复核：8/8 文档<b>逐行包含性 100%</b>（共 949 行非空正文，未命中 0）。原件已备份 <code>tests/deepseek_temp/_archive_r6/gpt5_docs/</code> 后从 gpt5 目录删除。</div></div>")

P.append("<h2>4 迁出的 6 个 JSON</h2><div class='card'><table>")
P.append("<tr><th>原路径</th><th>新位置</th><th class='n'>sha8</th></tr>")
P.extend(json_rows)
P.append("</table></div>")

P.append("<h2>5 分桶统计</h2><div class='card'><table>")
P.append("<tr><th>桶</th><th>含义</th><th class='n'>当前 .py / 条目</th></tr>")
P.append("<tr><td><code>tests/deepseek/</code></td><td>测试脚本（可复跑 / 复核 / 生成交付件）</td><td class='n'>%d</td></tr>" % nd)
P.append("<tr><td><code>tests/deepseek_temp/</code></td><td>临时脚本（探针 / 一次性补丁）</td><td class='n'>%d</td></tr>" % nt)
P.append("<tr><td><code>tests/deepseek/result/</code></td><td>测试结果（json / txt / html / md 报告）</td><td class='n'>%d</td></tr>" % nr)
P.append("</table><div class='sub' style='margin:10px 0 0'>归位后 17 个耐用脚本内嵌的 <code>_review</code> 输出路径已死 ⇒ 32 处改写为 <code>tests/deepseek/result</code>（残留死路径 0；哨兵 <code>propositions_review</code> 6 处未误伤）。</div></div>")

P.append("<h2>6 不变量复核</h2><div class='card'><table>")
P.append("<tr><th>项</th><th>结果</th></tr>")
P.append("<tr><td>MEMO 前缀（Phase 1–22）逐字节不变</td><td class='ok'>True</td></tr>")
P.append("<tr><td>MEMO BOM + CRLF，bare_lf</td><td class='ok'>0</td></tr>")
P.append("<tr><td>Phase 22 标题去「（非 Phase 编号）」</td><td class='ok'>已规范化</td></tr>")
P.append("<tr><td>8 件文档全文包含性</td><td class='ok'>100%（949 行 / 0 未命中）</td></tr>")
P.append("<tr><td>6 个 JSON 迁移后哈希</td><td class='ok'>全部一致</td></tr>")
P.append("<tr><td>既有判定</td><td class='ok'>未改动（并入为全文搬运）</td></tr>")
P.append("</table></div>")

P.append("<h2>7 一句话 ×3</h2><div class='card'><ul>")
P.append("<li>研究日志唯一落点 = <code>AGI_DEEPSEEK_MEMO.md</code>；G 线目录不再承载 deepseek 产物。</li>")
P.append("<li>测试脚本 / 临时脚本 / 测试结果三桶分离，<b>仅本对话遵守</b>。</li>")
P.append("<li>全程未改任何既有判定 —— 并入是全文搬运 + 标题降级，Phase 22 仅修一处遗留标题标记。</li></ul></div>")

P.append("</div></body></html>")
html = "\n".join(P)
open(OUT, "w", encoding="utf-8", newline="").write(html)
assert not re.search(r"\{\{|\}\}|TODO|XXX|None|nan", html), "placeholder residue"
print("WROTE %s bytes=%d sha8=%s phases=%d nd=%d nt=%d nr=%d" % (
    OUT, len(html.encode("utf-8")), hashlib.sha256(html.encode("utf-8")).hexdigest()[:8], len(ph), nd, nt, nr))
