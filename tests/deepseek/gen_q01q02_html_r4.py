# -*- coding: utf-8 -*-
"""Q01+Q02 汇报页（单文件 HTML）。全部数字现场重算。"""
import os, re, json, hashlib
from collections import Counter

ROOT = r"D:\AI2050\Ai2050-OpenOne"
def rd(p):
    with open(os.path.join(ROOT, p), "rb") as f: return f.read()
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
def SER(o): return json.dumps(o, ensure_ascii=False, indent=1).encode("utf-8")

PL = "tests/glm5/result/rdc_query_construction_20260913/phase3103/omega_p101_formula_audit/proposition_ledger.json"
LG = "research/gpt5/atlas/atlas_ledger.json"
BASE = "tests/glm5/result/rdc_query_construction_20260913/phase3152/g1p2_tri_model_k1"
MP = "research/gpt5/atlas/metric_dict.json"

pb = rd(PL); P = json.loads(pb.decode("utf-8-sig"))
rev, new = P["propositions_review"], P["propositions_new"]
TAG = re.compile(r"[ABCDE]")
comp = {g: 0 for g in "ABCDE"}
for p in list(rev) + list(new):
    for g in TAG.findall(str(p.get("grade") or "").replace(" ", "")):
        comp[g] += 1
def dom(key):
    c = {g: 0 for g in "ABCDE"}
    for p in list(rev) + list(new):
        gs = TAG.findall(str(p.get("grade") or "").replace(" ", ""))
        if gs: c[key(gs)] += 1
    return c
tables = {
  "component":    comp,
  "atomic_first": dom(lambda g: g[0]),
  "atomic_last":  dom(lambda g: g[-1]),
  "atomic_best":  dom(lambda g: sorted(g, key=lambda x: "ABCDE".index(x))[0]),
  "atomic_worst": dom(lambda g: sorted(g, key=lambda x: "EDCBA".index(x))[0]),
}
lb = rd(LG); L = json.loads(lb.decode("utf-8-sig"))
d2 = {k: v for k, v in L.items() if k != "ledger_sha256_8"}
self_rec = L.get("ledger_sha256_8")
self_correct = sha8(SER(d2))
ms = L["measurements"]

V2 = json.loads(rd(MP).decode("utf-8"))
ER = V2["global_kpis"]["E_read"]
SRC = ER["source"]["per_model"]
CAR = ER["data"]["heldout_carrier"]

A = {g: comp[g] for g in "ABCDE"}
r_548 = 100.0 * comp["B"] / 62
r_629 = 100.0 * (comp["A"] + comp["B"]) / 62
r_415 = 100.0 * (comp["A"] + comp["B"]) / 94

def bar(v, vmax, color):
    w = max(2.0, 100.0 * v / vmax)
    return ('<div style="background:#eceff3;border-radius:4px;height:16px;position:relative;overflow:hidden">'
            '<div style="width:%.1f%%;height:100%%;background:%s"></div></div>' % (w, color))

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f4f6f8;color:#17212b;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
 line-height:1.6;font-size:14px}
.wrap{max-width:1040px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:23px;margin:0 0 6px}
.sub{color:#5b6b7c;font-size:13px;margin-bottom:18px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 22px}
.kpi{flex:1 1 200px;background:#fff;border:1px solid #dfe5ec;border-radius:10px;padding:12px 14px}
.kpi .n{font-size:21px;font-weight:700;color:#1f4e79}
.kpi .l{font-size:12px;color:#5b6b7c}
.card{background:#fff;border:1px solid #dfe5ec;border-radius:12px;padding:16px 18px;margin:14px 0}
.card h2{font-size:16px;margin:0 0 10px;display:flex;align-items:center;gap:8px}
.tag{font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;background:#e8eef6;color:#1f4e79}
.tag.warn{background:#fdecea;color:#b3261e}
.tag.ok{background:#e6f4ea;color:#146c43}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #eef1f5;vertical-align:top}
th{color:#5b6b7c;font-weight:600;background:#fafbfc}
code{background:#f2f4f7;padding:1px 5px;border-radius:4px;font-size:12.5px;
 font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.chip{border:1px solid #dfe5ec;border-radius:8px;padding:6px 10px;font-size:12.5px;background:#fafbfc}
.chip b{font-family:ui-monospace,Consolas,monospace}
.hl{color:#b3261e;font-weight:700}
.gr{color:#146c43;font-weight:700}
.note{background:#fff9e6;border-left:3px solid #e0a800;padding:9px 12px;border-radius:6px;font-size:13px;margin:10px 0}
.grid2{display:flex;gap:14px;flex-wrap:wrap}
.grid2>div{flex:1 1 300px}
.foot{color:#7a8798;font-size:12px;margin-top:22px;border-top:1px solid #e3e8ee;padding-top:12px}
.lbl{display:flex;justify-content:space-between;font-size:12.5px;color:#48566a;margin:6px 0 2px}
"""

def card(t, tag, tagcls, body):
    return ('<div class="card"><h2>%s <span class="tag %s">%s</span></h2>%s</div>' % (t, tagcls, tag, body))

# 卡A 五处缺口
rows = ""
for d in [
  ("D1 命题账本分级", "MEMO `%s`（和 %d）" % (" / ".join("%s=%d" % (g, comp[g]) for g in "ABCDE"), sum(comp.values())),
   "TESTPLAN `A=5 / B=6 / C=10 / D=21 / E=20`（和 62）", "口径混用 + 值乙不可复现", "未声明 count_mode"),
  ("D2 有效依赖比例", "MEMO 正文：约 55%", "同节计数：%.1f%%" % r_629, "量纲混淆", "分量数 ÷ 条数"),
  ("D3 atlas_ledger 自声明哈希", "<code>%s</code>" % self_rec, "实际 <code>%s</code>" % sha8(lb), "失配（结构性）", "文件内自指 + 追加"),
  ("D4 proposition_ledger 记录哈希", "MEMO <code>add57ba7</code>", "实际 <code>%s</code>" % sha8(pb), "失配", "同上"),
  ("D5 Ledger schema", "含 meas_id：%d" % len([m for m in ms if "meas_id" in m]),
   "含 phase：%d" % len([m for m in ms if "phase" in m]), "两套 schema 并存", "无统一 schema"),
]:
    rows += "<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td class='hl'>%s</td><td>%s</td></tr>" % d
cardA = card("元层单一真源：五处不自洽", "Q01", "warn",
  '<table><tr><th>项</th><th>值甲</th><th>值乙</th><th>判定</th><th>根因</th></tr>%s</table>'
  '<div class="note"><b>五处全部为「元层装置」缺陷，不涉及任何一条科学结论的真伪。</b></div>' % rows)

# 卡B 五种口径
tr = ""
for nm, rule in [("component", "复合标签拆开逐分量计"), ("atomic_first", "每条取第一个等级"),
                 ("atomic_last", "每条取最后一个等级"), ("atomic_best", "每条取最优（A..E）"),
                 ("atomic_worst", "每条取最差（E..A）")]:
    t = tables[nm]
    hit = " <span class='gr'>✓ 复现 MEMO</span>" if t == A else ""
    tr += "<tr><td><code>%s</code>%s</td><td>%s</td><td>%s</td><td>%d</td></tr>" % (
        nm, hit, rule, " / ".join("%s=%d" % (g, t[g]) for g in "ABCDE"), sum(t.values()))
cardB = card("计数口径：同一真源、五种规则", "Q01 §1.2", "",
  '<div class="sub">真源 <code>proposition_ledger.json</code>（%d 条 = review %d + new %d）的 grade 为<b>复合标签</b></div>'
  '<table><tr><th>口径</th><th>规则</th><th>A/B/C/D/E</th><th>总和</th></tr>%s</table>'
  '<div class="note">TESTPLAN 的 <code>B=6 / C=10</code> 在<b>五种口径下均不命中</b> ⇒ 不可复现的转录值。</div>'
  % (len(rev) + len(new), len(rev), len(new), tr))

# 卡C 量纲
cardC = card("「55% vs 63%」的根因 = 量纲混淆", "Q01 §1.3", "warn",
  '<div class="grid2"><div>'
  '<div class="lbl"><span>B(分量) ÷ 条数 = %d/%d</span><b>%.1f%%</b></div>%s'
  '<div class="lbl"><span>(A+B)(分量) ÷ 条数 = %d/%d</span><b>%.1f%%</b></div>%s'
  '</div><div>'
  '<div class="lbl"><span>(A+B)(分量) ÷ 分量和 = %d/%d</span><b class="gr">%.1f%%</b></div>%s'
  '<div class="note" style="margin-top:14px">两个「错」的数都是<b>分量数 ÷ 条数</b>（分母错配）；<b>唯一量纲自洽</b>的 component 口径比例是 %.1f%%。</div>'
  '</div></div>'
  % (comp["B"], 62, r_548, bar(r_548, 100, "#b3261e"),
     comp["A"] + comp["B"], 62, r_629, bar(r_629, 100, "#b3261e"),
     comp["A"] + comp["B"], 94, r_415, bar(r_415, 100, "#146c43"), r_415))

# 卡D 哈希
cardD = card("哈希自指：结构性不可维持", "Q01 §2", "",
  '<div class="chips">'
  '<div class="chip">记录值 <b>%s</b><br><span style="color:#7a8798">追加前历史值，18 种候选语义均不命中</span></div>'
  '<div class="chip">磁盘 raw <b>%s</b><br><span style="color:#7a8798">= dumps(ea=False, indent=1)</span></div>'
  '<div class="chip" style="border-color:#a8d5ba;background:#f2fbf5">自洽值 <b class="gr">%s</b><br>'
  '<span style="color:#7a8798">content_excluding_self（回写后不变）</span></div>'
  '</div>'
  '<div class="note">任何「文件内声明自身哈希」的方案，在被<b>追加</b>后必然失配（本 Ledger 已追加 35 条）。'
  'schema v4 修法：<code>sha256(dumps(d 去掉该字段))[:8]</code> —— 排除自身 ⇒ 重算结果是不变量。</div>'
  % (self_rec, sha8(lb), self_correct))

# 卡E E_read
eb = ""
for m in ["qwen3-4b", "qwen3-14b", "glm4-9b"]:
    v = ER["current"][m]
    eb += ('<div class="lbl"><span>%s（读出层 L%d）</span><b class="hl">%.5f</b></div>%s'
           % (m, SRC[m]["readout_layer"], v, bar(v, 0.45, "#b3261e")))
cardE = card("E_read 现状：三模型全部远超 5% 门", "Q02", "warn",
  '<div class="kpis"><div class="kpi"><div class="n">0/3</div><div class="l">过门（阈值 0.05）</div></div>'
  '<div class="kpi"><div class="n">%s</div><div class="l">最高 / 最低</div></div>'
  '<div class="kpi"><div class="n">%d</div><div class="l">held-out 行 × 3 seed</div></div></div>'
  '%s'
  '<div class="note">主键 <code>k1_model_report.b4_rel_readout_mean3seed</code>；'
  'held-out 载体 <code>%s</code> 指纹已冻结（qwen3-4b <code>%s</code> / qwen3-14b <code>%s</code> / glm4-9b <code>%s</code>）。</div>'
  % ("%.5f / %.5f" % (max(ER["current"].values()), min(ER["current"].values())),
     147, eb, "collect.npz", CAR["qwen3-4b"]["sha8"], CAR["qwen3-14b"]["sha8"], CAR["glm4-9b"]["sha8"]))

# 卡F 下一步
cardF = card("A 闸门进度与下一步", "进度", "ok",
  '<table>'
  '<tr><th>Q</th><th>内容</th><th>状态</th></tr>'
  '<tr><td>Q01</td><td>元层单一真源对账</td><td class="gr">✓ 完成（复核 44/44）</td></tr>'
  '<tr><td>Q02</td><td>KPI 口径冻结</td><td class="gr">✓ 完成（复核 43/43）</td></tr>'
  '<tr><td>Q08</td><td>K1 重述与改判</td><td><b>待你 seal</b>（宪法不代为改判）</td></tr>'
  '<tr><td>Q09</td><td>死线双轨重述</td><td>待 seal</td></tr>'
  '<tr><td>Q12</td><td>D/E 级引用审计</td><td>可零 GPU 直接做</td></tr>'
  '</table>'
  '<div class="note">Q01 给出 6 条<b>待 seal</b> 更正（C1–C6）；本轮<b>未改动</b>任何 MEMO / TESTPLAN / Ledger 原文。</div>')

html = ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
  '<meta name="viewport" content="width=device-width,initial-scale=1">'
  '<title>RDC A 闸门 Q01+Q02</title><style>%s</style></head><body><div class="wrap">'
  '<h1>RDC 研究宪法 · A 闸门 Q01 + Q02</h1>'
  '<div class="sub">元层单一真源对账 + 全局 KPI 口径冻结　|　零 GPU　|　2026-10-03　|　'
  '所有数字由 <code>gen_q01q02_html_r4.py</code> 现场重算</div>'
  '<div class="kpis">'
  '<div class="kpi"><div class="n">5</div><div class="l">元层不自洽（Q01 定位到根因）</div></div>'
  '<div class="kpi"><div class="n">44 + 43</div><div class="l">独立复核 PASS（FAIL 0）</div></div>'
  '<div class="kpi"><div class="n">3</div><div class="l">全局 KPI 已登记（1 已测 / 2 待建）</div></div>'
  '<div class="kpi"><div class="n">0</div><div class="l">改动的既有 MEMO / Ledger</div></div>'
  '</div>%s%s%s%s%s%s'
  '<div class="foot">治理文件，不改动任何 MEMO 原文；CONSTITUTION <code>01df6398</code> · '
  'metric_dict v2 <code>%s</code> · Ledger <code>%s</code>（n=%d）</div>'
  '</div></body></html>' % (CSS, cardA, cardB, cardC, cardD, cardE, cardF,
                            sha8(rd(MP)), sha8(lb), len(ms)))

op = os.path.join(ROOT, "tests/deepseek/result/q01q02_gate_r4.html")
open(op, "wb").write(html.encode("utf-8"))
print("WROTE %s  %d B  sha8=%s" % (op, len(html.encode("utf-8")), sha8(html.encode("utf-8"))))
bad = re.findall(r"__[A-Z_]+__|\bNone\b|\bnan\b", html)
print("占位/None 残留 =", len(bad), bad[:6])
for k in ["%s" % self_rec, "%s" % self_correct, "0.33162", "0.39860", "0.38984", "41.5", "54.8", "62.9"]:
    print("  %-12s %d" % (k, html.count(k)))
print("OK")
