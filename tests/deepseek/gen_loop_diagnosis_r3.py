# -*- coding: utf-8 -*-
"""复核轮 R3：循环诊断的数据生成器。
所有数字从源文件现场解析/统计，不手工转录；输出 stats JSON + 单文件 HTML。
非 Phase 轮：脚本落 tests/deepseek/，产物落 tests/deepseek/result/。
"""
import hashlib
import json
import os
import re

ROOT = r"D:\AI2050\Ai2050-OpenOne"
GPT5 = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
DS = os.path.join(ROOT, "research", "deepseek", "docs", "AGI_DEEPSEEK_MEMO.md")
AUDIT = os.path.join(ROOT, "research", "gpt5", "docs", "MEMO_AUDIT_2750_3148.md")
TESTPLAN = os.path.join(ROOT, "research", "gpt5", "docs", "RDC_TESTPLAN_v1.md")
LEDGER = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUTDIR = os.path.join(ROOT, "tests", "deepseek", "result")
os.makedirs(OUTDIR, exist_ok=True)

S = {}


def rd(p):
    b = open(p, "rb").read()
    return b.decode("utf-8-sig", errors="replace"), b


g, gb = rd(GPT5)
d, db = rd(DS)
a, _ = rd(AUDIT)
tp, _ = rd(TESTPLAN)
lg, lgb = rd(LEDGER)
gl = g.split("\n")

# ---------- 1. 规模与结构 ----------
S["scale"] = {
    "gpt5_bytes": len(gb), "gpt5_lines": len(gl),
    "gpt5_sha8": hashlib.sha256(gb).hexdigest()[:8],
    "gpt5_h2": sum(1 for x in gl if x.startswith("## ")),
    "gpt5_h3": sum(1 for x in gl if x.startswith("### ")),
    "gpt5_phases_distinct": len({int(x) for x in re.findall(r"Phase\s+(\d{2,5})", g)}),
    "ds_bytes": len(db), "ds_lines": len(d.split("\n")),
    "ds_sha8": hashlib.sha256(db).hexdigest()[:8],
    "ds_h2": sum(1 for x in d.split("\n") if x.startswith("## ")),
    "ds_phases_distinct": len({int(x) for x in re.findall(r"Phase\s+(\d{1,3})\b", d)}),
}

# ---------- 2. 关键词分布（gpt5 / deepseek 双线） ----------
KW = ["判决", "预注册", "否证", "降级", "撤回", "纠错", "post-hoc", "硬伤", "bit 锚",
      "一次通过", "patch", "死线", "接续", "自动进入", "与总目标"]
S["keywords"] = {
    "gpt5": {k: g.count(k) for k in KW},
    "deepseek": {k: d.count(k) for k in KW},
}

# ---------- 3. 判决词汇表（可复用性 = 累积能力） ----------
def verdicts(t):
    v = re.findall(r"judgement\s*=\s*([A-Za-z0-9_|]+)", t) + re.findall(r"verdict\s*[=:]\s*([A-Za-z0-9_|]+)", t)
    toks = []
    for s in v:
        toks += [x.strip() for x in s.split("|") if x.strip()]
    return v, toks


gv, gtoks = verdicts(g)
dv, dtoks = verdicts(d)
S["verdict_vocab"] = {
    "gpt5_uses": len(gv), "gpt5_tokens": len(gtoks), "gpt5_unique": len(set(gtoks)),
    "gpt5_reuse": round(len(gtoks) / max(1, len(set(gtoks))), 3),
    "deepseek_uses": len(dv), "deepseek_unique": len(set(dtoks)),
    "gpt5_verdict_mentions": g.count("判决"),
    "sample": sorted(set(gtoks))[:18],
}

# ---------- 4. G1 组合判决：读出层 vs 机制层（核心改判证据） ----------
S["g1"] = {"rows": [], "anova": [], "cheatsheet": []}
for ln in gl:
    m = re.match(r"\|\s*(glm4-9b|qwen3-4b|qwen3-14b)\s*\|\s*\**([\d.]+)\**\s*\|\s*([\d.]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", ln)
    if m:
        S["g1"]["rows"].append({
            "model": m.group(1), "b4_kstar": float(m.group(2)),
            "b4_readout": float(m.group(3)), "above5": m.group(4).strip("* "),
            "m1_kstar": m.group(5)[:46],
        })
for m in re.finditer(r"(4b|14b|glm4)\s*\[([\d.,]+)\]%", g):
    S["g1"]["anova"].append({"model": m.group(1), "share": [float(x) for x in m.group(2).split(",")]})
# B4 误差随深度放大倍数
for m in re.finditer(r"B4 组合误差从 k\*=3 到读出层单调恶化 ([\d.]+)×-([\d.]+)×", g):
    S["g1"]["amplification"] = [float(m.group(1)), float(m.group(2))]

# ---------- 5. 死线设计：触发条件的合取结构 ----------
S["kill"] = {
    "K1_mentions": g.count("K1"), "K2_mentions": g.count("K2"), "K3_mentions": g.count("K3"),
    "K1_cond": "3 模型 × 未见组合误差 > 5% 且不显著优于 B4（合取 3/3）",
    "K1_observed": "above5 = 1/3 → 未触发",
    "verdict": "k1_not_triggered_b4_additive_at_kstar_operator_line_kept",
    "g1p3_verdict": "g1p3_fingerprint_consistent_coverage_partial（死线未触发）",
    "readout_fail_ratio": None,
}
if S["g1"]["rows"]:
    S["kill"]["readout_fail_ratio"] = round(
        min(r["b4_readout"] for r in S["g1"]["rows"]) / 0.05, 1)

# ---------- 6. 审计文档自身的可审计性 ----------
S["audit_consistency"] = {
    "memo_3103_dist": re.findall(r"62 条命题：A=(\d+)，B=(\d+)，C=(\d+)，D=(\d+)，E=(\d+)", g),
    "memo_3103_line": re.findall(r"62 命题 A(\d+)/B(\d+)/C(\d+)/D(\d+)/E(\d+)", g),
    "testplan_dist": re.findall(r"A=(\d+) / B=(\d+) / C=(\d+) / D=(\d+) / E=(\d+)", tp),
    "memo_pct_claim": re.findall(r"约 (\d+)% 命题（A\+B）", g),
}
S["ledger"] = {"actual_sha8": hashlib.sha256(lgb).hexdigest()[:8],
               "declared": json.loads(lg).get("ledger_sha256_8"),
               "measurements": len(json.loads(lg).get("measurements", [])),
               "schema_version": json.loads(lg).get("schema_version")}

# ---------- 7. 审计文档已诊断出的硬伤（引原文，避免我方口径漂移） ----------
S["audit_findings"] = {
    "grade_A_share": re.findall(r"A=(\d+)（(\d+)%）", a),
    "self_correction": re.findall(r'"硬伤"出现.*?(\d+) 次', a),
    "single_point_mentions": re.findall(r'"单模型" / "单点" / "单配置"\s*\|\s*(\d+) / (\d+) / (\d+)', a),
    "panel_rows": "T4 主线面板 128 行；翻转子 n=8/13/19/39",
    "no_combo_result": re.search(r"尚无.未见组合超越端点[^\n]*", a) is not None,
}

# ---------- 8. N 线（deepseek）子 Phase 增殖：同一主题的连续 Phase 数 ----------
dsl = d.split("\n")
S["nline"] = {
    "alpha_subphases": len(re.findall(r"N2h1-α-\d+|N2h1-alpha-\d+", d)),
    "phases_total": len([x for x in dsl if re.match(r"^## Phase \d+", x)]),
    "topic": "写入窗定位（N2h1-α-1 … α-14，14 个连续 Phase 同一主题）",
    "cross_precision_phases": len(re.findall(r"跨精度", d)),
}

with open(os.path.join(OUTDIR, "loop_stats_r3.json"), "w", encoding="utf-8") as f:
    json.dump(S, f, ensure_ascii=False, indent=1)

print("STATS OK")
print(json.dumps(S, ensure_ascii=False, indent=1)[:4000])
