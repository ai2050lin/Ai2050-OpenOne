# -*- coding: utf-8 -*-
"""
p3090_reuse_inventory.py — R1 (复用拓扑全景 v1) 数据底册构建
读取 research/gpt5/atlas/atlas_ledger.json 全部 measurements，
按复用机制标签 x 模型 x 层位分类，输出:
  - tests/gpt5_temp/reuse_inventory.json        (全量带标签清单)
  - tests/gpt5_temp/reuse_inventory_report.txt  (覆盖矩阵 + 缺口分析)
仅做文本分类与统计，不做任何前向，不修改 Ledger。
"""
import json
import re
import io
import sys
from collections import defaultdict, Counter

LEDGER = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json"
OUT_JSON = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\reuse_inventory.json"
OUT_TXT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\reuse_inventory_report.txt"

# 复用机制标签: (标签名, 正则, 说明)
TAG_RULES = [
    ("pool_reversal",      r"pool|reversal|反转|S_A|top.?128|cond_reversal|direction_pool", "条件反转池/池复用"),
    ("gamma_readout",      r"gamma|whiten|白化|colstd|reweight|反方差|readout_pc|PC1", "γ 反方差重加权读出"),
    ("focal_heads",        r"focal|head|h20|h7|h1|h14|h26|capture8|attention_head|attn", "焦点头"),
    ("hill_capacity",      r"hill|capacity|d_eff|effective_dim|a=0\.7|submod|超模", "Hill 容量/双区"),
    ("registry",           r"registry|登记|word_coord|coordinate|坐标册|atlas_axis", "坐标登记册"),
    ("spectrum_continuum", r"spectrum|谱|participation|PR_|trunk|dispersed|mixed_|continuum|keff", "谱连续统 trunk/mixed/dispersed"),
    ("relay_damper",       r"relay|damper|阻尼|中继|transport_mid|L3_|l3_|midlayer", "L3 中继/阻尼场"),
    ("operator_transfer",  r"jacobian|operator|gear|齿轮|transmission|transfer|dz_|carrier|传动", "算子/传动复用"),
    ("alignment_base",     r"base|chat|instruct|align|procrustes|cross_family|跨族", "base-vs-chat 对齐"),
    ("long_context",       r"long|4k|8k|kv|span|length|context_len|跨距", "长上下文/KV"),
]

MODEL_RULES = [
    ("qwen3-4b",  r"qwen.?3.?4b|qwen4b|4b\b", "qwen3-4b"),
    ("qwen3-14b", r"qwen.?3.?14b|qwen14b|14b\b", "qwen3-14b"),
    ("glm4",      r"glm.?4|glm4", "glm4"),
    ("ds7b",      r"ds.?7b|deepseek|ds7b", "ds7b"),
    ("qwen25",    r"qwen.?2\.?5|qwen25|distill", "qwen2.5"),
]

NEG_PAT = re.compile(r"\bfalse\b|no_|refut|incompatib|null|closed|fail|offdiag|degener|absent|reject", re.I)
POS_PAT = re.compile(r"\btrue\b|confirm|pass\b|\bok\b|real|stable|hold|robust|locked|migrate", re.I)

def classify_text(text, rules):
    tags = []
    for name, pat, _ in rules:
        if re.search(pat, text, re.I):
            tags.append(name)
    return tags

def extract_layers(text):
    # 匹配 L13 / L38 / layer 39 / L_INJ=38 等
    layers = set()
    for m in re.finditer(r"\bL(\d{1,2})\b", text):
        layers.add(int(m.group(1)))
    for m in re.finditer(r"layer[s]?[ =:](\d{1,2})", text, re.I):
        layers.add(int(m.group(1)))
    return sorted(layers)

def polarity(verdict):
    n = len(NEG_PAT.findall(verdict))
    p = len(POS_PAT.findall(verdict))
    if n and p:
        return "mixed"
    if p:
        return "positive"
    if n:
        return "negative"
    return "neutral"

def main():
    with io.open(LEDGER, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    meas = ledger.get("measurements", [])
    items = []
    for e in meas:
        mid = e.get("meas_id") or e.get("name") or "?"
        typ = e.get("type", "")
        verdict = e.get("verdict", "")
        src = e.get("source", {}) or {}
        if isinstance(src, str):
            path = src
            src = {"path": path, "sha256_8": "", "phase": None}
        else:
            path = src.get("path", "")
        phase = src.get("phase")
        if phase is None:
            phase = e.get("phase")
        if phase is None:
            m = re.search(r"(?:^|[^0-9])(\d{3,4})", mid)
            phase = int(m.group(1)) if m else -1
        # 富字段条目: question/design/tests 也纳入分类文本
        extra = " ".join(str(e.get(k, "")) for k in ("question", "design", "tests"))
        blob = " | ".join([mid, typ, verdict, path, extra])
        tags = classify_text(blob, TAG_RULES)
        # 模型判定: 默认 ledger 顶层 model=qwen3-4b, 除非文本显式提及其他模型
        models = classify_text(blob, MODEL_RULES)
        if not models:
            models = [ledger.get("model", "qwen3-4b")]
        layers = extract_layers(blob)
        items.append({
            "meas_id": mid,
            "phase": phase,
            "type": typ,
            "verdict": verdict,
            "path": path,
            "sha8": src.get("sha256_8", ""),
            "tags": tags,
            "models": models,
            "layers": layers,
            "polarity": polarity(verdict),
        })

    # ===== 统计 =====
    lines = []
    W = 78
    lines.append("=" * W)
    lines.append("R1 数据底册: 复用机制证据清单 (source: atlas_ledger.json v2)")
    lines.append("生成: p3090_reuse_inventory.py | 纯文本分类, 无前向")
    lines.append("=" * W)
    lines.append("total_measurements: %d" % len(items))
    lines.append("")

    # tag x model 覆盖矩阵
    all_tags = [t[0] for t in TAG_RULES]
    all_models = ["qwen3-4b", "qwen3-14b", "glm4", "ds7b", "qwen25"]
    mat = defaultdict(int)          # (tag, model) -> count
    tag_total = Counter()
    for it in items:
        for t in it["tags"]:
            tag_total[t] += 1
            for m in it["models"]:
                mat[(t, m)] += 1

    lines.append("--- 覆盖矩阵: 机制标签 x 模型 (条数) ---")
    header = "%-20s" % "tag" + "".join("%12s" % m for m in all_models) + "%8s" % "TOTAL"
    lines.append(header)
    for t in all_tags:
        row = "%-20s" % t + "".join("%12d" % mat.get((t, m), 0) for m in all_models)
        row += "%8d" % tag_total.get(t, 0)
        lines.append(row)
    lines.append("")

    # 标签说明
    lines.append("--- 标签图例 ---")
    for name, _, desc in TAG_RULES:
        lines.append("%-20s %s" % (name, desc))
    lines.append("")

    # 层位分布 (仅 L13-L45 感兴趣区)
    layer_counter = Counter()
    for it in items:
        for L in it["layers"]:
            if 8 <= L <= 48:
                layer_counter[L] += 1
    lines.append("--- 层位覆盖 (L8-L48 提及次数) ---")
    if layer_counter:
        for L in sorted(layer_counter):
            bar = "#" * min(50, layer_counter[L])
            lines.append("L%-3d %4d %s" % (L, layer_counter[L], bar))
    else:
        lines.append("(无层位信息)")
    lines.append("")

    # polarity 分布
    pol_counter = Counter(it["polarity"] for it in items)
    lines.append("--- 判决极性分布 (关键词启发式, 仅供参考) ---")
    for k in ["positive", "negative", "mixed", "neutral"]:
        lines.append("%-10s %4d" % (k, pol_counter.get(k, 0)))
    lines.append("")

    # 缺口分析: 空格 = 机制标签在哪些模型上 0 证据
    lines.append("--- 缺口分析 (tag x model = 0) ---")
    gaps = []
    for t in all_tags:
        for m in all_models:
            if mat.get((t, m), 0) == 0 and tag_total.get(t, 0) > 0:
                gaps.append((t, m))
            elif tag_total.get(t, 0) == 0 and m == all_models[0]:
                gaps.append((t, "<ALL: tag无任何证据>"))
    if gaps:
        for t, m in gaps:
            lines.append("GAP  %-20s x %-10s = 0" % (t, m))
    else:
        lines.append("(无空格)")
    lines.append("")

    # 无标签测量清单
    untagged = [it for it in items if not it["tags"]]
    lines.append("--- 未匹配任何复用标签的测量 (%d 条) ---" % len(untagged))
    for it in untagged[:40]:
        lines.append("  %s (P%s) [%s] %s" % (it["meas_id"], it["phase"], it["polarity"], it["verdict"][:60]))
    if len(untagged) > 40:
        lines.append("  ... 共 %d 条" % len(untagged))
    lines.append("")

    # 每标签 top 代表条目 (按 phase 降序取 5)
    lines.append("--- 每标签代表条目 (phase 降序, 最多 5 条) ---")
    for t in all_tags:
        members = [it for it in items if t in it["tags"]]
        members.sort(key=lambda x: x["phase"], reverse=True)
        lines.append("[%s] n=%d" % (t, len(members)))
        for it in members[:5]:
            lines.append("   %s (P%s) [%s] %s" % (it["meas_id"], it["phase"], it["polarity"], it["verdict"][:70]))
    lines.append("=" * W)
    lines.append("END")

    report = "\n".join(lines)
    with io.open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"n_total": len(items), "items": items,
                   "tag_legend": {n: d for n, _, d in TAG_RULES}},
                  f, ensure_ascii=False, indent=1)
    print("OK items=%d report=%s" % (len(items), OUT_TXT))

if __name__ == "__main__":
    main()
