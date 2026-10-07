# -*- coding: utf-8 -*-
"""R5 探针 E：区域边界 + 等级计数 + R55 定位 + 文本指纹规模。只读。"""
import os, re, json, glob, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "probe_q09q12e_r5.txt")
R = []
def A(s=""):
    R.append(s)

MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
t = open(MEMO, "rb").read().decode("utf-8-sig")
L = t.split("\n")

# --- 区域边界 ---
h2 = [(i, l) for i, l in enumerate(L, 1) if l.startswith("## ")]
A("=== 3100+ 的 h2 标题（定位 3103 之后的下一个 Phase 起点） ===")
for i, l in h2:
    m = re.search(r"Phase\s+(\d{4})", l)
    if m and 3099 <= int(m.group(1)) <= 3160:
        A("  L%-6d %s" % (i, l[:140]))

# --- 等级计数 ---
led_path = glob.glob(os.path.join(ROOT, "tests", "glm5", "result", "**", "proposition_ledger.json"), recursive=True)[0]
led = json.loads(open(led_path, "rb").read().decode("utf-8-sig"))
props = []
for k in ["propositions_review", "propositions_new"]:
    for p in (led.get(k) or []):
        props.append(p)

def split_g(g):
    return [x.strip() for x in re.split(r"[+/、,\s]+", str(g or "")) if x.strip()] or ["?"]

A("")
A("=== 等级口径计数 ===")
from collections import Counter
raw = Counter(str(p.get("grade")) for p in props)
A("  原始复合标签 %d 种: %s" % (len(raw), dict(raw)))
pure = Counter()
for p in props:
    c = split_g(p.get("grade"))
    if len(c) == 1:
        pure[c[0]] += 1
A("  纯标签（单分量）计数: %s" % dict(pure))
comp_cnt = sum(1 for p in props if len(split_g(p.get("grade"))) > 1)
A("  复合标签条数 = %d / %d" % (comp_cnt, len(props)))
de = [p for p in props if set(split_g(p.get("grade"))) & {"D", "E"}]
A("  D∪E 条数 = %d (%.2f%%)" % (len(de), 100.0 * len(de) / len(props)))
pureDE = [p for p in de if set(split_g(p.get("grade"))) <= {"D", "E"}]
A("  纯 D/E（不含 A/B/C）= %d" % len(pureDE))
A("  纯 E（grade=='E'）= %d : %s" % (pure.get("E", 0), [p["id"] for p in props if str(p.get("grade")) == "E"]))
A("  纯 D（grade=='D'）= %d : %s" % (pure.get("D", 0), [p["id"] for p in props if str(p.get("grade")) == "D"]))
A("  D∪E 中复合（含 A/B/C 之一）= %d" % (len(de) - len(pureDE)))

# --- R55 定位（3103 之后的 Phase） ---
A("")
A("=== R55 在 3103 之后区域的命中与所属 Phase ===")
idx3104 = None
for i, l in enumerate(L, 1):
    if l.startswith("## ") and re.search(r"Phase\s+(\d{4})", l):
        n = int(re.search(r"Phase\s+(\d{4})", l).group(1))
        if n > 3103:
            idx3104 = i
            break
A("  下一个 >3103 的 Phase 标题行 = L%s: %s" % (idx3104, L[idx3104 - 1][:120] if idx3104 else "N/A"))
if idx3104:
    region = list(enumerate(L[idx3104 - 1:], idx3104))
    cur_phase = None
    hits = []
    for i, l in region:
        if l.startswith("## "):
            cur_phase = l[:100]
        if re.search(r"(?<![A-Za-z0-9])R55(?![0-9])", l):
            hits.append((i, cur_phase, l.strip()))
    A("  3104+ 区域 R55 命中 %d 次：" % len(hits))
    for i, ph, l in hits:
        A("    L%-6d [%s]" % (i, ph))
        A("       %s" % l[:220])

# --- 其他文档的 R 命名空间占用 ---
A("")
A("=== 各文档中 R\\d\\d 模式的 id 集合（判命名空间碰撞） ===")
docs = sorted(glob.glob(os.path.join(ROOT, "research", "gpt5", "docs", "*.md")))
for d in docs:
    txt = open(d, "rb").read().decode("utf-8-sig", "ignore")
    s = sorted(set(re.findall(r"(?<![A-Za-z0-9])(R\d{2})(?![0-9])", txt)))
    if s:
        A("  %-46s %3d 个: %s" % (os.path.basename(d), len(s), ",".join(s[:20]) + ("..." if len(s) > 20 else "")))

# --- 文本指纹（6-CJK-gram）匹配规模 ---
A("")
A("=== 文本指纹（claim 的 6-CJK-gram）在 3104+ 区域的命中规模 ===")
def cjk_runs(s):
    return re.findall(r"[\u4e00-\u9fff]{6,}", str(s or ""))

def grams6(s):
    out = set()
    for run in cjk_runs(s):
        for i in range(len(run) - 5):
            out.add(run[i:i + 6])
    return out

tail = "\n".join(L[(idx3104 - 1):]) if idx3104 else ""
rows = []
for p in de:
    gs = grams6(p.get("claim"))
    if not gs:
        continue
    hits = [g for g in gs if g in tail]
    if hits:
        rows.append((p["id"], p.get("grade"), len(gs), len(hits), sorted(hits)[:3]))
rows.sort(key=lambda r: -r[3])
A("  命中 ≥1 的 D/E 命题 %d 条（共 %d 条可构造指纹）" % (len(rows), len([p for p in de if grams6(p.get("claim"))])))
for pid, g, n, m, ex in rows[:25]:
    A("    [%s] %-6s %2d/%2d 例: %s" % (pid, g, m, n, " / ".join(ex)))

# 反向：3104+ 区域真实内容样本（前 40 行）
A("")
A("=== 3104+ 区域样本（前 40 非空行，了解新推理链内容） ===")
cnt = 0
for i in range(idx3104 - 1, len(L)):
    if L[i].strip():
        A("  L%-6d %s" % (i + 1, L[i][:190]))
        cnt += 1
        if cnt >= 40:
            break

open(OUT, "w", encoding="utf-8").write("\n".join(R))
print("WROTE", OUT, len(R), "lines")
