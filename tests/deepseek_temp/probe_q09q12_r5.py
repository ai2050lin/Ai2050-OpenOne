# -*- coding: utf-8 -*-
"""R5 探针：为 Q09（死线双轨重述）与 Q12（D/E 级引用审计）取证。
只读，不改任何文件。输出写盘到 tests/deepseek_temp/_review/probe_q09q12_r5.txt。
"""
import os, re, json, hashlib, glob

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "probe_q09q12_r5.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

R = []
def A(s=""):
    R.append(s)

def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
t = open(MEMO, "rb").read().decode("utf-8-sig")
L = t.split("\n")
A("=== [0] MEMO 概览 ===")
A("MEMO bytes=%d lines=%d sha8=%s" % (os.path.getsize(MEMO), len(L), sha8(MEMO)))

# ---------- Part A: K1/K2/K3 定义定位 ----------
A("")
A("=== [A] K1/K2/K3 出现位置（含 死线/触发/判据 上下文的行） ===")
pat_kk = re.compile(r"\bK[123]\b")
key = re.compile(r"死线|触发|判据|kill|条件|门槛|门\b|阈值")
hits = []
for i, l in enumerate(L, 1):
    if pat_kk.search(l) and key.search(l):
        hits.append((i, l.strip()))
A("命中行数 = %d" % len(hits))
for i, l in hits[:80]:
    A("  L%-6d %s" % (i, l[:210]))

# 各 K 关键词出现总数
A("")
A("--- 各标记计数 ---")
for k in ["K1", "K2", "K3", "K 死线", "kill criteria", "死线", "触发"]:
    A("  %-16s %d" % (k, t.count(k)))

# ---------- Part B: 命题账本 ----------
A("")
A("=== [B] 命题账本真源 ===")
cands = glob.glob(os.path.join(ROOT, "tests", "glm5", "result", "**", "proposition_ledger.json"), recursive=True)
A("glob 命中 %d 个：" % len(cands))
for c in cands:
    A("   %s  (%d B, sha8 %s)" % (os.path.relpath(c, ROOT), os.path.getsize(c), sha8(c)))

led = None
if cands:
    led_path = cands[0]
    led = json.loads(open(led_path, "rb").read().decode("utf-8-sig"))
    A("")
    A("--- schema ---")
    if isinstance(led, dict):
        for k, v in led.items():
            if isinstance(v, list):
                A("  %s -> list(%d)" % (k, len(v)))
            elif isinstance(v, dict):
                A("  %s -> dict keys=%s" % (k, list(v.keys())[:14]))
            else:
                A("  %s = %r" % (k, v))
    if isinstance(led, dict):
        for keyname in ["propositions_review", "propositions_new"]:
            v = led.get(keyname)
            if isinstance(v, list) and v:
                A("")
                A("--- %s 首条键集 ---" % keyname)
                A("  %s" % sorted(v[0].keys()))
                A("  首条样例: %s" % json.dumps(v[0], ensure_ascii=False)[:600])

# 分级分布（component 口径）
def cmp_split(g):
    parts = [x.strip() for x in re.split(r"[+/、,\s]+", str(g or "")) if x.strip()]
    return parts or ["?"]

if isinstance(led, dict):
    allp = []
    for keyname in ["propositions_review", "propositions_new"]:
        for p in (led.get(keyname) or []):
            allp.append(p)
    A("")
    A("--- 命题总条数 = %d ---" % len(allp))
    gk = None
    for cand_key in ["grade", "level", "tier", "rank"]:
        if allp and cand_key in allp[0]:
            gk = cand_key
            break
    A("grade 键名 = %s" % gk)
    if gk:
        from collections import Counter
        raw = Counter(str(p.get(gk)) for p in allp)
        comp = Counter()
        for p in allp:
            for c in cmp_split(p.get(gk)):
                comp[c] += 1
        A("  原始标签分布 %s" % dict(raw))
        A("  分量口径分布 %s" % dict(comp))
        A("  条数=%d  分量和=%d" % (len(allp), sum(comp.values())))
        # D/E 级条目
        for lv in ["D", "E"]:
            sub = [p for p in allp if lv in cmp_split(p.get(gk))]
            A("")
            A("--- %s 级（分量口径）%d 条 ---" % (lv, len(sub)))
            for p in sub[:12]:
                txt = ""
                for tk in ["claim", "text", "prop", "statement", "desc", "title"]:
                    if tk in p:
                        txt = str(p[tk])
                        break
                pid = p.get("id") or p.get("pid") or p.get("prop_id") or "?"
                A("   [%s] %s | %s" % (pid, str(p.get(gk)), txt[:150]))

# ---------- Part C: 候选引用位点 ----------
A("")
A("=== [C] 候选引用位点（3103 之后的 MEMO 节 + docs 目录） ===")
# MEMO 中 3103 之后是否出现 D/E 命题 id 或文本
idx3103 = None
for i, l in enumerate(L, 1):
    if l.startswith("## ") and "3103" in l:
        idx3103 = i
        A("3103 节标题 L%d: %s" % (i, l[:120]))
A("3103 之后的 MEMO 行数 = %d" % (len(L) - (idx3103 or 0)))

docs = sorted(glob.glob(os.path.join(ROOT, "research", "gpt5", "docs", "*.md")))
A("")
A("docs 目录 .md 共 %d 个：" % len(docs))
for d in docs:
    A("   %-58s %8d B  %s" % (os.path.basename(d), os.path.getsize(d), sha8(d)))

open(OUT, "w", encoding="utf-8").write("\n".join(R))
print("WROTE", OUT, len(R), "lines")
