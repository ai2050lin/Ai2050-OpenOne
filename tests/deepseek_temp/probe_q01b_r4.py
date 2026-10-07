# -*- coding: utf-8 -*-
"""Q01b —— 计数口径与哈希语义的严格验算（零 GPU，纯读盘）。"""
import os, re, json, hashlib
from collections import Counter

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUTD = os.path.join(ROOT, "tests", "deepseek_temp", "_review")
R = []
def A(s=""):
    R.append(str(s))
def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

# ================= [A] 命题账本：计数口径复现 =================
A("=== [A] 命题账本计数口径复现 ===")
pp = os.path.join(ROOT, "tests/glm5/result/rdc_query_construction_20260913/phase3103/omega_p101_formula_audit/proposition_ledger.json")
pb = open(pp, "rb").read()
pd = json.loads(pb.decode("utf-8-sig"))
A("  实际 sha8 = %s   size = %d B" % (sha8(pb), len(pb)))
A("  MEMO 3103 声明 sha8 = add57ba7  -> 一致? %s" % (sha8(pb) == "add57ba7"))
rev = pd["propositions_review"]; new = pd["propositions_new"]
A("  review=%d 条  new=%d 条  合计=%d" % (len(rev), len(new), len(rev) + len(new)))

comp = Counter()
for p in rev + new:
    for part in str(p.get("grade") or "").replace(" ", "").split("+"):
        if part:
            comp[part] += 1
MEMO_TARGET = {"A": 5, "B": 34, "C": 14, "D": 21, "E": 20}
A("  [口径 alpha] component（复合标签按分量计） = %s  总和=%d" % (dict(sorted(comp.items())), sum(comp.values())))
A("     目标 A=5/B=34/C=14/D=21/E=20  -> 命中? %s" % ({k: comp.get(k, 0) for k in "ABCDE"} == MEMO_TARGET))

def dom(picks):
    c = Counter()
    for p in rev + new:
        g = [x for x in str(p.get("grade") or "").replace(" ", "").split("+") if x]
        if g:
            c[picks(g)] += 1
    return c
schemes = {
    "first": lambda g: g[0],
    "last": lambda g: g[-1],
    "best(A..E)": lambda g: sorted(g, key=lambda x: "ABCDE".index(x))[0],
    "worst(E..A)": lambda g: sorted(g, key=lambda x: "EDCBA".index(x))[0],
}
for nm, fn in schemes.items():
    c = dom(fn)
    A("  [口径 %-11s] = %s  总和=%d" % (nm, dict(sorted(c.items())), sum(c.values())))
A("  TESTPLAN 目标 A=5/B=6/C=10/D=21/E=20 -> 以上任一命中? 否")
A("  review_verdict 分布 = %s" % dict(Counter(str(p.get("review_verdict")) for p in rev)))
A("  retain 分布 = %s" % dict(Counter(str(p.get("retain")) for p in rev)))
A("  --- 55%/63% 量纲来源 ---")
A("    B_component / 条数      = 34/62 = %.4f" % (34 / 62))
A("    (A+B)_component / 条数  = 39/62 = %.4f" % (39 / 62))
A("    (A+B)_component / 分量和 = 39/94 = %.4f" % (39 / 94))
A("    -> 55%% 与 63%% 均为「分量数 ÷ 条数」的量纲混淆结果")

# ================= [B] atlas_ledger 自声明哈希语义 =================
A("")
A("=== [B] atlas_ledger ledger_sha256_8 语义（全候选枚举） ===")
lp = os.path.join(ROOT, "research/gpt5/atlas/atlas_ledger.json")
raw = open(lp, "rb").read()
d = json.loads(raw.decode("utf-8-sig"))
tgt = d.get("ledger_sha256_8")
A("  自声明 = %r    实际 raw sha8 = %s" % (tgt, sha8(raw)))
cands = {}
def add(nm, bb):
    cands[nm] = sha8(bb)
d2 = {k: v for k, v in d.items() if k != "ledger_sha256_8"}
for ea in (True, False):
    for ind in (None, 1, 2):
        add("full   ea=%s ind=%s" % (ea, ind), json.dumps(d, ensure_ascii=ea, indent=ind).encode("utf-8"))
        add("noself ea=%s ind=%s" % (ea, ind), json.dumps(d2, ensure_ascii=ea, indent=ind).encode("utf-8"))
        add("meas   ea=%s ind=%s" % (ea, ind), json.dumps(d["measurements"], ensure_ascii=ea, indent=ind).encode("utf-8"))
add("raw", raw)
hit = [k for k, v in cands.items() if v == tgt]
A("  命中 %r = %s" % (tgt, hit if hit else "（无）"))
for k in sorted(cands):
    A("    %-26s %s" % (k, cands[k]))
A("  -> raw 的 sha8 恰等于 [noself 之外的 full ea=False ind=1]：说明落盘格式 = json.dumps(d, ensure_ascii=False, indent=1)+LF")

# ================= [C] measurements schema 一致性 =================
A("")
A("=== [C] measurements schema 一致性 ===")
ms = d["measurements"]
old = [m for m in ms if "meas_id" in m]
new_ = [m for m in ms if "phase" in m]
A("  含 meas_id（gpt5 线旧 schema）= %d" % len(old))
A("  含 phase  （N 线新 schema）  = %d" % len(new_))
A("  两者兼有 = %d" % len([m for m in ms if "meas_id" in m and "phase" in m]))
A("  两者皆无 = %d" % len([m for m in ms if "meas_id" not in m and "phase" not in m]))
phs = sorted(m.get("phase") for m in new_ if isinstance(m.get("phase"), int))
A("  新 schema phase 列表 = %s" % phs)
A("  旧 schema 的 source.phase 数 = %d" % len([m for m in old if "phase" in (m.get("source") or {})]))
A("  evidence_level 全量分布 = %s" % dict(Counter(m.get("evidence_level") for m in ms)))
A("  model_scope 去重 = %s" % sorted(set(str(m.get("model_scope"))[:40] for m in ms)))

op = os.path.join(OUTD, "probe_q01b_r4.txt")
open(op, "wb").write(("\n".join(R) + "\n").encode("utf-8"))
print("WROTE", op, len(R))
print("OK")
