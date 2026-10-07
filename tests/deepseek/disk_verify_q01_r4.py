# -*- coding: utf-8 -*-
"""Q01 独立磁盘复核：从源文件重算，与交付件比对。不引用生成器中间态。
产物：tests/deepseek/result/disk_verify_q01_r4.txt
"""
import os, re, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "deepseek", "result")
PASS, FAIL = [], []
def chk(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
def rd(p): return open(os.path.join(ROOT, p), "rb").read()

PL   = "tests/glm5/result/rdc_query_construction_20260913/phase3103/omega_p101_formula_audit/proposition_ledger.json"
LG   = "research/gpt5/atlas/atlas_ledger.json"
MEMO = "research/gpt5/docs/AGI_GPT5_MEMO.md"
TP   = "research/gpt5/docs/RDC_TESTPLAN_v1.md"
MD   = "research/gpt5/docs/META_SINGLE_SOURCE_Q01.md"
JS   = "research/gpt5/atlas/meta_single_source_v4.json"

pb = rd(PL); P = json.loads(pb.decode("utf-8-sig"))
lb = rd(LG); L = json.loads(lb.decode("utf-8-sig"))
md = rd(MD).decode("utf-8")
js = json.loads(rd(JS).decode("utf-8"))
memo = rd(MEMO).decode("utf-8-sig")
tp   = rd(TP).decode("utf-8-sig")

# ---------- 1 源指纹 ----------
chk("源 proposition_ledger sha8=9a3c6ff4", sha8(pb) == "9a3c6ff4", sha8(pb))
chk("源 atlas_ledger sha8=bbda63df", sha8(lb) == "bbda63df", sha8(lb))
chk("交付 md 已落盘(>3000B)", len(md.encode("utf-8")) > 3000, len(md.encode("utf-8")))
chk("交付 json schema==rdc_meta_single_source_v4",
    isinstance(js, dict) and js.get("schema") == "rdc_meta_single_source_v4")

# ---------- 2 条数 ----------
rev, new = P["propositions_review"], P["propositions_new"]
chk("62 条 = review 57 + new 5", len(rev) == 57 and len(new) == 5, "%d+%d" % (len(rev), len(new)))

# ---------- 3 component 独立重算（regex 分词，非 Counter 拆分） ----------
TAG = re.compile(r"[ABCDE]")
comp = {g: 0 for g in "ABCDE"}
for p in list(rev) + list(new):
    for g in TAG.findall(str(p.get("grade") or "").replace(" ", "")):
        comp[g] += 1
chk("component 重算 == A5/B34/C14/D21/E20", comp == {"A": 5, "B": 34, "C": 14, "D": 21, "E": 20}, str(comp))
chk("component 分量和 == 94", sum(comp.values()) == 94, sum(comp.values()))

# ---------- 4 atomic 四口径独立重算 ----------
first = {g: 0 for g in "ABCDE"}; last = {g: 0 for g in "ABCDE"}
best  = {g: 0 for g in "ABCDE"}; worst = {g: 0 for g in "ABCDE"}
for p in list(rev) + list(new):
    gs = TAG.findall(str(p.get("grade") or "").replace(" ", ""))
    if not gs: continue
    first[gs[0]] += 1
    last[gs[-1]] += 1
    best[sorted(gs, key=lambda x: "ABCDE".index(x))[0]] += 1
    worst[sorted(gs, key=lambda x: "EDCBA".index(x))[0]] += 1
chk("atomic_first  == A3/B27/C10/D9/E13",  first == {"A": 3, "B": 27, "C": 10, "D": 9, "E": 13},  str(first))
chk("atomic_last   == A5/B18/C11/D18/E10", last  == {"A": 5, "B": 18, "C": 11, "D": 18, "E": 10}, str(last))
chk("atomic_best   == A5/B34/C11/D9/E3",   best  == {"A": 5, "B": 34, "C": 11, "D": 9, "E": 3},   str(best))
chk("atomic_worst  == A3/B11/C10/D18/E20", worst == {"A": 3, "B": 11, "C": 10, "D": 18, "E": 20}, str(worst))

# ---------- 5 比例 ----------
r1 = 100.0 * comp["B"] / 62                 # 54.8
r2 = 100.0 * (comp["A"] + comp["B"]) / 62   # 62.9
r3 = 100.0 * (comp["A"] + comp["B"]) / 94   # 41.5
chk("34/62 = 54.84%", abs(r1 - 54.84) < 0.02, "%.4f" % r1)
chk("39/62 = 62.90%", abs(r2 - 62.90) < 0.02, "%.4f" % r2)
chk("39/94 = 41.49%", abs(r3 - 41.49) < 0.02, "%.4f" % r3)

# ---------- 6 哈希语义 ----------
SER = lambda o: json.dumps(o, ensure_ascii=False, indent=1).encode("utf-8")
d2 = {k: v for k, v in L.items() if k != "ledger_sha256_8"}
chk("落盘格式 == dumps(d,ea=False,ind=1)（sha8 相等）", sha8(SER(L)) == sha8(lb), "%s vs %s" % (sha8(SER(L)), sha8(lb)))
chk("content_excluding_self == 0dc6e57a", sha8(SER(d2)) == "0dc6e57a", sha8(SER(d2)))
rec = L.get("ledger_sha256_8")
hits = []
for ea in (True, False):
    for ind in (None, 1, 2):
        for obj, nm in ((L, "full"), (d2, "noself"), (L["measurements"], "meas")):
            if sha8(json.dumps(obj, ensure_ascii=ea, indent=ind).encode("utf-8")) == rec:
                hits.append((nm, ea, ind))
chk("自声明 41d65a13 在 18 候选下全不命中", rec == "41d65a13" and len(hits) == 0, "rec=%s hits=%s" % (rec, hits))

# ---------- 7 measurements schema ----------
ms = L["measurements"]
nm_ = len([m for m in ms if "meas_id" in m])
np_ = len([m for m in ms if "phase" in m])
nb_ = len([m for m in ms if "meas_id" in m and "phase" in m])
chk("304/270/199/165", (len(ms), nm_, np_, nb_) == (304, 270, 199, 165), "%d/%d/%d/%d" % (len(ms), nm_, np_, nb_))
evset = set(m.get("evidence_level") for m in ms)
chk("evidence_level 恒为 statistical", evset == {"statistical"}, str(evset))

# ---------- 8 交付件与重算一致 ----------
for s in ["9a3c6ff4", "bbda63df", "0dc6e57a", "41d65a13", "add57ba7", "41.5%", "54.8%", "62.9%"]:
    chk("md 含 %s" % s, s in md)
chk("md 含 component 表行", "A=5 / B=34 / C=14 / D=21 / E=20" in md)
for bad in ["None", "nan", "{", "}"]:
    if bad in ("{", "}"):
        continue
    chk("md 无 %s 残留" % bad, bad not in md)
chk("json canonical==component", js["grade_counting"]["canonical"] == "component")
chk("json component 表 == 重算", js["grade_counting"]["tables"]["component"] == comp, str(js["grade_counting"]["tables"]["component"]))
chk("json atomic_best 表 == 重算", js["grade_counting"]["tables"]["atomic_best"] == best)
chk("json atomic_first 表 == 重算", js["grade_counting"]["tables"]["atomic_first"] == first)
chk("json reproduced == {memo:True, testplan:False}", js["grade_counting"]["reproduced"] == {"memo_3103": True, "testplan": False})
chk("json hash correct==0dc6e57a", js["hash_policy"]["atlas_ledger"]["correct_self_excluding"] == "0dc6e57a")
chk("json discrepancies 5 条", len(js["discrepancies"]) == 5, len(js["discrepancies"]))
chk("json corrections 6 条全 awaiting_seal",
    len(js["corrections"]) == 6 and all(c.get("status") == "awaiting_seal" for c in js["corrections"]))
chk("json measurements 计数 == 重算",
    (js["measurements_schema"]["n_total"], js["measurements_schema"]["n_has_meas_id"],
     js["measurements_schema"]["n_has_phase"], js["measurements_schema"]["n_both"]) == (len(ms), nm_, np_, nb_))

# ---------- 9 源文本核对（证明 O1/O2 的存在） ----------
chk("MEMO 原文含 add57ba7（D4 值甲）", "add57ba7" in memo)
chk("MEMO 原文含 A=5，B=34（D1 值甲）", "A=5，B=34" in memo)
chk("TESTPLAN 原文含 A=5 / B=6 / C=10 / D=21 / E=20（D1 值乙）",
    "A=5 / B=6 / C=10 / D=21 / E=20" in tp)

# ---------- 10 上游治理文件未变（上位依据指纹） ----------
CONST_REF = "01df6398"
chk("宪法 sha8 仍为 %s（未被本轮改动）" % CONST_REF,
    sha8(rd("research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md")) == CONST_REF,
    sha8(rd("research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md")))
chk("ledger 未被本轮改动（仍 304 条 / bbda63df）", len(ms) == 304 and sha8(lb) == "bbda63df")

# ---------- 汇总 ----------
L_ = ["Q01 独立磁盘复核（源文件重算）", "=" * 60]
for n, d in PASS:
    L_.append("[PASS] %s%s" % (n, ("  <%s>" % d) if d else ""))
for n, d in FAIL:
    L_.append("[FAIL] %s%s" % (n, ("  <%s>" % d) if d else ""))
L_.append("=" * 60)
L_.append("PASS=%d  FAIL=%d  ->  %s" % (len(PASS), len(FAIL), "ALL_PASS" if not FAIL else "HAS_FAIL"))
open(os.path.join(OUT, "disk_verify_q01_r4.txt"), "wb").write(("\n".join(L_) + "\n").encode("utf-8"))
print("PASS=%d FAIL=%d -> %s" % (len(PASS), len(FAIL), "ALL_PASS" if not FAIL else "HAS_FAIL"))
for n, d in FAIL:
    print("  FAIL:", n, d)
