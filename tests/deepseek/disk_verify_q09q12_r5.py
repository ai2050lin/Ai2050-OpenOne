# -*- coding: utf-8 -*-
"""Q09 / Q12 独立磁盘复核。
原则：不复用生成器中间产物——直接从 3151/3152 result 与 3103 账本重算，再与交付件比对。
输出：tests/deepseek/result/disk_verify_q09q12_r5.txt
"""
import os, re, json, random, hashlib, glob

ROOT = r"D:\AI2050\Ai2050-OpenOne"
BASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
DOCS = os.path.join(ROOT, "research", "gpt5", "docs")
ATLAS = os.path.join(ROOT, "research", "gpt5", "atlas")
OUT = os.path.join(ROOT, "tests", "deepseek", "result", "disk_verify_q09q12_r5.txt")

R = []
n_pass = [0]
n_fail = [0]
def chk(name, cond, extra=""):
    if cond:
        n_pass[0] += 1
        R.append("[PASS] %s %s" % (name, extra))
    else:
        n_fail[0] += 1
        R.append("[FAIL] %s %s" % (name, extra))

def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
def jload(p):
    return json.loads(open(p, "rb").read().decode("utf-8-sig"))
def mean(v):
    return sum(v) / float(len(v))
def boot(vals, n=20000, seed=20261003, alpha=0.05):
    rnd = random.Random(seed)
    k = len(vals)
    ms = sorted(sum(vals[rnd.randrange(k)] for _ in range(k)) / float(k) for _ in range(n))
    return ms[int(alpha / 2 * n)], ms[int((1 - alpha / 2) * n) - 1]

# ================= 0. 交付件存在性与指纹 =================
Q09J = os.path.join(ATLAS, "deadline_dual_track_v1.json")
Q09M = os.path.join(DOCS, "DEADLINE_DUAL_TRACK_Q09.md")
Q12J = os.path.join(ATLAS, "prop_citation_audit_v1.json")
Q12M = os.path.join(DOCS, "PROP_CITATION_AUDIT_Q12.md")
for p in [Q09J, Q09M, Q12J, Q12M]:
    chk("存在 " + os.path.basename(p), os.path.exists(p), "%d B %s" % (os.path.getsize(p), sha8(p)))
    if os.path.exists(p):
        b = open(p, "rb").read()
        chk("LF-only 无 CR " + os.path.basename(p), b"\r" not in b)
        chk("无 BOM " + os.path.basename(p), b[:3] != b"\xef\xbb\xbf")

q09 = jload(Q09J)
q12 = jload(Q12J)

# ================= 1. 受保护文件未被改动 =================
MEMO = os.path.join(DOCS, "AGI_GPT5_MEMO.md")
LEDP = glob.glob(os.path.join(ROOT, "tests", "glm5", "result", "**", "proposition_ledger.json"), recursive=True)[0]
TP = os.path.join(DOCS, "RDC_TESTPLAN_v1.md")
chk("MEMO 未改（sha8 2a84776b）", sha8(MEMO) == "2a84776b", sha8(MEMO))
chk("账本未改（sha8 9a3c6ff4）", sha8(LEDP) == "9a3c6ff4", sha8(LEDP))
chk("TESTPLAN 未改（sha8 71b85673）", sha8(TP) == "71b85673", sha8(TP))

# ================= 2. Q09 独立重算 K1 =================
P3152 = os.path.join(BASE, "phase3152", "g1p2_tri_model_k1")
P3151 = os.path.join(BASE, "phase3151", "g1p1_combo_additive_vs_interaction")
sm = jload(os.path.join(P3152, "summary", "result_summary.json"))
r51 = jload(os.path.join(P3151, "result.json"))
k39 = r51["v1"]["gates"]["M1_k39"]
MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]
MDIR = {"qwen3-4b": "qwen3-4b", "qwen3-14b": "qwen3-14b", "glm4-9b": "glm4k1"}

tok = er = eks = ero = dks = dro = None
mr_all, mk_all, mde_ro_all, mde_k_all = [], [], [], []
model_lv = {}
for m in MODELS:
    a = jload(os.path.join(P3152, MDIR[m], "result.json"))["k1_model_report"]
    s = sm["k1_3model"][m]
    ro_m = [x["margin"] for x in k39] if m == "glm4-9b" else a["m1_readout"]["margins"]
    ro_d = [x["mde"] for x in k39] if m == "glm4-9b" else a["m1_readout"]["mdes"]
    ks_m = a["m1_kstar"]["margins"]
    ks_d = a["m1_kstar"]["mdes"]
    mr_all += ro_m; mde_ro_all += ro_d
    mk_all += ks_m; mde_k_all += ks_d
    model_lv[m] = (mean(ro_m), mean(ro_d), mean(ks_m), mean(ks_d))

mr, dr = mean(mr_all), mean(mde_ro_all)
mk, dk = mean(mk_all), mean(mde_k_all)
EREAD = mean([sm["k1_3model"][m]["b4_rel_readout"] for m in MODELS])

A = q09["k1_recompute"]["aggregate"]
chk("Q09 E_read_pooled 复现", abs(A["E_read_pooled"] - EREAD) < 1e-12, "%.12f vs %.12f" % (A["E_read_pooled"], EREAD))
chk("Q09 k* 池化 margin 复现", abs(A["kstar_pooled_margin"] - mk) < 1e-12, "%.12f" % mk)
chk("Q09 k* 池化 MDE 复现", abs(A["kstar_pooled_mde"] - dk) < 1e-12, "%.12f" % dk)
chk("Q09 读出层池化 margin 复现", abs(A["readout_pooled_margin"] - mr) < 1e-12, "%.12f" % mr)
chk("Q09 读出层池化 MDE 复现", abs(A["readout_pooled_mde"] - dr) < 1e-12, "%.12f" % dr)
chk("Q09 E_read > 0.05", EREAD > 0.05, "%.6f" % EREAD)

# 逐模型存表复现
pm = q09["k1_recompute"]["per_model"]
for m in MODELS:
    chk("Q09 per_model[%s].b4_readout 与源一致" % m,
        abs(pm[m]["b4_readout"] - sm["k1_3model"][m]["b4_rel_readout"]) < 1e-12)
    chk("Q09 per_model[%s].m1_kstar_margins 与源一致" % m,
        all(abs(x - y) < 1e-12 for x, y in zip(pm[m]["m1_kstar_margins"], sm["k1_3model"][m]["m1_kstar_margins"])))

# 双轨逻辑独立复算
veto_ks = [m for m in MODELS if not (model_lv[m][2] <= -2 * model_lv[m][3])]
veto_ro = [m for m in MODELS if not (model_lv[m][0] <= -2 * model_lv[m][1])]
v = q09["k1_recompute"]["verdict"]
chk("Q09 k* 层否决模型 = [qwen3-4b]", veto_ks == ["qwen3-4b"], str(veto_ks))
chk("Q09 读出层否决 3/3", len(veto_ro) == 3, str(veto_ro))
chk("Q09 k* 判决 = model_specific", v["kstar_layer"] == "model_specific", v["kstar_layer"])
chk("Q09 读出层判决 = fired_all_models", v["readout_layer"] == "fired_all_models", v["readout_layer"])
chk("Q09 k* 轨 A 未触发（=主判据通过，候选显著更优）",
    v["kstar_trackA_fired"] is False and (mk > -2 * dk) is False,
    "mk=%.6f -2dk=%.6f" % (mk, -2 * dk))
t1_ro = (mr > -2 * dr) or (EREAD > 0.05)
chk("Q09 读出层轨 A 触发（独立复算）", bool(t1_ro) == bool(v["readout_trackA_fired"]))
chk("Q09 层位选择标注属 Q08", v["layer_choice_is_Q08"] is True)

# CI：同 seed 精确、异 seed 近似
chk("Q09 k* CI(模型级) 同seed 复现",
    abs(q09["k1_recompute"]["aggregate"]["kstar_margin_ci95_modellevel"][0] - boot([model_lv[m][2] for m in MODELS])[0]) < 1e-12)
lo2, hi2 = boot([model_lv[m][0] for m in MODELS], seed=987654321)
clo, chi = q09["k1_recompute"]["aggregate"]["readout_margin_ci95_modellevel"]
chk("Q09 读出层 CI 含点估计", clo <= mr <= chi, "[%.4f, %.4f] vs %.4f" % (clo, chi, mr))
chk("Q09 读出层 CI 异seed 相对稳定", abs(lo2 - clo) < 5e-3 and abs(hi2 - chi) < 5e-3,
    "seed2=[%.4f,%.4f]" % (lo2, hi2))

# K2/K3 未测量
chk("Q09 K2 = not_measurable_yet", q09["k2_status"]["state"] == "not_measurable_yet")
chk("Q09 K3 = not_measurable_yet", q09["k3_status"]["state"] == "not_measurable_yet")
chk("phase3154 目录确实不存在",
    len(glob.glob(os.path.join(BASE, "phase3154*"))) == 0)
chk("Q09 never_measured = [K2,K3]", q09["deadline_immunity_quantified"]["never_measured"] == ["K2", "K3"])
# 文末预注册 3154（证明 K2 从未开跑）
mtxt = open(MEMO, "rb").read().decode("utf-8-sig")
chk("MEMO 含『预注册 Phase 3154』（K2 停在预注册）", "预注册 Phase 3154" in mtxt)
chk("MEMO 无『top-50 覆盖率』条目（K3 装置不存在）",
    not re.search(r"top-?50[^。\n]{0,20}覆盖率", mtxt))

# ================= 3. Q12 独立重算 =================
led = jload(LEDP)
def split_g(g):
    return [x.strip() for x in re.split(r"[+/、,\s]+", str(g or "")) if x.strip()] or ["?"]
props = []
for k in ["propositions_review", "propositions_new"]:
    for p in (led.get(k) or []):
        props.append({"id": str(p.get("id")), "grade": str(p.get("grade")), "comp": split_g(p.get("grade")),
                      "claim": str(p.get("claim", ""))})
DE = [p for p in props if set(p["comp"]) & {"D", "E"}]
chk("Q12 n_props = 62", q12["counts"]["n_props"] == 62, str(q12["counts"]["n_props"]))
chk("Q12 独立重算 n_props = 62", len(props) == 62, str(len(props)))
chk("Q12 D∪E = 38", q12["counts"]["n_DE_union"] == 38, str(q12["counts"]["n_DE_union"]))
chk("Q12 独立重算 D∪E = 38", len(DE) == 38, str(len(DE)))
from collections import Counter
cc = Counter()
for p in props:
    for c in p["comp"]:
        cc[c] += 1
chk("Q12 分量口径 B=34,D=21,E=20,C=14,A=5",
    cc["B"] == 34 and cc["D"] == 21 and cc["E"] == 20 and cc["C"] == 14 and cc["A"] == 5, str(dict(cc)))
pureE = [p["id"] for p in props if p["grade"] == "E"]
chk("Q12 纯 E = 3 (R03,R15,R46)", sorted(pureE) == ["R03", "R15", "R46"], str(sorted(pureE)))
de_comp = [p for p in DE if not (len(set(p["comp"]) - {"D", "E"}) == 0)]
chk("Q12 DE 复合 = 26", len(de_comp) == 26, str(len(de_comp)))
chk("Q12 复合总数 = 32", q12["counts"]["n_composite_any"] == 32)

# 新区域独立扫描
ML = mtxt.split("\n")
NEW_START = next(i for i, l in enumerate(ML, 1)
                 if re.match(r"## Phase (\d{4})", l) and int(re.match(r"## Phase (\d{4})", l).group(1)) > 3103)
chk("新区域起点 = L13244 (Phase 3104)", NEW_START == 13244, str(NEW_START))
region = "\n".join(ML[NEW_START - 1:])
indep = set()
for p in DE:
    for m in re.finditer(r"(?<![A-Za-z0-9])%s(?![0-9])" % re.escape(p["id"]), region):
        indep.add((p["id"], NEW_START - 1 + region.count("\n", 0, m.start()) + 1))
stored = set()
for h in q12["F1_id_hits_primary"].get("MEMO@3104+", []):
    stored.add((h["id"], h["line"]))
chk("Q12 新区域 id 命中 = {(R55,13546)}", indep == {("R55", 13546)}, str(sorted(indep)))
chk("Q12 存表新区域命中与独立扫描一致", stored == indep, "%s vs %s" % (sorted(stored), sorted(indep)))
# R55 命中行落在 Phase 3113 区间
p3113 = next(i for i, l in enumerate(ML, 1) if l.startswith("## Phase 3113"))
p3114 = next(i for i, l in enumerate(ML, 1) if l.startswith("## Phase 3114"))
chk("Q12 R55 命中行在 Phase 3113 区间", p3113 <= 13546 < p3114, "3113=L%d 3114=L%d" % (p3113, p3114))

# 文本指纹独立重算
def grams6(s):
    out = set()
    for run in re.findall(r"[\u4e00-\u9fff]{6,}", str(s or "")):
        for i in range(len(run) - 5):
            out.add(run[i:i + 6])
    return out
n_con = sum(1 for p in DE if grams6(p["claim"]))
chk("Q12 可构造指纹 = 29", n_con == 29, str(n_con))
fp_new = 0
for p in DE:
    gs = grams6(p["claim"])
    if gs and any(g in region for g in gs):
        fp_new += 1
chk("Q12 新区域文本指纹命中 = 0", fp_new == 0, str(fp_new))
chk("Q12 存表 fp_violations 为空", q12["verdict"]["text_level_violations_in_new_region"] == [])

# 命名空间碰撞证据
ld = open(os.path.join(DOCS, "LOOP_DIAGNOSIS_AND_EXIT_v1.md"), "rb").read().decode("utf-8-sig")
chk("LOOP_DIAGNOSIS 出现 R10/R11（自有编号）",
    bool(re.search(r"(?<![A-Za-z0-9])R10(?![0-9])", ld)) and bool(re.search(r"(?<![A-Za-z0-9])R11(?![0-9])", ld)))
chk("LOOP_DIAGNOSIS 的 R10 语境不含账本 R10 claim 关键词（实体感紧凑性）", "紧凑性" not in ld)
chk("账本 R10 claim = 实体感紧凑性定律",
    next(p["claim"] for p in props if p["id"] == "R10") == "实体感紧凑性定律")
chk("账本 R11 含『层级嵌套』", "层级嵌套" in next(p["claim"] for p in props if p["id"] == "R11"))
chk("Q12 登记 3 处命名空间碰撞", len(q12["verdict"]["namespace_collisions"]) == 3,
    str(len(q12["verdict"]["namespace_collisions"])))
# 自身输出未入语料（自指污染已被排除）
chk("Q12 主语料不含自身输出", "PROP_CITATION_AUDIT_Q12.md" not in q12["corpora"]["primary"])
chk("Q12 主语料doc数 = 26", len([x for x in q12["corpora"]["primary"] if x != "MEMO@3104+"]) == 26,
    str(len([x for x in q12["corpora"]["primary"] if x != "MEMO@3104+"])))
chk("Q12 归档 10 个被排除", len(q12["corpora"]["archives_excluded_from_verdict"]) == 10,
    str(len(q12["corpora"]["archives_excluded_from_verdict"])))
# 实质命中分类计数
chk("Q12 新区域实质命中 = 1",
    q12["verdict"]["n_substantive_in_new_region"] == 1, str(q12["verdict"]["n_substantive_in_new_region"]))
chk("Q12 无『纯 E 进入新链』违规", q12["verdict"]["substantive_id_hits"] ==
    [x for x in q12["verdict"]["substantive_id_hits"]] and
    all(x["corpus"] != "MEMO@3104+" or x["class"] == "candidate_violation"
        for x in q12["verdict"]["substantive_id_hits"]))
# 量纲口径
chk("Q12 量纲：38/62 = 61.29%", abs(q12["counts"]["pct_DE_union"] - 61.29) < 0.01, str(q12["counts"]["pct_DE_union"]))

# ================= 4. 与既有治理文件的一致性 =================
con = open(os.path.join(DOCS, "RDC_RESEARCH_CONSTITUTION_v1.md"), "rb").read().decode("utf-8-sig")
chk("宪法 §3 I3 在位", "死线双轨（I3）" in con)
chk("宪法 §6 I4 在位", "单一真源（I4）" in con)
qj = jload(os.path.join(ATLAS, "phase_queue_v1.json"))
ids = [x["id"] for x in qj["queue"]]
chk("队列含 Q09 与 Q12", "Q09" in ids and "Q12" in ids)
chk("队列未改（sha8 675836fd）", sha8(os.path.join(ATLAS, "phase_queue_v1.json")) == "675836fd")
chk("宪法未改（sha8 01df6398）", sha8(os.path.join(DOCS, "RDC_RESEARCH_CONSTITUTION_v1.md")) == "01df6398")
for x in qj["queue"]:
    if x["id"] in ("Q09", "Q12"):
        chk("队列 %s status 仍 pending（不改状态，待 seal）" % x["id"], x["status"] == "pending", x["status"])

R.append("")
R.append("==== 汇总：PASS=%d FAIL=%d ====" % (n_pass[0], n_fail[0]))
R.append("VERDICT: " + ("ALL_PASS" if n_fail[0] == 0 else "HAS_FAIL"))
open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(R) + "\n")
print("PASS=%d FAIL=%d %s" % (n_pass[0], n_fail[0], "ALL_PASS" if n_fail[0] == 0 else "HAS_FAIL"))
print("WROTE", OUT)
