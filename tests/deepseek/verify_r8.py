# -*- coding: utf-8 -*-
"""R8 独立复核（新进程）：独立重算 + 指纹比对。输出 tests/deepseek/result/verify_r8.txt"""
import os, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek\result\verify_r8.txt")
EXPECT_PREFIX_BYTES = None  # 由备份长度确定

R = []
def add(s): R.append(s)
def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
def rj(p): return json.loads(open(p, "rb").read().decode("utf-8-sig"))

PASS = 0; FAIL = 0
def chk(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; add("  [PASS] %s %s" % (name, detail))
    else:    FAIL += 1; add("  [FAIL] %s %s" % (name, detail))

MEMO  = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
BACK  = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r8\AGI_DEEPSEEK_MEMO.md")
QUEUE = os.path.join(ROOT, r"research\deepseek\atlas\phase_queue_v1.json")
QBACK = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r8\phase_queue_v1.json")
META  = os.path.join(ROOT, r"tests\deepseek\result\meta_single_source_v4.json")
MBACK = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r8\meta_single_source_v4.json")
CLOSE = os.path.join(ROOT, r"research\deepseek\atlas\a_gate_closure_v1.json")
PATCH = os.path.join(ROOT, r"research\deepseek\atlas\ledger_corrections_v1.json")
LEDGER= os.path.join(ROOT, r"research\gpt5\atlas\atlas_ledger.json")

add("=== R8 独立复核 @ 新进程 ===")
add("")

# ---- 1) memo ----
add("== 1) memo append-only ==")
raw = open(MEMO, "rb").read()
bak = open(BACK, "rb").read()
chk("前缀逐字节不变", raw[:len(bak)] == bak, "前 %d B" % len(bak))
chk("前缀 sha8 == b257f54f", hashlib.sha256(raw[:len(bak)]).hexdigest()[:8] == "b257f54f")
t = raw.decode("utf-8-sig")
chk("BOM", raw.startswith(b"\xef\xbb\xbf"))
chk("bare_lf == 0", t.count("\n") - t.count("\r\n") == 0, "bare=%d" % (t.count("\n") - t.count("\r\n")))
chk("Phase 35 在位", "## Phase 35:" in t)
chk("Phase 标题数 == 35", sum(1 for x in t.replace("\r\n", "\n").split("\n") if x.startswith("## Phase ")) == 35)
chk("seal 原文在位", "seal: Q08=甲 | C=全接受 | I1=确认 | I9=确认" in t)
chk("K1 判决词在位", "fired_all_models" in t and "model_specific" in t)
chk("C 表数字在位", "0dc6e57a" in t and "9a3c6ff4" in t and "41.5%" in t)
add("  memo = %d B  %s" % (len(raw), sha8(MEMO)))

# ---- 2) 队列 ----
add("")
add("== 2) phase_queue_v1.json ==")
q = rj(QUEUE); qb = rj(QBACK)
chk("原 30 项", q["count"] == 30 and len(q["queue"]) == 30)
sealed = sorted(it["id"] for it in q["queue"] if it["status"] == "sealed")
chk("sealed == Q01/Q02/Q08/Q09/Q12", sealed == ["Q01", "Q02", "Q08", "Q09", "Q12"], str(sealed))
chk("其余仍 pending", sum(1 for it in q["queue"] if it["status"] == "pending") == 25)
chk("条目数不变（仅状态变）", len(qb["queue"]) == len(q["queue"]))
add("  queue = %d B  %s（前 %s）" % (os.path.getsize(QUEUE), sha8(QUEUE), sha8(QBACK)))

# ---- 3) meta 更正回填 ----
add("")
add("== 3) meta_single_source_v4.json ==")
m = rj(META)
chk("6 条更正全 accepted", len(m["corrections"]) == 6 and all(c["status"].startswith("accepted") for c in m["corrections"]))
chk("seal_verbatim 记录", m.get("seal_verbatim", "").startswith("seal: Q08=甲"))
c4 = [c for c in m["corrections"] if c["id"] == "C4"][0]
chk("C4 更正值为 0dc6e57a", c4["resolved_value"]["correct_self_excluding"] == "0dc6e57a")
add("  meta = %d B  %s（前 %s）" % (os.path.getsize(META), sha8(META), sha8(MBACK)))

# ---- 4) 关闭记录：独立重算 ----
add("")
add("== 4) a_gate_closure_v1.json（独立重算）==")
cl = rj(CLOSE)
dl = rj(os.path.join(ROOT, r"tests\deepseek\result\deadline_dual_track_v1.json"))["k1_recompute"]
agg = dl["aggregate"]; ml = dl["model_level"]
E_pool = sum(agg["E_read_per_model"]) / len(agg["E_read_per_model"])
chk("E_read 池化独立重算 == JSON", abs(E_pool - agg["E_read_pooled"]) < 1e-12, "%.8f vs %.8f" % (E_pool, agg["E_read_pooled"]))
chk("E_read 池化 > 5% 门", E_pool > 0.05, "%.4f (%.1f×)" % (E_pool, E_pool / 0.05))
chk("读出层 3/3 否决", all(ml[m]["readout_pass"] is False for m in ml))
chk("读出层 margin 未达负向门", agg["readout_pooled_margin"] > -2 * agg["readout_pooled_mde"])
chk("k* 层 1 模型否决 ⇒ model_specific", sum(1 for m in ml if not ml[m]["kstar_pass"]) == 1)
chk("closure.verdict 与 deadline JSON 一致", cl["k1_reverdict"]["verdict"] == dl["verdict"])
chk("gate_closed", cl["gate_closed"] is True)
# C4 独立重算（不读 closure）
lg = rj(LEDGER)
lg_wo = {k: v for k, v in lg.items() if k != "ledger_sha256_8"}
re_c4 = hashlib.sha256(json.dumps(lg_wo, ensure_ascii=False, indent=1).encode("utf-8")).hexdigest()[:8]
chk("C4 content_excluding_self 重算 == 0dc6e57a", re_c4 == "0dc6e57a", re_c4)
chk("账本当前 recorded 仍为陈旧 41d65a13（未施加）", lg.get("ledger_sha256_8") == "41d65a13")
add("  closure = %d B  %s" % (os.path.getsize(CLOSE), sha8(CLOSE)))

# ---- 5) 跨线补丁 spec ----
add("")
add("== 5) ledger_corrections_v1.json ==")
pt = rj(PATCH)
chk("状态 SEALED_BUT_NOT_APPLIED", pt["status"] == "SEALED_BUT_NOT_APPLIED")
chk("4 条 op", len(pt["operations"]) == 4)
chk("记录正确值 0dc6e57a", pt["recorded_correct_value"] == "0dc6e57a")
add("  patch = %d B  %s" % (os.path.getsize(PATCH), sha8(PATCH)))

# ---- 6) 跨线受保护文件未被触碰 ----
add("")
add("== 6) 跨线受保护文件指纹 ==")
for rel, exp in [(r"research\gpt5\docs\AGI_GPT5_MEMO.md", "2a84776b"),
                 (r"research\gpt5\atlas\atlas_ledger.json", "bbda63df"),
                 (r"tests\glm5\result\rdc_query_construction_20260913\phase3103\omega_p101_formula_audit\proposition_ledger.json", "9a3c6ff4"),
                 (r"research\deepseek\atlas\metric_dict.json", "03887e51"),
                 (r"tests\deepseek\result\deadline_dual_track_v1.json", "4d1853d3"),
                 (r"tests\deepseek\result\prop_citation_audit_v1.json", "4c2ea9d2"),
                 (r"tests\deepseek\result\seal_request_v1.json", "d871a4b4")]:
    p = os.path.join(ROOT, rel); g = sha8(p)
    chk("未变 %s" % os.path.basename(rel), g == exp, "exp=%s got=%s" % (exp, g))

# ---- 7) 备份可回滚 ----
add("")
add("== 7) 备份 ==")
for b, exp in [(BACK, "b257f54f"), (QBACK, "675836fd"), (MBACK, "f9d7ede6")]:
    chk("备份 %s" % os.path.basename(b), os.path.exists(b) and sha8(b) == exp, sha8(b) if os.path.exists(b) else "MISSING")

add("")
add("==== 汇总：PASS=%d  FAIL=%d  %s ====" % (PASS, FAIL, "ALL_PASS" if FAIL == 0 else "HAS_FAIL"))
open(OUT, "w", encoding="utf-8").write("\n".join(R))
print("\n".join(R))
