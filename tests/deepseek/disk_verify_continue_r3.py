# -*- coding: utf-8 -*-
"""R3 续研：对本轮两份新交付件做独立磁盘复核。
原则：所有判据从源文件（MEMO / ledger）重算，不信任生成器产出的中间 json。"""
import os, re, json, hashlib

root = r"D:\AI2050\Ai2050-OpenOne"
T = os.path.join(root, "tests/deepseek/result")
R = []


def chk(name, cond, detail=""):
    R.append(("PASS" if cond else "FAIL", name, detail))
    return cond


# ---------- 1. 源文件级重算 ----------
memo_b = open(os.path.join(root, "research/gpt5/docs/AGI_GPT5_MEMO.md"), "rb").read()
memo = memo_b.decode("utf-8-sig")
lines = memo.split("\n")

# 1a. Phase 节重算（独立正则）
idx = [i for i, l in enumerate(lines) if re.match(r"^##\s+.*Phase", l)]
chk("源:Phase 节重算 >=390", len(idx) >= 390, "n=%d" % len(idx))

# 1b. 严格共享指标 delta 重算（独立写法）
# 宽词表（规格口径，含 rho/相关系数）
MW = r"(?:误差|error|rel-?L2|AUC|cos|余弦|相关系数|rho)"
DW = r"(?:→|->|降到|降至|下降|降低|减少|改善|恶化|上升|增至|提升|拉近)"
pat = re.compile(MW + r"[^。\n]{0,45}?\d+\.\d+[^。\n]{0,14}?" + DW + r"[^。\n]{0,14}?\d+\.\d+")
# 窄词表（敏感性对照）
patn = re.compile(r"(?:误差|error|rel-?L2|AUC|cos|余弦)[^。\n]{0,45}?\d+\.\d+[^。\n]{0,14}?"
                  + DW + r"[^。\n]{0,14}?\d+\.\d+")
n_adv = 0
n_nar = 0
for j, i in enumerate(idx):
    end = idx[j + 1] if j + 1 < len(idx) else len(lines)
    body = "\n".join(lines[i:end])
    if pat.search(body):
        n_adv += 1
    if patn.search(body):
        n_nar += 1
chk("源:宽词表 advance 精确复现 26", n_adv == 26, "n=%d" % n_adv)
chk("源:窄词表 advance 落在 [10,30]（敏感性）", 10 <= n_nar <= 30, "n_narrow=%d" % n_nar)

# 1c. 关键词重算
chk("源:判决计数 >600", memo.count("判决") > 600, "n=%d" % memo.count("判决"))
chk("源:接续计数 >300", memo.count("接续") > 300, "n=%d" % memo.count("接续"))

# 1d. 三个 KPI 名在 MEMO 中不存在
for k in ["E_ar", "E_read", "C_steer"]:
    chk("源:MEMO 无 %s" % k, memo.count(k) == 0, "n=%d" % memo.count(k))

# 1e. ledger 自声明 vs 实际
lb = open(os.path.join(root, "research/gpt5/atlas/atlas_ledger.json"), "rb").read()
led = json.loads(lb.decode("utf-8-sig"))
chk("源:ledger 自声明 != 实际（缺口成立）",
    led.get("ledger_sha256_8") != hashlib.sha256(lb).hexdigest()[:8],
    "decl=%s act=%s" % (led.get("ledger_sha256_8"), hashlib.sha256(lb).hexdigest()[:8]))

# ---------- 2. 宪法文件 ----------
cp = os.path.join(root, "research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md")
cb = open(cp, "rb").read()
ct = cb.decode("utf-8")
chk("宪法:存在且非空", len(cb) > 8000, "%d B" % len(cb))
bad = re.findall(r"\{[a-z_\[\]'\"]+\}|None|nan", ct)
chk("宪法:0 未填充占位", len(bad) == 0, str(bad[:5]))
for k in ["6.55", "93.45", "371", "0.3316", "0.3986", "0.3898", "6.6",
          "41d65a13", "bbda63df", "1.104", "E_ar", "C_steer", "E_read",
          "Q30", "phase_queue_v1.json", "013d08a9", "可识别性", "未识别", "窄词表"]:
    chk("宪法:含 %s" % k, k in ct)
chk("宪法:声明不改 MEMO", "不改动任何 MEMO" in ct)

# ---------- 3. 队列文件 ----------
qp = os.path.join(root, "research/gpt5/atlas/phase_queue_v1.json")
qb = open(qp, "rb").read()
q = json.loads(qb.decode("utf-8-sig"))
chk("队列:count==30", q.get("count") == 30)
chk("队列:queue 长度==30", len(q.get("queue", [])) == 30)
ids = [e["id"] for e in q["queue"]]
chk("队列:id 唯一且为 Q01..Q30",
    ids == ["Q{:02d}".format(i) for i in range(1, 31)], str(ids[:3]))
allowed_kpi = {"none", "all", "E_read", "E_ar", "C_steer"}
chk("队列:kpi 取值合法", all(e["kpi"] in allowed_kpi for e in q["queue"]))
chk("队列:每项有 deliverable", all(e.get("deliverable") for e in q["queue"]))
chk("队列:禁止自动派生的规则在位", "禁止从任意 Phase 的未解释残差派生" in q.get("rule", ""))
blocks = sorted(set(e["block"] for e in q["queue"]))
chk("队列:区块覆盖 A-F", any(b.startswith("A") for b in blocks) and any(b.startswith("F") for b in blocks), str(blocks))
chk("队列:引用 KPI 定义的宪法锚", "RDC_RESEARCH_CONSTITUTION_v1" in q.get("kpi_definition_ref", ""))
chk("队列:source_stats 文件存在",
    all(os.path.exists(os.path.join(T, v)) for v in q["source_stats"].values()))

# ---------- 4. 与 loop_stats 交叉核对 ----------
S = json.loads(open(os.path.join(T, "loop_stats_r3.json"), "rb").read().decode("utf-8-sig"))
for r in S["g1"]["rows"]:
    chk("交叉:E_read %s 落在宪法" % r["model"],
        ("%.4f" % r["b4_readout"]) in ct)
chk("交叉:anova 60.4 落在宪法", "60.4" in ct and "56.8" in ct)

# ---------- 汇总 ----------
nfail = sum(1 for s, _, _ in R if s == "FAIL")
out = ["=== R3 续研 独立复核 ===",
       "constitution sha8 = %s  (%d B)" % (hashlib.sha256(cb).hexdigest()[:8], len(cb)),
       "queue sha8        = %s  (%d B)" % (hashlib.sha256(qb).hexdigest()[:8], len(qb)),
       "PASS=%d FAIL=%d" % (len(R) - nfail, nfail), ""]
for s, n, d in R:
    out.append("[%s] %s %s" % (s, n, ("| " + d) if d else ""))
out.append("")
out.append("ALL_PASS" if nfail == 0 else "HAS_FAILURE")
open(os.path.join(T, "disk_verify_continue_r3.txt"), "w", encoding="utf-8").write("\n".join(out))
print("PASS=%d FAIL=%d" % (len(R) - nfail, nfail))
