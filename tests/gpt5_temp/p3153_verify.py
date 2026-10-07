# -*- coding: utf-8 -*-
"""3153 daily 补写 + 独立磁盘复核（不复读 closeout 内部状态）。"""
import io, os, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
DAILY = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-01.md")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
LEDGER = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
B = os.path.join(ROOT, "tests", "glm5", "result",
                 "rdc_query_construction_20260913", "phase3153",
                 "g1p3_failure_mode_anatomy")
out = []

# --- daily 补写（精确检查完成行） ---
d = io.open(DAILY, encoding="utf-8").read()
marker = "3153 G1-P3 失败模态解剖闭环"
if marker not in d:
    line = ("- Phase 3153 G1-P3 失败模态解剖闭环：全零 GPU 三模型（4b/14b/glm4 读 3151/3152 npz）。"
            "主判决 **fingerprint_consistent_coverage_partial**——读出层残差 ANOVA 交互(i,c)≈0.0–0.3% "
            "三模型一致（fp_corr 0.995–0.999、spec_corr 0.968–0.991，死线未触发），"
            "闭合 3152 读出层 rank10 之谜（残差无网格坐标）；M1 因子列空间≈Pc(k*) 主角 "
            "0.979/0.889/0.977 vs Pc(KOUT) 0.14/0.34/0.38 = k* 交互=类子空间第 4 确证；"
            "worst20 模态分布模型特异（glm4 错配 13/20、4b 散布 8/20、14b 均衡）；cov partial（4b 0.60）。"
            "修 1 次（patch2：M1 门段 Mte/e_m1 误入 t 循环→移出，与 3152 位级一致后全锚过）。"
            "锚：4b c222c7ff/dad158d2、14b a0049e59/a600c4b5、glm4 18f2ef01/d36c7627、summary b9846738/8236f974；"
            "ledger n=290。3154（G2-P1 多关系族 K2 可分离性+held-out 门）已预注册。\n")
    io.open(DAILY, "a", encoding="utf-8").write(line)
    out.append("daily: appended 3153 completion")
else:
    out.append("daily: completion already present")

# --- 独立磁盘复核 ---
ok = True
def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    out.append("[%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))

# 1. 四 result.json 存在 + verdict 锚
exp = {"qwen3-4b": ("c222c7ff", "g1p3_qwen3-4b"),
       "qwen3-14b": ("a0049e59", "g1p3_qwen3-14b"),
       "glm4": ("18f2ef01", "g1p3_glm4"),
       "summary": ("b9846738", "g1p3_summary")}
for m, (sha, vpre) in exp.items():
    p = os.path.join(B, m, "result.json" if m != "summary" else "result_summary.json")
    if not os.path.exists(p):
        chk("result %s" % m, False, "missing")
        continue
    r = json.load(io.open(p, encoding="utf-8"))
    chk("result %s" % m, r.get("res_sha8") == sha and
        r["verdict"].startswith(vpre), "sha=%s" % r.get("res_sha8"))
    seal = hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
    chk("seal %s" % m, seal == r.get("seal_sha8"), seal)

# 2. execution.json ×4
for m in exp:
    p = os.path.join(B, m, "execution.json")
    chk("execution %s" % m, os.path.exists(p))

# 3. MEMO 3153 节 + 3154 预注册
mm = io.open(MEMO, encoding="utf-8").read()
chk("memo 3153 section", "## Phase 3153: 失败模态解剖" in mm)
chk("memo 3154 prereg", "预注册 Phase 3154" in mm)
chk("memo verdict string", "g1p3_fingerprint_consistent_coverage_partial" in mm)

# 4. ledger n=290 + 3153 条目
led = json.load(io.open(LEDGER, encoding="utf-8"))
chk("ledger n=290", len(led["measurements"]) == 290, str(len(led["measurements"])))
m3153 = [m for m in led["measurements"] if m.get("phase") == 3153]
chk("ledger 3153 entry", len(m3153) == 1 and
    m3153[0]["verdict"] == "g1p3_fingerprint_consistent_coverage_partial")

# 5. daily 完成行
d = io.open(DAILY, encoding="utf-8").read()
chk("daily 3153 completion", marker in d)

# 6. MEMORY 重写含 3153
mem = io.open(MEMORY, encoding="utf-8").read()
chk("memory 3153 row", "fingerprint_consistent_coverage_partial" in mem)
chk("memory size", len(mem.encode("utf-8")) < 5200,
    "%d bytes" % len(mem.encode("utf-8")))

out.append("VERIFY " + ("OK — all checks passed" if ok else "FAILED"))
io.open(os.path.join(ROOT, "tests", "gpt5_temp", "p3153_verify_out.txt"),
        "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
