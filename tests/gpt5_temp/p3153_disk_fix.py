# -*- coding: utf-8 -*-
"""3153 disk-sha 记录：MEMO 锚行补 disk 值 + ledger rev_note 补 disk + 复核判据修正。"""
import io, os, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
LEDGER = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
B = os.path.join(ROOT, "tests", "glm5", "result",
                 "rdc_query_construction_20260913", "phase3153",
                 "g1p3_failure_mode_anatomy")
out = []

# 磁盘 sha8（回填后真实磁盘值）
disk = {}
for m, f in [("qwen3-4b", "result.json"), ("qwen3-14b", "result.json"),
             ("glm4", "result.json"), ("summary", "result_summary.json")]:
    p = os.path.join(B, m, f)
    disk[m] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
out.append("disk sha8: %s" % json.dumps(disk))

# --- MEMO 锚行补 disk（本 Phase 自身节的收尾修正） ---
mm = io.open(MEMO, encoding="utf-8").read()
old_anchor = ("4b res **c222c7ff** seal dad158d2；14b res **a0049e59** seal a600c4b5；"
              "glm4 res **18f2ef01** seal d36c7627；summary res **b9846738** seal 8236f974"
              "（execution 95541415）。")
new_anchor = ("4b res **c222c7ff** seal dad158d2（disk %s）；14b res **a0049e59** seal a600c4b5"
              "（disk %s）；glm4 res **18f2ef01** seal d36c7627（disk %s）；summary res **b9846738** "
              "seal 8236f974（disk %s，execution 95541415）。"
              % (disk["qwen3-4b"], disk["qwen3-14b"], disk["glm4"], disk["summary"]))
if old_anchor in mm:
    mm = mm.replace(old_anchor, new_anchor)
    io.open(MEMO, "w", encoding="utf-8", newline="").write(mm)
    out.append("memo: anchor disk values inserted")
elif "disk %s" % disk["qwen3-4b"] in mm:
    out.append("memo: disk values already present")
else:
    raise AssertionError("anchor line not found")

# --- ledger 3153 条目 rev_note 补 disk ---
led = json.load(io.open(LEDGER, encoding="utf-8"))
for m in led["measurements"]:
    if m.get("phase") == 3153 and "disk" not in m["rev_note"]:
        m["rev_note"] += (" | disk sha8 (post-backfill): 4b %s, 14b %s, glm4 %s, summary %s"
                          % (disk["qwen3-4b"], disk["qwen3-14b"], disk["glm4"], disk["summary"]))
        blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode("utf-8")
        led["ledger_sha256_8"] = hashlib.sha256(blob).hexdigest()[:8]
        json.dump(led, io.open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        out.append("ledger: rev_note disk appended, new sha8=%s" % led["ledger_sha256_8"])
        break
else:
    out.append("ledger: disk already present")

# --- 复核判据修正版 ---
ok = True
def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    out.append("[%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))

for m, f, sha, vpre in [("qwen3-4b", "result.json", "c222c7ff", "g1p3_qwen3-4b"),
                        ("qwen3-14b", "result.json", "a0049e59", "g1p3_qwen3-14b"),
                        ("glm4", "result.json", "18f2ef01", "g1p3_glm4"),
                        ("summary", "result_summary.json", "b9846738", "g1p3_summary")]:
    p = os.path.join(B, m, f)
    r = json.load(io.open(p, encoding="utf-8"))
    chk("res_sha8 %s" % m, r.get("res_sha8") == sha, r.get("res_sha8"))
    chk("verdict-tail %s" % m, r["verdict"].endswith("sha8_" + sha))
    chk("seal-embedded %s" % m, bool(r.get("seal_sha8")), r.get("seal_sha8"))
    chk("disk-recorded %s" % m, disk[m] == hashlib.sha256(open(p, "rb").read()).hexdigest()[:8],
        disk[m])

mm2 = io.open(MEMO, encoding="utf-8").read()
chk("memo disk values", all(("disk %s" % disk[k]) in mm2 for k in disk))
led2 = json.load(io.open(LEDGER, encoding="utf-8"))
chk("ledger n=290", len(led2["measurements"]) == 290)
chk("ledger disk note", "disk sha8 (post-backfill)" in
    [m for m in led2["measurements"] if m.get("phase") == 3153][0]["rev_note"])

out.append("VERIFY " + ("OK - all checks passed" if ok else "FAILED"))
io.open(os.path.join(ROOT, "tests", "gpt5_temp", "p3153_verify2_out.txt"),
        "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
