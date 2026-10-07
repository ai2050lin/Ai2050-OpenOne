# -*- coding: utf-8 -*-
"""Q02 独立磁盘复核：从源文件重算，与 metric_dict v2 比对。"""
import os, re, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "deepseek", "result")
PASS, FAIL = [], []
def chk(n, c, d=""):
    (PASS if c else FAIL).append((n, d))
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
def rd(p):
    with open(os.path.join(ROOT, p), "rb") as f: return f.read()
def sha8_file(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()[:8]
def SER(o): return json.dumps(o, ensure_ascii=False, indent=1).encode("utf-8")

MP   = "research/gpt5/atlas/metric_dict.json"
BP   = "research/gpt5/atlas/metric_dict_v1_backup.json"
MDD  = "research/gpt5/docs/METRIC_DICT_Q02.md"
BASE = "tests/glm5/result/rdc_query_construction_20260913/phase3152/g1p2_tri_model_k1"
NPZG = "tests/glm5/result/rdc_query_construction_20260913/phase3151/g1p1_combo_additive_vs_interaction/collect.npz"

raw = rd(MP); V2 = json.loads(raw.decode("utf-8"))
md  = rd(MDD).decode("utf-8")

# 1 版本与备份
chk("v2 format", V2.get("format") == "metric_dict_v2", V2.get("format"))
chk("v2 version==2", V2.get("version") == 2)
chk("v1 备份存在且 sha8=469c0ad1", sha8(rd(BP)) == "469c0ad1", sha8(rd(BP)))
chk("v1 备份内容 == supersedes.sha8", V2["supersedes"]["sha8"] == sha8(rd(BP)))

# 2 兼容性（模拟 P3150 历史断言）
chk("兼容: len(metrics)==7", len(V2["metrics"]) == 7, len(V2["metrics"]))
chk("兼容: meta_rules 含 F2/F7",
    "F2_verdict_grading" in V2["meta_rules"] and "F7_bit_anchor_status" in V2["meta_rules"])

# 3 E_read 现场重算
ARMS = [("qwen3-4b", "qwen3-4b"), ("qwen3-14b", "qwen3-14b"), ("glm4-9b", "glm4k1")]
recomputed = {}
for mname, arm in ARMS:
    d = json.loads(rd("%s/%s/result.json" % (BASE, arm)).decode("utf-8-sig"))
    k = d["k1_model_report"]
    recomputed[mname] = k["b4_rel_readout_mean3seed"]
    ps = k["b4_rel_readout_per_seed"]
    chk("E_read[%s] 与 v2 逐位相等" % mname,
        V2["global_kpis"]["E_read"]["current"][mname] == k["b4_rel_readout_mean3seed"],
        "%r vs %r" % (V2["global_kpis"]["E_read"]["current"][mname], k["b4_rel_readout_mean3seed"]))
    chk("E_read[%s] per_seed 均值 == mean3seed" % mname,
        len(ps) == 3 and abs(sum(ps) / 3.0 - k["b4_rel_readout_mean3seed"]) < 1e-12)

# 4 gate 判定
chk("E_read 3/3 均 > 0.05", all(v > 0.05 for v in recomputed.values()), str(recomputed))
_npass = sum(1 for v in recomputed.values() if v <= 0.05)
_exp_vd = "all_above_gate (%d/3 过门)" % _npass
chk("current_verdict 与重算一致", V2["global_kpis"]["E_read"]["current_verdict"] == _exp_vd,
    "%s vs %s" % (V2["global_kpis"]["E_read"]["current_verdict"], _exp_vd))

# 5 载体指纹（分块哈希，避免大内存）
carrier = V2["global_kpis"]["E_read"]["data"]["heldout_carrier"]
for mk, relp in [("qwen3-4b", BASE + "/qwen3-4b/collect.npz"),
                 ("qwen3-14b", BASE + "/qwen3-14b/collect.npz"),
                 ("glm4-9b", NPZG)]:
    chk("carrier[%s].sha8 == 实测" % mk, carrier[mk]["sha8"] == sha8_file(relp),
        "%s vs %s" % (carrier[mk]["sha8"], sha8_file(relp)))
    chk("carrier[%s].bytes == 实测" % mk, carrier[mk]["bytes"] == os.path.getsize(os.path.join(ROOT, relp)))
g = json.loads(rd(BASE + "/glm4k1/result.json").decode("utf-8-sig"))
chk("glm4 source_npz_sha8 == 实测 3151 npz", g.get("source_npz_sha8") == sha8_file(NPZG),
    "%s vs %s" % (g.get("source_npz_sha8"), sha8_file(NPZG)))

# 6 source 记录的文件 sha8 与盘上一致
for mname, arm in ARMS:
    p = "%s/%s/result.json" % (BASE, arm)
    chk("source[%s].result_file_sha8 与盘上一致" % mname,
        V2["global_kpis"]["E_read"]["source"]["per_model"][mname]["result_file_sha8"] == sha8(rd(p)))

# 7 content_excluding_self 自洽
c = V2.pop("content_sha256_8", None)
chk("content_sha256_8 自洽（排除自身重算）", c == sha8(SER(V2)), "%s vs %s" % (c, sha8(SER(V2))))
V2["content_sha256_8"] = c

# 8 三 KPI 结构
gk = V2["global_kpis"]
chk("三 KPI 齐备", set(gk.keys()) == {"E_read", "E_ar", "C_steer"}, str(list(gk.keys())))
chk("E_ar.status == not_built", gk["E_ar"]["status"] == "not_built")
chk("C_steer.status == not_measured", gk["C_steer"]["status"] == "not_measured")
chk("E_ar 含 to_build_in", "to_build_in" in gk["E_ar"])
chk("C_steer 含 to_measure_in", "to_measure_in" in gk["C_steer"])
chk("kpi_registration_rule 含 catalog/advance",
    "catalog" in V2["kpi_registration_rule"]["rule"] and "advance" in V2["kpi_registration_rule"]["rule"])

# 9 报告文本
for s in ["0.33162", "0.39860", "0.38984", "03887e51", "469c0ad1", "5ce974ee", "c38b97ff"]:
    chk("报告 md 含 %s" % s, s in md)
for bad in ["None", "nan"]:
    chk("报告 md 无 %s" % bad, bad not in md)

# 10 上游未改动
chk("Ledger 未变（bbda63df / 304）",
    sha8(rd("research/gpt5/atlas/atlas_ledger.json")) == "bbda63df"
    and len(json.loads(rd("research/gpt5/atlas/atlas_ledger.json").decode("utf-8-sig"))["measurements"]) == 304)
chk("宪法未变（01df6398）", sha8(rd("research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md")) == "01df6398")
chk("Q01 交付物仍在（meta_single_source_v4.json）",
    os.path.exists(os.path.join(ROOT, "research/gpt5/atlas/meta_single_source_v4.json")))

# 汇总
L = ["Q02 独立磁盘复核（源文件重算）", "=" * 60]
for n, d in PASS: L.append("[PASS] %s%s" % (n, ("  <%s>" % d) if d else ""))
for n, d in FAIL: L.append("[FAIL] %s%s" % (n, ("  <%s>" % d) if d else ""))
L += ["=" * 60, "PASS=%d  FAIL=%d  ->  %s" % (len(PASS), len(FAIL), "ALL_PASS" if not FAIL else "HAS_FAIL")]
open(os.path.join(OUT, "disk_verify_q02_r4.txt"), "wb").write(("\n".join(L) + "\n").encode("utf-8"))
print("PASS=%d FAIL=%d -> %s" % (len(PASS), len(FAIL), "ALL_PASS" if not FAIL else "HAS_FAIL"))
for n, d in FAIL: print("  FAIL:", n, d)
