# -*- coding: utf-8 -*-
"""3153 结果提取（读 summary + 三模型 result 关键字段）。"""
import json, io, os

B = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3153\g1p3_failure_mode_anatomy"
out = []
s = json.load(io.open(os.path.join(B, "summary", "result_summary.json"), encoding="utf-8"))
out.append("verdict: %s" % s["verdict"])
out.append("runtime: %s" % s["runtime_s"])
out.append("fp_corr: %s" % json.dumps(s["fp_corr"]))
out.append("spec_corr: %s" % json.dumps(s["spec_corr"]))
out.append("fingerprint_consistent: %s" % s["fingerprint_consistent"])
out.append("dead_line_triggered: %s" % s["dead_line_triggered"])
out.append("coverage_all_pass: %s  cov_mean: %s" % (s["coverage_all_pass"], s["coverage_mean"]))
out.append("jaccard_sample: %s" % json.dumps(s["worst20_jaccard_sample"]))
out.append("jaccard_pair: %s" % json.dumps(s["worst20_jaccard_pair"]))
out.append("modes_table: %s" % json.dumps(s["modes_table"], ensure_ascii=False))
out.append("align_note: %s" % json.dumps(s["align_note"], ensure_ascii=False))
for m in ["qwen3-4b", "qwen3-14b", "glm4"]:
    r = json.load(io.open(os.path.join(B, m, "result.json"), encoding="utf-8"))
    out.append("== %s ==" % m)
    out.append("  verdict: %s" % r["verdict"])
    out.append("  anchors: %s" % json.dumps(r["anchor_checks"], ensure_ascii=False))
    out.append("  hard: %s s2=%s" % (r["hard_classes"], {k: round(v, 3) for k, v in list(r["s2_b4_readout"].items())[:3]}))
    out.append("  fp_kout: %s fp_kstar: %s" % (r["anatomy_kout"]["fp"], r["anatomy_kstar"]["fp"]))
    out.append("  spec_cell_kout[:6]: %s" % [round(x, 3) for x in r["anatomy_kout"]["spec_cell"][:6]])
    out.append("  spec_cell_kstar[:6]: %s" % [round(x, 3) for x in r["anatomy_kstar"]["spec_cell"][:6]])
    out.append("  align: %s" % json.dumps({k: round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v] for k, v in r["m1_align"].items() if k in ("margin_s7", "e_m1_mean", "e_b4_k", "principal_vs_pckstar", "principal_vs_pckout")}))
    out.append("  Eg of M3 samples: %s" % [round(w["Eg"], 3) for w in r["worst20_modes"] if w["mode"] == "M3_scatter"])
    out.append("  proj_pcko of M3 samples: %s" % [round(w["proj_pcko"], 3) for w in r["worst20_modes"] if w["mode"] == "M3_scatter"])
io.open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3153_extract.txt", "w", encoding="utf-8").write("\n".join(out))
print("written")
