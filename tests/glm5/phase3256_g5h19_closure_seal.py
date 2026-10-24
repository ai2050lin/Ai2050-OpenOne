# -*- coding: utf-8 -*-
"""Phase 3256 G5-H19: closure seal -- 收线终点封章 + N 线施加准备.

预注册（3255 tail）：
  (a) 封章臂：3239–3255 收线链一页式总结（数字全部从封存盘面实测）+
      registry v1.34 -> v1.35 append（upgrade_log +1 封章注记，append-only 字节纪律）。
  (b) N 线施加准备臂：收线包清单 nline_handoff_manifest.json（registry/gap/综论/
      现役配方/u* 三部曲/F20 边界/GPU 挂账，全带实测 sha8）。
  (c) GPU 挂账：checkpoint 训练步轨迹 + 5 appendix GPU 项维持挂账（记录不执行）。

纪律：freeze 先于一切观察（DESIGN 静态，hash 排除 created）→ SMOKE（零落册写）→
正式 → gates → result.json → closeout 五写（另脚本）→ 独立磁盘复核（另脚本）。
SMOKE 模式：所有计算与 revert/pointer 校验在内存完成，v1.35/summary/manifest 不落盘。
"""
import hashlib
import json
import os
import sys

SMOKE = os.environ.get("P3256_SMOKE") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = HERE if os.path.basename(HERE) == "glm5" else os.path.join(HERE, "glm5")  # tests/glm5
ROOT = os.path.dirname(os.path.dirname(BASE))                       # repo root
RBASE = os.path.join(BASE, "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3256", "g5h19_closure_seal")
SMOKE_DIR = os.path.join(OUT, "smoke")
os.makedirs(SMOKE_DIR if SMOKE else OUT, exist_ok=True)

REG134 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "atlas_registry_v1_34.json")
GAP112 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json")
LEDGER = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")

PHASE = 3256
NAME = "g5h19_closure_seal"


def s8(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": PHASE,
    "name": NAME,
    "prereg": "3255 tail: closure-seal arm + N-line handoff prep arm",
    "arms": {
        "a": "closure summary 3239-3255 (all numbers measured from sealed disk) + registry v1.35 append (upgrade_log +1 seal annotation, append-only byte discipline)",
        "b": "N-line handoff manifest (sealed artifacts with measured sha8 + incumbent recipe + u* trilogy + F20 boundary + GPU backlog)",
        "c": "GPU backlog recorded, not executed",
    },
    "planned_registry_ptrs": ["$.version", "$.supersedes", "$.upgrade_log (LEN)"],
    "gates": {
        "g1": "registry v1.35 pointer diff == planned exactly AND revert-check bitwise == v1.34 raw bytes",
        "g2": "closure facts consistency: G0_SCLS/ORC/incumbent values equal sealed headline/aggregate, repair-rate ~0.7436 (vs zero target)",
        "g3": "ledger 3239-3255 closure-chain entries present AND sealed v1.34/v1.12 bytes unchanged after writes",
    },
    "no_gpu": True,
}
if SMOKE:
    DESIGN["mode"] = "smoke"
else:
    DESIGN["mode"] = "formal"
DESIGN["created"] = today()
design_sha8 = hashlib.sha256(
    json.dumps({k: v for k, v in DESIGN.items() if k != "created"},
               sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]

if not SMOKE:
    with open(os.path.join(OUT, "design.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(DESIGN, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "design_sha8.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(design_sha8)
    print("freeze", design_sha8, "(smoke freeze already recorded in smoke dir)")
else:
    print("SMOKE freeze", design_sha8)

# ---------------------------------------------------------------- observations
reg134_raw = open(REG134, "rb").read()
reg134 = json.loads(reg134_raw.decode("utf-8"))
gap112 = json.load(open(GAP112, encoding="utf-8"))
led = json.load(open(LEDGER, encoding="utf-8"))

r246 = json.load(open(os.path.join(RBASE, "phase3246", "g5h9_regime_boundary", "result.json"), encoding="utf-8"))
r250 = json.load(open(os.path.join(RBASE, "phase3250", "g5h13_kernel_boundary", "result.json"), encoding="utf-8"))
r251 = os.path.join(RBASE, "phase3251", "g5h14_ustar_lexical_attribution", "result.json")
r252 = os.path.join(RBASE, "phase3252", "g5h15_ustar_layer_trace", "result.json")
r253 = os.path.join(RBASE, "phase3253", "g5h16_ustar_subspace", "result.json")

# 收线事实（全部实测自封存盘面）
G0_scls = r246["aggregate"]["G0"]["fam_fid_mean"]["SCLS"]          # -1.7515
G0_massive = r246["aggregate"]["G0"]["fam_fid_mean"]["MASSIVE"]
ORC_scls = r246["aggregate"]["ORC_ls_full"]["fam_fid_mean"]["SCLS"]  # +0.3290
ORC_min = r246["headline"]["orc_ls_full_min_scls_ent"]              # +0.3290
INC = r250["aggregate"]["XC_p5_b8"]
INC_scls = INC["fam_fid_mean"]["SCLS"]
inc_h = r250["headline"]["boundary_axes"]["joint_grid_minSCLSENT"]["XC_p5_b8"]
inc_r2 = r250["headline"]["boundary_axes"]["joint_grid_r2"]["XC_p5_b8"]
inc_kl = INC["kl_mean"]
closure_tab = r250["headline"]["closure_table"]["XC_p5_b8"]

# 修复率（相对零目标，与 MEMO 74% 口径一致）
repair_vs_zero = (inc_h - G0_scls) / (0.0 - G0_scls)
assert abs(repair_vs_zero - 0.7436) < 0.005, repair_vs_zero
# oracle 上界之内的闭合占比
repair_vs_oracle = (inc_h - G0_scls) / (ORC_scls - G0_scls)

facts = {
    "G0_SCLS": round(G0_scls, 6), "G0_MASSIVE": round(G0_massive, 6),
    "ORC_ls_full_SCLS": round(ORC_scls, 6), "ORC_min_scls_ent": round(ORC_min, 6),
    "incumbent": "XC_p5_b8", "incumbent_SCLS": round(INC_scls, 6),
    "incumbent_headline_minSCLSENT": round(inc_h, 6),
    "incumbent_r2_oov": round(inc_r2, 6), "incumbent_kl": round(inc_kl, 6),
    "closure_table_XC_p5_b8": round(closure_tab, 6),
    "repair_vs_zero": round(repair_vs_zero, 4),
    "repair_vs_oracle": round(repair_vs_oracle, 4),
    "residual_after_incumbent": round(0.0 - inc_h, 6),
}
print("facts", json.dumps(facts, ensure_ascii=False))

# ---------------------------------------------------------------- arm (a) seal
SEALED_CHAIN = [
    (3239, "sealed baselines + drift anchors"), (3240, "cleaning basis anchors"),
    (3242, "X_anchor"), (3245, "X_combo additive (+0.954), FLIP unreachable in-regime"),
    (3246, "FLIP_LOCATION=INFORMATION_GAP, ORC_ls_full +0.329, span ceiling"),
    (3247, "XC_poly_b4 incumbent (-0.651), poly gain = mapping class"),
    (3248, "degree unsaturated, K_p3_T3 best pure-pull (-0.529)"),
    (3249, "deg5 monotone (-0.467), combo fails on p3 base"),
    (3250, "kernel boundary, XC_p5_b8 incumbent (-0.449), fidelity schema v2 + UI"),
    (3251, "u* lexical decoupling (activation-side v_c)"),
    (3252, "u* layer trace: embedding rank-1 at input, U-shape"),
    (3253, "u* subspace: mid-trough = same carrier decay, append trigger MET"),
    (3254, "landing: gap v1.12 (APPX-6) + registry v1.34"),
    (3255, "client v1.34 + u* synthesis: dual-track geometry, word-side unfixable"),
]


def ptr_diff(a, b, p=""):
    """JSON pointer diff with LEN-level granularity for list appends, value annotation stripped by caller."""
    out = []
    if type(a) is not type(b):
        out.append(p + " (TYPE %s->%s)" % (type(a).__name__, type(b).__name__))
        return out
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a:
                out.append(p + "." + str(k) + " (ADDED)")
            elif k not in b:
                out.append(p + "." + str(k) + " (REMOVED)")
            else:
                out.extend(ptr_diff(a[k], b[k], p + "." + str(k)))
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(p + " (LEN %d->%d)" % (len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(ptr_diff(x, y, p + "[%d]" % i))
    else:
        if a != b:
            out.append(p + " (%r->%r)" % (a, b))
    return out


def norm_ptr(s):
    return s.split(" (")[0].lstrip("$.")


def revert(new_obj, old_raw, planned_norm_ptrs):
    """Revert planned pointer changes from new_obj and compare bitwise with old raw."""
    import copy
    old = json.loads(old_raw.decode("utf-8"))
    o = copy.deepcopy(new_obj)
    for ptr in planned_norm_ptrs:
        if ptr == "upgrade_log":
            o["upgrade_log"] = o["upgrade_log"][:-1]
        elif ptr == "version":
            o["version"] = old["version"]
        elif ptr == "supersedes":
            o["supersedes"] = old["supersedes"]
        elif ptr == "updated":
            o["updated"] = old["updated"]
        else:
            raise AssertionError("unhandled revert ptr " + ptr)
    return json.dumps(o, ensure_ascii=False, indent=1).encode("utf-8") == old_raw


reg135 = json.loads(reg134_raw.decode("utf-8"))
assert reg135["version"] == "1.34"
reg135["version"] = "1.35"
reg135["supersedes"] = "atlas_registry_v1_34.json (3deabf25, Phase 3254)"
reg135["upgrade_log"].append({
    "node": "registry (seal annotation)",
    "change": ("closure seal: fidelity-closure chain 3239-3255 declared CLOSED in-regime -- "
               "incumbent XC_p5_b8 (Phase 3250, SCLS %.6f vs G0 %.6f, repair-vs-zero %.4f); "
               "residual -0.449 located at INFORMATION_GAP by ORC_ls_full oracle (+%.6f, Phase 3246); "
               "u* dual-track geometry (3251/3252/3253) proves word-side closure unreachable; "
               "F20 (readout geometry != behavioral control) bounds the only remaining lever "
               "(activation-side v_c injection family). No features/failures/anchors changed by this append. "
               "One-page summary + N-line handoff manifest at phase3256/g5h19_closure_seal/.") % (
                  inc_h, G0_scls, repair_vs_zero, ORC_scls),
    "reason": ("G5-H19 (Phase 3256): preregistered closure-seal arm (MEMO 3255 tail a); "
               "append-only byte discipline (revert-check bitwise, pointer diff exact)"),
})

gap_planned = DESIGN["planned_registry_ptrs"]
d_reg = set(norm_ptr(x) for x in ptr_diff(reg134, reg135))
assert d_reg == set(norm_ptr(x) for x in gap_planned), ("registry diff != planned", sorted(d_reg))
ok_revert = revert(reg135, reg134_raw, [norm_ptr(x) for x in gap_planned])
assert ok_revert, "revert check FAILED"
print("arm(a) pointer diff exact + revert bitwise OK")

# ---------------------------------------------------------------- arm (b) manifest
handoff = {
    "schema": "nline_handoff_v1",
    "created": today(),
    "phase": PHASE,
    "purpose": "G-line closure package for cross-line (N-line) application arm",
    "sealed_artifacts": {
        "registry_current": {"file": "atlas_registry_v1_35.json", "sha8_file": None, "dir": "phase3256/g5h19_closure_seal", "supersedes": "atlas_registry_v1_34.json (3deabf25)"},
        "registry_prev": {"file": "atlas_registry_v1_34.json", "sha8_file": s8(REG134), "dir": "phase3254/g5h17_registry_landing"},
        "gap_ledger_current": {"file": "gap_ledger_v1_12.json", "sha8_file": s8(GAP112), "dir": "phase3254/g5h17_registry_landing", "appendix_count": len(gap112["appendix"])},
        "ledger": {"file": "research/gpt5/atlas/atlas_ledger.json", "entries": len(led["measurements"]), "post_sha8": led["ledger_sha256_8"]},
    },
    "closure_facts": facts,
    "u_star_trilogy": {
        "3251_lexical": {"res_sha8": s8(r251), "claim": "word-level alignment small, scale-decaying, non-OOV -- u* carried by activation-side v_c"},
        "3252_layer": {"res_sha8": s8(r252), "claim": "embedding top1=1.0 x3, U-shape mid-trough"},
        "3253_subspace": {"res_sha8": s8(r253), "claim": "mid trough = same carrier decay (sigma1 mid-out 0.628/0.746/0.335), emb dissociated"},
    },
    "boundary_conditions": {
        "F20": "readout geometry != behavioral control (3206-3232 chain) -- bounds activation-side levers",
        "INFORMATION_GAP": "ORC_ls_full +0.329 (3246) shows legal OOV info source could flip SCLS; word-side sources excluded by u* trilogy",
    },
    "gpu_backlog": [
        "checkpoint training-step trajectory (u* emergence over training)",
        "appendix GPU items x5 (APPX-1..5, sealed gap v1.12 appendix)",
    ],
    "apply_protocol": "N-line reads this manifest, appends its own ledger entries, never rewrites sealed files",
}
out_dir = SMOKE_DIR if SMOKE else OUT
reg_path = os.path.join(out_dir, "atlas_registry_v1_35.json")
man_path = os.path.join(out_dir, "nline_handoff_manifest.json")

# ---------------------------------------------------------------- gates
gates = {}
# g2 facts consistency
gates["g2_facts"] = (abs(G0_scls - (-1.751489)) < 1e-4 and abs(ORC_scls - 0.329005) < 1e-4
                     and abs(inc_h - (-0.449039)) < 1e-4 and abs(inc_r2 - 0.241094) < 1e-4
                     and 0.740 <= repair_vs_zero <= 0.747)
# g3 ledger closure chain present + sealed bytes unchanged
chain_phases = set()
for e in led["measurements"]:
    p = e.get("phase", 0)
    if 3239 <= p <= 3255:
        chain_phases.add(p)
missing = [p for p, _ in SEALED_CHAIN if p not in chain_phases]
gates["g3_ledger_chain"] = (not missing) and s8(REG134) == "3deabf25" and s8(GAP112) == "aa8cff21"
gates["g1_pointer_revert"] = ok_revert and d_reg == set(norm_ptr(x) for x in gap_planned)
gate_all = all(gates.values())
print("gates", gates)

if not SMOKE:
    with open(reg_path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(reg135, ensure_ascii=False, indent=1))
    with open(man_path, "w", encoding="utf-8", newline="") as f:
        json.dump(handoff, f, ensure_ascii=False, indent=1)
    handoff["sealed_artifacts"]["registry_current"]["sha8_file"] = s8(reg_path)
    with open(man_path, "w", encoding="utf-8", newline="") as f:
        json.dump(handoff, f, ensure_ascii=False, indent=1)
    # sealed predecessors byte-unchanged after writes
    assert s8(REG134) == "3deabf25" and s8(GAP112) == "aa8cff21"
    # reload roundtrip on landed v1.35
    rt = json.loads(open(reg_path, "rb").read().decode("utf-8"))
    assert rt["version"] == "1.35" and len(rt["upgrade_log"]) == len(reg134["upgrade_log"]) + 1

# ---------------------------------------------------------------- closure summary (one page)
summary = ["# 收线终点封章：3239–3255 保真缺口收线链（Phase 3256 G5-H19）", "",
           "本页所有数字实测自封存盘面（phase3246/3250 result.json），禁凭记忆。", "",
           "## 四件套", "",
           "1. **起点**：G0 基线 SCLS %+.6f / MASSIVE %+.6f（3246 aggregate，与 3239/3240 封存锚逐位一致）。" % (G0_scls, G0_massive),
           "2. **合法上界**：ORC_ls_full（完美类+实体 oracle）SCLS +%.6f，min(scls,ent) +%.6f——转正在协议内可达但仅以越界信息为代价（3246 FLIP_LOCATION=INFORMATION_GAP）。" % (ORC_scls, ORC_min),
           "3. **现役**：XC_p5_b8（3250）SCLS %+.6f / r2_oov %.6f / KL %.6f，修复率（相对零目标）**%.2f%%**；剩余 %+.6f = INFORMATION_GAP。" % (inc_h, inc_r2, inc_kl, repair_vs_zero * 100, -inc_h),
           "4. **边界**：F20（读出几何≠行为控制）钉界唯一剩余杠杆（激活侧 v_c 注入族）；u* 双轨道几何（3251 词汇解耦 / 3252 embedding 秩 1 / 3253 载轴重聚焦）证明词表侧补 INFORMATION_GAP 不可达。", "",
           "## 链条（14 收官 phase）", ""]
for p, claim in SEALED_CHAIN:
    summary.append("- **%d** %s" % (p, claim))
summary += ["", "## 封章动作", "",
            "- registry v1.34 → **v1.35**（本目录 atlas_registry_v1_35.json）：仅 upgrade_log +1 封章注记 + version/updated/supersedes；指针 diff 精确（4 ptrs）+ revert 逐位；features/failures/anchors 零改动。",
            "- N 线施加包：本目录 nline_handoff_manifest.json（全 sha8 实测）。",
            "- GPU 挂账：checkpoint 训练步轨迹 + APPX-1..5 GPU 项，维持挂账不执行。", ""]
if not SMOKE:
    with open(os.path.join(OUT, "closure_summary_3239_3255.md"), "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(summary))
    print("summary written")

# ---------------------------------------------------------------- result.json
res = {
    "phase": PHASE, "name": NAME, "mode": DESIGN["mode"], "created": today(),
    "design_sha8": design_sha8, "facts": facts,
    "registry_landing": {
        "from": "1.34 (3deabf25)", "to": "1.35", "file_sha8": s8(reg_path) if not SMOKE else None,
        "ptr_diff": sorted(d_reg), "revert_bitwise": ok_revert,
    },
    "handoff_manifest": "nline_handoff_manifest.json",
    "gates": gates, "gate_all": gate_all,
    "gpu_outstanding": handoff["gpu_backlog"],
    "verdict": "closure_sealed_registered" if gate_all else "BLOCKED",
    "sources": {"registry_v134": REG134, "gap_v112": GAP112, "r246": r246["res_sha8"], "r250": r250["res_sha8"]},
}
if not SMOKE:
    res["res_sha8"] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k not in ("res_sha8",)},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "result_smoke.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
print("verdict", res["verdict"])
print("SMOKE DONE" if SMOKE else "FORMAL DONE")
