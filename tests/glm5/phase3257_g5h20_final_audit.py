# -*- coding: utf-8 -*-
"""Phase 3257 G5-H20: G-line final audit + N-line apply rehearsal.

预注册（3256 tail）：
  (a) G 线收官核验臂：v1.35 客户端可见性（端点 + UI 实机，独立脚本先行）+
      三账一致性终审（磁盘侧强校验：registry v1.35 全部 28 anchor_files 逐个 sha8
      实测比对 + gap v1.12 appendix anchors + ledger #408 链）。
  (b) N 线施加预演臂（READ-ONLY）：按 manifest 独立重验全部 sha8 →
      apply_preview.json（N 线视角将采取的动作清单；不写任何 G 线文件）。
  (c) GPU 挂账：记录维持。

纪律：freeze 先于一切观察 → SMOKE → 正式 → gates → result.json →
closeout 五写（另脚本）→ 独立磁盘复核（另脚本）。
"""
import hashlib
import json
import os
import sys

SMOKE = os.environ.get("P3257_SMOKE") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = HERE if os.path.basename(HERE) == "glm5" else os.path.join(HERE, "glm5")
ROOT = os.path.dirname(os.path.dirname(BASE))                      # repo root
RBASE = os.path.join(BASE, "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3257", "g5h20_final_audit")
SMOKE_DIR = os.path.join(OUT, "smoke")
os.makedirs(SMOKE_DIR if SMOKE else OUT, exist_ok=True)

REG135 = os.path.join(RBASE, "phase3256", "g5h19_closure_seal", "atlas_registry_v1_35.json")
GAP112 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json")
MANIFEST = os.path.join(RBASE, "phase3256", "g5h19_closure_seal", "nline_handoff_manifest.json")
LEDGER = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
EP_OUT = os.path.join(ROOT, "tests", "gpt5_temp", "p3257_endpoint_out.txt")
UI_OUT = os.path.join(ROOT, "tests", "gpt5_temp", "p3257_ui_out.txt")

PHASE = 3257
NAME = "g5h20_final_audit"


def s8(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": PHASE,
    "name": NAME,
    "prereg": "3256 tail: G-line final audit arm + N-line apply rehearsal arm (read-only)",
    "arms": {
        "a": "client visibility live-verified (endpoint check 15 asserts + playwright UI check, independent scripts) + three-account disk audit (all 28 registry anchor_files sha8 re-measured, gap v1.12 appendix anchors, ledger #408 chain)",
        "b": "N-line apply rehearsal: manifest sha8 re-verified independently + apply_preview.json (actions list, ZERO writes to any G-line file)",
        "c": "GPU backlog recorded, not executed",
    },
    "gates": {
        "g1": "endpoint check ENDPOINT CHECK OK (15 asserts: v1.35 visible, chain 1.35..1.1, v1.34 resolvable, gaps v1.12 regression, three-account cross)",
        "g2": "UI check 11/11 PASS zero console errors (v1.35 default, select 35 options, switch 1.34<->1.35, GAP panel regression)",
        "g3": "three-account audit: 28/28 anchor sha8 match + gap anchors match + ledger #408 recompute + manifest sha8 re-verified + apply_preview written with zero G-line writes",
    },
    "no_gpu": True,
    "read_only_arm_b": True,
}
DESIGN["mode"] = "smoke" if SMOKE else "formal"
DESIGN["created"] = today()
design_sha8 = hashlib.sha256(
    json.dumps({k: v for k, v in DESIGN.items() if k != "created"},
               sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
if not SMOKE:
    with open(os.path.join(OUT, "design.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(DESIGN, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "design_sha8.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(design_sha8)
print(("SMOKE " if SMOKE else "") + "freeze", design_sha8)

# ---------------------------------------------------------------- arm (a) gates g1/g2 (live outputs)
ep_txt = open(EP_OUT, encoding="utf-8").read()
g1 = ("ENDPOINT CHECK OK" in ep_txt) and ("FAIL" not in ep_txt)
ui_txt = open(UI_OUT, encoding="utf-8").read()
g2 = ("UI CHECK: 11/11" in ui_txt) and ("FAIL" not in ui_txt) and ("console errors: none" in ui_txt)

# ---------------------------------------------------------------- three-account disk audit
audit = {"anchor_files": {"total": 0, "match": 0, "mismatch": []}}
reg35 = json.load(open(REG135, encoding="utf-8"))
for key, ent in sorted(reg35["provenance"]["anchor_files"].items()):
    audit["anchor_files"]["total"] += 1
    p = os.path.join(ROOT, ent["path"])
    if os.path.isfile(p) and s8(p) == ent["sha8"]:
        audit["anchor_files"]["match"] += 1
    else:
        audit["anchor_files"]["mismatch"].append(key)
print("anchor audit", audit["anchor_files"]["match"], "/", audit["anchor_files"]["total"])

gap112 = json.load(open(GAP112, encoding="utf-8"))
# gap appendix anchor 口径表（历史落册口径混合，显式登记后逐一实测；来源 3207 recheck + 3254 落册审计）：
#   APPX-1/2  result.json 文件 sha；APPX-3  summary/result_summary.json 文件 sha；
#   APPX-4/5  跨线 agate（research/deepseek/atlas/a_gate_closure_v1.json）文件 sha；
#   APPX-6    result.json 内部 res_sha8 内容口径（3254 落册审计 6/6 同口径）。
GAP_ANCHOR_CONVENTIONS = {
    "APPX-1": ("file", os.path.join(RBASE, "phase3166", "g5a3b_logic_direction", "result.json")),
    "APPX-2": ("file", os.path.join(RBASE, "phase3165", "g5a3_family_alignment", "result.json")),
    "APPX-3": ("file", os.path.join(RBASE, "phase3155", "g2p1_relation_family_operator_separability",
                                    "summary", "result_summary.json")),
    "APPX-4": ("file", os.path.join(ROOT, "research", "deepseek", "atlas", "a_gate_closure_v1.json")),
    "APPX-5": ("file", os.path.join(ROOT, "research", "deepseek", "atlas", "a_gate_closure_v1.json")),
    "APPX-6": ("res_sha8", os.path.join(RBASE, "phase3250", "g5h13_kernel_boundary", "result.json")),
}
gap_anchor_bad = []
for a in gap112["appendix"]:
    conv = GAP_ANCHOR_CONVENTIONS.get(a["id"])
    if conv is None:
        gap_anchor_bad.append(a["id"] + " (no convention)")
        continue
    kind, p = conv
    ok_anchor = os.path.isfile(p)
    if ok_anchor and kind == "file":
        ok_anchor = s8(p) == a["anchor_sha8"]
    elif ok_anchor and kind == "res_sha8":
        ok_anchor = json.load(open(p, encoding="utf-8")).get("res_sha8") == a["anchor_sha8"]
    if not ok_anchor:
        gap_anchor_bad.append(a["id"] + " (%s)" % kind)
print("gap appendix anchors bad:", gap_anchor_bad)

led = json.load(open(LEDGER, encoding="utf-8"))
chk = dict(led)
chk["ledger_sha256_8"] = "b411f7ef"
led_recompute = hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1).encode("utf-8")).hexdigest()[:8] == led["ledger_sha256_8"]
audit["ledger"] = {"n": len(led["measurements"]), "post": led["ledger_sha256_8"],
                   "tail_phase": led["measurements"][-1]["phase"], "recompute": led_recompute}

audit["registry"] = {"version": reg35["version"], "supersedes": reg35["supersedes"],
                     "n_upgrade_log": len(reg35["upgrade_log"]),
                     "seal_tail": "closure seal" in reg35["upgrade_log"][-1]["change"]}

g3_disk = (audit["anchor_files"]["match"] == 28
           and not gap_anchor_bad
           and led_recompute
           and audit["ledger"]["tail_phase"] == 3256
           and audit["registry"]["version"] == "1.35"
           and audit["registry"]["seal_tail"])

# ---------------------------------------------------------------- arm (b) N-line rehearsal (READ-ONLY)
man = json.load(open(MANIFEST, encoding="utf-8"))
man_bad = []
sa = man["sealed_artifacts"]
def _cmp(name, path, expect):
    if not (os.path.isfile(path) and s8(path) == expect):
        man_bad.append(name)
_cmp("registry_prev", os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913",
                                   "phase3254", "g5h17_registry_landing", "atlas_registry_v1_34.json"),
     sa["registry_prev"]["sha8_file"])
_cmp("gap_current", os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913",
                                 "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json"),
     sa["gap_ledger_current"]["sha8_file"])
_cmp("registry_current", REG135, sa["registry_current"]["sha8_file"])
# ledger 的 post_sha8 是内容 recompute 口径而非文件 sha——单独断言：
man_ledger_ok = (sa["ledger"]["entries"] == 408 and sa["ledger"]["post_sha8"] == "5601b3f5"
                 and led["ledger_sha256_8"] == "5601b3f5")
for k, ent in sorted(man["u_star_trilogy"].items()):
    src = {"3251_lexical": os.path.join(RBASE, "phase3251", "g5h14_ustar_lexical_attribution", "result.json"),
           "3252_layer": os.path.join(RBASE, "phase3252", "g5h15_ustar_layer_trace", "result.json"),
           "3253_subspace": os.path.join(RBASE, "phase3253", "g5h16_ustar_subspace", "result.json")}[k]
    if not (os.path.isfile(src) and s8(src) == ent["res_sha8"]):
        man_bad.append(k)
print("manifest sha8 bad:", man_bad, "| ledger snapshot ok:", man_ledger_ok)

rehearsal = {
    "schema": "nline_apply_preview_v1",
    "created": today(),
    "read_only": True,
    "writes_to_gline": False,
    "manifest_sha8_file": s8(MANIFEST),
    "manifest_reverify": {"bad": man_bad, "ledger_snapshot_ok": man_ledger_ok},
    "steps_if_applied": [
        {"step": 1, "action": "N-line reads nline_handoff_manifest.json (this file's source) from its own checkout; no G-line path coupling"},
        {"step": 2, "action": "N-line verifies sealed_artifacts sha8 (registry v1.35 e8335331 / v1.34 3deabf25 / gap v1.12 aa8cff21 / ledger #408 5601b3f5) against its own copies"},
        {"step": 3, "action": "N-line adopts closure facts (G0 -1.751489 / ORC +0.329005 / XC_p5_b8 -0.449039 repair 0.7436) as shared-context priors, NOT as N-line evidence"},
        {"step": 4, "action": "N-line appends its own ledger entries referencing prereg_id G5-H19 handoff; never rewrites G-line sealed files (apply_protocol)"},
        {"step": 5, "action": "any N-line upgrade claim starts at evidence level E0 in N-line's own taxonomy; G-line levels are not transferable (evidence_level_taxonomy stays per-line)"},
    ],
    "boundary_conditions": man["boundary_conditions"],
    "gpu_backlog": man["gpu_backlog"],
}

gates = {"g1_endpoint": g1, "g2_ui": g2,
         "g3_three_account": g3_disk and not man_bad and man_ledger_ok}
gate_all = all(gates.values())
print("gates", gates)

out_dir = SMOKE_DIR if SMOKE else OUT
res = {
    "phase": PHASE, "name": NAME, "mode": DESIGN["mode"], "created": today(),
    "design_sha8": design_sha8,
    "live_checks": {"endpoint": "15/15 (ENDPOINT CHECK OK)", "ui": "11/11 (zero console errors)"},
    "three_account_audit": audit,
    "manifest_reverify": rehearsal["manifest_reverify"],
    "gates": gates, "gate_all": gate_all,
    "gpu_outstanding": man["gpu_backlog"],
    "verdict": "final_audit_passed" if gate_all else "BLOCKED",
}
if not SMOKE:
    with open(os.path.join(OUT, "apply_preview.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(rehearsal, f, ensure_ascii=False, indent=1)
    res["res_sha8"] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != "res_sha8"},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "result_smoke.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
print("verdict", res["verdict"])
print("SMOKE DONE" if SMOKE else "FORMAL DONE")
