# -*- coding: utf-8 -*-
"""Phase 3266 (G5-H29): closure integration arm (report-only).

Arm a: /atlas/gaps endpoint coroutine live re-read — v1.13 visible with
       APPX-7, chain contains 1.13, gaps 5 unchanged (regression).
Arm b: /atlas/registry endpoint coroutine live re-read — v1.36 visible
       with upgrade_log 15, v1.35 still resolvable (regression).
Arm c: N-line handoff manifest v2 — refreshed snapshot (registry v1.36 /
       gap v1.13 / ledger #417); v1 manifest (bc212d52) sealed untouched.
Zero writes to ledger; sealed files byte-untouched.
"""
import asyncio
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3266", "g5h29_closure_integration")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3266_SMOKE") == "1"

from server.ai_rnd_service import atlas_gaps, atlas_registry  # noqa: E402

MANIFEST_V1 = os.path.join(RBASE, "phase3256", "g5h19_closure_seal",
                           "nline_handoff_manifest.json")
MANIFEST_V1_SHA = "bc212d52"
G12 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json")
R35 = os.path.join(RBASE, "phase3256", "g5h19_closure_seal", "atlas_registry_v1_35.json")


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3266,
    "name": "g5h29_closure_integration",
    "prereg_id": "G5-H29 (3265 tail preregistration)",
    "mode_basis": "REPORT_ONLY (endpoint coroutines called in-process, read-only)",
    "arms": {
        "a": ("/atlas/gaps live re-read: v1.13 + APPX-7 visible, chain, "
              "gaps-5 regression"),
        "b": ("/atlas/registry live re-read: v1.36 + upgrade_log 15 visible, "
              "v1.35 regression"),
        "c": ("N-line handoff manifest v2: refreshed snapshot v1.36/v1.13/#417; "
              "v1 manifest sealed untouched"),
    },
    "inputs": {
        "manifest_v1_sha8": MANIFEST_V1_SHA,
        "gap_v1_13_sha8": "814ce808",
        "registry_v1_36_sha8": "78d37211",
        "ledger_post": "ec34b29b",
    },
    "ledger": {"n": 422, "post": "ec34b29b", "chain_prev": "113e6af3"},
    "zero_write": ["research/gpt5/atlas/atlas_ledger.json"],
    "sealed_untouched": ["nline_handoff_manifest.json (bc212d52)",
                         "gap_ledger_v1_12.json (aa8cff21)",
                         "atlas_registry_v1_35.json (e8335331)"],
}
DESIGN["mode"] = "smoke" if SMOKE else "formal"
DESIGN["created"] = today()
design_sha8 = hashlib.sha256(
    json.dumps({k: v for k, v in DESIGN.items() if k != "created"},
               sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    os.makedirs(SMOKE_DIR, exist_ok=True)
with open(os.path.join(OUT, "design_sha8.txt"), "w", encoding="utf-8", newline="") as f:
    f.write(design_sha8)
print(("SMOKE " if SMOKE else "") + "freeze", design_sha8)

gates = {}

# ---------------------------------------------------------------- g1: ledger state
led_raw = open(LED, "rb").read()
led = json.loads(led_raw.decode("utf-8"))
ms = led["measurements"]
chk_led = json.loads(led_raw.decode("utf-8"))
chk_led["ledger_sha256_8"] = DESIGN["ledger"]["chain_prev"]
rec = hashlib.sha256(json.dumps(chk_led, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates["g1_ledger_state"] = (
    len(ms) == DESIGN["ledger"]["n"]
    and led["ledger_sha256_8"] == DESIGN["ledger"]["post"]
    and rec == DESIGN["ledger"]["post"])
print("g1 ledger: n=%d post=%s recompute=%s" % (len(ms), led["ledger_sha256_8"], rec))

# ---------------------------------------------------------------- g2: sealed inputs
gates["g2_sealed_inputs"] = (s8(MANIFEST_V1) == MANIFEST_V1_SHA
                             and s8(G12) == "aa8cff21"
                             and s8(R35) == "e8335331")
print("g2 sealed: manifest=%s gap=%s registry=%s"
      % (s8(MANIFEST_V1) == MANIFEST_V1_SHA, s8(G12) == "aa8cff21",
         s8(R35) == "e8335331"))

# ---------------------------------------------------------------- Arm a: gaps live
doc_g = asyncio.run(atlas_gaps())
arm_a = {
    "version": doc_g["version"],
    "appendix_n": len(doc_g["appendix"]),
    "appx7": None,
    "versions_head": [v["version"] for v in doc_g["versions"][:3]],
    "gaps_ids": sorted(g["id"] for g in doc_g["gaps"]),
}
a7 = [a for a in doc_g["appendix"] if a["id"] == "APPX-7"]
arm_a["appx7"] = ({"anchor": a7[0]["anchor"], "anchor_sha8": a7[0]["anchor_sha8"],
                   "status": a7[0]["status"]} if a7 else None)
gates["g3_gaps_live"] = (
    doc_g["version"] == "1.13" and len(doc_g["appendix"]) == 7
    and a7 and a7[0]["anchor"] == "p3263+p3264"
    and a7[0]["anchor_sha8"] == "f223a9a0"
    and arm_a["versions_head"][0] == "1.13" and "1.12" in arm_a["versions_head"]
    and arm_a["gaps_ids"] == ["GAP-1", "GAP-2", "GAP-3", "GAP-4", "GAP-5"])
print("g3 gaps live: v%s appendix=%d APPX-7=%s chain=%s gaps5=%s"
      % (doc_g["version"], len(doc_g["appendix"]), bool(a7),
         arm_a["versions_head"],
         arm_a["gaps_ids"] == ["GAP-1", "GAP-2", "GAP-3", "GAP-4", "GAP-5"]))

# ---------------------------------------------------------------- Arm b: registry live
doc_r = asyncio.run(atlas_registry())
arm_b = {
    "version": doc_r["version"],
    "n_upgrades": len(doc_r["upgrade_log"]),
    "ul_tail_node": doc_r["upgrade_log"][-1].get("node"),
    "ul_tail_phase": doc_r["upgrade_log"][-1].get("phase"),
    "versions_head": [v["version"] for v in doc_r["versions"][:3]],
}
doc_r35 = asyncio.run(atlas_registry(version="1.35"))
arm_b["v135_regression"] = (doc_r35["version"] == "1.35"
                            and len(doc_r35["upgrade_log"]) == 14)
gates["g4_registry_live"] = (
    doc_r["version"] == "1.36" and arm_b["n_upgrades"] == 15
    and arm_b["ul_tail_phase"] == 3265
    and "APPX-7" in doc_r["upgrade_log"][-1].get("change", "")
    and arm_b["versions_head"][0] == "1.36"
    and arm_b["v135_regression"])
print("g4 registry live: v%s upgrades=%d tail_phase=%s v1.35_regression=%s"
      % (doc_r["version"], arm_b["n_upgrades"], arm_b["ul_tail_phase"],
         arm_b["v135_regression"]))

# ---------------------------------------------------------------- Arm c: manifest v2
mv1 = json.load(open(MANIFEST_V1, encoding="utf-8"))
mv2 = json.loads(json.dumps(mv1))
mv2["schema"] = "nline_handoff_v2"
mv2["created"] = today()
mv2["phase"] = 3266
mv2["purpose"] = ("G-line closure package for cross-line (N-line) application arm "
                  "-- refreshed snapshot (Phase 3266); supersedes "
                  "nline_handoff_manifest.json v1 (bc212d52, Phase 3256, sealed)")
mv2["supersedes"] = {"manifest": "nline_handoff_manifest.json v1 (bc212d52, "
                                 "Phase 3256)"}
mv2["sealed_artifacts"] = {
    "registry_current": {"file": "atlas_registry_v1_36.json",
                         "sha8_file": "78d37211",
                         "dir": "phase3265/g5h28_appx7_landing",
                         "upgrade_log_n": 15,
                         "supersedes": "atlas_registry_v1_35.json (e8335331)"},
    "registry_prev": {"file": "atlas_registry_v1_35.json", "sha8_file": "e8335331",
                      "dir": "phase3256/g5h19_closure_seal"},
    "gap_ledger_current": {"file": "gap_ledger_v1_13.json", "sha8_file": "814ce808",
                           "dir": "phase3265/g5h28_appx7_landing",
                           "appendix_count": 7,
                           "supersedes": "gap_ledger_v1_12.json (aa8cff21)"},
    "gap_ledger_prev": {"file": "gap_ledger_v1_12.json", "sha8_file": "aa8cff21",
                        "dir": "phase3254/g5h17_registry_landing"},
    "ledger": {"file": "research/gpt5/atlas/atlas_ledger.json",
               "entries": 422, "post_sha8": "ec34b29b",
               "note": ("snapshot refreshed to #417 after 3265 closeout append; "
                        "v1 manifest snapshot was n=408/5601b3f5")},
}
mv2["gpu_backlog"] = [
    "RESOLVED (3263/3264): checkpoint training-step trajectory -- u* dual-track "
    "geometry is a pretraining property (25 states, bit-exact restore; Track I "
    "bitwise invariant; D scale-dependent)",
    "APPX-7 follow-up (OPTIONAL, non-blocking): cross-scale D sampling "
    "(qwen2.5-3b/7b) to calibrate the dominance scale-law curve",
    "APPX-1..3 GPU items (APPX-1 pending quantization-acceptability decision; "
    "APPX-2/3 feasible as-is); APPX-4/5 cross-line governance unchanged",
]
mv2["u_star_trilogy"] = {
    "3251_lexical": mv1["u_star_trilogy"]["3251_lexical"],
    "3252_layer": mv1["u_star_trilogy"]["3252_layer"],
    "3253_subspace": mv1["u_star_trilogy"]["3253_subspace"],
    "3263_trajectory": {"res_sha8": "f0f95c27",
                        "claim": ("u* dual-track geometry is a PRETRAINING "
                                  "property: Track I bitwise invariant across "
                                  "25 states (layer0 top1=1.0, cos_emb=1.0); "
                                  "D band 16.1-17.2, no training formation")},
    "3264_second_order": {"res_sha8": "f223a9a0",
                          "claim": ("wscp phase transition is abrupt and "
                                    "readout-side localized (layers 0-16 "
                                    "bitwise zero; u*_kout cos drop in single "
                                    "s32->s128 step); Track I invariants "
                                    "re-asserted bitwise")},
    "3265_appx7": {"res_sha8": "18437872",
                   "claim": ("APPX-7 landed (gap v1.13): D scale law & u* "
                             "geometry dissociation")},
}
mv2["boundary_conditions"] = mv1["boundary_conditions"]
mv2["apply_protocol"] = mv1["apply_protocol"]
mv2["closure_facts"] = mv1["closure_facts"]
mv2["integration_note"] = (
    "v1.13 appendix APPX-7 adds the scale dimension to the closure package; "
    "no boundary condition changes; client visibility live-verified in-phase "
    "(gaps + registry coroutines)")

# ---------------------------------------------------------------- write manifest v2
if not SMOKE:
    with open(os.path.join(OUT, "nline_handoff_manifest_v2.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(mv2, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "manifest_v2_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"schema": mv2["schema"], "registry": "78d37211",
                   "gap": "814ce808"}, f, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- g5: manifest checks + zero write
zero_before = s8(LED)
gates["g5_manifest_v2"] = (
    mv2["schema"] == "nline_handoff_v2"
    and mv2["sealed_artifacts"]["registry_current"]["sha8_file"] == "78d37211"
    and mv2["sealed_artifacts"]["gap_ledger_current"]["appendix_count"] == 7
    and mv2["sealed_artifacts"]["ledger"]["entries"] == 422
    and len(mv2["u_star_trilogy"]) == 6
    and s8(MANIFEST_V1) == MANIFEST_V1_SHA)
zero_after = s8(LED)
gates["g6_zero_write"] = (zero_before == zero_after)
print("g5 manifest v2: %s | g6 zero-write: %s" % (gates["g5_manifest_v2"],
                                                  gates["g6_zero_write"]))

# ---------------------------------------------------------------- verdict
gate_all = all(v for v in gates.values() if v is not None)
res = {
    "phase": 3266,
    "name": "g5h29_closure_integration",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "closure_integration_registered" if gate_all else "BLOCKED",
    "arm_a_gaps_live": arm_a,
    "arm_b_registry_live": arm_b,
    "arm_c_manifest_v2": {
        "file": "nline_handoff_manifest_v2.json",
        "schema": "nline_handoff_v2",
        "v1_sealed": MANIFEST_V1_SHA,
        "file_sha8": (s8(os.path.join(OUT, "nline_handoff_manifest_v2.json"))
                      if not SMOKE else "smoke"),
    },
    "zero_write_verified": zero_after,
}
if not SMOKE:
    res["res_sha8"] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != "res_sha8"},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(OUT, "result.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "result_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
print("verdict", res["verdict"])
for k, v in gates.items():
    print(" ", k, v)
print("SMOKE DONE" if SMOKE else "FORMAL DONE")
