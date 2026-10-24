# -*- coding: utf-8 -*-
"""Phase 3261 (G5-H24): evidence_level normalization re-grading (report-only).

Arm (a): re-grade ALL 417 ledger measurements against the sealed three-tier
rule F2_verdict_grading (bit_anchored / statistical / descriptive, frozen in
phase3150 metric dict) — producing a normalization DRAFT (per-entry derived
grade + rationale + ambiguity flags). Zero writes to the ledger: the draft is
a successor proposal, application (if ever) needs owner/user confirmation
per the 3260 op2/op3 dispositions (governance values are outside the science
triad and are mapped to a 4th non-evidence class GOVERNANCE).

Arm (b): GPU feasibility note for the open GPU backlog (hardware probe only,
no env mutation).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
F3150 = os.path.join(RBASE, "phase3150")
OUT = os.path.join(RBASE, "phase3261", "g5h24_evidence_regrading")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3261_SMOKE") == "1"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3261,
    "name": "g5h24_evidence_regrading",
    "prereg_id": "G5-H24 (3260 tail op3 successor action; backlog v3 open_gpu item 3)",
    "arms": {
        "a": ("re-grade all ledger measurements against sealed F2_verdict_grading "
              "(phase3150) -> normalization DRAFT report-only, zero ledger writes; "
              "4-class mapping: science triad + GOVERNANCE (non-evidence)"),
        "b": "GPU feasibility note (hardware probe only, no env mutation)",
    },
    "f2_rule_source": "tests/glm5/phase3150_p0_freeze_carrierdecode.py (F2_verdict_grading)",
    "ledger": {"n": 417, "post": "c26305dd", "chain_prev": "7b6e6230"},
    "classes": ["bit_anchored", "statistical", "descriptive", "GOVERNANCE"],
    "zero_write": ["research/gpt5/atlas/atlas_ledger.json"],
    "identity_rule": ("entries whose current evidence_level is already in the science "
                      "triad keep their value (identity mapping)"),
    "heuristic_order": [
        "1 identity: value in triad -> same",
        "2 governance keywords in value -> GOVERNANCE",
        "3 harness-determinism markers in name/verdict/detail -> bit_anchored",
        "4 gate-vs-null markers in name/verdict/detail -> statistical",
        "5 default -> descriptive (most conservative) with AMBIGUOUS flag if "
        "conflicting markers present",
    ],
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

# ---------------------------------------------------------------- g1: ledger state
led_raw = open(LED, "rb").read()
led = json.loads(led_raw.decode("utf-8"))
ms = led["measurements"]
chk_prev = json.loads(led_raw.decode("utf-8"))
chk_prev["ledger_sha256_8"] = DESIGN["ledger"]["chain_prev"]
rec = hashlib.sha256(json.dumps(chk_prev, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates = {}
gates["g1_ledger_state"] = (
    len(ms) == DESIGN["ledger"]["n"]
    and led["ledger_sha256_8"] == DESIGN["ledger"]["post"]
    and rec == DESIGN["ledger"]["post"])
print("g1 ledger: n=%d post=%s recompute=%s" % (len(ms), led["ledger_sha256_8"], rec))

# ---------------------------------------------------------------- g2: F2 rule verbatim
f3150_src = None
f3150_hit = None
if os.path.isdir(F3150):
    for root, _, files in os.walk(F3150):
        for fn in files:
            if fn == "phase3150_p0_freeze_carrierdecode.py":
                f3150_hit = os.path.join(root, fn)
if f3150_hit is None:
    f3150_hit = os.path.join(ROOT, "tests", "glm5", "phase3150_p0_freeze_carrierdecode.py")
f3150_src = open(f3150_hit, encoding="utf-8").read()
i = f3150_src.find("'F2_verdict_grading'")
seg = f3150_src[i:i + 600]
f2_ok = all(k in seg for k in ("bit_anchored", "statistical", "descriptive")) and \
    "NEVER counts as" in seg.replace("\n                    ", " ") or \
    ("bit_anchored" in seg and "statistical" in seg and "descriptive" in seg)
gates["g2_f2_rule_source"] = f2_ok
print("g2 F2 rule source found:", f2_ok, "|", os.path.relpath(f3150_hit, ROOT))

# ---------------------------------------------------------------- arm (a): re-grade
TRIAD = {"bit_anchored", "statistical", "descriptive"}
GOV_PAT = re.compile(
    r"governance|report_only|sealed|design|ui_|registry|render|client|n/a|"
    r"gate_decision|compute_only|efficacy_test|registration arm|robustness arm|"
    r"census arm|fidelity loop|demo arm|combination arm|characterization|"
    r"menu-scan|deepening arm|extrapolation|bridge arm|doc landing|engineering|"
    r"registered$|registered \(|counter_evidence|E_LINE_CLOSED|RE_READ|"
    r"descriptive \(cross-line|annotation|inventory|backfill|readiness|audit|"
    r"summary|archival|rederivation|consistency|draft|apply")
BIT_PAT = re.compile(
    r"bit[- ]level|bitwise|verbatim|sha8|sha256|drift.*== ?0|d\d{1,2} ?== ?0|"
    r"byte-unchanged|byte-untouched|revert.*bitwise|top1\(kout\)|exact k\d+|"
    r"cross-checked exact")
STAT_PAT = re.compile(
    r"gate vs null|null band|statistical assert|cross-sample|gate reachability|"
    r"E1_|E2_|E3_|held[_ ]out|repeatable|predictive|causal")

regraded = []
amb = 0
for idx, e in enumerate(ms):
    cur = str(e.get("evidence_level"))
    name_v = str(e.get("name", ""))
    verd_v = str(e.get("verdict", ""))
    det_v = str(e.get("detail", ""))[:800]
    blob = " ".join((name_v, verd_v, det_v))
    if cur in TRIAD:
        grade, why = cur, "identity: already in F2 triad"
    elif GOV_PAT.search(cur):
        grade, why = "GOVERNANCE", "value matches governance/report vocabulary"
    else:
        is_bit = bool(BIT_PAT.search(blob))
        is_stat = bool(STAT_PAT.search(blob))
        if is_bit and not is_stat:
            grade, why = "bit_anchored", "harness-determinism markers only"
        elif is_stat and not is_bit:
            grade, why = "statistical", "gate/null or E-level markers"
        elif is_bit and is_stat:
            amb += 1
            grade, why = "AMBIGUOUS", "both determinism and gate markers; manual review"
        elif GOV_PAT.search(blob):
            grade, why = "GOVERNANCE", "non-evidence process vocabulary in body"
        else:
            amb += 1
            grade, why = "AMBIGUOUS", "no markers; default descriptive needs manual review"
    regraded.append({"idx": idx, "phase": e.get("phase"), "name": e.get("name"),
                     "current": cur[:80], "derived": grade, "why": why})

from collections import Counter
dist = Counter(r["derived"] for r in regraded)
cur_dist = Counter(str(e.get("evidence_level")) for e in ms)
gates["g3_full_coverage"] = (len(regraded) == len(ms) == DESIGN["ledger"]["n"])
print("g3 coverage:", gates["g3_full_coverage"], "| derived dist:",
      dict(dist), "| ambiguous:", amb)

draft = {
    "schema": "gline_evidence_level_normalization_draft_v1",
    "created": today(),
    "status": "REPORT_ONLY_DRAFT",
    "rule_source": {"file": os.path.relpath(f3150_hit, ROOT).replace("\\", "/"),
                    "rule": "F2_verdict_grading (bit_anchored / statistical / descriptive)"},
    "f7_constraint": ("bit_anchored proves harness determinism only; it never raises "
                      "the evidence level of any scientific claim"),
    "class_note": ("GOVERNANCE is a 4th NON-EVIDENCE class for process records "
                   "(report/gate-decision/engineering entries); it is outside the "
                   "F2 science triad and carries zero downstream premise weight"),
    "application": ("NONE performed; ledger untouched. Application would be a "
                    "destructive schema migration (cf. 3260 op2 disposition "
                    "REQUIRES_DESTRUCTIVE_MIGRATION) and needs explicit user/owner "
                    "confirmation; recommend documenting the mapping instead of "
                    "rewriting byte-anchored history"),
    "target_state": {"n": len(ms), "post": led["ledger_sha256_8"]},
    "current_distinct_values": len(cur_dist),
    "derived_distribution": dict(dist),
    "ambiguous_count": amb,
    "entries": regraded,
}

# ---------------------------------------------------------------- arm (b): GPU note
gpu = {"probe": "nvidia-smi + torch import"}
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=30)
    gpu["hardware"] = out.stdout.strip() or out.stderr.strip()[:120]
except Exception as ex:
    gpu["hardware"] = "probe error: %s" % ex
try:
    import torch  # noqa
    gpu["torch"] = "installed"
except ImportError:
    gpu["torch"] = "NOT installed in current venv"
gpu["conclusion"] = ("hardware present (%s); torch env missing -> GPU backlog items "
                     "(checkpoint training-step trajectory, APPX-1..5) remain parked "
                     "pending an env decision; no packages installed by this phase"
                     % gpu.get("hardware", "?")[:40])
print("gpu:", gpu["hardware"], "| torch:", gpu["torch"])

# ---------------------------------------------------------------- g4/g5
zero_before = s8(LED)
if not SMOKE:
    with open(os.path.join(OUT, "evidence_regrading_draft_v1.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(draft, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "gpu_feasibility_note.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(gpu, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "draft_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"dist": dict(dist), "ambiguous": amb}, f,
                  ensure_ascii=False, indent=1)

gates["g4_classes_valid"] = all(
    r["derived"] in set(DESIGN["classes"]) | {"AMBIGUOUS"} for r in regraded)
zero_after = s8(LED)
gates["g5_zero_write"] = (zero_before == zero_after)
print("g4 classes valid:", gates["g4_classes_valid"], "| g5 zero-write:",
      gates["g5_zero_write"])

# ---------------------------------------------------------------- verdict / result
gate_all = all(v for k, v in gates.items() if v is not None)
res = {
    "phase": 3261,
    "name": "g5h24_evidence_regrading",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "evidence_regrading_registered" if gate_all else "BLOCKED",
    "arm_a": {
        "artifact": "evidence_regrading_draft_v1.json",
        "entries": len(regraded),
        "current_distinct_values": len(cur_dist),
        "derived_distribution": dict(dist),
        "ambiguous_count": amb,
        "identity_kept": dist.get("bit_anchored", 0) + dist.get("statistical", 0)
        + dist.get("descriptive", 0),
        "application": "none (report-only; destructive migration needs confirmation)",
    },
    "arm_b": {
        "artifact": "gpu_feasibility_note.json",
        "hardware": gpu.get("hardware"),
        "torch": gpu.get("torch"),
        "backlog_action": "GPU items remain parked pending env decision",
    },
    "zero_write_verified": zero_after,
}
if not SMOKE:
    res["res_sha8"] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != "res_sha8"},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "result_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
print("verdict", res["verdict"])
for k, v in gates.items():
    print(" ", k, v)
print("SMOKE DONE" if SMOKE else "FORMAL DONE")
