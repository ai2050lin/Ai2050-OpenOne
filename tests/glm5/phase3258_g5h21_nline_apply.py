# -*- coding: utf-8 -*-
"""Phase 3258 (G5-H21): N-line apply arm (append-type) + lcorr staleness disposition.

Arm (a): apply the SEALED N-line backfill draft f29da2de (phase3208 g5g7, 5 entries
N-P3..N-P7) to the shared atlas ledger research/gpt5/atlas/atlas_ledger.json
(target per draft: "#measurements (append 5)"). Append-only; draft file byte-unchanged;
chain recompute per house convention new_post = sha8(dumps(ledger with
ledger_sha256_8 = prev_post)).

Arm (b): disposition of research/deepseek/atlas/ledger_corrections_v1.json (beb775bd,
SEALED_BUT_NOT_APPLIED, 4 ops computed against a n=304 ledger state): verify staleness
on-disk, record STALE_AS_SEALED, apply nothing, touch nothing in deepseek/.

Readiness chain: 3218 readiness assessment N-line = ready_append_type; applied under
user continuation authorization announced at 3257 tail ("从挂账项中择序推进").
"""
import copy
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
DRAFT = os.path.join(RBASE, "phase3208", "g5g7_nline_backfill_draft",
                     "n_line_backfill_draft_v1.json")
LCORR = os.path.join(ROOT, "research", "deepseek", "atlas", "ledger_corrections_v1.json")
OUT = os.path.join(RBASE, "phase3258", "g5h21_nline_apply")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3258_SMOKE") == "1"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3258,
    "name": "g5h21_nline_apply",
    "prereg_id": "G5-H21 (3257 tail backlog order: N-line apply arm per apply_preview)",
    "arms": {
        "a": "apply sealed N-line backfill draft (5 entries N-P3..N-P7) to shared atlas ledger, append-only",
        "b": "lcorr beb775bd staleness disposition (report-only, zero deepseek writes)",
    },
    "expected_draft_sha8": "f29da2de",
    "expected_lcorr_sha8": "beb775bd",
    "ledger_prev_post": "15875b87",
    "ledger_n_before": 409,
    "ledger_n_after": 414,
    "planned_ledger_ptrs": ["$.measurements", "$.ledger_sha256_8"],
    "entry_names": [
        "e2_conditioned_compare_layer_scan",
        "n1_main_axis_layer_scan",
        "n2_reconstruction_source",
        "n2h1_permutation_ablation",
        "n3_device_transfer",
    ],
    "entry_phases": ["N-P3", "N-P4", "N-P5", "N-P6", "N-P7"],
    "lcorr_expected_disposition": "STALE_AS_SEALED",
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

# ---------------------------------------------------------------- pre-state
draft_sha_before = s8(DRAFT)
lcorr_sha_before = s8(LCORR)
ledger_raw = open(LED, "rb").read()
ledger_file_sha_before = hashlib.sha256(ledger_raw).hexdigest()
led = json.loads(ledger_raw.decode("utf-8"))
assert draft_sha_before == DESIGN["expected_draft_sha8"], draft_sha_before
assert lcorr_sha_before == DESIGN["expected_lcorr_sha8"], lcorr_sha_before
assert led["ledger_sha256_8"] == DESIGN["ledger_prev_post"], led["ledger_sha256_8"]
assert len(led["measurements"]) == DESIGN["ledger_n_before"]
# chain verify of current post (field = prev-prev post 5601b3f5)
_chk = copy.deepcopy(led)
_chk["ledger_sha256_8"] = "5601b3f5"
assert hashlib.sha256(json.dumps(_chk, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8] == DESIGN["ledger_prev_post"]
# serialization roundtrip (writer byte convention)
assert json.dumps(led, ensure_ascii=False, indent=1).encode("utf-8") == ledger_raw
print("pre-state OK: draft", draft_sha_before, "| lcorr", lcorr_sha_before,
      "| ledger n=%d post=%s (chain verify OK)"
      % (len(led["measurements"]), led["ledger_sha256_8"]))

# idempotency guard
present = [e for e in led["measurements"] if str(e.get("phase", "")).startswith("N-P")]
assert not present, "N-P entries already present: %s" % [e.get("phase") for e in present]

# ---------------------------------------------------------------- arm (a): build entries
draft = json.load(open(DRAFT, encoding="utf-8"))
assert draft["status"] == "draft_pending_cross_line"
assert len(draft["entries"]) == 5
TODAY = today()
entries = []
for de in draft["entries"]:
    ent = {
        "phase": de["phase"],                 # verbatim string phase (N-P3..N-P7)
        "name": de["name"],                   # verbatim
        "line": "N",
        "date": draft["created"].split(" ")[0],
        "title": de["title"],                 # verbatim
        "sources": de["sources"],             # verbatim
        "status": de["status"],               # verbatim draft status at sealing
        "evidence_level": (
            "descriptive (cross-line backfill from sealed N-line draft f29da2de; "
            "evidence levels are per-line and non-transferable per G5-H20 "
            "apply_preview step 5; N-line taxonomy starts at E0)"),
        "model": "qwen3-4b (N-line sealed sources; registration of prior work only)",
        "prereg_id": "N-line backfill draft (phase3208 g5g7, seal f29da2de)",
        "backfill": {
            "applied_by_phase": 3258,
            "applied_date": TODAY,
            "draft_file": "tests/glm5/result/rdc_query_construction_20260913/"
                          "phase3208/g5g7_nline_backfill_draft/"
                          "n_line_backfill_draft_v1.json",
            "draft_sha8": draft_sha_before,
            "ledger_post_before": DESIGN["ledger_prev_post"],
            "governance": ("APPX-4/5 同型：只补不改；draft file byte-unchanged "
                           "(f29da2de asserted before==after); readiness "
                           "ready_append_type per 3218; user continuation "
                           "authorization announced at 3257 tail"),
        },
        "superseded_by": None,
        "detail": (
            "Cross-line backfill of N-line phases P3-P7 into the shared atlas "
            "ledger per the sealed draft f29da2de (target: measurements append 5). "
            "Draft entry fields verbatim (name/phase/sources/status/title); this "
            "record adds registration provenance only. No G-line evidence claims "
            "are made; string phase 'N-P*' is the draft's registered namespace "
            "(G-line entries use int phases; the ledger is historically "
            "mixed-schema per lcorr C6 observation). Sources are sealed "
            "N-line reports/memo appends/design seals listed in sources[]."),
    }
    entries.append(ent)
assert [e["phase"] for e in entries] == DESIGN["entry_phases"]
assert [e["name"] for e in entries] == DESIGN["entry_names"]

# ---------------------------------------------------------------- arm (b): lcorr staleness
lcorr = json.load(open(LCORR, encoding="utf-8"))
assert lcorr["status"] == "SEALED_BUT_NOT_APPLIED"
stale = {}
stale["target_sha_before_mismatch"] = (
    lcorr["target_sha256_before"] != ledger_file_sha_before)
op1, op2, op3, op4 = lcorr["operations"]
stale["op1_from_sha_stale"] = op1["from"] != led["ledger_sha256_8"]
stale["op2_n_total_stale"] = op2["detail"]["n_total"] != len(led["measurements"])
mid = sum(1 for e in led["measurements"] if "meas_id" in e)
stale["op2_meas_id_count_drift_INFO"] = (mid != op2["detail"]["n_has_meas_id"])
dist = {}
for e in led["measurements"]:
    ev = str(e.get("evidence_level", "")).split(" ")[0]
    dist[ev] = dist.get(ev, 0) + 1
stale["op3_distribution_stale"] = dist != op3["current_distribution"]
stale["op3_enum_superseded"] = not set(
    dist.keys()).issubset(set(op3["allowed"]))
agate_present = any(
    str(e.get("phase")) == "A-gate"
    or "a_gate" in str(e.get("name", "")).lower()
    for e in led["measurements"])
stale["op4_agate_absent_not_redundant"] = not agate_present
lcorr_stale_confirmed = all(v for k, v in stale.items() if not k.endswith("_INFO"))
print("lcorr staleness:", json.dumps(stale))
print("lcorr evidence_level distribution now:", dist)

# ---------------------------------------------------------------- apply (in-memory)
new_led = copy.deepcopy(led)
new_led["measurements"].extend(entries)
# chain recompute BEFORE diff (both modes): field=prev -> sha8 -> field=new_post
new_led["ledger_sha256_8"] = DESIGN["ledger_prev_post"]
new_post = hashlib.sha256(
    json.dumps(new_led, ensure_ascii=False, indent=1).encode("utf-8")).hexdigest()[:8]
new_led["ledger_sha256_8"] = new_post
# planned pointer diff
def ptr_diff(a, b, p="$"):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append("%s.%s (KEY)" % (p, k))
            else:
                out.extend(ptr_diff(a[k], b[k], "%s.%s" % (p, k)))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s (LEN %d->%d)" % (p, len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(ptr_diff(x, y, "%s[%d]" % (p, i)))
    elif a != b:
        out.append("%s (VAL)" % p)
    return out


def norm_ptr(s):
    return s.split(" (")[0]


diffs = set(norm_ptr(x) for x in ptr_diff(led, new_led))
planned = set(DESIGN["planned_ledger_ptrs"])
g2_append_only = diffs == planned
# revert bitwise
rev = copy.deepcopy(new_led)
rev["measurements"] = rev["measurements"][:-5]
rev["ledger_sha256_8"] = DESIGN["ledger_prev_post"]
g3_revert_bitwise = json.dumps(rev, ensure_ascii=False, indent=1).encode("utf-8") == ledger_raw
print("g2 append-only:", g2_append_only, sorted(diffs))
print("g3 revert bitwise:", g3_revert_bitwise)

# ---------------------------------------------------------------- gates
gates = {}
gates["g1_draft_seal_intact"] = (s8(DRAFT) == DESIGN["expected_draft_sha8"])
gates["g2_append_only"] = g2_append_only
gates["g3_revert_bitwise"] = g3_revert_bitwise
gates["g5_lcorr_stale_confirmed"] = lcorr_stale_confirmed
gates["g6_lcorr_untouched"] = (s8(LCORR) == lcorr_sha_before)

new_post_written = None
if not SMOKE:
    shutil.copy2(LED, LED + ".bak_3258")
    with open(LED, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(new_led, ensure_ascii=False, indent=1))
    new_post_written = new_post
    # post-write self-check
    chk = json.load(open(LED, encoding="utf-8"))
    assert len(chk["measurements"]) == DESIGN["ledger_n_after"]
    assert chk["ledger_sha256_8"] == new_post
    chk["ledger_sha256_8"] = DESIGN["ledger_prev_post"]
    rec = hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1)
                         .encode("utf-8")).hexdigest()[:8]
    gates["g4_chain_recompute"] = (rec == new_post)
    gates["g1_draft_seal_intact"] = gates["g1_draft_seal_intact"] and (
        s8(DRAFT) == DESIGN["expected_draft_sha8"])
    gates["g6_lcorr_untouched"] = gates["g6_lcorr_untouched"] and (
        s8(LCORR) == lcorr_sha_before)
    print("applied: n=%d post=%s (pre file sha %s)"
          % (DESIGN["ledger_n_after"], new_post, ledger_file_sha_before[:8]))
else:
    gates["g4_chain_recompute"] = None  # deferred to formal
    print("SMOKE: no ledger write (in-memory post=%s)" % new_post)

gate_all = all(v for k, v in gates.items() if v is not None)

# ---------------------------------------------------------------- result
res = {
    "phase": 3258,
    "name": "g5h21_nline_apply",
    "line": "G",
    "date": TODAY,
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": ("nline_apply_registered" if gate_all else "BLOCKED"),
    "arm_a": {
        "draft": {"file": DRAFT, "sha8": draft_sha_before,
                  "status": "draft_pending_cross_line", "entries": 5},
        "applied_entry_phases": DESIGN["entry_phases"],
        "ledger": {"file": "research/gpt5/atlas/atlas_ledger.json",
                   "n_before": DESIGN["ledger_n_before"],
                   "n_after": DESIGN["ledger_n_after"],
                   "post_before": DESIGN["ledger_prev_post"],
                   "post_after": new_post,
                   "backup": LED + ".bak_3258" if not SMOKE else None,
                   "pre_file_sha8": ledger_file_sha_before[:8]},
        "authorization": ("readiness ready_append_type (3218) + sealed draft "
                          "target '#measurements (append 5)' + user continuation "
                          "authorization at 3257 tail"),
        "evidence_policy": ("entries are descriptive backfill registrations; no "
                            "G-line evidence level claimed; N-line levels start "
                            "at E0 (apply_preview step 5)"),
    },
    "arm_b": {
        "lcorr": {"file": LCORR, "sha8": lcorr_sha_before,
                  "status": "SEALED_BUT_NOT_APPLIED (unchanged)"},
        "staleness_checks": stale,
        "evidence_level_distribution_now": dist,
        "disposition": ("STALE_AS_SEALED — all 4 operations computed against a "
                        "n=304 ledger state (2026-10-03); target_sha256_before "
                        "mismatches current file; op3 enum does not cover current "
                        "evidence_level practice; op4 (A-gate registration) is "
                        "absent from the ledger and would need re-derivation, not "
                        "verbatim application. Not applied; deepseek files "
                        "untouched; backlog item updated to 'lcorr re-derivation "
                        "against current state'"),
        "writes": 0,
    },
    "gpu_outstanding": ["checkpoint training-step trajectory (u* emergence)",
                        "appendix GPU items x5 (APPX-1..3 re-measure; APPX-5 "
                        "re-baseline; APPX-4 agate re-derive)"],
}
if not SMOKE:
    res["res_sha8"] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != "res_sha8"},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "result_smoke.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
print("verdict", res["verdict"])
for k, v in gates.items():
    print(" ", k, v)
print("SMOKE DONE" if SMOKE else "FORMAL DONE")
