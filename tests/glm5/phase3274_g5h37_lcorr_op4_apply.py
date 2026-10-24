# -*- coding: utf-8 -*-
"""Phase 3274 (G5-H37): lcorr op4 apply (owner-approved) + APPX-4/5 resolution
landing.

Owner adjudication (2026-10-21, interactive selection): 批准 lcorr op4 施加.

Arm (a): apply op4 of the sealed deepseek cross-line patch
`ledger_corrections_v1.json` (beb775bd, op `append_measurement`, re-derived
READY in lcorr_rederivation_v2.json ef72c733 at 3260) to the shared atlas
ledger research/gpt5/atlas/atlas_ledger.json — append ONE A-gate governance
registration entry (payload fields verbatim + registration provenance block).
Append-only protocol per 3258 precedent: chain recompute at apply time (the
dry-run post d165ee30 recorded at 3272 expired with appends #424/#425 and is
NOT reused), revert-proof, deepseek sealed files byte-untouched.

Arm (b): APPX-4/5 auto-clear landing (linked per PENDING_ITEMS_IMPACT_PREVIEW
section ⑤): gap ledger v1.13 -> v1.14 appendix resolution note (APPX-4
STALE->RESOLVED: N-P3..P7 in ledger since 3258; APPX-5 ->RESOLVED: op4 applied
this phase) + registry v1.36 -> v1.37 upgrade_log annotation. Append-only;
v1.13 (814ce808) and v1.36 (78d37211) bytes untouched; REVERT bitwise.
"""
import copy
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
LCORR_V1 = os.path.join(ROOT, "research", "deepseek", "atlas",
                        "ledger_corrections_v1.json")
LCORR_V2 = os.path.join(RBASE, "phase3260", "g5h23_lcorr_rederivation",
                        "lcorr_rederivation_v2.json")
CLOSURE = os.path.join(ROOT, "research", "deepseek", "atlas",
                       "a_gate_closure_v1.json")
G13 = os.path.join(RBASE, "phase3265", "g5h28_appx7_landing",
                   "gap_ledger_v1_13.json")
R36 = os.path.join(RBASE, "phase3265", "g5h28_appx7_landing",
                   "atlas_registry_v1_36.json")
OUT = os.path.join(RBASE, "phase3274", "g5h37_lcorr_op4_apply")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3274_SMOKE") == "1"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3274,
    "name": "g5h37_lcorr_op4_apply",
    "prereg_id": "G5-H37 (owner adjudication 2026-10-21: 批准 lcorr op4 施加)",
    "arms": {
        "a": ("apply lcorr op4 (A-gate governance registration, payload verbatim "
              "from sealed patch beb775bd op append_measurement, re-verified vs "
              "closure record 24c60160) to shared atlas ledger, append-only, "
              "chain recompute at apply time, revert-proof"),
        "b": ("APPX-4/5 auto-clear landing: gap v1.13->v1.14 appendix resolution "
              "note + registry v1.36->v1.37 upgrade_log; append-only, "
              "predecessor bytes untouched, REVERT bitwise"),
    },
    "expected_sha8": {
        "lcorr_v1": "beb775bd",
        "lcorr_v2": "ef72c733",
        "closure": "24c60160",
        "gap_v1_13": "814ce808",
        "registry_v1_36": "78d37211",
    },
    "payload_fields": ["phase", "name", "verdict", "evidence_level", "note"],
    "payload": {
        "phase": "A-gate",
        "name": "rdc_a_gate_closure_q01_q02_q08_q09_q12",
        "verdict": "k1_fired_at_readout__kspecific__k2_k3_not_measurable__A_gate_closed",
        "evidence_level": "bit_anchored",
        "note": "Q08=甲 seal 后登记；属治理记录非科学测量",
    },
    "ledger": {"n_before": 430, "post": "20ecd6c2", "chain_prev": "3f210467",
               "n_after": 431},
    "planned_ledger_ptrs": ["$.measurements", "$.ledger_sha256_8"],
    "gap": {"version": "1.14", "appendix_n_after": 8},
    "registry": {"version": "1.37", "upgrade_log_n_after": 16,
                 "no_updated_key": True},
    "sealed_untouched": ["beb775bd (lcorr v1)", "ef72c733 (lcorr v2)",
                         "814ce808 (gap v1.13)", "78d37211 (registry v1.36)"],
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
led_raw = open(LED, "rb").read()
ledger_file_sha_before = hashlib.sha256(led_raw).hexdigest()
led = json.loads(led_raw.decode("utf-8"))
ms = led["measurements"]
gates = {}

# g1: ledger state + chain recompute (field=chain_prev -> head post)
chk = copy.deepcopy(led)
chk["ledger_sha256_8"] = DESIGN["ledger"]["chain_prev"]
rec = hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates["g1_ledger_state"] = (
    len(ms) == DESIGN["ledger"]["n_before"]
    and led["ledger_sha256_8"] == DESIGN["ledger"]["post"]
    and rec == DESIGN["ledger"]["post"]
    and json.dumps(led, ensure_ascii=False, indent=1).encode("utf-8") == led_raw)
print("g1 ledger: n=%d post=%s recompute=%s roundtrip=%s"
      % (len(ms), led["ledger_sha256_8"], rec,
         json.dumps(led, ensure_ascii=False, indent=1).encode("utf-8") == led_raw))

# g2: inputs on disk + statuses
lcorr_v1 = json.load(open(LCORR_V1, encoding="utf-8"))
lcorr_v2 = json.load(open(LCORR_V2, encoding="utf-8"))
closure = json.load(open(CLOSURE, encoding="utf-8"))
gates["g2_inputs"] = (
    s8(LCORR_V1) == DESIGN["expected_sha8"]["lcorr_v1"]
    and s8(LCORR_V2) == DESIGN["expected_sha8"]["lcorr_v2"]
    and s8(CLOSURE) == DESIGN["expected_sha8"]["closure"]
    and s8(G13) == DESIGN["expected_sha8"]["gap_v1_13"]
    and s8(R36) == DESIGN["expected_sha8"]["registry_v1_36"]
    and lcorr_v1["status"] == "SEALED_BUT_NOT_APPLIED"
    and lcorr_v2["status"] == "REPORT_ONLY_DRAFT"
    and closure["gate_closed"] is True)
print("g2 inputs: v1=%s v2=%s closure=%s gap=%s registry=%s"
      % (s8(LCORR_V1), s8(LCORR_V2), s8(CLOSURE), s8(G13), s8(R36)))

# g3: payload verbatim + closure-record cross-check + non-redundancy
op4_v1 = dict([o for o in lcorr_v1["operations"]
               if o["op"] == "append_measurement"][0]["entry"])
op4_v2 = dict(lcorr_v2["operations"]["op4_append_agate"]["rederived_entry"])
agate_present = any(
    str(e.get("phase")) == "A-gate"
    or "a_gate" in str(e.get("name", "")).lower() for e in ms)
gates["g3_payload_verbatim"] = (
    op4_v1 == DESIGN["payload"] and op4_v2 == DESIGN["payload"]
    and "Q08=甲" in closure["seal_verbatim"]
    and not agate_present)
print("g3 payload verbatim (v1==v2==payload):", op4_v1 == DESIGN["payload"],
      op4_v2 == DESIGN["payload"], "| agate_present:", agate_present)

# ---------------------------------------------------------------- Arm a: apply (in-memory)
TODAY = today()
ent = dict(DESIGN["payload"])
ent["lcorr_apply"] = {
    "applied_by_phase": 3274,
    "applied_date": TODAY,
    "source_draft": ("research/deepseek/atlas/ledger_corrections_v1.json "
                     "(beb775bd, op append_measurement; SEALED_BUT_NOT_APPLIED "
                     "byte-untouched)"),
    "rederivation": ("lcorr_rederivation_v2.json (ef72c733, Phase 3260 "
                     "op4_append_agate READY_PENDING_CONFIRMATION)"),
    "closure_record": ("research/deepseek/atlas/a_gate_closure_v1.json "
                       "(24c60160, seal_verbatim 'Q08=甲')"),
    "ledger_post_before": DESIGN["ledger"]["post"],
    "authorization": ("owner adjudication 2026-10-21 (interactive selection): "
                      "批准 lcorr op4 施加"),
    "governance": ("append-only registration per 3258 precedent; payload fields "
                   "verbatim; provenance block adds registration metadata only; "
                   "deepseek sealed files byte-untouched; revert-proof"),
}
new_led = copy.deepcopy(led)
new_led["measurements"].extend([ent])
# chain recompute BEFORE diff: field=prev -> sha8 -> field=new_post
new_led["ledger_sha256_8"] = DESIGN["ledger"]["chain_prev"]
new_post = hashlib.sha256(json.dumps(new_led, ensure_ascii=False, indent=1)
                          .encode("utf-8")).hexdigest()[:8]
new_led["ledger_sha256_8"] = new_post
print("in-memory op4 post:", new_post, "(n=%d)" % len(new_led["measurements"]))


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
gates["g5_append_only"] = diffs == set(DESIGN["planned_ledger_ptrs"])
print("g5 append-only ptrs:", sorted(diffs))
# revert bitwise
rev = copy.deepcopy(new_led)
rev["measurements"] = rev["measurements"][:-1]
rev["ledger_sha256_8"] = DESIGN["ledger"]["post"]
gates["g6_revert_bitwise"] = (
    json.dumps(rev, ensure_ascii=False, indent=1).encode("utf-8") == led_raw)
print("g6 revert bitwise:", gates["g6_revert_bitwise"])

# ---------------------------------------------------------------- Arm b: gap v1.14 + registry v1.37
g13d = json.load(open(G13, encoding="utf-8"))
r36d = json.load(open(R36, encoding="utf-8"))
appx45_entry = {
    "id": "APPX-4/5",
    "title": "跨线治理结清注记（APPX-4 STALE→RESOLVED；APPX-5→RESOLVED via op4）",
    "anchor": "p3258+p3274",
    "status": "appendix_recorded",
    "note": ("APPX-4（N 线跨线补丁 re-derive）：N-P3..N-P7 五条自 3258 起实测在账"
             "（shared ledger #410 post a59da52b），v1.13 沿用的 status=open_crossline "
             "系滞后注记，现定谳 RESOLVED（STALE_STATUS→已结清，N 线自 E0 起算口径不变）。"
             "APPX-5（lcorr ledger_corrections_v1.json beb775bd 跨线补丁）：四操作于 3260 "
             "全处置（op1 作废/op2 不迁移/op3 被实践超越/op4 READY），唯一活口 op4（A-gate "
             "治理登记，closure 24c60160）经 owner 裁决（2026-10-21 批准）于 3274 施加至 "
             "shared ledger（post %s，n 430→431），payload 逐字、revert-proof、封存文件"
             "字节不动。v1.13 字节不动（814ce808）。" % new_post),
    "anchor_sha8": design_sha8,
}
g14 = json.loads(json.dumps(g13d))
g14["version"] = "1.14"
g14["appendix"] = g13d["appendix"] + [appx45_entry]
g14["supersedes"] = "gap_ledger_v1_13.json (%s, Phase 3265)" % DESIGN["expected_sha8"]["gap_v1_13"]
g14["updated"] = TODAY

r37 = json.loads(json.dumps(r36d))
r37["version"] = "1.37"
r37["upgrade_log"] = r36d["upgrade_log"] + [{
    "node": "gap ledger appendix",
    "change": ("APPX-4/5 cross-line governance resolution (gap v1.13 -> v1.14): "
               "APPX-4 STALE->RESOLVED (N-P3..P7 in shared ledger since 3258, "
               "post a59da52b); APPX-5 RESOLVED (lcorr beb775bd op4 A-gate "
               "governance registration applied at 3274, ledger post %s, n "
               "430->431, owner-approved 2026-10-21; payload verbatim, "
               "revert-proof, sealed deepseek files byte-untouched). Anchors "
               "unchanged; no feature/failure changes." % new_post),
    "phase": 3274,
    "gap_ledger": "gap_ledger_v1_14.json",
}]
r37["supersedes"] = "atlas_registry_v1_36.json (%s, Phase 3265)" % DESIGN["expected_sha8"]["registry_v1_36"]
# NOTE: registry chain has NO 'updated' key (3256 SMOKE lesson) — do not add.

# REVERT rebuilds (byte-identical to sealed predecessors)
rev_g = json.loads(json.dumps(g14))
rev_g["version"] = "1.13"
rev_g["appendix"] = g13d["appendix"]
rev_g["supersedes"] = g13d["supersedes"]
rev_g["updated"] = g13d["updated"]
rev_r = json.loads(json.dumps(r37))
rev_r["version"] = "1.36"
rev_r["upgrade_log"] = r36d["upgrade_log"]
rev_r["supersedes"] = r36d["supersedes"]
gates["g7_revert_gap_registry"] = (
    hashlib.sha256(json.dumps(rev_g, ensure_ascii=False, indent=1,
                              sort_keys=False).encode("utf-8")).hexdigest()[:8]
    == DESIGN["expected_sha8"]["gap_v1_13"]
    and hashlib.sha256(json.dumps(rev_r, ensure_ascii=False, indent=1,
                                  sort_keys=False).encode("utf-8")).hexdigest()[:8]
    == DESIGN["expected_sha8"]["registry_v1_36"])
print("g7 revert gap/registry:", gates["g7_revert_gap_registry"])

# pointer-diff audit: gaps/features/failures/anchors untouched
gates["g8_pointer_diff"] = (
    g14["gaps"] == g13d["gaps"]
    and r37["features"] == r36d["features"]
    and r37["failures"] == r36d["failures"]
    and r37["provenance"]["anchor_files"] == r36d["provenance"]["anchor_files"]
    and r37["census_3238"] == r36d["census_3238"]
    and r37["evidence_level_taxonomy"] == r36d["evidence_level_taxonomy"])
print("g8 pointer diff:", gates["g8_pointer_diff"])

# ---------------------------------------------------------------- write
zero_targets = [LED, LCORR_V1, LCORR_V2, CLOSURE, G13, R36]
zero_before = {p: s8(p) for p in zero_targets}
if not SMOKE:
    shutil.copy2(LED, LED + ".bak_3274")
    with open(LED, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(new_led, ensure_ascii=False, indent=1))
    # post-write self-check
    chk2 = json.load(open(LED, encoding="utf-8"))
    assert len(chk2["measurements"]) == DESIGN["ledger"]["n_after"]
    assert chk2["ledger_sha256_8"] == new_post
    chk2["ledger_sha256_8"] = DESIGN["ledger"]["chain_prev"]
    rec2 = hashlib.sha256(json.dumps(chk2, ensure_ascii=False, indent=1)
                          .encode("utf-8")).hexdigest()[:8]
    gates["g4_chain_recompute"] = (rec2 == new_post)
    with open(os.path.join(OUT, "gap_ledger_v1_14.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(g14, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "atlas_registry_v1_37.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(r37, f, ensure_ascii=False, indent=1)
    g14d = json.load(open(os.path.join(OUT, "gap_ledger_v1_14.json"),
                          encoding="utf-8"))
    r37d = json.load(open(os.path.join(OUT, "atlas_registry_v1_37.json"),
                          encoding="utf-8"))
    gates["g9_landed_files"] = (
        g14d["version"] == "1.14" and len(g14d["appendix"]) == 8
        and g14d["appendix"][-1]["id"] == "APPX-4/5"
        and r37d["version"] == "1.37" and len(r37d["upgrade_log"]) == 16
        and "updated" not in r37d)
    print("applied: ledger n=%d post=%s (pre file sha %s)"
          % (DESIGN["ledger"]["n_after"], new_post, ledger_file_sha_before[:8]))
else:
    gates["g4_chain_recompute"] = None   # deferred to formal
    gates["g9_landed_files"] = None      # deferred to formal
    with open(os.path.join(SMOKE_DIR, "apply_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"op4_post": new_post, "n_after": len(new_led["measurements"]),
                   "gates": {k: v for k, v in gates.items() if v is not None}},
                  f, ensure_ascii=False, indent=1)
    print("SMOKE: no writes (in-memory op4 post=%s)" % new_post)

# ---------------------------------------------------------------- zero/sealed guard
zero_after = {p: s8(p) for p in zero_targets}
sealed_ok = all(zero_after[p] == zero_before[p]
                for p in zero_targets if p != LED)
if not SMOKE:
    # LED changed by design (op4 apply); verify new content matches in-memory
    sealed_ok = sealed_ok and (
        open(LED, "rb").read() == json.dumps(
            new_led, ensure_ascii=False, indent=1).encode("utf-8"))
gates["g10_sealed_untouched"] = sealed_ok
print("g10 sealed untouched (lcorr/closure/gap13/reg36 byte-intact):", sealed_ok)

# ---------------------------------------------------------------- verdict
gate_all = all(v for k, v in gates.items() if v is not None)
res = {
    "phase": 3274,
    "name": "g5h37_lcorr_op4_apply",
    "line": "G",
    "date": TODAY,
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "lcorr_op4_applied" if gate_all else "BLOCKED",
    "arm_a": {
        "payload": DESIGN["payload"],
        "payload_fields_verbatim": True,
        "ledger": {"file": "research/gpt5/atlas/atlas_ledger.json",
                   "n_before": DESIGN["ledger"]["n_before"],
                   "n_after": DESIGN["ledger"]["n_after"],
                   "post_before": DESIGN["ledger"]["post"],
                   "post_after": new_post,
                   "backup": LED + ".bak_3274" if not SMOKE else None,
                   "pre_file_sha8": ledger_file_sha_before[:8]},
        "dry_run_post_3272": "d165ee30 (expired, not reused)",
        "authorization": "owner adjudication 2026-10-21: 批准 lcorr op4 施加",
        "sealed_untouched": {"lcorr_v1": "beb775bd", "lcorr_v2": "ef72c733",
                             "closure": "24c60160"},
    },
    "arm_b": {
        "gap_v1_14": {"file": "gap_ledger_v1_14.json",
                      "appendix_n": 8, "entry_id": "APPX-4/5",
                      "supersedes": g14["supersedes"],
                      "file_sha8": (s8(os.path.join(OUT, "gap_ledger_v1_14.json"))
                                    if not SMOKE else "smoke")},
        "registry_v1_37": {"file": "atlas_registry_v1_37.json",
                           "upgrade_log_n": 16, "anchors_unchanged": True,
                           "supersedes": r37["supersedes"],
                           "file_sha8": (s8(os.path.join(
                               OUT, "atlas_registry_v1_37.json"))
                               if not SMOKE else "smoke")},
        "sealed_untouched": {"gap_v1_13": "814ce808",
                             "registry_v1_36": "78d37211"},
    },
    "revert_check": {"ledger": gates["g6_revert_bitwise"],
                     "gap_registry": gates["g7_revert_gap_registry"]},
    "pending_items_after": [
        "水果类/K4 MEMO 挂账删除（待 owner）",
        "APPX-1 量化口径三选一（待 owner）",
        "APPX-2/3 执行相（协议 ce73984d 已备，待 owner 点名）",
        "lcorr op2/op3 以 v2 稿结论归档，不再执行（3260 定谳）",
    ],
}
if not SMOKE:
    res["res_sha8"] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != "res_sha8"},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8",
              newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
else:
    with open(os.path.join(SMOKE_DIR, "result_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
print("verdict", res["verdict"])
for k, v in gates.items():
    print(" ", k, v)
print("SMOKE DONE" if SMOKE else "FORMAL DONE")
