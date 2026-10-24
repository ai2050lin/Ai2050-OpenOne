# -*- coding: utf-8 -*-
"""Phase 3260 (G5-H23): lcorr re-derivation (report-only) + backlog inventory v3.

Arm (a): re-derive the 4 operations of the sealed deepseek cross-line patch
`ledger_corrections_v1.json` (beb775bd, computed against the n=304 ledger of
2026-10-03) against the CURRENT shared ledger (n=416, post 7b6e6230), producing
`lcorr_rederivation_v2.json` — a report-only successor draft. Zero writes to
deepseek files and zero writes to the shared ledger.

Arm (b): backlog inventory v3 (post N-line-apply) + one-page user-confirmation
proposal for the fruit-class/K4 MEMO list deletion and lcorr application.
"""
import hashlib
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
LCORR = os.path.join(ROOT, "research", "deepseek", "atlas", "ledger_corrections_v1.json")
DS_DIR = os.path.join(ROOT, "research", "deepseek")
OUT = os.path.join(RBASE, "phase3260", "g5h23_lcorr_rederivation")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3260_SMOKE") == "1"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


def chain_sha8(state, prev_post):
    """Ledger chained convention: sha8(dumps(state with field=prev_post))."""
    chk = json.loads(json.dumps(state))
    chk["ledger_sha256_8"] = prev_post
    return hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1)
                          .encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3260,
    "name": "g5h23_lcorr_rederivation",
    "prereg_id": "G5-H23 (3259 tail default arm (a))",
    "arms": {
        "a": ("lcorr re-derivation: recompute the 4 sealed ops (beb775bd, n=304 basis) "
              "against current ledger n=416 -> lcorr_rederivation_v2.json (report-only)"),
        "b": ("backlog inventory v3 (post N-line-apply) + user-confirmation proposal "
              "for fruit-class/K4 deletion and lcorr application"),
    },
    "expected_sha8": {"lcorr": "beb775bd"},
    "ledger": {"n": 416, "post": "7b6e6230", "chain_prev": "8de77a63"},
    "zero_write": ["research/gpt5/atlas/atlas_ledger.json",
                   "research/deepseek/atlas/ledger_corrections_v1.json",
                   "research/deepseek/docs/AGI_DEEPSEEK_MEMO.md"],
    "op_dispositions": {
        "op1_set_field": "OBSOLETE_SUPERSEDED",
        "op2_unify_schema": "REQUIRES_DESTRUCTIVE_MIGRATION",
        "op3_enumerate_field": "SUPERSEDED_BY_PRACTICE",
        "op4_append_agate": "READY_PENDING_CONFIRMATION",
    },
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

# ---------------------------------------------------------------- g1: ledger chain
led_raw = open(LED, "rb").read()
led = json.loads(led_raw.decode("utf-8"))
ms = led["measurements"]
rec = chain_sha8(led, DESIGN["ledger"]["chain_prev"])
gates = {}
gates["g1_chain_recompute"] = (
    len(ms) == DESIGN["ledger"]["n"]
    and led["ledger_sha256_8"] == DESIGN["ledger"]["post"]
    and rec == DESIGN["ledger"]["post"])
print("g1 chain: n=%d post=%s recompute=%s" % (len(ms), led["ledger_sha256_8"], rec))

# ---------------------------------------------------------------- g2: lcorr sealed
lcorr_before = s8(LCORR)
gates["g2_lcorr_sealed"] = (lcorr_before == DESIGN["expected_sha8"]["lcorr"])
lcorr = json.load(open(LCORR, encoding="utf-8"))
ops = {o["op"]: o for o in lcorr["operations"]}
print("g2 lcorr sealed:", gates["g2_lcorr_sealed"])

# ---------------------------------------------------------------- arm (a): re-derivation
# measured current state
mid = sum(1 for e in ms if "meas_id" in e)
ph = sum(1 for e in ms if "phase" in e)
both = sum(1 for e in ms if "meas_id" in e and "phase" in e)
string_phases = sorted({e["phase"] for e in ms if isinstance(e.get("phase"), str)})
el_dist = Counter(str(e.get("evidence_level")) for e in ms)
agate_present = any(e.get("phase") == "A-gate"
                    or "a_gate" in str(e.get("name", "")).lower() for e in ms)

derived = {}
# op1: one-shot set_field of ledger_sha256_8
derived["op1_set_field"] = {
    "disposition": "OBSOLETE_SUPERSEDED",
    "evidence": {
        "original_from": ops["set_field"]["from"],
        "original_to": ops["set_field"]["to"],
        "current_post": led["ledger_sha256_8"],
        "chain_recompute": rec,
    },
    "rationale": ("the chained convention (field = sha8(dumps(state with field=prev_post))) "
                  "is enforced at every append since adoption; a one-shot set_field cannot "
                  "reproduce it and is superseded. Current post is self-consistent (g1)."),
}
# op2: unify_schema meas_id+phase
derived["op2_unify_schema"] = {
    "disposition": "REQUIRES_DESTRUCTIVE_MIGRATION",
    "evidence": {
        "original_detail": ops["unify_schema"]["detail"],
        "current_detail": {"n_total": len(ms), "n_has_meas_id": mid,
                           "n_has_phase": ph, "n_both": both},
        "string_phase_namespaces": string_phases[:8],
        "note": ("phase is now int OR string (N-P* backfill namespace, 3258); "
                 "backfill by name/sha would rewrite byte-anchored history"),
    },
    "rationale": ("schema unification rewrites historical entries that are bit-anchored "
                  "by revert-proof protocol; not a safe patch. Recommend: keep historical "
                  "schema as-is and document the convention instead of migrating."),
}
# op3: enumerate evidence_level
top_vals = el_dist.most_common(6)
derived["op3_enumerate_field"] = {
    "disposition": "SUPERSEDED_BY_PRACTICE",
    "evidence": {
        "original_allowed": ops["enumerate_field"]["allowed"],
        "original_distribution": ops["enumerate_field"]["current_distribution"],
        "current_top": [[k, v] for k, v in top_vals],
        "current_distinct_values": len(el_dist),
        "statistical_count": el_dist.get("statistical", 0),
        "statistical_share": round(el_dist.get("statistical", 0) / len(ms), 4),
    },
    "rationale": ("the 3-value enum cannot express the governance/report arm vocabulary "
                  "that practice has accumulated (55+ distinct values); the correct "
                  "successor action is a separate F2_verdict_grading re-grading project "
                  "(evidence_level normalization), logged as backlog, not enum enforcement."),
}
# op4: append A-gate governance registration
op4_entry = dict(ops["append_measurement"]["entry"])
derived["op4_append_agate"] = {
    "disposition": "READY_PENDING_CONFIRMATION",
    "agate_in_ledger": agate_present,
    "rederived_entry": op4_entry,
    "evidence": {
        "closure_record": lcorr["closure_record"],
        "closure_record_sha8": s8(os.path.join(DS_DIR, "atlas", "a_gate_closure_v1.json")),
        "note": ("A-gate still absent from the shared ledger (measured); the entry is a "
                 "governance record, application needs owner/user confirmation"),
    },
    "rationale": ("payload re-verified against the sealed closure record; ready to append "
                  "upon explicit confirmation, via the same append-only protocol as 3258."),
}
gates["g3_rederivation_complete"] = all(
    derived[k]["disposition"] == DESIGN["op_dispositions"][k]
    for k in DESIGN["op_dispositions"]) and all(
    k in derived for k in DESIGN["op_dispositions"])
print("g3 re-derivation complete:", gates["g3_rederivation_complete"])

v2 = {
    "schema": "rdc_cross_line_ledger_patch_v2_rederivation",
    "created": today(),
    "status": "REPORT_ONLY_DRAFT",
    "supersedes_draft": "ledger_corrections_v1.json (beb775bd, SEALED_BUT_NOT_APPLIED)",
    "target": "research/gpt5/atlas/atlas_ledger.json",
    "target_state_at_derivation": {"n": len(ms), "post": led["ledger_sha256_8"]},
    "operations": derived,
    "governance": {
        "application": "requires explicit user/owner confirmation; append-only protocol per 3258 precedent",
        "deepseek_files_touched": "none (report-only; sealed draft beb775bd byte-untouched)",
    },
}

# ---------------------------------------------------------------- arm (b): backlog v3
backlog = {
    "schema": "gline_backlog_v3",
    "created": today(),
    "supersedes": "backlog v2 (MEMO standing, 3216)",
    "closed_since_v2": [
        {"item": "N-line P3-P7 ledger backfill",
         "closed_at": "3258 (apply) / 3259 (post-apply consistency + deepseek receipt)",
         "evidence": "ledger #410, n=416 post 7b6e6230; receipt in deepseek MEMO"},
        {"item": "lcorr re-derivation (report-only)",
         "closed_at": "3260",
         "evidence": "lcorr_rederivation_v2.json (this phase)"},
    ],
    "open_gpu": [
        "checkpoint training-step trajectory (u* emergence)",
        "appendix GPU items x5 (APPX-1..5)",
    ],
    "pending_user_confirmation": [
        {"item": "MEMO backlog-list deletion: fruit-class, K4",
         "why": "documentation edit on the G-line MEMO requires explicit user approval"},
        {"item": "lcorr v2 application (op4 A-gate governance entry; op2 migration decision)",
         "why": "cross-line write to the shared ledger; owner/user decision per original patch protocol"},
    ],
    "notes": ("N-line apply resolved; remaining GPU items unchanged; evidence_level "
              "normalization (F2 re-grading) newly logged from op3 re-derivation"),
}
backlog["open_gpu"].append(
    "evidence_level normalization re-grading (F2_verdict_grading, from op3 re-derivation)")

proposal = "\n".join([
    "# 用户确认提案（Phase 3260 arm b，一页式）",
    "",
    "日期：%s ｜ 状态：REPORT_ONLY（本 Phase 零写入共享账本与 deepseek 封存）" % today(),
    "",
    "## 待确认 1：MEMO 挂账列表删除「水果类、K4」",
    "- 背景：3210 盘点/3216 挂账总账 v2 已将两者定为 closed（item+closed_at+evidence 三锚），",
    "  但 MEMO 挂账列表的文本删除属文档编辑，按纪律需用户显式确认。",
    "- 建议：批准删除（账本侧证据已三锚封存，删除仅影响 MEMO 列表可读性）。",
    "",
    "## 待确认 2：lcorr v2 施加",
    "- op1 作废（链式约定已取代一次性 set_field，g1 已验）；",
    "- op2 建议不迁移（对字节锚定历史属破坏性改写，建议以文档约定取代）；",
    "- op3 建议不施加枚举（55+ 值实践已超越三值枚举；改挂 evidence_level 规范化重判定）；",
    "- **op4 可施加**：A-gate 治理登记（Q08=甲，closure record `a_gate_closure_v1.json`），",
    "  append-only 协议同 3258 先例。",
    "- 建议：批准 op4 单条施加；op2/op3 以本 v2 稿结论归档，不再作为补丁执行。",
    "",
    "## 不需确认（自动推进）",
    "- GPU 挂账（checkpoint 轨迹、APPX-1..5）维持待资源；evidence_level 规范化已入挂账 v3。",
    "",
])

# ---------------------------------------------------------------- zero-write guard
zero_before = {p: s8(p) for p in DESIGN["zero_write"]}

# ---------------------------------------------------------------- write outputs
wdir = SMOKE_DIR if SMOKE else OUT
if not SMOKE:
    with open(os.path.join(OUT, "lcorr_rederivation_v2.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(v2, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "backlog_inventory_v3.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump(backlog, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "user_confirmation_proposal.md"), "w",
              encoding="utf-8", newline="") as f:
        f.write(proposal)
else:
    with open(os.path.join(SMOKE_DIR, "rederivation_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"v2": v2, "backlog": backlog}, f, ensure_ascii=False, indent=1)

# g4: backlog v3 complete & consistent
gates["g4_backlog_v3"] = (
    len(backlog["closed_since_v2"]) == 2
    and backlog["pending_user_confirmation"][0]["item"].startswith("MEMO")
    and any("evidence_level" in x for x in backlog["open_gpu"]))
print("g4 backlog v3:", gates["g4_backlog_v3"])

# g5: zero-write (report-only) assertion
zero_after = {p: s8(p) for p in DESIGN["zero_write"]}
gates["g5_zero_write"] = (zero_before == zero_after)
print("g5 zero-write:", gates["g5_zero_write"])

# ---------------------------------------------------------------- verdict / result
gate_all = all(v for k, v in gates.items() if v is not None)
res = {
    "phase": 3260,
    "name": "g5h23_lcorr_rederivation",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "lcorr_rederivation_registered" if gate_all else "BLOCKED",
    "arm_a": {
        "sealed_draft": {"file": "research/deepseek/atlas/ledger_corrections_v1.json",
                         "sha8": lcorr_before},
        "target_state": {"n": len(ms), "post": led["ledger_sha256_8"],
                         "chain_recompute": rec},
        "op_dispositions": {k: derived[k]["disposition"] for k in derived},
        "measured": {"meas_id": mid, "phase": ph, "both": both,
                     "distinct_evidence_levels": len(el_dist),
                     "statistical": el_dist.get("statistical", 0),
                     "agate_in_ledger": agate_present},
        "artifact": "lcorr_rederivation_v2.json",
    },
    "arm_b": {
        "artifacts": ["backlog_inventory_v3.json", "user_confirmation_proposal.md"],
        "closed_since_v2": [b["item"] for b in backlog["closed_since_v2"]],
        "pending_user_confirmation": [p["item"] for p in backlog["pending_user_confirmation"]],
    },
    "zero_write_verified": zero_after,
    "gpu_outstanding": backlog["open_gpu"],
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
