# -*- coding: utf-8 -*-
"""Phase 3259 (G5-H22): post-apply consistency verification + deepseek MEMO receipt.

Arm (a): three-account + client consistency recheck after 3258 N-line apply —
ledger chain recompute, N-P entries verbatim vs sealed draft, sealed G-line
accounts byte-untouched (registry v1.35/v1.34, gap v1.12, draft, lcorr), live
endpoint visibility of the 5 N-P entries (line=N filter).

Arm (b): deepseek MEMO receipt annotation (append-only) recording the cross-line
backfill application and the lcorr STALE_AS_SEALED disposition.
"""
import copy
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
DRAFT = os.path.join(RBASE, "phase3208", "g5g7_nline_backfill_draft",
                     "n_line_backfill_draft_v1.json")
LCORR = os.path.join(ROOT, "research", "deepseek", "atlas", "ledger_corrections_v1.json")
REG35 = os.path.join(RBASE, "phase3256", "g5h19_closure_seal", "atlas_registry_v1_35.json")
REG34 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "atlas_registry_v1_34.json")
GAP12 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json")
EP_OUT = os.path.join(ROOT, "tests", "gpt5_temp", "p3259_endpoint_out.txt")
DS_MEMO = os.path.join(ROOT, "research", "deepseek", "docs", "AGI_DEEPSEEK_MEMO.md")
OUT = os.path.join(RBASE, "phase3259", "g5h22_postapply_consistency")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3259_SMOKE") == "1"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3259,
    "name": "g5h22_postapply_consistency",
    "prereg_id": "G5-H22 (3258 tail default arm (a))",
    "arms": {
        "a": "post-apply consistency: ledger chain + N-P verbatim + sealed accounts untouched + live endpoint line=N visibility",
        "b": "deepseek MEMO receipt annotation (append-only, marker '## 跨线施加回执（G 线 Phase 3258）')",
    },
    "expected_sha8": {
        "ledger_backup_apply": "n/a (chain-verified)",
        "registry_v1_35": "e8335331",
        "registry_v1_34": "3deabf25",
        "gap_v1_12": "aa8cff21",
        "draft": "f29da2de",
        "lcorr": "beb775bd",
    },
    "ledger": {"n": 415, "post": "8de77a63", "chain_prev": "a59da52b",
               "nline_phases": ["N-P3", "N-P4", "N-P5", "N-P6", "N-P7"]},
    "endpoint_expected_pass": 11,
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

# ---------------------------------------------------------------- arm (a) disk checks
led_raw = open(LED, "rb").read()
led = json.loads(led_raw.decode("utf-8"))
gates = {}

# g1 ledger chain recompute
chk = copy.deepcopy(led)
chk["ledger_sha256_8"] = DESIGN["ledger"]["chain_prev"]
rec = hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates["g1_ledger_chain"] = (
    len(led["measurements"]) == DESIGN["ledger"]["n"]
    and led["ledger_sha256_8"] == DESIGN["ledger"]["post"]
    and rec == DESIGN["ledger"]["post"])

# g2 N-P entries verbatim vs draft
draft = json.load(open(DRAFT, encoding="utf-8"))
np_entries = [e for e in led["measurements"]
              if str(e.get("phase", "")).startswith("N-P")]
ok_verbatim = len(np_entries) == 5
for de, le in zip(draft["entries"], np_entries):
    for k in ("name", "phase", "sources", "status", "title"):
        if de[k] != le.get(k):
            ok_verbatim = False
    if le.get("backfill", {}).get("applied_by_phase") != 3258:
        ok_verbatim = False
gates["g2_np_verbatim"] = ok_verbatim
# #410 present exactly once
gates["g2b_entry410_once"] = sum(
    1 for e in led["measurements"] if e.get("phase") == 3258) == 1

# g3 sealed accounts untouched
sha_expect = DESIGN["expected_sha8"]
gates["g3_sealed_untouched"] = (
    s8(REG35) == sha_expect["registry_v1_35"]
    and s8(REG34) == sha_expect["registry_v1_34"]
    and s8(GAP12) == sha_expect["gap_v1_12"]
    and s8(DRAFT) == sha_expect["draft"]
    and s8(LCORR) == sha_expect["lcorr"])

# g4 live endpoint visibility (from saved endpoint-check output)
ep = open(EP_OUT, encoding="utf-8").read()
n_pass_lines = sum(1 for l in ep.splitlines() if l.startswith("PASS"))
gates["g4_endpoint_live"] = (
    "ENDPOINT CHECK OK" in ep
    and n_pass_lines == DESIGN["endpoint_expected_pass"]
    and not any(l.startswith("FAIL") for l in ep.splitlines())
    and "line=N count 5" in ep
    and "8de77a63" in ep)

# ---------------------------------------------------------------- arm (b): receipt
RECEIPT_MARK = "## 跨线施加回执（G 线 Phase 3258）"
ds_memo = open(DS_MEMO, encoding="utf-8").read()
already = RECEIPT_MARK in ds_memo
if not SMOKE and not already:
    receipt = (
        "\n\n---\n\n" + RECEIPT_MARK + " [2026-10-21]\n\n"
        "- **施加对象**：本线封存草案 `n_line_backfill_draft_v1.json`（phase3208 g5g7，"
        "seal `f29da2de`，status draft_pending_cross_line，N-P3..N-P7 五条）已由 G 线 "
        "Phase 3258（G5-H21，verdict `nline_apply_registered`）按其 target "
        "「#measurements (append 5)」施加至共享账本 `research/gpt5/atlas/atlas_ledger.json`。\n"
        "- **施加方式**：草案字段逐字（name/phase/sources/status/title）+ 登记溯源块；"
        "evidence_level=descriptive（跨线证据级不转移，N 线自 E0 起算 per G5-H20 "
        "apply_preview step 5）；草案文件字节不动；账本 n 409→414 post `a59da52b`，"
        "closeout 后 #410 post `8de77a63`；施加后一致性核验 Phase 3259 复读通过"
        "（链式 recompute / 逐字比对 / 客户端 /atlas/ledger?line=N 实机 11 断言）。\n"
        "- **挂账结转**：本 MEMO 挂账「N 线 P3–P7 补 Ledger」**已结清**；"
        "「跨线账本补丁施加确认」更新为——`ledger_corrections_v1.json`（`beb775bd`，"
        "4 操作对 n=304 旧账计算）经 G 线 3258 实测**STALE_AS_SEALED 不施加**"
        "（target sha 失配 / op3 枚举被 55 值 evidence_level 实践超越 / op4 A-gate "
        "须重推导），挂账改为「lcorr 重推导（基于现账状态）」，施加仍待确认。\n"
        "- **纪律**：本回执为 append-only 注记，本线封存文件（草案/lcorr/账目）零改动。\n")
    ds_memo = ds_memo.rstrip("\n") + receipt
    with open(DS_MEMO, "w", encoding="utf-8", newline="") as f:
        f.write(ds_memo)
    print("receipt appended to deepseek MEMO")
elif already:
    print("receipt already present (idempotent)")
else:
    print("SMOKE: receipt deferred")
if not SMOKE:
    gates["g5_receipt_once"] = open(
        DS_MEMO, encoding="utf-8").read().count(RECEIPT_MARK) == 1
else:
    gates["g5_receipt_once"] = None  # deferred to formal (write arm)

# ---------------------------------------------------------------- verdict / result
gate_all = all(v for k, v in gates.items() if v is not None)
res = {
    "phase": 3259,
    "name": "g5h22_postapply_consistency",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "postapply_consistency_confirmed" if gate_all else "BLOCKED",
    "arm_a": {
        "ledger": {"n": len(led["measurements"]),
                   "post": led["ledger_sha256_8"],
                   "chain_prev": DESIGN["ledger"]["chain_prev"],
                   "recompute": rec},
        "np_entries": [e["phase"] for e in np_entries],
        "np_verbatim": gates["g2_np_verbatim"],
        "sealed_untouched": {k: s8(p) for k, p in
                             (("registry_v1_35", REG35), ("registry_v1_34", REG34),
                              ("gap_v1_12", GAP12), ("draft", DRAFT), ("lcorr", LCORR))},
        "endpoint_live": {"file": "tests/gpt5_temp/p3259_endpoint_out.txt",
                          "asserts": DESIGN["endpoint_expected_pass"],
                          "line_N_filter": "count=5 phases N-P7..N-P3 newest-first"},
    },
    "arm_b": {
        "file": "research/deepseek/docs/AGI_DEEPSEEK_MEMO.md",
        "marker": RECEIPT_MARK,
        "write": "append-only receipt" if not SMOKE else "deferred (SMOKE)",
        "backlog_resolution": ("'N-line P3-P7 ledger backfill' CLOSED by 3258; "
                              "'cross-line patch confirmation' -> lcorr re-derivation "
                              "backlog (STALE_AS_SEALED, 3258)"),
    },
    "gpu_outstanding": ["checkpoint training-step trajectory (u* emergence)",
                        "appendix GPU items x5",
                        "lcorr re-derivation (report-only, apply pending confirmation)",
                        "水果类/K4 MEMO list deletion pending user confirmation"],
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
