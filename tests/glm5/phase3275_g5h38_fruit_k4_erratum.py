# -*- coding: utf-8 -*-
"""Phase 3275 (G5-H38): fruit-class/K4 MEMO backlog erratum (owner-approved,
append-only).

Owner adjudication (2026-10-21, interactive selection): 批准水果类/K4 注记
(append-only erratum, no history rewrite; physical deletion requires a second
confirmation and was NOT requested).

Arm (a): census re-verification on the current baseline (6d0ed4fb = d3f62d35
+ 3274 receipt append): 水果类 x26 / K4 x26 across 35 unique lines; per-line
classification (science/protocol/standing-backlog) recomputed and asserted.

Arm (b): append ONE dated erratum section at MEMO end declaring the standing
backlog names 「水果类」「K4」 closed per the triple anchors (3210 G5-G9 audit
disposition + 3216 standing ledger v2 closed-x5) + owner approval; per-line
annotation table (52 occurrences / 35 lines); prefix bytes immutable
(byte-level append), backup .bak_3275, revert-proof.
"""
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result",
                     "rdc_query_construction_20260913")
MEMO = os.path.join(ROOT, "research", "deepseek", "docs",
                    "AGI_DEEPSEEK_MEMO.md")
GMEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(RBASE, "phase3275", "g5h38_fruit_k4_erratum")
SMOKE_DIR = os.path.join(OUT, "smoke")

SMOKE = os.environ.get("P3275_SMOKE") == "1"


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


# classified line numbers (measured 2026-10-21 on baseline 6d0ed4fb)
L_SCI = [1106, 1160, 1318, 3097, 6678, 7396, 7399]          # science/stats (fruit)
L_DEF = [311, 386, 432]                                      # protocol defs (K4)
L_BKG = [1478, 1486, 1509, 1561, 1766, 1769, 2003, 2005,     # standing backlog
         2279, 2530, 2834, 3124, 3406, 3657, 3824, 4077,
         4340, 4480, 4667, 4768, 4826, 7153, 7257, 7360, 7409]

# ---------------------------------------------------------------- DESIGN freeze
DESIGN = {
    "phase": 3275,
    "name": "g5h38_fruit_k4_erratum",
    "prereg_id": "G5-H38 (owner adjudication 2026-10-21: 批准水果类/K4 注记)",
    "arms": {
        "a": ("census re-verification on baseline 6d0ed4fb: 水果类 x26 / K4 x26 "
              "across 35 unique lines; classification recomputed & asserted"),
        "b": ("append-only erratum: one dated section at MEMO end declaring "
              "standing items 水果类/K4 closed per 3210 (e678dc42/3ae23744/"
              "23008571) + 3216 closed-x5 anchors + owner approval; per-line "
              "table; prefix bytes immutable; backup .bak_3275"),
    },
    "memo_baseline_sha8": "6d0ed4fb",
    "memo_provenance": ("d3f62d35 (3271 baseline) + 3274 receipt append "
                        "(owner-approved op4 回执) -> 6d0ed4fb"),
    "census_expected": {"fruit": 26, "k4": 26, "unique_lines": 35},
    "classification": {"science_retained": L_SCI, "protocol_retained": L_DEF,
                       "standing_backlog_annotated": L_BKG},
    "closure_anchors": {
        "p3210": ("G5-G9 audit: design e678dc42 / res 3ae23744 / seal "
                  "23008571, verdict standing_items_audit_registered; D3 "
                  "placeholder origin (3177 tail); fruit = LPF legacy topic "
                  "(2767/2800/2802); K4 = P-K4 criterion 2810 FALSE + 2833 "
                  "dual-falsify, superseded by delta_specific(cond) channel "
                  "separation"),
        "p3216": ("standing backlog ledger v2: closed x5 (item+closed_at+"
                  "evidence) incl 水果类/K4; pending_user_confirmation = "
                  "MEMO list edit"),
        "owner": "批准水果类/K4 注记, 2026-10-21 (interactive selection)",
    },
    "ledger": {"n_before": 432, "post": "98c772a8"},
}
DESIGN["mode"] = "smoke" if SMOKE else "formal"
DESIGN["created"] = today()
design_sha8 = hashlib.sha256(
    json.dumps({k: v for k, v in DESIGN.items() if k != "created"},
               sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    os.makedirs(SMOKE_DIR, exist_ok=True)
with open(os.path.join(OUT, "design_sha8.txt"), "w", encoding="utf-8",
          newline="") as f:
    f.write(design_sha8)
print(("SMOKE " if SMOKE else "") + "freeze", design_sha8)

gates = {}

# ---------------------------------------------------------------- g1: memo state
memo_raw = open(MEMO, "rb").read()
gates["g1_memo_baseline"] = (
    s8(MEMO) == DESIGN["memo_baseline_sha8"]
    and "G 线 3274 回执" in memo_raw.decode("utf-8"))
print("g1 memo baseline:", s8(MEMO), "(expect 6d0ed4fb)")

# ---------------------------------------------------------------- g2: census + classification
text = memo_raw.decode("utf-8")
lines = text.split("\n")
fruit_lines = sorted(i + 1 for i, l in enumerate(lines) if "水果类" in l)
k4_lines = sorted(i + 1 for i, l in enumerate(lines) if "K4" in l)
unique = sorted(set(fruit_lines) | set(k4_lines))
cls_ok = (
    all("水果类" in lines[n - 1] for n in L_SCI)
    and all("K4" in lines[n - 1] for n in L_DEF)
    and all(("水果类" in lines[n - 1] or "K4" in lines[n - 1]) for n in L_BKG)
    and sorted(L_SCI + L_DEF + L_BKG) == unique)
gates["g2_census_classification"] = (
    len(fruit_lines) == 26 and len(k4_lines) == 26 and len(unique) == 35
    and cls_ok)
print("g2 census: fruit=%d k4=%d unique=%d classification_ok=%s"
      % (len(fruit_lines), len(k4_lines), len(unique), cls_ok))

# ---------------------------------------------------------------- g3: closure anchors on disk
gtxt = open(GMEMO, encoding="utf-8").read()
gates["g3_closure_anchors"] = (
    "## Phase 3210 — G5-G9 水果类/K4 挂账盘点臂" in gtxt
    and "### disposition（两项均建议 close" in gtxt
    and "P-K4 判据 2810 FALSE" in gtxt
    and "closed ×5" in gtxt and "挂账总账 v2" in gtxt
    and "pending_user_confirmation" in gtxt)
print("g3 closure anchors in G-line MEMO:", gates["g3_closure_anchors"])

# ---------------------------------------------------------------- Arm b: erratum text
erratum = (
    "\n## 勘误注记：挂账「水果类」「K4」结清（2026-10-21，G 线 Phase 3275，"
    "append-only 勘误）\n\n"
    "- **裁决**：owner 于 2026-10-21 批准「水果类/K4 MEMO 挂账删除」，机制 = "
    "append-only 勘误注记（即本节），不重写历史字节；物理删除属破坏性操作须二次确认，"
    "未批准、未执行。\n"
    "- **结清锚（item+closed_at+evidence 三锚）**：3210 G5-G9 盘点臂（design "
    "e678dc42 / res 3ae23744 / seal 23008571，verdict standing_items_audit_"
    "registered）——「水果类、K4」首现于 3177 尾挂账列表，系继承占位项、无 G 线未结"
    "工作产物；水果类 = LPF 线历史研究主题（2767/2800/2802 MEMO 锚定，若未来恢复"
    "水果类协议工作可转 appendix）；K4 = P-K4 判据 2810 FALSE（func 1.400 > "
    "content 1.269）+ 2833 与 P-K2 双否证，被 Δ_specific(cond) 通道分离线取代。"
    "3216 挂账总账 v2：closed ×5（含水果类/K4）。R1-P6 之「K4 执行或正式改判挂起」"
    "处置要求，经 owner 批准按上述三锚改判关闭，不再排期。\n"
    "- **处置**：全 MEMO 挂账清单行中的 standing 名目「水果类」「K4」自本注记起视为"
    "已结清（stale-as-closed）；历史行字节不动，本注记为唯一权威处置记录。行内其他"
    "名目（N2h1-α-1、N3 链、跨线项等）处置不变。\n"
    "- **逐行注记（35 行 52 处；行号按现役基线 6d0ed4fb）**：\n"
    "  - 保留-科学发现/统计事实（水果类崩塌 0.04/0.05 等，不在勘误范围）："
    + ", ".join(str(n) for n in L_SCI) + "\n"
    "  - 保留-协议判据定义（死线 K4 定义行，非挂账状态行）："
    + ", ".join(str(n) for n in L_DEF) + "\n"
    "  - 挂账清单行（standing 名目 → 已结清；N2h1-β 任务行之挂账状态随名目「水果类」"
    "一并结清，其科学事实保留）："
    + ", ".join(str(n) for n in L_BKG) + "\n"
    "- **状态更新**：本注记后 pending_user_confirmation「MEMO 挂账列表删除水果类/"
    "K4」结清；跨线剩余待决项见 research/gpt5/docs/PENDING_ITEMS_IMPACT_PREVIEW.md"
    "（APPX-1 三选一、APPX-2/3 点名）。\n")

# append-only construction: new_bytes = old_bytes + suffix
new_raw = memo_raw + erratum.encode("utf-8")
gates["g4_append_only_suffix"] = new_raw.startswith(memo_raw)
print("g4 append-only suffix construction:", gates["g4_append_only_suffix"])

# ---------------------------------------------------------------- write
if not SMOKE:
    shutil.copy2(MEMO, MEMO + ".bak_3275")
    with open(MEMO, "wb") as f:
        f.write(new_raw)
    # post-write self-checks
    after = open(MEMO, "rb").read()
    gates["g5_prefix_immutable"] = after[:len(memo_raw)] == memo_raw
    gates["g6_erratum_present"] = (
        "## 勘误注记：挂账「水果类」「K4」结清" in after.decode("utf-8")
        and all(str(n) in after.decode("utf-8") for n in L_BKG))
    lines_after = after.decode("utf-8").split("\n")
    # expected extra 水果类 lines come from the erratum text itself (measured,
    # not assumed): title/裁决/锚/处置/逐行注记 bullets may each contain the name
    err_fruit_extra = sum(1 for l in erratum.split("\n") if "水果类" in l)
    fruit_after = sum(1 for l in lines_after if "水果类" in l)
    gates["g7_census_stable"] = (fruit_after == 26 + err_fruit_extra)
    # original classified lines must be byte-unchanged in place
    gates["g7b_original_lines_unchanged"] = all(
        lines_after[n - 1] == lines[n - 1] for n in unique)
    print("applied: memo %s -> %s (bytes +%d)"
          % (DESIGN["memo_baseline_sha8"], s8(MEMO),
             len(new_raw) - len(memo_raw)))
else:
    gates["g5_prefix_immutable"] = None
    gates["g6_erratum_present"] = None
    gates["g7_census_stable"] = None
    gates["g7b_original_lines_unchanged"] = None
    with open(os.path.join(SMOKE_DIR, "erratum_smoke.json"), "w",
              encoding="utf-8", newline="") as f:
        json.dump({"erratum_chars": len(erratum),
                   "gates": {k: v for k, v in gates.items() if v is not None}},
                  f, ensure_ascii=False, indent=1)
    print("SMOKE: no MEMO write (erratum %d chars)" % len(erratum))

# ---------------------------------------------------------------- zero-write guard
led_now = json.load(open(LED, encoding="utf-8"))
gates["g8_ledger_untouched"] = (
    len(led_now["measurements"]) == DESIGN["ledger"]["n_before"]
    and led_now["ledger_sha256_8"] == DESIGN["ledger"]["post"])
print("g8 ledger untouched:", gates["g8_ledger_untouched"])

# ---------------------------------------------------------------- verdict
gate_all = all(v for k, v in gates.items() if v is not None)
res = {
    "phase": 3275,
    "name": "g5h38_fruit_k4_erratum",
    "line": "G",
    "date": today(),
    "design_sha8": design_sha8,
    "mode": DESIGN["mode"],
    "gates": gates,
    "gate_all": gate_all,
    "verdict": "fruit_k4_erratum_registered" if gate_all else "BLOCKED",
    "arm_a": {
        "memo_baseline_sha8": s8(MEMO) if SMOKE else DESIGN["memo_baseline_sha8"],
        "census": {"fruit_lines": len(fruit_lines), "k4_lines": len(k4_lines),
                   "unique_lines": len(unique)},
        "classification": DESIGN["classification"],
    },
    "arm_b": {
        "target": "research/deepseek/docs/AGI_DEEPSEEK_MEMO.md",
        "mechanism": "append-only erratum (suffix bytes; no history rewrite)",
        "backup": (MEMO + ".bak_3275") if not SMOKE else None,
        "memo_sha8_after": (s8(MEMO) if not SMOKE else "smoke"),
        "erratum_chars": len(erratum),
        "closure_anchors": DESIGN["closure_anchors"],
    },
    "pending_items_after": [
        "APPX-1 量化口径三选一（待 owner）",
        "APPX-2/3 执行相（协议 ce73984d 已备，待 owner 点名）",
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
