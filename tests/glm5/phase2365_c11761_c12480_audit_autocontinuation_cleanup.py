#!/usr/bin/env python3
"""Audit Phase2358-2364, auto-continue matched template research, publish and clean raw fields."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
P2358 = RESULT / "phase2358_c10161_c10320_external_hypergraph_factorial_contract"
P2359 = RESULT / "phase2359_c10321_c10560_qwen4b_hypergraph_full_field"
P2360 = RESULT / "phase2360_c10561_c10800_factorial_coordinate_route_scan"
P2361 = RESULT / "phase2361_c10801_c11040_layer_token_coordinate_dynamics"
P2362 = RESULT / "phase2362_c11041_c11280_composition_prediction_tournament"
P2363 = RESULT / "phase2363_c11281_c11520_balanced_generation_realization"
P2364 = RESULT / "phase2364_c11521_c11760_crossmodel_hypergraph_structure"
OUT2365 = RESULT / "phase2365_c11761_c12000_stage_audit_and_auto_continuation"
OUT2366 = RESULT / "phase2366_c12001_c12240_matched_template_hierarchy"
OUT2367 = RESULT / "phase2367_c12241_c12480_publication_cleanup_audit"
MATERIAL = P2358 / "material/bilingual_typed_hypergraph_factorial.jsonl"
STATES = P2359 / "raw/qwen4b_boundary_all_checkpoints.float16.npy"
TOKEN_FIELD = P2359 / "raw/qwen4b_reference_all_token_all_checkpoints.float16.npy"
COEFF = P2360 / "raw/qwen4b_boolean_factorial_coefficients.float16.npy"
TRAJECTORY = P2363 / "raw/qwen4b_generation_trajectory.float16.npy"
IMPROVEMENT = OUT2366 / "derived/matched_template_coordinate_improvement.float32.npy"
FAMILIES = (
    "taxonomy", "attribute", "attitude", "grammar", "coreference", "translation",
    "causal", "temporal", "spatial", "possession", "partwhole", "negation",
)
LANGUAGES = ("en", "zh")
FACTORS = ("lexical_realization", "relation_variant", "branch_edge", "conflict_edge", "query_role")
UNITS = 8
CELLS = 32
CANDIDATES = ("global", "category", "same_family", "global_structure", "category_structure", "same_family_structure")

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2359_c10321_c11520_qwen_hypergraph_field_campaign as campaign  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def append_memo(phase: int, title: str, campaign_id: str, body: str) -> None:
    if f"## Phase {phase}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n\n## Phase {phase}: {title}（{campaign_id}） [{stamp}]\n\n{body}\n")


def phase_result_path(phase: int) -> Path:
    mapping = {
        2358: P2358, 2359: P2359, 2360: P2360, 2361: P2361, 2362: P2362,
        2363: P2363, 2364: P2364,
    }
    return mapping[phase] / "analysis/final.json"


def stage_audit() -> dict:
    records = {}
    memo_text = MEMO.read_text(encoding="utf-8")
    for phase in range(2358, 2365):
        path = phase_result_path(phase)
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        records[str(phase)] = {
            "final_exists": path.exists(), "memo_heading_count": memo_text.count(f"## Phase {phase}:"),
            "engineering_checks": bool(result.get("all_checks_passed")),
        }
    phase60 = json.loads(phase_result_path(2360).read_text(encoding="utf-8"))["analysis"]
    selected_q = int(phase60["selected_qpoint"])
    energy = np.asarray(phase60["order_energy_by_qpoint"][selected_q], dtype=np.float64)
    corrected_fraction = (energy[1:] / energy[1:].sum()).tolist()
    phase62 = json.loads(phase_result_path(2362).read_text(encoding="utf-8"))["analysis"]
    phase64 = json.loads(phase_result_path(2364).read_text(encoding="utf-8"))
    return {
        "phase": 2365, "campaign": "C11761-C12000", "phase_records": records,
        "continuous": all(row["final_exists"] and row["memo_heading_count"] == 1 and row["engineering_checks"]
                          for row in records.values()),
        "numerical_correction": {
            "phase2360_field": "selected_nonzero_order_energy_fraction_order1_to_order5",
            "corrected": corrected_fraction, "sum": float(sum(corrected_fraction)),
            "reason": "The earlier displayed vector accidentally included order0 while dividing by the order1-5 denominator; selected qpoint and every downstream computation were unaffected.",
        },
        "evidence_adjudication": {
            "retained": [
                "All 12 Qwen4B families pass the target-over-one-foil qualification minimum >=0.75.",
                "At q32, paired order1-3 interaction fields have mean bilingual cosine 0.582, but this is observation, not decoding.",
                "Whole-family order3 R2 is positive in Qwen4B, Qwen14B, GLM4 and DeepSeek; exact coordinates beat sorted controls in every tested model.",
                "Qwen14B reproduces Qwen4B late relative depth and bilingual positive transfer on behavior-qualified material.",
            ],
            "not_retained": [
                "A universal higher-order gear: order3 fails to improve additive unseen-unit prediction and is not consistently better cross-model.",
                "Universal bilingual transport: GLM4/DeepSeek bilingual prediction is negative and those models have nonqualified families.",
                "A universal generation-success marker: query difficulty remains large in Phase2363.",
                "Internal hypergraph isomorphism, a sheaf/groupoid, a causal coordinate gear, or a new closed mathematics.",
            ],
            "key_confound": "Phase2362 unseen-unit and whole-family results used different target groups, so their R2 values cannot establish that cross-family reuse is stronger than within-family reuse.",
        },
        "auto_continuation": {
            "same_goal": True, "executed_next_phase": 2366,
            "question": "On identical confirmation and fresh-unit target groups, which external conditioning level best transports exact-coordinate factorial effects?",
            "candidates": list(CANDIDATES),
            "selection": "choose on units4-5; freeze and adjudicate on units6-7; compare order1/order3 and sorted controls",
        },
        "source_metrics": {"phase2362": phase62["gate"], "phase2364": phase64["adjudication"]},
    }


def signs_orders() -> tuple[np.ndarray, np.ndarray]:
    signs = np.empty((CELLS, CELLS), dtype=np.float32)
    for cell in range(CELLS):
        for subset in range(CELLS):
            signs[cell, subset] = -1.0 if ((cell & subset).bit_count() % 2) else 1.0
    return signs, np.asarray([subset.bit_count() for subset in range(CELLS)])


def group_index(family: int, unit: int, language: int) -> int:
    return (family * UNITS + unit) * 2 + language


def unit_features(unit: int) -> tuple[int, str]:
    return 2 + unit % 3, "enumerated" if unit % 2 == 0 else "independent_prose"


def matched_train_units(target_unit: int) -> list[int]:
    target_depth, target_surface = unit_features(target_unit)
    scores = {}
    for unit in range(4):
        depth, surface = unit_features(unit)
        scores[unit] = abs(depth - target_depth) + 2 * int(surface != target_surface)
    minimum = min(scores.values())
    return [unit for unit, score in scores.items() if score == minimum]


def template_groups(candidate: str, family: int, unit: int, language: int, categories: list[str]) -> list[int]:
    train_units = list(range(4))
    if candidate.endswith("_structure"):
        train_units = matched_train_units(unit)
    if candidate.startswith("same_family"):
        families = [family]
    elif candidate.startswith("category"):
        families = [f for f in range(8) if categories[f] == categories[family]]
    else:
        families = list(range(8))
    return [group_index(f, u, language) for f in families for u in train_units]


def evaluate_candidate(coeff: np.ndarray, cube: np.ndarray, targets: list[tuple[int, int, int]],
                       candidate: str, qpoint: int, categories: list[str], coordinate_detail: bool = False) -> dict:
    signs, orders = signs_orders()
    selected1 = np.where(orders == 1)[0]
    selected3 = np.where((orders >= 1) & (orders <= 3))[0]
    rng = np.random.default_rng(2366); permutation = rng.permutation(cube.shape[-1])
    sse0 = sse1 = sse3 = sorted_sse = permuted_sse = 0.0
    err0 = np.zeros(cube.shape[-1], dtype=np.float64)
    err3 = np.zeros_like(err0)
    factor_sse = np.zeros(len(FACTORS), dtype=np.float64)
    factor_base = np.zeros(len(FACTORS), dtype=np.float64)
    for family, unit, language in targets:
        group = group_index(family, unit, language)
        actual = np.asarray(cube[group, :, qpoint], dtype=np.float32)
        base = actual[0]; truth = actual[1:]; baseline = np.repeat(base[None], CELLS - 1, axis=0)
        groups = template_groups(candidate, family, unit, language, categories)
        template = np.asarray(coeff[groups, :, qpoint], dtype=np.float32).mean(axis=0)
        pred1 = base + (signs[1:, selected1] - signs[0, selected1]) @ template[selected1]
        pred3 = base + (signs[1:, selected3] - signs[0, selected3]) @ template[selected3]
        pred_sorted = base + (signs[1:, selected3] - signs[0, selected3]) @ np.sort(template[selected3], axis=1)
        pred_permuted = base + (signs[1:, selected3] - signs[0, selected3]) @ template[selected3][:, permutation]
        e0 = np.square(truth - baseline); e3 = np.square(truth - pred3)
        sse0 += float(e0.sum()); sse1 += float(np.square(truth - pred1).sum()); sse3 += float(e3.sum())
        sorted_sse += float(np.square(truth - pred_sorted).sum()); permuted_sse += float(np.square(truth - pred_permuted).sum())
        err0 += e0.sum(axis=0); err3 += e3.sum(axis=0)
        for factor in range(len(FACTORS)):
            cell = 1 << factor
            truth_delta = actual[cell] - base
            predicted_delta = (signs[cell, selected3] - signs[0, selected3]) @ template[selected3]
            factor_base[factor] += float(np.square(truth_delta).sum())
            factor_sse[factor] += float(np.square(truth_delta - predicted_delta).sum())
    result = {
        "candidate": candidate, "target_groups": len(targets),
        "order1_r2": 1 - sse1 / max(sse0, 1e-20), "order3_r2": 1 - sse3 / max(sse0, 1e-20),
        "order3_sorted_r2": 1 - sorted_sse / max(sse0, 1e-20),
        "order3_permuted_r2": 1 - permuted_sse / max(sse0, 1e-20),
        "factor_single_toggle_r2": {factor: 1 - factor_sse[i] / max(factor_base[i], 1e-20)
                                    for i, factor in enumerate(FACTORS)},
    }
    if coordinate_detail:
        result["coordinate_improvement"] = ((err0 - err3) / max(len(targets) * 31, 1)).astype(np.float32)
    return result


def template_hierarchy() -> dict:
    phase60 = json.loads(phase_result_path(2360).read_text(encoding="utf-8"))["analysis"]
    qpoint = int(phase60["selected_qpoint"])
    rows = io.read_rows(MATERIAL)
    categories = [next(row["category"] for row in rows if row["family"] == family) for family in FAMILIES]
    coeff = np.load(COEFF, mmap_mode="r")
    states = np.load(STATES, mmap_mode="r")
    cube = states.reshape(len(FAMILIES) * UNITS * 2, CELLS, states.shape[1], states.shape[2])
    confirmation = [(f, u, language) for f in range(8) for u in (4, 5) for language in range(2)]
    fresh = [(f, u, language) for f in range(8) for u in (6, 7) for language in range(2)]
    selection = [evaluate_candidate(coeff, cube, confirmation, candidate, qpoint, categories) for candidate in CANDIDATES]
    winner = max(selection, key=lambda row: row["order3_r2"])["candidate"]
    lockbox = [evaluate_candidate(coeff, cube, fresh, candidate, qpoint, categories, coordinate_detail=True)
               for candidate in CANDIDATES]
    opposite_language = []
    signs, orders = signs_orders(); selected3 = np.where((orders >= 1) & (orders <= 3))[0]
    sse0 = sse3 = 0.0
    for family, unit, language in fresh:
        group = group_index(family, unit, language); donor = group_index(family, unit, 1 - language)
        actual = np.asarray(cube[group, :, qpoint], dtype=np.float32); base = actual[0]; truth = actual[1:]
        template = np.asarray(coeff[donor, :, qpoint], dtype=np.float32)
        predicted = base + (signs[1:, selected3] - signs[0, selected3]) @ template[selected3]
        sse0 += float(np.square(truth - base).sum()); sse3 += float(np.square(truth - predicted).sum())
    opposite_language.append({"diagnostic": "same_unit_opposite_language", "order3_r2": 1 - sse3 / max(sse0, 1e-20),
                              "not_a_contender": "uses the target unit in the other language"})
    matrix = np.stack([row.pop("coordinate_improvement") for row in lockbox])
    IMPROVEMENT.parent.mkdir(parents=True, exist_ok=True); np.save(IMPROVEMENT, matrix)
    lockbox_by_name = {row["candidate"]: row for row in lockbox}
    winner_result = lockbox_by_name[winner]
    gate = {
        "winner_positive_lockbox": winner_result["order3_r2"] > 0,
        "winner_beats_global_lockbox": winner_result["order3_r2"] > lockbox_by_name["global"]["order3_r2"],
        "winner_higher_order": winner_result["order3_r2"] > winner_result["order1_r2"],
        "winner_physical_over_sorted": winner_result["order3_r2"] > winner_result["order3_sorted_r2"],
        "winner_physical_over_permuted": winner_result["order3_r2"] > winner_result["order3_permuted_r2"],
    }
    result = {
        "phase": 2366, "campaign": "C12001-C12240", "selected_qpoint": qpoint,
        "selection_targets": "families0-7 units4-5 bilingual", "lockbox_targets": "same families0-7 units6-7 bilingual",
        "selection": selection, "selected_candidate": winner, "lockbox": lockbox,
        "opposite_language_diagnostic": opposite_language, "gate": gate,
        "conditional_template_candidate_passed": all(gate.values()),
        "interpretation_boundary": "A winning external-condition template is a predictive regularity, not evidence that the model stores the external graph verbatim.",
    }
    close(coeff); close(states)
    return result


def publish_phase2366(result: dict) -> dict:
    metadata = [{"candidate": candidate, "qpoint": result["selected_qpoint"],
                 "meaning": "positive = reference-cell error minus order<=3 template error per physical coordinate"}
                for candidate in CANDIDATES]
    return campaign.publish_array(
        "c12001_qwen4b_matched_template_coordinate_improvement",
        "Qwen3-4B matched-template coordinate prediction improvement", IMPROVEMENT, metadata,
        "Qwen3-4B-FP16", "matched_template_coordinate_improvement_v1", "held-out template hierarchy",
        "winner selected on units4-5 and adjudicated on units6-7",
        "per-coordinate baseline squared error minus order<=3 template squared error",
        2366, "C12001-C12240", np.float32, {"selected_candidate": result["selected_candidate"]},
    )


def publish_remaining_full_fields(selected_q: int) -> list[dict]:
    rows = io.read_rows(MATERIAL)
    states = np.load(STATES, mmap_mode="r")
    metadata = []
    for row in rows:
        for qpoint in range(states.shape[1]):
            metadata.append({"case_id": row["case_id"], "family": row["family"], "category": row["category"],
                             "language": row["language"], "surface": row["surface"], "unit": row["unit"],
                             "cell": row["cell"], "query": row["query"], "qpoint": qpoint,
                             "checkpoint": "embedding" if qpoint == 0 else ("final_norm" if qpoint == states.shape[1] - 1 else f"block_{qpoint:02d}_post")})
    asset1 = campaign.publish_array(
        "c12241_qwen4b_hypergraph_all_layer_full_field", "Qwen3-4B hypergraph complete all-layer full-coordinate field",
        STATES, metadata, "Qwen3-4B-FP16", "hypergraph_all_layer_full_coordinate_v1",
        "observational complete field", "all 6144 prompts x embedding+36 blocks+final norm",
        "raw activation at every model-local physical coordinate; activation, not trained parameter",
        2367, "C12241-C12480", np.float16, {"shape3d": list(states.shape)})
    close(states)

    coeff = np.load(COEFF, mmap_mode="r")
    selected_groups = [group_index(f, u, language) for f in (10, 11) for u in (6, 7) for language in range(2)]
    values = np.asarray(coeff[selected_groups, 1:], dtype=np.float16)
    coeff_meta = []
    _, orders = signs_orders()
    for group in selected_groups:
        language = group % 2; temp = group // 2; unit = temp % UNITS; family = temp // UNITS
        for subset in range(1, CELLS):
            for qpoint in range(coeff.shape[2]):
                coeff_meta.append({"family": FAMILIES[family], "unit": unit, "language": LANGUAGES[language],
                                   "subset": subset, "order": int(orders[subset]), "qpoint": qpoint,
                                   "field": "factorial_coefficient_layer_trajectory"})
    asset2 = campaign.publish_array(
        "c12242_qwen4b_whole_family_factorial_layer_trajectory",
        "Qwen3-4B whole-family lockbox factorial layer trajectory", values, coeff_meta,
        "Qwen3-4B-FP16", "whole_family_factorial_layer_trajectory_v1", "observational interaction dynamics",
        "partwhole and negation fresh-unit lockboxes, every nonzero subset and checkpoint",
        "signed Walsh-Mobius coefficient in every physical coordinate",
        2367, "C12241-C12480", np.float16, {"selected_qpoint_from_confirmation": selected_q})
    close(coeff)
    return [asset1, asset2]


def verify_existing_visuals() -> dict:
    ids = (
        "c10321_qwen4b_hypergraph_boundary_field", "c10561_qwen4b_factorial_first_order",
        "c10562_qwen4b_factorial_second_order", "c10563_qwen4b_factorial_third_order",
        "c10564_qwen4b_conditional_sign_information", "c10565_qwen4b_tensor_coordinate_loadings",
        "c10801_qwen4b_reference_all_token_field", "c10802_qwen4b_token_increment_field",
        "c10803_qwen4b_coordinate_cooperation_matrix", "c11041_qwen4b_composition_prediction_improvement",
        "c11281_qwen4b_balanced_generation_trajectory", "c11521_qwen14b_hypergraph_interaction_field",
        "c11521_glm4_hypergraph_interaction_field", "c11521_deepseek7b_hypergraph_interaction_field",
    )
    rows = []
    for dataset_id in ids:
        metadata_path = VIS / f"{dataset_id}.json"
        if not metadata_path.exists():
            rows.append({"id": dataset_id, "metadata": False, "binary": False, "sha256": False}); continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        binary = VIS / Path(metadata["binary_url"]).name
        rows.append({"id": dataset_id, "metadata": True, "binary": binary.exists(),
                     "sha256": binary.exists() and sha256(binary) == metadata["binary_sha256"],
                     "phase": metadata.get("phase"), "coordinate_count": metadata.get("coordinate_count")})
    return {"rows": rows, "all_present_and_hashed": all(row["metadata"] and row["binary"] and row["sha256"] for row in rows)}


def cleanup_raw_fields() -> dict:
    candidates = [STATES, TOKEN_FIELD, COEFF, TRAJECTORY]
    for key in ("qwen14b", "glm4", "deepseek7b"):
        candidates.append(P2364 / key / "derived/factorial_coefficients.float16.npy")
    deleted = []; reclaimed = 0
    for path in candidates:
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents:
            raise RuntimeError(("unsafe_cleanup_target", resolved))
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            reclaimed += size; deleted.append({"path": str(path), "bytes": size})
    remaining = []
    for base in (P2359, P2360, P2363, P2364):
        for path in base.rglob("*.npy"):
            lowered = path.name.lower()
            if any(term in lowered for term in ("hidden", "boundary", "token_field", "trajectory", "factorial_coefficients")):
                remaining.append(str(path))
    return {"deleted": deleted, "bytes_reclaimed": reclaimed, "remaining_unpublished_raw_field_candidates": remaining}


def main() -> None:
    final67 = OUT2367 / "analysis/final.json"
    if final67.exists():
        result = json.loads(final67.read_text(encoding="utf-8")); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    audit = stage_audit()
    save(OUT2365 / "analysis/final.json", audit)
    append_memo(2365, "大阶段证据审计、数值纠错与自动续研裁决", "C11761-C12000", rf"""
**测试原理与公式。** 对Phase2358–2364的final、MEMO连续性、证据等级和跨目标集可比性进行审计。Phase2360展示字段纠正为仅对非零阶归一：

$$p_k=E_k/\sum_{{r=1}}^5E_r,\quad k=1,\ldots,5.$$

**结果汇总。** `{json.dumps(audit, ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2365_c11761_c12480_audit_autocontinuation_cleanup.py`；结果 `tests/glm5/result/phase2365_c11761_c12000_stage_audit_and_auto_continuation`。

**理论进展、问题硬伤与结论。** 纠错只影响Phase2360一个展示向量，不影响q32选择或后续预测。whole-family正R²与unseen-unit负R²来自不同目标组，不能比较强弱。总目标仍相同，因此本Phase没有停止，而是冻结并立即执行Phase2366同目标集模板层级竞赛；这正是“因果失败不清空观察证据”的实际落实。
""")
    hierarchy = template_hierarchy()
    phase66 = {**hierarchy, "checks": {"six_candidates": len(hierarchy["selection"]) == 6,
                                       "selection_frozen": hierarchy["selected_candidate"] in CANDIDATES,
                                       "improvement_exists": IMPROVEMENT.exists()}}
    phase66["all_checks_passed"] = all(phase66["checks"].values())
    save(OUT2366 / "analysis/final.json", phase66)
    asset66 = publish_phase2366(phase66)
    append_memo(2366, "同目标集条件模板层级与具体坐标复用裁决", "C12001-C12240", rf"""
**测试原理与公式。** 在完全相同的family0–7目标集上，用unit0–3训练，unit4–5选择，unit6–7锁箱，公平比较global/category/same-family及其结构匹配版本。每个目标只观察cell0：

$$\widehat H_c(x)=H_{{target}}(0)+\sum_{{1\leq|S|\leq3}}[\chi_S(x)-\chi_S(0)]\overline{{H}}_{{S,c}}.$$

**结果汇总。** 选择与锁箱 `{json.dumps({k: v for k, v in phase66.items() if k != 'selection' and k != 'lockbox'}, ensure_ascii=False)}`；选择表 `{json.dumps(phase66['selection'], ensure_ascii=False)}`；锁箱表 `{json.dumps(phase66['lockbox'], ensure_ascii=False)}`；热力图`c12001`。

**相关文件。** 本脚本；结果 `tests/glm5/result/phase2366_c12001_c12240_matched_template_hierarchy`。

**理论进展、问题硬伤与结论。** 只有在unit4–5选出的模板于unit6–7保持正R²、优于global、三阶优于一阶并击败排序/置乱，才升级为条件模板候选。factor单翻转结果用于定位词汇、关系、支路、冲突和查询中哪些可复用；即便全门通过也仍是预测规律，不是内部超图同构或因果闭合。
""")
    selected_q = int(hierarchy["selected_qpoint"])
    remaining_assets = publish_remaining_full_fields(selected_q)
    assets = [asset66, *remaining_assets]
    verification = [atlas.verify(asset) for asset in assets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    existing = verify_existing_visuals()
    catalog = atlas.update_catalog(assets)
    frontend = atlas.frontend_build()
    if not (verified and existing["all_present_and_hashed"] and frontend["passed"]):
        raise RuntimeError(("publication_audit_failed", verification, existing, frontend))
    cleanup = cleanup_raw_fields()
    memo_before = MEMO.read_text(encoding="utf-8")
    phase_counts = {str(phase): memo_before.count(f"## Phase {phase}:") for phase in range(2358, 2367)}
    result67 = {
        "phase": 2367, "campaign": "C12241-C12480", "prior_audit": audit,
        "automatic_successor": {"phase2366": phase66, "same_goal_executed": True},
        "publication": {"datasets": json.loads(json.dumps(assets, default=str)), "verification": verification,
                        "existing_visuals": existing, "catalog": catalog, "frontend": frontend},
        "cleanup": cleanup, "memo_phase_counts_before_2367": phase_counts,
        "next_stage": {
            "same_overall_goal": True,
            "different_immediate_target": "Move from transport of averaged factorial templates to sample-conditioned local update laws for the factors that passed Phase2366; do not repeat another global/family mean campaign.",
            "required": [
                "Use concrete semantic chains and role-compositional sentences across many lexical domains, not only explicit graph-following instructions.",
                "Predict the next-layer coordinate update from the current sample state and external typed edge, then test on unseen lexical domains.",
                "Only after a local update law predicts should small norm-preserving response tests examine compensation; final-answer deletion is not the sole gate.",
            ],
        },
        "checks": {"new_assets_verified": verified, "existing_assets_hashed": existing["all_present_and_hashed"],
                   "frontend": frontend["passed"], "raw_fields_clean": not cleanup["remaining_unpublished_raw_field_candidates"],
                   "memo_2358_2366_once": all(value == 1 for value in phase_counts.values())},
    }
    result67["all_checks_passed"] = all(result67["checks"].values())
    save(final67, result67)
    append_memo(2367, "全层全坐标发布、来源验证、场数据清理与下一阶段边界", "C12241-C12480", rf"""
**测试原理与公式。** 将此前只展示3个checkpoint的Qwen4B边界场扩展为6144×38×2560完整客户端热力图，并发布whole-family锁箱全部非零析因项的层轨迹；逐文件验证shape、row metadata、finite与SHA256，前端构建通过后才删除重复/未发布原始场。

$$\mathrm{{SHA256}}(B_{{client}})=h_{{metadata}},\qquad
\text{{cleanup}}
\Leftarrow \text{{publication verified}}\land\text{{frontend passed}}.$$

**结果汇总。** 新发布 `{json.dumps(result67['publication']['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result67['checks'], ensure_ascii=False)}`；清理 `{json.dumps(cleanup, ensure_ascii=False)}`。

**相关文件。** 本脚本；结果 `tests/glm5/result/phase2367_c12241_c12480_publication_cleanup_audit`；客户端`c12241/c12242`及Phase2366的`c12001`。

**理论进展、问题硬伤与结论。** 本轮没有得到普遍高阶齿轮或新数学闭合；得到的是可定位到具体坐标的五因子响应场、跨族弱共享预测、模型依赖的语言运输差异，以及对平均模板极限的明确反证。下一即时目标已经变化为“样本条件化局部更新律”，而不是再复制平均模板或以最终答案删除作唯一标准；需要新的自然语义材料和新模型前向，故在完成本轮全部发布与清理后形成清晰阶段边界。
""")
    if not result67["all_checks_passed"]:
        raise RuntimeError(result67["checks"])
    print(json.dumps(result67, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
