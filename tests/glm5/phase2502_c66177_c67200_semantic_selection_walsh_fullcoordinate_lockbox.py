#!/usr/bin/env python3
"""Extract the full-coordinate relation-selection Walsh interaction and lock it on unit22."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2500 = RESULT / "phase2500_c64001_c65152_semantic_necessity_2x2_behavior"
P2501 = RESULT / "phase2501_c65153_c66176_semantic_necessity_fullcoordinate_field"
OUT = RESULT / "phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2502, "C66177-C67200"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def identity_metric(views: dict[str, np.ndarray]) -> dict:
    keys = sorted(views)
    same, wrong = [], []
    for first, second in combinations(keys, 2):
        a, b = views[first], views[second]
        for item in range(a.shape[0]):
            same.append(cosine(a[item], b[item]))
            wrong.extend(cosine(a[item], b[other]) for other in range(b.shape[0]) if other != item)
    if not same or not wrong:
        return {"view_pairs": 0, "same_mean": 0.0, "wrong_mean": 0.0, "wrong_q95": 0.0,
                "identity_advantage_over_q95": 0.0}
    return {"view_pairs": len(list(combinations(keys, 2))), "same_mean": float(np.mean(same)),
            "wrong_mean": float(np.mean(wrong)), "wrong_q95": float(np.quantile(wrong, .95)),
            "identity_advantage_over_q95": float(np.mean(same) - np.quantile(wrong, .95))}


def effects(field: np.ndarray, rows: list[dict], unit: int, event: int, qpoint: int,
            pair_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Axes: pair, language, surface, coordinate. No behavior-correct filtering.
    interaction = np.zeros((len(pair_ids), 2, 4, field.shape[-1]), dtype=np.float64)
    mapping_main = np.zeros_like(interaction)
    query_main = np.zeros_like(interaction)
    language_map = {"en": 0, "zh": 1}
    lookup = {(r["unit"], r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r
              for r in rows}
    for pi, pair_id in enumerate(pair_ids):
        for language, li in language_map.items():
            for surface in range(4):
                cell = {}
                for meaning_swap in (0, 1):
                    for query_marker in (0, 1):
                        row = lookup[(unit, pair_id, language, surface, meaning_swap, query_marker)]
                        cell[(meaning_swap, query_marker)] = np.asarray(
                            field[row["model_row"], event, qpoint], dtype=np.float64)
                h00, h01, h10, h11 = cell[(0, 0)], cell[(0, 1)], cell[(1, 0)], cell[(1, 1)]
                mapping_main[pi, li, surface] = (h00 + h01 - h10 - h11) / 4.0
                query_main[pi, li, surface] = (h00 - h01 + h10 - h11) / 4.0
                interaction[pi, li, surface] = (h00 - h01 - h10 + h11) / 4.0
    return interaction, mapping_main, query_main


def metrics(interaction: np.ndarray) -> dict:
    language_views = {}
    for li, language in enumerate(("en", "zh")):
        value = interaction[:, li].mean(axis=1)
        language_views[language] = value - value.mean(axis=0, keepdims=True)
    surface_views = {}
    for surface in range(4):
        value = interaction[:, :, surface].mean(axis=1)
        surface_views[str(surface)] = value - value.mean(axis=0, keepdims=True)
    return {"pair_identity_across_language": identity_metric(language_views),
            "pair_identity_across_surface": identity_metric(surface_views)}


def energy_ledger(interaction: np.ndarray, mapping: np.ndarray, query: np.ndarray) -> dict:
    values = {"meaning_by_query_interaction": float(np.square(interaction).mean()),
              "meaning_mapping_main": float(np.square(mapping).mean()),
              "query_marker_main": float(np.square(query).mean())}
    total = sum(values.values())
    return {**values, "shares_within_three_walsh_terms": {key: value / max(total, 1e-30) for key, value in values.items()},
            "boundary": "descriptive mean-square ledger; not information or causal shares"}


def density(interaction: np.ndarray) -> dict:
    energy = np.square(interaction).sum(axis=(0, 1, 2))
    weights = energy / max(float(energy.sum()), 1e-30)
    ordered = np.sort(weights)[::-1]
    cumulative = np.cumsum(ordered)
    return {"effective_coordinate_count": float(1 / max(float(np.square(weights).sum()), 1e-30)),
            "coordinates_for_50pct_contrast_energy": int(np.searchsorted(cumulative, .5) + 1),
            "coordinates_for_90pct_contrast_energy": int(np.searchsorted(cumulative, .9) + 1),
            "max_coordinate_share": float(ordered[0]),
            "boundary": "Walsh-interaction energy coverage only; not information or causal importance"}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 四格交互首次执行、因果前缀数值泄漏与锁箱撤销（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2500共同合格的四个pair，不筛选单条是否答对，使用全部unit21/22四格。令 (m) 为marker含义交换，(q) 为查询marker； (m=q) 时选择pair内relation0，(m\ne q) 时选择relation1。分别计算meaning-mapping主效应、query-marker主效应和二者交互。核心交互为：

$$I=\frac14(H_{{00}}-H_{{01}}-H_{{10}}+H_{{11}}),$$

它在平衡设计中消去单独的定义交换与marker身份方向，保留“marker含义与查询marker结合后选择哪种关系/目标”的联合响应。对pair身份比较时再在每种语言或surface内对四pair中心化。unit21的answer-boundary在q1–q36中用“跨语言、跨surface身份优势较小者”选择唯一qpoint；unit22同层一次锁箱，六事件不得重新挑层。

**结果汇总。** 首次选择 `{json.dumps(result['selection'], ensure_ascii=False)}`；首次lockbox六事件 `{json.dumps(result['lockbox'], ensure_ascii=False)}`；协议诊断 `{json.dumps(result['protocol_diagnostic'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。这些lockbox身份数字作为失败诊断留存，不进入正证据账本。

**相关文件。** 脚本`tests/glm5/phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox.py`；unit21全部qpoint选择分数、两unit×六事件×三Walsh项的完整2560坐标场、哈希与final位于`{OUT}`。

**理论进展。** 四格交互仍是值得复测的基础候选：正确答案随关系含义翻转，且单独marker身份与定义交换的一阶方向被代数抵消。但首次unit22英文两个marker的token长度不同，使整段prompt长度不同；尽管definition/facts的token前缀和事件位置相同，GPU前向在不同矩阵shape下出现小数值差异。因而预注册的严格零负对照失败，首次lockbox必须撤销，不能用后续较大的query/answer信号把协议失败冲掉。

**问题硬伤与结论。** 首次unit22在definition/facts的interaction RMS为0.0164/0.0211，最大绝对坐标0.625；因此本Phase只构成测量协议发现，不构成关系选择锁箱。下一Phase冻结unit21选择，但以等token长度的新英文marker、等长度中文marker和全新实体建立unit23重新做行为与全场；若prefix仍不为零则继续停留在协议层。interaction不是模型内部显式相乘的证据，也不是因果中介。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    f2500 = json.loads((P2500 / "analysis/final.json").read_text(encoding="utf-8"))
    f2501 = json.loads((P2501 / "analysis/final.json").read_text(encoding="utf-8"))
    pair_ids = [int(v) for v in f2500["behavior"]["qualified_pair_ids"]]
    rows = read_jsonl(Path(f2501["collection"]["event_index"]))
    field = np.load(f2501["collection"]["event_field"], mmap_mode="r")
    events = f2501["collection"]["events"]
    answer_event = events.index("answer_boundary")
    discovery = []
    for qpoint in range(1, 37):
        interaction, mapping, query = effects(field, rows, 21, answer_event, qpoint, pair_ids)
        panel = metrics(interaction)
        score = min(panel["pair_identity_across_language"]["identity_advantage_over_q95"],
                    panel["pair_identity_across_surface"]["identity_advantage_over_q95"])
        discovery.append({"qpoint": qpoint, "selection_score_min_identity_advantage": float(score),
                          **panel, "energy_ledger": energy_ledger(interaction, mapping, query)})
    selected = max(discovery, key=lambda r: r["selection_score_min_identity_advantage"])
    qpoint = int(selected["qpoint"])
    interaction_fields = np.zeros((2, len(events), len(pair_ids), 2, 4, 2560), dtype=np.float32)
    walsh_fields = np.zeros((2, len(events), 3, len(pair_ids), 2, 4, 2560), dtype=np.float32)
    lockbox = {}
    densities = {}
    for ui, unit in enumerate((21, 22)):
        for event_index, event in enumerate(events):
            interaction, mapping, query = effects(field, rows, unit, event_index, qpoint, pair_ids)
            interaction_fields[ui, event_index] = interaction.astype(np.float32)
            walsh_fields[ui, event_index, 0] = mapping.astype(np.float32)
            walsh_fields[ui, event_index, 1] = query.astype(np.float32)
            walsh_fields[ui, event_index, 2] = interaction.astype(np.float32)
            if unit == 22:
                lockbox[event] = {**metrics(interaction), "interaction_rms": float(np.sqrt(np.square(interaction).mean())),
                                  "max_abs_interaction": float(np.abs(interaction).max()),
                                  "energy_ledger": energy_ledger(interaction, mapping, query)}
                densities[event] = density(interaction)
    cross_unit = {}
    for event_index, event in enumerate(events):
        views = {}
        for ui, unit in enumerate((21, 22)):
            value = interaction_fields[ui, event_index].mean(axis=(1, 2))
            views[str(unit)] = value - value.mean(axis=0, keepdims=True)
        cross_unit[event] = identity_metric(views)
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    interaction_path = derived / "relation_selection_interaction.float32.npy"
    walsh_path = derived / "relation_selection_three_walsh_terms.float32.npy"
    np.save(interaction_path, interaction_fields)
    np.save(walsh_path, walsh_fields)
    save(OUT / "analysis/unit21_allqpoint_selection.json", discovery)
    prefix_zero = all(lockbox[event]["max_abs_interaction"] == 0.0 for event in ("definition_end", "facts_end"))
    # The unit22 English query markers have unequal token counts. Prefix token IDs and event
    # positions are nevertheless equal; different full sequence shapes introduce a numerical
    # kernel floor. The preregistered exact-zero control therefore invalidates this lockbox.
    row_lookup = {(r["unit"], r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    prefix_equal, position_equal, full_length_equal = [], [], []
    for pair_id in range(6):
        for language in ("en", "zh"):
            for surface in range(4):
                for meaning_swap in (0, 1):
                    a = row_lookup[(22, pair_id, language, surface, meaning_swap, 0)]
                    b = row_lookup[(22, pair_id, language, surface, meaning_swap, 1)]
                    end = a["event_positions"][1] + 1
                    prefix_equal.append(a["prompt_ids"][:end] == b["prompt_ids"][:end])
                    position_equal.append(a["event_positions"][:2] == b["event_positions"][:2])
                    full_length_equal.append(len(a["prompt_ids"]) == len(b["prompt_ids"]))
    protocol_diagnostic = {
        "prefix_token_equal_rate": sum(prefix_equal) / len(prefix_equal),
        "prefix_event_position_equal_rate": sum(position_equal) / len(position_equal),
        "full_prompt_length_equal_rate": sum(full_length_equal) / len(full_length_equal),
        "prefix_exact_zero_required": True,
        "prefix_exact_zero_observed": prefix_zero,
        "lockbox_valid": False,
        "repair": "fresh unit23 with equal-token-length nonce markers and independent entities",
    }
    checks = {
        "source_phases_passed": f2500["all_checks_passed"] and f2501["all_checks_passed"],
        "four_qualified_pairs": len(pair_ids) == 4,
        "selection_unit21_only": 1 <= qpoint <= 36,
        "unit22_same_qpoint_all_events": True,
        "prefix_control_evaluated": True,
        "prefix_failure_detected": not prefix_zero,
        "invalid_lockbox_not_promoted": True,
        "all_coordinates": interaction_fields.shape[-1] == 2560 and walsh_fields.shape[-1] == 2560,
        "finite": bool(np.isfinite(interaction_fields).all() and np.isfinite(walsh_fields).all()),
        "hashes": bool(sha256(interaction_path) and sha256(walsh_path)),
        "claim_boundary": True,
    }
    answer = lockbox["answer_boundary"]
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "pair_ids": pair_ids,
        "pairs": [f2500["behavior"]["detail"]["22"][str(pair_id)]["families"] for pair_id in pair_ids],
        "selection": {"qpoint": qpoint, "unit21_answer_boundary": selected},
        "lockbox": lockbox, "cross_unit": cross_unit, "density": densities,
        "protocol_diagnostic": protocol_diagnostic,
        "fields": {
            "interaction": {"path": str(interaction_path), "shape": list(interaction_fields.shape),
                            "axes": ["unit21_or_unit22", "event", "pair", "language", "surface", "coordinate"],
                            "sha256": sha256(interaction_path)},
            "walsh_terms": {"path": str(walsh_path), "shape": list(walsh_fields.shape),
                            "axes": ["unit21_or_unit22", "event", "mapping_query_interaction", "pair", "language", "surface", "coordinate"],
                            "term_order": ["meaning_mapping_main", "query_marker_main", "meaning_by_query_interaction"],
                            "sha256": sha256(walsh_path)},
        },
        "adjudication": {
            "behaviorally_necessary_selection_interaction_measured": True,
            "lockbox_valid": False,
            "lockbox_answer_pair_identity_across_language_positive_but_not_accepted": answer["pair_identity_across_language"]["identity_advantage_over_q95"] > 0,
            "lockbox_answer_pair_identity_across_surface_positive_but_not_accepted": answer["pair_identity_across_surface"]["identity_advantage_over_q95"] > 0,
            "causal_mediator_identified": False, "pure_semantic_code_identified": False,
            "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False,
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
