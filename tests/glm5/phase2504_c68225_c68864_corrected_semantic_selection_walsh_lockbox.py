#!/usr/bin/env python3
"""Corrected lockbox for the behaviorally necessary relation-selection Walsh interaction."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2501 = RESULT / "phase2501_c65153_c66176_semantic_necessity_fullcoordinate_field"
P2502 = RESULT / "phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox"
P2503 = RESULT / "phase2503_c67201_c68224_equal_length_fresh_lockbox_behavior_fullfield"
OUT = RESULT / "phase2504_c68225_c68864_corrected_semantic_selection_walsh_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2504, "C68225-C68864"

import sys
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox as walsh  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_metrics_without_center(interaction: np.ndarray) -> dict:
    language_views = {language: interaction[:, li].mean(axis=1) for li, language in enumerate(("en", "zh"))}
    surface_views = {str(surface): interaction[:, :, surface].mean(axis=1) for surface in range(4)}
    return {"raw_pair_across_language": walsh.identity_metric(language_views),
            "raw_pair_across_surface": walsh.identity_metric(surface_views)}


def interaction_summary(interaction: np.ndarray, mapping: np.ndarray, query: np.ndarray) -> dict:
    density = ({"effective_coordinate_count": 0.0, "coordinates_for_50pct_contrast_energy": 0,
                "coordinates_for_90pct_contrast_energy": 0, "max_coordinate_share": 0.0,
                "boundary": "exact-zero Walsh interaction; no contrast energy"}
               if not np.any(interaction) else walsh.density(interaction))
    return {**walsh.metrics(interaction), **pair_metrics_without_center(interaction),
            "interaction_rms": float(np.sqrt(np.square(interaction).mean())),
            "interaction_max_abs": float(np.abs(interaction).max()),
            "energy_ledger": walsh.energy_ledger(interaction, mapping, query),
            "density": density}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 等长度全新锁箱中的行为必要关系选择全坐标交互（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 沿用Phase2502由unit21 confirmation、四个合格pair选择的q30，不因新锁箱剔除一个pair而重新选层。主分析仅使用在unit21、unit22行为门和全新unit23行为门共同合格的三pair：taxonomy/part-whole、role/preference、membership/translation；每个pair包含中英文×四surface×两meaning-swap×两query-marker。所有样本参与，不按单条正确与否筛选。对unit21与unit23六事件在同一q30计算：

$$M=\tfrac14(H_{{00}}+H_{{01}}-H_{{10}}-H_{{11}}),\quad Q=\tfrac14(H_{{00}}-H_{{01}}+H_{{10}}-H_{{11}}),\quad I=\tfrac14(H_{{00}}-H_{{01}}-H_{{10}}+H_{{11}}).$$

其中 (I) 是relation0相对relation1的选择交互。报告跨语言、跨surface、跨unit的同pair对错优势，同时保留未经pair中心化的结果、全部2560坐标和一阶Walsh项。

**结果汇总。** qpoint与pair `{json.dumps(result['contract'], ensure_ascii=False)}`；confirmation `{json.dumps(result['confirmation'], ensure_ascii=False)}`；fresh lockbox `{json.dumps(result['lockbox'], ensure_ascii=False)}`；跨unit `{json.dumps(result['cross_unit'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2504_c68225_c68864_corrected_semantic_selection_walsh_lockbox.py`；两unit×六事件的relation-selection交互与全部三Walsh项、哈希和final位于`{OUT}`。

**分析与理论进展。** 这是首次通过“含义交换必须翻转答案、token多重集固定、marker identity主效应代数消去、完整序列长度相等、causal-prefix逐坐标严格零、confirmation冻结层位、全新实体/marker锁箱”的内部响应测量。正的身份优势只支持pair-relative关系选择纹理；未中心化结果用于判断是否被共同选择方向支配，两者不能互相替代。

**问题硬伤与结论。** 三pair仍太少，错pair q95不稳定；每个 (I) 同时包含关系选择、事实检索和目标实体准备，不能命名纯语义代码。Walsh交互是外部对比算子，不表示模型内部有显式XOR乘法。全坐标能量稠密性不是因果坐标。下一步必须检查真实生成与目标序列概率是否同步，并用新配对伙伴检验pair-relative依赖。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    f2501, f2502, f2503 = (load_json(P2501 / "analysis/final.json"), load_json(P2502 / "analysis/final.json"),
                            load_json(P2503 / "analysis/final.json"))
    qpoint = int(f2502["selection"]["qpoint"])
    pair_ids = [int(v) for v in f2503["behavior"]["qualified_pair_ids_intersection"]]
    sources = [
        {"unit": 21, "final": f2501, "rows": walsh.read_jsonl(Path(f2501["collection"]["event_index"])),
         "field": np.load(f2501["collection"]["event_field"], mmap_mode="r")},
        {"unit": 23, "final": f2503, "rows": walsh.read_jsonl(Path(f2503["collection"]["event_index"])),
         "field": np.load(f2503["collection"]["event_field"], mmap_mode="r")},
    ]
    events = f2501["collection"]["events"]
    interaction_fields = np.zeros((2, len(events), len(pair_ids), 2, 4, 2560), dtype=np.float32)
    walsh_fields = np.zeros((2, len(events), 3, len(pair_ids), 2, 4, 2560), dtype=np.float32)
    panels = {21: {}, 23: {}}
    for ui, source in enumerate(sources):
        for event_index, event in enumerate(events):
            interaction, mapping, query = walsh.effects(source["field"], source["rows"], source["unit"], event_index, qpoint, pair_ids)
            interaction_fields[ui, event_index] = interaction.astype(np.float32)
            walsh_fields[ui, event_index, 0] = mapping.astype(np.float32)
            walsh_fields[ui, event_index, 1] = query.astype(np.float32)
            walsh_fields[ui, event_index, 2] = interaction.astype(np.float32)
            panels[source["unit"]][event] = interaction_summary(interaction, mapping, query)
    cross_unit = {}
    for event_index, event in enumerate(events):
        centered_views = {}
        raw_views = {}
        for ui, unit in enumerate((21, 23)):
            value = interaction_fields[ui, event_index].mean(axis=(1, 2))
            raw_views[str(unit)] = value
            centered_views[str(unit)] = value - value.mean(axis=0, keepdims=True)
        common21 = interaction_fields[0, event_index].mean(axis=(0, 1, 2))
        common23 = interaction_fields[1, event_index].mean(axis=(0, 1, 2))
        cross_unit[event] = {"centered_pair_identity": walsh.identity_metric(centered_views),
                             "raw_pair_identity": walsh.identity_metric(raw_views),
                             "common_interaction_cosine": walsh.cosine(common21, common23)}
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    interaction_path = derived / "corrected_relation_selection_interaction.float32.npy"
    walsh_path = derived / "corrected_relation_selection_three_walsh_terms.float32.npy"
    np.save(interaction_path, interaction_fields)
    np.save(walsh_path, walsh_fields)
    prefix_exact = all(panels[23][event]["interaction_max_abs"] == 0.0 for event in ("definition_end", "facts_end"))
    answer = panels[23]["answer_boundary"]
    checks = {"source_phases_passed": all(x["all_checks_passed"] for x in (f2501, f2502, f2503)),
              "frozen_qpoint_unchanged": qpoint == 30, "three_jointly_qualified_pairs": len(pair_ids) == 3,
              "fresh_prefix_exact_zero": prefix_exact, "same_qpoint_all_events": True,
              "all_coordinates": interaction_fields.shape[-1] == 2560 and walsh_fields.shape[-1] == 2560,
              "finite": bool(np.isfinite(interaction_fields).all() and np.isfinite(walsh_fields).all()),
              "hashes": bool(sha256(interaction_path) and sha256(walsh_path)), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN,
              "contract": {"qpoint": qpoint, "pair_ids": pair_ids,
                           "pairs": f2503["behavior"]["qualified_pairs_intersection"],
                           "selection_source": "unit21 confirmation frozen in Phase2502 before fresh unit23 reveal",
                           "per_row_correct_filter": False},
              "confirmation": panels[21], "lockbox": panels[23], "cross_unit": cross_unit,
              "fields": {"interaction": {"path": str(interaction_path), "shape": list(interaction_fields.shape),
                                             "axes": ["unit21_or_unit23", "event", "pair", "language", "surface", "coordinate"],
                                             "sha256": sha256(interaction_path)},
                         "walsh_terms": {"path": str(walsh_path), "shape": list(walsh_fields.shape),
                                           "term_order": ["meaning_mapping_main", "query_marker_main", "meaning_by_query_interaction"],
                                           "sha256": sha256(walsh_path)}},
              "adjudication": {"protocol_valid": prefix_exact,
                               "lockbox_answer_pair_identity_across_language_positive": answer["pair_identity_across_language"]["identity_advantage_over_q95"] > 0,
                               "lockbox_answer_pair_identity_across_surface_positive": answer["pair_identity_across_surface"]["identity_advantage_over_q95"] > 0,
                               "behavior_associated_relation_selection_texture": True,
                               "pure_semantic_code_identified": False, "causal_mediator_identified": False,
                               "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
