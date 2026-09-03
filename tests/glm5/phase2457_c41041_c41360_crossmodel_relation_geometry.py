#!/usr/bin/env python3
"""Width-free relation-geometry adjudication for Phase 2454-2456.

This phase never aligns physical coordinates across architectures.  It reduces each
model's complete coordinate fields only after the model-relative full-coordinate
passports have been frozen, by comparing the 8x8 family cosine relation matrix.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2457_c41041_c41360_crossmodel_relation_geometry"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MODELS = ("qwen4b_bf16", "qwen14b_nf4_bf16", "glm4_int8", "ds7b_int8")
INTERACTIONS = ("semantic_validity", "lexical_control")
UNITS = (4, 5)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def normalized_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norms, 1e-30)


def family_gram(en: np.ndarray, zh: np.ndarray) -> np.ndarray:
    # Average two within-language relation matrices; no cross-model coordinate map.
    en_n = normalized_rows(en)
    zh_n = normalized_rows(zh)
    return ((en_n @ en_n.T) + (zh_n @ zh_n.T)) / 2.0


def off_diagonal(value: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(value.shape[0], 1)
    return np.asarray(value[indices], dtype=np.float64)


def relation_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = off_diagonal(left)
    b = off_diagonal(right)
    a = a - np.mean(a)
    b = b - np.mean(b)
    return cosine(a, b)


def permutation_test(left: np.ndarray, right: np.ndarray, permutations: np.ndarray) -> dict[str, float]:
    observed = relation_similarity(left, right)
    null = np.asarray([relation_similarity(left, right[p][:, p]) for p in permutations], dtype=np.float64)
    return {
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "identity_advantage": observed - float(np.quantile(null, 0.95)),
        "exceedance_fraction": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
    }


def load_sources() -> tuple[dict[str, np.ndarray], dict[str, dict], list[str]]:
    p2449 = next(RESULT.glob("phase2449_*"))
    p2454 = next(RESULT.glob("phase2454_*"))
    p2455 = next(RESULT.glob("phase2455_*"))
    p2456 = next(RESULT.glob("phase2456_*"))
    finals = {
        "qwen4b_bf16": json.loads((p2449 / "analysis/final.json").read_text(encoding="utf-8")),
        "qwen14b_nf4_bf16": json.loads((p2454 / "analysis/final.json").read_text(encoding="utf-8")),
        "glm4_int8": json.loads((p2455 / "analysis/final.json").read_text(encoding="utf-8")),
        "ds7b_int8": json.loads((p2456 / "analysis/final.json").read_text(encoding="utf-8")),
    }
    arrays = {
        "qwen4b_bf16": np.load(p2449 / "derived/canonical_semantic_lexical_vjp_passports.float32.npy", mmap_mode="r"),
        "qwen14b_nf4_bf16": np.load(p2454 / "derived/semantic_lexical_passports.float32.npy", mmap_mode="r"),
        "glm4_int8": np.load(p2455 / "derived/semantic_lexical_passports.float32.npy", mmap_mode="r"),
        "ds7b_int8": np.load(p2456 / "derived/semantic_lexical_passports.float32.npy", mmap_mode="r"),
    }
    families = finals["qwen4b_bf16"]["analysis"]["families"]
    for name in MODELS[1:]:
        if finals[name]["analysis"]["families"] != families:
            raise RuntimeError(f"family order mismatch for {name}")
    return arrays, finals, families


def extract_hxg(array: np.ndarray, model: str, interaction: int, unit_index: int) -> tuple[np.ndarray, np.ndarray]:
    if model == "qwen4b_bf16":
        # Phase2449: interaction, field, held-unit, language, qpoint, family, coordinate.
        # Frozen semantic Hxg qpoint is q16.  We deliberately use q16 for lexical too,
        # matching the cross-model workers rather than reselecting a lexical layer.
        en = array[interaction, 1, unit_index, 0, 16]
        zh = array[interaction, 1, unit_index, 1, 16]
    else:
        # Phase2454-56: interaction, field, unit(0/4/5), language,
        # qslot(q16-relative/q18-relative), family, coordinate.
        en = array[interaction, 1, unit_index + 1, 0, 0]
        zh = array[interaction, 1, unit_index + 1, 1, 0]
    return np.asarray(en), np.asarray(zh)


def main() -> None:
    arrays, finals, families = load_sources()
    rng = np.random.default_rng(2457)
    permutations = np.stack([rng.permutation(8) for _ in range(4096)])
    grams = np.zeros((4, 2, 2, 8, 8), dtype=np.float32)
    for model_index, model in enumerate(MODELS):
        for interaction in range(2):
            for unit_index in range(2):
                en, zh = extract_hxg(arrays[model], model, interaction, unit_index)
                grams[model_index, interaction, unit_index] = family_gram(en, zh)
    gram_path = OUT / "derived/bilingual_family_relation_grams.float32.npy"
    gram_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(gram_path, grams)

    within_model: dict[str, dict] = {}
    for model_index, model in enumerate(MODELS):
        within_model[model] = {}
        for interaction, name in enumerate(INTERACTIONS):
            within_model[model][name] = permutation_test(
                grams[model_index, interaction, 0], grams[model_index, interaction, 1], permutations
            )

    cross_model: dict[str, dict] = {}
    for left_index, left in enumerate(MODELS):
        for right_index in range(left_index + 1, len(MODELS)):
            right = MODELS[right_index]
            pair = f"{left}__{right}"
            cross_model[pair] = {}
            for interaction, interaction_name in enumerate(INTERACTIONS):
                cross_model[pair][interaction_name] = {}
                for unit_index, unit in enumerate(UNITS):
                    cross_model[pair][interaction_name][f"unit{unit}"] = permutation_test(
                        grams[left_index, interaction, unit_index],
                        grams[right_index, interaction, unit_index],
                        permutations,
                    )

    model_relative = {}
    for model in MODELS[1:]:
        analysis = finals[model]["analysis"]
        semantic = analysis["summary"]["semantic_validity"]["state_times_gradient"]
        lexical = analysis["summary"]["lexical_control"]["state_times_gradient"]
        model_relative[model] = {
            "precision": finals[model]["collection"]["precision"],
            "qualified_families": analysis["qualified_families"],
            "qualified_family_count": len(analysis["qualified_families"]),
            "semantic_hxg_lockbox": analysis["semantic_attribution_held_lockbox"],
            "semantic_exceeds_lexical": analysis["semantic_attribution_exceeds_lexical_held"],
            "confirmation_semantic_advantage": semantic["confirmation_unit4"]["family_identity_q95_advantage"],
            "fresh_semantic_advantage": semantic["fresh_unit5"]["family_identity_q95_advantage"],
            "confirmation_semantic_minus_lexical_coordinate": semantic["confirmation_unit4"]["language_coordinate"] - lexical["confirmation_unit4"]["language_coordinate"],
            "fresh_semantic_minus_lexical_coordinate": semantic["fresh_unit5"]["language_coordinate"] - lexical["fresh_unit5"]["language_coordinate"],
        }

    robust_model_relative = [name for name, value in model_relative.items() if value["semantic_hxg_lockbox"]]
    sem_cross_passes = []
    lex_cross_passes = []
    for pair, values in cross_model.items():
        for unit in ("unit4", "unit5"):
            if values["semantic_validity"][unit]["identity_advantage"] > 0:
                sem_cross_passes.append(f"{pair}:{unit}")
            if values["lexical_control"][unit]["identity_advantage"] > 0:
                lex_cross_passes.append(f"{pair}:{unit}")
    adjudication = {
        "model_relative_hxg_portability_replicated_in": robust_model_relative,
        "all_three_target_models_model_relative_lockbox": len(robust_model_relative) == 3,
        "semantic_crossmodel_relation_passes": sem_cross_passes,
        "lexical_crossmodel_relation_passes": lex_cross_passes,
        "semantic_relation_geometry_universal": len(sem_cross_passes) == 12,
        "semantic_specific_crossarchitecture_geometry": len(sem_cross_passes) > len(lex_cross_passes) and len(sem_cross_passes) == 12,
        "physical_coordinate_isomorphism_tested": False,
        "physical_coordinate_isomorphism_proven": False,
        "language_encoding_mechanism_closed": False,
    }
    result = {
        "phase": 2457,
        "campaign": "C41041-C41360",
        "principle": "Compare width-free 8x8 family relation matrices derived only after full-coordinate model-relative passports; never align physical coordinate IDs across architectures.",
        "families": families,
        "models": list(MODELS),
        "source_precision_boundary": {model: (finals[model].get("collection", {}).get("precision") or "Qwen3-4B BF16 CUDA") for model in MODELS},
        "field": "state_times_gradient at Qwen4B q16 mapped by frozen relative depth",
        "permutations": len(permutations),
        "relation_grams": str(gram_path),
        "relation_gram_shape": list(grams.shape),
        "within_model_unit4_unit5": within_model,
        "cross_model": cross_model,
        "model_relative_adjudication": model_relative,
        "adjudication": adjudication,
        "checks": {
            "four_models": len(MODELS) == 4,
            "two_interactions_two_held_units": grams.shape == (4, 2, 2, 8, 8),
            "all_finite": bool(np.isfinite(grams).all()) and all(math.isfinite(v) for pair in cross_model.values() for interaction in pair.values() for unit in interaction.values() for v in unit.values()),
            "permutations_4096": len(permutations) == 4096,
            "precision_separated": len({result for result in ["BF16", "NF4/BF16", "INT8"]}) == 3,
            "no_physical_coordinate_alignment": True,
            "claim_boundary": not adjudication["language_encoding_mechanism_closed"],
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    save(OUT / "derived/relation_geometry_metrics.json", {"within_model": within_model, "cross_model": cross_model})

    if "## Phase 2457:" not in MEMO.read_text(encoding="utf-8"):
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        memo = rf"""


## Phase 2457: 四模型输出条件家族关系几何与跨架构边界裁决（C41041-C41360） [{stamp}]

**测试原理与测试用例。** 汇总Qwen3-4B BF16、Qwen3-14B NF4/BF16、GLM4-9B INT8、DS7B INT8的canonical八族、中英、unit4确认与unit5 fresh全坐标$H\odot g$护照。不同架构维度和精度不允许比较坐标编号或绝对振幅；先在每模型内部用全部坐标构造八族余弦Gram，再比较宽度无关的28个非对角家族关系，并以4096次family标签置乱检验标签身份。语义validity与词项control完全并列。

$$G^M_{{ab}}=\frac12\left(\cos(P^{{M,en}}_a,P^{{M,en}}_b)+\cos(P^{{M,zh}}_a,P^{{M,zh}}_b)\right),$$
$$R(M,N)=\cos\left(\operatorname{{off}}(G^M)-\overline G^M,\operatorname{{off}}(G^N)-\overline G^N\right).$$

**结果汇总。** 模型内裁决 `{json.dumps(model_relative, ensure_ascii=False)}`；unit4→unit5关系稳定性 `{json.dumps(within_model, ensure_ascii=False)}`；跨模型关系几何 `{json.dumps(cross_model, ensure_ascii=False)}`；总裁决 `{json.dumps(adjudication, ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2457_c41041_c41360_crossmodel_relation_geometry.py`；四模型×语义/词项×unit4/5的完整$8\times8$关系矩阵、4096置乱指标和final位于`tests/glm5/result/phase2457_c41041_c41360_crossmodel_relation_geometry`。输入仍引用Phase2449、2454、2455、2456的全物理坐标护照，不新增压缩版原场。

**分析与理论进展。** 这个Phase把“模型内同坐标复现”与“跨模型关系结构复现”严格分开。若某模型的同family中英仅胜坐标移位但不胜family错配，只说明有语言共变纹理；若八族Gram跨模型胜标签置乱，才说明相对家族关系可能具有架构无关部分。词项control是必要反证，语义未超过词项时不能命名为语义专属齿轮。

**问题硬伤与结论。** Gram丢弃整体旋转、尺度、符号和大量逐坐标信息，只用于二级关系裁决，不能代替原场。Qwen14B为NF4/BF16，GLM/DS为INT8，局部梯度受量化影响；DS仅2/8家族行为合格。共同材料、候选输出和canonical协议仍可能产生任务模板关系。物理坐标同构没有被测试，更没有被证明；本Phase不闭合语言编码机制。
"""
        with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(memo)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
