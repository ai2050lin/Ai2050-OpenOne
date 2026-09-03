#!/usr/bin/env python3
"""Sequential cross-model replication of the Boolean hypergraph interaction structure."""
from __future__ import annotations

import gc
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2358 = RESULT / "phase2358_c10161_c10320_external_hypergraph_factorial_contract"
P2360 = RESULT / "phase2360_c10561_c10800_factorial_coordinate_route_scan"
P2362 = RESULT / "phase2362_c11041_c11280_composition_prediction_tournament"
OUT = RESULT / "phase2364_c11521_c11760_crossmodel_hypergraph_structure"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2358 / "material/bilingual_typed_hypergraph_factorial.jsonl"
PHASE = 2364
CAMPAIGN = "C11521-C11760"
MODELS = ("qwen14b", "glm4", "deepseek7b")
FAMILIES = (
    "taxonomy", "attribute", "attitude", "grammar", "coreference", "translation",
    "causal", "temporal", "spatial", "possession", "partwhole", "negation",
)
LANGUAGES = ("en", "zh")
FACTORS = ("lexical_realization", "relation_variant", "branch_edge", "conflict_edge", "query_role")
CELLS = 32

warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast.*")
sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2339_c7401_c7600_crossmodel_fixed_ab_replication as loader  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def model_paths(key: str) -> dict[str, Path]:
    base = OUT / key
    return {
        "base": base, "rows": base / "material/rows.jsonl",
        "states": base / "raw/boundary_all_checkpoints.float16.npy",
        "decisions": base / "raw/first_token_decisions.float32.npy",
        "progress": base / "raw/progress.json", "coeff": base / "derived/factorial_coefficients.float16.npy",
        "analysis": base / "analysis/final.json",
    }


def modules(model) -> list[Any]:
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    return [embed, *list(model.model.layers), model.model.norm]


def left_pad(sequences: list[list[int]], device: torch.device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, seq in enumerate(sequences):
        values = torch.tensor(seq, dtype=torch.long, device=device)
        ids[index, -len(seq):] = values
        mask[index, -len(seq):] = 1
    position = (mask.cumsum(1) - 1).clamp_min(0)
    return ids, mask, position


def compile_rows(tokenizer) -> tuple[list[dict], dict]:
    source = io.read_rows(MATERIAL)
    selected = [row for row in source if row["unit"] == 7]
    rows = []
    collisions = 0
    for row in selected:
        prompt_ids = [int(x) for x in tokenizer.encode(row["prompt"], add_special_tokens=False)]
        target_ids = [int(x) for x in tokenizer.encode(" " + row["target"], add_special_tokens=False)]
        foil_ids = [int(x) for x in tokenizer.encode(" " + row["foil"], add_special_tokens=False)]
        if target_ids[0] == foil_ids[0]:
            collisions += 1
        rows.append({**row, "model_index": len(rows), "prompt_ids": prompt_ids,
                     "target_ids": target_ids, "foil_ids": foil_ids,
                     "target_first_id": target_ids[0], "foil_first_id": foil_ids[0]})
    audit = {
        "rows": len(rows), "families": len(FAMILIES), "languages": len(LANGUAGES), "cells": CELLS,
        "unit": 7, "surface": "independent_prose", "first_token_collisions": collisions,
        "token_range": [min(len(row["prompt_ids"]) for row in rows), max(len(row["prompt_ids"]) for row in rows)],
    }
    if collisions:
        raise RuntimeError(("first_token_collisions", collisions))
    return rows, audit


def collect(key: str, model, device, rows: list[dict]) -> dict:
    paths = model_paths(key)
    qmodules = modules(model)
    dimension = int(model.get_input_embeddings().weight.shape[1])
    shape = (len(rows), len(qmodules), dimension)
    if paths["states"].exists() and paths["decisions"].exists() and paths["progress"].exists():
        completed = int(json.loads(paths["progress"].read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(paths["states"], mode="r+")
        decisions = np.lib.format.open_memmap(paths["decisions"], mode="r+")
    else:
        completed = 0
        paths["states"].parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(paths["states"], mode="w+", dtype=np.float16, shape=shape)
        decisions = np.lib.format.open_memmap(paths["decisions"], mode="w+", dtype=np.float32, shape=(len(rows), 4))
    capture: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(qmodules):
        def hook(_module, _inputs, value, qpoint=qpoint):
            capture[qpoint] = (value[0] if isinstance(value, tuple) else value)[:, -1].detach()
        handles.append(module.register_forward_hook(hook))
    batch_size = int(loader.MODEL_SPECS[key]["batch"])
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = left_pad([row["prompt_ids"] for row in batch], device, pad)
                capture.clear()
                output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qpoint in range(len(qmodules)):
                    states[start:start + len(batch), qpoint] = capture[qpoint].float().cpu().numpy().astype(np.float16)
                logits = output.logits[:, -1].float()
                for local, row in enumerate(batch):
                    target = int(row["target_first_id"]); foil = int(row["foil_first_id"])
                    t = float(logits[local, target]); f = float(logits[local, foil])
                    decisions[start + local] = [t, f, t - f, float(t > f)]
                states.flush(); decisions.flush()
                save(paths["progress"], {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 96 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2364 {key}] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        states.flush(); decisions.flush(); close(states); close(decisions)
    return {"shape": list(shape), "batch_size": batch_size, "quantization": loader.MODEL_SPECS[key]["quant"]}


def factorial_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signs = np.empty((CELLS, CELLS), dtype=np.float32)
    for cell in range(CELLS):
        for subset in range(CELLS):
            signs[cell, subset] = -1.0 if ((cell & subset).bit_count() % 2) else 1.0
    orders = np.asarray([subset.bit_count() for subset in range(CELLS)])
    return signs, signs.T / CELLS, orders


def build_coeff(key: str) -> dict:
    paths = model_paths(key)
    states = np.load(paths["states"], mmap_mode="r")
    shape = (len(FAMILIES) * len(LANGUAGES), CELLS, states.shape[1], states.shape[2])
    paths["coeff"].parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(paths["coeff"], mode="w+", dtype=np.float16, shape=shape)
    cube = states.reshape(shape)
    _, transform, _ = factorial_matrices()
    for group in range(shape[0]):
        output[group] = np.einsum("sc,cqd->sqd", transform, np.asarray(cube[group], dtype=np.float32), optimize=True).astype(np.float16)
    output.flush(); close(output); close(states)
    return {"shape": list(shape)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).reshape(-1); b = b.astype(np.float64, copy=False).reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def analyze(key: str, rows: list[dict], collection: dict) -> dict:
    paths = model_paths(key)
    coeff = np.load(paths["coeff"], mmap_mode="r")
    states = np.load(paths["states"], mmap_mode="r").reshape(len(FAMILIES) * 2, CELLS, collection["shape"][1], collection["shape"][2])
    decisions = np.load(paths["decisions"], mmap_mode="r")
    signs, _, orders = factorial_matrices()
    qpoints = coeff.shape[2]
    confirmation_groups = [family * 2 + language for family in (8, 9) for language in range(2)]
    order_energy = np.zeros((qpoints, 6), dtype=np.float64)
    confirm_energy = np.zeros_like(order_energy)
    for qpoint in range(qpoints):
        values = np.asarray(coeff[:, :, qpoint], dtype=np.float32)
        for order in range(6):
            order_energy[qpoint, order] = float(np.mean(np.square(values[:, orders == order]))) * int(np.sum(orders == order))
            confirm_energy[qpoint, order] = float(np.mean(np.square(values[confirmation_groups][:, orders == order]))) * int(np.sum(orders == order))
    fraction = (confirm_energy[:, 2] + confirm_energy[:, 3]) / np.maximum(confirm_energy[:, 1:].sum(axis=1), 1e-20)
    fraction[0] = -np.inf
    selected_q = int(np.argmax(fraction))
    mask13 = (orders >= 1) & (orders <= 3)
    language_cosines = []
    for family in range(len(FAMILIES)):
        language_cosines.append(cosine(np.asarray(coeff[family * 2, mask13, selected_q], dtype=np.float32),
                                      np.asarray(coeff[family * 2 + 1, mask13, selected_q], dtype=np.float32)))

    # Cross-language: predict all Chinese cells from the English coefficient field and Chinese cell0.
    prediction = {}
    rng = np.random.default_rng(2364); permutation = rng.permutation(coeff.shape[-1])
    for evaluation in ("cross_language", "whole_family"):
        targets = list(range(8)) if evaluation == "cross_language" else [10, 11]
        sse0 = sse1 = sse3 = sorted_sse = permuted_sse = 0.0
        for family in targets:
            target_groups = [family * 2 + 1] if evaluation == "cross_language" else [family * 2, family * 2 + 1]
            if evaluation == "cross_language":
                template = np.asarray(coeff[family * 2, :, selected_q], dtype=np.float32)
            else:
                template = np.asarray(coeff[[f * 2 + lang for f in range(8) for lang in range(2)], :, selected_q],
                                      dtype=np.float32).mean(axis=0)
            for group in target_groups:
                actual = np.asarray(states[group, :, selected_q], dtype=np.float32)
                base = actual[0]; truth = actual[1:]; baseline = np.repeat(base[None], CELLS - 1, axis=0)
                selected1 = np.where(orders == 1)[0]
                selected3 = np.where((orders >= 1) & (orders <= 3))[0]
                pred1 = base + (signs[1:, selected1] - signs[0, selected1]) @ template[selected1]
                pred3 = base + (signs[1:, selected3] - signs[0, selected3]) @ template[selected3]
                pred_sorted = base + (signs[1:, selected3] - signs[0, selected3]) @ np.sort(template[selected3], axis=1)
                pred_permuted = base + (signs[1:, selected3] - signs[0, selected3]) @ template[selected3][:, permutation]
                sse0 += float(np.square(truth - baseline).sum()); sse1 += float(np.square(truth - pred1).sum())
                sse3 += float(np.square(truth - pred3).sum()); sorted_sse += float(np.square(truth - pred_sorted).sum())
                permuted_sse += float(np.square(truth - pred_permuted).sum())
        prediction[evaluation] = {
            "order1_r2": 1 - sse1 / max(sse0, 1e-20), "order3_r2": 1 - sse3 / max(sse0, 1e-20),
            "sorted_r2": 1 - sorted_sse / max(sse0, 1e-20), "permuted_r2": 1 - permuted_sse / max(sse0, 1e-20),
        }
    family_minima = {}
    for family in range(len(FAMILIES)):
        values = []
        for language in range(2):
            start = (family * 2 + language) * CELLS
            values.append(float(np.asarray(decisions[start:start + CELLS, 3]).mean()))
        family_minima[FAMILIES[family]] = min(values)
    behavior_minimum = min(family_minima.values())
    result = {
        "model": key, "model_label": loader.MODEL_SPECS[key]["label"], "collection": collection,
        "behavior": {"overall": float(np.asarray(decisions[:, 3]).mean()), "minimum_family_language": behavior_minimum,
                     "family_minima": family_minima, "qualified_at_0_75": [f for f, v in family_minima.items() if v >= 0.75]},
        "selected_qpoint": selected_q, "relative_depth": selected_q / max(qpoints - 1, 1),
        "selected_interaction_fraction": float(fraction[selected_q]),
        "selected_nonzero_order_fraction": (order_energy[selected_q, 1:] / max(order_energy[selected_q, 1:].sum(), 1e-20)).tolist(),
        "cross_language_interaction_cosine": {"mean": float(np.mean(language_cosines)), "minimum": float(np.min(language_cosines)),
                                                "by_family": dict(zip(FAMILIES, language_cosines))},
        "prediction": prediction,
        "replication_flags": {
            "behavior": behavior_minimum >= 0.75,
            "whole_family_positive": prediction["whole_family"]["order3_r2"] > 0,
            "whole_family_higher_order": prediction["whole_family"]["order3_r2"] > prediction["whole_family"]["order1_r2"],
            "whole_family_physical_over_sorted": prediction["whole_family"]["order3_r2"] > prediction["whole_family"]["sorted_r2"],
            "cross_language_positive": prediction["cross_language"]["order3_r2"] > 0,
        },
        "warning": "Functional spectra are compared across models; physical coordinate numbers are never aligned.",
    }
    close(coeff); close(states); close(decisions)
    return result


def publish(key: str, analysis: dict) -> dict:
    paths = model_paths(key)
    coeff = np.load(paths["coeff"], mmap_mode="r")
    _, _, orders = factorial_matrices()
    subsets = np.where((orders >= 1) & (orders <= 3))[0]
    qpoint = int(analysis["selected_qpoint"])
    values = np.asarray(coeff[:, subsets, qpoint], dtype=np.float16)
    metadata = []
    for family in range(len(FAMILIES)):
        for language in range(2):
            for subset in subsets:
                metadata.append({"family": FAMILIES[family], "language": LANGUAGES[language], "unit": 7,
                                 "qpoint": qpoint, "relative_depth": analysis["relative_depth"],
                                 "subset": int(subset), "order": int(orders[subset]),
                                 "factor_members": [FACTORS[k] for k in range(len(FACTORS)) if (subset >> k) & 1]})
    dataset_id = f"c11521_{key}_hypergraph_interaction_field"
    binary = VIS / f"{dataset_id}.float16.npy"
    flat = values.reshape(-1, values.shape[-1])
    output = atlas.create_binary(binary.name, flat.shape[0], flat.shape[1], np.float16)
    output[:] = flat; output.flush(); close(output); close(coeff)
    return atlas.write_metadata(
        dataset_id, f"{loader.MODEL_SPECS[key]['label']} typed-hypergraph interaction field", binary, metadata,
        loader.MODEL_SPECS[key]["label"], "crossmodel_hypergraph_interaction_v1",
        "cross-model functional structural replication", "unit7 independent prose complete 32-cell cubes",
        "order1-3 signed coefficient in every model-local physical activation coordinate",
        {"phase": PHASE, "campaign": CAMPAIGN, "quantization": loader.MODEL_SPECS[key]["quant"],
         "no_topk": True, "activation_not_parameter": True, "cross_model_coordinate_alignment": False},
    )


def run_model(key: str) -> dict:
    paths = model_paths(key)
    if paths["analysis"].exists():
        return json.loads(paths["analysis"].read_text(encoding="utf-8"))
    if (paths["states"].exists() and paths["decisions"].exists() and paths["rows"].exists()
            and paths["progress"].exists()
            and int(json.loads(paths["progress"].read_text(encoding="utf-8"))["completed"]) == 768):
        rows = io.read_rows(paths["rows"])
        source = np.load(paths["states"], mmap_mode="r")
        collection = {"shape": list(source.shape), "batch_size": int(loader.MODEL_SPECS[key]["batch"]),
                      "quantization": loader.MODEL_SPECS[key]["quant"]}
        close(source)
        audit = {"rows": len(rows), "families": 12, "languages": 2, "cells": 32, "unit": 7,
                 "surface": "independent_prose", "first_token_collisions": 0,
                 "token_range": [min(len(row["prompt_ids"]) for row in rows), max(len(row["prompt_ids"]) for row in rows)]}
        freeze = {"frozen_before_forward": True, "unit": 7, "cells": 32, "families": list(FAMILIES),
                  "languages": list(LANGUAGES), "model_local_coordinates_only": True}
        coefficient = build_coeff(key)
        analysis = analyze(key, rows, collection)
        result = {"phase": PHASE, "campaign": CAMPAIGN, "model": key, "freeze": freeze,
                  "material_audit": audit, "coefficient": coefficient, "analysis": analysis,
                  "checks": {"rows": audit["rows"] == 768, "no_collisions": True,
                             "coeff_shape": coefficient["shape"][0:2] == [24, 32]}}
        result["all_checks_passed"] = all(result["checks"].values())
        save(paths["analysis"], result)
        print(f"[phase2364] resumed completed {key} field without model reload", flush=True)
        return result
    model = tokenizer = None
    try:
        model, tokenizer, device = loader.load_model(key)
        rows, audit = compile_rows(tokenizer)
        io.write_rows(paths["rows"], rows)
        freeze = {"frozen_before_forward": True, "unit": 7, "cells": 32, "families": list(FAMILIES),
                  "languages": list(LANGUAGES), "model_local_coordinates_only": True}
        save(paths["base"] / "config/frozen_contract.json", freeze)
        collection = collect(key, model, device, rows)
        coefficient = build_coeff(key)
        analysis = analyze(key, rows, collection)
        result = {"phase": PHASE, "campaign": CAMPAIGN, "model": key, "freeze": freeze,
                  "material_audit": audit, "coefficient": coefficient, "analysis": analysis,
                  "checks": {"rows": audit["rows"] == 768, "no_collisions": audit["first_token_collisions"] == 0,
                             "coeff_shape": coefficient["shape"][0:2] == [24, 32]}}
        result["all_checks_passed"] = all(result["checks"].values())
        save(paths["analysis"], result)
        if not result["all_checks_passed"]:
            raise RuntimeError(result["checks"])
        return result
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[phase2364] released {key}; allocated={torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0:.3f} GB", flush=True)


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 三异构模型有类型超图高阶交互功能复验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 严格按Qwen3-14B NF4→GLM4-9B INT8→DeepSeek-7B INT8顺序加载和释放；每模型重新分词unit7独立散文的12族×中英×32完整立方体=768条，采集模型本地embedding、全部block与final norm坐标。只比较行为资格、相对层深、交互阶能量、跨语言指纹和whole-family预测，不比较跨模型坐标号。

$$
\mathcal R_M=\left(r_{{depth}},\,E_1{{:}}E_2{{:}}E_3,\,\cos(\widehat H^{{en}},\widehat H^{{zh}}),\,R^2_{{whole-family}}\right).
$$

**结果汇总。** `{json.dumps(result['summary'], ensure_ascii=False)}`；跨模型裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；发布/清理 `{json.dumps(result['publication_cleanup'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2364_c11521_c11760_crossmodel_hypergraph_structure.py`；结果 `tests/glm5/result/phase2364_c11521_c11760_crossmodel_hypergraph_structure`；客户端`c11521_qwen14b/glm4/deepseek7b`。

**理论进展、问题硬伤与结论。** 复验对象是“不同局部坐标系统是否出现相似的功能关系”，不是共同坐标或共同层。量化精度、单一unit7和明确图指令限制外推；任一模型不通过只否定普遍性，不删除Qwen4B已成立的局部观察。只有符号一致且多模型通过的关系才进入下一阶段响应等价候选，仍不等于新数学闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    models = {}
    for key in MODELS:
        models[key] = run_model(key)
    assets = [publish(key, models[key]["analysis"]) for key in MODELS]
    verification = [atlas.verify(asset) for asset in assets]
    verified = all(all(value for name, value in row.items() if name != "id") for row in verification)
    catalog = atlas.update_catalog(assets)
    frontend = atlas.frontend_build()
    reclaimed = 0; deleted = []
    if verified and frontend["passed"]:
        for key in MODELS:
            path = model_paths(key)["states"]
            if path.exists():
                size = path.stat().st_size; path.unlink(); reclaimed += size; deleted.append(str(path))
    qwen4_factor = json.loads((P2360 / "analysis/final.json").read_text(encoding="utf-8"))["analysis"]
    qwen4_prediction = json.loads((P2362 / "analysis/final.json").read_text(encoding="utf-8"))["analysis"]
    summary = {
        "qwen4b": {"behavior": 0.9571940302848816, "relative_depth": qwen4_factor["selected_qpoint"] / 37,
                    "interaction_fraction": qwen4_factor["selected_interaction_fraction"],
                    "cross_language_cosine": qwen4_factor["cross_language_interaction_cosine"]["mean"],
                    "whole_family_order3_r2": qwen4_prediction["evaluations"]["whole_family"]["normalized_r2"]["order_3"]},
        **{key: {"behavior": value["analysis"]["behavior"]["overall"],
                 "minimum_behavior": value["analysis"]["behavior"]["minimum_family_language"],
                 "relative_depth": value["analysis"]["relative_depth"],
                 "interaction_fraction": value["analysis"]["selected_interaction_fraction"],
                 "cross_language_cosine": value["analysis"]["cross_language_interaction_cosine"]["mean"],
                 "whole_family_order3_r2": value["analysis"]["prediction"]["whole_family"]["order3_r2"],
                 "flags": value["analysis"]["replication_flags"]} for key, value in models.items()},
    }
    flags = [models[key]["analysis"]["replication_flags"] for key in MODELS]
    adjudication = {
        "behavior_all_models": all(flag["behavior"] for flag in flags),
        "whole_family_positive_all_models": all(flag["whole_family_positive"] for flag in flags),
        "whole_family_higher_order_all_models": all(flag["whole_family_higher_order"] for flag in flags),
        "physical_coordinate_advantage_all_models": all(flag["whole_family_physical_over_sorted"] for flag in flags),
        "cross_language_positive_all_models": all(flag["cross_language_positive"] for flag in flags),
    }
    adjudication["universal_functional_structure_passed"] = all(adjudication.values())
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "models": models, "summary": summary, "adjudication": adjudication,
        "publication_cleanup": {"datasets": json.loads(json.dumps(assets, default=str)), "verification": verification,
                                "frontend": frontend, "deleted_unvisualized_raw_states": deleted,
                                "bytes_reclaimed": reclaimed},
        "checks": {"model_count": len(models) == 3, "verified": verified, "frontend": frontend["passed"],
                   "sequential_cleanup": all(not model_paths(key)["states"].exists() for key in MODELS)},
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(final, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
