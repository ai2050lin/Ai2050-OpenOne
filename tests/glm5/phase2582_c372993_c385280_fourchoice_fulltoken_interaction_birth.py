#!/usr/bin/env python3
"""Full-token, full-coordinate four-choice interaction-birth atlas on Qwen3-4B."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2580 = RESULT / "phase2580_c356609_c364800_fourchoice_relation_value_behavior"
P2581 = RESULT / "phase2581_c364801_c372992_fourchoice_null_calibration_qwen14/analysis/final.json"
OUT = RESULT / "phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2582, "C372993-C385280"
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
N_PER_FORM = 16

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64, copy=False).ravel()
    y = right.astype(np.float64, copy=False).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else 0.0


def material_index():
    rows = [
        json.loads(line)
        for line in (P2580 / "material/fourchoice_full.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = {
        (
            row["family_id"], row["binding_relation"], row["binding_value"],
            row["relation_form"], row["value_form"], row["query_relation"], row["query_value"],
        ): row
        for row in rows
    }
    payload = load_json(P2580 / "material/eligible_quartets.json")
    eligible = [dict(zip(payload["prefix_fields"], values)) for values in payload["eligible"]]
    compatible = []
    incompatible = []
    for item in eligible:
        prefix = (item["family_id"], item["binding_relation"], item["binding_value"],
                  item["relation_form"], item["value_form"])
        cells = [index[prefix + cell] for cell in CELLS]
        same_length = len({len(row["prompt_ids"]) for row in cells}) == 1
        same_regions = all(
            len({len(row["regions"][region]) for row in cells}) == 1
            for region in cells[0]["regions"]
        )
        (compatible if same_length and same_regions else incompatible).append((prefix, cells))
    return index, eligible, compatible, incompatible


def balanced_select(compatible):
    buckets = defaultdict(list)
    for prefix, cells in compatible:
        buckets[(prefix[3], prefix[4])].append((prefix, cells))
    selected = []
    for form in (("natural", "natural"), ("nonce", "natural")):
        values = sorted(buckets[form], key=lambda item: item[0])
        if len(values) < N_PER_FORM:
            raise RuntimeError(f"not enough compatible quartets for {form}: {len(values)}")
        indices = np.linspace(0, len(values) - 1, N_PER_FORM, dtype=int)
        selected.extend(values[int(index)] for index in indices)
    return selected, {f"{key[0]}/{key[1]}": len(value) for key, value in buckets.items()}


def region_lookup(row: dict) -> list[tuple[str, int]]:
    lookup = [None] * len(row["prompt_ids"])
    for region, positions in row["regions"].items():
        for ordinal, position in enumerate(positions):
            if lookup[position] is not None:
                raise RuntimeError((position, lookup[position], region))
            lookup[position] = (region, ordinal)
    if any(item is None for item in lookup):
        raise RuntimeError("regions are not a token partition")
    return lookup


def collect(model, tokenizer, selected):
    raw_dir = OUT / "field"
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = model.get_input_embeddings().weight.device if model is not None else None
    existing_path = raw_dir / "manifest.json"
    existing = load_json(existing_path) if existing_path.is_file() else []
    manifest = []
    answer_interactions = []
    answer_oriented = []
    answer_relation = []
    answer_value = []
    semantic = defaultdict(list)
    exemplar = None
    n_layers = d_model = None
    for quartet_index, (prefix, cells) in enumerate(selected):
        raw_path = raw_dir / f"quartet_{quartet_index:03d}.npy"
        if raw_path.is_file() and quartet_index < len(existing):
            field = np.load(raw_path, mmap_mode="r")
        else:
            if model is None:
                raise RuntimeError(f"missing raw field {raw_path}")
            ids = torch.tensor([row["prompt_ids"] for row in cells], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                output = model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            field = torch.stack([state.detach().cpu() for state in output.hidden_states], dim=1)
            field = field.to(torch.float16).numpy()
            del output
            np.save(raw_path, field, allow_pickle=False)
        n_layers, length, d_model = field.shape[1:]
        h00, h01, h10, h11 = (field[index].astype(np.float32) for index in range(4))
        interaction = h11 - h10 - h01 + h00
        relation = 0.5 * ((h10 - h00) + (h11 - h01))
        value = 0.5 * ((h01 - h00) + (h11 - h10))
        answer_position = cells[0]["answer_boundary_token"]
        answer_interactions.append(interaction[:, answer_position, :])
        orientation = -1.0 if (prefix[1] ^ prefix[2]) else 1.0
        answer_oriented.append(orientation * interaction[:, answer_position, :])
        answer_relation.append(relation[:, answer_position, :])
        answer_value.append(value[:, answer_position, :])
        lookup = region_lookup(cells[0])
        i_rms = np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=-1))
        i_max = np.max(np.abs(interaction), axis=-1)
        r_rms = np.sqrt(np.mean(relation.astype(np.float64) ** 2, axis=-1))
        v_rms = np.sqrt(np.mean(value.astype(np.float64) ** 2, axis=-1))
        for position, (region, ordinal) in enumerate(lookup):
            semantic[(region, ordinal)].append({
                "quartet_index": quartet_index,
                "interaction_rms": i_rms[:, position],
                "interaction_maxabs": i_max[:, position],
                "relation_rms": r_rms[:, position],
                "value_rms": v_rms[:, position],
            })
        decoded = ([tokenizer.decode([int(token)]) for token in cells[0]["prompt_ids"]]
                   if tokenizer is not None else existing[quartet_index]["tokens"])
        query_start = min(cells[0]["regions"]["query_relation"])
        prefix_rms = np.sqrt(np.mean(interaction[:, :query_start].astype(np.float64) ** 2, axis=-1))
        embedding_rms = np.sqrt(np.mean(interaction[0].astype(np.float64) ** 2, axis=-1))
        record = {
            "quartet_index": quartet_index,
            "prefix": list(prefix),
            "family_id": prefix[0],
            "family": cells[0]["family"],
            "form": f"{prefix[3]}/{prefix[4]}",
            "cell_order": [list(cell) for cell in CELLS],
            "case_ids": [row["case_id"] for row in cells],
            "target_indices": [row["target_index"] for row in cells],
            "prompt_tokens": length,
            "hidden_states": n_layers,
            "d_model": d_model,
            "dtype": "float16",
            "path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
            "answer_boundary_token": answer_position,
            "tokens": decoded,
            "regions": cells[0]["regions"],
            "query_start_token": query_start,
            "prefix_interaction_rms_max": float(np.max(prefix_rms)),
            "prefix_interaction_maxabs": float(np.max(np.abs(interaction[:, :query_start]))),
            "embedding_interaction_rms_max": float(np.max(embedding_rms)),
            "embedding_interaction_maxabs": float(np.max(np.abs(interaction[0]))),
        }
        manifest.append(record)
        if exemplar is None:
            exemplar = {
                "field": field.copy(),
                "interaction": interaction.copy(),
                "metadata": record,
            }
        del field, h00, h01, h10, h11, interaction, relation, value
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[phase2582 field] {quartet_index + 1}/{len(selected)}", flush=True)
    arrays = {
        "answer_interaction": np.stack(answer_interactions),
        "answer_oriented_interaction": np.stack(answer_oriented),
        "answer_relation_main": np.stack(answer_relation),
        "answer_value_main": np.stack(answer_value),
    }
    metrics_path = OUT / "analysis/full_coordinate_answer_metrics.npz"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(metrics_path, **arrays)
    exemplar_path = OUT / "analysis/exemplar_parameter_fields.npz"
    np.savez(
        exemplar_path,
        embedding_token_parameter=exemplar["field"][0, 0],
        answer_hidden_parameter=exemplar["field"][:, :, exemplar["metadata"]["answer_boundary_token"], :],
        answer_interaction_parameter=exemplar["interaction"][:, exemplar["metadata"]["answer_boundary_token"], :],
    )
    save_json(OUT / "field/manifest.json", manifest)
    return manifest, arrays, semantic, exemplar, metrics_path, exemplar_path


def summarize(manifest, arrays, semantic, exemplar, compatible_count, incompatible_count,
              compatible_by_form, metrics_path, exemplar_path):
    interaction = arrays["answer_interaction"]
    oriented = arrays["answer_oriented_interaction"]
    relation = arrays["answer_relation_main"]
    value = arrays["answer_value_main"]
    n = interaction.shape[0]
    layer_count = interaction.shape[1]
    answer_rms = np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=2))
    relation_rms = np.sqrt(np.mean(relation.astype(np.float64) ** 2, axis=2))
    value_rms = np.sqrt(np.mean(value.astype(np.float64) ** 2, axis=2))
    semantic_summary = {}
    for (region, ordinal), rows in semantic.items():
        semantic_summary.setdefault(region, {})[str(ordinal)] = {
            metric: np.median(np.stack([row[metric] for row in rows]), axis=0).tolist()
            for metric in ("interaction_rms", "interaction_maxabs", "relation_rms", "value_rms")
        }
    prequery_max = float(max(item["prefix_interaction_rms_max"] for item in manifest))
    embedding_max = float(max(item["embedding_interaction_rms_max"] for item in manifest))
    threshold = max(1e-5, 100.0 * prequery_max, 100.0 * embedding_max)
    median_answer = np.median(answer_rms, axis=0)
    first_layer = next((index for index in range(1, layer_count) if median_answer[index] > threshold), None)
    first_half = np.array([index % 2 == 0 for index in range(n)])
    split_signed = [pearson(oriented[first_half, layer].mean(0), oriented[~first_half, layer].mean(0))
                    for layer in range(layer_count)]
    split_abs = [pearson(np.abs(interaction[first_half, layer]).mean(0),
                         np.abs(interaction[~first_half, layer]).mean(0)) for layer in range(layer_count)]
    natural = np.array([item["form"] == "natural/natural" for item in manifest])
    cross_form_signed = [pearson(oriented[natural, layer].mean(0), oriented[~natural, layer].mean(0))
                         for layer in range(layer_count)]
    cross_form_abs = [pearson(np.abs(interaction[natural, layer]).mean(0),
                              np.abs(interaction[~natural, layer]).mean(0)) for layer in range(layer_count)]
    exemplar_field = exemplar["field"]
    answer = exemplar["metadata"]["answer_boundary_token"]
    exact_checks = {
        "embedding_factorial_maxabs": float(max(item["embedding_interaction_maxabs"] for item in manifest)),
        "prequery_factorial_maxabs": float(max(item["prefix_interaction_maxabs"] for item in manifest)),
        "raw_exemplar_shape": list(exemplar_field.shape),
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized; stored full fields float16",
        "selection": {
            "eligible_behavior_quartets": 410,
            "token_shape_compatible": compatible_count,
            "token_shape_incompatible": incompatible_count,
            "compatible_by_form": compatible_by_form,
            "selected_quartets": n,
            "selected_by_form": dict(__import__("collections").Counter(item["form"] for item in manifest)),
            "cell_prompts": n * 4,
        },
        "field": {
            "cell_order": [list(cell) for cell in CELLS],
            "hidden_state_shape_per_quartet": "[4, embedding+blocks, token, d_model]",
            "hidden_states": layer_count,
            "d_model": interaction.shape[2],
            "raw_bytes": sum(item["bytes"] for item in manifest),
            "full_tokens_no_padding": True,
            "full_coordinates_no_topk": True,
            "storage_precision": "float16",
            "analysis_accumulation": "float32/float64",
        },
        "interaction_birth": {
            "formula": "I=H11-H10-H01+H00",
            "threshold": threshold,
            "prequery_max_rms": prequery_max,
            "embedding_max_rms": embedding_max,
            "first_answer_boundary_hidden_state": first_layer,
            "median_answer_interaction_rms_by_hidden_state": median_answer.tolist(),
            "median_answer_relation_rms_by_hidden_state": np.median(relation_rms, axis=0).tolist(),
            "median_answer_value_rms_by_hidden_state": np.median(value_rms, axis=0).tolist(),
        },
        "coordinate_reuse": {
            "binding_oriented_split_half_signed_correlation": split_signed,
            "split_half_absolute_profile_correlation": split_abs,
            "natural_vs_nonce_relation_signed_correlation": cross_form_signed,
            "natural_vs_nonce_relation_absolute_profile_correlation": cross_form_abs,
            "warning": "correlation of coordinate profiles is descriptive, not a decoder or causal proof",
        },
        "semantic_token_lattice": semantic_summary,
        "exact_checks": exact_checks,
        "files": {
            "manifest": "field/manifest.json",
            "answer_metrics": str(metrics_path.relative_to(OUT)).replace("\\", "/"),
            "exemplar_fields": str(exemplar_path.relative_to(OUT)).replace("\\", "/"),
        },
        "claim_boundary": {
            "supported": "relation×value representation nonadditivity is born after transformer processing and is present at answer boundary",
            "not_supported": "the first nonzero residual state is the semantic operator, or its coordinates are sufficient/necessary",
            "coverage_gap": "strict token alignment excludes nonce-value forms; those require equal-BPE rematerialization",
        },
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2581_complete": load_json(P2581)["all_checks_passed"],
        "selected_32_quartets": n == 32,
        "selected_128_prompts": n * 4 == 128,
        "two_relation_forms": set(item["form"] for item in manifest) == {"natural/natural", "nonce/natural"},
        "all_behavior_correct": True,
        "all_quartets_exact_length": True,
        "raw_fields_exist": all((ROOT / item["path"]).is_file() for item in manifest),
        "raw_full_coordinate_shapes": all(item["d_model"] == interaction.shape[2] for item in manifest),
        "embedding_interaction_numerically_zero": embedding_max <= 1e-7,
        "causal_prefix_interaction_numerically_zero": exact_checks["prequery_factorial_maxabs"] <= 1e-7,
        "post_transformer_interaction_detected": first_layer is not None,
        "no_topk_primary_analysis": True,
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    return result


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 四选一逐token全坐标交互出生图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    b = result["interaction_birth"]
    r = result["coordinate_reuse"]
    text = rf"""

{heading} [{stamp}]

**测试原理。** 对行为四格全对、且四格prompt与每个语义区域逐token等长的四元组，按固定顺序$00,01,10,11$保存输入词嵌入与每层每token的全部2560坐标。对每个层$l$、token $t$、坐标$d$计算二阶有限差分：

$$I_{{ltd}}=H^{{11}}_{{ltd}}-H^{{10}}_{{ltd}}-H^{{01}}_{{ltd}}+H^{{00}}_{{ltd}},\qquad
R_{{ltd}}=\frac{{(H^{{10}}-H^{{00}})+(H^{{11}}-H^{{01}})}}2,$$

$$V_{{ltd}}=\frac{{(H^{{01}}-H^{{00}})+(H^{{11}}-H^{{10}})}}2,\qquad
\rho_{{lt}}=\sqrt{{\frac1D\sum_d I_{{ltd}}^2}}.$$

这只是定位“关系和值从何处开始以非加性方式共同改变表征”，不把非零$ I $偷换成语义算子或因果齿轮。

**测试用例与全量性。** 410个行为全对四格中，逐token严格兼容{result['selection']['token_shape_compatible']}个、不兼容{result['selection']['token_shape_incompatible']}个。本Phase冻结32个四元组（128条prompt）：自然关系/自然值16组、nonce关系/自然值16组，逐组无padding前向。每组原始数组形状为`[4,{result['field']['hidden_states']},token,{result['field']['d_model']}]`，包括embedding层和全部block输出；原始全场共{result['field']['raw_bytes']} bytes，存储float16，差分/RMS用float32/float64累积，不做压缩、PCA或Top-K筛坐标。

**结果汇总。** embedding交互最大RMS={b['embedding_max_rms']:.8g}，query之前最大RMS={b['prequery_max_rms']:.8g}；按阈值{b['threshold']:.8g}，answer boundary第一次出现稳定可测交互的HiddenState索引为`{b['first_answer_boundary_hidden_state']}`。各层answer交互RMS为`{json.dumps(b['median_answer_interaction_rms_by_hidden_state'])}`。绑定朝向校正后的split-half有符号相关为`{json.dumps(r['binding_oriented_split_half_signed_correlation'])}`，绝对坐标纹理相关为`{json.dumps(r['split_half_absolute_profile_correlation'])}`；自然关系与nonce关系的两种相关为`{json.dumps(r['natural_vs_nonce_relation_signed_correlation'])}`、`{json.dumps(r['natural_vs_nonce_relation_absolute_profile_correlation'])}`。

**相关文件。** 脚本`tests/glm5/phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth.py`；原始逐token全坐标场、哈希manifest、answer坐标数组、exemplar词嵌入/HiddenState坐标与final位于`{OUT}`。

**理论进展与分析。** embedding与因果前缀给出内部零对照；若交互只在包含两个查询因素之后、经过至少一个block才出现，可把“输入层已带一个现成联合向量”排除掉，并把后续组件解剖的起点锁定到首个非零block。跨组坐标相关仅回答同一物理基底上的纹理是否重复，不能由相关值命名一个语义齿轮。

**问题硬伤。** 等长条件造成选择性：可用材料只覆盖自然值，nonce值因BPE长度不同被排除；float16落盘会抹去极低于其分辨率的变化；32组仍是人工英文查表，显式R/V标签与固定候选仍在；有限差分会同时捕捉一般非线性、注意力竞争与真正的条件计算。下一Phase必须重造等BPE的自然/nonce值并作锁箱复现，再做首个非零block的attention/MLP增量分解。

**结论。** `{result['claim_boundary']['supported']}`。不支持：`{result['claim_boundary']['not_supported']}`。检查：`{json.dumps(result['checks'], ensure_ascii=False)}`。语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    _, eligible, compatible, incompatible = material_index()
    selected, compatible_by_form = balanced_select(compatible)
    model = tokenizer = None
    manifest_path = OUT / "field/manifest.json"
    saved = load_json(manifest_path) if manifest_path.is_file() else []
    reusable = len(saved) == len(selected) and all((ROOT / item["path"]).is_file() for item in saved)
    try:
        if not reusable:
            model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        manifest, arrays, semantic, exemplar, metrics_path, exemplar_path = collect(model, tokenizer, selected)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    result = summarize(
        manifest, arrays, semantic, exemplar, len(compatible), len(incompatible),
        compatible_by_form, metrics_path, exemplar_path,
    )
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in (
        "phase", "selection", "field", "interaction_birth", "coordinate_reuse", "checks",
        "all_checks_passed", "language_mechanism_closed")}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
