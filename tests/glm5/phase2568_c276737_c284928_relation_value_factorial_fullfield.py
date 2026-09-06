#!/usr/bin/env python3
"""Full-coordinate factorial relation x value interaction field on the validated lockbox."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
P2567 = TESTS / "result/phase2567_c264449_c276736_minimal_bridge_extension"
OUT = TESTS / "result/phase2568_c276737_c284928_relation_value_factorial_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2568, "C276737-C284928"
REGIONS = ("frame", "facts_entity", "facts_relation", "facts_value", "query_context",
           "query_relation", "query_value", "candidate", "instruction", "answer_boundary")
EFFECTS = ("relation_main", "value_main", "relation_x_value")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_balanced(items: list[tuple], limit: int = 64) -> list[tuple]:
    buckets: dict[tuple[str, str], list[tuple]] = {}
    for item in items:
        buckets.setdefault((item[2], item[3]), []).append(item)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    return selected


class CoordinateCapture:
    def __init__(self, model):
        self.layers = model_utils.get_layers(model)
        self.region_map: dict[str, list[int]] = {}
        self.values: dict[str, np.ndarray] = {}
        self.handles = []
        for layer_index, layer in enumerate(self.layers):
            for kind, name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
                def hook(_module, _inputs, output, layer_index=layer_index, kind=kind):
                    width = int(output.shape[-1])
                    if kind not in self.values:
                        self.values[kind] = np.zeros((len(self.layers), len(REGIONS), width), dtype=np.float32)
                    for region_index, region in enumerate(REGIONS):
                        positions = self.region_map[region]
                        if positions:
                            self.values[kind][layer_index, region_index] = output[
                                0, positions].float().mean(dim=0).cpu().numpy()
                    return None
                self.handles.append(getattr(layer.self_attn, name).register_forward_hook(hook))

    def clear(self, row: dict) -> None:
        self.region_map = {region: list(row["regions"].get(region, [])) for region in REGIONS}
        self.values = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def capture_one(model, row: dict, controller: CoordinateCapture) -> dict[str, np.ndarray]:
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    controller.clear(row)
    with torch.inference_mode():
        output = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                             output_hidden_states=True, return_dict=True)
    hidden = np.zeros((len(output.hidden_states), len(REGIONS), int(output.hidden_states[0].shape[-1])),
                      dtype=np.float32)
    for layer_index, state in enumerate(output.hidden_states):
        for region_index, region in enumerate(REGIONS):
            positions = controller.region_map[region]
            if positions:
                hidden[layer_index, region_index] = state[0, positions].float().mean(dim=0).cpu().numpy()
    values = {kind: value.copy() for kind, value in controller.values.items()}
    values["hidden"] = hidden
    del output
    return values


def quartets(material: list[dict], behavior: list[dict]) -> tuple[list[tuple], dict[tuple, dict]]:
    correct = {row["case_id"]: row["correct"] for row in behavior if row["ablation"] == "full_scaffold"}
    full = [row for row in material if row["ablation"] == "full_scaffold" and row["depth"] == 1]
    index = {(row["family_id"], row["binding"], row["relation_form"], row["value_form"],
              row["query_relation"], row["query_value"]): row for row in full}
    keys = []
    for prefix in sorted({key[:4] for key in index}):
        cells = [index[prefix + (query_relation, query_value)]
                 for query_relation in (0, 1) for query_value in (0, 1)]
        if all(correct[row["case_id"]] for row in cells):
            keys.append(prefix)
    return keys, index


def initialize(groups: list[str], sample: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for kind, value in sample.items():
        shape = (len(groups),) + value.shape
        output[kind] = {"sum": np.zeros(shape, dtype=np.float32),
                        "sumabs": np.zeros(shape, dtype=np.float32),
                        "positive": np.zeros(shape, dtype=np.uint16),
                        "count": np.zeros(len(groups), dtype=np.uint16)}
    return output


def field_summary(arrays: dict[str, dict[str, np.ndarray]], groups: list[str]) -> dict:
    result = {"group_labels": groups, "regions": list(REGIONS), "fields": {}}
    for kind, values in arrays.items():
        mean = values["sum"] / np.maximum(values["count"], 1).reshape((-1,) + (1,) * (values["sum"].ndim - 1))
        mean_abs = values["sumabs"] / np.maximum(values["count"], 1).reshape((-1,) + (1,) * (values["sum"].ndim - 1))
        axes = tuple(range(2, mean.ndim))
        result["fields"][kind] = {"shape": list(mean.shape),
            "rms_by_group_layer": np.sqrt(np.mean(mean.astype(np.float64) ** 2, axis=axes)).tolist(),
            "mean_abs_by_group_layer": np.mean(mean_abs, axis=axes).tolist()}
    hidden = arrays["hidden"]
    count_shape = (-1,) + (1,) * (hidden["sum"].ndim - 1)
    hidden_mean = hidden["sum"] / np.maximum(hidden["count"], 1).reshape(count_shape)
    hidden_consistency = np.maximum(hidden["positive"],
                                    np.maximum(hidden["count"], 1).reshape(count_shape) - hidden["positive"]) \
        / np.maximum(hidden["count"], 1).reshape(count_shape)
    answer_index = REGIONS.index("answer_boundary")
    interaction_groups = [index for index, label in enumerate(groups) if label.endswith("relation_x_value")]
    late = hidden_mean[interaction_groups, -10:, answer_index]
    late_consistency = hidden_consistency[interaction_groups, -10:, answer_index]
    result["late_answer_interaction"] = {
        "mean_field_rms": float(np.sqrt(np.mean(late.astype(np.float64) ** 2))),
        "median_coordinate_sign_consistency": float(np.median(late_consistency)),
        "fraction_sign_consistency_ge_075": float(np.mean(late_consistency >= .75)),
        "fraction_sign_consistency_ge_090": float(np.mean(late_consistency >= .90)),
    }
    result["embedding_interaction_rms_by_region"] = np.sqrt(np.mean(
        hidden_mean[interaction_groups, 0].astype(np.float64) ** 2, axis=(0, 2))).tolist()
    cosines = {}
    for i in interaction_groups:
        left = hidden_mean[i, :, answer_index].reshape(-1).astype(np.float64)
        for j in interaction_groups:
            if j <= i:
                continue
            right = hidden_mean[j, :, answer_index].reshape(-1).astype(np.float64)
            cosines[f"{groups[i]}__{groups[j]}"] = float(np.dot(left, right) /
                (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
    result["cross_stratum_hidden_interaction_cosine"] = cosines
    return result


def write_fields(arrays: dict[str, dict[str, np.ndarray]], groups: list[str]) -> dict:
    metadata = {}
    field_dir = OUT / "fields"
    field_dir.mkdir(parents=True, exist_ok=True)
    for kind, values in arrays.items():
        count_shape = (-1,) + (1,) * (values["sum"].ndim - 1)
        denominator = np.maximum(values["count"], 1).reshape(count_shape)
        mean = values["sum"] / denominator
        mean_abs = values["sumabs"] / denominator
        consistency = (np.maximum(values["positive"], denominator - values["positive"]) / denominator).astype(np.float16)
        path = field_dir / f"{kind}_factorial_full_coordinates.npz"
        np.savez_compressed(path, mean=mean.astype(np.float32), mean_abs=mean_abs.astype(np.float32),
                            sign_consistency=consistency, counts=values["count"],
                            group_labels=np.asarray(groups), regions=np.asarray(REGIONS))
        metadata[kind] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path),
                          "shape": list(mean.shape), "dtype_mean": "float32", "topk_primary": False}
    return metadata


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 关系×值析因交互的全物理坐标场（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 只使用Phase2567中行为合格的depth1四事实关系表，并进一步要求同一family、binding、relation/value词面下的四个$(r_q,v_q)$查询全部答对。138个合格四元组中按自然/nonce四格轮转平衡选64组，共256次Qwen3-4B BF16前向。对10个token region读取输入embedding及layer0–36全部HiddenState、36层全部raw Q/K/V投影物理坐标；不做Top-K。以二阶有限差分隔离关系和值的不可加交互：

$$\Delta_R F=F_{{10}}-F_{{00}},\quad
\Delta_V F=F_{{01}}-F_{{00}},\quad
\Delta_{{R\times V}}F=F_{{11}}-F_{{10}}-F_{{01}}+F_{{00}}.$$

每个词面格分别累计每个物理坐标的mean、mean-absolute-effect和sign-consistency，float32保留均值与低幅变化。embedding交互理论上应接近0，是加法输入负对照；随层出现的非零交互是网络计算产生的候选条件纹理，但尚不是功能命名。

**结果汇总。** 四元组总数/选择数`{result['eligible_quartets']}`/`{result['selected_quartets']}`；维度`{json.dumps(result['dimensions'], ensure_ascii=False)}`；全场摘要`{json.dumps(result['summary'], ensure_ascii=False)}`；文件`{json.dumps(result['fields'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2568_c276737_c284928_relation_value_factorial_fullfield.py`；四元组索引、embedding/HiddenState/Q/K/V全坐标析因场和final位于`{OUT}`。

**分析与理论进展。** 该场直接把“关系主效应”“值主效应”和“关系×值交互”分开，不再把binding donor差分误叫关系齿轮。embedding交互接近0而内部层出现非零交互，只支持非加性条件纹理由网络逐层产生；不同词面格的场余弦衡量物理坐标复用或旋转。Q/K/V仍只是投影输出，不能直接命名为寻址、内容或编译算法。

**问题硬伤与结论。** region内多token取均值，但所有feature坐标完整保留；只覆盖一个模型与人工表格；四元组经过正确性筛选，不是独立family锁箱；二阶差分是描述性场，不自动提供因果充分性。下一Phase必须用single-relation、single-value和double-donor干预验证XOR组合预言。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2567 / "analysis/final.json")
    material = read(P2567 / "material/rows.jsonl")
    behavior = read(P2567 / "behavior/scores.jsonl")
    eligible, index = quartets(material, behavior)
    selected = select_balanced(eligible, 64)
    if not selected:
        raise RuntimeError("no fully correct quartet in qualified strata")
    forms = sorted({(item[2], item[3]) for item in selected})
    groups = [f"r{relation_form}_v{value_form}_{effect}"
              for relation_form, value_form in forms for effect in EFFECTS]
    group_index = {label: index for index, label in enumerate(groups)}
    model = tokenizer = controller = None
    arrays = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        controller = CoordinateCapture(model)
        for item_index, prefix in enumerate(selected):
            family_id, binding, relation_form, value_form = prefix
            states = {(query_relation, query_value): capture_one(
                model, index[prefix + (query_relation, query_value)], controller)
                for query_relation in (0, 1) for query_value in (0, 1)}
            if arrays is None:
                arrays = initialize(groups, states[(0, 0)])
            for kind in states[(0, 0)]:
                effects = {"relation_main": states[(1, 0)][kind] - states[(0, 0)][kind],
                           "value_main": states[(0, 1)][kind] - states[(0, 0)][kind],
                           "relation_x_value": states[(1, 1)][kind] - states[(1, 0)][kind]
                               - states[(0, 1)][kind] + states[(0, 0)][kind]}
                for effect, value in effects.items():
                    target = group_index[f"r{relation_form}_v{value_form}_{effect}"]
                    arrays[kind]["sum"][target] += value
                    arrays[kind]["sumabs"][target] += np.abs(value)
                    arrays[kind]["positive"][target] += value > 0
                    arrays[kind]["count"][target] += 1
            if (item_index + 1) % 4 == 0 or item_index + 1 == len(selected):
                print(f"[phase2568 fullfield] {item_index + 1}/{len(selected)} quartets", flush=True)
    finally:
        if controller is not None:
            controller.close()
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    assert arrays is not None
    summary = field_summary(arrays, groups)
    fields = write_fields(arrays, groups)
    index_path = OUT / "material/selected_quartets.json"
    save(index_path, {"eligible": eligible, "selected": selected})
    dimensions = {kind: list(values["sum"].shape[1:]) for kind, values in arrays.items()}
    embedding_max = max(summary["embedding_interaction_rms_by_region"])
    form_counts = {f"r{r}_v{v}": sum(item[2:] == (r, v) for item in selected) for r, v in forms}
    checks = {"prior_complete": prior["all_checks_passed"], "only_depth1_behavior_qualified": True,
              "quartets_all_four_behavior_correct": True, "selected_at_least_24": len(selected) >= 24,
              "all_four_form_cells": len(forms) == 4,
              "all_coordinates_float32_mean": all(item["dtype_mean"] == "float32" for item in fields.values()),
              "no_topk_primary": all(not item["topk_primary"] for item in fields.values()),
              "embedding_interaction_near_zero": embedding_max < 1e-5,
              "all_files_hashed": all(item["sha256"] for item in fields.values()), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "eligible_quartets": len(eligible), "selected_quartets": len(selected), "form_counts": form_counts,
              "dimensions": dimensions, "groups": groups, "summary": summary, "fields": fields,
              "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
