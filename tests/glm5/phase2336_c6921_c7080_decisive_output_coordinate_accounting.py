#!/usr/bin/env python3
"""Exact decisive-token HiddenState, unembedding and coordinate contribution accounting."""
from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2335 = RESULT / "phase2335_c6761_c6920_independent_construction_replication"
OUT = RESULT / "phase2336_c6921_c7080_decisive_output_coordinate_accounting"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2335 / "material/independent_constructions.jsonl"
STATES = OUT / "raw/decisive_boundary_all_checkpoints.float16.npy"
WEIGHTS = OUT / "raw/decisive_unembedding_delta.float32.npy"
CONTRIBUTIONS = OUT / "raw/decisive_coordinate_contribution.float32.npy"
LOGITS = OUT / "raw/decisive_logits.float32.npy"
PROGRESS = OUT / "raw/progress.json"
PHASE = 2336
CAMPAIGN = "C6921-C7080"
FAMILIES = ("coreference_anaphora", "translation_equivalence", "attribute_binding")
PARTITIONS = ("independent_development", "independent_lockbox")
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return io.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    io.write_rows(path, rows)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def decisive_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        if row["family"] not in FAMILIES: continue
        target, wrong = row["future_target_ids"], row["future_wrong_ids"]
        divergence = 0
        while divergence < min(len(target), len(wrong)) and target[divergence] == wrong[divergence]:
            divergence += 1
        if divergence >= min(len(target), len(wrong)):
            raise RuntimeError(("prefix_candidate", row["case_id"], target, wrong))
        prefix = row["future_prompt_ids"] + target[:divergence]
        output.append({
            **row, "decisive_prefix_ids": prefix, "divergence_index": divergence,
            "decisive_target_id": int(target[divergence]), "decisive_wrong_id": int(wrong[divergence]),
            "decisive_position": len(prefix) - 1,
        })
    output.sort(key=lambda row: row["design_index"])
    return output


def collect(model, device, rows: list[dict], batch_size: int = 12) -> dict:
    module_list = modules(model); dimension = int(model.config.hidden_size)
    shape = (len(rows), len(module_list), dimension)
    if STATES.exists() and WEIGHTS.exists() and CONTRIBUTIONS.exists() and LOGITS.exists() and PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text(encoding="utf-8")); completed = int(progress["completed"])
        states = np.lib.format.open_memmap(STATES, mode="r+")
        weights = np.lib.format.open_memmap(WEIGHTS, mode="r+")
        contributions = np.lib.format.open_memmap(CONTRIBUTIONS, mode="r+")
        logits_out = np.lib.format.open_memmap(LOGITS, mode="r+")
    else:
        completed = 0; STATES.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(STATES, mode="w+", dtype=np.float16, shape=shape)
        weights = np.lib.format.open_memmap(WEIGHTS, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        contributions = np.lib.format.open_memmap(CONTRIBUTIONS, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        logits_out = np.lib.format.open_memmap(LOGITS, mode="w+", dtype=np.float32, shape=(len(rows), 3))
    captures: dict[int, torch.Tensor] = {}; handles = []
    for q, module in enumerate(module_list):
        def hook(_module, _inputs, value, q=q): captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = baseline.pad_right([row["decisive_prefix_ids"] for row in batch], device, pad)
                captures.clear(); output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    states[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                for local, row in enumerate(batch):
                    target_id, wrong_id = row["decisive_target_id"], row["decisive_wrong_id"]
                    h = captures[len(module_list) - 1][local, ends[local]].float()
                    w = model.lm_head.weight[target_id].float() - model.lm_head.weight[wrong_id].float()
                    c = h * w
                    direct = output.logits[local, ends[local], target_id].float() - output.logits[local, ends[local], wrong_id].float()
                    weights[start + local] = w.cpu().numpy().astype(np.float32)
                    contributions[start + local] = c.cpu().numpy().astype(np.float32)
                    logits_out[start + local] = [float(direct.item()), float(c.sum().item()), float(abs(direct.item() - c.sum().item()))]
                states.flush(); weights.flush(); contributions.flush(); logits_out.flush()
                save(PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                print(f"[phase2336 collect] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        for value in (states, weights, contributions, logits_out): value.flush(); close_memmap(value)
    return {"rows": len(rows), "state_shape": list(shape), "weight_shape": [len(rows), dimension],
            "contribution_shape": [len(rows), dimension], "logit_shape": [len(rows), 3]}


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.square(actual - predicted, dtype=np.float64)) /
                 (np.sum(np.square(actual, dtype=np.float64)) + EPS))


def prototype_analysis(rows: list[dict]) -> dict:
    states = np.load(STATES, mmap_mode="r")[:, -1].astype(np.float32)
    weights = np.load(WEIGHTS, mmap_mode="r")
    contributions = np.load(CONTRIBUTIONS, mmap_mode="r")
    sources = {"hidden_final_norm": states, "unembedding_delta": weights, "coordinate_contribution": contributions}
    result = {"representations": {}, "family_replication": {}}
    for name, field in sources.items():
        prototypes = np.stack([field[[i for i, row in enumerate(rows) if row["family"] == family and row["partition"] == "independent_development"]].astype(np.float64).mean(axis=0)
                               for family in FAMILIES])
        test_indices = [i for i, row in enumerate(rows) if row["partition"] == "independent_lockbox"]
        records = []
        for index in test_indices:
            actual = field[index].astype(np.float64)
            errors = [relative_mse(actual, prototype) for prototype in prototypes]
            correct = FAMILIES.index(rows[index]["family"]); predicted = int(np.argmin(errors))
            records.append({"case_id": rows[index]["case_id"], "family": rows[index]["family"],
                            "representation": name, "predicted_family": FAMILIES[predicted], "correct": predicted == correct,
                            "correct_mse": errors[correct], "best_wrong_mse": min(v for j, v in enumerate(errors) if j != correct)})
        write_rows(OUT / f"analysis/{name}_prototype_records.jsonl", records)
        result["representations"][name] = {"rows": len(records), "accuracy": float(np.mean([r["correct"] for r in records])),
                                            "chance": 1 / len(FAMILIES),
                                            "median_correct_over_best_wrong_ratio": float(np.median([r["correct_mse"] / (r["best_wrong_mse"] + EPS) for r in records]))}
        result["family_replication"][name] = {}
        for family_index, family in enumerate(FAMILIES):
            lock = field[[i for i, row in enumerate(rows) if row["family"] == family and row["partition"] == "independent_lockbox"]].astype(np.float64).mean(axis=0)
            dev = prototypes[family_index]
            result["family_replication"][name][family] = {
                "sign_agreement": float(np.mean(dev * lock > 0)),
                "symmetric_relative_mse": float(np.sum(np.square(dev - lock)) / ((np.sum(np.square(dev)) + np.sum(np.square(lock))) / 2 + EPS)),
            }
    decision = np.load(LOGITS, mmap_mode="r")
    result["exact_accounting"] = {"max_abs_error": float(decision[:, 2].max()), "mean_abs_error": float(decision[:, 2].mean()),
                                   "decision_accuracy": float(np.mean(decision[:, 0] > 0))}
    for partition in PARTITIONS:
        idx = [i for i, row in enumerate(rows) if row["partition"] == partition]
        result["exact_accounting"][f"{partition}_decision_accuracy"] = float(np.mean(decision[idx, 0] > 0))
    cacc = result["representations"]["coordinate_contribution"]["accuracy"]
    wacc = result["representations"]["unembedding_delta"]["accuracy"]
    result["gate"] = {
        "contribution_accuracy": cacc, "weight_only_accuracy": wacc, "increment": cacc - wacc,
        "chance_twice": 2 / len(FAMILIES),
        "passed": cacc >= 2 / len(FAMILIES) and cacc >= wacc + 0.05 and result["exact_accounting"]["independent_lockbox_decision_accuracy"] >= 0.70,
        "interpretation": "requires contribution family information beyond output-token weight identity",
    }
    for value in (states, weights, contributions, decision): close_memmap(value)
    return result


def publish(rows: list[dict]) -> list[dict]:
    assets = []
    specs = (
        ("c6921_qwen4b_decisive_boundary_all_checkpoints", STATES, np.float16, "decisive HiddenState at embedding through final norm", "decisive_hiddenstate_full_coordinate_v1"),
        ("c6922_qwen4b_decisive_unembedding_delta", WEIGHTS, np.float32, "target minus wrong unembedding weight in every coordinate", "unembedding_weight_delta_full_coordinate_v1"),
        ("c6923_qwen4b_decisive_coordinate_contribution", CONTRIBUTIONS, np.float32, "exact per-coordinate target-minus-wrong logit contribution", "output_contribution_full_coordinate_v1"),
    )
    for dataset_id, source_path, dtype, description, schema in specs:
        source = np.load(source_path, mmap_mode="r")
        if source.ndim == 3:
            flat = source.reshape(-1, source.shape[-1]); metadata = [
                {"case_id": row["case_id"], "family": row["family"], "language": row["language"], "surface": row["surface"],
                 "unit": row["unit"], "state": row["state"], "partition": row["partition"], "qpoint": q,
                 "decisive_target_id": row["decisive_target_id"], "decisive_wrong_id": row["decisive_wrong_id"]}
                for row in rows for q in range(source.shape[1])]
        else:
            flat = source; metadata = [{"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                                        "surface": row["surface"], "unit": row["unit"], "state": row["state"],
                                        "partition": row["partition"], "decisive_target_id": row["decisive_target_id"],
                                        "decisive_wrong_id": row["decisive_wrong_id"]} for row in rows]
        binary = VIS / f"{dataset_id}.{np.dtype(dtype).name}.npy"
        out = atlas.create_binary(binary.name, flat.shape[0], flat.shape[1], dtype); out[:] = flat; out.flush(); close_memmap(out); close_memmap(source)
        assets.append(atlas.write_metadata(dataset_id, description, binary, metadata, "Qwen3-4B-FP16", schema,
                                           "exact output-sensitive field", "three behavior-qualified independent families",
                                           description, {"coordinate_count": 2560, "no_projection": True, "decisive_token": True}))
    return assets


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 行为合格三族的决策边界、输出权重与逐坐标贡献精确分账（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 不再搬运自然差分。只使用独立生成器中行为跨开发/lockbox通过的指代、翻译、属性三族共 `{result['collection']['rows']}` 行。对每个正确/错误完整候选找到第一个分叉token，在共同teacher-forced前缀末端保存embedding至final norm全部2560坐标；另保存目标/错误unembedding权重行之差及逐坐标乘积。分别用 HiddenState、权重差、乘积贡献的开发原型识别lockbox族，要求乘积携带的信息超过纯输出token身份。

$$
d=\min\{{r:y_r^+\ne y_r^-\}},\qquad
c_j=h_j(W_{{y_d^+,j}}-W_{{y_d^-,j}}),
$$

$$
z_{{y_d^+}}-z_{{y_d^-}}=\sum_{{j=1}}^{{2560}}c_j.
$$

**结果汇总、门槛与相关文件。** 精确账与三种表示比较 `{json.dumps(result['analysis'], ensure_ascii=False)}`；采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；发布 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。冻结门要求lockbox决策准确率≥0.70、贡献原型识别≥两倍机会水平且至少高于纯权重差0.05。脚本 `tests/glm5/phase2336_c6921_c7080_decisive_output_coordinate_accounting.py`；结果 `tests/glm5/result/phase2336_c6921_c7080_decisive_output_coordinate_accounting`。

**分析、理论进展、问题硬伤与结论。** 这一步把“状态携带什么”“输出token权重本身携带什么”“两者相乘后怎样雕刻当前竞争”分成三本全坐标账。逐坐标和精确等于logit差只闭合当前分叉token输出，不说明整句未来已存储，也不证明某坐标是语义原子。族识别可能来自模板、候选token形态或输出身份；只有贡献超过权重单独基线，才说明当前状态对权重方向进行了额外条件化。teacher forcing、程序材料、FP16状态写盘仍是硬伤。

**下一阶段路线判断。** 若贡献增量门通过，目标相同，自动用固定公共输出码A/B重新编译同三族，隔离自然token权重身份，再顺序做跨模型功能复验；若不通过，说明族区分主要由输出权重身份解释，下一步仍应先做固定输出码而不是因果闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    parent = json.loads((P2335 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]: raise RuntimeError("Phase2335 invalid")
    rows = decisive_rows(read_rows(MATERIAL)); write_rows(OUT / "material/decisive_rows.jsonl", rows)
    freeze = {"frozen_before_model_load": True, "families": list(FAMILIES), "representations": ["hidden_final_norm", "unembedding_delta", "coordinate_contribution"],
              "gate": {"decision_accuracy": 0.70, "contribution_twice_chance": True, "increment_over_weight": 0.05}}
    save(OUT / "config/frozen_contract.json", freeze)
    model = tokenizer = None
    try:
        model, tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect(model, device, rows)
    finally:
        if model is not None: model_utils.release_model(model)
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    analysis = prototype_analysis(rows); datasets = publish(rows)
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(v for k, v in row.items() if k != "id") for row in verification)
    if not verified: raise RuntimeError(("verify", verification))
    catalog = atlas.update_catalog(datasets); build = atlas.frontend_build()
    if not build["passed"]: raise RuntimeError(("build", build))
    checks = {"rows": len(rows) == 384, "exact_accounting": analysis["exact_accounting"]["max_abs_error"] < 0.01,
              "all_coordinates": collection["state_shape"][-1] == 2560, "assets_verified": verified, "frontend_build": build["passed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "collection": collection, "analysis": analysis,
              "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)), "verification": verification,
              "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)), "frontend_build": build,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]: raise RuntimeError(("phase2336_failed", checks))
    append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
