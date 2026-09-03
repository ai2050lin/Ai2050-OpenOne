#!/usr/bin/env python3
"""Exploratory all-coordinate causal tests on the behavior-qualified supported-task subdomain."""
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
P2344 = RESULT / "phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract"
P2346 = RESULT / "phase2346_c8481_c8600_factorial_coordinate_route_competition"
P2347 = RESULT / "phase2347_c8601_c8720_task_policy_formation_and_cleanup"
OUT = RESULT / "phase2348_c8721_c8840_supported_task_continuous_coalition_causality"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2344 / "material/bilingual_factorial_fixed_code.jsonl"
MASKS = P2347 / "derived/supported_task_family_continuous_masks.float32.npz"
OUTCOMES = OUT / "raw/intervention_outcomes.float32.npy"
OUTCOME_INDEX = OUT / "index/intervention_rows.jsonl"
PROGRESS = OUT / "raw/progress.json"
PHASE = 2348
CAMPAIGN = "C8721-C8840"
DOSES = (0.25, 0.50, 0.75)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract as contract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def module_list(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def intervention_specs(qpoint: int, checkpoint_count: int) -> list[dict]:
    wrong_q = qpoint - 4 if qpoint >= 5 else min(checkpoint_count - 1, qpoint + 4)
    return ([{"name": "baseline", "dose": 0.0, "qpoint": qpoint}]
            + [{"name": "matched_delete", "dose": dose, "qpoint": qpoint} for dose in DOSES]
            + [
                {"name": "wrong_family_delete", "dose": 0.50, "qpoint": qpoint},
                {"name": "permuted_mask_delete", "dose": 0.50, "qpoint": qpoint},
                {"name": "random_equal_l2_delete", "dose": 0.50, "qpoint": qpoint},
                {"name": "wrong_layer_matched_delete", "dose": 0.50, "qpoint": wrong_q},
                {"name": "prototype_rescue", "dose": 0.50, "qpoint": qpoint},
                {"name": "matched_invocation", "dose": 0.50, "qpoint": qpoint},
            ])


def test_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["partition"] == "fresh_lockbox" and row["task"] == "select_supported"
            and row["surface"] == "natural" and row["codebook"] == "AB" and row["condition"] == "original"]


def collect(model, device, rows: list[dict], qpoint: int, masks_file: Any, batch_size: int = 16) -> dict:
    specs = intervention_specs(qpoint, len(module_list(model)))
    families = list(contract.FAMILIES)
    family_index = {family: index for index, family in enumerate(families)}
    wrong_family = {family: families[(index + 1) % len(families)] for index, family in enumerate(families)}
    masks = {family: masks_file[f"mask__{family}"].astype(np.float32) for family in families}
    prototypes = {family: masks_file[f"prototype__{family}"].astype(np.float32) for family in families}
    grand = masks_file["grand"].astype(np.float32)
    rng = np.random.default_rng(8721)
    permuted = {family: rng.permutation(mask) for family, mask in masks.items()}
    random_masks = {}
    for family, mask in masks.items():
        value = np.abs(rng.standard_normal(mask.size)).astype(np.float32)
        value *= np.linalg.norm(mask) / max(np.linalg.norm(value), 1e-12)
        random_masks[family] = np.clip(value, 0.0, 1.0)
    expected = len(specs) * len(rows)
    if OUTCOMES.exists() and OUTCOME_INDEX.exists() and PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if progress.get("completed_specs") == len(specs):
            return {"rows": len(rows), "specs": specs, "outcome_shape": list(np.load(OUTCOMES, mmap_mode="r").shape),
                    "resumed_complete": True}
    OUTCOMES.parent.mkdir(parents=True, exist_ok=True)
    outcomes = np.lib.format.open_memmap(OUTCOMES, mode="w+", dtype=np.float32, shape=(expected, 4))
    metadata = []
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    cursor = 0
    with torch.inference_mode():
        for spec_index, spec in enumerate(specs):
            active_ends = None
            active_rows = None
            active_spec = spec

            def hook(_module, _inputs, value):
                tensor = value[0] if isinstance(value, tuple) else value
                edited = tensor.clone()
                if active_spec["name"] != "baseline":
                    for local, row in enumerate(active_rows):
                        end = int(active_ends[local].item())
                        family = row["family"]
                        vector = edited[local, end].float()
                        mask = torch.from_numpy(masks[family]).to(vector.device)
                        prototype = torch.from_numpy(prototypes[family]).to(vector.device)
                        if active_spec["name"] == "wrong_family_delete":
                            mask = torch.from_numpy(masks[wrong_family[family]]).to(vector.device)
                        elif active_spec["name"] == "permuted_mask_delete":
                            mask = torch.from_numpy(permuted[family]).to(vector.device)
                        elif active_spec["name"] == "random_equal_l2_delete":
                            mask = torch.from_numpy(random_masks[family]).to(vector.device)
                        dose = float(active_spec["dose"])
                        if active_spec["name"] == "prototype_rescue":
                            vector = vector * (1.0 - dose * mask) + dose * mask * prototype
                        elif active_spec["name"] == "matched_invocation":
                            vector = vector + dose * mask * (prototype - torch.from_numpy(grand).to(vector.device))
                        else:
                            vector = vector * (1.0 - dose * mask)
                        edited[local, end] = vector.to(edited.dtype)
                if isinstance(value, tuple):
                    return (edited, *value[1:])
                return edited

            handle = module_list(model)[int(spec["qpoint"])].register_forward_hook(hook)
            try:
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start + batch_size]
                    ids, mask_tensor, positions = baseline.pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                    active_ends = mask_tensor.sum(dim=1) - 1
                    active_rows = batch
                    output = model.model(input_ids=ids, attention_mask=mask_tensor, position_ids=positions,
                                         use_cache=False, return_dict=True)
                    selected = torch.stack([output.last_hidden_state[local, active_ends[local]] for local in range(len(batch))])
                    logits = model.lm_head(selected).float()
                    for local, row in enumerate(batch):
                        target, wrong = row["target_code_id"], row["wrong_code_id"]
                        margin = logits[local, target] - logits[local, wrong]
                        outcomes[cursor] = [float(margin.item()), float(margin.item() > 0),
                                            float(logits[local, target].item()), float(logits[local, wrong].item())]
                        metadata.append({"case_id": row["case_id"], "family": row["family"],
                                         "wrong_control_family": wrong_family[row["family"]], "language": row["language"],
                                         "lexical_set": row["lexical_set"], "unit": row["unit"], "state": row["state"],
                                         "intervention": spec["name"], "dose": spec["dose"],
                                         "intervention_qpoint": spec["qpoint"], "atlas_qpoint": qpoint})
                        cursor += 1
                    outcomes.flush()
            finally:
                handle.remove()
            save(PROGRESS, {"completed_specs": spec_index + 1, "cursor": cursor, "shape": list(outcomes.shape)})
            print(f"[phase2348 intervention] {spec_index + 1}/{len(specs)} {spec['name']} dose={spec['dose']}", flush=True)
    outcomes.flush(); close_memmap(outcomes)
    io.write_rows(OUTCOME_INDEX, metadata)
    return {"rows": len(rows), "specs": specs, "outcome_shape": [expected, 4], "resumed_complete": False,
            "wrong_family_map": wrong_family, "mask_policy": "continuous 2560-coordinate masks"}


def analyze() -> dict:
    rows = io.read_rows(OUTCOME_INDEX)
    outcomes = np.load(OUTCOMES, mmap_mode="r")
    by_case = defaultdict(dict)
    for index, row in enumerate(rows):
        by_case[row["case_id"]][(row["intervention"], float(row["dose"]))] = index
    specs = sorted({(row["intervention"], float(row["dose"])) for row in rows})
    aggregate = {}
    family_results = {}
    for spec in specs:
        idx = np.asarray([values[spec] for values in by_case.values()])
        baseline_idx = np.asarray([values[("baseline", 0.0)] for values in by_case.values()])
        delta = outcomes[idx, 0] - outcomes[baseline_idx, 0]
        aggregate[f"{spec[0]}:{spec[1]:.2f}"] = {"rows": len(idx), "forced_accuracy": float(np.mean(outcomes[idx, 1])),
                                                    "mean_margin": float(np.mean(outcomes[idx, 0])),
                                                    "mean_margin_delta": float(np.mean(delta)),
                                                    "median_margin_delta": float(np.median(delta))}
    for family in contract.FAMILIES:
        cases = [case_id for case_id, values in by_case.items()
                 if rows[values[("baseline", 0.0)]]["family"] == family]
        result = {}
        for spec in specs:
            idx = np.asarray([by_case[case_id][spec] for case_id in cases])
            base = np.asarray([by_case[case_id][("baseline", 0.0)] for case_id in cases])
            result[f"{spec[0]}:{spec[1]:.2f}"] = {"accuracy": float(np.mean(outcomes[idx, 1])),
                                                  "mean_margin_delta": float(np.mean(outcomes[idx, 0] - outcomes[base, 0]))}
        family_results[family] = result
    baseline_accuracy = aggregate["baseline:0.00"]["forced_accuracy"]
    d25 = aggregate["matched_delete:0.25"]["mean_margin_delta"]
    d50 = aggregate["matched_delete:0.50"]["mean_margin_delta"]
    d75 = aggregate["matched_delete:0.75"]["mean_margin_delta"]
    controls = [aggregate[name]["mean_margin_delta"] for name in
                ("wrong_family_delete:0.50", "permuted_mask_delete:0.50", "random_equal_l2_delete:0.50")]
    gate = {
        "baseline_behavior_pass": baseline_accuracy >= 0.70,
        "matched_delete_negative_dose_response": d25 < 0 and d50 < d25 and d75 < d50,
        "matched_delete_more_negative_than_all_same_layer_controls": d50 < min(controls),
        "matched_delete_more_negative_than_wrong_layer": d50 < aggregate["wrong_layer_matched_delete:0.50"]["mean_margin_delta"],
        "prototype_rescue_improves_over_matched_delete": aggregate["prototype_rescue:0.50"]["mean_margin_delta"] > d50,
        "matched_invocation_improves_margin": aggregate["matched_invocation:0.50"]["mean_margin_delta"] > 0,
    }
    gate["scoped_causal_candidate_passed"] = all(gate.values())
    close_memmap(outcomes)
    return {"aggregate": aggregate, "families": family_results, "gate": gate}


def publish_masks(masks_file: Any, qpoint: int) -> dict:
    values = []
    metadata = []
    for view in ("prototype", "mask"):
        for family in contract.FAMILIES:
            values.append(masks_file[f"{view}__{family}"].astype(np.float32))
            metadata.append({"view": view, "family": family, "macrotype": contract.MACROTYPE[family],
                             "qpoint": qpoint, "coordinate_count": 2560,
                             "coordinate_policy": "continuous all-coordinate; no Top-K"})
    values = np.stack(values)
    dataset_id = "c8721_qwen4b_supported_task_continuous_family_coalitions"
    binary = VIS / f"{dataset_id}.float32.npy"
    out = atlas.create_binary(binary.name, values.shape[0], values.shape[1], np.float32)
    out[:] = values
    out.flush(); close_memmap(out)
    return atlas.write_metadata(
        dataset_id, f"Qwen3-4B q{qpoint} supported-task continuous family coalitions",
        binary, metadata, "Qwen3-4B-FP16", "supported_task_continuous_coalitions_v1",
        "exploratory causal masks derived only from the behavior-qualified supported-task subdomain",
        "12 family prototypes and continuous masks; full task-inversion gate failed",
        "raw family prototype activations and continuous importance values for all 2560 coordinates",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "no_topk": True, "qpoint": qpoint},
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 行为合格支持任务的连续全坐标联盟删除—控制—救援（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不因`select_contradicted`失败而停止，而是把结论域严格缩到12族均行为合格的`select_supported`。使用Phase2346冻结q点和Phase2347从训练分区得到的2560维连续族掩码；在fresh_lockbox全部4 units、中英、两词汇、两状态、natural/AB/original共384条上运行baseline、0.25/0.50/0.75匹配删除、错族、坐标置乱、等L2随机、错层、原型救援和匹配调用。没有Top-K；每个坐标都可按连续权重参与。

$$
h'_j=h_j(1-\alpha M_{{f,j}}),\qquad
h_j^{{rescue}}=h'_j+\alpha M_{{f,j}}\mu_{{f,j}},
$$

$$
h_j^{{invoke}}=h_j+\alpha M_{{f,j}}(\mu_{{f,j}}-\mu_j).
$$

**结果汇总与相关文件。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；汇总与逐族结果 `{json.dumps(result['analysis'], ensure_ascii=False)}`；客户端具体坐标掩码 `{json.dumps(result['dataset'], ensure_ascii=False)}`；核验 `{json.dumps(result['verification'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2348_c8721_c8840_supported_task_continuous_coalition_causality.py`；结果 `tests/glm5/result/phase2348_c8721_c8840_supported_task_continuous_coalition_causality`。

**理论进展、问题硬伤与结论。** 这是通过特征上的局部因果探索，不是整个语言机制闭合。乘法删除可能主要改变层归一化前尺度；原型救援混合族模板和语义，且恢复不等于自然计算路径；错误族仍共享大量坐标，因此是严格控制。只有匹配删除呈单调负剂量、强于错族/置乱/随机/错层，原型救援相对恢复且匹配调用提高margin，才记为“支持任务局部连续联盟因果候选”。即便全部通过，也不能外推到矛盾任务、自然生成或跨模型。

**下一阶段路线判断。** 若局部因果候选通过，目标相同，自动在Qwen14B、GLM4、DS7B上顺序复验同一支持任务的行为与相对深度功能指纹，不比较坐标号；若未通过，则停止因果升级并回到更多行为合格任务政策/自然生成图谱。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    rows = test_rows(io.read_rows(MATERIAL))
    parent = json.loads((P2346 / "analysis/final.json").read_text(encoding="utf-8"))
    qpoint = int(parent["analysis"]["selected"]["qpoint"])
    masks_file = np.load(MASKS)
    freeze = {"frozen_before_model_load": True, "domain": "select_supported only", "qpoint": qpoint,
              "rows": len(rows), "doses": list(DOSES), "families": list(contract.FAMILIES),
              "coordinate_policy": "continuous all 2560 coordinates; no Top-K"}
    save(OUT / "config/frozen_contract.json", freeze)
    model = None
    try:
        model, _tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect(model, device, rows, qpoint, masks_file)
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    analysis = analyze()
    dataset = publish_masks(masks_file, qpoint)
    masks_file.close()
    verification = atlas.verify(dataset)
    verified = all(value for key, value in verification.items() if key != "id")
    catalog = atlas.update_catalog([dataset])
    build = atlas.frontend_build()
    checks = {"rows": len(rows) == 384, "outcomes": collection["outcome_shape"] == [3840, 4],
              "all_conditions": len(collection["specs"]) == 10, "asset_verified": verified,
              "frontend_build": build["passed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "collection": collection,
              "analysis": analysis, "dataset": json.loads(json.dumps(dataset, ensure_ascii=False, default=str)),
              "verification": verification, "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
              "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2348_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
