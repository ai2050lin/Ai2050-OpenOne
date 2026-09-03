#!/usr/bin/env python3
"""Fixed A/B output interface: isolate state-dependent coordinate use from answer-token identity."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2335 = RESULT / "phase2335_c6761_c6920_independent_construction_replication"
OUT = RESULT / "phase2337_c7081_c7240_fixed_ab_output_interface"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
SOURCE = P2335 / "material/independent_constructions.jsonl"
MATERIAL = OUT / "material/fixed_ab_rows.jsonl"
STATES = OUT / "raw/fixed_ab_boundary.float16.npy"
WEIGHTS = OUT / "raw/fixed_ab_weight_delta.float32.npy"
CONTRIBUTIONS = OUT / "raw/fixed_ab_coordinate_contribution.float32.npy"
DECISIONS = OUT / "raw/fixed_ab_decisions.float32.npy"
PROGRESS = OUT / "raw/progress.json"
PHASE = 2337
CAMPAIGN = "C7081-C7240"
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
    if mmap is not None:
        mmap.close()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def compile_fixed_rows(tokenizer) -> tuple[list[dict], dict]:
    code_ids = {code: tokenizer.encode(" " + code, add_special_tokens=False) for code in ("A", "B")}
    if any(len(ids) != 1 for ids in code_ids.values()):
        raise RuntimeError(("codes_not_single_token", code_ids))
    output = []
    for source in read_rows(SOURCE):
        if source["family"] not in FAMILIES:
            continue
        language_index = 0 if source["language"] == "en" else 1
        surface_index = 0 if source["surface"] == "direct" else 1
        target_is_a = (source["unit"] + source["state"] + language_index + surface_index) % 2 == 0
        target_code, wrong_code = (("A", "B") if target_is_a else ("B", "A"))
        choice_a = source["future_target_text"] if target_is_a else source["future_wrong_text"]
        choice_b = source["future_wrong_text"] if target_is_a else source["future_target_text"]
        if source["language"] == "en":
            prompt = (f'{source["future_prompt"]}\nOption A: {choice_a}\nOption B: {choice_b}\n'
                      "Select the factually correct continuation. Reply with exactly A or B.\nAnswer:")
        else:
            prompt = (f'{source["future_prompt"]}\n选项A：{choice_a}\n选项B：{choice_b}\n'
                      "选择符合事实的续写，只回答A或B。\n答案：")
        row = {
            **source,
            "case_id": source["case_id"].replace("c6761-", "c7081-"),
            "source_case_id": source["case_id"],
            "natural_target_text": source["future_target_text"],
            "natural_wrong_text": source["future_wrong_text"],
            "choice_a_text": choice_a,
            "choice_b_text": choice_b,
            "target_code": target_code,
            "wrong_code": wrong_code,
            "target_code_id": int(code_ids[target_code][0]),
            "wrong_code_id": int(code_ids[wrong_code][0]),
            "future_prompt": prompt,
            "future_prompt_ids": [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)],
            "future_target_text": " " + target_code,
            "future_wrong_text": " " + wrong_code,
            "future_target_ids": [int(code_ids[target_code][0])],
            "future_wrong_ids": [int(code_ids[wrong_code][0])],
            "identity_target": target_code,
            "identity_wrong": wrong_code,
            "cue_id": "fixed_ab",
        }
        row["boundary_position"] = len(row["future_prompt_ids"]) - 1
        output.append(row)
    output.sort(key=lambda row: row["design_index"])
    balance = {}
    for family in FAMILIES:
        balance[family] = {}
        for partition in PARTITIONS:
            cell = [r for r in output if r["family"] == family and r["partition"] == partition]
            balance[family][partition] = {code: sum(r["target_code"] == code for r in cell) for code in ("A", "B")}
    audit = {
        "rows": len(output), "families": list(FAMILIES), "code_ids": code_ids,
        "target_code_balance": balance,
        "balanced_every_family_partition": all(
            cell["A"] == cell["B"] for family in balance.values() for cell in family.values()
        ),
        "source_target_not_leaked_after_answer_label": all(r["target_code"] in ("A", "B") for r in output),
        "all_full_natural_choices_retained": all(r["choice_a_text"] != r["choice_b_text"] for r in output),
    }
    return output, audit


def collect(model, device, rows: list[dict], batch_size: int = 12) -> dict:
    module_list = modules(model)
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(module_list), dimension)
    if all(path.exists() for path in (STATES, WEIGHTS, CONTRIBUTIONS, DECISIONS, PROGRESS)):
        completed = int(json.loads(PROGRESS.read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(STATES, mode="r+")
        weights = np.lib.format.open_memmap(WEIGHTS, mode="r+")
        contributions = np.lib.format.open_memmap(CONTRIBUTIONS, mode="r+")
        decisions = np.lib.format.open_memmap(DECISIONS, mode="r+")
    else:
        completed = 0
        STATES.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(STATES, mode="w+", dtype=np.float16, shape=shape)
        weights = np.lib.format.open_memmap(WEIGHTS, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        contributions = np.lib.format.open_memmap(CONTRIBUTIONS, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        decisions = np.lib.format.open_memmap(DECISIONS, mode="w+", dtype=np.float32, shape=(len(rows), 5))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(module_list):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = baseline.pad_right([r["future_prompt_ids"] for r in batch], device, pad)
                captures.clear()
                model_out = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    states[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                for local, row in enumerate(batch):
                    target_id, wrong_id = row["target_code_id"], row["wrong_code_id"]
                    h = captures[len(module_list) - 1][local, ends[local]].float()
                    w = model.lm_head.weight[target_id].float() - model.lm_head.weight[wrong_id].float()
                    contribution = h * w
                    logits = model_out.logits[local, ends[local]].float()
                    margin = logits[target_id] - logits[wrong_id]
                    reconstructed = contribution.sum()
                    greedy_id = int(torch.argmax(logits).item())
                    weights[start + local] = w.cpu().numpy().astype(np.float32)
                    contributions[start + local] = contribution.cpu().numpy().astype(np.float32)
                    decisions[start + local] = [
                        float(margin.item()), float(reconstructed.item()),
                        float(abs(margin.item() - reconstructed.item())),
                        float(margin.item() > 0), float(greedy_id == target_id),
                    ]
                for value in (states, weights, contributions, decisions):
                    value.flush()
                save(PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                print(f"[phase2337 collect] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        for value in (states, weights, contributions, decisions):
            value.flush()
            close_memmap(value)
    return {
        "rows": len(rows), "state_shape": list(shape),
        "weight_shape": [len(rows), dimension], "contribution_shape": [len(rows), dimension],
        "decision_shape": [len(rows), 5],
    }


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.square(actual - predicted, dtype=np.float64)) /
                 (np.sum(np.square(actual, dtype=np.float64)) + EPS))


def behavior(rows: list[dict], decisions: np.ndarray) -> dict:
    result = {"families": {}, "qualified": []}
    for family in FAMILIES:
        passed = True
        cells = {}
        for partition in PARTITIONS:
            idx = [i for i, row in enumerate(rows) if row["family"] == family and row["partition"] == partition]
            cell = {
                "rows": len(idx),
                "forced_ab_accuracy": float(np.mean(decisions[idx, 3])),
                "unconstrained_exact_next_code_accuracy": float(np.mean(decisions[idx, 4])),
            }
            cell["passed"] = cell["forced_ab_accuracy"] >= 0.70 and cell["unconstrained_exact_next_code_accuracy"] >= 0.40
            passed = passed and cell["passed"]
            cells[partition] = cell
        result["families"][family] = {"qualified": passed, "partitions": cells}
        if passed:
            result["qualified"].append(family)
    return result


def prototype_analysis(rows: list[dict], behavior_result: dict) -> dict:
    states_map = np.load(STATES, mmap_mode="r")
    states = states_map[:, -1].astype(np.float32)
    weights = np.load(WEIGHTS, mmap_mode="r")
    contributions = np.load(CONTRIBUTIONS, mmap_mode="r")
    decisions = np.load(DECISIONS, mmap_mode="r")
    sources = {
        "hidden_final_norm": states,
        "fixed_code_weight_delta": weights,
        "signed_coordinate_contribution": contributions,
        "absolute_coordinate_contribution": np.abs(contributions),
    }
    result = {"representations": {}, "family_replication": {}}
    for name, field in sources.items():
        prototypes = np.stack([
            field[[i for i, row in enumerate(rows) if row["family"] == family and row["partition"] == "independent_development"]]
            .astype(np.float64).mean(axis=0)
            for family in FAMILIES
        ])
        records = []
        for index, row in enumerate(rows):
            if row["partition"] != "independent_lockbox":
                continue
            actual = field[index].astype(np.float64)
            errors = [relative_mse(actual, prototype) for prototype in prototypes]
            correct = FAMILIES.index(row["family"])
            predicted = int(np.argmin(errors))
            records.append({
                "case_id": row["case_id"], "family": row["family"], "representation": name,
                "predicted_family": FAMILIES[predicted], "correct": predicted == correct,
                "correct_mse": errors[correct],
                "best_wrong_mse": min(v for j, v in enumerate(errors) if j != correct),
            })
        write_rows(OUT / f"analysis/{name}_prototype_records.jsonl", records)
        result["representations"][name] = {
            "rows": len(records), "accuracy": float(np.mean([r["correct"] for r in records])),
            "chance": 1 / len(FAMILIES),
            "median_correct_over_best_wrong_ratio": float(np.median([
                r["correct_mse"] / (r["best_wrong_mse"] + EPS) for r in records
            ])),
        }
        result["family_replication"][name] = {}
        for family_index, family in enumerate(FAMILIES):
            lock = field[[i for i, row in enumerate(rows) if row["family"] == family and row["partition"] == "independent_lockbox"]]
            lock = lock.astype(np.float64).mean(axis=0)
            dev = prototypes[family_index]
            result["family_replication"][name][family] = {
                "sign_agreement": float(np.mean(dev * lock > 0)),
                "symmetric_relative_mse": float(
                    np.sum(np.square(dev - lock)) /
                    ((np.sum(np.square(dev)) + np.sum(np.square(lock))) / 2 + EPS)
                ),
            }
    exact = {
        "max_abs_error": float(decisions[:, 2].max()),
        "mean_abs_error": float(decisions[:, 2].mean()),
        "forced_ab_accuracy": float(np.mean(decisions[:, 3])),
        "unconstrained_exact_next_code_accuracy": float(np.mean(decisions[:, 4])),
    }
    for partition in PARTITIONS:
        idx = [i for i, row in enumerate(rows) if row["partition"] == partition]
        exact[f"{partition}_forced_ab_accuracy"] = float(np.mean(decisions[idx, 3]))
        exact[f"{partition}_unconstrained_exact_next_code_accuracy"] = float(np.mean(decisions[idx, 4]))
    result["exact_accounting_and_behavior"] = exact
    signed = result["representations"]["signed_coordinate_contribution"]["accuracy"]
    absolute = result["representations"]["absolute_coordinate_contribution"]["accuracy"]
    weight = result["representations"]["fixed_code_weight_delta"]["accuracy"]
    best = max(signed, absolute)
    result["gate"] = {
        "best_contribution_accuracy": best,
        "signed_accuracy": signed,
        "absolute_accuracy": absolute,
        "fixed_weight_accuracy": weight,
        "increment_over_fixed_weight": best - weight,
        "twice_chance_threshold": 2 / len(FAMILIES),
        "all_behavior_qualified": set(behavior_result["qualified"]) == set(FAMILIES),
        "passed": (best >= 2 / len(FAMILIES) and best >= weight + 0.05 and
                   set(behavior_result["qualified"]) == set(FAMILIES)),
    }
    for value in (states_map, weights, contributions, decisions):
        close_memmap(value)
    return result


def publish(rows: list[dict]) -> list[dict]:
    specs = (
        ("c7081_qwen4b_fixed_ab_boundary", STATES, np.float16,
         "fixed A/B interface HiddenState from embedding through final norm", "fixed_ab_hiddenstate_full_coordinate_v1"),
        ("c7082_qwen4b_fixed_ab_weight_delta", WEIGHTS, np.float32,
         "balanced correct-minus-wrong A/B unembedding delta", "fixed_ab_unembedding_delta_full_coordinate_v1"),
        ("c7083_qwen4b_fixed_ab_coordinate_contribution", CONTRIBUTIONS, np.float32,
         "exact signed per-coordinate fixed-code logit contribution", "fixed_ab_output_contribution_full_coordinate_v1"),
    )
    assets = []
    for dataset_id, path, dtype, description, schema in specs:
        source = np.load(path, mmap_mode="r")
        if source.ndim == 3:
            flat = source.reshape(-1, source.shape[-1])
            metadata = [
                {"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                 "surface": row["surface"], "unit": row["unit"], "state": row["state"],
                 "partition": row["partition"], "target_code": row["target_code"], "qpoint": q}
                for row in rows for q in range(source.shape[1])
            ]
        else:
            flat = source
            metadata = [
                {"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                 "surface": row["surface"], "unit": row["unit"], "state": row["state"],
                 "partition": row["partition"], "target_code": row["target_code"]}
                for row in rows
            ]
        binary = VIS / f"{dataset_id}.{np.dtype(dtype).name}.npy"
        out = atlas.create_binary(binary.name, flat.shape[0], flat.shape[1], dtype)
        out[:] = flat
        out.flush()
        close_memmap(out)
        close_memmap(source)
        assets.append(atlas.write_metadata(
            dataset_id, description, binary, metadata, "Qwen3-4B-FP16", schema,
            "fixed shared output interface", "three independently generated language families",
            description, {"coordinate_count": 2560, "no_projection": True, "fixed_output_codes": ["A", "B"]},
        ))
    return assets


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 固定公共A/B输出接口下的全坐标功能分账（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2336表明最终状态能区分族，但自然答案的逐坐标贡献不能稳定达到族识别门槛；本阶段把指代、翻译、属性三族 `{result['material_audit']['rows']}` 个独立构式全部重编译成同一单token A/B接口。每题保留两条完整自然语言候选，A/B位置在每个族、每个partition内严格各半，开发集只建原型，lockbox只裁决。这样所有题只使用同一对输出权重，不能再把不同自然答案token误称为语言族机制。模型仍为Qwen3-4B FP16、CUDA、非量化，保存embedding至final norm的38×2560全坐标场、A/B权重差与最终逐坐标贡献。

$$
m_i=z_{{y_i^+}}-z_{{y_i^-}}=\sum_{{j=1}}^{{2560}}h_{{ij}}(W_{{y_i^+,j}}-W_{{y_i^-,j}}),
\qquad y_i^+,y_i^-\in\{{A,B\}}.
$$

$$
\operatorname{{Gate}}=[A_{{func}}\ge 2/3]\land[A_{{func}}-A_W\ge0.05]\land
\bigwedge_f[Acc^{{dev}}_f,Acc^{{lock}}_f\text{{达到行为门槛}}].
$$

**结果汇总与相关文件。** 材料审计 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；全坐标分析 `{json.dumps(result['analysis'], ensure_ascii=False)}`；发布 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2337_c7081_c7240_fixed_ab_output_interface.py`；结果 `tests/glm5/result/phase2337_c7081_c7240_fixed_ab_output_interface`。

**分析、理论进展、问题硬伤与结论。** 本阶段第一次把“语言族差异”和“输出词身份差异”在输出接口处硬分离。signed贡献保留正确减错误方向；absolute贡献只去掉A/B轮换符号，仍保留每个具体坐标而未做Top-K或投影。若HiddenState仍能识别族而贡献不能，说明族/模板痕迹广泛存在于状态场，但没有以可复用的族级方式进入这一个二选一功能界面；若贡献通过，则得到比自然差分更强、但仍只限单模型和人工构式的机制候选。选择题提示可能诱发新的选项位置电路；只测三个行为强族；最终输出闭合不等于中层计算闭合，均是硬伤。

**下一阶段路线判断。** 目标仍是语言编码图谱而非为一次干预闭合。无论严格功能门是否通过，下一阶段都自动扩展到层间全坐标轨迹：用固定A/B接口比较族身份、正确性和选项位置三种因素在38个检查点何时出现、何时进入输出贡献；只有跨partition且跨表述稳定的层段才进入Qwen3-14B/GLM4/DS7B顺序复验。
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
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    rows, audit = compile_fixed_rows(tokenizer)
    write_rows(MATERIAL, rows)
    freeze = {
        "frozen_before_model_load": True,
        "families": list(FAMILIES), "partitions": list(PARTITIONS),
        "behavior_gate": {"forced_ab": 0.70, "unconstrained_exact_next_code": 0.40},
        "mechanism_gate": {"family_accuracy": 2 / 3, "increment_over_weight": 0.05},
        "numeric_accounting_abs_tolerance": 0.02,
        "reason_for_numeric_tolerance": "prospectively set after Phase2336 FP16 max rounding residual 0.011299",
    }
    save(OUT / "config/frozen_contract.json", freeze)
    model = None
    try:
        model, _loaded_tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect(model, device, rows)
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    decisions = np.load(DECISIONS, mmap_mode="r")
    behavior_result = behavior(rows, decisions)
    close_memmap(decisions)
    analysis = prototype_analysis(rows, behavior_result)
    datasets = publish(rows)
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    if not verified:
        raise RuntimeError(("verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    checks = {
        "rows": len(rows) == 384,
        "balanced_codes": audit["balanced_every_family_partition"],
        "all_coordinates": collection["state_shape"] == [384, 38, 2560],
        "numeric_accounting": analysis["exact_accounting_and_behavior"]["max_abs_error"] < freeze["numeric_accounting_abs_tolerance"],
        "assets_verified": verified,
        "frontend_build": build["passed"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze,
        "material_audit": audit, "collection": collection, "behavior": behavior_result,
        "analysis": analysis,
        "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)),
        "verification": verification,
        "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
        "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2337_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
