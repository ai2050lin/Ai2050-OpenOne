#!/usr/bin/env python3
"""Validate full-coordinate local response directions on qualified language families."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2309 = RESULT / "phase2309_c4321_c4440_multistep_future_contract"
P2310 = RESULT / "phase2310_c4441_c4580_qwen4b_multistep_field"
P2311 = RESULT / "phase2311_c4581_c4700_basic_future_accounting"
OUT = RESULT / "phase2312_c4701_c4820_qwen4b_local_response"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2309 / "material/multistep_future_bilingual.jsonl"
CONFIG_PATH = OUT / "config/frozen_local_response.json"
ACTIVATIONS = OUT / "atlas/selected_boundary_activations.float16.npy"
GRADIENTS = OUT / "atlas/fixed_margin_gradients.float16.npy"
ROW_INDEX = OUT / "index/selected_rows.jsonl"
PERTURBATIONS = OUT / "causal/structured_perturbations.jsonl"
PROGRESS = OUT / "raw/progress.json"
sys.path.insert(0, str(TESTS))

import phase1332_bf16_utils as model_base  # noqa: E402
import phase2309_c4321_c4440_multistep_future_contract as contract  # noqa: E402


PHASE = 2312
CAMPAIGN = "C4701-C4820"
SELECTED_UNITS = {
    "confirmation": (12, 13),
    "fresh_confirmation": (20, 21),
    "fresh_lockbox": (26, 27),
}
GRADIENT_QPOINTS = tuple(contract.QPOINTS_4B)
PERTURB_QPOINTS = (10, 20, 25, 30, 36)
DOSES = (0.01, 0.03)
DIRECTIONS = (
    "gradient_positive",
    "gradient_negative",
    "deterministic_rademacher",
    "coordinate_permuted_gradient",
)
EPS = 1e-12


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def checkpoint_modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def frozen_config() -> dict:
    sequence = json.loads((P2310 / "behavior/sequence_ledger.json").read_text(encoding="utf-8"))
    free = json.loads((P2310 / "behavior/free_ledger.json").read_text(encoding="utf-8"))
    eligible = sorted(set(sequence["qualified_families"]) & set(free["route_eligible_families"]))
    value = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "frozen_before_model_load": True,
        "source_phases": [2309, 2310, 2311],
        "eligible_families": eligible,
        "eligibility_rule": "intersection(complete_future_strict_gate, free_identity_gate)",
        "selected_units": {key: list(values) for key, values in SELECTED_UNITS.items()},
        "gradient_qpoints": list(GRADIENT_QPOINTS),
        "perturb_qpoints": list(PERTURB_QPOINTS),
        "doses_relative_to_hidden_l2": list(DOSES),
        "directions": list(DIRECTIONS),
        "fixed_margin": "state1_identity_first_token_minus_state0_identity_first_token",
        "confirmation_gates": {
            "gradient_sign_accuracy": contract.LOCAL_LINEAR_GATES["sign_accuracy"],
            "gradient_median_symmetric_relative_error": contract.LOCAL_LINEAR_GATES["median_relative_error"],
            "positive_above_negative_rate": contract.LOCAL_LINEAR_GATES["forward_reverse_order_rate"],
        },
        "fresh_policy": (
            "all selected fresh rows receive observational gradients; structured perturbation is restricted "
            "to family-checkpoint cells passing all confirmation gates"
        ),
        "coordinate_policy": "all_2560_original_coordinates_no_topk_no_projection",
        "claim_boundary": "local_fixed_margin_response_not_semantic_circuit_or_sufficient_state",
    }
    if CONFIG_PATH.exists():
        previous = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if previous != value:
            raise RuntimeError(("frozen_config_changed", previous, value))
    else:
        save(CONFIG_PATH, value)
    return value


def selected_rows(rows: list[dict], families: list[str]) -> list[dict]:
    selected = [row for row in rows
                if row["family"] in families
                and int(row["unit"]) in SELECTED_UNITS.get(row["partition"], ())]
    partition_order = {name: index for index, name in enumerate(SELECTED_UNITS)}
    selected.sort(key=lambda row: (
        partition_order[row["partition"]], row["family"], int(row["unit"]),
        row["language"], row["surface"], int(row["state"]),
    ))
    return selected


def fixed_tokens(row: dict) -> tuple[int, int]:
    if int(row["state"]) == 1:
        return int(row["future_identity_target_ids"][0]), int(row["future_identity_wrong_ids"][0])
    return int(row["future_identity_wrong_ids"][0]), int(row["future_identity_target_ids"][0])


def as_tensor(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def replace_tensor(value: Any, tensor: torch.Tensor) -> Any:
    if isinstance(value, tuple):
        return (tensor, *value[1:])
    return tensor


def capture_local_gradient(model, device, row: dict) -> tuple[np.ndarray, np.ndarray, float]:
    modules = checkpoint_modules(model)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q in GRADIENT_QPOINTS:
        module = modules[q]

        def hook(_module, _inputs, value, q=q):
            tensor = as_tensor(value)
            if q == 0:
                tensor = tensor.detach().requires_grad_(True)
                value = replace_tensor(value, tensor)
            captures[q] = tensor
            if not tensor.is_leaf:
                tensor.retain_grad()
            return value

        handles.append(module.register_forward_hook(hook))
    try:
        ids = torch.tensor([row["future_prompt_ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        result = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        positive, negative = fixed_tokens(row)
        margin = result.logits[0, -1, positive] - result.logits[0, -1, negative]
        margin.backward()
        activations, gradients = [], []
        for q in GRADIENT_QPOINTS:
            tensor = captures[q]
            if tensor.grad is None:
                raise RuntimeError(("missing_hidden_gradient", row["case_id"], q))
            activations.append(tensor[0, -1].detach().float().cpu().numpy())
            gradients.append(tensor.grad[0, -1].detach().float().cpu().numpy())
        return np.stack(activations), np.stack(gradients), float(margin.detach().item())
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        captures.clear()


def seeded_rng(case_id: str, q: int) -> np.random.Generator:
    raw = hashlib.sha256(f"{case_id}|{q}|phase2312".encode("utf-8")).digest()[:8]
    return np.random.default_rng(int.from_bytes(raw, "little"))


def perturbation_variants(row: dict, q: int, h: np.ndarray, g: np.ndarray) -> list[dict]:
    rng = seeded_rng(row["case_id"], q)
    dimension = len(g)
    g_norm = float(np.linalg.norm(g.astype(np.float64)))
    h_norm = float(np.linalg.norm(h.astype(np.float64)))
    if g_norm <= EPS or h_norm <= EPS:
        raise RuntimeError(("zero_norm_local_field", row["case_id"], q, h_norm, g_norm))
    random_direction = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dimension)
    random_direction /= np.linalg.norm(random_direction)
    permutation = rng.permutation(dimension)
    directions = {
        "gradient_positive": g / g_norm,
        "gradient_negative": -g / g_norm,
        "deterministic_rademacher": random_direction,
        "coordinate_permuted_gradient": g[permutation] / g_norm,
    }
    output = []
    for direction_name in DIRECTIONS:
        direction = directions[direction_name].astype(np.float32, copy=False)
        for dose in DOSES:
            delta = direction * (float(dose) * h_norm)
            output.append({
                "direction": direction_name,
                "dose": float(dose),
                "delta": delta,
                "predicted_delta": float(np.dot(g.astype(np.float64), delta.astype(np.float64))),
                "delta_l2": float(np.linalg.norm(delta.astype(np.float64))),
            })
    return output


def run_perturbation_batch(model, device, row: dict, q: int, variants: list[dict]) -> np.ndarray:
    modules = checkpoint_modules(model)
    deltas = torch.tensor(np.stack([value["delta"] for value in variants]),
                          dtype=torch.float32, device=device)

    def hook(_module, _inputs, value):
        tensor = as_tensor(value)
        changed = tensor.clone()
        changed[:, -1, :] = changed[:, -1, :] + deltas.to(dtype=changed.dtype)
        return replace_tensor(value, changed)

    handle = modules[q].register_forward_hook(hook)
    try:
        batch = len(variants)
        ids = torch.tensor([row["future_prompt_ids"]] * batch, dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
                           return_dict=True).logits[:, -1]
        positive, negative = fixed_tokens(row)
        return (logits[:, positive] - logits[:, negative]).float().cpu().numpy()
    finally:
        handle.remove()


def summarize_cells(records: list[dict], partitions: tuple[str, ...]) -> dict:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        if record["partition"] in partitions:
            grouped[(record["family"], int(record["checkpoint"]))].append(record)
    gates = {
        "sign_accuracy": contract.LOCAL_LINEAR_GATES["sign_accuracy"],
        "median_relative_error": contract.LOCAL_LINEAR_GATES["median_relative_error"],
        "positive_above_negative_rate": contract.LOCAL_LINEAR_GATES["forward_reverse_order_rate"],
    }
    cells = {}
    for (family, q), values in sorted(grouped.items()):
        gradients = [value for value in values if value["direction"].startswith("gradient_")]
        sign = [float(value["actual_delta"] * value["predicted_delta"] > 0) for value in gradients]
        relative = [abs(value["actual_delta"] - value["predicted_delta"])
                    / (abs(value["actual_delta"]) + abs(value["predicted_delta"]) + EPS)
                    for value in gradients]
        pairs: dict[tuple[str, float], dict[str, float]] = defaultdict(dict)
        for value in gradients:
            pairs[(value["case_id"], float(value["dose"]))][value["direction"]] = value["actual_delta"]
        ordering = [float(pair.get("gradient_positive", -math.inf)
                          > pair.get("gradient_negative", math.inf))
                    for pair in pairs.values()]
        controls = [abs(value["actual_delta"]) for value in values
                    if not value["direction"].startswith("gradient_")]
        grad_effects = [abs(value["actual_delta"]) for value in gradients]
        metric = {
            "family": family,
            "checkpoint": q,
            "rows": len(values),
            "gradient_rows": len(gradients),
            "gradient_sign_accuracy": float(np.mean(sign)),
            "gradient_median_symmetric_relative_error": float(np.median(relative)),
            "positive_above_negative_rate": float(np.mean(ordering)),
            "median_abs_gradient_effect": float(np.median(grad_effects)),
            "median_abs_control_effect": float(np.median(controls)),
        }
        metric["qualified"] = (
            metric["gradient_sign_accuracy"] >= gates["sign_accuracy"]
            and metric["gradient_median_symmetric_relative_error"] <= gates["median_relative_error"]
            and metric["positive_above_negative_rate"] >= gates["positive_above_negative_rate"]
        )
        cells[f"{family}@{q}"] = metric
    return {
        "partitions": list(partitions),
        "gates": gates,
        "cells": cells,
        "qualified_cells": [key for key, value in cells.items() if value["qualified"]],
    }


def initialize_arrays(count: int, dimension: int) -> tuple[np.memmap, np.memmap, int, list[dict]]:
    shape = (count, len(GRADIENT_QPOINTS), dimension)
    OUT.joinpath("atlas").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if progress["shape"] != list(shape):
            raise RuntimeError(("resume_shape", progress["shape"], shape))
        activation = np.lib.format.open_memmap(ACTIVATIONS, mode="r+")
        gradient = np.lib.format.open_memmap(GRADIENTS, mode="r+")
        records = read_rows(PERTURBATIONS) if PERTURBATIONS.exists() else []
        return activation, gradient, int(progress["completed"]), records
    activation = np.lib.format.open_memmap(ACTIVATIONS, mode="w+", dtype=np.float16, shape=shape)
    gradient = np.lib.format.open_memmap(GRADIENTS, mode="w+", dtype=np.float16, shape=shape)
    return activation, gradient, 0, []


def run_local_campaign(model, device, rows: list[dict], config: dict) -> tuple[dict, dict, dict]:
    dimension = int(model.config.hidden_size)
    activation_file, gradient_file, completed, records = initialize_arrays(len(rows), dimension)
    confirmation_count = sum(row["partition"] == "confirmation" for row in rows)
    confirmation_ledger_path = OUT / "analysis/confirmation_cells.json"
    eligible_cells: set[str] = set()
    try:
        for row_index in range(completed, len(rows)):
            if row_index == confirmation_count:
                confirmation = summarize_cells(records, ("confirmation",))
                save(confirmation_ledger_path, confirmation)
                eligible_cells = set(confirmation["qualified_cells"])
            row = rows[row_index]
            activations, gradients, baseline = capture_local_gradient(model, device, row)
            activation_file[row_index] = activations.astype(np.float16)
            gradient_file[row_index] = gradients.astype(np.float16)
            for q in PERTURB_QPOINTS:
                cell = f"{row['family']}@{q}"
                if row["partition"] != "confirmation" and cell not in eligible_cells:
                    continue
                qi = GRADIENT_QPOINTS.index(q)
                variants = perturbation_variants(row, q, activations[qi], gradients[qi])
                actual_margins = run_perturbation_batch(model, device, row, q, variants)
                for variant, actual_margin in zip(variants, actual_margins):
                    records.append({
                        "case_id": row["case_id"],
                        "family": row["family"],
                        "partition": row["partition"],
                        "language": row["language"],
                        "surface": row["surface"],
                        "unit": int(row["unit"]),
                        "state": int(row["state"]),
                        "checkpoint": int(q),
                        "direction": variant["direction"],
                        "dose": variant["dose"],
                        "baseline_margin": baseline,
                        "actual_margin": float(actual_margin),
                        "actual_delta": float(actual_margin - baseline),
                        "predicted_delta": variant["predicted_delta"],
                        "delta_l2": variant["delta_l2"],
                    })
            activation_file.flush()
            gradient_file.flush()
            write_rows(PERTURBATIONS, records)
            save(PROGRESS, {"completed": row_index + 1,
                            "shape": [len(rows), len(GRADIENT_QPOINTS), dimension],
                            "config_hash": file_hash(CONFIG_PATH)})
            print(f"[phase2312] {row_index + 1}/{len(rows)} {row['case_id']}", flush=True)
        if not confirmation_ledger_path.exists():
            save(confirmation_ledger_path, summarize_cells(records, ("confirmation",)))
        confirmation = json.loads(confirmation_ledger_path.read_text(encoding="utf-8"))
        fresh = summarize_cells(records, ("fresh_confirmation", "fresh_lockbox"))
        save(OUT / "analysis/fresh_cells.json", fresh)
    finally:
        close_memmap(activation_file)
        close_memmap(gradient_file)
    gradient_audit = {
        "activation_path": str(ACTIVATIONS.relative_to(ROOT)),
        "gradient_path": str(GRADIENTS.relative_to(ROOT)),
        "shape": [len(rows), len(GRADIENT_QPOINTS), dimension],
        "all_coordinates": True,
        "qpoints": list(GRADIENT_QPOINTS),
        "rows": len(rows),
    }
    return gradient_audit, confirmation, fresh


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 合格语言族的全坐标局部响应与冻结扰动（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本阶段在模型加载前读取 Phase2310 的完整未来门和自由身份门，冻结其交集 `{result['config']['eligible_families']}`，没有根据梯度结果增删语言族。每族使用 confirmation、fresh-confirmation、fresh-lockbox 各两个 unit，覆盖中英、叙述/对话、两种事实状态。对当前样本、当前检查点和同一 unit 固定的“状态1身份首 token 减状态0身份首 token”边界，保存全部 2560 个物理坐标的激活和精确局部导数：
$$
m_i(h_q)=z_{{a_i}}-z_{{b_i}},\qquad
g_{{i,q,j}}=\frac{{\partial m_i}}{{\partial h_{{i,q,j}}}},\quad j=1,\ldots,2560.
$$
在检查点 `10,20,25,30,36` 按当前隐藏状态的 L2 范数施加 `0.01/0.03` 两档扰动。正梯度、负梯度用于预测，确定性 Rademacher 与坐标置换梯度是控制；不读取 Attention/MLP，不用 Top-K/PCA，也不搬运其他样本差分：
$$
\widehat{{\Delta m}}=g_{{i,q}}^\top\delta,
\qquad
e_{{sym}}=\frac{{|\Delta m-\widehat{{\Delta m}}|}}{{|\Delta m|+|\widehat{{\Delta m}}|+\epsilon}}.
$$

**结果汇总。** confirmation 裁决 `{json.dumps(result['confirmation'], ensure_ascii=False)}`；只有其中同时满足符号准确率不低于 `0.75`、中位对称相对误差不高于 `0.30`、正梯度效果高于负梯度比例不低于 `0.75` 的族×检查点单元，才执行 fresh 扰动。fresh 裁决 `{json.dumps(result['fresh'], ensure_ascii=False)}`。全坐标资产 `{json.dumps(result['gradient_audit'], ensure_ascii=False)}`；扰动账 `{result['perturbation_rows']}` 行；模型 `{json.dumps(result['model'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。本阶段检验的是冻结输出边界附近的局部可调用方向，不是语言语义的充分状态，也不是唯一物理电路。梯度由模型的当前输入、检查点和输出 token 对共同决定；通过只说明有限剂量下的一阶预测可复现，失败则说明该剂量/检查点的一阶近似不足。控制方向不能穷尽 2560 维空间；固定首 token 边界不等于完整多 token 自由生成；样本仍来自人工微世界，且只有 Qwen3-4B。脚本 `tests/glm5/phase2312_c4701_c4820_qwen4b_local_response.py`；结果 `tests/glm5/result/phase2312_c4701_c4820_qwen4b_local_response`。下一步不比较跨模型坐标编号，只在 fresh 材料上比较行为资格、形成相对深度和局部方向是否存在。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for parent in (P2309, P2310, P2311):
        final = json.loads((parent / "analysis/final.json").read_text(encoding="utf-8"))
        if not final["all_checks_passed"]:
            raise RuntimeError(("parent_not_authorized", parent.name))
    config = frozen_config()
    all_rows = read_rows(ROWS_PATH)
    rows = selected_rows(all_rows, config["eligible_families"])
    write_rows(ROW_INDEX, [{
        "row": index,
        "case_id": row["case_id"],
        "family": row["family"],
        "partition": row["partition"],
        "language": row["language"],
        "surface": row["surface"],
        "unit": int(row["unit"]),
        "state": int(row["state"]),
    } for index, row in enumerate(rows)])
    expected = len(config["eligible_families"]) * 3 * 2 * 2 * 2 * 2
    if len(rows) != expected:
        raise RuntimeError(("selected_row_count", len(rows), expected))
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        gradient_audit, confirmation, fresh = run_local_campaign(model, device, rows, config)
        model_info = {
            "name": "Qwen3-4B",
            "precision": "bfloat16",
            "quantization": "none",
            "placement": placement,
            "hidden_size": int(model.config.hidden_size),
            "layers": len(model.model.layers),
        }
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    perturbation_rows = len(read_rows(PERTURBATIONS))
    confirmation_cells = set(confirmation["qualified_cells"])
    fresh_cells = set(fresh["qualified_cells"])
    checks = {
        "config_frozen_before_model_load": config["frozen_before_model_load"],
        "eligibility_is_behavior_intersection": len(config["eligible_families"]) == 4,
        "all_selected_rows": len(rows) == expected,
        "all_gradient_coordinates": gradient_audit["shape"] == [len(rows), 10, 2560],
        "confirmation_all_family_qpoint_cells": len(confirmation["cells"]) == len(config["eligible_families"]) * len(PERTURB_QPOINTS),
        "fresh_only_confirmation_qualified_cells": set(fresh["cells"]).issubset(confirmation_cells),
        "fresh_qualification_is_subset_of_tested": fresh_cells.issubset(set(fresh["cells"])),
        "all_perturbation_rows_recorded": perturbation_rows > 0,
        "no_topk_pca_or_donor_difference": True,
        "no_attention_or_mlp_internal_read": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "config": config,
        "model": model_info,
        "gradient_audit": gradient_audit,
        "confirmation": confirmation,
        "fresh": fresh,
        "perturbation_path": str(PERTURBATIONS.relative_to(ROOT)),
        "perturbation_rows": perturbation_rows,
        "hashes": {
            "config": file_hash(CONFIG_PATH),
            "activations": file_hash(ACTIVATIONS),
            "gradients": file_hash(GRADIENTS),
            "perturbations": file_hash(PERTURBATIONS),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            f"Confirmation qualified {len(confirmation_cells)}/{len(config['eligible_families']) * len(PERTURB_QPOINTS)} "
            f"family-checkpoint local response cells; fresh data qualified {len(fresh_cells)}/{len(fresh['cells'])} "
            "tested cells. This supports only sample- and boundary-conditioned local first-order control where "
            "it replicates, not a fixed semantic direction, coordinate dictionary, hologram, or complete mechanism."
        ),
        "next_authorization": (
            "Run fresh-partition functional replication sequentially on Qwen3-14B and DeepSeek-7B, comparing "
            "relative formation depth and gates rather than coordinate identity."
        ),
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
