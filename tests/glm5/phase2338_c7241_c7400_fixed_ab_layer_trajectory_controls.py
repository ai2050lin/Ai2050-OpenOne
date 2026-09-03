#!/usr/bin/env python3
"""All-checkpoint fixed A/B coordinate-use trajectories with a paired option-swap control."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2335 = RESULT / "phase2335_c6761_c6920_independent_construction_replication"
P2337 = RESULT / "phase2337_c7081_c7240_fixed_ab_output_interface"
OUT = RESULT / "phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
SOURCE = P2335 / "material/independent_constructions.jsonl"
ORIGINAL_ROWS = P2337 / "material/fixed_ab_rows.jsonl"
ORIGINAL_STATES = P2337 / "raw/fixed_ab_boundary.float16.npy"
ORIGINAL_WEIGHTS = P2337 / "raw/fixed_ab_weight_delta.float32.npy"
ORIGINAL_FINAL_CONTRIB = P2337 / "raw/fixed_ab_coordinate_contribution.float32.npy"
SWAP_ROWS = OUT / "material/option_swapped_rows.jsonl"
SWAP_STATES = OUT / "raw/option_swapped_boundary.float16.npy"
SWAP_FINAL_CONTRIB = OUT / "raw/option_swapped_final_contribution.float32.npy"
SWAP_DECISIONS = OUT / "raw/option_swapped_decisions.float32.npy"
SWAP_PROGRESS = OUT / "raw/swap_progress.json"
ORIGINAL_TRAJECTORY = OUT / "derived/original_logit_lens_contribution.float32.npy"
SWAP_TRAJECTORY = OUT / "derived/swapped_logit_lens_contribution.float32.npy"
PHASE = 2338
CAMPAIGN = "C7241-C7400"
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


def compile_swapped(tokenizer) -> tuple[list[dict], dict]:
    originals = read_rows(ORIGINAL_ROWS)
    natural = {row["case_id"]: row for row in read_rows(SOURCE)}
    output = []
    for row in originals:
        source = natural[row["source_case_id"]]
        target_code = "B" if row["target_code"] == "A" else "A"
        wrong_code = "A" if target_code == "B" else "B"
        choice_a, choice_b = row["choice_b_text"], row["choice_a_text"]
        if row["language"] == "en":
            prompt = (f'{source["future_prompt"]}\nOption A: {choice_a}\nOption B: {choice_b}\n'
                      "Select the factually correct continuation. Reply with exactly A or B.\nAnswer:")
        else:
            prompt = (f'{source["future_prompt"]}\n选项A：{choice_a}\n选项B：{choice_b}\n'
                      "选择符合事实的续写，只回答A或B。\n答案：")
        target_ids = tokenizer.encode(" " + target_code, add_special_tokens=False)
        wrong_ids = tokenizer.encode(" " + wrong_code, add_special_tokens=False)
        if len(target_ids) != 1 or len(wrong_ids) != 1:
            raise RuntimeError(("code_tokenization", target_ids, wrong_ids))
        swapped = {
            **row,
            "case_id": row["case_id"].replace("c7081-", "c7241-"),
            "paired_original_case_id": row["case_id"],
            "choice_a_text": choice_a, "choice_b_text": choice_b,
            "target_code": target_code, "wrong_code": wrong_code,
            "target_code_id": int(target_ids[0]), "wrong_code_id": int(wrong_ids[0]),
            "future_prompt": prompt,
            "future_prompt_ids": [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)],
            "future_target_text": " " + target_code, "future_wrong_text": " " + wrong_code,
            "future_target_ids": [int(target_ids[0])], "future_wrong_ids": [int(wrong_ids[0])],
            "identity_target": target_code, "identity_wrong": wrong_code,
            "cue_id": "fixed_ab_option_swapped",
        }
        swapped["boundary_position"] = len(swapped["future_prompt_ids"]) - 1
        output.append(swapped)
    audit = {
        "rows": len(output), "paired_one_to_one": len(output) == len(originals),
        "every_target_code_flipped": all(a["target_code"] != b["target_code"] for a, b in zip(originals, output)),
        "every_option_order_reversed": all(
            a["choice_a_text"] == b["choice_b_text"] and a["choice_b_text"] == b["choice_a_text"]
            for a, b in zip(originals, output)
        ),
        "semantic_answer_preserved": all(
            (b["choice_a_text"] if b["target_code"] == "A" else b["choice_b_text"]) == a["natural_target_text"]
            for a, b in zip(originals, output)
        ),
    }
    return output, audit


def collect_swap(model, device, rows: list[dict], batch_size: int = 12) -> dict:
    module_list = modules(model)
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(module_list), dimension)
    if all(path.exists() for path in (SWAP_STATES, SWAP_FINAL_CONTRIB, SWAP_DECISIONS, SWAP_PROGRESS)):
        completed = int(json.loads(SWAP_PROGRESS.read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(SWAP_STATES, mode="r+")
        contributions = np.lib.format.open_memmap(SWAP_FINAL_CONTRIB, mode="r+")
        decisions = np.lib.format.open_memmap(SWAP_DECISIONS, mode="r+")
    else:
        completed = 0
        SWAP_STATES.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(SWAP_STATES, mode="w+", dtype=np.float16, shape=shape)
        contributions = np.lib.format.open_memmap(SWAP_FINAL_CONTRIB, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        decisions = np.lib.format.open_memmap(SWAP_DECISIONS, mode="w+", dtype=np.float32, shape=(len(rows), 5))
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
                ids, mask, positions = baseline.pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    states[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                for local, row in enumerate(batch):
                    target_id, wrong_id = row["target_code_id"], row["wrong_code_id"]
                    h = captures[len(module_list) - 1][local, ends[local]].float()
                    w = model.lm_head.weight[target_id].float() - model.lm_head.weight[wrong_id].float()
                    contribution = h * w
                    logits = output.logits[local, ends[local]].float()
                    margin = logits[target_id] - logits[wrong_id]
                    reconstructed = contribution.sum()
                    contributions[start + local] = contribution.cpu().numpy().astype(np.float32)
                    decisions[start + local] = [
                        float(margin.item()), float(reconstructed.item()),
                        float(abs(margin.item() - reconstructed.item())),
                        float(margin.item() > 0), float(int(torch.argmax(logits).item()) == target_id),
                    ]
                for value in (states, contributions, decisions):
                    value.flush()
                save(SWAP_PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                print(f"[phase2338 swap] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        for value in (states, contributions, decisions):
            value.flush()
            close_memmap(value)
    return {"state_shape": list(shape), "contribution_shape": [len(rows), dimension], "decision_shape": [len(rows), 5]}


def final_norm_parameters() -> tuple[np.ndarray, float, dict]:
    model_path = Path(model_utils.MODEL_CONFIGS["qwen3"]["path"])
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    key = "model.norm.weight"
    shard = index["weight_map"][key]
    with safe_open(str(model_path / shard), framework="pt", device="cpu") as handle:
        gamma = handle.get_tensor(key).float().numpy()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    eps = float(config.get("rms_norm_eps", 1e-6))
    return gamma, eps, {"tensor_key": key, "shard": shard, "shape": list(gamma.shape), "rms_norm_eps": eps}


def rms_norm(field: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    values = field.astype(np.float32)
    return values * np.reciprocal(np.sqrt(np.mean(np.square(values), axis=1, keepdims=True) + eps)) * gamma[None, :]


def derive_trajectories(gamma: np.ndarray, eps: float) -> dict:
    original_states = np.load(ORIGINAL_STATES, mmap_mode="r")
    swapped_states = np.load(SWAP_STATES, mmap_mode="r")
    original_weights = np.load(ORIGINAL_WEIGHTS, mmap_mode="r")
    original_final = np.load(ORIGINAL_FINAL_CONTRIB, mmap_mode="r")
    swapped_final = np.load(SWAP_FINAL_CONTRIB, mmap_mode="r")
    shape = original_states.shape
    ORIGINAL_TRAJECTORY.parent.mkdir(parents=True, exist_ok=True)
    if not ORIGINAL_TRAJECTORY.exists() or not SWAP_TRAJECTORY.exists():
        original_out = np.lib.format.open_memmap(ORIGINAL_TRAJECTORY, mode="w+", dtype=np.float32, shape=shape)
        swapped_out = np.lib.format.open_memmap(SWAP_TRAJECTORY, mode="w+", dtype=np.float32, shape=shape)
        for q in range(shape[1] - 1):
            original_out[:, q] = rms_norm(original_states[:, q], gamma, eps) * original_weights
            swapped_out[:, q] = rms_norm(swapped_states[:, q], gamma, eps) * (-original_weights)
            original_out.flush(); swapped_out.flush()
            print(f"[phase2338 derive] q={q}/37", flush=True)
        original_out[:, -1] = original_final
        swapped_out[:, -1] = swapped_final
        original_out.flush(); swapped_out.flush()
        close_memmap(original_out); close_memmap(swapped_out)
    original_trajectory = np.load(ORIGINAL_TRAJECTORY, mmap_mode="r")
    swapped_trajectory = np.load(SWAP_TRAJECTORY, mmap_mode="r")
    final_errors = {
        "original_final_copy_max_abs_error": float(np.max(np.abs(original_trajectory[:, -1] - original_final))),
        "swapped_final_copy_max_abs_error": float(np.max(np.abs(swapped_trajectory[:, -1] - swapped_final))),
    }
    for value in (original_states, swapped_states, original_weights, original_final, swapped_final, original_trajectory, swapped_trajectory):
        close_memmap(value)
    return {"shape": list(shape), **final_errors}


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.square(actual - predicted, dtype=np.float64)) /
                 (np.sum(np.square(actual, dtype=np.float64)) + EPS))


def classify(train: np.ndarray, test: np.ndarray, train_rows: list[dict], test_rows: list[dict], label: str, labels: tuple[str, ...]) -> dict:
    prototypes = np.stack([
        train[[i for i, row in enumerate(train_rows) if row["partition"] == "independent_development" and row[label] == value]]
        .astype(np.float64).mean(axis=0)
        for value in labels
    ])
    records = []
    for index, row in enumerate(test_rows):
        if row["partition"] != "independent_lockbox":
            continue
        errors = [relative_mse(test[index].astype(np.float64), prototype) for prototype in prototypes]
        correct = labels.index(row[label])
        predicted = int(np.argmin(errors))
        records.append((predicted == correct, errors[correct], min(v for j, v in enumerate(errors) if j != correct)))
    return {
        "accuracy": float(np.mean([r[0] for r in records])),
        "median_correct_over_best_wrong_ratio": float(np.median([r[1] / (r[2] + EPS) for r in records])),
        "rows": len(records), "chance": 1 / len(labels),
    }


def pair_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    l = left.astype(np.float64)
    r = right.astype(np.float64)
    numerator = np.sum(np.square(l - r), axis=1)
    denominator = (np.sum(np.square(l), axis=1) + np.sum(np.square(r), axis=1)) / 2 + EPS
    cosine = np.sum(l * r, axis=1) / (np.sqrt(np.sum(np.square(l), axis=1) * np.sum(np.square(r), axis=1)) + EPS)
    return {"median_symmetric_relative_mse": float(np.median(numerator / denominator)), "median_cosine": float(np.median(cosine))}


def analyze(original_rows: list[dict], swapped_rows: list[dict]) -> dict:
    original_states = np.load(ORIGINAL_STATES, mmap_mode="r")
    swapped_states = np.load(SWAP_STATES, mmap_mode="r")
    original_contrib = np.load(ORIGINAL_TRAJECTORY, mmap_mode="r")
    swapped_contrib = np.load(SWAP_TRAJECTORY, mmap_mode="r")
    rows = []
    for q in range(original_states.shape[1]):
        os = original_states[:, q].astype(np.float32)
        ss = swapped_states[:, q].astype(np.float32)
        oc = original_contrib[:, q].astype(np.float32)
        sc = swapped_contrib[:, q].astype(np.float32)
        oa, sa = np.abs(oc), np.abs(sc)
        record = {
            "qpoint": q,
            "checkpoint_kind": "embedding" if q == 0 else "final_norm" if q == original_states.shape[1] - 1 else "block",
            "original_hidden_family": classify(os, os, original_rows, original_rows, "family", FAMILIES),
            "swapped_hidden_family": classify(ss, ss, swapped_rows, swapped_rows, "family", FAMILIES),
            "original_to_swapped_hidden_family": classify(os, ss, original_rows, swapped_rows, "family", FAMILIES),
            "swapped_to_original_hidden_family": classify(ss, os, swapped_rows, original_rows, "family", FAMILIES),
            "original_abs_contribution_family": classify(oa, oa, original_rows, original_rows, "family", FAMILIES),
            "swapped_abs_contribution_family": classify(sa, sa, swapped_rows, swapped_rows, "family", FAMILIES),
            "original_to_swapped_abs_family": classify(oa, sa, original_rows, swapped_rows, "family", FAMILIES),
            "swapped_to_original_abs_family": classify(sa, oa, swapped_rows, original_rows, "family", FAMILIES),
            "original_signed_target_code": classify(oc, oc, original_rows, original_rows, "target_code", ("A", "B")),
            "swapped_signed_target_code": classify(sc, sc, swapped_rows, swapped_rows, "target_code", ("A", "B")),
            "paired_absolute_contribution": pair_metrics(oa, sa),
        }
        rows.append(record)
        print(f"[phase2338 analyze] q={q}/37", flush=True)
    write_rows(OUT / "analysis/checkpoint_trajectory.jsonl", rows)
    qualifying = [
        row["qpoint"] for row in rows
        if min(row["original_to_swapped_abs_family"]["accuracy"], row["swapped_to_original_abs_family"]["accuracy"]) >= 2 / 3
    ]
    sustained = None
    for q in qualifying:
        if all((q + offset) in qualifying for offset in range(3)):
            sustained = q
            break
    final = rows[-1]
    return {
        "checkpoint_rows": len(rows), "qualifying_cross_swap_qpoints": qualifying,
        "first_three_checkpoint_sustained_onset": sustained,
        "peak": {
            key: max(rows, key=lambda row: row[key]["accuracy"])[key] | {
                "qpoint": max(rows, key=lambda row: row[key]["accuracy"])["qpoint"]
            }
            for key in (
                "original_hidden_family", "swapped_hidden_family",
                "original_abs_contribution_family", "swapped_abs_contribution_family",
                "original_to_swapped_abs_family", "swapped_to_original_abs_family",
            )
        },
        "final_checkpoint": final,
        "gate": {
            "threshold_cross_swap_accuracy": 2 / 3,
            "threshold_pair_symmetric_mse": 0.50,
            "both_cross_swap_family_accuracies_pass": min(
                final["original_to_swapped_abs_family"]["accuracy"],
                final["swapped_to_original_abs_family"]["accuracy"],
            ) >= 2 / 3,
            "paired_absolute_geometry_pass": final["paired_absolute_contribution"]["median_symmetric_relative_mse"] <= 0.50,
        },
    }


def behavior(rows: list[dict]) -> dict:
    decisions = np.load(SWAP_DECISIONS, mmap_mode="r")
    output = {"families": {}, "all_cells_forced_ab_pass": True,
              "overall_forced_ab_accuracy": float(np.mean(decisions[:, 3])),
              "overall_unconstrained_exact_next_code_accuracy": float(np.mean(decisions[:, 4])),
              "max_accounting_abs_error": float(np.max(decisions[:, 2]))}
    for family in FAMILIES:
        output["families"][family] = {}
        for partition in PARTITIONS:
            idx = [i for i, row in enumerate(rows) if row["family"] == family and row["partition"] == partition]
            cell = {"rows": len(idx), "forced_ab_accuracy": float(np.mean(decisions[idx, 3])),
                    "unconstrained_exact_next_code_accuracy": float(np.mean(decisions[idx, 4]))}
            cell["forced_ab_pass"] = cell["forced_ab_accuracy"] >= 0.70
            output["all_cells_forced_ab_pass"] = output["all_cells_forced_ab_pass"] and cell["forced_ab_pass"]
            output["families"][family][partition] = cell
    close_memmap(decisions)
    return output


def publish(rows: list[dict]) -> list[dict]:
    specs = (
        ("c7241_qwen4b_fixed_ab_original_logit_lens_contribution", ORIGINAL_TRAJECTORY, np.float32,
         "original fixed-A/B RMS-normalized layerwise coordinate contributions", "fixed_ab_layerwise_coordinate_contribution_v1", False),
        ("c7242_qwen4b_fixed_ab_option_swapped_boundary", SWAP_STATES, np.float16,
         "paired option-swapped HiddenState from embedding through final norm", "fixed_ab_option_swap_hiddenstate_v1", True),
        ("c7243_qwen4b_fixed_ab_option_swapped_logit_lens_contribution", SWAP_TRAJECTORY, np.float32,
         "option-swapped RMS-normalized layerwise coordinate contributions", "fixed_ab_option_swap_layerwise_contribution_v1", True),
    )
    assets = []
    for dataset_id, path, dtype, description, schema, swapped in specs:
        source = np.load(path, mmap_mode="r")
        flat = source.reshape(-1, source.shape[-1])
        metadata = [
            {"case_id": row["case_id"], "family": row["family"], "language": row["language"],
             "surface": row["surface"], "unit": row["unit"], "state": row["state"],
             "partition": row["partition"], "target_code": row["target_code"], "qpoint": q,
             "option_swapped": swapped}
            for row in rows for q in range(source.shape[1])
        ]
        binary = VIS / f"{dataset_id}.{np.dtype(dtype).name}.npy"
        out = atlas.create_binary(binary.name, flat.shape[0], flat.shape[1], dtype)
        out[:] = flat
        out.flush(); close_memmap(out); close_memmap(source)
        assets.append(atlas.write_metadata(
            dataset_id, description, binary, metadata, "Qwen3-4B-FP16", schema,
            "paired option-order control", "three independent families, 384 prompts, 38 checkpoints",
            description, {"coordinate_count": 2560, "no_projection": True, "rms_logit_lens": "q0-q36", "exact_final": "q37"},
        ))
    return assets


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 固定A/B逐层坐标使用轨迹与配对选项交换控制（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 针对Phase2337绝对逐坐标贡献识别三族的强信号，本阶段不立即宣称机制，而做更严格的配对控制：对原384题逐题交换A/B两条自然候选，正确自然答案不变、正确代码必翻转；Qwen3-4B FP16在CUDA上重新前向，保存38×2560全场。对embedding和36个block输出先施加模型真实最终RMSNorm参数，再与当前正确减错误A/B输出权重逐坐标相乘；q37直接使用真实最终贡献。所有层均保留全部坐标，不做Top-K/PCA。

$$
\widetilde h_{{i,q,j}}=\gamma_j\frac{{h_{{i,q,j}}}}{{\sqrt{{2560^{{-1}}\sum_k h_{{i,q,k}}^2+\epsilon}}}},
\qquad c_{{i,q,j}}=\widetilde h_{{i,q,j}}(W_{{y_i^+,j}}-W_{{y_i^-,j}}).
$$

$$
A^{{cross}}_q=\operatorname{{ProtoAcc}}\left(|c^{{original}}_{{dev,q}}|\to|c^{{swap}}_{{lock,q}}|\right),
\quad E^{{pair}}_q=\operatorname{{median}}_i\frac{{\| |c^o_{{i,q}}|-|c^s_{{i,q}}|\|^2}}{{(\|c^o_{{i,q}}\|^2+\|c^s_{{i,q}}\|^2)/2}}.
$$

**结果汇总与相关文件。** 交换审计 `{json.dumps(result['swap_audit'], ensure_ascii=False)}`；交换行为 `{json.dumps(result['swap_behavior'], ensure_ascii=False)}`；RMSNorm参数 `{json.dumps(result['norm_parameters'], ensure_ascii=False)}`；派生核验 `{json.dumps(result['derived'], ensure_ascii=False)}`；逐层分析 `{json.dumps(result['analysis'], ensure_ascii=False)}`；发布 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls.py`；结果 `tests/glm5/result/phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls`。

**分析、理论进展、问题硬伤与结论。** 这不是把中层状态当成真实当层logit：q0–q36的最终RMSNorm读出只是统一坐标尺，只有q37是模型真实输出分账。选项交换能排除固定A/B位置和符号映射，却不能排除两种提示都共享的模板、词汇长度与三族材料差异。绝对值描述“哪些坐标被用得强”，不描述促进A还是B；若跨交换仍稳定，它是可复用坐标使用图谱候选，不是语义原子或因果齿轮。跨层首次出现点只能作为轨迹定位，不能据此说该层首次计算了语义。

**下一阶段路线判断。** 若最终与连续层段的跨交换门通过，目标相同，自动把完全相同材料和冻结相对深度送到Qwen3-14B，再按显存规则顺序测试GLM4与DS7B，比较坐标编号不可比时的族×相对层×分布形状规律；若失败，则返回20族扩展固定接口而不做跨模型因果。两条路线都继续以观察图谱优先。
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
    parent = json.loads((P2337 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["analysis"]["gate"]["passed"]:
        raise RuntimeError("Phase2337 scientific gate did not pass")
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    original_rows = read_rows(ORIGINAL_ROWS)
    swapped_rows, swap_audit = compile_swapped(tokenizer)
    write_rows(SWAP_ROWS, swapped_rows)
    freeze = {
        "frozen_before_swap_model_load": True,
        "families": list(FAMILIES), "partitions": list(PARTITIONS),
        "cross_swap_family_accuracy": 2 / 3,
        "paired_absolute_symmetric_mse_max": 0.50,
        "forced_ab_behavior_each_cell": 0.70,
        "sustained_onset_width": 3,
        "q0_to_q36_are_rms_normalized_readouts_not_native_logits": True,
    }
    save(OUT / "config/frozen_contract.json", freeze)
    model = None
    try:
        model, _tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect_swap(model, device, swapped_rows)
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gamma, eps, norm_record = final_norm_parameters()
    derived = derive_trajectories(gamma, eps)
    swap_behavior = behavior(swapped_rows)
    analysis = analyze(original_rows, swapped_rows)
    analysis["gate"]["all_swap_behavior_cells_pass"] = swap_behavior["all_cells_forced_ab_pass"]
    analysis["gate"]["passed"] = all((
        analysis["gate"]["both_cross_swap_family_accuracies_pass"],
        analysis["gate"]["paired_absolute_geometry_pass"],
        analysis["gate"]["all_swap_behavior_cells_pass"],
        analysis["first_three_checkpoint_sustained_onset"] is not None,
    ))
    datasets = publish(swapped_rows)
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    if not verified:
        raise RuntimeError(("verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    checks = {
        "parent_scientific_gate": parent["analysis"]["gate"]["passed"],
        "paired_material": all(swap_audit.values()),
        "all_coordinates": collection["state_shape"] == [384, 38, 2560] and derived["shape"] == [384, 38, 2560],
        "final_copy_exact": derived["original_final_copy_max_abs_error"] == 0 and derived["swapped_final_copy_max_abs_error"] == 0,
        "assets_verified": verified, "frontend_build": build["passed"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze,
        "swap_audit": swap_audit, "collection": collection, "swap_behavior": swap_behavior,
        "norm_parameters": norm_record, "derived": derived, "analysis": analysis,
        "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)),
        "verification": verification, "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
        "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2338_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
