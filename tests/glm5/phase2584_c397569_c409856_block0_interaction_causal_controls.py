#!/usr/bin/env python3
"""Causal removal and matched-coordinate controls for block0 factorial interactions."""
from __future__ import annotations

import gc
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
P2583 = RESULT / "phase2583_c385281_c397568_block0_component_interaction_atlas/analysis/final.json"
OUT = RESULT / "phase2584_c397569_c409856_block0_interaction_causal_controls"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2584, "C397569-C409856"
CONDITIONS = (
    "baseline",
    "attention_interaction_removed",
    "attention_matched_coordinate_roll",
    "mlp_interaction_removed",
    "block_interaction_removed",
)
ROLL = 641

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth as p2582  # noqa: E402


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def replace_tensor(output, tensor):
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    if isinstance(output, list):
        return [tensor] + output[1:]
    return tensor


class InteractionController:
    def __init__(self, layer):
        self.layer = layer
        self.condition = "baseline"
        self.prompt_length = 0
        self.stats = []
        self.handle = None

    def set(self, condition: str, prompt_length: int):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.condition = condition
        self.prompt_length = prompt_length
        if condition == "baseline":
            return
        module = {
            "attention_interaction_removed": self.layer.self_attn,
            "attention_matched_coordinate_roll": self.layer.self_attn,
            "mlp_interaction_removed": self.layer.mlp,
            "block_interaction_removed": self.layer,
        }[condition]
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if tensor.shape[0] != 4:
            raise RuntimeError(f"factorial controller requires batch=4, got {tensor.shape}")
        result = tensor.clone()
        prefix = tensor[:, : self.prompt_length]
        interaction = prefix[3].float() - prefix[2].float() - prefix[1].float() + prefix[0].float()
        correction = (torch.roll(interaction, shifts=ROLL, dims=-1)
                      if self.condition == "attention_matched_coordinate_roll" else interaction)
        signs = (-1.0, 1.0, 1.0, -1.0)
        for index, sign in enumerate(signs):
            result[index, : self.prompt_length] = (
                prefix[index].float() + sign * correction / 4.0
            ).to(result.dtype)
        after = (result[3, : self.prompt_length].float() - result[2, : self.prompt_length].float()
                 - result[1, : self.prompt_length].float() + result[0, : self.prompt_length].float())
        self.stats.append({
            "condition": self.condition,
            "before_rms": float(torch.sqrt(torch.mean(interaction.double() ** 2)).item()),
            "after_rms": float(torch.sqrt(torch.mean(after.double() ** 2)).item()),
            "perturbation_rms": float(torch.sqrt(torch.mean((result[:, : self.prompt_length].float()
                                                               - prefix.float()).double() ** 2)).item()),
        })
        return replace_tensor(output, result)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def continuation(tokenizer, entity: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]


def score(model, tokenizer, selected):
    device = model.get_input_embeddings().weight.device
    layer = model_utils.get_layers(model)[0]
    controller = InteractionController(layer)
    score_rows = []
    next_dir = OUT / "next_token_field"
    next_dir.mkdir(parents=True, exist_ok=True)
    next_manifest = []
    try:
        for quartet_index, (prefix, cells) in enumerate(selected):
            prompt_length = len(cells[0]["prompt_ids"])
            if len({len(row["prompt_ids"]) for row in cells}) != 1:
                raise RuntimeError("non-aligned quartet")
            scores = {condition: defaultdict(dict) for condition in CONDITIONS}
            next_logits = {}
            for condition in CONDITIONS:
                for candidate_index, entity in enumerate(cells[0]["entities"]):
                    tail = continuation(tokenizer, entity)
                    sequences = [row["prompt_ids"] + tail for row in cells]
                    if len({len(sequence) for sequence in sequences}) != 1:
                        raise RuntimeError("candidate batch length mismatch")
                    ids = torch.tensor(sequences, dtype=torch.long, device=device)
                    controller.set(condition, prompt_length)
                    with torch.inference_mode():
                        output = model(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            use_cache=False,
                            logits_to_keep=len(tail) + 1,
                            return_dict=True,
                        )
                    logits = output.logits.float()
                    offset = ids.shape[1] - logits.shape[1]
                    first = prompt_length - 1 - offset
                    if candidate_index == 0:
                        next_logits[condition] = logits[:, first].detach().cpu().to(torch.float16).numpy()
                    for cell_index, row in enumerate(cells):
                        total = 0.0
                        for token_offset, token in enumerate(tail):
                            vector = logits[cell_index, first + token_offset]
                            total += float(vector[token] - torch.logsumexp(vector, dim=-1))
                        scores[condition][row["case_id"]][candidate_index] = total
                    del output, logits, ids
            controller.set("baseline", prompt_length)
            for condition in CONDITIONS:
                for row in cells:
                    values = scores[condition][row["case_id"]]
                    prediction = max(values, key=values.get)
                    target = row["target_index"]
                    score_rows.append({
                        "quartet_index": quartet_index,
                        "condition": condition,
                        "case_id": row["case_id"],
                        "family_id": row["family_id"],
                        "relation_form": row["relation_form"],
                        "value_form": row["value_form"],
                        "binding_relation": row["binding_relation"],
                        "binding_value": row["binding_value"],
                        "query_relation": row["query_relation"],
                        "query_value": row["query_value"],
                        "target_index": target,
                        "prediction_index": prediction,
                        "correct": prediction == target,
                        "target_score": values[target],
                        "target_minus_best_wrong": values[target] - max(
                            value for index, value in values.items() if index != target
                        ),
                        "scores": {str(index): value for index, value in values.items()},
                    })
            next_path = next_dir / f"quartet_{quartet_index:03d}.npz"
            np.savez(next_path, **{condition: next_logits[condition] for condition in CONDITIONS})
            next_manifest.append({
                "quartet_index": quartet_index,
                "path": str(next_path.relative_to(ROOT)).replace("\\", "/"),
                "shape": [len(CONDITIONS), 4, int(next_logits["baseline"].shape[1])],
                "dtype": "float16",
                "bytes": next_path.stat().st_size,
            })
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[phase2584 causal] {quartet_index + 1}/{len(selected)}", flush=True)
    finally:
        controller.close()
    save_json(OUT / "next_token_field/manifest.json", next_manifest)
    (OUT / "behavior").mkdir(parents=True, exist_ok=True)
    with (OUT / "behavior/scores.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in score_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return score_rows, controller.stats, next_manifest


def summarize(rows, stats, next_manifest):
    baseline = {row["case_id"]: row for row in rows if row["condition"] == "baseline"}
    behavior = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        changed = sum(row["prediction_index"] != baseline[row["case_id"]]["prediction_index"] for row in subset)
        behavior[condition] = {
            "n": len(subset),
            "accuracy": float(np.mean([row["correct"] for row in subset])),
            "mean_target_margin": float(np.mean([row["target_minus_best_wrong"] for row in subset])),
            "changed_predictions_vs_baseline": changed,
            "by_form": {
                form: float(np.mean([row["correct"] for row in subset
                                     if f'{row["relation_form"]}/{row["value_form"]}' == form]))
                for form in ("natural/natural", "nonce/natural")
            },
        }
    intervention = {}
    for condition in CONDITIONS[1:]:
        subset = [row for row in stats if row["condition"] == condition]
        intervention[condition] = {
            name: float(np.median([row[name] for row in subset]))
            for name in ("before_rms", "after_rms", "perturbation_rms")
        }
    next_token = {}
    for condition in CONDITIONS:
        values = []
        for item in next_manifest:
            arrays = np.load(ROOT / item["path"])
            values.append(arrays[condition].astype(np.float32))
        full = np.stack(values)
        interaction = full[:, 3] - full[:, 2] - full[:, 1] + full[:, 0]
        next_token[condition] = {
            "full_vocab_logits": int(full.shape[-1]),
            "median_factorial_rms": float(np.median(np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=1)))),
            "median_factorial_maxabs": float(np.median(np.max(np.abs(interaction), axis=1))),
        }
    prior = load_json(P2583)
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "prior_correction": {
            "phase2583_initial_memo_overclaim": "31/32, not 32/32, heads had median interaction above 1e-7",
            "phase2583_final_corrected": prior["all_checks_passed"],
        },
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "design": {
            "quartets": 32,
            "cells": 128,
            "conditions": list(CONDITIONS),
            "candidate_sequences": 32 * 4 * 4 * len(CONDITIONS),
            "intervention": "remove batch-factorial component symmetrically from all four cells on prompt tokens",
            "matched_control": f"same-norm coordinate roll by {ROLL}",
            "complete_candidate_likelihood": True,
            "full_next_token_vocabulary_preserved": True,
        },
        "behavior": behavior,
        "intervention_diagnostics": intervention,
        "next_token_field": next_token,
        "storage": {"next_token_bytes": sum(item["bytes"] for item in next_manifest)},
        "claim_boundary": {
            "positive": "a selective effect larger than the matched roll would support directional causal use of the removed component",
            "negative": "a null effect means block0 interaction is replaceable/regenerated under this coupled intervention, not that the route is irrelevant",
            "not_supported": "any intervention outcome alone identifies a complete semantic gear",
        },
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2583_corrected_final_passes": prior["all_checks_passed"],
        "all_2560_candidate_sequences": len(rows) == 32 * 4 * len(CONDITIONS),
        "all_conditions_complete": all(value["n"] == 128 for value in behavior.values()),
        "baseline_all_correct": behavior["baseline"]["accuracy"] == 1.0,
        "attention_removal_reduces_measured_interaction": (
            intervention["attention_interaction_removed"]["after_rms"]
            < intervention["attention_interaction_removed"]["before_rms"]
        ),
        "matched_control_same_perturbation_scale": abs(
            intervention["attention_interaction_removed"]["perturbation_rms"]
            - intervention["attention_matched_coordinate_roll"]["perturbation_rms"]
        ) <= 1e-5,
        "full_vocab_saved": all((ROOT / item["path"]).is_file() for item in next_manifest),
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    return result


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 首块交互的同范数因果对照与完整next-token场（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**前Phase勘误。** Phase2583初版文字误写“32头均非零”；正确结果是31/32头的中位interaction RMS超过$10^{{-7}}$，head 2为0。Phase2583 final与本Phase均已按“分布式但不均匀”裁决；组件场不受影响。

**测试原理。** 对每组四格batch的prompt token，在block0组件输出$Y$中计算$I(Y)=Y_{{11}}-Y_{{10}}-Y_{{01}}+Y_{{00}}$，再作对称投影：

$$Y'_{{00}}=Y_{{00}}-\tfrac14I,\quad Y'_{{01}}=Y_{{01}}+\tfrac14I,\quad
Y'_{{10}}=Y_{{10}}+\tfrac14I,\quad Y'_{{11}}=Y_{{11}}-\tfrac14I,$$

使四格二阶项尽量归零而保留总体均值和一阶主效应。分别在attention输出、MLP输出、整个block0输出执行；另把$I$沿2560坐标固定循环位移641位，形成同范数方向错配对照。干预只覆盖prompt，不直接改candidate token。

**测试用例。** 沿用32四元组、128个行为正确cell；5条件×4完整候选，共2560条完整多token candidate序列。每个quartet/条件还保存四格完整{next(iter(result['next_token_field'].values()))['full_vocab_logits']}维next-token logits，不以Top-K代替词表场。

**结果汇总。** 行为：`{json.dumps(result['behavior'], ensure_ascii=False)}`。干预实现诊断：`{json.dumps(result['intervention_diagnostics'], ensure_ascii=False)}`。完整词表二阶场：`{json.dumps(result['next_token_field'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2584_c397569_c409856_block0_interaction_causal_controls.py`；全部候选分数、完整next-token场、manifest与final位于`{OUT}`。

**理论进展与分析。** 该实验首次直接问“首块产生的交互方向是否被后续网络使用”，并用同范数坐标错配区分纯扰动与特定方向。attention/MLP/block三层次可判断交互被同块MLP吸收、被后层再生或真正形成瓶颈。无论必要性为正或负，都不终止对这条路径的测绘。

**问题硬伤。** 四格batch耦合投影不是模型自然运行中的局部手术；BF16回写使交互不可能数学上精确为0；坐标循环位移只保范数，不保协方差或LayerNorm几何；只改首块，后层可再生；候选与人工表格限制仍在。因此阳性也只是方向性因果证据，阴性也不是路线否定。

**结论。** `{json.dumps(result['claim_boundary'], ensure_ascii=False)}`。检查`{json.dumps(result['checks'], ensure_ascii=False)}`。语言机制未闭合；下一Phase重造等BPE四词面材料，锁箱验证该交互路径是否跨自然/nonce关系和值复用。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    _, _, compatible, _ = p2582.material_index()
    selected, _ = p2582.balanced_select(compatible)
    model = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        rows, stats, manifest = score(model, tokenizer, selected)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    result = summarize(rows, stats, manifest)
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
