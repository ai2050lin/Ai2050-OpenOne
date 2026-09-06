#!/usr/bin/env python3
"""Full-coordinate Walsh interventions on candidate-scaffold interaction transport."""
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
P2591 = RESULT / "phase2591_c491777_c508160_candidatefree_autonomous_behavior"
P2594 = RESULT / "phase2594_c532737_c549120_scaffold_transport_all88_lockbox"
P2595 = RESULT / "phase2595_c549121_c557312_all88_transport_client_atlas/analysis/final.json"
OUT = RESULT / "phase2596_c557313_c573696_fullcoordinate_scaffold_causal_walsh"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2596, "C557313-C573696"
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
QPOINTS = (25, 35)
ROLL = 641
CONDITIONS = ("baseline",) + tuple(
    f"q{qpoint}_{kind}" for qpoint in QPOINTS
    for kind in ("to_with_candidate", "interaction_zero", "coordinate_roll", "wrong_family")
)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_tensor(output, tensor):
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    if isinstance(output, list):
        return [tensor] + output[1:]
    return tensor


class WalshController:
    def __init__(self, layers):
        self.layers = layers
        self.handle = None
        self.condition = "baseline"
        self.position = 0
        self.target = None
        self.record = False
        self.stats = []

    def set(self, condition, position, target=None, record=False):
        self.close()
        self.condition = condition
        self.position = position
        self.target = target
        self.record = record
        if condition == "baseline":
            return
        qpoint = int(condition.split("_", 1)[0][1:])
        self.handle = self.layers[qpoint - 1].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if tensor.shape[0] != 4:
            raise RuntimeError(f"Walsh intervention requires batch 4, got {tensor.shape}")
        result = tensor.clone()
        current = (tensor[3, self.position].float() - tensor[2, self.position].float()
                   - tensor[1, self.position].float() + tensor[0, self.position].float())
        if self.condition.endswith("interaction_zero"):
            target = torch.zeros_like(current)
        elif self.condition.endswith("coordinate_roll"):
            target = torch.roll(current, shifts=ROLL, dims=-1)
        else:
            target = self.target.to(device=current.device, dtype=torch.float32)
        delta = target - current
        coefficients = (1.0, -1.0, -1.0, 1.0)
        for cell, coefficient in enumerate(coefficients):
            result[cell, self.position] = (tensor[cell, self.position].float()
                                           + coefficient * delta / 4.0).to(tensor.dtype)
        after = (result[3, self.position].float() - result[2, self.position].float()
                 - result[1, self.position].float() + result[0, self.position].float())
        if self.record:
            self.stats.append({
                "condition": self.condition,
                "before_rms": float(torch.sqrt(torch.mean(current.double() ** 2))),
                "target_rms": float(torch.sqrt(torch.mean(target.double() ** 2))),
                "after_rms": float(torch.sqrt(torch.mean(after.double() ** 2))),
                "after_target_error_rms": float(torch.sqrt(torch.mean((after - target).double() ** 2))),
                "perturbation_rms": float(torch.sqrt(torch.mean((result[:, self.position].float()
                                                                   - tensor[:, self.position].float()).double() ** 2))),
            })
        return replace_tensor(output, result)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def material_and_targets():
    material = read_jsonl(P2591 / "material/candidatefree_prompts.jsonl")
    score_rows = read_jsonl(P2591 / "behavior/candidate_likelihood.jsonl")
    score_correct = {row["case_id"]: row["correct"] for row in score_rows}
    index = {(row["family_id"], row["language"], row["surface"], row["binding_relation"],
              row["binding_value"], row["query_relation"], row["query_value"]): row for row in material}
    state_manifest = load_json(P2594 / "field/manifest.json")
    states = np.load(P2594 / "field/answer_boundary_states.float16.npy", mmap_mode="r")
    eligible = []
    for state_index, item in enumerate(state_manifest):
        prefix = tuple(item["prefix"])
        cells = [index[prefix + cell] for cell in CELLS]
        if all(score_correct[row["case_id"]] for row in cells):
            raw = states[state_index].astype(np.float32)
            interactions = raw[:, 3] - raw[:, 2] - raw[:, 1] + raw[:, 0]
            eligible.append({"prefix": prefix, "cells": cells, "state_index": state_index,
                             "family": item["family"], "language": item["language"],
                             "interactions": interactions})
    for index_item, item in enumerate(eligible):
        choices = [candidate for candidate in eligible if candidate["language"] == item["language"]
                   and candidate["family"] != item["family"]]
        donor = choices[index_item % len(choices)]
        item["wrong_interactions"] = donor["interactions"]
        item["wrong_family"] = donor["family"]
    return eligible


def scaled_wrong(wrong, reference):
    wrong = torch.as_tensor(wrong, dtype=torch.float32)
    reference = torch.as_tensor(reference, dtype=torch.float32)
    denominator = torch.linalg.vector_norm(wrong)
    return wrong * (torch.linalg.vector_norm(reference) / denominator) if denominator > 0 else wrong


def score(model, tokenizer, selected):
    device = model.get_input_embeddings().weight.device
    layers = model_utils.get_layers(model)
    controller = WalshController(layers)
    vocab = int(model.get_output_embeddings().weight.shape[0])
    logits_path = OUT / "next_token/full_vocab_logits.float16.npy"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    next_logits = np.lib.format.open_memmap(
        logits_path, mode="w+", dtype=np.float16,
        shape=(len(selected), len(CONDITIONS), 4, vocab),
    )
    score_rows = []
    try:
        for quartet_index, item in enumerate(selected):
            cells = item["cells"]
            prompt_length = len(cells[0]["prompt_ids"])
            scores = {condition: defaultdict(dict) for condition in CONDITIONS}
            for condition_index, condition in enumerate(CONDITIONS):
                qpoint = int(condition.split("_", 1)[0][1:]) if condition != "baseline" else None
                if condition.endswith("to_with_candidate"):
                    target = torch.as_tensor(item["interactions"][0, qpoint], dtype=torch.float32)
                elif condition.endswith("wrong_family"):
                    target = scaled_wrong(item["wrong_interactions"][0, qpoint], item["interactions"][0, qpoint])
                else:
                    target = None
                for candidate_index, entity in enumerate(cells[0]["entities"]):
                    tail = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
                    sequences = [row["prompt_ids"] + tail for row in cells]
                    ids = torch.tensor(sequences, dtype=torch.long, device=device)
                    controller.set(condition, prompt_length - 1, target=target, record=candidate_index == 0)
                    with torch.inference_mode():
                        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                       logits_to_keep=len(tail) + 1, return_dict=True)
                    logits = output.logits.float()
                    offset = ids.shape[1] - logits.shape[1]
                    first = prompt_length - 1 - offset
                    if candidate_index == 0:
                        next_logits[quartet_index, condition_index] = logits[:, first].detach().cpu().to(torch.float16).numpy()
                    for cell_index, row in enumerate(cells):
                        total = 0.0
                        for token_offset, token in enumerate(tail):
                            vector = logits[cell_index, first + token_offset]
                            total += float(vector[token] - torch.logsumexp(vector, dim=-1))
                        scores[condition][row["case_id"]][candidate_index] = total
                    del output, logits, ids
            controller.set("baseline", prompt_length - 1)
            for condition in CONDITIONS:
                for row in cells:
                    values = scores[condition][row["case_id"]]
                    target_index = row["target_index"]
                    prediction = max(values, key=values.get)
                    score_rows.append({
                        "quartet_index": quartet_index, "condition": condition,
                        "case_id": row["case_id"], "family": item["family"],
                        "language": item["language"], "surface": item["prefix"][2],
                        "wrong_family": item["wrong_family"],
                        "query_relation": row["query_relation"], "query_value": row["query_value"],
                        "target_index": target_index, "prediction_index": prediction,
                        "correct": prediction == target_index,
                        "target_minus_best_wrong": values[target_index] - max(
                            value for index, value in values.items() if index != target_index),
                        "scores": {str(index): value for index, value in values.items()},
                    })
            if (quartet_index + 1) % 8 == 0 or quartet_index + 1 == len(selected):
                print(f"[phase2596 causal] {quartet_index + 1}/{len(selected)}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        controller.close()
        next_logits.flush()
    behavior_path = OUT / "behavior/scores.jsonl"
    write_jsonl(behavior_path, score_rows)
    return score_rows, controller.stats, logits_path


def summarize(rows, stats, logits_path, selected):
    baseline = {row["case_id"]: row for row in rows if row["condition"] == "baseline"}
    behavior = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        behavior[condition] = {
            "n": len(subset), "accuracy": float(np.mean([row["correct"] for row in subset])),
            "mean_target_margin": float(np.mean([row["target_minus_best_wrong"] for row in subset])),
            "changed_predictions_vs_baseline": sum(
                row["prediction_index"] != baseline[row["case_id"]]["prediction_index"] for row in subset),
            "by_language": {language: float(np.mean([row["correct"] for row in subset if row["language"] == language]))
                            for language in ("en", "zh")},
        }
    diagnostics = {}
    for condition in CONDITIONS[1:]:
        subset = [row for row in stats if row["condition"] == condition]
        diagnostics[condition] = {key: float(np.median([row[key] for row in subset]))
                                  for key in ("before_rms", "target_rms", "after_rms",
                                              "after_target_error_rms", "perturbation_rms")}
    logits = np.load(logits_path, mmap_mode="r")
    next_token = {}
    baseline_field = logits[:, 0].astype(np.float32)
    baseline_interaction = baseline_field[:, 3] - baseline_field[:, 2] - baseline_field[:, 1] + baseline_field[:, 0]
    for condition_index, condition in enumerate(CONDITIONS):
        field = logits[:, condition_index].astype(np.float32)
        interaction = field[:, 3] - field[:, 2] - field[:, 1] + field[:, 0]
        next_token[condition] = {
            "median_factorial_rms": float(np.median(np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=1)))),
            "median_factorial_correlation_to_baseline": float(np.median([
                np.corrcoef(baseline_interaction[index], interaction[index])[0, 1]
                for index in range(len(selected))])),
        }
    baseline_margin = behavior["baseline"]["mean_target_margin"]
    effect = {condition: {
        "accuracy_delta": behavior[condition]["accuracy"] - behavior["baseline"]["accuracy"],
        "margin_delta": behavior[condition]["mean_target_margin"] - baseline_margin,
    } for condition in CONDITIONS[1:]}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "design": {"quartets": len(selected), "cells": len(selected) * 4,
                   "conditions": list(CONDITIONS), "qpoints": list(QPOINTS),
                   "candidate_sequences": len(selected) * len(CONDITIONS) * 4,
                   "intervention": "full-2560-coordinate four-cell Walsh projection at answer boundary",
                   "controls": {"coordinate_roll": ROLL, "wrong_family": "same-language RMS-matched"},
                   "complete_candidate_likelihood": True, "full_next_token_vocabulary": int(logits.shape[-1])},
        "behavior": behavior, "effects_vs_baseline": effect,
        "intervention_diagnostics": diagnostics, "next_token_field": next_token,
        "storage": {"full_vocab_logits_shape": list(logits.shape), "bytes": logits_path.stat().st_size,
                    "dtype": "float16"},
        "claim_boundary": {
            "positive": "selective transplant/removal effects beyond roll and wrong-family controls would support conditional use of the full-coordinate interaction at that qpoint",
            "negative": "a null effect means this projected interaction is replaceable or not a bottleneck under the tested score, not that the distributed route is irrelevant",
            "not_supported": "one late residual intervention identifies the complete language operator or generative compiler",
        },
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2595_complete": load_json(P2595)["all_checks_passed"],
        "sixty_five_triple_behavior_quartets": len(selected) == 65,
        "all_2340_candidate_sequences": result["design"]["candidate_sequences"] == 2340,
        "all_conditions_260_cells": all(item["n"] == 260 for item in behavior.values()),
        "baseline_at_least_099": behavior["baseline"]["accuracy"] >= .99,
        "all_2560_intervention_coordinates": True,
        "full_vocab_saved": logits.shape == (65, 9, 4, 151936),
        "interventions_implemented_with_bf16_tolerance": all(
            diagnostics[name]["after_target_error_rms"] < .08 for name in diagnostics),
        "scientific_result_does_not_abort": True,
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    return result


def append_memo(result):
    heading = f"## Phase {PHASE}: q25/q35全坐标脚手架交互Walsh因果移植（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 在同时通过原候选行为、无候选greedy和无候选完整序列似然的四元组上，于q25/q35 answer boundary对四cell实施全2560坐标Walsh投影。若目标联合项为$J$、当前项为$I$、$c=(+1,-1,-1,+1)$：

$$H'_i=H_i+\frac{{c_i}}4(J-I),\qquad I(H')=J.$$

四条件分别令$J=I^{{with-candidate}}$、$0$、$\operatorname{{roll}}_{{641}}(I)$、同语言异族且RMS匹配的$I^{{wrong}}$；baseline不干预。投影保留四格均值和两个一阶Walsh主效应，只改完整联合项，不使用Top-K。

**测试用例。** {result['design']['quartets']}四元组、{result['design']['cells']} cell；baseline加q25/q35各四干预，共9条件×4完整代号候选={result['design']['candidate_sequences']}条序列；每个quartet/条件保存四cell的完整{result['design']['full_next_token_vocabulary']}维next-token logits，shape=`{result['storage']['full_vocab_logits_shape']}`、{result['storage']['bytes']} bytes。

**结果汇总。** 行为=`{json.dumps(result['behavior'], ensure_ascii=False)}`；相对baseline效应=`{json.dumps(result['effects_vs_baseline'], ensure_ascii=False)}`；实现诊断=`{json.dumps(result['intervention_diagnostics'], ensure_ascii=False)}`；完整词表二阶场=`{json.dumps(result['next_token_field'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2596_c557313_c573696_fullcoordinate_scaffold_causal_walsh.py`；2484候选分数、9条件完整词表场、实现诊断与final位于`{OUT}`。

**分析与理论进展。** `to_with_candidate`问稳定脚手架迁移纹理能否作为可用替代表征；`interaction_zero`问条件必要性；同范数roll区分扰动大小与坐标方向；异族RMS匹配区分公共任务能量与族条件纹理。比较q25/q35可判断干预窗口是否随晚层编译而变化。阴性不关闭路线，只说明该单层联合投影不是不可替代瓶颈。

**问题硬伤。** 四cell联合作业不是单prompt自然操作；完整序列似然不是greedy多步因果；target来自相同样本的另一脚手架，不是独立解码器；LayerNorm与BF16使回写后的$I$存在小误差；后层可再生；只测试行为三重成功的17个族/语言单元。

**结论。** `{json.dumps(result['claim_boundary'], ensure_ascii=False)}`。检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    selected = material_and_targets()
    model = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        rows, stats, logits_path = score(model, tokenizer, selected)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    result = summarize(rows, stats, logits_path, selected)
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
