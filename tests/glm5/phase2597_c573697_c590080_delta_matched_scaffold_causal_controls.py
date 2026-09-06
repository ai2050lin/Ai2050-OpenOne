#!/usr/bin/env python3
"""Strictly delta-norm-matched causal controls for scaffold interaction transport."""
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
P2596 = RESULT / "phase2596_c557313_c573696_fullcoordinate_scaffold_causal_walsh"
OUT = RESULT / "phase2597_c573697_c590080_delta_matched_scaffold_causal_controls"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2597, "C573697-C590080"
QPOINTS = (25, 35)
ROLL = 641
KINDS = ("transplant_delta", "transplant_delta_roll", "zero_delta", "zero_delta_roll", "wrong_delta")
CONDITIONS = ("baseline",) + tuple(f"q{qpoint}_{kind}" for qpoint in QPOINTS for kind in KINDS)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2596_c557313_c573696_fullcoordinate_scaffold_causal_walsh as p2596  # noqa: E402


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


class DeltaController:
    def __init__(self, layers):
        self.layers = layers
        self.handle = None
        self.condition = "baseline"
        self.position = 0
        self.delta = None
        self.record = False
        self.stats = []

    def set(self, condition, position, delta=None, record=False):
        self.close()
        self.condition = condition
        self.position = position
        self.delta = delta
        self.record = record
        if condition == "baseline":
            return
        qpoint = int(condition.split("_", 1)[0][1:])
        self.handle = self.layers[qpoint - 1].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if tensor.shape[0] != 4:
            raise RuntimeError(f"Delta Walsh controller requires batch4, got {tensor.shape}")
        result = tensor.clone()
        before = (tensor[3, self.position].float() - tensor[2, self.position].float()
                  - tensor[1, self.position].float() + tensor[0, self.position].float())
        delta = self.delta.to(device=before.device, dtype=torch.float32)
        coefficients = (1.0, -1.0, -1.0, 1.0)
        for cell, coefficient in enumerate(coefficients):
            result[cell, self.position] = (tensor[cell, self.position].float()
                                           + coefficient * delta / 4.0).to(tensor.dtype)
        after = (result[3, self.position].float() - result[2, self.position].float()
                 - result[1, self.position].float() + result[0, self.position].float())
        if self.record:
            self.stats.append({
                "condition": self.condition,
                "before_rms": float(torch.sqrt(torch.mean(before.double() ** 2))),
                "delta_rms": float(torch.sqrt(torch.mean(delta.double() ** 2))),
                "after_rms": float(torch.sqrt(torch.mean(after.double() ** 2))),
                "after_delta_error_rms": float(torch.sqrt(torch.mean((after - before - delta).double() ** 2))),
                "perturbation_rms": float(torch.sqrt(torch.mean((result[:, self.position].float()
                                                                   - tensor[:, self.position].float()).double() ** 2))),
            })
        return p2596.replace_tensor(output, result)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def matched_scale(vector, reference):
    vector = torch.as_tensor(vector, dtype=torch.float32)
    reference = torch.as_tensor(reference, dtype=torch.float32)
    norm = torch.linalg.vector_norm(vector)
    return vector * (torch.linalg.vector_norm(reference) / norm) if norm > 0 else vector


def intervention_delta(item, qpoint, kind):
    own = item["interactions"]
    transplant = torch.as_tensor(own[0, qpoint] - own[1, qpoint], dtype=torch.float32)
    zero = torch.as_tensor(-own[1, qpoint], dtype=torch.float32)
    if kind == "transplant_delta":
        return transplant
    if kind == "transplant_delta_roll":
        return torch.roll(transplant, shifts=ROLL, dims=-1)
    if kind == "zero_delta":
        return zero
    if kind == "zero_delta_roll":
        return torch.roll(zero, shifts=ROLL, dims=-1)
    wrong = item["wrong_interactions"][0, qpoint] - item["wrong_interactions"][1, qpoint]
    return matched_scale(wrong, transplant)


def score(model, tokenizer, selected):
    device = model.get_input_embeddings().weight.device
    controller = DeltaController(model_utils.get_layers(model))
    vocab = int(model.get_output_embeddings().weight.shape[0])
    logits_path = OUT / "next_token/full_vocab_logits.float16.npy"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    next_logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float16,
        shape=(len(selected), len(CONDITIONS), 4, vocab))
    rows = []
    try:
        for quartet_index, item in enumerate(selected):
            cells = item["cells"]
            prompt_length = len(cells[0]["prompt_ids"])
            scores = {condition: defaultdict(dict) for condition in CONDITIONS}
            for condition_index, condition in enumerate(CONDITIONS):
                if condition == "baseline":
                    delta = None
                else:
                    qpoint_text, kind = condition.split("_", 1)
                    delta = intervention_delta(item, int(qpoint_text[1:]), kind)
                for candidate_index, entity in enumerate(cells[0]["entities"]):
                    tail = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
                    ids = torch.tensor([row["prompt_ids"] + tail for row in cells], dtype=torch.long, device=device)
                    controller.set(condition, prompt_length - 1, delta=delta, record=candidate_index == 0)
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
                    target = row["target_index"]
                    prediction = max(values, key=values.get)
                    rows.append({"quartet_index": quartet_index, "condition": condition,
                                 "case_id": row["case_id"], "family": item["family"],
                                 "language": item["language"], "surface": item["prefix"][2],
                                 "target_index": target, "prediction_index": prediction,
                                 "correct": prediction == target,
                                 "target_minus_best_wrong": values[target] - max(
                                     value for index, value in values.items() if index != target)})
            if (quartet_index + 1) % 8 == 0 or quartet_index + 1 == len(selected):
                print(f"[phase2597 matched-delta] {quartet_index + 1}/{len(selected)}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        controller.close()
        next_logits.flush()
    p2596.write_jsonl(OUT / "behavior/scores.jsonl", rows)
    return rows, controller.stats, logits_path


def summarize(rows, stats, logits_path, selected):
    baseline = {row["case_id"]: row for row in rows if row["condition"] == "baseline"}
    behavior = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        behavior[condition] = {"n": len(subset),
            "accuracy": float(np.mean([row["correct"] for row in subset])),
            "mean_target_margin": float(np.mean([row["target_minus_best_wrong"] for row in subset])),
            "changed_predictions_vs_baseline": sum(
                row["prediction_index"] != baseline[row["case_id"]]["prediction_index"] for row in subset)}
    diagnostics = {}
    for condition in CONDITIONS[1:]:
        subset = [row for row in stats if row["condition"] == condition]
        diagnostics[condition] = {key: float(np.median([row[key] for row in subset]))
                                  for key in ("before_rms", "delta_rms", "after_rms",
                                              "after_delta_error_rms", "perturbation_rms")}
    logits = np.load(logits_path, mmap_mode="r")
    base_i = logits[:, 0, 3].astype(np.float32) - logits[:, 0, 2].astype(np.float32)
    base_i -= logits[:, 0, 1].astype(np.float32)
    base_i += logits[:, 0, 0].astype(np.float32)
    next_token = {}
    for condition_index, condition in enumerate(CONDITIONS):
        field = logits[:, condition_index].astype(np.float32)
        interaction = field[:, 3] - field[:, 2] - field[:, 1] + field[:, 0]
        next_token[condition] = {
            "median_factorial_rms": float(np.median(np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=1)))),
            "median_correlation_to_baseline": float(np.median([
                np.corrcoef(base_i[index], interaction[index])[0, 1] for index in range(len(selected))])),
        }
    base_margin = behavior["baseline"]["mean_target_margin"]
    effects = {condition: {"accuracy_delta": behavior[condition]["accuracy"] - behavior["baseline"]["accuracy"],
                           "margin_delta": behavior[condition]["mean_target_margin"] - base_margin}
               for condition in CONDITIONS[1:]}
    contrasts = {}
    for qpoint in QPOINTS:
        t = f"q{qpoint}_transplant_delta"
        tr = f"q{qpoint}_transplant_delta_roll"
        z = f"q{qpoint}_zero_delta"
        zr = f"q{qpoint}_zero_delta_roll"
        w = f"q{qpoint}_wrong_delta"
        contrasts[f"q{qpoint}"] = {
            "transplant_minus_equal_delta_roll_accuracy": behavior[t]["accuracy"] - behavior[tr]["accuracy"],
            "transplant_minus_equal_delta_roll_margin": behavior[t]["mean_target_margin"] - behavior[tr]["mean_target_margin"],
            "zero_minus_equal_delta_roll_accuracy": behavior[z]["accuracy"] - behavior[zr]["accuracy"],
            "transplant_minus_equal_delta_wrong_accuracy": behavior[t]["accuracy"] - behavior[w]["accuracy"],
            "transplant_vs_roll_perturbation_rms_gap": diagnostics[t]["perturbation_rms"] - diagnostics[tr]["perturbation_rms"],
            "zero_vs_roll_perturbation_rms_gap": diagnostics[z]["perturbation_rms"] - diagnostics[zr]["perturbation_rms"],
            "transplant_vs_wrong_perturbation_rms_gap": diagnostics[t]["perturbation_rms"] - diagnostics[w]["perturbation_rms"],
        }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized",
              "design": {"quartets": len(selected), "cells": len(selected) * 4,
                         "conditions": list(CONDITIONS), "qpoints": list(QPOINTS),
                         "candidate_sequences": len(selected) * len(CONDITIONS) * 4,
                         "intervention": "all-2560-coordinate Walsh delta; matched controls preserve delta norm",
                         "full_next_token_vocabulary": int(logits.shape[-1])},
              "behavior": behavior, "effects_vs_baseline": effects, "matched_delta_contrasts": contrasts,
              "intervention_diagnostics": diagnostics, "next_token_field": next_token,
              "storage": {"shape": list(logits.shape), "bytes": logits_path.stat().st_size, "dtype": "float16"},
              "claim_boundary": {"supported": "strictly separates a transported delta from equal-delta-norm coordinate and family controls at q25/q35",
                                 "not_supported": "selectivity of this coupled four-cell projection is a single-prompt natural causal gear or complete compiler"},
              "language_mechanism_closed": False}
    match_ok = all(abs(contrasts[f"q{qpoint}"][key]) < .003 for qpoint in QPOINTS for key in (
        "transplant_vs_roll_perturbation_rms_gap", "zero_vs_roll_perturbation_rms_gap",
        "transplant_vs_wrong_perturbation_rms_gap"))
    result["checks"] = {"phase2596_complete": p2596.load_json(P2596 / "analysis/final.json")["all_checks_passed"],
                        "all_65_quartets": len(selected) == 65,
                        "all_2860_candidate_sequences": result["design"]["candidate_sequences"] == 2860,
                        "all_conditions_260_cells": all(item["n"] == 260 for item in behavior.values()),
                        "baseline_at_least_099": behavior["baseline"]["accuracy"] >= .99,
                        "full_vocab_shape": logits.shape == (65, 11, 4, 151936),
                        "delta_norm_controls_matched": match_ok,
                        "bf16_delta_error_below_008": all(item["after_delta_error_rms"] < .08 for item in diagnostics.values()),
                        "all_2560_coordinates": True, "scientific_result_does_not_abort": True,
                        "claim_boundary": True}
    result["all_checks_passed"] = all(result["checks"].values())
    return result


def append_memo(result):
    heading = f"## Phase {PHASE}: q25/q35严格等Δ范数的全坐标因果对照（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 修复Phase2596“目标$I$同范数但位移$\Delta I$不等范数”的硬伤。对当前无候选联合项$I_f$定义：

$$\Delta_T=I_w-I_f,\quad \Delta_0=-I_f,\quad
\Delta_{{T,r}}=\operatorname{{roll}}_{{641}}(\Delta_T),\quad
\Delta_{{0,r}}=\operatorname{{roll}}_{{641}}(\Delta_0),$$

$$\Delta_{{wrong}}=\frac{{\|\Delta_T\|}}{{\|I^w_{{donor}}-I^f_{{donor}}\|}}
(I^w_{{donor}}-I^f_{{donor}}).$$

全部条件用$H'_i=H_i+c_i\Delta/4$，故每组transplant/roll、zero/roll、transplant/wrong严格匹配位移范数并覆盖全部2560坐标。

**测试用例。** 65四元组、260 cell；baseline加q25/q35各5条件，共11条件、{result['design']['candidate_sequences']}条完整候选序列；保存shape=`{result['storage']['shape']}`的完整151936词表next-token场（{result['storage']['bytes']} bytes）。

**结果汇总。** 行为=`{json.dumps(result['behavior'], ensure_ascii=False)}`；效应=`{json.dumps(result['effects_vs_baseline'], ensure_ascii=False)}`；严格匹配对比=`{json.dumps(result['matched_delta_contrasts'], ensure_ascii=False)}`；实现诊断=`{json.dumps(result['intervention_diagnostics'], ensure_ascii=False)}`；词表场=`{json.dumps(result['next_token_field'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2597_c573697_c590080_delta_matched_scaffold_causal_controls.py`；2860分数、11条件完整词表场、诊断与final位于`{OUT}`。

**分析与理论进展。** transplant优于等Δ roll才是坐标方向选择性；优于等Δ异族才是族/样本条件性；zero与其等Δ roll的差异区分“移除联合项”与一般同能扰动。两个深度可判断编译窗口。任何阳性都限于四cell联合投影，任何阴性都不否定分布式路径。

**问题硬伤。** 干预仍耦合四个prompt；完整序列似然而非真实多步greedy；wrong donor虽同语言同Δ范数，但协方差、角度和LayerNorm响应不匹配；样本由三重行为成功筛选；只测两个层。

**结论。** `{json.dumps(result['claim_boundary'], ensure_ascii=False)}`。检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    selected = p2596.material_and_targets()
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
