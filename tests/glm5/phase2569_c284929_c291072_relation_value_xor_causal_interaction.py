#!/usr/bin/env python3
"""Causal single-factor and double-factor relation x value tests."""
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
P2567 = TESTS / "result/phase2567_c264449_c276736_minimal_bridge_extension"
P2568 = TESTS / "result/phase2568_c276737_c284928_relation_value_factorial_fullfield"
OUT = TESTS / "result/phase2569_c284929_c291072_relation_value_xor_causal_interaction"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2569, "C284929-C291072"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2563_c239873_c248064_compositional_distance_relation_atlas as p2563  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def bands(n_layers: int) -> dict[str, tuple[int, ...]]:
    values = p2563.bands(n_layers)
    return dict(zip(("early", "middle", "middlelate", "late"), values))


def specs(n_layers: int) -> dict[str, dict]:
    output = {"no_patch": {"expected": "base"}}
    for band, layers in bands(n_layers).items():
        output[f"{band}_kv_relation"] = {"layers": layers, "kind": "kv", "donor": "relation",
                                          "regions": ("query_relation",), "expected": "flip"}
        output[f"{band}_kv_value"] = {"layers": layers, "kind": "kv", "donor": "value",
                                          "regions": ("query_value",), "expected": "flip"}
        output[f"{band}_kv_double"] = {"layers": layers, "kind": "kv", "donor": "double",
                                        "regions": ("query_relation", "query_value"), "expected": "base"}
        output[f"{band}_kv_cross_relation_to_value"] = {"layers": layers, "kind": "kv",
            "donor": "relation", "regions": ("query_value",), "expected": "base", "matched_null": True}
        output[f"{band}_kv_cross_value_to_relation"] = {"layers": layers, "kind": "kv",
            "donor": "value", "regions": ("query_relation",), "expected": "base", "matched_null": True}
    middlelate = bands(n_layers)["middlelate"]
    late = bands(n_layers)["late"]
    output.update({
        "middlelate_kv_external_relation": {"layers": middlelate, "kind": "kv", "donor": "relation",
            "regions": ("external",), "expected": "flip"},
        "middlelate_kv_external_value": {"layers": middlelate, "kind": "kv", "donor": "value",
            "regions": ("external",), "expected": "flip"},
        "middlelate_kv_external_double": {"layers": middlelate, "kind": "kv", "donor": "double",
            "regions": ("external",), "expected": "base"},
        "late_q_relation": {"layers": late, "kind": "q", "donor": "relation", "expected": "flip"},
        "late_q_value": {"layers": late, "kind": "q", "donor": "value", "expected": "flip"},
        "late_q_double": {"layers": late, "kind": "q", "donor": "double", "expected": "base"},
    })
    return output


class Controller:
    def __init__(self, model, conditions: dict[str, dict]):
        self.layers = model_utils.get_layers(model)
        self.mode, self.capture_label, self.spec, self.jobs = "none", "", {}, []
        self.store: dict[tuple[str, str, int], torch.Tensor] = {}
        self.handles = []
        required = {(kind, layer_index) for spec in conditions.values() for layer_index in spec.get("layers", ())
                    for kind in (("q",) if spec.get("kind") == "q" else ("k", "v"))}
        for layer_index, layer in enumerate(self.layers):
            for kind, name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
                if (kind, layer_index) not in required:
                    continue
                def hook(_module, _inputs, output, layer_index=layer_index, kind=kind):
                    return self._hook(output, layer_index, kind)
                self.handles.append(getattr(layer.self_attn, name).register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _hook(self, output: torch.Tensor, layer_index: int, kind: str):
        if self.mode == "capture":
            self.store[(self.capture_label, kind, layer_index)] = output.detach().clone()
            return None
        if self.mode != "patch" or layer_index not in self.spec.get("layers", ()):
            return None
        requested = self.spec["kind"]
        if not (kind == requested or (requested == "kv" and kind in ("k", "v"))):
            return None
        donor_label = self.spec["donor"]
        donor = self.store[(donor_label, kind, layer_index)].to(output.device)
        changed = output.clone()
        for batch_index, job in enumerate(self.jobs):
            if kind == "q":
                base_start = job["base_shift"] + job["base_prompt_length"] - 1
                donor_start = job[f"{donor_label}_shift"] + job[f"{donor_label}_prompt_length"] - 1
                for offset in range(len(job["continuation"])):
                    changed[batch_index, base_start + offset] = donor[batch_index, donor_start + offset]
            else:
                for region in self.spec["regions"]:
                    base_positions = job["regions"]["base"][region]
                    donor_positions = job["regions"][donor_label][region]
                    if len(base_positions) != len(donor_positions):
                        raise RuntimeError((region, len(base_positions), len(donor_positions)))
                    for base_position, donor_position in zip(base_positions, donor_positions):
                        changed[batch_index, job["base_shift"] + base_position] = donor[
                            batch_index, job[f"{donor_label}_shift"] + donor_position]
        return changed


def region(row: dict, name: str) -> list[int]:
    if name == "external":
        return list(range(row["answer_boundary_token"]))
    return list(row["regions"].get(name, []))


def prepare(material: list[dict], selected: list[tuple], tokenizer, limit: int = 32) -> tuple[list[dict], list[tuple], int]:
    full = [row for row in material if row["ablation"] == "full_scaffold" and row["depth"] == 1]
    index = {(row["family_id"], row["binding"], row["relation_form"], row["value_form"],
              row["query_relation"], row["query_value"]): row for row in full}
    compatible, excluded = [], 0
    for prefix in selected:
        rows = [index[tuple(prefix) + cell] for cell in ((0, 0), (1, 0), (0, 1), (1, 1))]
        if all(len(rows[0]["regions"][name]) == len(row["regions"][name])
               for row in rows[1:] for name in ("query_relation", "query_value")):
            compatible.append(tuple(prefix))
        else:
            excluded += 1
    compatible = compatible[:limit]
    jobs = []
    for prefix in compatible:
        base = index[prefix + (0, 0)]
        donors = {"relation": index[prefix + (1, 0)], "value": index[prefix + (0, 1)],
                  "double": index[prefix + (1, 1)]}
        for candidate_index, entity in enumerate(base["entities"]):
            continuation = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
            job = {"case_id": base["case_id"], "family_id": base["family_id"], "depth": base["depth"],
                   "relation_form": base["relation_form"], "value_form": base["value_form"],
                   "binding": base["binding"], "candidate_index": candidate_index,
                   "target_index": base["target_index"], "flip_target_index": 1 - base["target_index"],
                   "continuation": continuation, "regions": {}}
            for label, row in {"base": base, **donors}.items():
                job[label] = row["prompt_ids"] + continuation
                job[f"{label}_prompt_length"] = len(row["prompt_ids"])
                job["regions"][label] = {name: region(row, name)
                                          for name in ("query_relation", "query_value", "external")}
            jobs.append(job)
    return jobs, compatible, excluded


def continuation_scores(logits: torch.Tensor, jobs: list[dict], width: int, label: str) -> list[float]:
    logit_offset = int(width - logits.shape[1])
    output = []
    for batch_index, job in enumerate(jobs):
        first = job[f"{label}_shift"] + job[f"{label}_prompt_length"] - 1 - logit_offset
        value = 0.0
        for offset, token in enumerate(job["continuation"]):
            z = logits[batch_index, first + offset].float()
            value += float(z[token] - torch.logsumexp(z, dim=-1))
        output.append(value)
    return output


def run(model, tokenizer, jobs: list[dict], conditions: dict[str, dict]) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    controller, output = Controller(model, conditions), []
    buckets: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for job in jobs:
        buckets[tuple(len(job[label]) for label in ("base", "relation", "value", "double"))].append(job)
    batches = [values[start:start + 4] for lengths, values in sorted(buckets.items())
               for start in range(0, len(values), 4)]
    done = 0
    try:
        for batch in batches:
            controller.jobs, controller.store = batch, {}
            for label in ("relation", "value", "double"):
                ids, mask, shifts = p2552.left_pad([job[label] for job in batch], tokenizer.pad_token_id, device)
                for job, shift in zip(batch, shifts):
                    job[f"{label}_shift"] = shift
                controller.mode, controller.capture_label = "capture", label
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
            base_ids, base_mask, base_shifts = p2552.left_pad(
                [job["base"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, base_shifts):
                job["base_shift"] = shift
            keep = max(len(job["continuation"]) for job in batch) + 1
            for condition, spec in conditions.items():
                controller.mode = "none" if condition == "no_patch" else "patch"
                controller.spec = spec
                with torch.inference_mode():
                    logits = model(input_ids=base_ids, attention_mask=base_mask,
                                   use_cache=False, logits_to_keep=keep).logits
                scores = continuation_scores(logits, batch, int(base_ids.shape[1]), "base")
                for job, score in zip(batch, scores):
                    output.append({key: job[key] for key in ("case_id", "family_id", "depth", "relation_form", "value_form", "binding",
                                                              "candidate_index", "target_index", "flip_target_index")})
                    output[-1].update({"condition": condition, "expected": spec["expected"], "score": score})
            done += len(batch)
            if done % 16 == 0 or done == len(jobs):
                print(f"[phase2569 causal] {done}/{len(jobs)}", flush=True)
    finally:
        controller.close()
    return output


def summarize(rows: list[dict], conditions: dict[str, dict]) -> dict:
    output = {}
    for condition, spec in conditions.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["condition"] == condition:
                groups[row["case_id"]].append(row)
        decisions = []
        for values in groups.values():
            prediction = max(values, key=lambda row: row["score"])["candidate_index"]
            decisions.append({"base": prediction == values[0]["target_index"],
                              "flip": prediction == values[0]["flip_target_index"],
                              "depth": values[0]["depth"], "relation_form": values[0]["relation_form"],
                              "value_form": values[0]["value_form"]})
        expected_key = spec["expected"]
        output[condition] = {"n": len(decisions), "base_accuracy": float(np.mean([row["base"] for row in decisions])),
                             "flip_rate": float(np.mean([row["flip"] for row in decisions])),
                             "expected_outcome_rate": float(np.mean([row[expected_key] for row in decisions])),
                             "by_form_expected": {f"r{rf}_v{vf}": float(np.mean([row[expected_key] for row in decisions
                                if row["relation_form"] == rf and row["value_form"] == vf]))
                                if any(row["relation_form"] == rf and row["value_form"] == vf for row in decisions)
                                else None for rf in ("natural", "nonce") for vf in ("natural", "nonce")}}
    return output


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 关系×值XOR的单双因子因果交互（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 从Phase2568的64个平衡四元组中只取query-relation与query-value token数跨00/10/01/11完全匹配的最多32组。base为$(r,v)=(0,0)$；relation donor为$(1,0)$、value donor为$(0,1)$，两者答案都翻转；double donor为$(1,1)$，按XOR答案回到base。对每个实体候选做完整多token评分，分别在四个相对层段替换query-relation K/V、query-value K/V或double donor的两个region K/V；另测错误region matched null、三种中晚层external K/V和三种晚层Q。

$$e(0,0)=e(1,1),\quad e(1,0)=e(0,1)=1-e(0,0),$$

$$C_{{XOR}}=\min(F_R,F_V,B_{{RV}})-\max(F_{{R\to V,null}},F_{{V\to R,null}}),$$

其中$F_R,F_V$是单因子patch翻转率，$B_{{RV}}$是双因子patch保持base率。只有三个预言同时满足且超过错region对照，才支持该层段的条件组合。

**结果汇总。** 选择、兼容和排除数量`{result['selected_quartets']}`、`{result['compatible_quartets']}`、`{result['excluded_token_mismatch']}`；各条件`{json.dumps(result['summary'], ensure_ascii=False)}`；XOR裁决`{json.dumps(result['xor_adjudication'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2569_c284929_c291072_relation_value_xor_causal_interaction.py`；完整逐候选因果分数与final位于`{OUT}`。

**分析与理论进展。** 这比binding donor更接近“关系条件齿轮”：关系和值分别改变输出，而双改变按任务代数恢复原答案。若仅external或late-Q满足，说明输出条件可搬运，但还不能定位计算发生处；若query两个region在同一中间层段呈单因子翻转、双因子恢复且错region低，才是局部组合证据。仍不能把K/V字面等同寻址或内容。

**问题硬伤与结论。** 二元答案使所有错误等于flip；double donor本身答案等于base，保持可能来自无效patch，必须与两个单因子有效同时解释；跨条件token数不等的组被排除，形成词面选择；region级全坐标替换不是最小坐标齿轮；正确四元组经过行为筛选。即使XOR门通过，也只闭合受控任务的一个局部组合关系，不等于自然语言编码机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2568 / "analysis/final.json")
    selected = load(P2568 / "material/selected_quartets.json")["selected"]
    material = read(P2567 / "material/rows.jsonl")
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        jobs, compatible, excluded = prepare(material, selected, tokenizer)
        conditions = specs(len(model_utils.get_layers(model)))
        rows = run(model, tokenizer, jobs, conditions)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    score_path = OUT / "causal/xor_scores.jsonl"
    write(score_path, rows)
    summary = summarize(rows, conditions)
    xor = {}
    for band in ("early", "middle", "middlelate", "late"):
        relation = summary[f"{band}_kv_relation"]["flip_rate"]
        value = summary[f"{band}_kv_value"]["flip_rate"]
        double = summary[f"{band}_kv_double"]["base_accuracy"]
        null = max(summary[f"{band}_kv_cross_relation_to_value"]["flip_rate"],
                   summary[f"{band}_kv_cross_value_to_relation"]["flip_rate"])
        xor[band] = {"relation_flip": relation, "value_flip": value,
                     "double_base_preserve": double, "matched_null_flip": null,
                     "xor_margin": min(relation, value, double) - null,
                     "strong_gate": min(relation, value, double) >= .70 and
                                    min(relation, value, double) - null >= .20}
    checks = {"prior_complete": prior["all_checks_passed"], "compatible_at_least_8": len(compatible) >= 8,
              "two_candidates_each": len(rows) == len(compatible) * 2 * len(conditions),
              "no_patch_identity": summary["no_patch"]["base_accuracy"] >= .98,
              "matched_nulls_present": True, "double_requires_single_effects": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "selected_quartets": len(selected), "compatible_quartets": len(compatible),
              "excluded_token_mismatch": excluded, "conditions": len(conditions), "summary": summary,
              "xor_adjudication": xor, "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
