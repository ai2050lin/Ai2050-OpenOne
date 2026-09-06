#!/usr/bin/env python3
"""Staged Q/K/V and recipient-region causal atlas on relation-necessary lockbox cases."""
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
P2554 = RESULT / "phase2554_c178433_c182528_independent_relation_lockbox_behavior"
OUT = RESULT / "phase2555_c182529_c190720_relation_stage_recipient_causal_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2555, "C182529-C190720"
EARLY, MIDDLE, MIDDLELATE, LATE = tuple(range(0, 9)), tuple(range(9, 18)), tuple(range(18, 27)), tuple(range(27, 36))

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def positions(row: dict, region: str) -> list[int]:
    if region == "facts_all":
        names = ("facts_entity", "facts_relation", "facts_value")
    elif region == "query_all":
        names = ("query_context", "query_relation", "query_value")
    elif region == "post_query":
        names = ("candidate", "instruction")
    elif region == "external":
        return list(range(row["answer_boundary_token"]))
    else:
        names = (region,)
    return sorted({position for name in names for position in row["regions"][name]})


def compile_jobs(tokenizer) -> tuple[list[dict], list[str]]:
    material = [row for row in read(P2554 / "material/u36_lockbox_token_atomic.jsonl") if row["ablation"] == "full_scaffold"]
    behavior = [row for row in read(P2554 / "behavior/u36_candidate_scores.jsonl") if row["ablation"] == "full_scaffold"]
    correct = {row["base_case_id"]: row["correct"] for row in behavior}
    index = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"],
              row["query_value"], row["binding"]): row for row in material}
    regions = ("facts_entity", "facts_relation", "facts_value", "facts_all", "query_context",
               "query_relation", "query_value", "query_all", "candidate", "instruction",
               "post_query", "answer_boundary", "external")
    jobs, eligible_ids = [], []
    for family_id in range(32):
        for relation_form in ("natural", "nonce"):
            for value_form in ("natural", "nonce"):
                for query_relation in (0, 1):
                    for query_value in (0, 1):
                        base = index[(family_id, relation_form, value_form, query_relation, query_value, 0)]
                        donor = index[(family_id, relation_form, value_form, query_relation, query_value, 1)]
                        if not (correct[base["base_case_id"]] and correct[donor["base_case_id"]]):
                            continue
                        pair_id = base["base_case_id"]
                        eligible_ids.append(pair_id)
                        for candidate_index, entity in enumerate(base["entities"]):
                            continuation = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
                            jobs.append({"case_id": pair_id, "family_id": family_id, "family": base["family"],
                                         "relation_form": relation_form, "value_form": value_form,
                                         "query_relation": query_relation, "query_value": query_value,
                                         "candidate_index": candidate_index, "candidate": entity,
                                         "target_index": base["target_index"], "donor_target_index": donor["target_index"],
                                         "base_prompt_length": len(base["prompt_ids"]),
                                         "donor_prompt_length": len(donor["prompt_ids"]),
                                         "base": base["prompt_ids"] + continuation,
                                         "donor": donor["prompt_ids"] + continuation,
                                         "continuation": continuation,
                                         "regions_base": {region: positions(base, region) for region in regions},
                                         "regions_donor": {region: positions(donor, region) for region in regions}})
    return jobs, eligible_ids


CONDITIONS: dict[str, dict] = {
    "no_patch": {},
    "early_k_facts_value": {"layers": EARLY, "kind": "k", "region": "facts_value"},
    "early_v_facts_entity": {"layers": EARLY, "kind": "v", "region": "facts_entity"},
    "early_v_facts_relation": {"layers": EARLY, "kind": "v", "region": "facts_relation"},
    "early_v_facts_value": {"layers": EARLY, "kind": "v", "region": "facts_value"},
    "early_kv_facts_value": {"layers": EARLY, "kind": "kv", "region": "facts_value"},
    "middle_k_facts_entity": {"layers": MIDDLE, "kind": "k", "region": "facts_entity"},
    "middle_k_facts_relation": {"layers": MIDDLE, "kind": "k", "region": "facts_relation"},
    "middle_k_facts_value": {"layers": MIDDLE, "kind": "k", "region": "facts_value"},
    "middle_v_facts_value": {"layers": MIDDLE, "kind": "v", "region": "facts_value"},
    "middle_kv_facts_value": {"layers": MIDDLE, "kind": "kv", "region": "facts_value"},
    "middlelate_kv_facts_value": {"layers": MIDDLELATE, "kind": "kv", "region": "facts_value"},
    "middlelate_kv_query_context": {"layers": MIDDLELATE, "kind": "kv", "region": "query_context"},
    "middlelate_kv_query_relation": {"layers": MIDDLELATE, "kind": "kv", "region": "query_relation"},
    "middlelate_kv_query_value": {"layers": MIDDLELATE, "kind": "kv", "region": "query_value"},
    "middlelate_kv_candidate": {"layers": MIDDLELATE, "kind": "kv", "region": "candidate"},
    "middlelate_kv_instruction": {"layers": MIDDLELATE, "kind": "kv", "region": "instruction"},
    "middlelate_kv_answer_boundary": {"layers": MIDDLELATE, "kind": "kv", "region": "answer_boundary"},
    "middlelate_kv_query_all": {"layers": MIDDLELATE, "kind": "kv", "region": "query_all"},
    "middlelate_kv_post_query": {"layers": MIDDLELATE, "kind": "kv", "region": "post_query"},
    "middlelate_kv_external": {"layers": MIDDLELATE, "kind": "kv", "region": "external"},
    "late_q": {"layers": LATE, "kind": "q", "region": "answer"},
    "late_kv_facts_all": {"layers": LATE, "kind": "kv", "region": "facts_all"},
}


class Controller:
    def __init__(self, model):
        self.layers = model_utils.get_layers(model)
        self.mode = "none"
        self.spec: dict = {}
        self.jobs: list[dict] = []
        self.store: dict[tuple[str, int], torch.Tensor] = {}
        self.handles = []
        required = {(kind, layer_index) for spec in CONDITIONS.values() for layer_index in spec.get("layers", ())
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
        key = (kind, layer_index)
        if self.mode == "capture":
            self.store[key] = output.detach().clone()
            return None
        if self.mode != "patch" or layer_index not in self.spec.get("layers", ()):
            return None
        requested = self.spec["kind"]
        if not (kind == requested or (requested == "kv" and kind in ("k", "v"))):
            return None
        changed = output.clone()
        donor = self.store[key].to(device=output.device, dtype=output.dtype)
        for batch_index, job in enumerate(self.jobs):
            if kind == "q":
                base_start = job["base_shift"] + job["base_prompt_length"] - 1
                donor_start = job["donor_shift"] + job["donor_prompt_length"] - 1
                for offset in range(len(job["continuation"])):
                    changed[batch_index, base_start + offset] = donor[batch_index, donor_start + offset]
            else:
                region = self.spec["region"]
                for base_position, donor_position in zip(job["regions_base"][region], job["regions_donor"][region]):
                    changed[batch_index, job["base_shift"] + base_position] = donor[
                        batch_index, job["donor_shift"] + donor_position
                    ]
        return changed


def forward(model, ids: torch.Tensor, mask: torch.Tensor, keep: int) -> torch.Tensor:
    return model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits


def scores(logits: torch.Tensor, jobs: list[dict], keep: int) -> list[float]:
    output = []
    for batch_index, job in enumerate(jobs):
        continuation = job["continuation"]
        first = keep - len(continuation) - 1
        value = 0.0
        for offset, token in enumerate(continuation):
            z = logits[batch_index, first + offset].float()
            value += float(z[token] - torch.logsumexp(z, dim=-1))
        output.append(value)
    return output


def run(model, tokenizer, jobs: list[dict]) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    controller = Controller(model)
    rows = []
    try:
        for start in range(0, len(jobs), 8):
            batch = jobs[start:start + 8]
            controller.jobs = batch
            donor_ids, donor_mask, donor_shifts = p2552.left_pad([job["donor"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, donor_shifts):
                job["donor_shift"] = shift
            keep = max(len(job["continuation"]) for job in batch) + 1
            controller.mode = "capture"
            controller.store.clear()
            with torch.inference_mode():
                donor_logits = forward(model, donor_ids, donor_mask, keep)
            donor_scores = scores(donor_logits, batch, keep)
            base_ids, base_mask, base_shifts = p2552.left_pad([job["base"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, base_shifts):
                job["base_shift"] = shift
            for condition, spec in CONDITIONS.items():
                controller.mode = "none" if condition == "no_patch" else "patch"
                controller.spec = spec
                with torch.inference_mode():
                    logits = forward(model, base_ids, base_mask, keep)
                values = scores(logits, batch, keep)
                for job, value, donor_value in zip(batch, values, donor_scores):
                    rows.append({"case_id": job["case_id"], "family_id": job["family_id"],
                                 "family": job["family"], "relation_form": job["relation_form"],
                                 "value_form": job["value_form"], "query_relation": job["query_relation"],
                                 "query_value": job["query_value"], "candidate_index": job["candidate_index"],
                                 "target_index": job["target_index"], "donor_target_index": job["donor_target_index"],
                                 "condition": condition, "score": value, "donor_baseline_score": donor_value})
            done = start + len(batch)
            if done % 80 == 0 or done == len(jobs):
                print(f"[phase2555] {done}/{len(jobs)} candidate jobs", flush=True)
    finally:
        controller.close()
    return rows


def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["case_id"])].append(row)
    panels = {}
    for condition in CONDITIONS:
        case_rows = [values for (name, _), values in grouped.items() if name == condition]
        base_correct, donor_flip, margins = [], [], []
        by_form = defaultdict(list)
        for values in case_rows:
            prediction = max(values, key=lambda row: row["score"])["candidate_index"]
            target, donor_target = values[0]["target_index"], values[0]["donor_target_index"]
            base_correct.append(prediction == target)
            donor_flip.append(prediction == donor_target)
            scores_by_index = {row["candidate_index"]: row["score"] for row in values}
            margins.append(scores_by_index[donor_target] - scores_by_index[target])
            by_form[(values[0]["relation_form"], values[0]["value_form"])].append(prediction == donor_target)
        panels[condition] = {"n": len(case_rows), "accuracy": float(np.mean(base_correct)),
                             "donor_flip": float(np.mean(donor_flip)), "mean_donor_margin": float(np.mean(margins)),
                             "donor_flip_by_form": {f"r={key[0]},v={key[1]}": float(np.mean(value))
                                                    for key, value in sorted(by_form.items())}}
    return panels


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 关系必要任务的分阶段Q/K/V与recipient因果图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    key = {name: result["summary"][name] for name in CONDITIONS if name != "no_patch"}
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 在Phase2554新实体锁箱的512个binding0/1对中，只使用base与donor均答对的{result['design']['eligible_pairs']}对，覆盖全部32族、自然/nonce关系与值以及四种query；每对用两个完整多token实体候选。meaning donor仅改变四个事实cell中的binding $b:0\to1$，query不变而正确实体必翻转。Qwen3-4B以BF16非量化CUDA运行，对0–8、9–17、18–26、27–35四段分别干预q_proj/k_proj/v_proj输出。

$$
do\!\left(P^c_{{l,j}}\leftarrow P^{{c,donor}}_{{l,j'}}\right),\quad c\in\{{Q,K,V\}},
\qquad e^*_{{donor}}=1-e^*_{{base}}.
$$

早中层将facts-entity/relation/value分开；中晚层不再使用混合external结论，而逐区拆开facts-value、query-context、query-relation、query-value、candidate、instruction、answer-boundary，并与query-all、post-query及全external累积组合比较；晚层复验答案位置Q和facts-all K/V。

**结果汇总。** 全部条件为`{json.dumps(key, ensure_ascii=False)}`。设计与裁决为`{json.dumps(result['design'], ensure_ascii=False)}`、`{json.dumps(result['adjudication'], ensure_ascii=False)}`；完整性检查为`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2555_c182529_c190720_relation_stage_recipient_causal_atlas.py`；每个case、候选和条件的完整logprob与final位于`{OUT}`。

**分析与理论进展。** 本Phase首次检验旧阶段顺序能否从“relation退化的value匹配”迁移到relation/value都行为必要的合取检索。若早层facts-value仍强，只能称绑定内容源；只有facts-relation也出现独立或联合贡献，才支持关系条件参与。中晚层单recipient效应回答信息究竟先写到query relation、query value、候选、指令还是答案边界；全external强而各单区弱则表明是冗余分布式接收，而非单一复制边。

**问题硬伤与结论。** 干预仍覆盖九层、全部head和相应region全部投影坐标；recipient K/V donor同时改变该区的地址与内容；单区零翻转不构成自然不必要；400对来自行为资格筛选；R0/R1脚手架和候选格式仍在。该Phase定位region级控制迁移，不称最小坐标齿轮、自然语义闭合或知识链算法。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2554 / "analysis/final.json")
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        jobs, eligible_ids = compile_jobs(tokenizer)
        rows = run(model, tokenizer, jobs)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
    scores_path = OUT / "causal/region_stage_candidate_scores.jsonl"
    p2552.write(scores_path, rows)
    summary = summarize(rows)
    adjudication = {
        "old_early_value_stage_replicates": summary["early_v_facts_value"]["donor_flip"] >= .70,
        "relation_region_has_independent_control": max(summary["early_v_facts_relation"]["donor_flip"],
                                                       summary["middle_k_facts_relation"]["donor_flip"]) >= .20,
        "middle_value_kv_stage_replicates": summary["middle_kv_facts_value"]["donor_flip"] >= .50,
        "middlelate_external_stage_replicates": summary["middlelate_kv_external"]["donor_flip"] >= .70,
        "single_recipient_sufficient": max(summary[name]["donor_flip"] for name in (
            "middlelate_kv_query_context", "middlelate_kv_query_relation", "middlelate_kv_query_value",
            "middlelate_kv_candidate", "middlelate_kv_instruction", "middlelate_kv_answer_boundary")) >= .70,
        "late_q_stage_replicates": summary["late_q"]["donor_flip"] >= .70,
        "late_fact_kv_absent_at_this_dose": summary["late_kv_facts_all"]["donor_flip"] <= .10,
        "language_mechanism_closed": False,
    }
    checks = {"phase2554_passed": prior["all_checks_passed"], "eligible_400": len(eligible_ids) == 400,
              "candidate_jobs_800": len(jobs) == 800, "conditions_23": len(CONDITIONS) == 23,
              "all_scores_complete": len(rows) == len(jobs) * len(CONDITIONS),
              "baseline_lockbox": summary["no_patch"]["accuracy"] >= .99,
              "all_forms_reported": all(len(panel["donor_flip_by_form"]) == 4 for panel in summary.values()),
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized",
              "design": {"eligible_pairs": len(eligible_ids), "candidate_jobs": len(jobs),
                         "conditions": len(CONDITIONS), "layer_bands": [list(EARLY), list(MIDDLE), list(MIDDLELATE), list(LATE)]},
              "summary": summary, "adjudication": adjudication, "checks": checks,
              "all_checks_passed": all(checks.values()), "files": {"scores": str(scores_path)}}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
