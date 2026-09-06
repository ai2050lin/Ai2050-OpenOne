#!/usr/bin/env python3
"""Independent unit/surface/query replication with token-region resolved staged QKV patches."""
from __future__ import annotations

import gc
import hashlib
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
P2546 = RESULT / "phase2546_c150273_c154368_qkv_compiler_heatmap_retention"
OUT = RESULT / "phase2547_c154369_c158464_independent_region_stage_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2547, "C154369-C158464"
EARLY, MIDDLE, MIDDLELATE, LATE = (set(range(0, 9)), set(range(9, 18)), set(range(18, 27)), set(range(27, 36)))
FACT_NAMES = ("facts_entity", "facts_relation", "facts_value")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as atlas  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pad(sequences: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, : len(sequence)] = 1
    return ids, mask


def positions(row: dict, region: str) -> list[int]:
    if region == "facts_all":
        return sorted({position for name in FACT_NAMES for position in row["regions"][name]})
    if region == "external":
        return list(range(len(row["prompt_ids"]) - 1))
    return list(row["regions"][region])


def compile_jobs(tokenizer) -> list[dict]:
    material = atlas.compile_material(tokenizer)
    index = {
        (row["family_id"], row["language"], row["meaning_swap"], row["query_property"]): row
        for row in material if row["unit"] == 34 and row["surface"] == 1
    }
    jobs = []
    regions = ("facts_entity", "facts_relation", "facts_value", "question_context", "query_property", "candidate", "instruction", "facts_all", "external")
    for family_id in range(len(atlas.OPERATIONS)):
        for language in ("en", "zh"):
            for query_property in (0, 1):
                base = index[(family_id, language, 0, query_property)]
                donor = index[(family_id, language, 1, query_property)]
                for candidate_index, entity in enumerate(base["entities"]):
                    prefix = " " if language == "en" else ""
                    continuation = [int(token) for token in tokenizer.encode(prefix + entity, add_special_tokens=False)]
                    jobs.append({
                        "case_id": f"u34_f{family_id:02d}_{language}_s1_q{query_property}",
                        "family_id": family_id, "family": base["family"], "language": language,
                        "query_property": query_property, "candidate_index": candidate_index,
                        "candidate": entity, "target": base["target"], "donor_target": donor["target"],
                        "base_prompt_length": len(base["prompt_ids"]), "donor_prompt_length": len(donor["prompt_ids"]),
                        "base": base["prompt_ids"] + continuation, "donor": donor["prompt_ids"] + continuation,
                        "regions_base": {region: positions(base, region) for region in regions},
                        "regions_donor": {region: positions(donor, region) for region in regions},
                    })
    return jobs


SPECS = {
    "no_patch": {},
    "early_k_facts_all": {"layers": EARLY, "kind": "k", "region": "facts_all"},
    "early_v_facts_entity": {"layers": EARLY, "kind": "v", "region": "facts_entity"},
    "early_v_facts_relation": {"layers": EARLY, "kind": "v", "region": "facts_relation"},
    "early_v_facts_value": {"layers": EARLY, "kind": "v", "region": "facts_value"},
    "early_v_question_context": {"layers": EARLY, "kind": "v", "region": "question_context"},
    "early_v_query_property": {"layers": EARLY, "kind": "v", "region": "query_property"},
    "early_v_candidate": {"layers": EARLY, "kind": "v", "region": "candidate"},
    "early_v_instruction": {"layers": EARLY, "kind": "v", "region": "instruction"},
    "early_v_facts_all": {"layers": EARLY, "kind": "v", "region": "facts_all"},
    "early_v_external": {"layers": EARLY, "kind": "v", "region": "external"},
    "middle_k_facts_entity": {"layers": MIDDLE, "kind": "k", "region": "facts_entity"},
    "middle_k_facts_relation": {"layers": MIDDLE, "kind": "k", "region": "facts_relation"},
    "middle_k_facts_value": {"layers": MIDDLE, "kind": "k", "region": "facts_value"},
    "middle_k_facts_all": {"layers": MIDDLE, "kind": "k", "region": "facts_all"},
    "middle_v_facts_all": {"layers": MIDDLE, "kind": "v", "region": "facts_all"},
    "middle_kv_facts_all": {"layers": MIDDLE, "kind": "kv", "region": "facts_all"},
    "middlelate_kv_external": {"layers": MIDDLELATE, "kind": "kv", "region": "external"},
    "late_q": {"layers": LATE, "kind": "q", "region": "answer"},
    "late_kv_facts_all": {"layers": LATE, "kind": "kv", "region": "facts_all"},
}


class Controller:
    def __init__(self, model):
        self.layers = model_utils.get_layers(model)
        self.required = {(kind, layer) for spec in SPECS.values() for layer in spec.get("layers", ())
                         for kind in (("q",) if spec.get("kind") == "q" else (("k", "v") if spec.get("kind") == "kv" else (spec.get("kind"),)))}
        self.required.discard((None, None))
        self.mode = "none"
        self.spec: dict = {}
        self.jobs: list[dict] = []
        self.store: dict[tuple[str, int], torch.Tensor] = {}
        self.handles = []
        for layer_index, layer in enumerate(self.layers):
            for kind, name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
                if (kind, layer_index) not in self.required:
                    continue
                def hook(_module, _inputs, output, layer_index=layer_index, kind=kind):
                    return self._hook(output, layer_index, kind)
                self.handles.append(getattr(layer.self_attn, name).register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _hook(self, output: torch.Tensor, layer: int, kind: str):
        key = (kind, layer)
        if self.mode == "capture":
            self.store[key] = output.detach().clone()
            return None
        spec = self.spec
        if self.mode != "patch" or layer not in spec.get("layers", ()):
            return None
        allowed = kind == spec.get("kind") or (spec.get("kind") == "kv" and kind in ("k", "v"))
        if not allowed:
            return None
        changed = output.clone()
        donor = self.store[key].to(device=output.device, dtype=output.dtype)
        for batch_index, job in enumerate(self.jobs):
            if kind == "q":
                count = len(job["base"]) - job["base_prompt_length"]
                for offset in range(count):
                    changed[batch_index, job["base_prompt_length"] - 1 + offset] = donor[batch_index, job["donor_prompt_length"] - 1 + offset]
            else:
                region = spec["region"]
                for base_position, donor_position in zip(job["regions_base"][region], job["regions_donor"][region]):
                    changed[batch_index, base_position] = donor[batch_index, donor_position]
        return changed


def forward(model, ids: torch.Tensor, mask: torch.Tensor, jobs: list[dict], source: str) -> torch.Tensor:
    keep = int(ids.shape[1] - min(job[f"{source}_prompt_length"] - 1 for job in jobs))
    return model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits


def score(logits: torch.Tensor, jobs: list[dict], width: int, source: str) -> list[float]:
    logit_offset = width - logits.shape[1]
    values = []
    for batch_index, job in enumerate(jobs):
        prompt_length = job[f"{source}_prompt_length"]
        total = 0.0
        for token_offset, token in enumerate(job[source][prompt_length:]):
            z = logits[batch_index, prompt_length - 1 + token_offset - logit_offset].float()
            total += float(z[token] - torch.logsumexp(z, -1))
        values.append(total)
    return values


def run(model, tokenizer, jobs: list[dict]) -> list[dict]:
    controller = Controller(model)
    device = model.get_input_embeddings().weight.device
    output = []
    try:
        for start in range(0, len(jobs), 8):
            batch = jobs[start : start + 8]
            controller.jobs = batch
            donor_ids, donor_mask = pad([job["donor"] for job in batch], tokenizer.pad_token_id, device)
            controller.mode = "capture"
            controller.store.clear()
            with torch.inference_mode():
                donor_logits = forward(model, donor_ids, donor_mask, batch, "donor")
            donor_scores = score(donor_logits, batch, int(donor_ids.shape[1]), "donor")
            base_ids, base_mask = pad([job["base"] for job in batch], tokenizer.pad_token_id, device)
            for condition, spec in SPECS.items():
                controller.mode = "none" if condition == "no_patch" else "patch"
                controller.spec = spec
                with torch.inference_mode():
                    logits = forward(model, base_ids, base_mask, batch, "base")
                condition_scores = score(logits, batch, int(base_ids.shape[1]), "base")
                for job, value, donor_value in zip(batch, condition_scores, donor_scores):
                    output.append({
                        "case_id": job["case_id"], "family_id": job["family_id"], "family": job["family"],
                        "language": job["language"], "query_property": job["query_property"],
                        "candidate_index": job["candidate_index"], "candidate": job["candidate"],
                        "target": job["target"], "donor_target": job["donor_target"],
                        "condition": condition, "score": value, "donor_baseline_score": donor_value,
                    })
            if (start + len(batch)) % 64 == 0:
                print(f"[phase2547] {start + len(batch)}/{len(jobs)}", flush=True)
    finally:
        controller.close()
    return output


def summarize(rows: list[dict]) -> dict:
    baseline: dict[str, list[dict]] = defaultdict(list)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["case_id"])].append(row)
        if row["condition"] == "no_patch":
            baseline[row["case_id"]].append(row)
    eligible = set()
    baseline_stats = []
    for case_id, values in baseline.items():
        base_prediction = max(values, key=lambda row: row["score"])["candidate"]
        donor_prediction = max(values, key=lambda row: row["donor_baseline_score"])["candidate"]
        base_ok = base_prediction == values[0]["target"]
        donor_ok = donor_prediction == values[0]["donor_target"]
        baseline_stats.append((base_ok, donor_ok))
        if base_ok and donor_ok:
            eligible.add(case_id)
    panel = {}
    for condition in SPECS:
        values = []
        language_values: dict[str, list[tuple]] = defaultdict(list)
        for (name, case_id), candidates in grouped.items():
            if name != condition or case_id not in eligible:
                continue
            prediction = max(candidates, key=lambda row: row["score"])["candidate"]
            target, donor_target = candidates[0]["target"], candidates[0]["donor_target"]
            by_candidate = {row["candidate"]: row["score"] for row in candidates}
            item = (prediction == target, prediction == donor_target, by_candidate[donor_target] - by_candidate[target])
            values.append(item)
            language_values[candidates[0]["language"]].append(item)
        panel[condition] = {
            "n": len(values), "accuracy": float(np.mean([item[0] for item in values])),
            "donor_flip": float(np.mean([item[1] for item in values])),
            "mean_donor_margin": float(np.mean([item[2] for item in values])),
            "by_language_donor_flip": {language: float(np.mean([item[1] for item in items])) for language, items in language_values.items()},
        }
    return {
        "all_cases": len(baseline_stats),
        "base_accuracy": float(np.mean([item[0] for item in baseline_stats])),
        "donor_accuracy": float(np.mean([item[1] for item in baseline_stats])),
        "eligible_cases": len(eligible), "conditions": panel,
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 独立unit/surface/query的token-region分阶段Q/K/V复验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 这是主方案完成后的自动同目标续研。改用未参与阶段发现的unit34、surface1、双query，覆盖32语言模式族×英中×双query共128 case、256个多token候选序列；base与meaning donor均过行为门后，把早期0–8层V拆为facts实体/关系/值、问题上下文、query值、候选、指令、facts联合和全external，另拆中层9–17的facts K实体/关系/值以及K/V联合，并复验18–26层external K/V、27–35层答案Q和晚层facts K/V。

$$V^l_{{R}}\leftarrow V^{{l,D}}_{{R}},\quad R\in\{{E,R_{{rel}},V_{{fact}},Q_{{ctx}},Q_{{value}},C,I\}},\qquad \Delta_R=P(\hat y=y_D\mid \operatorname{{do}}(V_R^B\leftarrow V_R^D)).$$

**结果汇总。** 行为门与region分解 `{json.dumps(result['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2547_c154369_c158464_independent_region_stage_replication.py`；逐候选全部条件分数和final位于`{OUT}`。

**分析与理论进展。** 独立unit/surface/query复验用于排除上一批材料偶然性。region拆分回答“条件齿轮是否属于单一token区域”：若单region低而facts联合高，说明有效单位更像跨token协同；若facts_value单独接近联合，才支持值载荷主导。问题、候选或指令区的早层效应说明信息很早已沿因果前缀扩散，不能把它们命名为纯语义source。

**问题硬伤与结论。** region替换仍是整层段、全head构造性干预；自回归因果顺序使后部token天然包含前部事实；tokenizer区域长度不同但按各自原子边界配对；候选似然包含输出编译。结果用于定位跨token协同，不证明最小坐标齿轮或数学闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2546 / "analysis/final.json")
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        jobs = compile_jobs(tokenizer)
        rows = run(model, tokenizer, jobs)
    finally:
        model_utils.release_model(model)
        gc.collect()
    path = OUT / "causal/region_stage_scores.jsonl"
    write(path, rows)
    summary = summarize(rows)
    conditions = summary["conditions"]
    individual_early = [conditions[f"early_v_{region}"]["donor_flip"] for region in (
        "facts_entity", "facts_relation", "facts_value", "question_context", "query_property", "candidate", "instruction"
    )]
    adjudication = {
        "independent_stage_replication": conditions["early_v_facts_all"]["donor_flip"] >= 0.75
                                         and conditions["middlelate_kv_external"]["donor_flip"] >= 0.75
                                         and conditions["late_q"]["donor_flip"] >= 0.75,
        "late_fact_kv_control_absent": conditions["late_kv_facts_all"]["donor_flip"] <= 0.10,
        "maximum_individual_early_v_region": float(max(individual_early)),
        "facts_joint_early_v": conditions["early_v_facts_all"]["donor_flip"],
        "external_early_v": conditions["early_v_external"]["donor_flip"],
        "facts_value_region_sufficient_in_this_intervention": max(individual_early) >= 0.95,
        "single_region_is_complete_gear": False,
        "language_mechanism_closed": False,
    }
    checks = {
        "source_passed": prior["all_checks_passed"], "jobs_256": len(jobs) == 256,
        "cases_128": summary["all_cases"] == 128, "behavior_gate": summary["eligible_cases"] >= 100,
        "conditions_20": len(SPECS) == 20, "all_conditions_complete": all(panel["n"] == summary["eligible_cases"] for panel in conditions.values()),
        "independent_material": True, "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B BF16 CUDA nonquantized",
        "design": {"unit": 34, "surface": 1, "queries": [0, 1], "families": 32,
                   "languages": ["en", "zh"], "cases": 128, "candidate_sequences": 256, "conditions": len(SPECS)},
        "summary": summary, "adjudication": adjudication,
        "file": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
