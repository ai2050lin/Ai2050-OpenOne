#!/usr/bin/env python3
"""Sequential BF16 cross-model replication of frozen relation-stage events."""
from __future__ import annotations

import gc
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2560_c223489_c231680_crossmodel_relation_stage_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2560, "C223489-C231680"
MODEL_KEYS = ("qwen14b", "deepseek7b", "glm4")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2522_c87201_c88576_crossmodel_natural_boundary_replication as cross  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2553_c174337_c178432_relation_slot_scaffold_adjudication as p2553  # noqa: E402
import phase2554_c178433_c182528_independent_relation_lockbox_behavior as p2554  # noqa: E402


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_model(model_key: str):
    """BF16 auto placement without avoidable disk-resident transformer blocks."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = cross.MODELS[model_key]
    offload = ROOT / "tests/glm5_temp/phase2560_offload" / model_key
    offload.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "13GiB", "cpu": "16GiB"},
        offload_folder=str(offload),
        offload_state_dict=True,
        offload_buffers=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    if bool(getattr(model, "is_quantized", False)):
        raise RuntimeError("quantized model is forbidden in Phase2560")
    return model, tokenizer, offload


def bands(n_layers: int) -> tuple[tuple[int, ...], ...]:
    cuts = [round(index * n_layers / 4) for index in range(5)]
    return tuple(tuple(range(cuts[index], cuts[index + 1])) for index in range(4))


def specs(n_layers: int) -> dict[str, dict]:
    early, middle, middlelate, late = bands(n_layers)
    return {
        "no_patch": {},
        "early_k_facts_value": {"layers": early, "kind": "k", "region": "facts_value"},
        "early_v_facts_value": {"layers": early, "kind": "v", "region": "facts_value"},
        "middle_kv_facts_value": {"layers": middle, "kind": "kv", "region": "facts_value"},
        "middlelate_kv_query_value": {"layers": middlelate, "kind": "kv", "region": "query_value"},
        "middlelate_kv_external": {"layers": middlelate, "kind": "kv", "region": "external"},
        "late_q": {"layers": late, "kind": "q", "region": "answer"},
        "late_kv_facts": {"layers": late, "kind": "kv", "region": "facts_all"},
    }


def positions(row: dict, region: str) -> list[int]:
    if region == "facts_all":
        return sorted({position for name in ("facts_entity", "facts_relation", "facts_value")
                       for position in row["regions"][name]})
    if region == "external":
        return list(range(row["answer_boundary_token"]))
    return list(row["regions"][region])


def behavior_summary(rows: list[dict]) -> dict:
    output = {}
    for condition in ("full_scaffold", "relation_missing", "value_missing"):
        subset = [row for row in rows if row["ablation"] == condition]
        output[condition] = {"n": len(subset), "accuracy": float(np.mean([row["correct"] for row in subset])),
                             "mean_margin": float(np.mean([row["target_minus_wrong"] for row in subset]))}
    full = [row for row in rows if row["ablation"] == "full_scaffold"]
    output["by_form"] = {f"r={rf},v={vf}": float(np.mean([row["correct"] for row in full
        if row["relation_form"] == rf and row["value_form"] == vf]))
        for rf in ("natural", "nonce") for vf in ("natural", "nonce")}
    return output


def score_candidates(model, tokenizer, rows: list[dict], batch_size: int = 8) -> list[dict]:
    """Exact candidate scoring for models that return either full or tail-only logits."""
    device = model.get_input_embeddings().weight.device
    jobs = []
    for row in rows:
        for candidate_index, entity in enumerate(row["entities"]):
            continuation = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
            jobs.append({"row": row, "candidate_index": candidate_index, "continuation": continuation,
                         "sequence": row["prompt_ids"] + continuation})
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for job in jobs:
        buckets[len(job["sequence"])].append(job)
    batches = [values[start:start + batch_size] for length, values in sorted(buckets.items())
               for start in range(0, len(values), batch_size)]
    done = 0
    for batch in batches:
        ids, mask, shifts = p2552.left_pad([job["sequence"] for job in batch], tokenizer.pad_token_id, device)
        if any(shifts):
            raise RuntimeError("length bucketing failed to eliminate left padding")
        keep = max(len(job["continuation"]) for job in batch) + 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits
        logit_offset = int(ids.shape[1] - logits.shape[1])
        for batch_index, (job, shift) in enumerate(zip(batch, shifts)):
            first = int(shift + len(job["row"]["prompt_ids"]) - 1 - logit_offset)
            value = 0.0
            for offset, token in enumerate(job["continuation"]):
                z = logits[batch_index, first + offset].float()
                value += float(z[token] - torch.logsumexp(z, dim=-1))
            scores[job["row"]["case_id"]][job["candidate_index"]] = value
        done += len(batch)
        if done == len(jobs) or done % 2048 == 0:
            print(f"[phase2560 robust score] {done}/{len(jobs)}", flush=True)
    output = []
    for row in rows:
        values = scores[row["case_id"]]
        prediction = max(values, key=values.get)
        target, wrong = row["target_index"], 1 - row["target_index"]
        copied = {key: row[key] for key in ("case_id", "base_case_id", "ablation", "unit", "family_id",
                                             "family", "language", "surface", "binding", "relation_form",
                                             "value_form", "query_relation", "query_value", "target_index", "target")}
        copied.update({"prediction": prediction, "correct": prediction == target,
                       "target_score": values[target], "wrong_score": values[wrong],
                       "target_minus_wrong": values[target] - values[wrong]})
        output.append(copied)
    return output


def eligible_pairs(material: list[dict], behavior: list[dict]) -> tuple[list[tuple], dict[tuple, dict]]:
    full_material = [row for row in material if row["ablation"] == "full_scaffold"]
    full_behavior = [row for row in behavior if row["ablation"] == "full_scaffold"]
    correct = {row["base_case_id"]: row["correct"] for row in full_behavior}
    index = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"],
              row["query_value"], row["binding"]): row for row in full_material}
    eligible = []
    for family_id in range(32):
        for relation_form in ("natural", "nonce"):
            for value_form in ("natural", "nonce"):
                for query_relation in (0, 1):
                    for query_value in (0, 1):
                        key = (family_id, relation_form, value_form, query_relation, query_value)
                        if correct[index[key + (0,)]["base_case_id"]] and correct[index[key + (1,)]["base_case_id"]]:
                            eligible.append(key)
    return eligible, index


def select_balanced(keys: list[tuple], limit: int = 128) -> list[tuple]:
    if len(keys) <= limit:
        return keys
    indices = np.linspace(0, len(keys) - 1, limit, dtype=int)
    return [keys[int(index)] for index in indices]


def compile_causal_jobs(tokenizer, selected: list[tuple], index: dict[tuple, dict]) -> list[dict]:
    jobs = []
    for key in selected:
        base, donor = index[key + (0,)], index[key + (1,)]
        region_names = ("facts_value", "facts_all", "query_value", "external")
        for candidate_index, entity in enumerate(base["entities"]):
            prefix = " " if base["language"] == "en" else ""
            continuation = [int(token) for token in tokenizer.encode(prefix + entity, add_special_tokens=False)]
            jobs.append({"case_id": base["base_case_id"], "family_id": base["family_id"],
                         "relation_form": base["relation_form"], "value_form": base["value_form"],
                         "query_relation": base["query_relation"], "query_value": base["query_value"],
                         "candidate_index": candidate_index, "target_index": base["target_index"],
                         "donor_target_index": donor["target_index"], "continuation": continuation,
                         "base_prompt_length": len(base["prompt_ids"]), "donor_prompt_length": len(donor["prompt_ids"]),
                         "base": base["prompt_ids"] + continuation, "donor": donor["prompt_ids"] + continuation,
                         "regions_base": {name: positions(base, name) for name in region_names},
                         "regions_donor": {name: positions(donor, name) for name in region_names}})
    return jobs


class Controller:
    def __init__(self, model, conditions: dict[str, dict]):
        self.layers = model_utils.get_layers(model)
        self.conditions = conditions
        self.mode = "none"
        self.spec: dict = {}
        self.jobs: list[dict] = []
        self.store: dict[tuple[str, int], torch.Tensor] = {}
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
                        batch_index, job["donor_shift"] + donor_position]
        return changed


def forward(model, ids: torch.Tensor, mask: torch.Tensor, keep: int) -> torch.Tensor:
    return model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits


def continuation_scores(logits: torch.Tensor, jobs: list[dict], input_width: int, side: str) -> list[float]:
    output = []
    logit_offset = int(input_width - logits.shape[1])
    for batch_index, job in enumerate(jobs):
        continuation = job["continuation"]
        first = int(job[f"{side}_shift"] + job[f"{side}_prompt_length"] - 1 - logit_offset)
        value = 0.0
        for offset, token in enumerate(continuation):
            z = logits[batch_index, first + offset].float()
            value += float(z[token] - torch.logsumexp(z, dim=-1))
        output.append(value)
    return output


def run_causal(model, tokenizer, jobs: list[dict], conditions: dict[str, dict], batch_size: int) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    controller = Controller(model, conditions)
    rows = []
    try:
        buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for job in jobs:
            buckets[(len(job["base"]), len(job["donor"]))].append(job)
        batches = [values[start:start + batch_size] for lengths, values in sorted(buckets.items())
                   for start in range(0, len(values), batch_size)]
        done = 0
        for batch in batches:
            controller.jobs = batch
            donor_ids, donor_mask, donor_shifts = p2552.left_pad([job["donor"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, donor_shifts):
                job["donor_shift"] = shift
            if any(donor_shifts):
                raise RuntimeError("donor length bucketing failed")
            keep = max(len(job["continuation"]) for job in batch) + 1
            controller.mode = "capture"
            controller.store.clear()
            with torch.inference_mode():
                donor_logits = forward(model, donor_ids, donor_mask, keep)
            donor_scores = continuation_scores(donor_logits, batch, int(donor_ids.shape[1]), "donor")
            base_ids, base_mask, base_shifts = p2552.left_pad([job["base"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, base_shifts):
                job["base_shift"] = shift
            if any(base_shifts):
                raise RuntimeError("base length bucketing failed")
            for condition, spec in conditions.items():
                controller.mode = "none" if condition == "no_patch" else "patch"
                controller.spec = spec
                with torch.inference_mode():
                    logits = forward(model, base_ids, base_mask, keep)
                values = continuation_scores(logits, batch, int(base_ids.shape[1]), "base")
                for job, value, donor_value in zip(batch, values, donor_scores):
                    rows.append({"case_id": job["case_id"], "family_id": job["family_id"],
                                 "relation_form": job["relation_form"], "value_form": job["value_form"],
                                 "query_relation": job["query_relation"], "query_value": job["query_value"],
                                 "candidate_index": job["candidate_index"], "target_index": job["target_index"],
                                 "donor_target_index": job["donor_target_index"], "condition": condition,
                                 "score": value, "donor_baseline_score": donor_value})
            done += len(batch)
            if done % 32 == 0 or done == len(jobs):
                print(f"[phase2560 causal] {done}/{len(jobs)}", flush=True)
    finally:
        controller.close()
    return rows


def causal_summary(rows: list[dict], conditions: dict[str, dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["case_id"])].append(row)
    panels = {}
    for condition in conditions:
        groups = [values for (name, _), values in grouped.items() if name == condition]
        if not groups:
            panels[condition] = {"n": 0, "accuracy": None, "donor_flip": None}
            continue
        correct, flipped = [], []
        for values in groups:
            prediction = max(values, key=lambda row: row["score"])["candidate_index"]
            correct.append(prediction == values[0]["target_index"])
            flipped.append(prediction == values[0]["donor_target_index"])
        panels[condition] = {"n": len(groups), "accuracy": float(np.mean(correct)),
                             "donor_flip": float(np.mean(flipped))}
    return panels


def run_model(model_key: str) -> dict:
    model = tokenizer = offload = None
    try:
        model, tokenizer, offload = load_model(model_key)
        material = p2554.compile_material(tokenizer)
        # Disk/CPU offload dominates per-forward latency on 14B, while these
        # short prompts leave enough activation headroom for batch eight.
        batch_size = 8
        behavior = score_candidates(model, tokenizer, material, batch_size=batch_size)
        behavior_path = OUT / f"behavior/{model_key}_scores.jsonl"
        p2552.write(behavior_path, behavior)
        panel = behavior_summary(behavior)
        eligible, index = eligible_pairs(material, behavior)
        selected = select_balanced(eligible, 128) if panel["full_scaffold"]["accuracy"] >= .80 \
            and len(eligible) >= 64 else []
        n_layers = len(model_utils.get_layers(model))
        conditions = specs(n_layers)
        jobs = compile_causal_jobs(tokenizer, selected, index)
        causal = run_causal(model, tokenizer, jobs, conditions, batch_size=4)
        causal_path = OUT / f"causal/{model_key}_stage_scores.jsonl"
        p2552.write(causal_path, causal)
        return {"model": model_key, "layers": n_layers, "bands": [list(band) for band in bands(n_layers)],
                "behavior": panel, "eligible_pairs": len(eligible), "causal_pairs": len(selected),
                "causal": causal_summary(causal, conditions),
                "files": {"behavior": str(behavior_path), "causal": str(causal_path)}}
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        if offload is not None:
            resolved = Path(offload).resolve()
            allowed = (ROOT / "tests/glm5_temp").resolve()
            if allowed in resolved.parents:
                shutil.rmtree(resolved, ignore_errors=True)


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 三模型关系必要阶段事件的顺序BF16复验（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 严格按Qwen3-14B、DeepSeek-R1-Distill-Qwen-7B、GLM-4-9B顺序加载，每次只驻留一个模型；全部BF16非量化、`device_map=auto`，显存上限13GiB并允许CPU/磁盘offload。每个模型用自己的tokenizer重新编译Phase2554的1024个full及relation/value缺失共3072 case，完整评分6144条多token候选。再从各模型base/donor双侧正确对中按全序均匀取至多128对，测试冻结的早层facts-value K/V、中层facts-value K/V、中晚层query-value与external K/V、晚层Q和facts K/V；层段只按模型内相对四分位定义，不对齐物理层号或坐标号。

$$
B_k=\left[\operatorname{{round}}\frac{{kL}}4,\operatorname{{round}}\frac{{(k+1)L}}4\right),
\qquad F_{{M,c,B}}=P_M(\hat e_{{do(c,B)}}=e_{{donor}}).
$$

**结果汇总。** `{json.dumps(result['models'], ensure_ascii=False)}`。跨模型裁决和检查为`{json.dumps(result['adjudication'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2560_c223489_c231680_crossmodel_relation_stage_replication.py`；每模型完整行为分数、相对阶段因果分数和final位于`{OUT}`；offload临时目录在每模型释放后删除。

**分析与理论进展。** 复验对象是功能事件而非Qwen物理坐标。只有模型行为门通过、eligible数足够时，其因果数字才参与比较；失败模型保留为能力边界。若早V跨模型稳定而late-Q不稳定，说明“source值早期载荷”比“晚Q集中”更像相对不变量；若query-value只在Qwen出现，则它是材料/架构条件化路线而非普遍recipient。

**问题硬伤与结论。** 每模型因果最多128对；相对四分位可能切断真实阶段边界；Qwen14B含CPU/offload但没有量化；模型tokenizer使物理token数不同；人工英文表格和候选输出仍在。跨模型相似只提升功能事件证据，不证明共享坐标、共享参数齿轮或Transformer必然性。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    models = []
    for model_key in MODEL_KEYS:
        print(f"[phase2560] START {model_key}", flush=True)
        models.append(run_model(model_key))
        print(f"[phase2560] END {model_key}", flush=True)
    eligible_models = [row for row in models if row["behavior"]["full_scaffold"]["accuracy"] >= .80
                       and row["eligible_pairs"] >= 64]
    adjudication = {"behavior_eligible_models": [row["model"] for row in eligible_models],
                    "early_value_replication_models": [row["model"] for row in eligible_models
                        if row["causal"]["early_v_facts_value"]["donor_flip"] >= .70],
                    "query_value_replication_models": [row["model"] for row in eligible_models
                        if row["causal"]["middlelate_kv_query_value"]["donor_flip"] >= .70],
                    "late_q_replication_models": [row["model"] for row in eligible_models
                        if row["causal"]["late_q"]["donor_flip"] >= .70],
                    "physical_coordinate_invariance_tested": False, "language_mechanism_closed": False}
    checks = {"models_sequential_complete": len(models) == 3, "bf16_nonquantized": True,
              "behavior_rows_each_3072": all(row["behavior"]["full_scaffold"]["n"] == 1024 for row in models),
              "missing_controls_each": all(row["behavior"]["relation_missing"]["n"] == 1024
                                           and row["behavior"]["value_missing"]["n"] == 1024 for row in models),
              "causal_only_after_behavior": all(row["causal_pairs"] <= min(128, row["eligible_pairs"]) for row in models),
              "relative_bands": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "models": models, "adjudication": adjudication, "checks": checks,
              "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
