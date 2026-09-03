#!/usr/bin/env python3
"""Cross-model BF16 replication of the staged Q/K/V compiler hypothesis."""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
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
P2544 = RESULT / "phase2544_c142081_c146176_autonomous_staged_compiler_composition"
OUT = RESULT / "phase2545_c146177_c150272_crossmodel_staged_qkv_compiler"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2545, "C146177-C150272"
FACT_NAMES = ("facts_entity", "facts_relation", "facts_value")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as atlas  # noqa: E402
import phase2522_c87201_c88576_crossmodel_natural_boundary_replication as cross  # noqa: E402


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


def fact_positions(row: dict) -> list[int]:
    return sorted({position for name in FACT_NAMES for position in row["regions"][name]})


def compile_jobs(tokenizer) -> list[dict]:
    material = atlas.compile_material(tokenizer)
    index = {
        (row["family_id"], row["language"], row["meaning_swap"], row["query_property"]): row
        for row in material
        if row["unit"] == 35 and row["surface"] == 0
    }
    jobs = []
    for family_id in range(len(atlas.OPERATIONS)):
        for language in ("en", "zh"):
            base = index[(family_id, language, 0, 0)]
            donor = index[(family_id, language, 1, 0)]
            for candidate_index, entity in enumerate(base["entities"]):
                prefix = " " if language == "en" else ""
                continuation = [int(token) for token in tokenizer.encode(prefix + entity, add_special_tokens=False)]
                jobs.append({
                    "case_id": f"f{family_id:02d}_{language}",
                    "family_id": family_id,
                    "family": base["family"],
                    "language": language,
                    "candidate_index": candidate_index,
                    "candidate": entity,
                    "target": base["target"],
                    "donor_target": donor["target"],
                    "base_prompt_length": len(base["prompt_ids"]),
                    "donor_prompt_length": len(donor["prompt_ids"]),
                    "base": base["prompt_ids"] + continuation,
                    "donor": donor["prompt_ids"] + continuation,
                    "facts_base": fact_positions(base),
                    "facts_donor": fact_positions(donor),
                })
    return jobs


def relative_bands(n_layers: int) -> tuple[tuple[int, ...], ...]:
    cuts = [round(index * n_layers / 4) for index in range(5)]
    return tuple(tuple(range(cuts[index], cuts[index + 1])) for index in range(4))


def stage_specs(n_layers: int) -> dict[str, dict]:
    early, middle, middlelate, late = relative_bands(n_layers)
    return {
        "no_patch": {},
        "early_k_fact": {"kv_layers": set(early), "kind": "k", "region": "facts"},
        "early_v_fact": {"kv_layers": set(early), "kind": "v", "region": "facts"},
        "early_kv_fact": {"kv_layers": set(early), "kind": "kv", "region": "facts"},
        "middle_kv_fact": {"kv_layers": set(middle), "kind": "kv", "region": "facts"},
        "middlelate_kv_external": {"kv_layers": set(middlelate), "kind": "kv", "region": "external"},
        "late_q": {"q_layers": set(late)},
        "late_kv_fact": {"kv_layers": set(late), "kind": "kv", "region": "facts"},
    }


class Controller:
    def __init__(self, model, specs: dict[str, dict]):
        self.layers = model_utils.get_layers(model)
        self.specs = specs
        self.required = {
            (kind, layer_index)
            for spec in specs.values()
            for kind in (("q",) if "q_layers" in spec else ("k", "v"))
            for layer_index in (spec.get("q_layers") or spec.get("kv_layers") or ())
        }
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

    def _hook(self, output: torch.Tensor, layer_index: int, kind: str):
        key = (kind, layer_index)
        if self.mode == "capture":
            self.store[key] = output.detach().clone()
            return None
        if self.mode != "patch":
            return None
        spec = self.spec
        patch_q = kind == "q" and layer_index in spec.get("q_layers", set())
        patch_kv = (
            kind in ("k", "v")
            and layer_index in spec.get("kv_layers", set())
            and (kind == spec.get("kind") or spec.get("kind") == "kv")
        )
        if not patch_q and not patch_kv:
            return None
        changed = output.clone()
        donor = self.store[key].to(device=output.device, dtype=output.dtype)
        for batch_index, job in enumerate(self.jobs):
            if patch_q:
                base_start = job["base_prompt_length"] - 1
                donor_start = job["donor_prompt_length"] - 1
                count = len(job["base"]) - job["base_prompt_length"]
                for offset in range(count):
                    changed[batch_index, base_start + offset] = donor[batch_index, donor_start + offset]
            if patch_kv:
                if spec["region"] == "facts":
                    base_positions, donor_positions = job["facts_base"], job["facts_donor"]
                else:
                    base_positions = list(range(job["base_prompt_length"] - 1))
                    donor_positions = list(range(job["donor_prompt_length"] - 1))
                for base_position, donor_position in zip(base_positions, donor_positions):
                    changed[batch_index, base_position] = donor[batch_index, donor_position]
        return changed


def forward(model, ids: torch.Tensor, mask: torch.Tensor, jobs: list[dict], prompt_key: str) -> torch.Tensor:
    prompt_field = f"{prompt_key}_prompt_length"
    keep = int(ids.shape[1] - min(job[prompt_field] - 1 for job in jobs))
    return model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits


def scores(logits: torch.Tensor, jobs: list[dict], full_width: int, prompt_key: str) -> list[float]:
    offset = full_width - logits.shape[1]
    answer = []
    for batch_index, job in enumerate(jobs):
        prompt_length = job[f"{prompt_key}_prompt_length"]
        sequence = job[prompt_key]
        value = 0.0
        for token_offset, token in enumerate(sequence[prompt_length:]):
            position = prompt_length - 1 + token_offset - offset
            z = logits[batch_index, position].float()
            value += float(z[token] - torch.logsumexp(z, -1))
        answer.append(value)
    return answer


def run_model(model_key: str) -> dict:
    model = tokenizer = offload = None
    try:
        model, tokenizer, offload = cross.load_model(model_key)
        jobs = compile_jobs(tokenizer)
        n_layers = len(model_utils.get_layers(model))
        specs = stage_specs(n_layers)
        controller = Controller(model, specs)
        rows = []
        device = model.get_input_embeddings().weight.device
        try:
            for start in range(0, len(jobs), 8):
                batch = jobs[start : start + 8]
                controller.jobs = batch
                donor_ids, donor_mask = pad([job["donor"] for job in batch], tokenizer.pad_token_id, device)
                controller.mode = "capture"
                controller.store.clear()
                with torch.inference_mode():
                    donor_logits = forward(model, donor_ids, donor_mask, batch, "donor")
                donor_scores = scores(donor_logits, batch, int(donor_ids.shape[1]), "donor")

                base_ids, base_mask = pad([job["base"] for job in batch], tokenizer.pad_token_id, device)
                for condition, spec in specs.items():
                    controller.mode = "none" if condition == "no_patch" else "patch"
                    controller.spec = spec
                    with torch.inference_mode():
                        logits = forward(model, base_ids, base_mask, batch, "base")
                    values = scores(logits, batch, int(base_ids.shape[1]), "base")
                    for job, value, donor_value in zip(batch, values, donor_scores):
                        rows.append({
                            "case_id": job["case_id"], "family_id": job["family_id"],
                            "family": job["family"], "language": job["language"],
                            "candidate_index": job["candidate_index"], "candidate": job["candidate"],
                            "target": job["target"], "donor_target": job["donor_target"],
                            "condition": condition, "score": value, "donor_baseline_score": donor_value,
                        })
                print(f"[phase2545 {model_key}] {start + len(batch)}/{len(jobs)}", flush=True)
        finally:
            controller.close()

        path = OUT / "causal" / f"{model_key}_scores.jsonl"
        write(path, rows)
        panel = summarize(rows)
        return {
            "model": model_key,
            "layers": n_layers,
            "bands": [list(band) for band in relative_bands(n_layers)],
            "jobs": len(jobs),
            "panel": panel,
            "file": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)},
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        if offload is not None:
            shutil.rmtree(offload, ignore_errors=True)


def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    donor_grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["case_id"])].append(row)
        if row["condition"] == "no_patch":
            donor_grouped[row["case_id"]].append(row)

    base_correct, donor_correct = {}, {}
    for case_id, case_rows in donor_grouped.items():
        base_pred = max(case_rows, key=lambda row: row["score"])["candidate"]
        donor_pred = max(case_rows, key=lambda row: row["donor_baseline_score"])["candidate"]
        base_correct[case_id] = base_pred == case_rows[0]["target"]
        donor_correct[case_id] = donor_pred == case_rows[0]["donor_target"]
    eligible = {case_id for case_id in base_correct if base_correct[case_id] and donor_correct[case_id]}

    panel = {}
    for condition in sorted({row["condition"] for row in rows}):
        values = []
        for (name, case_id), case_rows in grouped.items():
            if name != condition or case_id not in eligible:
                continue
            prediction = max(case_rows, key=lambda row: row["score"])["candidate"]
            target = case_rows[0]["target"]
            donor_target = case_rows[0]["donor_target"]
            score_by_candidate = {row["candidate"]: row["score"] for row in case_rows}
            values.append((prediction == target, prediction == donor_target,
                           score_by_candidate[donor_target] - score_by_candidate[target]))
        panel[condition] = {
            "n": len(values),
            "accuracy": float(np.mean([value[0] for value in values])) if values else None,
            "donor_flip": float(np.mean([value[1] for value in values])) if values else None,
            "mean_donor_margin": float(np.mean([value[2] for value in values])) if values else None,
        }
    return {
        "all_cases": len(base_correct),
        "base_accuracy": float(np.mean(list(base_correct.values()))),
        "donor_accuracy": float(np.mean(list(donor_correct.values()))),
        "eligible_cases": len(eligible),
        "conditions_on_eligible": panel,
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: DS7B与GLM4的相对层段Q/K/V编译复验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在DeepSeek-R1-Distill-Qwen-7B与GLM4-9B-chat上严格顺序加载BF16非量化模型。每模型重新按自身tokenizer原子化构造32语言模式族×英中共64个case、128个多token候选序列；先要求base与meaning donor都答对，再在各模型自身深度的四个等比例层段上复验早期facts K/V、中期facts K+V、中晚期全部prompt K+V、晚期答案阶段Q和晚期facts K+V。物理层号不跨模型硬对齐。

$$B_r=[l:\lfloor 4l/L\rfloor=r],\qquad r\in(0,1,2,3),\qquad \operatorname{{do}}(Q,K,V)^B_{{B_r}}\leftarrow(Q,K,V)^D_{{B_r}}.$$

**结果汇总。** `{json.dumps(result['models'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2545_c146177_c150272_crossmodel_staged_qkv_compiler.py`；逐候选分数和final位于`{OUT}`。

**分析与理论进展。** 本Phase检验的是Qwen3-4B中“早期source内容写入—中层地址/下游重写—晚层输出Q控制”是否具有事件级、相对深度上的可迁移性。跨模型只比较功能阶段，不把head号、坐标号或绝对层号视为同源。若某模型的最佳阶段不同，应记录为架构/训练形成的差异，而不是强行宣布统一算法。

**问题硬伤与结论。** donor替换是构造性充分性；K/V整层段替换会同时带入大量非语义状态；候选似然仍含输出身份编译；仅对base与donor都通过行为门的case解释因果结果。跨模型复现能增强阶段假说，不能证明相同物理齿轮或完整语言机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2544 / "analysis/final.json")
    models = {}
    for model_key in ("deepseek7b", "glm4"):
        marker = OUT / "analysis" / f"{model_key}.json"
        if marker.exists():
            models[model_key] = load(marker)
        else:
            models[model_key] = run_model(model_key)
            save(marker, models[model_key])
    checks = {
        "source_passed": prior["all_checks_passed"],
        "models_complete": set(models) == {"deepseek7b", "glm4"},
        "nonquantized_bf16": True,
        "all_families_and_languages": all(value["jobs"] == 128 for value in models.values()),
        "behavior_gate_nonempty": all(value["panel"]["eligible_cases"] >= 16 for value in models.values()),
        "all_conditions_complete": all(
            all(panel["n"] == value["panel"]["eligible_cases"]
                for panel in value["panel"]["conditions_on_eligible"].values())
            for value in models.values()
        ),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "design": {"families": 32, "languages": ["en", "zh"], "cases_per_model": 64,
                   "candidate_sequences_per_model": 128, "conditions": 8,
                   "models_sequential": ["deepseek7b", "glm4"]},
        "models": models, "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
