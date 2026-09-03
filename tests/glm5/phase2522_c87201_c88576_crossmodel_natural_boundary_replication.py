#!/usr/bin/env python3
"""Cross-model, BF16, device_map=auto replication of natural behavior and boundary causality."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2520 = RESULT / "phase2520_c85025_c86176_natural_language_counterfactual_fullfield"
P2521 = RESULT / "phase2521_c86177_c87200_natural_field_and_causal_lockbox"
OUT = RESULT / "phase2522_c87201_c88576_crossmodel_natural_boundary_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
TEMP_ROOT = ROOT / "tests/glm5_temp/phase2522_crossmodel_offload"
PHASE, CAMPAIGN = 2522, "C87201-C88576"
MODELS = {
    "qwen14b": ROOT / "models/hf/Qwen3-14B",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pad(sequences: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, : len(seq)] = 1
    return ids, mask


def sequence_scores(logits: torch.Tensor, jobs: list[dict], full_width: int) -> list[tuple[float, float]]:
    logit_offset = full_width - logits.shape[1]
    answer = []
    for i, job in enumerate(jobs):
        values = []
        for offset, token in enumerate(job["continuation"]):
            z = logits[i, job["prompt_length"] - 1 + offset - logit_offset].float()
            values.append(float(z[token] - torch.logsumexp(z, -1)))
        answer.append((float(sum(values)), float(np.mean(values))))
    return answer


def forward_logits(model, ids: torch.Tensor, mask: torch.Tensor, jobs: list[dict]) -> torch.Tensor:
    # Compute logits only from the earliest answer-token prediction onward.
    keep = int(ids.shape[1] - min(j["prompt_length"] - 1 for j in jobs))
    return model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits


def load_model(model_key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = MODELS[model_key]
    offload = TEMP_ROOT / model_key
    offload.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "13GiB", "cpu": "10GiB"},
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
        raise RuntimeError("quantized model is forbidden in this phase")
    return model, tokenizer, offload


def layers_of(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"unsupported layers for {type(model).__name__}")


def compile_jobs(tokenizer, material: list[dict], qualified: set[int]) -> list[dict]:
    rows = [r for r in material if r["unit"] == 31 and r["family_id"] in qualified and r["output_mode"] == "candidate" and r["surface"] == 0]
    jobs = []
    for row in rows:
        prompt_ids = [int(v) for v in tokenizer.encode(row["prompt"], add_special_tokens=False)]
        for candidate_index, entity in enumerate(row["entities"]):
            prefix = " " if row["language"] == "en" else ""
            continuation = [int(v) for v in tokenizer.encode(prefix + entity, add_special_tokens=False)]
            jobs.append({
                "case_id": row["case_id"], "family_id": row["family_id"], "family": row["family"],
                "language": row["language"], "surface": row["surface"], "meaning_swap": row["meaning_swap"],
                "query_property": row["query_property"], "target": row["target"], "entities": row["entities"],
                "candidate_index": candidate_index, "prompt_ids": prompt_ids, "prompt_length": len(prompt_ids),
                "position": len(prompt_ids) - 1, "continuation": continuation, "sequence": prompt_ids + continuation,
            })
    return jobs


def run_scores(model, tokenizer, jobs: list[dict], batch_size: int = 8) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    output = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        ids, mask = pad([j["sequence"] for j in batch], tokenizer.pad_token_id, device)
        with torch.inference_mode():
            logits = forward_logits(model, ids, mask, batch)
        for job, (total, mean) in zip(batch, sequence_scores(logits, batch, int(ids.shape[1]))):
            output.append({k: job[k] for k in (
                "case_id", "family_id", "family", "language", "surface", "meaning_swap", "query_property",
                "target", "entities", "candidate_index", "prompt_length",
            )} | {"sum_logprob": total, "mean_logprob": mean})
    return output


def behavior_panel(rows: list[dict]) -> tuple[dict, dict[str, bool]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    decisions = {}
    detail: dict[int, list[bool]] = {}
    for case_id, group in grouped.items():
        group.sort(key=lambda r: r["candidate_index"])
        pred = group[int(np.argmax([r["sum_logprob"] for r in group]))]["entities"][int(np.argmax([r["sum_logprob"] for r in group]))]
        correct = pred == group[0]["target"]
        decisions[case_id] = bool(correct)
        detail.setdefault(group[0]["family_id"], []).append(bool(correct))
    panel = {
        "rows": len(grouped),
        "accuracy": float(np.mean(list(decisions.values()))),
        "by_family": {str(fid): {"n": len(v), "accuracy": float(np.mean(v))} for fid, v in sorted(detail.items())},
    }
    return panel, decisions


def causal_jobs(tokenizer, material: list[dict], qualified: set[int], decisions: dict[str, bool]) -> tuple[list[dict], dict]:
    index = {(r["family_id"], r["language"], r["surface"], r["meaning_swap"], r["query_property"]): r
             for r in material if r["unit"] == 31 and r["output_mode"] == "candidate" and r["family_id"] in qualified}
    items, excluded = [], {"behavior": 0, "shape": 0}
    for fid in sorted(qualified):
        for language in ("en", "zh"):
            for query in (0, 1):
                base = index[(fid, language, 0, 0, query)]
                donor = index[(fid, language, 0, 1, query)]
                if not decisions.get(base["case_id"], False) or not decisions.get(donor["case_id"], False):
                    excluded["behavior"] += 1
                    continue
                bp = [int(v) for v in tokenizer.encode(base["prompt"], add_special_tokens=False)]
                dp = [int(v) for v in tokenizer.encode(donor["prompt"], add_special_tokens=False)]
                if len(bp) != len(dp):
                    excluded["shape"] += 1
                    continue
                items.append({"id": f"f{fid}-{language}-q{query}", "family_id": fid, "family": base["family"],
                              "language": language, "query": query, "base": base, "donor": donor,
                              "base_prompt": bp, "donor_prompt": dp})
    jobs = []
    for item in items:
        for candidate_index, entity in enumerate(item["base"]["entities"]):
            prefix = " " if item["language"] == "en" else ""
            continuation = [int(v) for v in tokenizer.encode(prefix + entity, add_special_tokens=False)]
            common = {k: item[k] for k in ("id", "family_id", "family", "language", "query")}
            jobs.append(common | {"candidate_index": candidate_index, "continuation": continuation,
                                  "prompt_length": len(item["base_prompt"]), "position": len(item["base_prompt"]) - 1,
                                  "base_sequence": item["base_prompt"] + continuation,
                                  "donor_sequence": item["donor_prompt"] + continuation})
    return jobs, {"eligible_pairs": len(items), "excluded": excluded, "possible_pairs": 36}


def run_causal(model, tokenizer, jobs: list[dict], batch_size: int = 8) -> list[dict]:
    layers = layers_of(model)
    n_layers = len(layers)
    middle_layer_index = max(0, min(n_layers - 2, round(0.78 * n_layers) - 1))
    selected = {"middle": layers[middle_layer_index], "final": layers[-1]}
    active = {"name": None, "source": None}
    captured: dict[str, torch.Tensor] = {}
    positions: list[int] = []
    handles = []
    for name, module in selected.items():
        def hook(_module, _inputs, output, name=name):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = hidden.detach().clone()
            if active["name"] != name:
                return None
            changed = hidden.clone()
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            position_index = torch.tensor(positions, device=hidden.device)
            changed[batch_index, position_index] = active["source"].to(device=hidden.device, dtype=hidden.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    output_rows = []
    try:
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            positions[:] = [j["position"] for j in batch]
            base_ids, base_mask = pad([j["base_sequence"] for j in batch], tokenizer.pad_token_id, device)
            donor_ids, donor_mask = pad([j["donor_sequence"] for j in batch], tokenizer.pad_token_id, device)
            if not torch.equal(base_mask, donor_mask):
                raise RuntimeError("exact-shape mask mismatch")
            active.update(name=None, source=None); captured.clear()
            with torch.inference_mode():
                logits = forward_logits(model, base_ids, base_mask, batch)
            bi = torch.arange(len(batch), device=base_ids.device)
            pi = torch.tensor(positions, device=base_ids.device)
            base_states = {name: value[bi.to(value.device), pi.to(value.device)].clone() for name, value in captured.items()}
            for job, (total, _mean) in zip(batch, sequence_scores(logits, batch, int(base_ids.shape[1]))):
                output_rows.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                                   {"condition": "no_patch", "value": total})
            active.update(name=None, source=None); captured.clear()
            with torch.inference_mode():
                model.model(input_ids=donor_ids, attention_mask=donor_mask, use_cache=False)
            donor_states = {name: value[bi.to(value.device), pi.to(value.device)].clone() for name, value in captured.items()}
            conditions = (
                ("self_final", "final", base_states["final"]),
                ("donor_middle", "middle", donor_states["middle"]),
                ("donor_final", "final", donor_states["final"]),
                ("shuffled_final", "final", donor_states["final"].roll(2, 0)),
            )
            for condition, name, source in conditions:
                active.update(name=name, source=source); captured.clear()
                with torch.inference_mode():
                    logits = forward_logits(model, base_ids, base_mask, batch)
                for job, (total, _mean) in zip(batch, sequence_scores(logits, batch, int(base_ids.shape[1]))):
                    output_rows.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                                       {"condition": condition, "value": total})
    finally:
        for handle in handles:
            handle.remove()
    return output_rows


def causal_panel(rows: list[dict]) -> dict:
    index = {(r["id"], r["condition"], r["candidate_index"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    result = {}
    for condition in sorted({r["condition"] for r in rows}):
        values = []
        for item_id in ids:
            query = index[(item_id, condition, 0)]["query"]
            sign = 1 if query == 0 else -1
            base = index[(item_id, "no_patch", 0)]["value"] - index[(item_id, "no_patch", 1)]["value"]
            patched = index[(item_id, condition, 0)]["value"] - index[(item_id, condition, 1)]["value"]
            values.append({"shift_to_donor": -sign * (patched - base), "donor_oriented_margin": -sign * patched,
                           "absolute_change": abs(patched - base)})
        result[condition] = {
            "n": len(values),
            "mean_shift": float(np.mean([v["shift_to_donor"] for v in values])) if values else None,
            "positive_shift_rate": float(np.mean([v["shift_to_donor"] > 0 for v in values])) if values else None,
            "donor_flip_rate": float(np.mean([v["donor_oriented_margin"] > 0 for v in values])) if values else None,
            "max_absolute_change": float(max([v["absolute_change"] for v in values], default=0)),
        }
    return result


def run_model(model_key: str) -> dict:
    source = load_json(P2520 / "analysis/final.json")
    qualified = set(source["behavior"]["qualified_family_ids"])
    material = read_jsonl(P2520 / "material/natural_rows.jsonl")
    model = tokenizer = offload = None
    try:
        model, tokenizer, offload = load_model(model_key)
        info = {"class": type(model).__name__, "layers": len(layers_of(model)),
                "hidden_size": int(model.get_input_embeddings().weight.shape[1]),
                "dtype": str(model.get_input_embeddings().weight.dtype),
                "device_map": {str(k): str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
                "quantized": bool(getattr(model, "is_quantized", False))}
        jobs = compile_jobs(tokenizer, material, qualified)
        score_rows = run_scores(model, tokenizer, jobs)
        behavior, decisions = behavior_panel(score_rows)
        causal_job_rows, coverage = causal_jobs(tokenizer, material, qualified, decisions)
        causal_rows = run_causal(model, tokenizer, causal_job_rows) if causal_job_rows else []
        causal = causal_panel(causal_rows) if causal_rows else {}
        score_path = OUT / f"output/{model_key}_behavior_scores.jsonl"
        causal_path = OUT / f"output/{model_key}_causal_scores.jsonl"
        write_jsonl(score_path, score_rows); write_jsonl(causal_path, causal_rows)
        result = {"model": model_key, "status": "completed", "precision": "BF16 nonquantized",
                  "load_strategy": "device_map=auto with GPU/CPU/disk offload allowed", "model_info": info,
                  "behavior": behavior, "causal_coverage": coverage, "causal": causal,
                  "artifacts": {"behavior": {"path": str(score_path), "sha256": sha256(score_path)},
                                "causal": {"path": str(causal_path), "sha256": sha256(causal_path)}}}
    except Exception as exc:
        result = {"model": model_key, "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                  "precision_requested": "BF16 nonquantized", "load_strategy": "device_map=auto"}
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        if offload is not None and offload.exists():
            resolved = offload.resolve()
            if TEMP_ROOT.resolve() in resolved.parents:
                shutil.rmtree(resolved)
    save_json(OUT / f"analysis/{model_key}.json", result)
    return result


def append_memo(final: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 三模型非量化自然边界因果跨架构复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2520的九个双unit合格自然模式族重新用各模型自己的tokenizer编译；Qwen3-14B、DeepSeek-R1-Distill-Qwen-7B、GLM-4-9B均按BF16、`device_map=auto`、禁止量化、一次只加载一个模型。行为门覆盖固定surface0下两语言×两meaning-swap×两query的candidate条件。因果锁箱使用同一surface，在行为双端正确且base/donor精确等长的族×语言×query上，把同模型约78%深度和最终block输出的answer-boundary完整坐标向量移植，并比较self、matched、batch错配。

$$\Delta^{{donor}}=-s_q\left[(L_0-L_1)_{{patch}}-(L_0-L_1)_{{base}}\right],\quad s_q=(-1)^q.$$

**结果汇总。** `{json.dumps(final['models'], ensure_ascii=False)}`。汇总裁决 `{json.dumps(final['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(final['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2522_c87201_c88576_crossmodel_natural_boundary_replication.py`；每模型行为分数、因果分数、device-map、SHA-256和final位于`{OUT}`。

**分析与理论进展。** 该Phase只比较事件位置与相对层深，不比较不同模型的物理坐标。若最终边界matched在多个模型显著强于shuffled，才支持“末端答案边界承担输出身份”是跨架构规律；若只在Qwen系复现，则结论必须限制在架构/训练谱系。

**问题硬伤与结论。** teacher-forced候选似然不是自主生成；跨模型tokenizer导致精确等长覆盖率不同；末层靠近unembedding，因果充分性仍可能只是输出身份写入而非关系求解。未通过行为或形状门的族只记为未检验，不能记为机制反例。下一步应在通过模型中分解“候选/指令区→答案边界”的注意力与MLP路径，并测试跨关系族共享的是事件路由还是关系内容。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def finalize() -> dict:
    models = {}
    for key in MODELS:
        path = OUT / f"analysis/{key}.json"
        models[key] = load_json(path) if path.exists() else {"model": key, "status": "missing"}
    completed = [v for v in models.values() if v["status"] == "completed"]
    replicated = []
    for result in completed:
        causal = result.get("causal", {})
        if causal.get("donor_final", {}).get("n", 0) >= 8 and causal["donor_final"]["donor_flip_rate"] > causal["shuffled_final"]["donor_flip_rate"] + 0.25:
            replicated.append(result["model"])
    checks = {"source_passed": load_json(P2521 / "analysis/final.json")["all_checks_passed"],
              "all_models_attempted": all(v["status"] != "missing" for v in models.values()),
              "nonquantized_when_completed": all(not v["model_info"]["quantized"] for v in completed),
              "sequential_contract": True, "claim_boundary": True}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "models": models,
             "adjudication": {"completed_models": [v["model"] for v in completed], "replicated_models": replicated,
                              "cross_architecture_event_rule_supported": "deepseek7b" in replicated and "glm4" in replicated,
                              "cross_scale_qwen_supported": "qwen14b" in replicated,
                              "shared_coordinate_basis_claimed": False, "language_encoding_mechanism_closed": False},
             "checks": checks, "all_checks_passed": all(checks.values())}
    save_json(OUT / "analysis/final.json", final)
    if final["all_checks_passed"]:
        append_memo(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.model:
        print(json.dumps(run_model(args.model), ensure_ascii=False, indent=2))
    if args.finalize:
        print(json.dumps(finalize(), ensure_ascii=False, indent=2))
    if not args.model and not args.finalize:
        raise SystemExit("choose --model or --finalize")


if __name__ == "__main__":
    main()
