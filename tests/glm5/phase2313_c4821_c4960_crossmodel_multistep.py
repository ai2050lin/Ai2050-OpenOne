#!/usr/bin/env python3
"""Sequential fresh multi-step behavior and full-coordinate cross-model panel."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2309 = RESULT / "phase2309_c4321_c4440_multistep_future_contract"
P2310 = RESULT / "phase2310_c4441_c4580_qwen4b_multistep_field"
P2312 = RESULT / "phase2312_c4701_c4820_qwen4b_local_response"
OUT = RESULT / "phase2313_c4821_c4960_crossmodel_multistep"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2309 / "material/multistep_future_bilingual.jsonl"
CONFIG_PATH = OUT / "config/frozen_crossmodel.json"
Q4_FIELD = P2310 / "raw/qwen4b_multistep_boundary_all_checkpoints.float16.npy"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as model_worker  # noqa: E402
import phase2297_c3161_c3260_qwen4b_ntp_predictive_field as field_base  # noqa: E402
import phase2309_c4321_c4440_multistep_future_contract as contract  # noqa: E402


PHASE = 2313
CAMPAIGN = "C4821-C4960"
MODEL_ORDER = ("qwen3_14b", "deepseek7b")
MODEL_LABELS = {"qwen3_14b": "Qwen3-14B", "deepseek7b": "DeepSeek-R1-Distill-Qwen-7B"}
EPS = 1e-12


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def frozen_config() -> dict:
    value = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "frozen_before_any_model_load": True,
        "models_sequential": list(MODEL_ORDER),
        "model_claim_boundary": {
            "qwen3_14b": "same model family and larger scale, not independent architecture",
            "deepseek7b": "different weights/training but Qwen2-like architecture, not fully independent architecture",
        },
        "partitions": ["fresh_confirmation", "fresh_lockbox"],
        "families": list(contract.FAMILIES),
        "rows": 768,
        "complete_future_gate": contract.BEHAVIOR_GATE,
        "free_identity_gate": contract.FREE_IDENTITY_GATE,
        "hiddenstate_qualification": "intersection(complete_future_strict, free_identity)",
        "field_policy": "qualified families; boundary embedding, every block, final norm, all model-local coordinates",
        "comparison_policy": "relative-depth state-response topology only; never compare physical coordinate IDs",
        "failure_policy": "model-local route failure does not stop the other model",
    }
    if CONFIG_PATH.exists():
        previous = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if previous != value:
            raise RuntimeError(("frozen_crossmodel_config_changed", previous, value))
    else:
        save(CONFIG_PATH, value)
    return value


def continuation_ids(tokenizer, text: str, language: str) -> list[int]:
    values = tokenizer.encode((" " + text) if language == "en" else text,
                              add_special_tokens=False)
    if not values:
        raise RuntimeError(("empty_continuation", language, text))
    return [int(value) for value in values]


def compile_rows(tokenizer, rows: list[dict], model_name: str) -> list[dict]:
    output = []
    for row in rows:
        output.append({
            **row,
            "future_prompt_ids": [int(value) for value in tokenizer.encode(
                row["future_prompt"], add_special_tokens=False)],
            "future_target_ids": continuation_ids(tokenizer, row["future_target_text"], row["language"]),
            "future_wrong_ids": continuation_ids(tokenizer, row["future_wrong_text"], row["language"]),
            "future_identity_target_ids": continuation_ids(tokenizer, row["ntp_target_text"], row["language"]),
            "future_identity_wrong_ids": continuation_ids(tokenizer, row["ntp_wrong_text"], row["language"]),
            "tokenizer_model": MODEL_LABELS[model_name],
        })
    return output


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(sequences):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, :len(row)] = 1
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def sequence_scores(model, device, rows: list[dict], batch_size: int, model_name: str) -> list[dict]:
    items = []
    for row in rows:
        for kind, key in (("target", "future_target_ids"), ("wrong", "future_wrong_ids")):
            candidate = row[key]
            items.append((row, kind, row["future_prompt_ids"] + candidate,
                          len(row["future_prompt_ids"]), candidate))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    scores: dict[str, dict] = defaultdict(dict)
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            ids, mask, positions = pad_right([item[2] for item in batch], device, pad)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, return_dict=True).logits
            for local, (row, kind, _sequence, prompt_len, candidate) in enumerate(batch):
                selected = logits[local, prompt_len - 1:prompt_len - 1 + len(candidate)].float()
                token_ids = torch.tensor(candidate, dtype=torch.long, device=selected.device)
                token_logps = (selected.gather(1, token_ids[:, None])[:, 0]
                                - torch.logsumexp(selected, dim=-1)).cpu().tolist()
                scores[row["case_id"]][kind] = {
                    "token_logprobs": [float(value) for value in token_logps],
                    "sum_logprob": float(sum(token_logps)),
                    "mean_logprob": float(sum(token_logps) / len(token_logps)),
                    "token_count": len(token_logps),
                }
            del logits
            print(f"[phase2313 {model_name} sequence] {min(start + len(batch), len(items))}/{len(items)}",
                  flush=True)
    output = []
    for row in rows:
        target, wrong = scores[row["case_id"]]["target"], scores[row["case_id"]]["wrong"]
        output.append({
            "case_id": row["case_id"], "family": row["family"], "language": row["language"],
            "surface": row["surface"], "partition": row["partition"],
            "target_mention_order": row["target_mention_order"], "state": int(row["state"]),
            "target": target, "wrong": wrong,
            "correct_by_mean": target["mean_logprob"] > wrong["mean_logprob"],
            "correct_by_sum": target["sum_logprob"] > wrong["sum_logprob"],
            "mean_margin": target["mean_logprob"] - wrong["mean_logprob"],
            "sum_margin": target["sum_logprob"] - wrong["sum_logprob"],
        })
    return output


def slice_accuracy(rows: list[dict], score_by_id: dict[str, dict]) -> dict:
    values = [score_by_id[row["case_id"]] for row in rows]
    return {
        "rows": len(values),
        "mean_accuracy": float(np.mean([value["correct_by_mean"] for value in values])),
        "sum_accuracy": float(np.mean([value["correct_by_sum"] for value in values])),
        "mean_margin": float(np.mean([value["mean_margin"] for value in values])),
        "sum_margin": float(np.mean([value["sum_margin"] for value in values])),
    }


def sequence_ledger(rows: list[dict], scores: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in scores}
    families, qualified = {}, []
    for family in contract.FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        slices = {"overall:all": slice_accuracy(family_rows, by_id)}
        for key, values in (
            ("language", contract.LANGUAGES), ("surface", contract.SURFACES),
            ("partition", ("fresh_confirmation", "fresh_lockbox")),
            ("target_mention_order", ("first", "last")),
        ):
            for value in values:
                slices[f"{key}:{value}"] = slice_accuracy(
                    [row for row in family_rows if row[key] == value], by_id)
        passed = all(min(value["mean_accuracy"], value["sum_accuracy"]) >= contract.BEHAVIOR_GATE
                     for value in slices.values())
        families[family] = {"qualified": passed, "slices": slices}
        if passed:
            qualified.append(family)
    return {"families": families, "qualified_families": qualified,
            "overall": slice_accuracy(rows, by_id), "gate": contract.BEHAVIOR_GATE}


def normalized(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def free_generation(model, tokenizer, device, rows: list[dict], batch_size: int,
                    model_name: str) -> list[dict]:
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    max_new = min(10, max(max(len(row["future_identity_target_ids"]),
                              len(row["future_identity_wrong_ids"])) for row in rows) + 3)
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(row["future_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for local, row in enumerate(batch):
            sequence = row["future_prompt_ids"]
            ids[local, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[local, width - len(sequence):] = 1
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new,
                use_cache=True, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
            )
        for row, token_ids in zip(batch, generated[:, width:].detach().cpu().tolist()):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            clean = normalized(text)
            target, wrong = normalized(row["ntp_target_text"]), normalized(row["ntp_wrong_text"])
            target_pos, wrong_pos = clean.find(target), clean.find(wrong)
            output.append({
                "case_id": row["case_id"], "family": row["family"],
                "language": row["language"], "surface": row["surface"],
                "partition": row["partition"], "state": int(row["state"]),
                "generated_ids": [int(value) for value in token_ids], "generated_text": text,
                "first_identity_correct": target_pos >= 0 and (wrong_pos < 0 or target_pos < wrong_pos),
                "target_found": target_pos >= 0, "wrong_found": wrong_pos >= 0,
            })
        print(f"[phase2313 {model_name} free] {min(start + len(batch), len(rows))}/{len(rows)}",
              flush=True)
    return output


def free_ledger(rows: list[dict]) -> dict:
    def summarize(values: list[dict]) -> dict:
        return {"rows": len(values),
                "first_identity_accuracy": float(np.mean([row["first_identity_correct"] for row in values])),
                "target_found": float(np.mean([row["target_found"] for row in values])),
                "wrong_found": float(np.mean([row["wrong_found"] for row in values]))}
    families = {family: summarize([row for row in rows if row["family"] == family])
                for family in contract.FAMILIES}
    return {"families": families, "overall": summarize(rows), "gate": contract.FREE_IDENTITY_GATE,
            "route_eligible_families": [family for family, value in families.items()
                                        if value["first_identity_accuracy"] >= contract.FREE_IDENTITY_GATE]}


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def capture_field(model, device, rows: list[dict], worker: Path, model_name: str) -> dict:
    path = worker / "raw/boundary_all_checkpoints.float16.npy"
    progress_path = worker / "raw/field_progress.json"
    index_path = worker / "index/field_rows.jsonl"
    module_list = modules(model)
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(module_list), dimension)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["shape"] != list(shape):
            raise RuntimeError(("field_resume_shape", model_name, progress["shape"], shape))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(path, mode="r+")
    else:
        field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(module_list):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    batch_size = 16 if model_name == "qwen3_14b" else 8
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                if len(captures) != len(module_list):
                    raise RuntimeError(("checkpoint_count", model_name, len(captures), len(module_list)))
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]]
                                            for local in range(len(batch))])
                    field[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                field.flush()
                save(progress_path, {"completed": start + len(batch), "shape": list(shape)})
                print(f"[phase2313 {model_name} field] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        close_memmap(field)
    write_rows(index_path, [{
        "hidden_index": index, "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "surface": row["surface"],
        "partition": row["partition"], "unit": int(row["unit"]), "state": int(row["state"]),
    } for index, row in enumerate(rows)])
    return {"ran": True, "path": str(path.relative_to(ROOT)), "shape": list(shape),
            "index": str(index_path.relative_to(ROOT)), "all_model_local_coordinates": True,
            "checkpoints": len(module_list), "coordinates": dimension}


def state_topology(field_path: Path, index_rows: list[dict], label: str) -> dict:
    field = np.load(field_path, mmap_mode="r")
    grouped: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index_rows:
        key = (row["family"], row["language"], row["surface"], row["partition"], int(row["unit"]))
        grouped[key][int(row["state"])] = row
    output = {}
    try:
        families = sorted(set(row["family"] for row in index_rows))
        for family in families:
            pairs = [states for key, states in grouped.items()
                     if key[0] == family and set(states) == {0, 1}]
            curves = []
            for q in range(field.shape[1]):
                left = np.asarray(field[[pair[0]["hidden_index"] for pair in pairs], q], np.float32)
                right = np.asarray(field[[pair[1]["hidden_index"] for pair in pairs], q], np.float32)
                response_rms = float(np.sqrt(np.mean((right - left).astype(np.float64) ** 2)))
                base_rms = float(np.sqrt(np.mean(0.5 * (
                    left.astype(np.float64) ** 2 + right.astype(np.float64) ** 2))))
                curves.append({
                    "checkpoint": q, "relative_depth": q / max(1, field.shape[1] - 1),
                    "response_rms": response_rms, "base_rms": base_rms,
                    "relative_response": response_rms / (base_rms + EPS),
                })
            peak = max(value["relative_response"] for value in curves)
            threshold = 0.5 * peak
            formation = next((value for value in curves if value["relative_response"] >= threshold), curves[-1])
            output[family] = {
                "pairs": len(pairs), "curve": curves,
                "peak_relative_response": peak,
                "half_peak_formation_checkpoint": formation["checkpoint"],
                "half_peak_formation_relative_depth": formation["relative_depth"],
            }
    finally:
        close_memmap(field)
    return {"model": label, "families": output,
            "metric_boundary": "state-response RMS divided by base-state RMS; descriptive, not a logit lens"}


def load_named_model(model_name: str):
    if model_name == "qwen3_14b":
        import phase2145_c611_natural_qwen14_worker as q14

        def reduced_map() -> dict:
            value = {"model.embed_tokens": 0, "model.rotary_emb": 0,
                     "model.norm": "disk", "lm_head": "disk"}
            value.update({f"model.layers.{index}": 0 if index < 12 else "disk"
                          for index in range(40)})
            return value

        q14.device_map = reduced_map
        model, tokenizer, device, placement, loader = model_worker.load_model(model_name)
        placement = {
            "placement": placement,
            "engineering_note": (
                "The first disk-offload timing attempt was interrupted after 48/1536 sequence items "
                "without writing behavior results. A CUDA+CPU attempt was then killed during weight loading "
                "because the host has 32 GB RAM. The same local FP16 nonquantized weights, rows, decoding, "
                "and gates were finally rerun with disk offload and larger execution batches. The first "
                "24-row generation timing batch was also stopped before results were written and rerun at 64."
            ),
        }
        return model, tokenizer, device, placement, loader
    return model_worker.load_model(model_name)


def run_model(model_name: str, raw_rows: list[dict]) -> dict:
    worker = OUT / model_name
    final_path = worker / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    model = tokenizer = None
    try:
        model, tokenizer, device, placement, loader = load_named_model(model_name)
        compiled = compile_rows(tokenizer, raw_rows, model_name)
        write_rows(worker / "material/fresh_compiled.jsonl", compiled)
        score_path = worker / "behavior/sequence_scores.jsonl"
        scores = (read_rows(score_path) if score_path.exists() else
                  sequence_scores(model, device, compiled, 48 if model_name == "qwen3_14b" else 8,
                                  model_name))
        if not score_path.exists():
            write_rows(score_path, scores)
        sequence = sequence_ledger(compiled, scores)
        save(worker / "behavior/sequence_ledger.json", sequence)
        free_path = worker / "behavior/free_generation.jsonl"
        free_rows = (read_rows(free_path) if free_path.exists() else
                     free_generation(model, tokenizer, device, compiled,
                                     64 if model_name == "qwen3_14b" else 8, model_name))
        if not free_path.exists():
            write_rows(free_path, free_rows)
        free = free_ledger(free_rows)
        save(worker / "behavior/free_ledger.json", free)
        qualified = sorted(set(sequence["qualified_families"]) & set(free["route_eligible_families"]))
        observed = [row for row in compiled if row["family"] in qualified]
        field = capture_field(model, device, observed, worker, model_name) if observed else {
            "ran": False, "reason": "no_family_passed_both_behavior_routes"}
        topology = (state_topology(ROOT / field["path"], read_rows(ROOT / field["index"]),
                                   MODEL_LABELS[model_name]) if field["ran"] else
                    {"model": MODEL_LABELS[model_name], "families": {}})
        save(worker / "analysis/state_response_topology.json", topology)
        result = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL_LABELS[model_name],
            "model_key": model_name, "timestamp": datetime.now().astimezone().isoformat(),
            "status": "closed", "loader": loader, "placement": placement,
            "precision": "float16" if model_name == "qwen3_14b" else "bfloat16",
            "quantization": "none", "material_rows": len(compiled),
            "sequence": sequence, "free": free, "qualified_families": qualified,
            "field": field, "topology": topology,
            "checks": {
                "own_tokenizer": True, "all_fresh_rows": len(compiled) == 768,
                "all_sequence_scores": len(scores) == len(compiled),
                "all_free_generations": len(free_rows) == len(compiled),
                "field_matches_qualification": field["ran"] == bool(qualified),
                "all_model_local_coordinates": (not field["ran"]) or field["all_model_local_coordinates"],
                "no_crossmodel_coordinate_alignment": True,
            },
        }
        result["all_checks_passed"] = all(result["checks"].values())
        save(final_path, result)
        return result
    finally:
        if model is not None:
            model_worker.release_model(model_name, model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def q4_topology(raw_rows: list[dict]) -> dict:
    index_rows = [{
        "hidden_index": index, "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "surface": row["surface"],
        "partition": row["partition"], "unit": int(row["unit"]), "state": int(row["state"]),
    } for index, row in enumerate(read_rows(ROWS_PATH)) if row["partition"] in
                  ("fresh_confirmation", "fresh_lockbox")]
    # The source field contains all 2048 rows; retain the source row indices.
    source = read_rows(ROWS_PATH)
    index_rows = [{**row, "hidden_index": index} for index, row in enumerate(source)
                  if row["partition"] in ("fresh_confirmation", "fresh_lockbox")
                  and row["family"] in json.loads(
                      (P2312 / "config/frozen_local_response.json").read_text(encoding="utf-8")
                  )["eligible_families"]]
    return state_topology(Q4_FIELD, index_rows, "Qwen3-4B")


def compare_topologies(q4: dict, models: dict) -> dict:
    output = {}
    for model_name, result in models.items():
        topology = result.get("topology", {"families": {}})
        common = sorted(set(q4["families"]) & set(topology["families"]))
        output[model_name] = {
            "common_families": common,
            "formation_relative_depth": {
                family: {
                    "qwen3_4b": q4["families"][family]["half_peak_formation_relative_depth"],
                    model_name: topology["families"][family]["half_peak_formation_relative_depth"],
                    "absolute_difference": abs(
                        q4["families"][family]["half_peak_formation_relative_depth"]
                        - topology["families"][family]["half_peak_formation_relative_depth"]),
                } for family in common
            },
            "claim_boundary": "descriptive relative-depth topology; no coordinate or circuit identity",
        }
    return output


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {key: {
        "sequence_qualified": value.get("sequence", {}).get("qualified_families", []),
        "free_qualified": value.get("free", {}).get("route_eligible_families", []),
        "field_qualified": value.get("qualified_families", []),
        "status": value.get("status"),
    } for key, value in result["models"].items()}
    text = rf"""

## Phase {PHASE}: 多步未来的顺序跨模型功能复验（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 在任何模型加载前冻结 Phase2309 的两个 fresh 分区、八个语言族和 768 行原始文本。Qwen3-14B 与 DeepSeek-R1-Distill-Qwen-7B 严格逐个加载、释放，各自 tokenizer 重新编译裸文本前缀、完整正确/错误未来和身份词。每族的完整未来总分及长度均值必须在语言、表面、fresh 分区和提及顺序全部达到 `0.75`；自由生成中目标身份先于错误身份出现的族准确率必须达到 `0.50`。两门交集才保存回答边界的 embedding、每个 block 后状态、final norm 和全部模型本地坐标。没有搬运 4B token ID 或坐标编号。
$$
S_M(y_{{1:K}}\mid x)=\sum_k\log p_M(y_k\mid x,y_{{<k}}),
\qquad
\rho_{{M,f}}(q)=\frac{{\operatorname{{RMS}}(H^1_{{M,f,q}}-H^0_{{M,f,q}})}}
{{\operatorname{{RMS}}(H^0_{{M,f,q}},H^1_{{M,f,q}})+\epsilon}}.
$$
形成相对深度只记 `rho` 首次达到本模型本族峰值一半的位置，是描述性状态响应拓扑，不是 logit lens 或因果形成时钟。

**结果汇总。** 模型行为与场资格 `{json.dumps(compact, ensure_ascii=False)}`；完整模型账 `{json.dumps(result['models'], ensure_ascii=False)}`；4B 参考拓扑与跨模型相对深度比较 `{json.dumps(result['comparison'], ensure_ascii=False)}`；执行与检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。Qwen3-14B 属同一模型家族的跨规模复验；DeepSeek-R1-Distill-Qwen-7B 虽训练与权重不同，但底层是 Qwen2 类架构，不能冒充完全独立架构。跨模型共同过门只说明这套材料和接口上的功能可比较；半峰相对深度接近不证明相同算法，物理坐标更完全不可对齐。自由生成只观察有限首段，完整未来使用 teacher forcing；人工材料无独立人类盲评。脚本 `tests/glm5/phase2313_c4821_c4960_crossmodel_multistep.py`；结果 `tests/glm5/result/phase2313_c4821_c4960_crossmodel_multistep`。下一步只发布可逐坐标复核的代表场与完整局部梯度，构建通过后清理未展示的大型词表/HiddenState 原始副本。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for parent in (P2309, P2310, P2312):
        final = json.loads((parent / "analysis/final.json").read_text(encoding="utf-8"))
        if not final["all_checks_passed"]:
            raise RuntimeError(("parent_not_authorized", parent.name))
    config = frozen_config()
    source = read_rows(ROWS_PATH)
    raw_rows = [row for row in source
                if row["partition"] in ("fresh_confirmation", "fresh_lockbox")]
    if len(raw_rows) != config["rows"]:
        raise RuntimeError(("fresh_row_count", len(raw_rows), config["rows"]))
    models = {}
    for model_name in MODEL_ORDER:
        try:
            models[model_name] = run_model(model_name, raw_rows)
        except Exception as error:
            failure = {
                "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL_LABELS[model_name],
                "model_key": model_name, "timestamp": datetime.now().astimezone().isoformat(),
                "status": "worker_error", "error_type": type(error).__name__, "error": str(error),
                "all_checks_passed": False,
                "strict_boundary": "model-local failure; the next frozen model route remains authorized",
            }
            save(OUT / model_name / "analysis/final.json", failure)
            models[model_name] = failure
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    q4 = q4_topology(raw_rows)
    comparison = compare_topologies(q4, models)
    save(OUT / "analysis/qwen4_reference_topology.json", q4)
    save(OUT / "analysis/crossmodel_relative_topology.json", comparison)
    checks = {
        "config_frozen_before_models": config["frozen_before_any_model_load"],
        "fresh_rows_complete": len(raw_rows) == 768,
        "models_attempted_in_order": list(models) == list(MODEL_ORDER),
        "route_failure_did_not_stop_next": len(models) == 2,
        "qwen4_reference_all_four_families": len(q4["families"]) == 4,
        "no_coordinate_id_comparison": True,
    }
    successful = [key for key, value in models.items() if value.get("all_checks_passed")]
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
        "config": config, "models": models, "successful_model_workers": successful,
        "qwen4_reference_topology": q4, "comparison": comparison,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            f"Both frozen model routes were attempted sequentially and {len(successful)}/2 completed their "
            "engineering audits. Behavior-qualified family sets and model-local relative-depth topologies are "
            "valid only for this interface. Neither same-family scale replication nor the distilled Qwen2-like "
            "DeepSeek panel establishes shared coordinates, an independent-architecture invariant, or a complete gear."
        ),
        "next_authorization": "Publish exact-coordinate derivatives, verify the generic atlas build, then clean undisplayed raw fields.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
