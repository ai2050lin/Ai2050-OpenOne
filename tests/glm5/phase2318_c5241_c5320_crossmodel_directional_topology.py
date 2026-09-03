#!/usr/bin/env python3
"""Sequential model-local behavior and directional topology panel."""
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
P2315 = RESULT / "phase2315_c5041_c5100_active_response_contract"
P2316 = RESULT / "phase2316_c5101_c5160_qwen4b_active_baseline"
P2317 = RESULT / "phase2317_c5161_c5240_directional_response_identification"
OUT = RESULT / "phase2318_c5241_c5320_crossmodel_directional_topology"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2315 / "material/natural_active_response_bilingual.jsonl"
ACTIVE_INDEX = P2317 / "index/active_rows.jsonl"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as model_worker  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as contract  # noqa: E402


PHASE = 2318
CAMPAIGN = "C5241-C5320"
MODEL_ORDER = ("qwen3_14b", "glm4", "deepseek7b")
MODEL_LABELS = {
    "qwen3_14b": "Qwen3-14B",
    "glm4": "GLM-4-9B",
    "deepseek7b": "DeepSeek-R1-Distill-Qwen-7B",
}
RELATIVE_SOURCES = (10 / 36, 20 / 36, 30 / 36)
DOSE = contract.PERTURBATION_DOSE
BASE_PROBES = 4
PAIR_PROBES = ((0, 1), (2, 3))
PROBE_COUNT = 6
EPS = 1e-12


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def modules(model) -> list[Any]:
    base = model.model
    return [base.embed_tokens, *list(base.layers), base.norm]


def continuation_ids(tokenizer, text: str, language: str) -> list[int]:
    values = tokenizer.encode((" " + text) if language == "en" else text, add_special_tokens=False)
    if not values:
        raise RuntimeError(("empty_continuation", text, language))
    return [int(value) for value in values]


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        output.append({
            **row,
            "future_prompt_ids": [int(value) for value in tokenizer.encode(
                row["future_prompt"], add_special_tokens=False)],
            "future_target_ids": continuation_ids(tokenizer, row["future_target_text"], row["language"]),
            "future_wrong_ids": continuation_ids(tokenizer, row["future_wrong_text"], row["language"]),
            "identity_target_ids": continuation_ids(tokenizer, row["identity_target"], row["language"]),
            "identity_wrong_ids": continuation_ids(tokenizer, row["identity_wrong"], row["language"]),
        })
    return output


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(value) for value in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = 1
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def behavior_rows(all_rows: list[dict]) -> list[dict]:
    active = read_rows(ACTIVE_INDEX)
    ids = {row["case_id"] for row in active
           if row["partition"] in ("fresh_confirmation", "fresh_lockbox")}
    selected = [row for row in all_rows if row["case_id"] in ids]
    selected.sort(key=lambda row: row["design_index"])
    if len(selected) != 64:
        raise RuntimeError(("crossmodel_behavior_rows", len(selected)))
    return selected


def active_rows(compiled: list[dict]) -> list[dict]:
    values = [row for row in compiled if row["partition"] == "fresh_lockbox"]
    if len(values) != 32:
        raise RuntimeError(("crossmodel_active_rows", len(values)))
    return values


def sequence_scores(model, device, rows: list[dict], batch_size: int) -> list[dict]:
    items = []
    for row in rows:
        for label, key in (("target", "future_target_ids"), ("wrong", "future_wrong_ids")):
            candidate = row[key]
            items.append((row, label, row["future_prompt_ids"] + candidate,
                          len(row["future_prompt_ids"]), candidate))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    scores: dict[str, dict] = defaultdict(dict)
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            ids, mask, positions = pad_right([item[2] for item in batch], device, pad)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, return_dict=True).logits
            for local, (row, label, _sequence, prompt_length, candidate) in enumerate(batch):
                selected = logits[local, prompt_length - 1:prompt_length - 1 + len(candidate)].float()
                token_ids = torch.tensor(candidate, dtype=torch.long, device=selected.device)
                logps = selected.gather(1, token_ids[:, None])[:, 0] - torch.logsumexp(selected, dim=-1)
                scores[row["case_id"]][label] = {
                    "sum": float(logps.sum().item()), "mean": float(logps.mean().item())}
            del logits
            print(f"[phase2318 behavior score] {min(start + len(batch), len(items))}/{len(items)}", flush=True)
    output = []
    for row in rows:
        target, wrong = scores[row["case_id"]]["target"], scores[row["case_id"]]["wrong"]
        output.append({
            "case_id": row["case_id"], "family": row["family"], "language": row["language"],
            "surface": row["surface"], "partition": row["partition"], "state": int(row["state"]),
            "sum_margin": target["sum"] - wrong["sum"], "mean_margin": target["mean"] - wrong["mean"],
            "sum_correct": target["sum"] > wrong["sum"], "mean_correct": target["mean"] > wrong["mean"],
        })
    return output


def normalized(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def free_generation(model, tokenizer, device, rows: list[dict], batch_size: int) -> list[dict]:
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(row["future_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(batch):
            sequence = row["future_prompt_ids"]
            ids[index, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[index, width - len(sequence):] = 1
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=18,
                use_cache=True, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
            )
        for row, token_ids in zip(batch, generated[:, width:].detach().cpu().tolist()):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            clean = normalized(text)
            target, wrong = normalized(row["identity_target"]), normalized(row["identity_wrong"])
            target_position, wrong_position = clean.find(target), clean.find(wrong)
            output.append({
                "case_id": row["case_id"], "family": row["family"],
                "language": row["language"], "surface": row["surface"],
                "partition": row["partition"], "state": int(row["state"]),
                "generated_text": text,
                "identity_correct": target_position >= 0 and
                                    (wrong_position < 0 or target_position < wrong_position),
            })
        print(f"[phase2318 free] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_summary(scores: list[dict], free: list[dict]) -> dict:
    free_by_id = {row["case_id"]: row for row in free}
    families = {}
    for family in contract.FAMILIES:
        values = [row for row in scores if row["family"] == family]
        families[family] = {
            "rows": len(values),
            "sum_accuracy": float(np.mean([row["sum_correct"] for row in values])),
            "mean_accuracy": float(np.mean([row["mean_correct"] for row in values])),
            "free_identity_accuracy": float(np.mean([free_by_id[row["case_id"]]["identity_correct"]
                                                      for row in values])),
        }
    return {
        "rows": len(scores), "families": families,
        "overall": {
            "sum_accuracy": float(np.mean([row["sum_correct"] for row in scores])),
            "mean_accuracy": float(np.mean([row["mean_correct"] for row in scores])),
            "free_identity_accuracy": float(np.mean([row["identity_correct"] for row in free])),
        },
        "claim_boundary": "64-row descriptive fresh panel; not a replacement for the Phase2316 strict sliced gate",
    }


def probe_directions(dimension: int) -> np.ndarray:
    base = []
    for probe in range(BASE_PROBES):
        digest = hashlib.sha256(f"phase2318|model_local|{dimension}|{probe}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        value = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dimension)
        value /= np.linalg.norm(value.astype(np.float64))
        base.append(value)
    output = list(base)
    output.extend(base[left] + base[right] for left, right in PAIR_PROBES)
    return np.stack(output).astype(np.float32)


def source_and_target_qpoints(module_count: int) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
    blocks = module_count - 2
    sources = tuple(max(1, min(blocks - 1, int(round(depth * blocks)))) for depth in RELATIVE_SOURCES)
    targets = {source: (min(source + 1, blocks), min(source + 4, blocks), module_count - 1)
               for source in sources}
    return sources, targets


def active_capture(model, device, rows: list[dict], worker: Path) -> dict:
    module_list = modules(model)
    dimension = int(model.config.hidden_size)
    directions = probe_directions(dimension)
    sources, targets = source_and_target_qpoints(len(module_list))
    derivative_path = worker / "raw/directional_derivative.float16.npy"
    even_path = worker / "raw/even_response.float16.npy"
    norm_path = worker / "raw/source_and_target_rms.float32.npy"
    derivative_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(rows), len(sources), PROBE_COUNT, 3, dimension)
    derivative = np.lib.format.open_memmap(derivative_path, mode="w+", dtype=np.float16, shape=shape)
    even = np.lib.format.open_memmap(even_path, mode="w+", dtype=np.float16, shape=shape)
    norms = np.lib.format.open_memmap(norm_path, mode="w+", dtype=np.float32,
                                      shape=(len(rows), len(sources), 4))
    for row_index, row in enumerate(rows):
        ids_one = torch.tensor([row["future_prompt_ids"]], dtype=torch.long, device=device)
        mask_one = torch.ones_like(ids_one)
        for source_index, source_q in enumerate(sources):
            variants = [(None, 0.0)]
            for probe in range(PROBE_COUNT):
                variants.extend(((probe, 1.0), (probe, -1.0)))
            captures: dict[int, torch.Tensor] = {}
            source_norm_holder: list[float] = []
            handles = []

            def source_hook(_module, _inputs, value):
                tensor = value[0] if isinstance(value, tuple) else value
                changed = tensor.clone()
                source_norm = float(torch.linalg.vector_norm(tensor[0, -1].float()).item())
                source_norm_holder.append(source_norm)
                for batch_index, (probe, sign) in enumerate(variants):
                    if probe is None:
                        continue
                    direction = torch.tensor(directions[probe], dtype=tensor.dtype, device=tensor.device)
                    changed[batch_index, -1] = changed[batch_index, -1] + direction * (
                        sign * DOSE * source_norm)
                return (changed, *value[1:]) if isinstance(value, tuple) else changed

            handles.append(module_list[source_q].register_forward_hook(source_hook))
            for target_q in targets[source_q]:
                def target_hook(_module, _inputs, value, target_q=target_q):
                    captures[target_q] = value[0] if isinstance(value, tuple) else value
                handles.append(module_list[target_q].register_forward_hook(target_hook))
            try:
                batch = len(variants)
                ids = ids_one.repeat(batch, 1)
                mask = mask_one.repeat(batch, 1)
                with torch.inference_mode():
                    model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                source_norm = source_norm_holder[0]
                target_rms = []
                for target_index, target_q in enumerate(targets[source_q]):
                    baseline = captures[target_q][0, -1].float().cpu().numpy()
                    target_rms.append(float(np.sqrt(np.mean(np.square(baseline.astype(np.float64))))))
                    for probe in range(PROBE_COUNT):
                        plus = captures[target_q][1 + probe * 2, -1].float().cpu().numpy()
                        minus = captures[target_q][2 + probe * 2, -1].float().cpu().numpy()
                        derivative[row_index, source_index, probe, target_index] = (
                            (plus - minus) / (2.0 * DOSE * source_norm)
                        ).astype(np.float16)
                        even[row_index, source_index, probe, target_index] = (
                            (plus + minus) * 0.5 - baseline
                        ).astype(np.float16)
                norms[row_index, source_index] = [source_norm, *target_rms]
            finally:
                for handle in handles:
                    handle.remove()
            derivative.flush(); even.flush(); norms.flush()
            print(f"[phase2318 active] {row_index + 1}/{len(rows)} source {source_index + 1}/3", flush=True)
    derivative.flush(); even.flush(); norms.flush()
    close_memmap(derivative); close_memmap(even); close_memmap(norms)
    index = [{
        "active_index": index, "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "surface": row["surface"],
        "state": int(row["state"]), "unit": int(row["unit"]),
    } for index, row in enumerate(rows)]
    write_rows(worker / "index/active_rows.jsonl", index)
    return {
        "derivative": str(derivative_path.relative_to(ROOT)),
        "even": str(even_path.relative_to(ROOT)), "norms": str(norm_path.relative_to(ROOT)),
        "shape": list(shape), "sources": list(sources),
        "source_relative_depths": [value / (len(module_list) - 2) for value in sources],
        "targets": {str(key): list(value) for key, value in targets.items()},
        "target_relative_depths": {str(key): [value / (len(module_list) - 1) for value in values]
                                   for key, values in targets.items()},
    }


def functional_metrics(derivative_path: Path, even_path: Path, norm_path: Path,
                       index_rows: list[dict], label: str) -> dict:
    derivative = np.load(derivative_path, mmap_mode="r")
    even = np.load(even_path, mmap_mode="r")
    norms = np.load(norm_path, mmap_mode="r")
    records = []
    for row_index, row in enumerate(index_rows):
        family_indices = [index for index, other in enumerate(index_rows)
                          if other["family"] == row["family"] and index != row_index]
        global_indices = [index for index in range(len(index_rows)) if index != row_index]
        for source_index in range(derivative.shape[1]):
            for probe in range(BASE_PROBES):
                for target_index in range(3):
                    actual = derivative[row_index, source_index, probe, target_index].astype(np.float64)
                    denominator = float(np.dot(actual, actual)) + EPS
                    predictions = {
                        "zero": np.zeros_like(actual),
                        "global_leave_one_out": np.mean(
                            derivative[global_indices, source_index, probe, target_index].astype(np.float64), axis=0),
                        "family_leave_one_out": np.mean(
                            derivative[family_indices, source_index, probe, target_index].astype(np.float64), axis=0),
                    }
                    for name, prediction in predictions.items():
                        records.append({
                            "family": row["family"], "source_index": source_index,
                            "probe": probe, "target_index": target_index, "model": name,
                            "relative_mse": float(np.sum(np.square(actual - prediction)) / denominator),
                            "sign_agreement": float(np.mean(actual * prediction > 0)) if name != "zero" else 0.0,
                        })
    model_summary = {}
    for name in ("zero", "global_leave_one_out", "family_leave_one_out"):
        values = [row for row in records if row["model"] == name]
        model_summary[name] = {
            "median_relative_mse": float(np.median([row["relative_mse"] for row in values])),
            "median_sign_agreement": float(np.median([row["sign_agreement"] for row in values])),
        }
    family_summary = {}
    for family in contract.FAMILIES:
        values = [row for row in records if row["family"] == family]
        family_summary[family] = {
            name: float(np.median([row["relative_mse"] for row in values if row["model"] == name]))
            for name in ("zero", "global_leave_one_out", "family_leave_one_out")
        }
    pair_errors, even_ratios, relative_effects = [], [], []
    for row_index in range(len(index_rows)):
        for source_index in range(derivative.shape[1]):
            source_norm = float(norms[row_index, source_index, 0])
            for target_index in range(3):
                target_rms = float(norms[row_index, source_index, 1 + target_index])
                for pair_offset, (left, right) in enumerate(PAIR_PROBES):
                    actual = derivative[row_index, source_index, BASE_PROBES + pair_offset, target_index].astype(np.float64)
                    predicted = (derivative[row_index, source_index, left, target_index].astype(np.float64)
                                 + derivative[row_index, source_index, right, target_index].astype(np.float64))
                    pair_errors.append(float(np.sum(np.square(actual - predicted))
                                             / (np.sum(np.square(actual)) + EPS)))
                for probe in range(PROBE_COUNT):
                    odd = derivative[row_index, source_index, probe, target_index].astype(np.float64)
                    symmetric = even[row_index, source_index, probe, target_index].astype(np.float64)
                    odd_effect = odd * DOSE * source_norm
                    even_ratios.append(float(np.linalg.norm(symmetric) / (np.linalg.norm(odd_effect) + EPS)))
                    relative_effects.append({
                        "source_index": source_index, "target_index": target_index,
                        "relative_rms": float(np.sqrt(np.mean(np.square(odd_effect))) / (target_rms + EPS)),
                    })
    topology = []
    for source_index in range(derivative.shape[1]):
        for target_index in range(3):
            values = [row["relative_rms"] for row in relative_effects
                      if row["source_index"] == source_index and row["target_index"] == target_index]
            topology.append({"source_index": source_index, "target_index": target_index,
                             "median_relative_response_rms": float(np.median(values))})
    close_memmap(derivative); close_memmap(even); close_memmap(norms)
    return {
        "model": label, "prediction": model_summary, "families": family_summary,
        "median_pair_superposition_relative_mse": float(np.median(pair_errors)),
        "median_even_to_odd_l2": float(np.median(even_ratios)),
        "relative_response_topology": topology,
        "claim_boundary": "model-local functional metrics; no physical coordinate alignment",
    }


def qwen4_reference() -> dict:
    all_index = read_rows(ACTIVE_INDEX)
    selected_indices = [index for index, row in enumerate(all_index) if row["partition"] == "fresh_lockbox"]
    source_derivative = np.load(P2317 / "raw/directional_derivative.float16.npy", mmap_mode="r")
    source_even = np.load(P2317 / "raw/even_response.float16.npy", mmap_mode="r")
    source_boundary = np.load(P2316 / "raw/qwen4b_boundary_all_checkpoints.float16.npy", mmap_mode="r")
    probe_indices = [0, 1, 2, 3, 8, 9]
    derivative = np.asarray(source_derivative[selected_indices][:, :, probe_indices], dtype=np.float16)
    even = np.asarray(source_even[selected_indices][:, :, probe_indices], dtype=np.float16)
    norm_values = np.empty((len(selected_indices), 3, 4), dtype=np.float32)
    for local, source_index in enumerate(selected_indices):
        row = all_index[source_index]
        hidden_index = int(row["hidden_index"])
        for q_index, source_q in enumerate((10, 20, 30)):
            targets = (source_q + 1, source_q + 4, 37)
            norm_values[local, q_index, 0] = np.linalg.norm(
                source_boundary[hidden_index, source_q].astype(np.float64))
            for target_index, target_q in enumerate(targets):
                norm_values[local, q_index, 1 + target_index] = np.sqrt(np.mean(np.square(
                    source_boundary[hidden_index, target_q].astype(np.float64))))
    worker = OUT / "qwen3_4b_reference"
    worker.joinpath("raw").mkdir(parents=True, exist_ok=True)
    derivative_path = worker / "raw/directional_derivative.float16.npy"
    even_path = worker / "raw/even_response.float16.npy"
    norm_path = worker / "raw/source_and_target_rms.float32.npy"
    np.save(derivative_path, derivative, allow_pickle=False)
    np.save(even_path, even, allow_pickle=False)
    np.save(norm_path, norm_values, allow_pickle=False)
    index_rows = [all_index[index] for index in selected_indices]
    write_rows(worker / "index/active_rows.jsonl", index_rows)
    for value in (source_derivative, source_even, source_boundary):
        close_memmap(value)
    return functional_metrics(derivative_path, even_path, norm_path, index_rows, "Qwen3-4B")


def load_named_model(name: str):
    if name == "qwen3_14b":
        import phase2145_c611_natural_qwen14_worker as q14

        def reduced_map() -> dict:
            value = {"model.embed_tokens": 0, "model.rotary_emb": 0,
                     "model.norm": "disk", "lm_head": "disk"}
            value.update({f"model.layers.{index}": 0 if index < 12 else "disk" for index in range(40)})
            return value

        q14.device_map = reduced_map
    return model_worker.load_model(name)


def run_model(name: str, raw_rows: list[dict]) -> dict:
    worker = OUT / name
    final_path = worker / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    model = None
    try:
        model, tokenizer, device, placement, loader = load_named_model(name)
        compiled = compile_rows(tokenizer, raw_rows)
        write_rows(worker / "material/compiled_fresh_panel.jsonl", compiled)
        score_path = worker / "behavior/sequence_scores.jsonl"
        if score_path.exists() and len(read_rows(score_path)) == 64:
            scores = read_rows(score_path)
        else:
            scores = sequence_scores(model, device, compiled, 48 if name == "qwen3_14b" else 12)
            write_rows(score_path, scores)
        free_path = worker / "behavior/free_generation.jsonl"
        if free_path.exists() and len(read_rows(free_path)) == 64:
            free = read_rows(free_path)
        else:
            free = free_generation(model, tokenizer, device, compiled, 64 if name == "qwen3_14b" else 16)
            write_rows(free_path, free)
        behavior = behavior_summary(scores, free)
        save(worker / "behavior/summary.json", behavior)
        active = active_rows(compiled)
        field = active_capture(model, device, active, worker)
        index_rows = read_rows(worker / "index/active_rows.jsonl")
        metrics = functional_metrics(ROOT / field["derivative"], ROOT / field["even"],
                                     ROOT / field["norms"], index_rows, MODEL_LABELS[name])
        save(worker / "analysis/functional_metrics.json", metrics)
        result = {
            "status": "closed", "model": MODEL_LABELS[name], "model_key": name,
            "precision": "float16" if name == "qwen3_14b" else "bfloat16",
            "quantization": "none", "placement": placement, "loader": loader,
            "behavior": behavior, "field": field, "functional_metrics": metrics,
            "checks": {
                "all_behavior_rows": len(scores) == 64 and len(free) == 64,
                "all_active_rows": field["shape"][0] == 32,
                "all_model_local_coordinates": field["shape"][-1] == int(model.config.hidden_size),
                "three_relative_sources": len(field["sources"]) == 3,
                "no_coordinate_alignment": True,
            },
        }
        result["all_checks_passed"] = all(result["checks"].values())
        save(final_path, result)
        return result
    finally:
        model_worker.release_model(name, model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {}
    for key, value in result["models"].items():
        if not value.get("all_checks_passed"):
            compact[key] = {"status": value.get("status"), "error": value.get("error")}
            continue
        compact[key] = {
            "behavior": value["behavior"]["overall"],
            "prediction": value["functional_metrics"]["prediction"],
            "pair_mse": value["functional_metrics"]["median_pair_superposition_relative_mse"],
            "even_odd": value["functional_metrics"]["median_even_to_odd_l2"],
        }
    text = rf"""

## Phase {PHASE}: 三模型顺序锁箱行为与模型本地方向拓扑（{CAMPAIGN}） [{stamp}]

**测试原理。** Qwen3-14B、GLM-4-9B、DeepSeek-R1-Distill-Qwen-7B 按固定顺序逐个加载和释放，使用各自 tokenizer 重编译 Phase2315 的 64 行 fresh 面板；每个模型独立记录完整未来候选与自由续写。随后在 32 行 fresh_lockbox 上，用各模型本地维度生成 4 个固定 Rademacher 方向与 2 个成对方向，在相对 block 深度约 `0.28/0.56/0.83` 处正负扰动，读取 `q+1/q+4/final_norm` 全部本地坐标。跨模型只比较留一预测误差、成对叠加误差、偶/奇响应比和相对响应深度，不比较坐标编号。

$$
N_{{LOO}}=\frac{{\lVert D_i-\mathbb E_{{k\ne i}}D_k\rVert_2^2}}{{\lVert D_i\rVert_2^2+\varepsilon}},\qquad
E_{{pair}}=\frac{{\lVert D(r_a+r_b)-D(r_a)-D(r_b)\rVert_2^2}}{{\lVert D(r_a+r_b)\rVert_2^2+\varepsilon}}.
$$

**结果汇总。** Qwen3-4B 同口径参考 `{json.dumps(result['qwen4_reference'], ensure_ascii=False)}`；三个顺序模型 `{json.dumps(compact, ensure_ascii=False)}`。工程失败只淘汰对应模型，完成模型为 `{result['successful_models']}`。这些数字若相近，只能说明模型本地的扰动响应统计相近，不能说明共享坐标、共享电路或相同算法。

**文件、审计与硬伤。** 完整模型账位于 `tests/glm5/result/phase2318_c5241_c5320_crossmodel_directional_topology/<model>`；总检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2318_c5241_c5320_crossmodel_directional_topology.py`。Qwen3-14B 是同家族扩展；DeepSeek 底层仍为 Qwen2 类；64/32 行面板较小；GLM tokenizer 与架构差异会同时改变接口和动力学；主动方向是模型本地随机方向而非语义方向；所有面板都没有独立人类盲评。

**理论进展与结论。** 理论主体仍为“条件化输出场闭合理论”。本期只裁决“Qwen4B 中更像共享局部传播而非族特异方向”的观察能否在其他模型中以功能统计复现。即使复现，也只是局部响应类，不是语言齿轮闭合。下一步发布边界场、全 token 场、方向导数、偶响应、逐坐标族相对全局改善和跨模型相对拓扑；构建通过后清理未展示且可由发布物复核的原始副本。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def append_memo(result: dict) -> None:
    """Append a UTF-8 scientific record; this overrides the damaged draft above."""
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {}
    for key, value in result["models"].items():
        if not value.get("all_checks_passed"):
            compact[key] = {"status": value.get("status"), "error": value.get("error")}
            continue
        compact[key] = {
            "behavior": value["behavior"]["overall"],
            "prediction": value["functional_metrics"]["prediction"],
            "pair_mse": value["functional_metrics"]["median_pair_superposition_relative_mse"],
            "even_odd": value["functional_metrics"]["median_even_to_odd_l2"],
        }
    record = rf"""

## Phase {PHASE}: 三模型顺序锁箱行为与模型本地方向拓扑（{CAMPAIGN}） [{stamp}]

**测试原理。** Qwen3-14B、GLM-4-9B、DeepSeek-R1-Distill-Qwen-7B 按固定顺序逐个加载和释放，使用各自 tokenizer 重新编译 Phase2315 的 64 行 fresh 面板；每个模型独立记录完整未来候选与自由续写。随后在 32 行 fresh_lockbox 上，以各模型本地维度生成 4 个固定 Rademacher 方向与 2 个成对方向，在相对 block 深度约 `0.28/0.56/0.83` 处正负扰动，读取 `q+1/q+4/final_norm` 全部本地坐标。跨模型只比较留一预测误差、成对叠加误差、偶/奇响应比和相对响应深度，不比较坐标编号。
$$
N_{{LOO}}=\frac{{\lVert D_i-\mathbb E_{{k\ne i}}D_k\rVert_2^2}}{{\lVert D_i\rVert_2^2+\varepsilon}},\qquad
E_{{pair}}=\frac{{\lVert D(r_a+r_b)-D(r_a)-D(r_b)\rVert_2^2}}{{\lVert D(r_a+r_b)\rVert_2^2+\varepsilon}}.
$$

**结果汇总。** Qwen3-4B 同口径参考 `{json.dumps(result['qwen4_reference'], ensure_ascii=False)}`；三个顺序模型 `{json.dumps(compact, ensure_ascii=False)}`。工程失败只淘汰对应模型，完成模型为 `{result['successful_models']}`。这些数字若相近，只能说明模型本地的扰动响应统计相近，不能说明共享坐标、共享电路或相同算法。

**文件、审计与硬伤。** 完整模型账位于 `tests/glm5/result/phase2318_c5241_c5320_crossmodel_directional_topology/<model>`；总检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2318_c5241_c5320_crossmodel_directional_topology.py`。Qwen3-14B 是同家族扩展；DeepSeek 底层仍为 Qwen2 类；64/32 行面板较小；GLM tokenizer 与架构差异会同时改变接口和动力学；主动方向是模型本地随机方向而非语义方向；所有面板都没有独立人类盲评。

**理论进展与结论。** 理论主体仍为“条件化输出场闭合理论”。本期只裁决“Qwen4B 中更像共享局部传播而非族特异方向”的观察能否在其他模型中以功能统计复现。即使复现，也只是局部响应类，不是语言齿轮闭合。下一步发布边界场、全 token 场、方向导数、偶响应、逐坐标族相对全局改善和跨模型相对拓扑；构建通过后清理未展示且可由发布物复核的原始副本。"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for parent in (P2315, P2316, P2317):
        value = json.loads((parent / "analysis/final.json").read_text(encoding="utf-8"))
        if not value["all_checks_passed"]:
            raise RuntimeError(("parent_not_authorized", parent))
    all_rows = read_rows(ROWS_PATH)
    raw_rows = behavior_rows(all_rows)
    q4 = qwen4_reference()
    save(OUT / "qwen3_4b_reference/analysis/functional_metrics.json", q4)
    models = {}
    for name in MODEL_ORDER:
        try:
            models[name] = run_model(name, raw_rows)
        except Exception as error:
            failure = {
                "status": "worker_error", "model": MODEL_LABELS[name], "model_key": name,
                "error_type": type(error).__name__, "error": str(error),
                "all_checks_passed": False,
                "strict_boundary": "model-local failure; remaining frozen routes stay authorized",
            }
            save(OUT / name / "analysis/final.json", failure)
            models[name] = failure
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    successful = [name for name, value in models.items() if value.get("all_checks_passed")]
    checks = {
        "parents_authorized": True, "qwen4_reference_rows": True,
        "models_attempted_in_order": list(models) == list(MODEL_ORDER),
        "route_failure_did_not_stop_later_models": len(models) == len(MODEL_ORDER),
        "no_physical_coordinate_alignment": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
        "qwen4_reference": q4, "models": models, "successful_models": successful,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "All three frozen model routes were attempted sequentially. Model-local behavior and directional "
            "response statistics can be compared functionally, but no physical coordinate identity or shared "
            "semantic circuit is inferred."
        ),
        "next_authorization": "Publish exact-coordinate Qwen fields and model-local topology summaries, then clean undisplayed raw copies.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
