#!/usr/bin/env python3
"""Run held-out matched-difference interventions for Phase556.

Candidate coordinates are selected without confirmation worlds 48-95.  This
executor reads only behavior-qualified anchors in that reserved open subset.
It patches one query-end component with a matched one-factor state difference;
wrong-depth and channel-roll differences are simultaneous magnitude controls.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase556_fruit_encoding_protocol import CELLS  # noqa: E402


MODELS = ("qwen3", "glm4")
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
CASES_PATH = OUT_DIR / "phase556_open_cases.jsonl"
QUALIFICATION_PATH = OUT_DIR / "phase556_behavior_qualification.jsonl"
CANDIDATES_PATH = OUT_DIR / "phase556_causal_candidate_registry.json"
HOLDOUT_START = 48
HOLDOUT_STOP = 96
DEFAULT_ANCHOR_LIMIT = 12
SCENARIOS = ("matched_factor_delta", "wrong_depth_delta", "channel_roll_delta")
READOUT_CONTRACT = "first_non_whitespace_candidate_content_token_v2"


def observer_prompt(model: str, prompt: str) -> str:
    return prompt + "\n" if model == "glm4" else prompt


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(model: str) -> Path:
    return OUT_DIR / "causal_intervention" / model / "phase556_causal_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / "causal_intervention" / model / "phase556_causal_execution_summary.json"


def tensor_from_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected hook output: {type(value).__name__}")
    return value


def replace_primary(output: Any, primary: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (primary, *output[1:])
    return primary


def candidate_id(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['model']}__{candidate['mechanism']}__{candidate['component']}__"
        f"L{candidate['layer']}__rank{candidate['component_rank']}"
    )


def word_token_ids(tokenizer: Any, word: str) -> list[int]:
    ids: set[int] = set()
    for text in (word, " " + word, "\n" + word):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        for token_id in encoded:
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            if piece.strip():
                ids.add(int(token_id))
                break
    if not ids:
        raise ValueError(f"No token id for candidate {word!r}")
    return sorted(ids)


def word_scores(logits: torch.Tensor, tokenizer: Any, words: list[str]) -> dict[str, float]:
    token_sets = {word: set(word_token_ids(tokenizer, word)) for word in words}
    for index, word in enumerate(words):
        for other in words[index + 1:]:
            overlap = token_sets[word] & token_sets[other]
            if overlap:
                raise ValueError(
                    f"Restricted candidate first-token collision: {word!r}/{other!r} -> {sorted(overlap)}"
                )
    result: dict[str, float] = {}
    for word in words:
        token_ids = sorted(token_sets[word])
        values = logits[token_ids].float()
        finite = values[torch.isfinite(values)]
        result[word] = float(finite.max().item()) if finite.numel() else -1.0e30
    return result


def scores_are_valid(scores: dict[str, float]) -> bool:
    return bool(scores) and all(math.isfinite(value) and value > -1.0e29 for value in scores.values())


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def safe_fraction(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-8:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def authorized_holdout_anchors(model: str, anchor_limit: int) -> list[str]:
    qualifications = {row["model"]: row for row in read_jsonl(QUALIFICATION_PATH)}
    report = qualifications.get(model)
    if not report or not report["internal_collection_authorized"]:
        return []
    behavior = read_jsonl(OUT_DIR / f"phase556_{model}_behavior_rows.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in behavior:
        if (
            row["split"] == "independent_confirmation"
            and row["case_type"] == "controlled_factorial"
            and HOLDOUT_START <= int(row["world_index"]) < HOLDOUT_STOP
        ):
            grouped.setdefault(row["anchor_id"], []).append(row)
    qualified = sorted(
        anchor for anchor, rows in grouped.items()
        if len(rows) == 16 and all(row["semantic_correct"] for row in rows)
    )
    return qualified[:anchor_limit]


def matched_pairs(rows: list[dict[str, Any]], mechanism: str) -> list[dict[str, Any]]:
    by_factors = {
        tuple(int(row["factor_values"][factor]) for factor in ("entity", "category", "query", "binding")): row
        for row in rows
    }
    pairs: list[dict[str, Any]] = []
    if mechanism == "category_reuse":
        factor = "category"
        target_query = 0
        for entity in (0, 1):
            for query in (0, 1):
                for binding in (0, 1):
                    recipient = by_factors[(entity, 0, query, binding)]
                    donor = by_factors[(entity, 1, query, binding)]
                    pairs.append({
                        "factor": factor,
                        "pair_role": "target" if query == target_query else "specificity_control",
                        "recipient": recipient,
                        "donor": donor,
                    })
    elif mechanism == "attribute_binding":
        factor = "binding"
        target_query = 1
        for entity in (0, 1):
            for category in (0, 1):
                for query in (0, 1):
                    recipient = by_factors[(entity, category, query, 0)]
                    donor = by_factors[(entity, category, query, 1)]
                    pairs.append({
                        "factor": factor,
                        "pair_role": "target" if query == target_query else "specificity_control",
                        "recipient": recipient,
                        "donor": donor,
                    })
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    return pairs


def run(model_key: str, restart: bool, anchor_limit: int, batch_size: int) -> Path:
    anchors = authorized_holdout_anchors(model_key, anchor_limit)
    if not anchors:
        raise RuntimeError(f"No Phase556 causal holdout anchors authorized for {model_key}")
    registry = read_json(CANDIDATES_PATH)
    candidates = [row for row in registry["candidates"] if row["model"] == model_key]
    if not candidates:
        raise RuntimeError(f"No frozen Phase556 candidates for {model_key}")
    cases = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_key
        and row["split"] == "independent_confirmation"
        and row["case_type"] == "controlled_factorial"
        and row["anchor_id"] in anchors
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        grouped.setdefault(row["anchor_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: CELLS.index(row["factorial_cell"]))
        if [row["factorial_cell"] for row in rows] != list(CELLS):
            raise RuntimeError("Phase556 causal factorial order drift")

    output = output_path(model_key)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(model_key).unlink(missing_ok=True)
    completed = {row["anchor_id"] for row in read_jsonl(output)} if output.exists() else set()
    loaded = None
    started = time.monotonic()
    new_anchor_count = 0
    try:
        loaded = load_probe_model(model_key)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        expected_layer_count = {int(row["layer_count"]) for row in candidates}
        if expected_layer_count != {len(layers)}:
            raise RuntimeError(f"Phase556 candidate layer-count drift: {expected_layer_count} != {len(layers)}")

        coordinate_specs: set[tuple[str, int]] = set()
        wrong_layers: dict[str, int] = {}
        for candidate in candidates:
            layer = int(candidate["layer"])
            wrong_layer = (layer + max(2, len(layers) // 2)) % len(layers)
            if wrong_layer == layer:
                wrong_layer = (layer + 1) % len(layers)
            wrong_layers[candidate_id(candidate)] = wrong_layer
            coordinate_specs.add((candidate["component"], layer))
            coordinate_specs.add((candidate["component"], wrong_layer))

        for anchor_index, (anchor_id, rows) in enumerate(sorted(grouped.items()), 1):
            if anchor_id in completed:
                continue
            case_index = {row["case_id"]: index for index, row in enumerate(rows)}
            captures: dict[tuple[str, int], torch.Tensor] = {}
            handles: list[Any] = []

            def make_capture_pre(coordinate: tuple[str, int]):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    captures[coordinate] = inputs[0][:, -1, :].detach().float().cpu()
                return hook

            def make_capture_forward(coordinate: tuple[str, int]):
                def hook(_module: Any, _inputs: tuple[Any, ...], output_value: Any) -> None:
                    captures[coordinate] = tensor_from_output(output_value)[:, -1, :].detach().float().cpu()
                return hook

            for component, layer_index in sorted(coordinate_specs):
                if component == "layer_input":
                    handles.append(layers[layer_index].register_forward_pre_hook(make_capture_pre((component, layer_index))))
                elif component == "attention_output":
                    handles.append(layers[layer_index].self_attn.register_forward_hook(make_capture_forward((component, layer_index))))
                elif component == "mlp_output":
                    handles.append(layers[layer_index].mlp.register_forward_hook(make_capture_forward((component, layer_index))))
                else:
                    raise ValueError(f"Unsupported component: {component}")
            encoded = loaded.tokenizer(
                [observer_prompt(model_key, row["prompt"]) for row in rows], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                natural = loaded.model(**encoded, use_cache=False)
            natural_logits = natural.logits[:, -1, :].detach().float().cpu()
            for handle in handles:
                handle.remove()
            if set(captures) != coordinate_specs:
                missing = sorted(coordinate_specs - set(captures))
                raise RuntimeError(f"Missing Phase556 natural captures: {missing}")

            baseline_scores: dict[str, dict[str, float]] = {}
            for index, row in enumerate(rows):
                words = sorted(set(row["all_candidates"]))
                baseline_scores[row["case_id"]] = word_scores(natural_logits[index], loaded.tokenizer, words)

            anchor_output: list[dict[str, Any]] = []
            for candidate in candidates:
                cid = candidate_id(candidate)
                component = candidate["component"]
                layer_index = int(candidate["layer"])
                wrong_layer = wrong_layers[cid]
                coordinate = (component, layer_index)
                wrong_coordinate = (component, wrong_layer)
                tasks: list[dict[str, Any]] = []
                for pair in matched_pairs(rows, candidate["mechanism"]):
                    directions = (
                        ("factor_0_to_1", pair["recipient"], pair["donor"]),
                        ("factor_1_to_0", pair["donor"], pair["recipient"]),
                    )
                    for intervention_direction, recipient, donor in directions:
                        recipient_index = case_index[recipient["case_id"]]
                        donor_index = case_index[donor["case_id"]]
                        correct_delta = captures[coordinate][donor_index] - captures[coordinate][recipient_index]
                        wrong_delta = captures[wrong_coordinate][donor_index] - captures[wrong_coordinate][recipient_index]
                        roll_shift = 1 + int(hashlib.sha256(
                            f"{cid}|{anchor_id}|{recipient['case_id']}".encode("utf-8")
                        ).hexdigest()[:8], 16) % max(1, correct_delta.numel() - 1)
                        deltas = {
                            "matched_factor_delta": correct_delta,
                            "wrong_depth_delta": wrong_delta,
                            "channel_roll_delta": torch.roll(correct_delta, shifts=roll_shift, dims=-1),
                        }
                        for scenario in SCENARIOS:
                            tasks.append({
                                "factor": pair["factor"],
                                "pair_role": pair["pair_role"],
                                "recipient": recipient,
                                "donor": donor,
                                "intervention_direction": intervention_direction,
                                "scenario": scenario,
                                "delta": deltas[scenario],
                                "roll_shift": roll_shift if scenario == "channel_roll_delta" else None,
                            })

                for batch_start in range(0, len(tasks), batch_size):
                    batch_tasks = tasks[batch_start:batch_start + batch_size]
                    batch = loaded.tokenizer(
                        [observer_prompt(model_key, task["recipient"]["prompt"]) for task in batch_tasks],
                        return_tensors="pt", padding=True, truncation=True, max_length=512,
                    )
                    batch = {key: value.to(loaded.input_device) for key, value in batch.items()}
                    patch_deltas = torch.stack([task["delta"] for task in batch_tasks]).to(loaded.input_device)

                    def patch_primary(primary: torch.Tensor) -> torch.Tensor:
                        value = primary.clone()
                        value[:, -1, :] = value[:, -1, :] + patch_deltas.to(value.dtype)
                        return value

                    if component == "layer_input":
                        def patch_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                            return (patch_primary(inputs[0]), *inputs[1:])
                        patch_handle = layers[layer_index].register_forward_pre_hook(patch_hook)
                    else:
                        def patch_hook(_module: Any, _inputs: tuple[Any, ...], output_value: Any) -> Any:
                            return replace_primary(output_value, patch_primary(tensor_from_output(output_value)))
                        module = layers[layer_index].self_attn if component == "attention_output" else layers[layer_index].mlp
                        patch_handle = module.register_forward_hook(patch_hook)
                    with torch.inference_mode():
                        patched = loaded.model(**batch, use_cache=False)
                    patch_handle.remove()
                    patched_logits = patched.logits[:, -1, :].detach().float().cpu()
                    for offset, task in enumerate(batch_tasks):
                        recipient = task["recipient"]
                        donor = task["donor"]
                        words = sorted(set(recipient["all_candidates"] + donor["all_candidates"]))
                        recipient_scores = baseline_scores[recipient["case_id"]]
                        donor_scores = baseline_scores[donor["case_id"]]
                        patched_scores = word_scores(patched_logits[offset], loaded.tokenizer, words)
                        recipient_target = recipient["target"]
                        donor_target = donor["target"]
                        baseline_valid = scores_are_valid(recipient_scores)
                        donor_valid = scores_are_valid(donor_scores)
                        patched_valid = scores_are_valid(patched_scores)
                        numerical_valid = baseline_valid and donor_valid and patched_valid
                        baseline_choice = max(recipient_scores, key=recipient_scores.get) if baseline_valid else None
                        patched_choice = max(patched_scores, key=patched_scores.get) if patched_valid else None
                        donor_choice = max(donor_scores, key=donor_scores.get) if donor_valid else None
                        if recipient_target != donor_target and numerical_valid:
                            baseline_margin = recipient_scores[donor_target] - recipient_scores[recipient_target]
                            donor_margin = donor_scores[donor_target] - donor_scores[recipient_target]
                            patched_margin = patched_scores[donor_target] - patched_scores[recipient_target]
                            transfer_fraction = safe_fraction(
                                patched_margin - baseline_margin,
                                donor_margin - baseline_margin,
                            )
                        else:
                            baseline_margin = None
                            donor_margin = None
                            patched_margin = None
                            transfer_fraction = None
                        anchor_output.append({
                            "schema_version": "phase556_causal_intervention_row.v1",
                            "phase_id": "Phase556",
                            "created_at": now(),
                            "model": model_key,
                            "torch_dtype": run_dtype,
                            "split": "independent_confirmation_causal_holdout",
                            "anchor_id": anchor_id,
                            "world_index": int(recipient["world_index"]),
                            "candidate_id": cid,
                            "mechanism": candidate["mechanism"],
                            "component": component,
                            "layer": layer_index,
                            "relative_depth": candidate["relative_depth"],
                            "wrong_depth_layer": wrong_layer,
                            "component_rank": candidate["component_rank"],
                            "pair_role": task["pair_role"],
                            "manipulated_factor": task["factor"],
                            "scenario": task["scenario"],
                            "intervention_direction": task["intervention_direction"],
                            "recipient_case_id": recipient["case_id"],
                            "donor_case_id": donor["case_id"],
                            "recipient_factor_values": recipient["factor_values"],
                            "donor_factor_values": donor["factor_values"],
                            "recipient_target": recipient_target,
                            "donor_target": donor_target,
                            "baseline_choice": baseline_choice,
                            "natural_donor_choice": donor_choice,
                            "patched_choice": patched_choice,
                            "baseline_semantic_correct_restricted": baseline_choice == recipient_target,
                            "natural_donor_semantic_correct_restricted": donor_choice == donor_target,
                            "patched_donor_selected": patched_choice == donor_target,
                            "patched_recipient_preserved": patched_choice == recipient_target,
                            "baseline_donor_margin": baseline_margin,
                            "natural_donor_margin": donor_margin,
                            "patched_donor_margin": patched_margin,
                            "transfer_fraction": transfer_fraction,
                            "recipient_target_logit_delta": finite_or_none(
                                patched_scores[recipient_target] - recipient_scores[recipient_target]
                            ) if numerical_valid else None,
                            "patch_delta_norm": finite_or_none(float(task["delta"].norm().item())),
                            "baseline_logits_valid": baseline_valid,
                            "natural_donor_logits_valid": donor_valid,
                            "patched_logits_valid": patched_valid,
                            "numerical_valid": numerical_valid,
                            "channel_roll_shift": task["roll_shift"],
                            "matched_pair_changes_exactly_one_factor": True,
                            "candidate_selected_without_causal_holdout": True,
                            "query_end_only": model_key != "glm4",
                            "single_semantic_position_only": True,
                            "semantic_position": (
                                "answer_content_boundary_after_natural_newline"
                                if model_key == "glm4" else "query_end"
                            ),
                            "observer_prefix": "\n" if model_key == "glm4" else "",
                            "restricted_readout_contract": READOUT_CONTRACT,
                            "compute_edge": component != "layer_input",
                            "intervention_executed": True,
                            "causal_qualified": False,
                            "sealed": False,
                        })
                    del patched, patched_logits, batch, patch_deltas
            append_jsonl(output, anchor_output)
            new_anchor_count += 1
            del natural, natural_logits, encoded, captures, anchor_output
            if new_anchor_count == 1 or new_anchor_count % 4 == 0 or anchor_index == len(grouped):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_key} Phase556 causal "
                    f"{len(completed) + new_anchor_count}/{len(grouped)} anchors",
                    flush=True,
                )

        final_rows = read_jsonl(output)
        observed = {row["anchor_id"] for row in final_rows}
        expected_per_anchor = len(candidates) * 16 * len(SCENARIOS)
        expected_rows = len(grouped) * expected_per_anchor
        if observed != set(grouped) or len(final_rows) != expected_rows:
            raise RuntimeError(
                f"Incomplete Phase556 causal rows {model_key}: {len(final_rows)}/{expected_rows}"
            )
        summary = {
            "schema_version": "phase556_causal_execution_summary.v1",
            "phase_id": "Phase556",
            "created_at": now(),
            "status": "complete",
            "model": model_key,
            "torch_dtype": run_dtype,
            "anchor_count": len(grouped),
            "candidate_count": len(candidates),
            "scenarios": list(SCENARIOS),
            "directed_pairs_per_candidate_anchor": 16,
            "row_count": len(final_rows),
            "new_anchor_count_this_invocation": new_anchor_count,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "holdout_world_range": [HOLDOUT_START, HOLDOUT_STOP - 1],
            "candidate_selection_consumed_holdout": False,
            "restricted_readout_contract": READOUT_CONTRACT,
            "sealed_split_read": False,
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
        }
        write_json(summary_path(model_key), summary)
        print(summary_path(model_key))
        return summary_path(model_key)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--anchor-limit", type=int, default=DEFAULT_ANCHOR_LIMIT)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run(args.model, args.restart, args.anchor_limit, args.batch_size)
