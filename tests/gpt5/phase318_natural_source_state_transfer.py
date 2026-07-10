#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase305_internal_semantic_physical_path_probe as p305  # noqa: E402
import phase311_core_language_physical_atlas as p311  # noqa: E402
import phase317_natural_source_boundary_case_bank as p317  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
from model_utils import get_layers  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor  # noqa: E402


PHASE = "Phase318"
SCHEMA_VERSION = "4.1.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "natural_source_state_transfer"
OUT = ROOT / "tests/gpt5/result/phase318_natural_source_state_transfer"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"
LAYER_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
CALIBRATION_CONDITIONS = ["baseline", "source_replace", "unrelated_replace", "functional_control_replace", "wrong_position_replace", "feature_permute_replace"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def replace_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple) and output:
        return (tensor, *output[1:])
    return output


def candidate_layers(n_layers: int) -> list[int]:
    return sorted({min(n_layers - 1, max(0, round((n_layers - 1) * fraction))) for fraction in LAYER_FRACTIONS})


def encode_case(tokenizer: Any, case: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    encoded = tokenizer(str(case["prompt"]), return_tensors="pt", truncation=True, max_length=1536)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    positions = p311.locate_positions(tokenizer, case, str(case["prompt"]), last_pos)
    return {key: value.to(device) for key, value in encoded.items()}, positions


def token_group(tokenizer: Any, aliases: list[str]) -> list[int]:
    return p305.token_ids(tokenizer, [str(x) for x in aliases])


def group_logit(logits: torch.Tensor, ids: list[int]) -> float:
    valid = [int(i) for i in ids if 0 <= int(i) < logits.shape[-1]]
    return float(torch.logsumexp(logits[valid].float(), dim=0).item()) if valid else float("-inf")


def boundary_diag(tokenizer: Any, logits: torch.Tensor, donor: dict[str, Any], recipient: dict[str, Any]) -> dict[str, Any]:
    donor_ids = token_group(tokenizer, donor["target_aliases"])
    recipient_ids = token_group(tokenizer, recipient["target_aliases"])
    donor_logit = group_logit(logits, donor_ids)
    recipient_logit = group_logit(logits, recipient_ids)
    top1_id = int(torch.argmax(logits).item())
    return {
        "donor_target_logit": round(donor_logit, 6),
        "recipient_target_logit": round(recipient_logit, 6),
        "donor_vs_recipient_margin": round(donor_logit - recipient_logit, 6),
        "donor_wins": donor_logit > recipient_logit,
        "full_vocab_top1_id": top1_id,
        "full_vocab_top1_text": tokenizer.decode([top1_id]),
    }


def distribution_shift(base_logits: torch.Tensor, patched_logits: torch.Tensor) -> dict[str, float]:
    base_logp = torch.log_softmax(base_logits.float(), dim=-1)
    patched_logp = torch.log_softmax(patched_logits.float(), dim=-1)
    base_p = base_logp.exp()
    patched_p = patched_logp.exp()
    midpoint = 0.5 * (base_p + patched_p)
    midpoint_log = torch.log(midpoint.clamp_min(1e-12))
    js = 0.5 * torch.sum(base_p * (base_logp - midpoint_log)) + 0.5 * torch.sum(patched_p * (patched_logp - midpoint_log))
    return {
        "js_divergence": round(float(js.item()), 6),
        "logit_l2": round(float(torch.linalg.vector_norm(patched_logits.float() - base_logits.float()).item()), 6),
    }


def projection_fraction(base: torch.Tensor, patched: torch.Tensor, donor: torch.Tensor) -> float:
    direction = donor.float() - base.float()
    denominator = float(torch.dot(direction, direction).item())
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot((patched.float() - base.float()), direction).item() / denominator)


def patch_hook(position: int, vector: torch.Tensor, mode: str = "replace"):
    source = vector.detach().float().cpu()

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = extract_tensor(output)
        if tensor is None or not torch.is_tensor(tensor) or tensor.ndim != 3 or position >= tensor.shape[1]:
            return output
        patched = tensor.clone()
        vec = source.to(device=patched.device, dtype=patched.dtype)
        if mode == "permute":
            vec = torch.roll(vec, shifts=max(1, vec.shape[-1] // 7), dims=-1)
        patched[0, position, :] = vec
        return replace_tensor(output, patched)

    return hook


def forward_states(
    model_obj: Any,
    encoded: dict[str, torch.Tensor],
    positions: dict[str, dict[str, Any]],
    patch_layer: int | None = None,
    patch_position: int | None = None,
    patch_vector: torch.Tensor | None = None,
    patch_mode: str = "replace",
) -> tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]]]:
    handles = []
    if patch_layer is not None and patch_position is not None and patch_vector is not None:
        handles.append(get_layers(model_obj)[int(patch_layer)].register_forward_hook(patch_hook(int(patch_position), patch_vector, patch_mode)))
    try:
        with torch.inference_mode():
            output = model_obj(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
        last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
        logits = output.logits[0, last_pos].detach().float().cpu()
        states: dict[str, dict[int, torch.Tensor]] = {role: {} for role in positions}
        n_layers = len(output.hidden_states) - 1
        for role, meta in positions.items():
            position = int(meta["token_position"])
            for layer in range(n_layers):
                states[role][layer] = output.hidden_states[layer + 1][0, position].detach().float().cpu()
        del output
        return logits, states
    finally:
        for handle in handles:
            handle.remove()


def get_case_maps() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cases = read_jsonl(V2 / "phase317_natural_source_case_bank.jsonl")
    pairs = read_jsonl(V2 / "phase317_natural_source_pair_bank.jsonl")
    if not cases or not pairs:
        p317.prepare()
        cases = read_jsonl(V2 / "phase317_natural_source_case_bank.jsonl")
        pairs = read_jsonl(V2 / "phase317_natural_source_pair_bank.jsonl")
    return {str(row["case_id"]): row for row in cases}, pairs


def load_baseline(
    cache: dict[str, tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]], dict[str, dict[str, Any]], dict[str, torch.Tensor]]],
    case: dict[str, Any],
    model_obj: Any,
    tokenizer: Any,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]], dict[str, dict[str, Any]], dict[str, torch.Tensor]]:
    case_id = str(case["case_id"])
    if case_id not in cache:
        encoded, positions = encode_case(tokenizer, case, device)
        logits, states = forward_states(model_obj, encoded, positions)
        cache[case_id] = (logits, states, positions, encoded)
    return cache[case_id]


def transfer_row(
    model: str,
    pair: dict[str, Any],
    layer: int,
    condition: str,
    base_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    donor: dict[str, Any],
    recipient: dict[str, Any],
    tokenizer: Any,
    patch_position_role: str,
) -> dict[str, Any]:
    base_diag = boundary_diag(tokenizer, base_logits, donor, recipient)
    patched_diag = boundary_diag(tokenizer, patched_logits, donor, recipient)
    shift = safe_float(patched_diag["donor_vs_recipient_margin"]) - safe_float(base_diag["donor_vs_recipient_margin"])
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "pair_id": pair["pair_id"],
        "family_id": pair["family_id"],
        "mechanism_id": pair["mechanism_id"],
        "split": pair["split"],
        "template_id": pair["template_id"],
        "recipient_case_id": pair["recipient_case_id"],
        "donor_case_id": pair["donor_case_id"],
        "condition": condition,
        "source_layer": int(layer),
        "patch_position_role": patch_position_role,
        "base_donor_vs_recipient_margin": base_diag["donor_vs_recipient_margin"],
        "patched_donor_vs_recipient_margin": patched_diag["donor_vs_recipient_margin"],
        "donor_transfer_shift": round(shift, 6),
        "base_donor_wins": base_diag["donor_wins"],
        "patched_donor_wins": patched_diag["donor_wins"],
        "full_vocab_top1_changed": base_diag["full_vocab_top1_id"] != patched_diag["full_vocab_top1_id"],
        "base_top1_text": base_diag["full_vocab_top1_text"],
        "patched_top1_text": patched_diag["full_vocab_top1_text"],
        **distribution_shift(base_logits, patched_logits),
    }


def propagation_rows(
    model: str,
    pair: dict[str, Any],
    source_layer: int,
    base_states: dict[str, dict[int, torch.Tensor]],
    patched_states: dict[str, dict[int, torch.Tensor]],
    donor_states: dict[str, dict[int, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_layers = len(base_states["last"])
    for role in ["query", "last"]:
        for layer in range(source_layer, n_layers):
            fraction = projection_fraction(base_states[role][layer], patched_states[role][layer], donor_states[role][layer])
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": model,
                    "pair_id": pair["pair_id"],
                    "family_id": pair["family_id"],
                    "mechanism_id": pair["mechanism_id"],
                    "split": pair["split"],
                    "source_layer": int(source_layer),
                    "downstream_layer": int(layer),
                    "position_role": role,
                    "donor_direction_projection_fraction": round(fraction, 6),
                    "patched_delta_norm": round(float(torch.linalg.vector_norm(patched_states[role][layer] - base_states[role][layer]).item()), 6),
                    "donor_recipient_distance": round(float(torch.linalg.vector_norm(donor_states[role][layer] - base_states[role][layer]).item()), 6),
                }
            )
    return rows


def select_layers(discovery_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in discovery_rows:
        grouped[(str(row["family_id"]), str(row["mechanism_id"]), int(row["source_layer"]), str(row["condition"]))].append(safe_float(row["donor_transfer_shift"]))
    selections: list[dict[str, Any]] = []
    mechanisms = sorted({(str(r["family_id"]), str(r["mechanism_id"])) for r in discovery_rows})
    for family, mechanism in mechanisms:
        layers = sorted({key[2] for key in grouped if key[0] == family and key[1] == mechanism})
        scored = []
        for layer in layers:
            correct = mean(grouped.get((family, mechanism, layer, "source_replace"), [0.0]))
            unrelated = mean(grouped.get((family, mechanism, layer, "unrelated_replace"), [0.0]))
            scored.append((correct - unrelated, correct, unrelated, layer))
        corrected, correct, unrelated, layer = max(scored)
        selections.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "family_id": family,
                "mechanism_id": mechanism,
                "selected_source_layer": int(layer),
                "discovery_correct_transfer_mean": round(correct, 6),
                "discovery_unrelated_transfer_mean": round(unrelated, 6),
                "discovery_control_corrected_transfer": round(corrected, 6),
                "selection_split": "discovery_only",
            }
        )
    return selections


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    case_map, pairs = get_case_maps()
    model_obj = tokenizer = None
    discovery_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    propagation: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    cache: dict[str, tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]], dict[str, dict[str, Any]], dict[str, torch.Tensor]]] = {}
    selections: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        layers = get_layers(model_obj)
        scan_layers = candidate_layers(len(layers))
        discovery_pairs = [row for row in pairs if row["split"] == "discovery"]
        if args.limit:
            discovery_pairs = discovery_pairs[: args.limit]
        for index, pair in enumerate(discovery_pairs, 1):
            try:
                recipient = case_map[str(pair["recipient_case_id"])]
                donor = case_map[str(pair["donor_case_id"])]
                unrelated = case_map[str(pair["unrelated_control_case_id"])]
                rec_logits, rec_states, rec_positions, rec_encoded = load_baseline(cache, recipient, model_obj, tokenizer, device)
                _, donor_states, _, _ = load_baseline(cache, donor, model_obj, tokenizer, device)
                _, unrelated_states, _, _ = load_baseline(cache, unrelated, model_obj, tokenizer, device)
                source_position = int(rec_positions["source"]["token_position"])
                for layer in scan_layers:
                    patched_logits, patched_states = forward_states(model_obj, rec_encoded, rec_positions, layer, source_position, donor_states["source"][layer])
                    unrelated_logits, _ = forward_states(model_obj, rec_encoded, rec_positions, layer, source_position, unrelated_states["source"][layer])
                    discovery_rows.append(transfer_row(args.model, pair, layer, "source_replace", rec_logits, patched_logits, donor, recipient, tokenizer, "source"))
                    discovery_rows.append(transfer_row(args.model, pair, layer, "unrelated_replace", rec_logits, unrelated_logits, donor, recipient, tokenizer, "source"))
                    propagation.extend(propagation_rows(args.model, pair, layer, rec_states, patched_states, donor_states))
                print(f"{args.model}: discovery source scan {index}/{len(discovery_pairs)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                missing.append({"phase_id": PHASE, "model": args.model, "pair_id": pair["pair_id"], "split": "discovery", "reason": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        selections = select_layers(discovery_rows)
        selection_map = {(str(row["family_id"]), str(row["mechanism_id"])): int(row["selected_source_layer"]) for row in selections}
        calibration_pairs = [row for row in pairs if row["split"] == "calibration"]
        if args.limit:
            calibration_pairs = calibration_pairs[: args.limit]
        for index, pair in enumerate(calibration_pairs, 1):
            try:
                recipient = case_map[str(pair["recipient_case_id"])]
                donor = case_map[str(pair["donor_case_id"])]
                unrelated = case_map[str(pair["unrelated_control_case_id"])]
                functional = case_map[str(pair["same_target_control_case_id"])]
                layer = selection_map[(str(pair["family_id"]), str(pair["mechanism_id"]))]
                rec_logits, rec_states, rec_positions, rec_encoded = load_baseline(cache, recipient, model_obj, tokenizer, device)
                _, donor_states, _, _ = load_baseline(cache, donor, model_obj, tokenizer, device)
                _, unrelated_states, _, _ = load_baseline(cache, unrelated, model_obj, tokenizer, device)
                _, functional_states, _, _ = load_baseline(cache, functional, model_obj, tokenizer, device)
                source_position = int(rec_positions["source"]["token_position"])
                query_position = int(rec_positions["query"]["token_position"])
                condition_specs = {
                    "source_replace": (source_position, donor_states["source"][layer], "replace", "source"),
                    "unrelated_replace": (source_position, unrelated_states["source"][layer], "replace", "source"),
                    "functional_control_replace": (source_position, functional_states["source"][layer], "replace", "source"),
                    "wrong_position_replace": (query_position, donor_states["source"][layer], "replace", "query"),
                    "feature_permute_replace": (source_position, donor_states["source"][layer], "permute", "source"),
                }
                base_row = transfer_row(args.model, pair, layer, "baseline", rec_logits, rec_logits, donor, recipient, tokenizer, "none")
                calibration_rows.append(base_row)
                for condition, (position, vector, mode, role) in condition_specs.items():
                    patched_logits, patched_states = forward_states(model_obj, rec_encoded, rec_positions, layer, position, vector, mode)
                    calibration_rows.append(transfer_row(args.model, pair, layer, condition, rec_logits, patched_logits, donor, recipient, tokenizer, role))
                    if condition == "source_replace":
                        propagation.extend(propagation_rows(args.model, pair, layer, rec_states, patched_states, donor_states))
                print(f"{args.model}: calibration controls {index}/{len(calibration_pairs)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                missing.append({"phase_id": PHASE, "model": args.model, "pair_id": pair["pair_id"], "split": "calibration", "reason": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out_dir = OUT / args.round_name
        for row in selections:
            row["model"] = args.model
            row["attention_implementation"] = attn_impl
        summary = summarize_model(args.model, pairs, discovery_rows, calibration_rows, propagation, selections, missing, attn_impl)
        write_json(out_dir / f"phase318_{args.model}_summary.json", summary)
        write_jsonl(out_dir / f"phase318_{args.model}_discovery_scan_rows.jsonl", discovery_rows)
        write_jsonl(out_dir / f"phase318_{args.model}_calibration_control_rows.jsonl", calibration_rows)
        write_jsonl(out_dir / f"phase318_{args.model}_propagation_rows.jsonl", propagation)
        write_jsonl(out_dir / f"phase318_{args.model}_source_layer_selection_rows.jsonl", selections)
        write_jsonl(out_dir / f"phase318_{args.model}_missing_rows.jsonl", missing)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        cache.clear()
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def summarize_model(
    model: str,
    pairs: list[dict[str, Any]],
    discovery: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    propagation: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    attn_impl: str,
) -> dict[str, Any]:
    calibration_source = [r for r in calibration if r["condition"] == "source_replace"]
    calibration_unrelated = [r for r in calibration if r["condition"] == "unrelated_replace"]
    by_condition = {
        condition: mean_safe([safe_float(r["donor_transfer_shift"]) for r in calibration if r["condition"] == condition])
        for condition in CALIBRATION_CONDITIONS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if not missing else "complete_with_missing",
        "model": model,
        "attention_implementation": attn_impl,
        "planned_discovery_pairs": len([r for r in pairs if r["split"] == "discovery"]),
        "planned_calibration_pairs": len([r for r in pairs if r["split"] == "calibration"]),
        "discovery_scan_rows": len(discovery),
        "calibration_control_rows": len(calibration),
        "propagation_rows": len(propagation),
        "source_layer_selections": len(selections),
        "missing_pairs": len(missing),
        "calibration_mean_transfer_by_condition": by_condition,
        "calibration_source_donor_win_rate": mean_safe([1.0 if r["patched_donor_wins"] else 0.0 for r in calibration_source]),
        "calibration_source_top1_change_rate": mean_safe([1.0 if r["full_vocab_top1_changed"] else 0.0 for r in calibration_source]),
        "calibration_source_minus_unrelated_mean": round(
            mean_safe([safe_float(r["donor_transfer_shift"]) for r in calibration_source])
            - mean_safe([safe_float(r["donor_transfer_shift"]) for r in calibration_unrelated]),
            6,
        ),
        "query_positive_propagation_rate": mean_safe([1.0 if safe_float(r["donor_direction_projection_fraction"]) > 0 else 0.0 for r in propagation if r["position_role"] == "query"]),
        "last_positive_propagation_rate": mean_safe([1.0 if safe_float(r["donor_direction_projection_fraction"]) > 0 else 0.0 for r in propagation if r["position_role"] == "last"]),
    }


def collect(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    summaries = []
    discovery: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    propagation: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        path = out_dir / f"phase318_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        discovery.extend(read_jsonl(out_dir / f"phase318_{model}_discovery_scan_rows.jsonl"))
        calibration.extend(read_jsonl(out_dir / f"phase318_{model}_calibration_control_rows.jsonl"))
        propagation.extend(read_jsonl(out_dir / f"phase318_{model}_propagation_rows.jsonl"))
        selections.extend(read_jsonl(out_dir / f"phase318_{model}_source_layer_selection_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase318_{model}_missing_rows.jsonl"))
    source = [r for r in calibration if r["condition"] == "source_replace"]
    unrelated = [r for r in calibration if r["condition"] == "unrelated_replace"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(summaries) == len(MODELS) and not missing else "partial_or_missing",
        "model_summaries": summaries,
        "discovery_scan_rows": len(discovery),
        "calibration_control_rows": len(calibration),
        "propagation_rows": len(propagation),
        "source_layer_selections": len(selections),
        "missing_pairs": len(missing),
        "calibration_source_transfer_mean": mean_safe([safe_float(r["donor_transfer_shift"]) for r in source]),
        "calibration_unrelated_transfer_mean": mean_safe([safe_float(r["donor_transfer_shift"]) for r in unrelated]),
        "calibration_control_corrected_transfer": round(
            mean_safe([safe_float(r["donor_transfer_shift"]) for r in source])
            - mean_safe([safe_float(r["donor_transfer_shift"]) for r in unrelated]),
            6,
        ),
        "calibration_donor_win_rate": mean_safe([1.0 if r["patched_donor_wins"] else 0.0 for r in source]),
        "calibration_top1_change_rate": mean_safe([1.0 if r["full_vocab_top1_changed"] else 0.0 for r in source]),
        "evidence_level": "L4_natural_state_intervention",
        "heldout_used_for_selection": False,
    }
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase318_natural_source_state_transfer_summary.json", summary)
        write_jsonl(base / "phase318_discovery_scan_rows.jsonl", discovery)
        write_jsonl(base / "phase318_calibration_control_rows.jsonl", calibration)
        write_jsonl(base / "phase318_source_to_query_last_propagation_rows.jsonl", propagation)
        write_jsonl(base / "phase318_source_layer_selection_rows.jsonl", selections)
        write_jsonl(base / "phase318_missing_rows.jsonl", missing)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        collect(args.round_name)
    elif args.model:
        run_model(args)
    else:
        raise SystemExit("use --model or --summarize")


if __name__ == "__main__":
    main()
