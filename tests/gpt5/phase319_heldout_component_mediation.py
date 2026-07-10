#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase317_natural_source_boundary_case_bank as p317  # noqa: E402
import phase318_natural_source_state_transfer as p318  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
from model_utils import get_layers  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402


PHASE = "Phase319"
SCHEMA_VERSION = "4.2.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "heldout_component_mediation"
PHASE318_ROUND = "natural_source_state_transfer"
OUT = ROOT / "tests/gpt5/result/phase319_heldout_component_mediation"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"
CHANNEL_GROUPS = 16
HELDOUT_CONDITIONS = [
    "baseline",
    "source_replace",
    "source_attention_head_restore",
    "source_mlp_product_restore",
    "source_both_restore",
    "unrelated_replace",
    "wrong_position_replace",
    "feature_permute_replace",
]


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def get_down_proj(layer: Any) -> Any | None:
    mlp = getattr(layer, "mlp", None)
    return getattr(mlp, "down_proj", None) if mlp is not None else None


def downstream_layers(source_layer: int, n_layers: int) -> list[int]:
    values = [source_layer, source_layer + 1, round(0.6 * (n_layers - 1)), round(0.8 * (n_layers - 1)), n_layers - 1]
    return sorted({min(n_layers - 1, max(source_layer, int(value))) for value in values})


def group_ranges(width: int, groups: int = CHANNEL_GROUPS) -> list[tuple[int, int]]:
    return [(round(i * width / groups), round((i + 1) * width / groups)) for i in range(groups)]


def source_patch_handle(model_obj: Any, layer: int, position: int, vector: torch.Tensor, mode: str = "replace") -> Any:
    return get_layers(model_obj)[layer].register_forward_hook(p318.patch_hook(position, vector, mode))


def capture_component_vectors(
    model_obj: Any,
    encoded: dict[str, torch.Tensor],
    positions: dict[str, dict[str, Any]],
    layers: list[int],
    install_extra: Callable[[], list[Any]] | None = None,
) -> tuple[torch.Tensor, dict[tuple[int, str], torch.Tensor], dict[tuple[int, str], torch.Tensor]]:
    attention: dict[tuple[int, str], torch.Tensor] = {}
    products: dict[tuple[int, str], torch.Tensor] = {}
    handles = install_extra() if install_extra else []
    role_positions = {role: int(positions[role]["token_position"]) for role in ["query", "last"]}
    for layer_idx in layers:
        layer = get_layers(model_obj)[layer_idx]
        try:
            o_proj, _heads, _head_dim = head_meta(model_obj, layer_idx)

            def attention_pre(_module: Any, inputs: tuple[Any, ...], layer_idx: int = layer_idx) -> None:
                if not inputs or not torch.is_tensor(inputs[0]):
                    return None
                tensor = inputs[0]
                for role, position in role_positions.items():
                    if position < tensor.shape[1]:
                        attention[(layer_idx, role)] = tensor[0, position].detach().float().cpu()
                return None

            handles.append(o_proj.register_forward_pre_hook(attention_pre))
        except Exception:  # noqa: BLE001
            pass
        down_proj = get_down_proj(layer)
        if down_proj is not None:

            def product_pre(_module: Any, inputs: tuple[Any, ...], layer_idx: int = layer_idx) -> None:
                if not inputs or not torch.is_tensor(inputs[0]):
                    return None
                tensor = inputs[0]
                for role, position in role_positions.items():
                    if position < tensor.shape[1]:
                        products[(layer_idx, role)] = tensor[0, position].detach().float().cpu()
                return None

            handles.append(down_proj.register_forward_pre_hook(product_pre))
    try:
        with torch.inference_mode():
            output = model_obj(**encoded, use_cache=False, return_dict=True)
        last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
        logits = output.logits[0, last_pos].detach().float().cpu()
        return logits, attention, products
    finally:
        for handle in handles:
            handle.remove()


def discovery_component_rows(
    model: str,
    pair: dict[str, Any],
    source_layer: int,
    baseline_attn: dict[tuple[int, str], torch.Tensor],
    patched_attn: dict[tuple[int, str], torch.Tensor],
    baseline_product: dict[tuple[int, str], torch.Tensor],
    patched_product: dict[tuple[int, str], torch.Tensor],
    model_obj: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (layer, role), base in baseline_attn.items():
        patched = patched_attn.get((layer, role))
        if patched is None:
            continue
        _o_proj, n_heads, head_dim = head_meta(model_obj, layer)
        if base.numel() != n_heads * head_dim:
            continue
        base_heads = base.view(n_heads, head_dim)
        patched_heads = patched.view(n_heads, head_dim)
        for head in range(n_heads):
            delta = patched_heads[head] - base_heads[head]
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
                    "source_layer": source_layer,
                    "component_type": "attention_head_input",
                    "component_layer": layer,
                    "position_role": role,
                    "component_index": head,
                    "component_start": head * head_dim,
                    "component_end": (head + 1) * head_dim,
                    "natural_delta_norm": round(float(torch.linalg.vector_norm(delta).item()), 6),
                    "relative_delta_norm": round(float(torch.linalg.vector_norm(delta).item() / max(1e-8, torch.linalg.vector_norm(base_heads[head]).item())), 6),
                }
            )
    for (layer, role), base in baseline_product.items():
        patched = patched_product.get((layer, role))
        if patched is None:
            continue
        for group, (start, end) in enumerate(group_ranges(base.numel())):
            delta = patched[start:end] - base[start:end]
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
                    "source_layer": source_layer,
                    "component_type": "mlp_product_group",
                    "component_layer": layer,
                    "position_role": role,
                    "component_index": group,
                    "component_start": start,
                    "component_end": end,
                    "natural_delta_norm": round(float(torch.linalg.vector_norm(delta).item()), 6),
                    "relative_delta_norm": round(float(torch.linalg.vector_norm(delta).item() / max(1e-8, torch.linalg.vector_norm(base[start:end]).item())), 6),
                }
            )
    return rows


def select_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str, int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["family_id"]),
            str(row["mechanism_id"]),
            str(row["component_type"]),
            int(row["component_layer"]),
            str(row["position_role"]),
            int(row["component_index"]),
            int(row["component_start"]),
            int(row["component_end"]),
        )
        grouped[key].append(safe_float(row["relative_delta_norm"]))
    selections: list[dict[str, Any]] = []
    mechanisms = sorted({(str(r["family_id"]), str(r["mechanism_id"])) for r in rows})
    for family, mechanism in mechanisms:
        for component_type in ["attention_head_input", "mlp_product_group"]:
            candidates = []
            for key, values in grouped.items():
                if key[0] == family and key[1] == mechanism and key[2] == component_type:
                    candidates.append((mean(values), key))
            if not candidates:
                continue
            score, key = max(candidates)
            selections.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "family_id": family,
                    "mechanism_id": mechanism,
                    "component_type": component_type,
                    "component_layer": key[3],
                    "position_role": key[4],
                    "component_index": key[5],
                    "component_start": key[6],
                    "component_end": key[7],
                    "discovery_mean_relative_delta_norm": round(score, 6),
                    "selection_split": "discovery_only",
                }
            )
    return selections


def restore_attention_handle(model_obj: Any, spec: dict[str, Any], baseline: torch.Tensor) -> Any:
    o_proj, n_heads, head_dim = head_meta(model_obj, int(spec["component_layer"]))
    role_position = int(spec["token_position"])
    head = int(spec["component_index"])
    baseline_vec = baseline.detach().float().cpu().view(n_heads, head_dim)[head]

    def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not inputs or not torch.is_tensor(inputs[0]) or role_position >= inputs[0].shape[1]:
            return None
        value = inputs[0].clone()
        view = value.view(value.shape[0], value.shape[1], n_heads, head_dim)
        view[0, role_position, head, :] = baseline_vec.to(device=value.device, dtype=value.dtype)
        return (value,) + tuple(inputs[1:])

    return o_proj.register_forward_pre_hook(pre_hook)


def restore_product_handle(model_obj: Any, spec: dict[str, Any], baseline: torch.Tensor) -> Any:
    down_proj = get_down_proj(get_layers(model_obj)[int(spec["component_layer"])])
    if down_proj is None:
        raise RuntimeError("selected MLP has no down_proj")
    role_position = int(spec["token_position"])
    start, end = int(spec["component_start"]), int(spec["component_end"])
    baseline_vec = baseline[start:end].detach().float().cpu()

    def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not inputs or not torch.is_tensor(inputs[0]) or role_position >= inputs[0].shape[1]:
            return None
        value = inputs[0].clone()
        value[0, role_position, start:end] = baseline_vec.to(device=value.device, dtype=value.dtype)
        return (value,) + tuple(inputs[1:])

    return down_proj.register_forward_pre_hook(pre_hook)


def forward_condition(
    model_obj: Any,
    encoded: dict[str, torch.Tensor],
    source_layer: int,
    source_position: int,
    source_vector: torch.Tensor | None,
    attention_spec: dict[str, Any] | None = None,
    attention_baseline: torch.Tensor | None = None,
    product_spec: dict[str, Any] | None = None,
    product_baseline: torch.Tensor | None = None,
    source_mode: str = "replace",
) -> torch.Tensor:
    handles: list[Any] = []
    if source_vector is not None:
        handles.append(source_patch_handle(model_obj, source_layer, source_position, source_vector, source_mode))
    if attention_spec is not None and attention_baseline is not None:
        handles.append(restore_attention_handle(model_obj, attention_spec, attention_baseline))
    if product_spec is not None and product_baseline is not None:
        handles.append(restore_product_handle(model_obj, product_spec, product_baseline))
    try:
        with torch.inference_mode():
            output = model_obj(**encoded, use_cache=False, return_dict=True)
        last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
        return output.logits[0, last_pos].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def phrase_logprob(
    model_obj: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    phrase: str,
    source_layer: int,
    source_position: int,
    source_vector: torch.Tensor | None,
) -> float:
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    phrase_ids = tokenizer.encode(" " + phrase.strip(), add_special_tokens=False)
    if not phrase_ids:
        return float("-inf")
    ids = torch.tensor([prefix_ids + phrase_ids], device=device)
    attention_mask = torch.ones_like(ids)
    handles = []
    if source_vector is not None:
        handles.append(source_patch_handle(model_obj, source_layer, source_position, source_vector))
    try:
        with torch.inference_mode():
            output = model_obj(input_ids=ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = output.logits[0]
        total = 0.0
        for index, token_id in enumerate(phrase_ids):
            logit_position = len(prefix_ids) - 1 + index
            total += float(torch.log_softmax(logits[logit_position].float(), dim=-1)[int(token_id)].item())
        return round(total, 6)
    finally:
        for handle in handles:
            handle.remove()


def generate_text(
    model_obj: Any,
    tokenizer: Any,
    encoded: dict[str, torch.Tensor],
    source_layer: int,
    source_position: int,
    source_vector: torch.Tensor | None,
    max_new_tokens: int,
) -> str:
    handles = []
    if source_vector is not None:
        handles.append(source_patch_handle(model_obj, source_layer, source_position, source_vector))
    input_len = int(encoded["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            output = model_obj.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0, input_len:].detach().cpu().tolist(), skip_special_tokens=False)
    finally:
        for handle in handles:
            handle.remove()


def condition_row(
    model: str,
    pair: dict[str, Any],
    condition: str,
    source_layer: int,
    base_logits: torch.Tensor,
    logits: torch.Tensor,
    donor: dict[str, Any],
    recipient: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    row = p318.transfer_row(model, pair, source_layer, condition, base_logits, logits, donor, recipient, tokenizer, "source")
    row["phase_id"] = PHASE
    row["schema_version"] = SCHEMA_VERSION
    return row


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    case_map, pairs = p318.get_case_maps()
    selection_path = ROOT / f"tests/gpt5/result/phase318_natural_source_state_transfer/{PHASE318_ROUND}/phase318_{args.model}_source_layer_selection_rows.jsonl"
    source_selections = read_jsonl(selection_path)
    if not source_selections:
        raise FileNotFoundError(f"run Phase318 first: {selection_path}")
    source_map = {(str(r["family_id"]), str(r["mechanism_id"])): int(r["selected_source_layer"]) for r in source_selections}
    model_obj = tokenizer = None
    component_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    mediation_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model_obj))
        discovery_pairs = [r for r in pairs if r["split"] == "discovery"]
        if args.limit:
            discovery_pairs = discovery_pairs[: args.limit]
        for index, pair in enumerate(discovery_pairs, 1):
            try:
                recipient = case_map[str(pair["recipient_case_id"])]
                donor = case_map[str(pair["donor_case_id"])]
                source_layer = source_map[(str(pair["family_id"]), str(pair["mechanism_id"]))]
                rec_encoded, rec_positions = p318.encode_case(tokenizer, recipient, device)
                donor_encoded, donor_positions = p318.encode_case(tokenizer, donor, device)
                _, donor_states = p318.forward_states(model_obj, donor_encoded, donor_positions)
                source_position = int(rec_positions["source"]["token_position"])
                layers = downstream_layers(source_layer, n_layers)
                _, base_attn, base_product = capture_component_vectors(model_obj, rec_encoded, rec_positions, layers)
                _, patched_attn, patched_product = capture_component_vectors(
                    model_obj,
                    rec_encoded,
                    rec_positions,
                    layers,
                    install_extra=lambda: [source_patch_handle(model_obj, source_layer, source_position, donor_states["source"][source_layer])],
                )
                component_rows.extend(discovery_component_rows(args.model, pair, source_layer, base_attn, patched_attn, base_product, patched_product, model_obj))
                print(f"{args.model}: discovery component response {index}/{len(discovery_pairs)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                missing.append({"phase_id": PHASE, "model": args.model, "pair_id": pair["pair_id"], "split": "discovery", "reason": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        selections = select_components(component_rows)
        for row in selections:
            row["model"] = args.model
            row["attention_implementation"] = attn_impl
        component_map: dict[tuple[str, str, str], dict[str, Any]] = {
            (str(r["family_id"]), str(r["mechanism_id"]), str(r["component_type"])): r for r in selections
        }
        heldout_pairs = [r for r in pairs if r["split"] == "heldout"]
        if args.limit:
            heldout_pairs = heldout_pairs[: args.limit]
        for index, pair in enumerate(heldout_pairs, 1):
            try:
                recipient = case_map[str(pair["recipient_case_id"])]
                donor = case_map[str(pair["donor_case_id"])]
                unrelated = case_map[str(pair["unrelated_control_case_id"])]
                family_mechanism = (str(pair["family_id"]), str(pair["mechanism_id"]))
                source_layer = source_map[family_mechanism]
                attention_spec = dict(component_map[(family_mechanism[0], family_mechanism[1], "attention_head_input")])
                product_spec = dict(component_map[(family_mechanism[0], family_mechanism[1], "mlp_product_group")])
                rec_encoded, rec_positions = p318.encode_case(tokenizer, recipient, device)
                donor_encoded, donor_positions = p318.encode_case(tokenizer, donor, device)
                unrelated_encoded, unrelated_positions = p318.encode_case(tokenizer, unrelated, device)
                _, donor_states = p318.forward_states(model_obj, donor_encoded, donor_positions)
                _, unrelated_states = p318.forward_states(model_obj, unrelated_encoded, unrelated_positions)
                source_position = int(rec_positions["source"]["token_position"])
                query_position = int(rec_positions["query"]["token_position"])
                attention_spec["token_position"] = int(rec_positions[str(attention_spec["position_role"])]["token_position"])
                product_spec["token_position"] = int(rec_positions[str(product_spec["position_role"])]["token_position"])
                selected_layers = sorted({int(attention_spec["component_layer"]), int(product_spec["component_layer"])})
                base_logits, base_attn, base_product = capture_component_vectors(model_obj, rec_encoded, rec_positions, selected_layers)
                source_vector = donor_states["source"][source_layer]
                source_logits, source_attn, source_product = capture_component_vectors(
                    model_obj,
                    rec_encoded,
                    rec_positions,
                    selected_layers,
                    install_extra=lambda: [source_patch_handle(model_obj, source_layer, source_position, source_vector)],
                )
                attn_key = (int(attention_spec["component_layer"]), str(attention_spec["position_role"]))
                product_key = (int(product_spec["component_layer"]), str(product_spec["position_role"]))
                attention_baseline = base_attn[attn_key]
                product_baseline = base_product[product_key]
                conditions = {
                    "baseline": base_logits,
                    "source_replace": source_logits,
                    "source_attention_head_restore": forward_condition(model_obj, rec_encoded, source_layer, source_position, source_vector, attention_spec, attention_baseline),
                    "source_mlp_product_restore": forward_condition(model_obj, rec_encoded, source_layer, source_position, source_vector, product_spec=product_spec, product_baseline=product_baseline),
                    "source_both_restore": forward_condition(model_obj, rec_encoded, source_layer, source_position, source_vector, attention_spec, attention_baseline, product_spec, product_baseline),
                    "unrelated_replace": forward_condition(model_obj, rec_encoded, source_layer, source_position, unrelated_states["source"][source_layer]),
                    "wrong_position_replace": forward_condition(model_obj, rec_encoded, source_layer, query_position, source_vector),
                    "feature_permute_replace": forward_condition(model_obj, rec_encoded, source_layer, source_position, source_vector, source_mode="permute"),
                }
                rows_by_condition = {}
                for condition, logits in conditions.items():
                    row = condition_row(args.model, pair, condition, source_layer, base_logits, logits, donor, recipient, tokenizer)
                    heldout_rows.append(row)
                    rows_by_condition[condition] = row
                source_shift = safe_float(rows_by_condition["source_replace"]["donor_transfer_shift"])
                attn_shift = safe_float(rows_by_condition["source_attention_head_restore"]["donor_transfer_shift"])
                mlp_shift = safe_float(rows_by_condition["source_mlp_product_restore"]["donor_transfer_shift"])
                both_shift = safe_float(rows_by_condition["source_both_restore"]["donor_transfer_shift"])
                unrelated_shift = safe_float(rows_by_condition["unrelated_replace"]["donor_transfer_shift"])
                wrong_shift = safe_float(rows_by_condition["wrong_position_replace"]["donor_transfer_shift"])
                attn_loss = source_shift - attn_shift
                mlp_loss = source_shift - mlp_shift
                both_loss = source_shift - both_shift
                interaction = both_loss - attn_loss - mlp_loss
                corrected = source_shift - max(unrelated_shift, wrong_shift)
                l5_candidate = (
                    source_shift > 0.5
                    and corrected > 0.5
                    and bool(rows_by_condition["source_replace"]["patched_donor_wins"])
                    and both_loss > max(0.5, 0.2 * abs(source_shift))
                )
                mediation_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": args.model,
                        "pair_id": pair["pair_id"],
                        "family_id": pair["family_id"],
                        "mechanism_id": pair["mechanism_id"],
                        "split": pair["split"],
                        "source_layer": source_layer,
                        "source_transfer_shift": round(source_shift, 6),
                        "unrelated_transfer_shift": round(unrelated_shift, 6),
                        "wrong_position_transfer_shift": round(wrong_shift, 6),
                        "control_corrected_transfer": round(corrected, 6),
                        "attention_head_mediation_loss": round(attn_loss, 6),
                        "mlp_product_mediation_loss": round(mlp_loss, 6),
                        "joint_mediation_loss": round(both_loss, 6),
                        "mediation_interaction": round(interaction, 6),
                        "joint_mediation_fraction": round(both_loss / max(1e-8, abs(source_shift)), 6),
                        "attention_spec": attention_spec,
                        "mlp_product_spec": product_spec,
                        "donor_winner_after_source_replace": rows_by_condition["source_replace"]["patched_donor_wins"],
                        "l5_candidate": l5_candidate,
                        "evidence_level": "L5_candidate" if l5_candidate else "L4_natural_state_intervention",
                    }
                )
                attn_delta = source_attn[attn_key] - attention_baseline
                product_delta = source_product[product_key] - product_baseline
                start, end = int(product_spec["component_start"]), int(product_spec["component_end"])
                event_rows.extend(
                    [
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE,
                            "model": args.model,
                            "pair_id": pair["pair_id"],
                            "family_id": pair["family_id"],
                            "mechanism_id": pair["mechanism_id"],
                            "event": "selected_attention_input_natural_response",
                            "layer": attention_spec["component_layer"],
                            "position_role": attention_spec["position_role"],
                            "delta_norm": round(float(torch.linalg.vector_norm(attn_delta).item()), 6),
                        },
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE,
                            "model": args.model,
                            "pair_id": pair["pair_id"],
                            "family_id": pair["family_id"],
                            "mechanism_id": pair["mechanism_id"],
                            "event": "selected_mlp_product_natural_response",
                            "layer": product_spec["component_layer"],
                            "position_role": product_spec["position_role"],
                            "delta_norm": round(float(torch.linalg.vector_norm(product_delta[start:end]).item()), 6),
                        },
                    ]
                )
                base_donor_phrase = phrase_logprob(model_obj, tokenizer, device, recipient["prompt"], donor["target"], source_layer, source_position, None)
                base_recipient_phrase = phrase_logprob(model_obj, tokenizer, device, recipient["prompt"], recipient["target"], source_layer, source_position, None)
                patch_donor_phrase = phrase_logprob(model_obj, tokenizer, device, recipient["prompt"], donor["target"], source_layer, source_position, source_vector)
                patch_recipient_phrase = phrase_logprob(model_obj, tokenizer, device, recipient["prompt"], recipient["target"], source_layer, source_position, source_vector)
                base_text = generate_text(model_obj, tokenizer, rec_encoded, source_layer, source_position, None, args.rollout_tokens)
                patched_text = generate_text(model_obj, tokenizer, rec_encoded, source_layer, source_position, source_vector, args.rollout_tokens)
                rollout_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": args.model,
                        "pair_id": pair["pair_id"],
                        "family_id": pair["family_id"],
                        "mechanism_id": pair["mechanism_id"],
                        "recipient_target": recipient["target"],
                        "donor_target": donor["target"],
                        "base_donor_minus_recipient_phrase_logprob": round(base_donor_phrase - base_recipient_phrase, 6),
                        "patched_donor_minus_recipient_phrase_logprob": round(patch_donor_phrase - patch_recipient_phrase, 6),
                        "phrase_transfer_shift": round((patch_donor_phrase - patch_recipient_phrase) - (base_donor_phrase - base_recipient_phrase), 6),
                        "baseline_rollout": base_text,
                        "source_patched_rollout": patched_text,
                        "rollout_changed": base_text != patched_text,
                        "patched_starts_with_donor": patched_text.strip().lower().startswith(str(donor["target"]).lower()),
                        "patched_starts_with_recipient": patched_text.strip().lower().startswith(str(recipient["target"]).lower()),
                    }
                )
                print(f"{args.model}: heldout mediation {index}/{len(heldout_pairs)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                missing.append({"phase_id": PHASE, "model": args.model, "pair_id": pair["pair_id"], "split": "heldout", "reason": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out_dir = OUT / args.round_name
        summary = summarize_model(args.model, component_rows, selections, heldout_rows, mediation_rows, event_rows, rollout_rows, missing, attn_impl)
        write_json(out_dir / f"phase319_{args.model}_summary.json", summary)
        write_jsonl(out_dir / f"phase319_{args.model}_discovery_component_rows.jsonl", component_rows)
        write_jsonl(out_dir / f"phase319_{args.model}_component_selection_rows.jsonl", selections)
        write_jsonl(out_dir / f"phase319_{args.model}_heldout_condition_rows.jsonl", heldout_rows)
        write_jsonl(out_dir / f"phase319_{args.model}_heldout_mediation_rows.jsonl", mediation_rows)
        write_jsonl(out_dir / f"phase319_{args.model}_natural_component_event_rows.jsonl", event_rows)
        write_jsonl(out_dir / f"phase319_{args.model}_phrase_rollout_rows.jsonl", rollout_rows)
        write_jsonl(out_dir / f"phase319_{args.model}_missing_rows.jsonl", missing)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def summarize_model(
    model: str,
    component_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    mediation: list[dict[str, Any]],
    events: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    attn_impl: str,
) -> dict[str, Any]:
    source = [r for r in conditions if r["condition"] == "source_replace"]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if not missing else "complete_with_missing",
        "model": model,
        "attention_implementation": attn_impl,
        "discovery_component_rows": len(component_rows),
        "component_selections": len(selections),
        "heldout_condition_rows": len(conditions),
        "heldout_mediation_cases": len(mediation),
        "natural_component_event_rows": len(events),
        "phrase_rollout_cases": len(rollouts),
        "missing_cases": len(missing),
        "heldout_source_transfer_mean": mean_safe([safe_float(r["donor_transfer_shift"]) for r in source]),
        "heldout_source_donor_win_rate": mean_safe([1.0 if r["patched_donor_wins"] else 0.0 for r in source]),
        "heldout_control_corrected_transfer_mean": mean_safe([safe_float(r["control_corrected_transfer"]) for r in mediation]),
        "heldout_joint_mediation_loss_mean": mean_safe([safe_float(r["joint_mediation_loss"]) for r in mediation]),
        "heldout_joint_mediation_fraction_mean": mean_safe([safe_float(r["joint_mediation_fraction"]) for r in mediation]),
        "l5_candidate_count": sum(1 for r in mediation if r["l5_candidate"]),
        "rollout_change_rate": mean_safe([1.0 if r["rollout_changed"] else 0.0 for r in rollouts]),
        "patched_donor_start_rate": mean_safe([1.0 if r["patched_starts_with_donor"] else 0.0 for r in rollouts]),
        "phrase_transfer_shift_mean": mean_safe([safe_float(r["phrase_transfer_shift"]) for r in rollouts]),
    }


def collect(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    summaries = []
    component_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    mediation: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        path = out_dir / f"phase319_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        component_rows.extend(read_jsonl(out_dir / f"phase319_{model}_discovery_component_rows.jsonl"))
        selections.extend(read_jsonl(out_dir / f"phase319_{model}_component_selection_rows.jsonl"))
        conditions.extend(read_jsonl(out_dir / f"phase319_{model}_heldout_condition_rows.jsonl"))
        mediation.extend(read_jsonl(out_dir / f"phase319_{model}_heldout_mediation_rows.jsonl"))
        events.extend(read_jsonl(out_dir / f"phase319_{model}_natural_component_event_rows.jsonl"))
        rollouts.extend(read_jsonl(out_dir / f"phase319_{model}_phrase_rollout_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase319_{model}_missing_rows.jsonl"))
    source = [r for r in conditions if r["condition"] == "source_replace"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(summaries) == len(MODELS) and not missing else "partial_or_missing",
        "model_summaries": summaries,
        "discovery_component_rows": len(component_rows),
        "component_selections": len(selections),
        "heldout_condition_rows": len(conditions),
        "heldout_mediation_cases": len(mediation),
        "natural_component_event_rows": len(events),
        "phrase_rollout_cases": len(rollouts),
        "missing_cases": len(missing),
        "heldout_source_transfer_mean": mean_safe([safe_float(r["donor_transfer_shift"]) for r in source]),
        "heldout_control_corrected_transfer_mean": mean_safe([safe_float(r["control_corrected_transfer"]) for r in mediation]),
        "heldout_donor_win_rate": mean_safe([1.0 if r["patched_donor_wins"] else 0.0 for r in source]),
        "joint_mediation_loss_mean": mean_safe([safe_float(r["joint_mediation_loss"]) for r in mediation]),
        "joint_mediation_fraction_mean": mean_safe([safe_float(r["joint_mediation_fraction"]) for r in mediation]),
        "l5_candidate_count": sum(1 for r in mediation if r["l5_candidate"]),
        "l5_candidate_rate": mean_safe([1.0 if r["l5_candidate"] else 0.0 for r in mediation]),
        "phrase_transfer_shift_mean": mean_safe([safe_float(r["phrase_transfer_shift"]) for r in rollouts]),
        "rollout_change_rate": mean_safe([1.0 if r["rollout_changed"] else 0.0 for r in rollouts]),
        "patched_donor_start_rate": mean_safe([1.0 if r["patched_starts_with_donor"] else 0.0 for r in rollouts]),
        "heldout_used_for_selection": False,
        "evidence_note": "L5_candidate is an individual screening label; aggregate replication is required before any edge is promoted to L5.",
    }
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase319_heldout_component_mediation_summary.json", summary)
        write_jsonl(base / "phase319_discovery_component_rows.jsonl", component_rows)
        write_jsonl(base / "phase319_component_selection_rows.jsonl", selections)
        write_jsonl(base / "phase319_heldout_condition_rows.jsonl", conditions)
        write_jsonl(base / "phase319_heldout_mediation_rows.jsonl", mediation)
        write_jsonl(base / "phase319_natural_component_event_rows.jsonl", events)
        write_jsonl(base / "phase319_phrase_rollout_rows.jsonl", rollouts)
        write_jsonl(base / "phase319_missing_rows.jsonl", missing)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--rollout-tokens", type=int, default=8)
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
