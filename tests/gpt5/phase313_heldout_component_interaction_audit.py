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
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase305_internal_semantic_physical_path_probe as p305  # noqa: E402
import phase311_core_language_physical_atlas as p311  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
from model_utils import get_layers  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_mlp  # noqa: E402


PHASE = "Phase313"
SCHEMA_VERSION = "3.2.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "heldout_component_interaction_audit"
SOURCE = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/gpt5/result/phase313_heldout_component_interaction_audit"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"
CONDITIONS = ["baseline", "attention_half", "mlp_half", "attention_mlp_half", "attention_permute", "mlp_permute"]


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


def replace_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple) and output:
        return (tensor, *output[1:])
    return output


def patch_hook(position: int, mode: str):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        y = extract_tensor(output)
        if y is None or not torch.is_tensor(y):
            return output
        patched = y.clone()
        if patched.ndim == 3:
            if position >= patched.shape[1]:
                return output
            vec = patched[:, position, :]
            if mode == "half":
                patched[:, position, :] = vec * 0.5
            elif mode == "permute":
                patched[:, position, :] = torch.roll(vec, shifts=max(1, vec.shape[-1] // 7), dims=-1)
        elif patched.ndim == 2:
            pos = position if position < patched.shape[0] else patched.shape[0] - 1
            vec = patched[pos, :]
            if mode == "half":
                patched[pos, :] = vec * 0.5
            elif mode == "permute":
                patched[pos, :] = torch.roll(vec, shifts=max(1, vec.shape[-1] // 7), dims=-1)
        return replace_tensor(output, patched)

    return hook


def install_hooks(model_obj: Any, layer_idx: int, position: int, condition: str) -> list[Any]:
    layer = get_layers(model_obj)[layer_idx]
    handles = []
    if condition in {"attention_half", "attention_mlp_half", "attention_permute"}:
        attn = get_attn(layer)
        if attn is None:
            raise ValueError(f"no attention module at layer {layer_idx}")
        mode = "permute" if condition == "attention_permute" else "half"
        handles.append(attn.register_forward_hook(patch_hook(position, mode)))
    if condition in {"mlp_half", "attention_mlp_half", "mlp_permute"}:
        mlp = get_mlp(layer)
        if mlp is None:
            raise ValueError(f"no MLP module at layer {layer_idx}")
        mode = "permute" if condition == "mlp_permute" else "half"
        handles.append(mlp.register_forward_hook(patch_hook(position, mode)))
    return handles


def forward_logits(model_obj: Any, tokenizer: Any, device: torch.device, case: dict[str, Any], layer_idx: int, position: int, condition: str) -> tuple[torch.Tensor, int, int]:
    handles = [] if condition == "baseline" else install_hooks(model_obj, layer_idx, position, condition)
    encoded = tokenizer(case["prompt"], return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            output = model_obj(**encoded, use_cache=False, return_dict=True)
        logits = output.logits[0, last_pos].detach().float().cpu()
        return logits, int(torch.argmax(logits).item()), last_pos
    finally:
        for handle in handles:
            handle.remove()


def generate_text(model_obj: Any, tokenizer: Any, device: torch.device, case: dict[str, Any], layer_idx: int, position: int, condition: str, max_new_tokens: int) -> str:
    handles = [] if condition == "baseline" else install_hooks(model_obj, layer_idx, position, condition)
    encoded = tokenizer(case["prompt"], return_tensors="pt", truncation=True, max_length=1536).to(device)
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


def candidate_specs(model: str, cases: list[dict[str, Any]], component_rows: list[dict[str, Any]], per_family: int) -> list[dict[str, Any]]:
    train_ids = {str(r["case_id"]) for r in cases if r["model"] == model and r["split"] in {"discovery", "calibration"}}
    heldout = {(str(r["family_id"]), str(r["mechanism_id"])): r for r in cases if r["model"] == model and r["split"] == "heldout"}
    buckets: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in component_rows:
        if row["model"] != model or str(row["case_id"]) not in train_ids:
            continue
        family = str(row["family_id"])
        mechanism = str(row["mechanism_id"])
        position = str(row["position_role"])
        layer = int(row["layer_index"])
        strength = abs(safe_float(row.get("delta_attn_semantic_margin"))) + abs(safe_float(row.get("delta_mlp_semantic_margin")))
        buckets[(family, mechanism, position, layer)].append(strength)
    mechanism_best: dict[tuple[str, str], tuple[float, str, int]] = {}
    for (family, mechanism, position, layer), vals in buckets.items():
        score = mean(vals)
        old = mechanism_best.get((family, mechanism))
        if old is None or score > old[0]:
            mechanism_best[(family, mechanism)] = (score, position, layer)
    selected = []
    for family in sorted({key[0] for key in mechanism_best}):
        candidates = sorted(
            [(score, mechanism, position, layer) for (fam, mechanism), (score, position, layer) in mechanism_best.items() if fam == family],
            reverse=True,
        )[:per_family]
        for score, mechanism, position, layer in candidates:
            case = heldout.get((family, mechanism))
            if case:
                selected.append(
                    {
                        "model": model,
                        "family_id": family,
                        "mechanism_id": mechanism,
                        "case": case,
                        "selected_position_role": position,
                        "selected_layer": layer,
                        "train_selection_strength": round(score, 6),
                        "selection_data": "discovery_and_calibration_only",
                        "audit_data": "heldout_only",
                    }
                )
    return selected


def score_condition(tokenizer: Any, logits: torch.Tensor, case: dict[str, Any]) -> dict[str, Any]:
    target_ids, distractor_ids, _targets, _distractors = p311.semantic_groups(tokenizer, case)
    return p305.semantic_readout(logits, target_ids, distractor_ids)


def audit_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT / args.round_name
    all_cases = read_jsonl(SOURCE / "phase311_core_language_case_result_rows.jsonl")
    component_rows = read_jsonl(SOURCE / "phase311_core_language_component_rows.jsonl")
    selected = candidate_specs(args.model, all_cases, component_rows, args.candidates_per_family)
    audit_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model_obj = tokenizer = None
    try:
        model_obj, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for index, spec in enumerate(selected, 1):
            case = spec["case"]
            try:
                encoded = tokenizer(case["prompt"], return_tensors="pt", truncation=True, max_length=1536)
                last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
                positions = p311.locate_positions(tokenizer, case, case["prompt"], last_pos)
                position = int(positions[spec["selected_position_role"]]["token_position"])
                condition_scores: dict[str, dict[str, Any]] = {}
                top1_ids: dict[str, int] = {}
                for condition in CONDITIONS:
                    logits, top1_id, _ = forward_logits(model_obj, tokenizer, device, case, int(spec["selected_layer"]), position, condition)
                    score = score_condition(tokenizer, logits, case)
                    condition_scores[condition] = score
                    top1_ids[condition] = top1_id
                    audit_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE,
                            "created_at": now(),
                            "audit_id": f"phase313:audit:{args.model}:{case['case_id']}:{condition}",
                            "model": args.model,
                            "case_id": case["case_id"],
                            "family_id": case["family_id"],
                            "mechanism_id": case["mechanism_id"],
                            "split": case["split"],
                            "condition": condition,
                            "selected_layer": spec["selected_layer"],
                            "selected_position_role": spec["selected_position_role"],
                            "selected_token_position": position,
                            "train_selection_strength": spec["train_selection_strength"],
                            "selection_data": spec["selection_data"],
                            "audit_data": spec["audit_data"],
                            "attention_implementation": attn_impl,
                            "semantic_margin": round(safe_float(score["semantic_margin"]), 6),
                            "semantic_winner": score["semantic_winner"],
                            "target_semantic_logit": round(safe_float(score["target_semantic_logit"]), 6),
                            "distractor_semantic_logit": round(safe_float(score["distractor_semantic_logit"]), 6),
                            "full_vocab_top1_token_id": top1_id,
                        }
                    )
                base_margin = safe_float(condition_scores["baseline"]["semantic_margin"])
                da = safe_float(condition_scores["attention_half"]["semantic_margin"]) - base_margin
                dm = safe_float(condition_scores["mlp_half"]["semantic_margin"]) - base_margin
                dab = safe_float(condition_scores["attention_mlp_half"]["semantic_margin"]) - base_margin
                interaction = dab - da - dm
                winner_base = condition_scores["baseline"]["semantic_winner"]
                winner_both = condition_scores["attention_mlp_half"]["semantic_winner"]
                interaction_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "interaction_id": f"phase313:interaction:{args.model}:{case['case_id']}",
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case["family_id"],
                        "mechanism_id": case["mechanism_id"],
                        "selected_layer": spec["selected_layer"],
                        "selected_position_role": spec["selected_position_role"],
                        "base_semantic_margin": round(base_margin, 6),
                        "delta_attention_half": round(da, 6),
                        "delta_mlp_half": round(dm, 6),
                        "delta_attention_mlp_half": round(dab, 6),
                        "interaction_value": round(interaction, 6),
                        "nonlinear_interaction_abs_gt_1": abs(interaction) > 1.0,
                        "base_winner": winner_base,
                        "both_half_winner": winner_both,
                        "winner_changed": winner_base != winner_both,
                        "attention_permute_delta": round(safe_float(condition_scores["attention_permute"]["semantic_margin"]) - base_margin, 6),
                        "mlp_permute_delta": round(safe_float(condition_scores["mlp_permute"]["semantic_margin"]) - base_margin, 6),
                        "full_vocab_top1_changed": top1_ids["baseline"] != top1_ids["attention_mlp_half"],
                        "evidence_level": "L5_candidate" if winner_base == "target" and winner_both != "target" else "L4_intervention_effect",
                    }
                )
                base_text = generate_text(model_obj, tokenizer, device, case, int(spec["selected_layer"]), position, "baseline", args.rollout_tokens)
                both_text = generate_text(model_obj, tokenizer, device, case, int(spec["selected_layer"]), position, "attention_mlp_half", args.rollout_tokens)
                rollout_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "rollout_id": f"phase313:rollout:{args.model}:{case['case_id']}",
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case["family_id"],
                        "mechanism_id": case["mechanism_id"],
                        "base_text": base_text[:400],
                        "attention_mlp_half_text": both_text[:400],
                        "rollout_changed": base_text != both_text,
                        "rollout_tokens": args.rollout_tokens,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": args.model,
                        "case_id": case.get("case_id"),
                        "reason": repr(exc),
                    }
                )
            print(f"{args.model}: heldout interaction audit {index}/{len(selected)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(interaction_rows) + len(missing_rows) == len(selected) else "partial",
        "model": args.model,
        "selected_heldout_cases": len(selected),
        "audit_rows": len(audit_rows),
        "interaction_rows": len(interaction_rows),
        "rollout_rows": len(rollout_rows),
        "missing_rows": len(missing_rows),
        "family_counts": dict(Counter(str(r["family_id"]) for r in interaction_rows)),
        "winner_changed_count": sum(1 for r in interaction_rows if r["winner_changed"]),
        "nonlinear_interaction_count": sum(1 for r in interaction_rows if r["nonlinear_interaction_abs_gt_1"]),
        "full_vocab_top1_changed_count": sum(1 for r in interaction_rows if r["full_vocab_top1_changed"]),
        "mean_interaction_value": mean_safe([safe_float(r["interaction_value"]) for r in interaction_rows]),
        "mean_abs_interaction_value": mean_safe([abs(safe_float(r["interaction_value"])) for r in interaction_rows]),
        "rollout_changed_count": sum(1 for r in rollout_rows if r["rollout_changed"]),
    }
    write_json(out_dir / f"phase313_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase313_{args.model}_audit_rows.jsonl", audit_rows)
    write_jsonl(out_dir / f"phase313_{args.model}_interaction_rows.jsonl", interaction_rows)
    write_jsonl(out_dir / f"phase313_{args.model}_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase313_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def collect(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    summaries = []
    audit: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        path = out_dir / f"phase313_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        audit.extend(read_jsonl(out_dir / f"phase313_{model}_audit_rows.jsonl"))
        interactions.extend(read_jsonl(out_dir / f"phase313_{model}_interaction_rows.jsonl"))
        rollouts.extend(read_jsonl(out_dir / f"phase313_{model}_rollout_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase313_{model}_missing_rows.jsonl"))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_family[str(row["family_id"])].append(row)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(summaries) == len(MODELS) and not missing else "partial",
        "model_summaries": summaries,
        "selected_heldout_cases": len(interactions),
        "audit_rows": len(audit),
        "interaction_rows": len(interactions),
        "rollout_rows": len(rollouts),
        "missing_rows": len(missing),
        "winner_changed_count": sum(1 for r in interactions if r["winner_changed"]),
        "nonlinear_interaction_count": sum(1 for r in interactions if r["nonlinear_interaction_abs_gt_1"]),
        "full_vocab_top1_changed_count": sum(1 for r in interactions if r["full_vocab_top1_changed"]),
        "mean_abs_interaction_value": mean_safe([abs(safe_float(r["interaction_value"])) for r in interactions]),
        "family_mean_abs_interaction": {
            family: mean_safe([abs(safe_float(r["interaction_value"])) for r in vals]) for family, vals in sorted(by_family.items())
        },
        "evidence_level_counts": dict(Counter(str(r["evidence_level"]) for r in interactions)),
        "caution": [
            "Candidate layer and position are selected from discovery/calibration cases; intervention is evaluated on heldout lexical/rule cases.",
            "Half scaling and feature permutation are diagnostic interventions, not natural gate reconstruction.",
            "Winner changes are target-vs-distractor changes, not strict full-vocabulary clean closure.",
        ],
    }
    write_json(out_dir / "phase313_heldout_component_interaction_summary.json", payload)
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase313_heldout_component_interaction_summary.json", payload)
        write_jsonl(base / "phase313_heldout_component_audit_rows.jsonl", audit)
        write_jsonl(base / "phase313_nonlinear_interaction_rows.jsonl", interactions)
        write_jsonl(base / "phase313_heldout_rollout_rows.jsonl", rollouts)
        write_jsonl(base / "phase313_missing_rows.jsonl", missing)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--candidates-per-family", type=int, default=2)
    parser.add_argument("--rollout-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        collect(args.round_name)
    elif args.model:
        audit_model(args)
    else:
        for model in MODELS:
            args.model = model
            audit_model(args)
        collect(args.round_name)


if __name__ == "__main__":
    main()
