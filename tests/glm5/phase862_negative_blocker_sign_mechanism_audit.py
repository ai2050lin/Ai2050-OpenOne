#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase854_full_vocab_blocker_min_cut_validation as p854  # noqa: E402
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 862
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase862_negative_blocker_sign_mechanism_audit")
PHASE861_SUMMARY = Path("tests/result/phase861_high_confidence_domain_gear_structure_comparison/phase861_summary.json")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_gear_key(key: str) -> dict[str, Any] | None:
    return p854.gear_from_key(key)


def gear_key(gear: dict[str, Any]) -> str:
    return p854.gear_key(gear)


def selected_signatures(model_name: str, min_level: int) -> list[dict[str, Any]]:
    if not PHASE861_SUMMARY.exists():
        raise FileNotFoundError(f"missing Phase 861 summary: {PHASE861_SUMMARY}")
    payload = read_json(PHASE861_SUMMARY)
    out = []
    for row in payload.get("signatures") or []:
        if str(row.get("model")) != model_name:
            continue
        if int(row.get("level") or 0) < int(min_level):
            continue
        gears = [parse_gear_key(str(key)) for key in row.get("gear_keys") or []]
        gears = [gear for gear in gears if gear is not None]
        if not gears:
            continue
        copied = dict(row)
        copied["gears"] = gears
        out.append(copied)
    return out


def selected_cases(domain: str, max_cases: int) -> list[dict[str, Any]]:
    rows = [dict(case) for case in p856.base_cases() if str(case.get("domain")) == domain]
    if int(max_cases) > 0:
        rows = rows[: int(max_cases)]
    return rows


def install_scaled_gear_edit(model, gears: list[dict[str, Any]], mode: str, scale_up_factor: float) -> list[Any]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for gear in gears:
        by_layer[int(gear["layer_idx"])].append(int(gear["channel_id"]))
    handles: list[Any] = []
    layers = get_layers(model)

    def make_hook(channels: list[int]):
        idx_cpu = torch.tensor([int(x) for x in channels], dtype=torch.long)

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]) or idx_cpu.numel() == 0:
                return inputs
            patched = inputs[0].clone()
            idx = idx_cpu.to(device=patched.device)
            valid = idx[(idx >= 0) & (idx < int(patched.shape[-1]))]
            if valid.numel() == 0:
                return inputs
            if mode == "zero":
                patched[:, -1, valid] = 0
            elif mode == "flip":
                patched[:, -1, valid] = -patched[:, -1, valid]
            elif mode in {"half", "scale_down"}:
                patched[:, -1, valid] = patched[:, -1, valid] * 0.5
            elif mode == "scale_up":
                patched[:, -1, valid] = patched[:, -1, valid] * float(scale_up_factor)
            else:
                raise ValueError(f"unknown gear edit mode: {mode}")
            return (patched, *inputs[1:])

        return hook

    for layer_idx, channels in by_layer.items():
        if 0 <= int(layer_idx) < len(layers) and hasattr(layers[int(layer_idx)].mlp, "down_proj"):
            handles.append(layers[int(layer_idx)].mlp.down_proj.register_forward_pre_hook(make_hook(channels)))
    return handles


def first_logits_with_scaled_gears(
    model,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    mode: str,
    scale_up_factor: float,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    if mode != "original" and gears:
        handles = install_scaled_gear_edit(model, gears, mode, scale_up_factor)
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()


def greedy_with_scaled_gears(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    mode: str,
    max_new_tokens: int,
    scale_up_factor: float,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles = []
        if step == 0 and mode != "original" and gears:
            handles = install_scaled_gear_edit(model, gears, mode, scale_up_factor)
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
        finally:
            for handle in handles:
                handle.remove()
        next_id = int(torch.argmax(logits).item())
        new_ids.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def best_score_for_ids(logits: torch.Tensor, token_ids: list[int]) -> tuple[float | None, int | None]:
    return p854.best_score_for_ids(logits, token_ids)


def token_delta(logits: torch.Tensor, original_logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None:
        return None
    if not (0 <= int(token_id) < int(logits.numel()) and 0 <= int(token_id) < int(original_logits.numel())):
        return None
    return float(logits[int(token_id)].item() - original_logits[int(token_id)].item())


def original_blocker_deltas(
    logits: torch.Tensor,
    original_logits: torch.Tensor,
    original_blockers: list[dict[str, Any]],
    max_blockers: int,
) -> dict[str, Any]:
    deltas = []
    by_role: dict[str, list[float]] = defaultdict(list)
    for blocker in (original_blockers or [])[: int(max_blockers)]:
        token_id = int(blocker.get("token_id"))
        delta = token_delta(logits, original_logits, token_id)
        if delta is None:
            continue
        deltas.append(delta)
        by_role[str(blocker.get("role") or "unknown")].append(delta)
    return {
        "original_blocker_delta_mean": mean(deltas),
        "original_blocker_delta_top1": deltas[0] if deltas else None,
        "original_blocker_delta_negative_count": sum(1 for value in deltas if value < 0),
        "original_blocker_delta_positive_count": sum(1 for value in deltas if value > 0),
        "original_blocker_delta_by_role": {role: mean(values) for role, values in sorted(by_role.items())},
    }


def spec_rows_for_signature(signature: dict[str, Any], modes: list[str], include_single: bool) -> list[dict[str, Any]]:
    gears = list(signature["gears"])
    specs = [
        {
            "condition_type": "full_set",
            "candidate_key": "+".join(gear_key(gear) for gear in gears) + f":{mode}",
            "subset_name": "full",
            "mode": mode,
            "gears": gears,
        }
        for mode in modes
    ]
    if include_single:
        for idx, gear in enumerate(gears):
            for mode in modes:
                specs.append(
                    {
                        "condition_type": "single_channel",
                        "candidate_key": gear_key(gear) + f":single{idx}:{mode}",
                        "subset_name": f"single{idx}",
                        "mode": mode,
                        "gears": [gear],
                    }
                )
    return specs


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    signatures = selected_signatures(args.model, int(args.min_level))
    prompt_variants = parse_csv(args.prompt_variants)
    modes = parse_csv(args.edit_modes)
    if args.dry_run or not signatures:
        payload = {
            "phase": PHASE,
            "title": "Negative-Blocker Sign Mechanism Audit",
            "model": args.model,
            "round": args.round_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "no_level6_target" if not signatures else "dry_run",
            "target_signatures": [
                {key: row.get(key) for key in ("model", "domain", "gear_keys", "level", "candidate_role", "best_mode")}
                for row in signatures
            ],
            "prompt_variants": prompt_variants,
            "edit_modes": modes,
        }
        p846.write_json(out_dir / f"phase862_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase862_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        for sig_idx, signature in enumerate(signatures, 1):
            domain = str(signature["domain"])
            cases = selected_cases(domain, int(args.max_cases_per_domain))
            specs = spec_rows_for_signature(signature, modes, bool(args.include_single_channels))
            for case_idx, case in enumerate(cases, 1):
                sets = p856.token_sets(tokenizer, case)
                for prompt_variant in prompt_variants:
                    prompt = p856.prompt_for_case(case, prompt_variant)
                    prompt_ids = p844.encode_prompt(tokenizer, prompt)
                    original_logits = first_logits_with_scaled_gears(
                        model, device, prompt_ids, [], "original", float(args.scale_up_factor)
                    )
                    original_first = p856.first_token_metrics(tokenizer, original_logits, sets, int(args.topk_tokens))
                    original_blockers = p854.blocker_metrics(tokenizer, original_logits, sets, int(args.topk_blockers))
                    original_generated, original_token_ids = greedy_with_scaled_gears(
                        model,
                        tokenizer,
                        device,
                        prompt_ids,
                        [],
                        "original",
                        int(args.max_new_tokens),
                        float(args.scale_up_factor),
                    )
                    original_rollout = p856.classify_rollout(original_generated, case)
                    class_score_orig, class_id_orig = best_score_for_ids(original_logits, sets["class_target_ids"])
                    object_score_orig, object_id_orig = best_score_for_ids(original_logits, sets["object_ids"])
                    strict_score_orig, strict_id_orig = best_score_for_ids(original_logits, sets["strict_target_ids"])

                    base_common = {
                        "row_kind": "phase862_negative_blocker_sign_mechanism_audit",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "domain": domain,
                        "case_id": case["case_id"],
                        "object": case["object"],
                        "canonical_answer": case["canonical_answer"],
                        "prompt_variant": prompt_variant,
                        "prompt": prompt,
                        "source_level": signature.get("level"),
                        "source_gear_keys": signature.get("gear_keys"),
                        "source_depth_band": signature.get("depth_band"),
                        "source_candidate_role": signature.get("candidate_role"),
                        "source_best_mode": signature.get("best_mode"),
                        **sets,
                    }
                    rows.append(
                        {
                            **base_common,
                            "condition_type": "original",
                            "candidate_key": "original",
                            "subset_name": "original",
                            "edit_mode": "original",
                            "gear_count": 0,
                            "gear_keys": [],
                            "token_ids": original_token_ids,
                            "generated_clean": p856.clean_text(original_generated),
                            **original_first,
                            **{f"blocker_{k}": v for k, v in original_blockers.items()},
                            **original_rollout,
                            "class_answer_delta": 0.0,
                            "object_delta": 0.0,
                            "strict_delta": 0.0,
                            "original_blocker_delta_mean": 0.0,
                            "original_blocker_delta_top1": 0.0,
                            "original_blocker_delta_negative_count": 0,
                            "original_blocker_delta_positive_count": 0,
                        }
                    )

                    for spec in specs:
                        valid_gears = [
                            gear
                            for gear in spec["gears"]
                            if 0 <= int(gear["layer_idx"]) < n_layers and int(gear["channel_id"]) >= 0
                        ]
                        logits = first_logits_with_scaled_gears(
                            model, device, prompt_ids, valid_gears, str(spec["mode"]), float(args.scale_up_factor)
                        )
                        first = p856.first_token_metrics(tokenizer, logits, sets, int(args.topk_tokens))
                        blocker = p854.blocker_metrics(tokenizer, logits, sets, int(args.topk_blockers))
                        generated, token_ids = greedy_with_scaled_gears(
                            model,
                            tokenizer,
                            device,
                            prompt_ids,
                            valid_gears,
                            str(spec["mode"]),
                            int(args.max_new_tokens),
                            float(args.scale_up_factor),
                        )
                        rollout = p856.classify_rollout(generated, case)
                        class_score, _ = best_score_for_ids(logits, sets["class_target_ids"])
                        object_score, _ = best_score_for_ids(logits, sets["object_ids"])
                        strict_score, _ = best_score_for_ids(logits, sets["strict_target_ids"])
                        blocker_deltas = original_blocker_deltas(
                            logits,
                            original_logits,
                            original_blockers.get("class_top_blockers") or [],
                            int(args.topk_blockers),
                        )
                        rows.append(
                            {
                                **base_common,
                                "condition_type": spec["condition_type"],
                                "candidate_key": spec["candidate_key"],
                                "subset_name": spec["subset_name"],
                                "edit_mode": spec["mode"],
                                "scale_up_factor": float(args.scale_up_factor),
                                "gear_count": len(valid_gears),
                                "gear_keys": [gear_key(gear) for gear in valid_gears],
                                "token_ids": token_ids,
                                "generated_clean": p856.clean_text(generated),
                                **first,
                                **{f"blocker_{k}": v for k, v in blocker.items()},
                                **rollout,
                                "class_answer_delta": None if class_score is None or class_score_orig is None else float(class_score - class_score_orig),
                                "object_delta": None if object_score is None or object_score_orig is None else float(object_score - object_score_orig),
                                "strict_delta": None if strict_score is None or strict_score_orig is None else float(strict_score - strict_score_orig),
                                "original_class_token_delta": token_delta(logits, original_logits, class_id_orig),
                                "original_object_token_delta": token_delta(logits, original_logits, object_id_orig),
                                "original_strict_token_delta": token_delta(logits, original_logits, strict_id_orig),
                                **blocker_deltas,
                            }
                        )
            log(f"{args.model}/{args.round_name}: signature {sig_idx}/{len(signatures)} domain={domain} rows={len(rows)}")
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(args, signatures, rows, attn_impl)
    p846.write_jsonl(out_dir / f"phase862_{args.model}_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase862_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "status": summary.get("status"),
                "rows": len(rows),
                "domains": summary.get("domains"),
                "sign_mechanism": summary.get("sign_mechanism_by_domain"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("domain")), str(row.get("case_id")), str(row.get("prompt_variant")))


def pair_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    originals = {row_key(row): row for row in rows if row.get("condition_type") == "original"}
    grouped: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row.get("condition_type") == "original":
            continue
        base = originals.get(row_key(row))
        if base is not None:
            grouped[
                (
                    str(row.get("domain")),
                    str(row.get("condition_type")),
                    str(row.get("subset_name")),
                    str(row.get("edit_mode")),
                )
            ].append((base, row))

    out: list[dict[str, Any]] = []
    for (domain, condition_type, subset_name, edit_mode), pairs in grouped.items():
        clear_gain = sum(1 for base, row in pairs if not base.get("rollout_clear_answer_class") and row.get("rollout_clear_answer_class"))
        clear_loss = sum(1 for base, row in pairs if base.get("rollout_clear_answer_class") and not row.get("rollout_clear_answer_class"))
        rollout_gain = sum(1 for base, row in pairs if not base.get("rollout_answer_class") and row.get("rollout_answer_class"))
        rollout_loss = sum(1 for base, row in pairs if base.get("rollout_answer_class") and not row.get("rollout_answer_class"))
        first_gain = sum(1 for base, row in pairs if not base.get("first_token_answer_class") and row.get("first_token_answer_class"))
        first_loss = sum(1 for base, row in pairs if base.get("first_token_answer_class") and not row.get("first_token_answer_class"))
        blocker_reduction = [
            finite(base.get("class_blocker_count")) - finite(row.get("class_blocker_count"))
            for base, row in pairs
            if base.get("class_blocker_count") is not None and row.get("class_blocker_count") is not None
        ]
        margin_gain = [
            finite(row.get("class_minus_object_logit")) - finite(base.get("class_minus_object_logit"))
            for base, row in pairs
            if base.get("class_minus_object_logit") is not None and row.get("class_minus_object_logit") is not None
        ]
        original_blocker_delta = [
            finite(row.get("original_blocker_delta_mean"))
            for _, row in pairs
            if row.get("original_blocker_delta_mean") is not None
        ]
        answer_delta = [finite(row.get("class_answer_delta")) for _, row in pairs if row.get("class_answer_delta") is not None]
        object_delta = [finite(row.get("object_delta")) for _, row in pairs if row.get("object_delta") is not None]
        strict_delta = [finite(row.get("strict_delta")) for _, row in pairs if row.get("strict_delta") is not None]
        out.append(
            {
                "domain": domain,
                "condition_type": condition_type,
                "subset_name": subset_name,
                "edit_mode": edit_mode,
                "n_pairs": len(pairs),
                "clear_rollout_gain": clear_gain,
                "clear_rollout_loss": clear_loss,
                "rollout_gain": rollout_gain,
                "rollout_loss": rollout_loss,
                "first_gain": first_gain,
                "first_loss": first_loss,
                "mean_class_blocker_reduction": mean(blocker_reduction),
                "mean_class_minus_object_gain": mean(margin_gain),
                "mean_original_blocker_delta": mean(original_blocker_delta),
                "mean_answer_delta": mean(answer_delta),
                "mean_object_delta": mean(object_delta),
                "mean_strict_delta": mean(strict_delta),
                "blocker_weakening_supported": bool(
                    clear_gain > 0
                    and (mean(original_blocker_delta) or 0.0) < 0
                    and (mean(blocker_reduction) or 0.0) > 0
                ),
                "answer_lift_supported": bool(clear_gain > 0 and (mean(answer_delta) or 0.0) > 0),
            }
        )
    out.sort(key=lambda row: (row["domain"], row["condition_type"], row["subset_name"], row["edit_mode"]))
    return out


def sign_mechanism_summary(effects: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        if row.get("condition_type") == "full_set":
            by_domain[str(row.get("domain"))].append(row)
    out: dict[str, Any] = {}
    for domain, rows in sorted(by_domain.items()):
        by_mode = {str(row.get("edit_mode")): row for row in rows}
        clear_modes = [
            mode
            for mode, row in by_mode.items()
            if int(row.get("clear_rollout_gain") or 0) > 0 and int(row.get("clear_rollout_loss") or 0) == 0
        ]
        weakening_modes = [
            mode
            for mode, row in by_mode.items()
            if bool(row.get("blocker_weakening_supported"))
        ]
        answer_lift_modes = [
            mode
            for mode, row in by_mode.items()
            if bool(row.get("answer_lift_supported"))
        ]
        out[domain] = {
            "clear_gain_modes": sorted(clear_modes),
            "blocker_weakening_modes": sorted(weakening_modes),
            "answer_lift_modes": sorted(answer_lift_modes),
            "zero_and_flip_both_clear": "zero" in clear_modes and "flip" in clear_modes,
            "zero_and_flip_both_weaken_blockers": "zero" in weakening_modes and "flip" in weakening_modes,
            "scale_down_clear": "half" in clear_modes or "scale_down" in clear_modes,
            "scale_up_clear": "scale_up" in clear_modes,
            "interpretation": (
                "shared_blocker_weakening"
                if "zero" in weakening_modes and "flip" in weakening_modes
                else "mode_specific_or_unresolved"
            ),
        }
    return out


def compact_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "rollout_clear_answer_class": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "rollout_answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "first_token_answer_class": sum(1 for row in rows if row.get("first_token_answer_class")),
        "mean_class_blocker_count": mean([finite(row.get("class_blocker_count")) for row in rows if row.get("class_blocker_count") is not None]),
        "mean_class_minus_object_logit": mean(
            [finite(row.get("class_minus_object_logit")) for row in rows if row.get("class_minus_object_logit") is not None]
        ),
        "mean_original_blocker_delta": mean(
            [finite(row.get("original_blocker_delta_mean")) for row in rows if row.get("original_blocker_delta_mean") is not None]
        ),
        "rollout_labels": dict(Counter(str(row.get("rollout_label")) for row in rows)),
    }


def summarize(args: argparse.Namespace, signatures: list[dict[str, Any]], rows: list[dict[str, Any]], attn_impl: str | None) -> dict[str, Any]:
    effects = pair_effects(rows)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition_type"))].append(row)
        by_domain[str(row.get("domain"))].append(row)
    return {
        "phase": PHASE,
        "title": "Negative-Blocker Sign Mechanism Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "target_signatures": [
            {key: row.get(key) for key in ("model", "domain", "gear_keys", "level", "candidate_role", "best_mode", "depth_band")}
            for row in signatures
        ],
        "domains": sorted({str(row.get("domain")) for row in rows if row.get("domain")}),
        "prompt_variants": parse_csv(args.prompt_variants),
        "edit_modes": parse_csv(args.edit_modes),
        "include_single_channels": bool(args.include_single_channels),
        "n_rows": len(rows),
        "condition_summary": {key: compact_condition(group) for key, group in sorted(by_condition.items())},
        "domain_summary": {key: compact_condition(group) for key, group in sorted(by_domain.items())},
        "pair_effects": effects,
        "sign_mechanism_by_domain": sign_mechanism_summary(effects),
        "boundary": (
            "This phase audits why Level 6 negative-blocker gears show sign ambiguity. "
            "It does not search for new gears and does not prove language closure."
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 862 Negative-Blocker Sign Mechanism Audit ({payload['round']})",
        "",
        "- Source: Phase 861 Level 6 high-confidence signatures.",
        "- Boundary: sign-mechanism audit, not gear search and not language closure.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | status | rows | domains | clear modes by domain | interpretation |",
        "|---|---|---:|---|---|---|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        sign = summary.get("sign_mechanism_by_domain") or {}
        clear = {domain: row.get("clear_gain_modes") for domain, row in sign.items()}
        interp = {domain: row.get("interpretation") for domain, row in sign.items()}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {summary.get('n_rows', 0)} | "
            f"`{summary.get('domains', [])}` | `{clear}` | `{interp}` |"
        )
    lines += [
        "",
        "## Full-Set Effects",
        "",
        "| model | domain | mode | clear gain/loss | blocker reduction | original blocker delta | answer delta | object delta | weaken? | answer lift? |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        for row in summary.get("pair_effects") or []:
            if row.get("condition_type") != "full_set":
                continue
            lines.append(
                f"| {model_name} | {row.get('domain')} | `{row.get('edit_mode')}` | "
                f"{row.get('clear_rollout_gain', 0)}/{row.get('clear_rollout_loss', 0)} | "
                f"{fmt(row.get('mean_class_blocker_reduction'))} | "
                f"{fmt(row.get('mean_original_blocker_delta'))} | "
                f"{fmt(row.get('mean_answer_delta'))} | "
                f"{fmt(row.get('mean_object_delta'))} | "
                f"{row.get('blocker_weakening_supported')} | {row.get('answer_lift_supported')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase862_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase862_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase862_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--min-level", type=int, default=6)
    parser.add_argument("--max-cases-per-domain", type=int, default=5)
    parser.add_argument("--prompt-variants", default="natural_question,natural_category,classification")
    parser.add_argument("--edit-modes", default="zero,flip,half,scale_up")
    parser.add_argument("--include-single-channels", action="store_true")
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--topk-blockers", type=int, default=10)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(
            json.dumps(
                {"phase": PHASE, "round": args.round_name, "status": payload.get("status"), "models": payload.get("models")},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
