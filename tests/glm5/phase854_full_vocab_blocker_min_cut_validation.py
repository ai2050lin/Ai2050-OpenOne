#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase843_core_channel_natural_route_validation as p843  # noqa: E402
import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase853_strong_edge_expansion_natural_closure_validation as p853  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 854
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase854_full_vocab_blocker_min_cut_validation")
PHASE853_ROOT = Path("tests/result/phase853_strong_edge_expansion_natural_closure_validation")
PHASE851_ROOT = Path("tests/result/phase851_global_atlas_schema_orthogonality_audit")
TARGET_CLASS = "target_equivalent"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def first_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    return int(ids[0]) if ids else None


def decode_token(tokenizer, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    return tokenizer.decode([int(token_id)])


def gear_key(gear: dict[str, Any]) -> str:
    return str(gear.get("gear_key") or f"L{int(gear['layer_idx'])}C{int(gear['channel_id'])}")


def parse_gear_key(text: str) -> tuple[int, int] | None:
    key = str(text)
    if not key.startswith("L") or "C" not in key:
        return None
    try:
        layer_text, channel_text = key[1:].split("C", 1)
        return int(layer_text), int(channel_text)
    except ValueError:
        return None


def gear_from_key(text: str) -> dict[str, Any] | None:
    parsed = parse_gear_key(text)
    if not parsed:
        return None
    layer_idx, channel_id = parsed
    return {"layer_idx": int(layer_idx), "channel_id": int(channel_id), "gear_key": f"L{layer_idx}C{channel_id}"}


def combo_key(gears: list[dict[str, Any]]) -> str:
    return "+".join(gear_key(gear) for gear in gears) if gears else "original"


def row_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(name, ""))
        for name in ["model", "case_id", "prompt_variant", "combo_type", "edit_mode", "combo_key"]
    )


def residual_label(row: dict[str, Any], threshold: float) -> str:
    return p853.residual_class(finite(row.get("interaction_residual")), threshold)


def is_strong(row: dict[str, Any], threshold: float) -> bool:
    return p853.class_is_strong(residual_label(row, threshold))


def is_interaction(row: dict[str, Any]) -> bool:
    return row.get("combo_type") in {"pair", "triplet", "focus"} and row.get("interaction_residual") is not None


def phase853_rows_path(round_name: str, model_name: str) -> Path:
    return PHASE853_ROOT / round_name / f"phase853_{model_name}_rows.jsonl"


def load_phase853_rows(model_name: str, round_name: str) -> list[dict[str, Any]]:
    path = phase853_rows_path(round_name, model_name)
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 853 rows: {path}")
    return p846.read_jsonl(path)


def load_phase851_min_cut_keys(model_name: str, round_name: str) -> set[str]:
    path = PHASE851_ROOT / round_name / f"phase851_{model_name}_atlas_audit.json"
    if not path.exists():
        return set()
    data = p846.read_json(path)
    keys: set[str] = set()
    for row in data.get("gear_min_cut_candidates") or []:
        if row.get("audit_status") != "counterfactual_min_cut_candidate":
            continue
        parsed = parse_gear_key(str(row.get("gear")))
        if parsed:
            keys.add(f"L{parsed[0]}C{parsed[1]}")
    return keys


def classify_source_row(row: dict[str, Any], threshold: float) -> str:
    if is_strong(row, threshold) and row.get("target_transition") and not row.get("exact_natural_consistency"):
        return "strong_target_not_exact"
    if is_strong(row, threshold):
        return "strong_non_target_or_unknown"
    return "additive_control"


def select_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in load_phase853_rows(args.model, args.source_round) if is_interaction(row)]
    threshold = float(args.interaction_threshold)
    seen: set[str] = set()

    def dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in candidates:
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    target_fail = [
        row for row in rows if is_strong(row, threshold) and row.get("target_transition") and not row.get("exact_natural_consistency")
    ]
    strong_other = [row for row in rows if is_strong(row, threshold) and row not in target_fail]
    controls = [row for row in rows if not is_strong(row, threshold)]

    target_fail.sort(
        key=lambda row: (
            finite(row.get("best_target_rank"), 999999.0),
            -abs(finite(row.get("interaction_residual"))),
            str(row.get("object")),
            str(row.get("prompt_variant")),
        )
    )
    strong_other.sort(
        key=lambda row: (
            -abs(finite(row.get("interaction_residual"))),
            finite(row.get("best_target_rank"), 999999.0),
            str(row.get("object")),
        )
    )
    controls.sort(
        key=lambda row: (
            abs(finite(row.get("interaction_residual"))),
            finite(row.get("best_target_rank"), 999999.0),
            str(row.get("object")),
        )
    )
    selected = []
    selected.extend(dedupe(target_fail[: int(args.max_target_fail_rows)]))
    selected.extend(dedupe(strong_other[: int(args.max_strong_non_target_rows)]))
    selected.extend(dedupe(controls[: int(args.max_control_rows)]))
    return selected[: int(args.max_source_rows)] if int(args.max_source_rows) > 0 else selected


def case_from_source(row: dict[str, Any]) -> dict[str, Any]:
    obj = str(row.get("object") or "")
    return {
        "case_id": str(row.get("case_id") or f"p854_{obj}_case"),
        "object": obj,
        "question": f"Which category best describes a {obj}?",
        "answer": str(row.get("target_answer") or "geometric shape"),
        "contrast_answer": str(row.get("contrast_answer") or "living thing"),
        "distractors": ["hand tool", "public transport", "musical instrument", "warm color"],
        "synthetic_case": bool(row.get("synthetic_case")),
    }


def token_variant_ids(tokenizer, variants: list[str]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for text in variants:
        token_id = first_token_id(tokenizer, text)
        if token_id is not None and token_id not in seen:
            ids.append(int(token_id))
            seen.add(int(token_id))
    return ids


def answer_aliases(answer: str) -> list[str]:
    base = str(answer or "").strip()
    aliases = [base, base.capitalize(), base.title(), "geometric", "Geometric", "geometry", "Geometry", "shape", "Shape", "polygon", "Polygon"]
    with_space = [f" {item}" for item in aliases if item]
    return [item for item in [*aliases, *with_space] if item]


def object_aliases(obj: str) -> list[str]:
    base = str(obj or "").strip()
    aliases = [base, base.capitalize(), base.title()]
    with_space = [f" {item}" for item in aliases if item]
    return [item for item in [*aliases, *with_space] if item]


def target_token_sets(tokenizer, case: dict[str, Any]) -> dict[str, Any]:
    answer = str(case.get("answer") or "")
    obj = str(case.get("object") or "")
    strict_target_ids = token_variant_ids(tokenizer, [answer, "polygon"])
    class_target_ids = token_variant_ids(tokenizer, answer_aliases(answer))
    object_ids = token_variant_ids(tokenizer, object_aliases(obj))
    return {
        "strict_target_ids": strict_target_ids,
        "strict_target_tokens": [decode_token(tokenizer, token_id) for token_id in strict_target_ids],
        "class_target_ids": class_target_ids,
        "class_target_tokens": [decode_token(tokenizer, token_id) for token_id in class_target_ids],
        "object_ids": object_ids,
        "object_tokens": [decode_token(tokenizer, token_id) for token_id in object_ids],
    }


def rank_of_score(logits: torch.Tensor, score: float | None) -> int | None:
    if score is None or not math.isfinite(float(score)):
        return None
    return int((logits > float(score)).sum().item()) + 1


def best_score_for_ids(logits: torch.Tensor, token_ids: list[int]) -> tuple[float | None, int | None]:
    best_score: float | None = None
    best_id: int | None = None
    for token_id in token_ids:
        if 0 <= int(token_id) < int(logits.numel()):
            score = float(logits[int(token_id)].item())
            if best_score is None or score > best_score:
                best_score = score
                best_id = int(token_id)
    return best_score, best_id


def token_role(token_text: str, token_id: int, strict_ids: set[int], class_ids: set[int], object_ids: set[int]) -> str:
    if token_id in strict_ids:
        return "strict_target"
    if token_id in class_ids:
        return "answer_class"
    if token_id in object_ids:
        return "object_echo"
    stripped = token_text.strip()
    if not stripped:
        return "format_space"
    if re.fullmatch(r"[\W_]+", stripped, flags=re.UNICODE):
        return "format_punct"
    if stripped[:1].isdigit():
        return "number"
    if stripped.lower() in {"the", "a", "an", "it", "item", "category", "answer"}:
        return "protocol_word"
    return "other_blocker"


def blocker_metrics(tokenizer, logits: torch.Tensor, token_sets: dict[str, Any], topk: int) -> dict[str, Any]:
    strict_ids = {int(x) for x in token_sets["strict_target_ids"]}
    class_ids = {int(x) for x in token_sets["class_target_ids"]}
    object_ids = {int(x) for x in token_sets["object_ids"]}
    strict_score, strict_id = best_score_for_ids(logits, token_sets["strict_target_ids"])
    class_score, class_id = best_score_for_ids(logits, token_sets["class_target_ids"])
    object_score, object_id = best_score_for_ids(logits, token_sets["object_ids"])

    def count_blockers(score: float | None, excluded: set[int]) -> int | None:
        if score is None:
            return None
        mask = logits > float(score)
        if excluded:
            idx = torch.tensor([x for x in excluded if 0 <= x < int(logits.numel())], dtype=torch.long, device=logits.device)
            if idx.numel() > 0:
                mask[idx] = False
        return int(mask.sum().item())

    def top_blockers(score: float | None, excluded: set[int]) -> list[dict[str, Any]]:
        if score is None:
            return []
        mask = logits > float(score)
        if excluded:
            idx = torch.tensor([x for x in excluded if 0 <= x < int(logits.numel())], dtype=torch.long, device=logits.device)
            if idx.numel() > 0:
                mask[idx] = False
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        if ids.numel() == 0:
            return []
        vals = logits[ids]
        k = min(int(topk), int(ids.numel()))
        top = torch.topk(vals, k=k)
        out: list[dict[str, Any]] = []
        for local_idx, value in zip(top.indices.tolist(), top.values.tolist(), strict=False):
            token_id = int(ids[int(local_idx)].item())
            text = decode_token(tokenizer, token_id) or ""
            out.append(
                {
                    "token_id": token_id,
                    "token": text,
                    "logit": float(value),
                    "gap_vs_threshold": float(value) - float(score),
                    "role": token_role(text, token_id, strict_ids, class_ids, object_ids),
                }
            )
        return out

    top = torch.topk(logits, k=min(int(topk), int(logits.numel())))
    top_tokens = []
    for token_id, value in zip(top.indices.tolist(), top.values.tolist(), strict=False):
        text = decode_token(tokenizer, int(token_id)) or ""
        top_tokens.append(
            {
                "token_id": int(token_id),
                "token": text,
                "logit": float(value),
                "role": token_role(text, int(token_id), strict_ids, class_ids, object_ids),
            }
        )
    strict_blockers = top_blockers(strict_score, strict_ids)
    class_blockers = top_blockers(class_score, class_ids)
    return {
        "strict_best_target_logit": strict_score,
        "strict_best_target_id": strict_id,
        "strict_best_target_token": decode_token(tokenizer, strict_id),
        "strict_best_target_rank": rank_of_score(logits, strict_score),
        "strict_blocker_count": count_blockers(strict_score, strict_ids),
        "strict_top_blockers": strict_blockers,
        "strict_top_blocker_token": strict_blockers[0]["token"] if strict_blockers else None,
        "strict_top_blocker_role": strict_blockers[0]["role"] if strict_blockers else None,
        "strict_top_blocker_gap": strict_blockers[0]["gap_vs_threshold"] if strict_blockers else None,
        "class_best_target_logit": class_score,
        "class_best_target_id": class_id,
        "class_best_target_token": decode_token(tokenizer, class_id),
        "class_best_target_rank": rank_of_score(logits, class_score),
        "class_blocker_count": count_blockers(class_score, class_ids),
        "class_top_blockers": class_blockers,
        "class_top_blocker_token": class_blockers[0]["token"] if class_blockers else None,
        "class_top_blocker_role": class_blockers[0]["role"] if class_blockers else None,
        "class_top_blocker_gap": class_blockers[0]["gap_vs_threshold"] if class_blockers else None,
        "object_best_logit": object_score,
        "object_best_id": object_id,
        "object_best_token": decode_token(tokenizer, object_id),
        "object_rank": rank_of_score(logits, object_score),
        "class_minus_object_logit": None
        if class_score is None or object_score is None
        else float(class_score) - float(object_score),
        "strict_minus_object_logit": None
        if strict_score is None or object_score is None
        else float(strict_score) - float(object_score),
        "top_tokens": top_tokens,
        "strict_closure": count_blockers(strict_score, strict_ids) == 0 if strict_score is not None else False,
        "answer_class_closure": count_blockers(class_score, class_ids) == 0 if class_score is not None else False,
    }


def ordered_candidate_keys(row: dict[str, Any], min_cut_keys: set[str], max_candidates: int) -> list[str]:
    keys = [str(key) for key in row.get("gear_keys") or p846.split_combo_key(str(row.get("combo_key") or ""))]
    keys = [key for key in keys if parse_gear_key(key)]
    preferred = [key for key in keys if key in min_cut_keys]
    rest = [key for key in keys if key not in set(preferred)]
    out: list[str] = []
    for key in [*preferred, *rest]:
        if key not in out:
            out.append(key)
        if len(out) >= int(max_candidates):
            break
    return out


def condition_specs(row: dict[str, Any], min_cut_keys: set[str], max_candidates: int) -> list[dict[str, Any]]:
    row_gears = [gear_from_key(key) for key in row.get("gear_keys", [])]
    row_gears = [gear for gear in row_gears if gear is not None]
    mode = str(row.get("edit_mode") or "zero")
    candidates = ordered_candidate_keys(row, min_cut_keys, max_candidates)
    specs: list[dict[str, Any]] = [
        {"condition_type": "original", "candidate_key": None, "mode": "original", "gears": []},
        {"condition_type": "full_combo", "candidate_key": None, "mode": mode, "gears": row_gears},
    ]
    by_key = {gear_key(gear): gear for gear in row_gears}
    for key in candidates:
        if key not in by_key:
            continue
        specs.append({"condition_type": "candidate_only", "candidate_key": key, "mode": mode, "gears": [by_key[key]]})
        remain = [gear for gear in row_gears if gear_key(gear) != key]
        if remain:
            specs.append({"condition_type": "without_candidate", "candidate_key": key, "mode": mode, "gears": remain})
    return specs


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = select_source_rows(args)
    min_cut_keys = load_phase851_min_cut_keys(args.model, args.phase851_round)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "source_round": args.source_round,
            "selected_source_rows": [
                {
                    "source_group": classify_source_row(row, float(args.interaction_threshold)),
                    "case_id": row.get("case_id"),
                    "object": row.get("object"),
                    "prompt_variant": row.get("prompt_variant"),
                    "combo_type": row.get("combo_type"),
                    "edit_mode": row.get("edit_mode"),
                    "combo_key": row.get("combo_key"),
                    "interaction_residual": row.get("interaction_residual"),
                    "boundary_class": row.get("boundary_class"),
                    "best_target_rank": row.get("best_target_rank"),
                    "exact_natural_consistency": row.get("exact_natural_consistency"),
                }
                for row in source_rows
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    if not source_rows:
        summary = summarize(args, None, [], [], [], skipped=True, skip_reason="no selected Phase 853 source rows")
        p846.write_jsonl(out_dir / f"phase854_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase854_{args.model}_edge_rows.jsonl", [])
        p846.write_json(out_dir / f"phase854_{args.model}_summary.json", summary)
        return summary

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        for idx, source in enumerate(source_rows, 1):
            case = case_from_source(source)
            prompt = str(source.get("prompt") or p844.prompt_for_case(case, str(source.get("prompt_variant") or "natural_question")))
            prompt_ids = p844.encode_prompt(tokenizer, prompt)
            token_sets = target_token_sets(tokenizer, case)
            specs = condition_specs(source, min_cut_keys, int(args.max_candidates_per_row))
            for spec in specs:
                valid_gears = [
                    gear
                    for gear in spec["gears"]
                    if 0 <= int(gear["layer_idx"]) < n_layers and int(gear["channel_id"]) >= 0
                ]
                logits = p844.first_logits_with_gears(model, device, prompt_ids, valid_gears, str(spec["mode"]))
                metrics = blocker_metrics(tokenizer, logits, token_sets, int(args.topk_blockers))
                row = {
                    "row_kind": "phase854_full_vocab_blocker_min_cut_validation",
                    "phase": PHASE,
                    "model": args.model,
                    "round": args.round_name,
                    "source_round": args.source_round,
                    "phase851_round": args.phase851_round,
                    "source_row_key": row_key(source),
                    "source_group": classify_source_row(source, float(args.interaction_threshold)),
                    "source_combo_type": source.get("combo_type"),
                    "source_edit_mode": source.get("edit_mode"),
                    "source_combo_key": source.get("combo_key"),
                    "source_gear_keys": source.get("gear_keys"),
                    "source_interaction_residual": source.get("interaction_residual"),
                    "source_boundary_class": source.get("boundary_class"),
                    "source_target_transition": source.get("target_transition"),
                    "source_exact_natural_consistency": source.get("exact_natural_consistency"),
                    "source_best_target_rank": source.get("best_target_rank"),
                    "case_id": case["case_id"],
                    "object": case.get("object"),
                    "target_answer": case.get("answer"),
                    "prompt_variant": source.get("prompt_variant"),
                    "prompt": prompt,
                    "condition_type": spec["condition_type"],
                    "candidate_key": spec["candidate_key"],
                    "candidate_is_phase851_min_cut": bool(spec["candidate_key"] in min_cut_keys) if spec["candidate_key"] else False,
                    "edit_mode": spec["mode"],
                    "gear_count": len(valid_gears),
                    "gear_keys": [gear_key(gear) for gear in valid_gears],
                    "condition_combo_key": combo_key(valid_gears),
                    **token_sets,
                    **metrics,
                }
                rows.append(row)
            if idx % max(1, int(args.log_every)) == 0 or idx == len(source_rows):
                log(f"{args.model}/{args.round_name}: blocker audit source rows {idx}/{len(source_rows)} emitted_rows={len(rows)}")
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    edge_rows = build_edge_rows(rows)
    summary = summarize(args, attn_impl, source_rows, rows, edge_rows, skipped=False, skip_reason=None)
    p846.write_jsonl(out_dir / f"phase854_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase854_{args.model}_edge_rows.jsonl", edge_rows)
    p846.write_json(out_dir / f"phase854_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "source_rows": len(source_rows),
                "audit_rows": len(rows),
                "full_combo_rows": summary["transition_summary"]["full_combo_rows"],
                "full_combo_answer_class_closure": summary["transition_summary"]["full_combo_answer_class_closure"],
                "necessary_min_cut_candidates": summary["min_cut_summary"]["necessary_blocker_reducer"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def build_edge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_row_key"))].append(row)
    out: list[dict[str, Any]] = []
    for source_key, group in by_source.items():
        original = next((row for row in group if row.get("condition_type") == "original"), None)
        full = next((row for row in group if row.get("condition_type") == "full_combo"), None)
        if not original or not full:
            continue
        candidates = sorted({str(row.get("candidate_key")) for row in group if row.get("candidate_key")})
        for candidate in candidates:
            only = next(
                (row for row in group if row.get("condition_type") == "candidate_only" and row.get("candidate_key") == candidate),
                None,
            )
            without = next(
                (row for row in group if row.get("condition_type") == "without_candidate" and row.get("candidate_key") == candidate),
                None,
            )
            orig_blockers = finite(original.get("class_blocker_count"))
            full_blockers = finite(full.get("class_blocker_count"))
            orig_rank = finite(original.get("class_best_target_rank"), 999999.0)
            full_rank = finite(full.get("class_best_target_rank"), 999999.0)
            without_blockers = finite(without.get("class_blocker_count")) if without else None
            without_rank = finite(without.get("class_best_target_rank"), 999999.0) if without else None
            only_blockers = finite(only.get("class_blocker_count")) if only else None
            only_rank = finite(only.get("class_best_target_rank"), 999999.0) if only else None
            full_reduces = full_blockers < orig_blockers or full_rank < orig_rank
            rebound = None if without_blockers is None else float(without_blockers - full_blockers)
            rank_loss = None if without_rank is None else float(without_rank - full_rank)
            sufficient_reduction = None if only_blockers is None else float(orig_blockers - only_blockers)
            if rebound is not None and rebound > 0 and full_reduces:
                label = "necessary_blocker_reducer"
            elif sufficient_reduction is not None and sufficient_reduction > 0:
                label = "single_sufficient_partial_reducer"
            elif rebound is not None and rebound < 0:
                label = "candidate_harmful_or_antagonistic"
            else:
                label = "weak_or_no_min_cut_effect"
            candidate_is_phase851 = bool(
                (only or {}).get("candidate_is_phase851_min_cut")
                or (without or {}).get("candidate_is_phase851_min_cut")
            )
            out.append(
                {
                    "row_kind": "phase854_min_cut_edge_summary",
                    "phase": PHASE,
                    "model": full.get("model"),
                    "round": full.get("round"),
                    "source_row_key": source_key,
                    "source_group": full.get("source_group"),
                    "case_id": full.get("case_id"),
                    "object": full.get("object"),
                    "prompt_variant": full.get("prompt_variant"),
                    "source_combo_key": full.get("source_combo_key"),
                    "source_edit_mode": full.get("source_edit_mode"),
                    "candidate_key": candidate,
                    "candidate_is_phase851_min_cut": candidate_is_phase851,
                    "label": label,
                    "original_class_blocker_count": int(orig_blockers),
                    "full_class_blocker_count": int(full_blockers),
                    "without_candidate_class_blocker_count": None if without_blockers is None else int(without_blockers),
                    "candidate_only_class_blocker_count": None if only_blockers is None else int(only_blockers),
                    "full_minus_original_blockers": int(full_blockers - orig_blockers),
                    "without_minus_full_blockers": None if rebound is None else int(rebound),
                    "original_class_best_target_rank": int(orig_rank),
                    "full_class_best_target_rank": int(full_rank),
                    "without_candidate_class_best_target_rank": None if without_rank is None else int(without_rank),
                    "candidate_only_class_best_target_rank": None if only_rank is None else int(only_rank),
                    "full_minus_original_rank": int(full_rank - orig_rank),
                    "without_minus_full_rank": None if rank_loss is None else int(rank_loss),
                    "full_answer_class_closure": bool(full.get("answer_class_closure")),
                    "full_strict_closure": bool(full.get("strict_closure")),
                    "full_class_top_blocker_token": full.get("class_top_blocker_token"),
                    "without_class_top_blocker_token": None if without is None else without.get("class_top_blocker_token"),
                }
            )
    return out


def compact_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "strict_closure": sum(1 for row in rows if row.get("strict_closure")),
        "answer_class_closure": sum(1 for row in rows if row.get("answer_class_closure")),
        "mean_strict_blocker_count": mean([finite(row.get("strict_blocker_count")) for row in rows]),
        "mean_class_blocker_count": mean([finite(row.get("class_blocker_count")) for row in rows]),
        "mean_strict_best_target_rank": mean([finite(row.get("strict_best_target_rank")) for row in rows]),
        "mean_class_best_target_rank": mean([finite(row.get("class_best_target_rank")) for row in rows]),
        "mean_class_minus_object_logit": mean(
            [finite(row.get("class_minus_object_logit")) for row in rows if row.get("class_minus_object_logit") is not None]
        ),
        "top_class_blocker_roles": dict(Counter(str(row.get("class_top_blocker_role")) for row in rows)),
    }


def transition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_row_key"))].append(row)
    full_rows = []
    improved_blockers = 0
    improved_rank = 0
    worsened_blockers = 0
    target_fail_rows = 0
    target_fail_class_closure = 0
    target_fail_strict_closure = 0
    for group in by_source.values():
        original = next((row for row in group if row.get("condition_type") == "original"), None)
        full = next((row for row in group if row.get("condition_type") == "full_combo"), None)
        if not original or not full:
            continue
        full_rows.append(full)
        orig_blockers = finite(original.get("class_blocker_count"))
        full_blockers = finite(full.get("class_blocker_count"))
        orig_rank = finite(original.get("class_best_target_rank"), 999999.0)
        full_rank = finite(full.get("class_best_target_rank"), 999999.0)
        improved_blockers += 1 if full_blockers < orig_blockers else 0
        worsened_blockers += 1 if full_blockers > orig_blockers else 0
        improved_rank += 1 if full_rank < orig_rank else 0
        if full.get("source_group") == "strong_target_not_exact":
            target_fail_rows += 1
            target_fail_class_closure += 1 if full.get("answer_class_closure") else 0
            target_fail_strict_closure += 1 if full.get("strict_closure") else 0
    return {
        "full_combo_rows": len(full_rows),
        "full_combo_improved_class_blockers": improved_blockers,
        "full_combo_worsened_class_blockers": worsened_blockers,
        "full_combo_improved_class_rank": improved_rank,
        "full_combo_answer_class_closure": sum(1 for row in full_rows if row.get("answer_class_closure")),
        "full_combo_strict_closure": sum(1 for row in full_rows if row.get("strict_closure")),
        "strong_target_not_exact_rows": target_fail_rows,
        "strong_target_not_exact_answer_class_closure": target_fail_class_closure,
        "strong_target_not_exact_strict_closure": target_fail_strict_closure,
        "full_combo_top_blocker_roles": dict(Counter(str(row.get("class_top_blocker_role")) for row in full_rows)),
    }


def min_cut_summary(edge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "edge_rows": len(edge_rows),
        "labels": dict(Counter(str(row.get("label")) for row in edge_rows)),
        "necessary_blocker_reducer": sum(1 for row in edge_rows if row.get("label") == "necessary_blocker_reducer"),
        "single_sufficient_partial_reducer": sum(
            1 for row in edge_rows if row.get("label") == "single_sufficient_partial_reducer"
        ),
        "phase851_candidate_edges": sum(1 for row in edge_rows if row.get("candidate_is_phase851_min_cut")),
        "necessary_phase851_candidate_edges": sum(
            1
            for row in edge_rows
            if row.get("candidate_is_phase851_min_cut") and row.get("label") == "necessary_blocker_reducer"
        ),
    }


def summarize(
    args: argparse.Namespace,
    attn_impl: str | None,
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    skipped: bool,
    skip_reason: str | None,
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition_type"))].append(row)
        by_group[str(row.get("source_group"))].append(row)
    return {
        "phase": PHASE,
        "title": "Full-Vocabulary Blocker Field and Min-Cut Causal Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_round": args.source_round,
        "phase851_round": args.phase851_round,
        "skipped_model_load": skipped,
        "skip_reason": skip_reason,
        "n_source_rows": len(source_rows),
        "source_groups": dict(Counter(classify_source_row(row, float(args.interaction_threshold)) for row in source_rows)),
        "n_rows": len(rows),
        "n_edge_rows": len(edge_rows),
        "condition_summary": {key: compact_condition(group) for key, group in sorted(by_condition.items())},
        "source_group_condition_summary": {
            key: compact_condition([row for row in group if row.get("condition_type") == "full_combo"])
            for key, group in sorted(by_group.items())
        },
        "transition_summary": transition_summary(rows),
        "min_cut_summary": min_cut_summary(edge_rows),
        "top_edge_rows": sorted(
            edge_rows,
            key=lambda row: (
                int(row.get("label") == "necessary_blocker_reducer"),
                abs(finite(row.get("without_minus_full_blockers"))),
                -finite(row.get("full_class_blocker_count")),
            ),
            reverse=True,
        )[:80],
        "boundary": (
            "This phase audits full-vocabulary blockers and leave-one-gear min-cut behavior for Phase 853 strong edges. "
            "It does not by itself prove rollout closure or large-model invariance."
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
        f"# Phase 854 Full-Vocabulary Blocker Field and Min-Cut Validation ({payload['round']})",
        "",
        "- Source: Phase 853 strong / control interaction rows.",
        "- Boundary: first-step full-vocabulary blocker audit; no rollout closure claim.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | source rows | full rows | class closure | strict closure | improved blockers | worsened blockers | min-cut necessary | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        transition = data.get("transition_summary") or {}
        mincut = data.get("min_cut_summary") or {}
        lines.append(
            f"| {model_name} | {data.get('n_source_rows', 0)} | {transition.get('full_combo_rows', 0)} | "
            f"{transition.get('full_combo_answer_class_closure', 0)} | {transition.get('full_combo_strict_closure', 0)} | "
            f"{transition.get('full_combo_improved_class_blockers', 0)} | {transition.get('full_combo_worsened_class_blockers', 0)} | "
            f"{mincut.get('necessary_blocker_reducer', 0)} | "
            f"`{json.dumps(mincut.get('labels') or {}, ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Condition Means",
        "",
        "| model | condition | n | answer class closure | strict closure | mean class blockers | mean class rank | mean class-object logit |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for condition in ["original", "full_combo", "candidate_only", "without_candidate"]:
            stats = (data.get("condition_summary") or {}).get(condition) or {}
            lines.append(
                f"| {model_name} | `{condition}` | {stats.get('n', 0)} | {stats.get('answer_class_closure', 0)} | "
                f"{stats.get('strict_closure', 0)} | {fmt(stats.get('mean_class_blocker_count'))} | "
                f"{fmt(stats.get('mean_class_best_target_rank'))} | {fmt(stats.get('mean_class_minus_object_logit'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase854_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = p846.read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase854_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase854_cross_model_summary.md", payload)
    print(json.dumps({"status": payload["status"], "round": round_name, "models": payload["models"]}, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--phase851-round", default="confirm")
    parser.add_argument("--interaction-threshold", type=float, default=0.5)
    parser.add_argument("--max-target-fail-rows", type=int, default=8)
    parser.add_argument("--max-strong-non-target-rows", type=int, default=4)
    parser.add_argument("--max-control-rows", type=int, default=4)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--max-candidates-per-row", type=int, default=2)
    parser.add_argument("--topk-blockers", type=int, default=30)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    eval_model(args)


if __name__ == "__main__":
    main()
