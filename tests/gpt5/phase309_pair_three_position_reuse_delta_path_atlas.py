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
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase268_attention_mlp_continuation_path_attribution as p268  # noqa: E402
import phase301_semantic_reuse_delta_case_bank as p301  # noqa: E402
import phase305_internal_semantic_physical_path_probe as p305  # noqa: E402
import phase307_three_position_semantic_path_trace as p307  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = "Phase309"
SCHEMA_VERSION = "2.36.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "pair_three_position_reuse_delta_path_atlas"

RESULT_ROOT = ROOT / "tests/gpt5/result"
OUT = RESULT_ROOT / "phase309_pair_three_position_reuse_delta_path_atlas"
V2 = RESULT_ROOT / "pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"

SHARED_PAIRS = [
    ("lemon", "lime", "citrus"),
    ("orange", "lemon", "citrus"),
    ("apple", "pear", "tree_fruit"),
    ("banana", "mango", "tropical"),
    ("strawberry", "blueberry", "berry_like"),
]

DELTA_PAIRS = [
    ("apple", "banana", "shape_taste_delta"),
    ("fruit", "chair", "food_furniture_delta"),
    ("lemon", "knife", "citrus_tool_delta"),
    ("blueberry", "stone", "berry_mineral_delta"),
    ("orange", "stone", "fruit_mineral_delta"),
]

ATTRIBUTE_TYPES = ["category", "subclass", "color", "taste", "use"]


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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def cosine(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        return 0.0
    n = min(len(xs), len(ys))
    a = xs[:n]
    b = ys[:n]
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return round(dot / (na * nb), 6)


def objects() -> dict[str, dict[str, Any]]:
    by_id = {row["object_id"]: row for row in p301.OBJECTS}
    by_id["fruit"] = {
        "object_id": "fruit",
        "label": "fruit",
        "category": "food",
        "subclass": "plant_food",
        "color": "varied",
        "shape": "varied",
        "taste": "sweet",
        "texture": "varied",
        "part": "flesh",
        "use": "eating",
        "features": ["entity", "plant", "food", "fruit"],
    }
    return by_id


def prompt_for(obj: dict[str, Any], attr: str) -> str:
    templates = {row["prompt_type"]: row for row in p301.PROMPT_TYPES}
    if attr in templates:
        return str(templates[attr]["template"]).format(**obj)
    return f"A {obj['label']} is associated with ___. Answer briefly."


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    toks = tokenizer.encode(str(text), add_special_tokens=False)
    return [int(x) for x in toks]


def find_span(haystack: list[int], needle: list[int]) -> tuple[int | None, int | None]:
    if not needle or len(needle) > len(haystack):
        return None, None
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i, i + len(needle) - 1
    return None, None


def locate_positions_with_span(tokenizer: Any, case: dict[str, Any], prompt: str, last_pos: int) -> dict[str, dict[str, Any]]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    ids = [int(x) for x in encoded["input_ids"][0].tolist()]
    object_label = str(case.get("object_label") or case.get("object_id") or "")
    attr = str(case.get("attribute_type") or "")
    out: dict[str, dict[str, Any]] = {}

    object_match = (None, None)
    object_surface = ""
    for candidate in [object_label, " " + object_label, str(case.get("object_id") or ""), " " + str(case.get("object_id") or "")]:
        object_match = find_span(ids, encode_ids(tokenizer, candidate))
        if object_match[0] is not None:
            object_surface = candidate
            break
    if object_match[1] is None:
        out["object"] = {
            "token_position": max(0, min(last_pos, 1)),
            "token_start": None,
            "token_end": None,
            "match_confidence": 0.25,
            "match_surface": object_label,
            "multi_token_pooling_method": "fallback_single_position",
        }
    else:
        out["object"] = {
            "token_position": int(object_match[1]),
            "token_start": int(object_match[0]),
            "token_end": int(object_match[1]),
            "match_confidence": 1.0,
            "match_surface": object_surface.strip(),
            "multi_token_pooling_method": "last_token_of_span",
        }

    query_terms = [attr, str(case.get("semantic_field") or ""), str(case.get("prompt_type") or "")]
    query_terms += {
        "category": ["type", " type"],
        "subclass": ["associated", "categories"],
        "color": ["color", "usual color"],
        "taste": ["tastes", "mostly"],
        "use": ["use", "common use"],
    }.get(attr, [])
    query_match = (None, None)
    query_surface = ""
    for candidate in query_terms:
        query_match = find_span(ids, encode_ids(tokenizer, candidate))
        if query_match[0] is not None:
            query_surface = candidate
            break
    if query_match[1] is None:
        out["query"] = {
            "token_position": max(0, last_pos - 2),
            "token_start": None,
            "token_end": None,
            "match_confidence": 0.25,
            "match_surface": attr,
            "multi_token_pooling_method": "fallback_last_minus_2",
        }
    else:
        out["query"] = {
            "token_position": int(query_match[1]),
            "token_start": int(query_match[0]),
            "token_end": int(query_match[1]),
            "match_confidence": 1.0,
            "match_surface": query_surface.strip(),
            "multi_token_pooling_method": "last_token_of_span",
        }

    out["last"] = {
        "token_position": int(last_pos),
        "token_start": int(last_pos),
        "token_end": int(last_pos),
        "match_confidence": 1.0,
        "match_surface": "<last>",
        "multi_token_pooling_method": "last_context_token",
    }
    return out


def make_pairs(pairs_per_group: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group, pairs in [("shared_backbone", SHARED_PAIRS[:pairs_per_group]), ("delta_control", DELTA_PAIRS[:pairs_per_group])]:
        for left, right, relation in pairs:
            selected.append(
                {
                    "pair_id": f"{left}__{right}",
                    "left_object_id": left,
                    "right_object_id": right,
                    "pair_group": group,
                    "expected_relation": relation,
                }
            )
    return selected


def make_cases(model: str, pairs_per_group: int, attrs: list[str]) -> list[dict[str, Any]]:
    by_id = objects()
    cases: list[dict[str, Any]] = []
    for pair in make_pairs(pairs_per_group):
        for side in ["left", "right"]:
            obj = by_id[str(pair[f"{side}_object_id"])]
            for attr in attrs:
                target = str(obj.get(attr, "unknown"))
                cases.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "model": model,
                        "case_id": f"phase309:{model}:{pair['pair_id']}:{side}:{attr}",
                        "pair_id": pair["pair_id"],
                        "pair_group": pair["pair_group"],
                        "expected_relation": pair["expected_relation"],
                        "pair_side": side,
                        "object_id": obj["object_id"],
                        "object_label": obj["label"],
                        "contrast_object_id": pair["right_object_id"] if side == "left" else pair["left_object_id"],
                        "attribute_type": attr,
                        "prompt_type": attr,
                        "semantic_field": attr,
                        "prompt": prompt_for(obj, attr),
                        "target": target,
                        "target_aliases": p301.aliases(target),
                    }
                )
    return cases


def semantic_groups(tokenizer: Any, case: dict[str, Any]) -> tuple[list[int], list[int], list[str], list[str]]:
    target_aliases = [str(x) for x in case.get("target_aliases") or [case.get("target", "")]]
    attr = str(case.get("attribute_type") or "unknown")
    distractors = [x for x in p305.DISTRACTORS.get(attr, []) if x not in target_aliases]
    if not distractors:
        distractors = ["fruit", "vegetable", "tool", "red", "yellow", "sweet", "sour"]
    return p305.token_ids(tokenizer, target_aliases), p305.token_ids(tokenizer, distractors), target_aliases, distractors


def trace_case(model_obj: Any, tokenizer: Any, device: torch.device, case: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = str(case["prompt"])
    captured, _final_logits, last_pos = p268.capture_components(model_obj, tokenizer, device, prompt)
    positions = locate_positions_with_span(tokenizer, case, prompt, last_pos)
    target_ids, distractor_ids, target_aliases, distractors = semantic_groups(tokenizer, case)
    final_norm = p268.get_final_norm(model_obj)
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for role, meta in positions.items():
        rows, summary = p307.decompose_position(
            model_obj,
            final_norm,
            captured,
            int(meta["token_position"]),
            case,
            role,
            target_ids,
            distractor_ids,
            target_aliases,
            distractors,
        )
        for row in rows:
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "atlas_record_type": "pair_three_position_component",
                    "pair_id": case["pair_id"],
                    "pair_group": case["pair_group"],
                    "pair_side": case["pair_side"],
                    "expected_relation": case["expected_relation"],
                    "token_start": meta["token_start"],
                    "token_end": meta["token_end"],
                    "token_match_confidence": meta["match_confidence"],
                    "match_surface": meta["match_surface"],
                    "multi_token_pooling_method": meta["multi_token_pooling_method"],
                }
            )
        summary.update(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "atlas_record_type": "pair_three_position_summary",
                "pair_id": case["pair_id"],
                "pair_group": case["pair_group"],
                "pair_side": case["pair_side"],
                "expected_relation": case["expected_relation"],
                "token_start": meta["token_start"],
                "token_end": meta["token_end"],
                "token_match_confidence": meta["match_confidence"],
                "match_surface": meta["match_surface"],
                "multi_token_pooling_method": meta["multi_token_pooling_method"],
            }
        )
        component_rows.extend(rows)
        summary_rows.append(summary)
    return component_rows, summary_rows


def profile(rows: list[dict[str, Any]], component: str) -> list[float]:
    field = {
        "attention": "delta_attn_semantic_margin",
        "mlp": "delta_mlp_semantic_margin",
        "residual": "delta_residual_semantic_margin",
    }[component]
    return [safe_float(r.get(field)) for r in sorted(rows, key=lambda r: int(r.get("layer_index", 0)))]


def build_pair_rows(component_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        by_key[
            (
                str(row.get("model")),
                str(row.get("pair_id")),
                str(row.get("pair_side")),
                str(row.get("attribute_type")),
                str(row.get("position_role")),
            )
        ].append(row)

    summary_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in summary_rows:
        summary_key[
            (
                str(row.get("model")),
                str(row.get("pair_id")),
                str(row.get("pair_side")),
                str(row.get("attribute_type")),
                str(row.get("position_role")),
            )
        ] = row

    pair_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in summary_rows:
        pair_meta[(str(row.get("model")), str(row.get("pair_id")), str(row.get("attribute_type")))] = row

    matrix_rows: list[dict[str, Any]] = []
    component_similarity_rows: list[dict[str, Any]] = []
    pair_path_rows: list[dict[str, Any]] = []
    components = ["attention", "mlp", "residual"]
    positions = ["object", "query", "last"]
    for model, pair_id, attr in sorted(pair_meta):
        meta = pair_meta[(model, pair_id, attr)]
        per_position_scores: list[float] = []
        per_position_delta: list[float] = []
        dominant_components: list[str] = []
        for position in positions:
            comp_scores: dict[str, float] = {}
            for component in components:
                left = by_key.get((model, pair_id, "left", attr, position), [])
                right = by_key.get((model, pair_id, "right", attr, position), [])
                sim = cosine(profile(left, component), profile(right, component))
                reuse_score = round((sim + 1.0) / 2.0, 6)
                delta_score = round(1.0 - reuse_score, 6)
                left_summary = summary_key.get((model, pair_id, "left", attr, position), {})
                right_summary = summary_key.get((model, pair_id, "right", attr, position), {})
                target_agreement = 0.0
                if left_summary and right_summary:
                    winners = [
                        str(left_summary.get("final_layer_out_semantic_winner")),
                        str(right_summary.get("final_layer_out_semantic_winner")),
                    ]
                    target_agreement = winners.count("target") / 2.0
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "atlas_record_type": "pair_three_position_matrix",
                    "model": model,
                    "pair_id": pair_id,
                    "pair_group": meta.get("pair_group"),
                    "expected_relation": meta.get("expected_relation"),
                    "attribute_type": attr,
                    "position_role": position,
                    "component": component,
                    "path_cosine_similarity": sim,
                    "reuse_path_score": reuse_score,
                    "delta_path_score": delta_score,
                    "target_agreement_rate": round(target_agreement, 6),
                    "left_layers": len(left),
                    "right_layers": len(right),
                    "left_match_confidence": left_summary.get("token_match_confidence"),
                    "right_match_confidence": right_summary.get("token_match_confidence"),
                    "left_winner": left_summary.get("final_layer_out_semantic_winner"),
                    "right_winner": right_summary.get("final_layer_out_semantic_winner"),
                }
                matrix_rows.append(row)
                component_similarity_rows.append(row)
                comp_scores[component] = reuse_score
            dominant = max(comp_scores.items(), key=lambda kv: kv[1])[0] if comp_scores else "unknown"
            dominant_components.append(dominant)
            per_position_scores.append(comp_scores.get(dominant, 0.0))
            per_position_delta.append(1.0 - comp_scores.get(dominant, 0.0))
        path_type = "->".join(dominant_components)
        pair_path_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "atlas_record_type": "pair_path_route",
                "model": model,
                "pair_id": pair_id,
                "pair_group": meta.get("pair_group"),
                "expected_relation": meta.get("expected_relation"),
                "attribute_type": attr,
                "dominant_pair_route_type": path_type,
                "mean_dominant_reuse_path_score": mean_safe(per_position_scores),
                "mean_dominant_delta_path_score": mean_safe(per_position_delta),
                "object_component": dominant_components[0] if len(dominant_components) > 0 else None,
                "query_component": dominant_components[1] if len(dominant_components) > 1 else None,
                "last_component": dominant_components[2] if len(dominant_components) > 2 else None,
            }
        )
    return pair_path_rows, matrix_rows, component_similarity_rows


def summarize(model: str, cases: list[dict[str, Any]], component_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], matrix_rows: list[dict[str, Any]], missing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group = defaultdict(list)
    for row in pair_rows:
        by_group[str(row.get("pair_group"))].append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "model": model,
        "selected_cases": len(cases),
        "component_rows": len(component_rows),
        "summary_rows": len(summary_rows),
        "pair_path_rows": len(pair_rows),
        "pair_matrix_rows": len(matrix_rows),
        "missing_rows": len(missing_rows),
        "pair_group_counts": dict(Counter(str(r.get("pair_group")) for r in pair_rows)),
        "attribute_counts": dict(Counter(str(r.get("attribute_type")) for r in pair_rows)),
        "route_type_counts": dict(Counter(str(r.get("dominant_pair_route_type")) for r in pair_rows)),
        "mean_reuse_by_pair_group": {k: mean_safe([safe_float(r.get("mean_dominant_reuse_path_score")) for r in vals]) for k, vals in sorted(by_group.items())},
        "mean_delta_by_pair_group": {k: mean_safe([safe_float(r.get("mean_dominant_delta_path_score")) for r in vals]) for k, vals in sorted(by_group.items())},
    }


def run_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    out_dir = OUT / args.round_name
    attrs = [x.strip() for x in args.attributes.split(",") if x.strip()]
    cases = make_cases(model, args.pairs_per_group, attrs)
    model_obj = tokenizer = None
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, case in enumerate(cases, 1):
            try:
                rows, summaries = trace_case(model_obj, tokenizer, device, case)
                component_rows.extend(rows)
                summary_rows.extend(summaries)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": model,
                        "case_id": case.get("case_id"),
                        "pair_id": case.get("pair_id"),
                        "attribute_type": case.get("attribute_type"),
                        "reason": repr(exc),
                    }
                )
            print(f"{model}: pair three-position traced {idx}/{len(cases)}", flush=True)
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
    pair_rows, matrix_rows, component_similarity_rows = build_pair_rows(component_rows, summary_rows)
    payload = summarize(model, cases, component_rows, summary_rows, pair_rows, matrix_rows, missing_rows)
    write_json(out_dir / f"phase309_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase309_{model}_component_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase309_{model}_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase309_{model}_pair_path_rows.jsonl", pair_rows)
    write_jsonl(out_dir / f"phase309_{model}_pair_matrix_rows.jsonl", matrix_rows)
    write_jsonl(out_dir / f"phase309_{model}_component_similarity_rows.jsonl", component_similarity_rows)
    write_jsonl(out_dir / f"phase309_{model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def collect(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    for model in MODELS:
        summary_path = out_dir / f"phase309_{model}_summary.json"
        if summary_path.exists():
            model_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        component_rows.extend(read_jsonl(out_dir / f"phase309_{model}_component_rows.jsonl"))
        summary_rows.extend(read_jsonl(out_dir / f"phase309_{model}_summary_rows.jsonl"))
        pair_rows.extend(read_jsonl(out_dir / f"phase309_{model}_pair_path_rows.jsonl"))
        matrix_rows.extend(read_jsonl(out_dir / f"phase309_{model}_pair_matrix_rows.jsonl"))
        sim_rows.extend(read_jsonl(out_dir / f"phase309_{model}_component_similarity_rows.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase309_{model}_missing_rows.jsonl"))

    by_group = defaultdict(list)
    by_pos = defaultdict(list)
    for row in matrix_rows:
        by_group[str(row.get("pair_group"))].append(row)
        by_pos[str(row.get("position_role"))].append(row)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "round_name": round_name,
        "model_summaries": model_summaries,
        "component_rows": len(component_rows),
        "summary_rows": len(summary_rows),
        "pair_path_rows": len(pair_rows),
        "pair_matrix_rows": len(matrix_rows),
        "component_similarity_rows": len(sim_rows),
        "missing_rows": len(missing_rows),
        "models": MODELS,
        "pair_group_counts": dict(Counter(str(r.get("pair_group")) for r in pair_rows)),
        "attribute_counts": dict(Counter(str(r.get("attribute_type")) for r in pair_rows)),
        "route_type_counts": dict(Counter(str(r.get("dominant_pair_route_type")) for r in pair_rows)),
        "mean_reuse_by_pair_group": {k: mean_safe([safe_float(r.get("reuse_path_score")) for r in vals]) for k, vals in sorted(by_group.items())},
        "mean_delta_by_pair_group": {k: mean_safe([safe_float(r.get("delta_path_score")) for r in vals]) for k, vals in sorted(by_group.items())},
        "mean_reuse_by_position": {k: mean_safe([safe_float(r.get("reuse_path_score")) for r in vals]) for k, vals in sorted(by_pos.items())},
        "mean_delta_by_position": {k: mean_safe([safe_float(r.get("delta_path_score")) for r in vals]) for k, vals in sorted(by_pos.items())},
        "token_match_confidence_mean": mean_safe(
            [
                safe_float(r.get("token_match_confidence"))
                for r in summary_rows
                if r.get("token_match_confidence") is not None
            ]
        ),
    }
    write_json(out_dir / "phase309_reuse_delta_path_summary.json", payload)
    write_jsonl(out_dir / "phase309_pair_component_rows.jsonl", component_rows)
    write_jsonl(out_dir / "phase309_pair_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / "phase309_shared_delta_pair_path_rows.jsonl", pair_rows)
    write_jsonl(out_dir / "phase309_three_position_pair_matrix_rows.jsonl", matrix_rows)
    write_jsonl(out_dir / "phase309_component_similarity_rows.jsonl", sim_rows)
    write_jsonl(out_dir / "phase309_missing_rows.jsonl", missing_rows)
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase309_reuse_delta_path_summary.json", payload)
        write_jsonl(base / "phase309_pair_component_rows.jsonl", component_rows)
        write_jsonl(base / "phase309_pair_summary_rows.jsonl", summary_rows)
        write_jsonl(base / "phase309_shared_delta_pair_path_rows.jsonl", pair_rows)
        write_jsonl(base / "phase309_three_position_pair_matrix_rows.jsonl", matrix_rows)
        write_jsonl(base / "phase309_component_similarity_rows.jsonl", sim_rows)
        write_jsonl(base / "phase309_missing_rows.jsonl", missing_rows)
        write_report(base / "phase309_reuse_delta_path_report.md", payload, matrix_rows, pair_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def grouped_mean(rows: list[dict[str, Any]], keys: list[str], value_field: str) -> list[tuple[tuple[str, ...], float, int]]:
    buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(k)) for k in keys)].append(safe_float(row.get(value_field)))
    return [(key, mean_safe(vals), len(vals)) for key, vals in sorted(buckets.items())]


def write_report(path: Path, payload: dict[str, Any], matrix_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase309 Shared Backbone and Delta Pair Three-Position Path Atlas",
        "",
        "## Summary",
        "",
        f"- component_rows: {payload['component_rows']}",
        f"- summary_rows: {payload['summary_rows']}",
        f"- pair_path_rows: {payload['pair_path_rows']}",
        f"- pair_matrix_rows: {payload['pair_matrix_rows']}",
        f"- missing_rows: {payload['missing_rows']}",
        f"- token_match_confidence_mean: {payload['token_match_confidence_mean']}",
        "",
        "## Mean Reuse By Pair Group",
        "",
    ]
    for group, value in payload["mean_reuse_by_pair_group"].items():
        lines.append(f"- {group}: reuse={value}, delta={payload['mean_delta_by_pair_group'].get(group)}")
    lines += ["", "## Mean Reuse By Position", ""]
    for pos, value in payload["mean_reuse_by_position"].items():
        lines.append(f"- {pos}: reuse={value}, delta={payload['mean_delta_by_position'].get(pos)}")
    lines += ["", "## Pair Group x Position", ""]
    for key, value, count in grouped_mean(matrix_rows, ["pair_group", "position_role"], "reuse_path_score"):
        lines.append(f"- {key[0]} / {key[1]}: reuse={value}, n={count}")
    lines += ["", "## Pair Group x Attribute", ""]
    for key, value, count in grouped_mean(matrix_rows, ["pair_group", "attribute_type"], "reuse_path_score"):
        lines.append(f"- {key[0]} / {key[1]}: reuse={value}, n={count}")
    lines += ["", "## Component", ""]
    for key, value, count in grouped_mean(matrix_rows, ["component"], "reuse_path_score"):
        lines.append(f"- {key[0]}: reuse={value}, n={count}")
    lines += ["", "## Top Route Types", ""]
    for route, count in Counter(str(row.get("dominant_pair_route_type")) for row in pair_rows).most_common(12):
        lines.append(f"- {route}: {count}")
    lines += [
        "",
        "## Caution",
        "",
        "This is an observational path-signature atlas. It compares per-layer component margin profiles, not direct causal necessity.",
        "The current linear target-distractor readout direction is a probe, not a final mechanism formula.",
        "",
    ]
    write_json(path.with_suffix(".summary.json"), payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--pairs-per-group", type=int, default=5)
    parser.add_argument("--attributes", default="category,subclass,color,taste,use")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        collect(args.round_name)
        return
    if args.model:
        run_model(args, args.model)
        return
    for model in MODELS:
        run_model(args, model)
    collect(args.round_name)


if __name__ == "__main__":
    main()
