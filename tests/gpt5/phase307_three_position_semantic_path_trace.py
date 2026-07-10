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
import phase305_internal_semantic_physical_path_probe as p305  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = "Phase307"
SCHEMA_VERSION = "2.34.0"
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase307_three_position_semantic_path_trace"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "three_position_semantic_path_trace"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    toks = tokenizer.encode(str(text), add_special_tokens=False)
    return [int(x) for x in toks]


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i + len(needle) - 1
    return None


def locate_positions(tokenizer: Any, case: dict[str, Any], prompt: str, last_pos: int) -> dict[str, int]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    ids = [int(x) for x in encoded["input_ids"][0].tolist()]
    object_label = str(case.get("object_label") or case.get("object_id") or "")
    attr = str(case.get("attribute_type") or case.get("semantic_field") or "")
    object_pos = None
    for candidate in [object_label, " " + object_label, str(case.get("object_id") or ""), " " + str(case.get("object_id") or "")]:
        object_pos = find_subsequence(ids, encode_ids(tokenizer, candidate))
        if object_pos is not None:
            break
    query_pos = None
    query_terms = [attr, str(case.get("semantic_field") or ""), str(case.get("prompt_type") or "")]
    if attr == "shared":
        query_terms += ["both", " both"]
    if attr == "difference":
        query_terms += ["Compared", " associated", "more"]
    for candidate in query_terms:
        query_pos = find_subsequence(ids, encode_ids(tokenizer, candidate))
        if query_pos is not None:
            break
    return {
        "object": int(object_pos if object_pos is not None else max(0, min(last_pos, 1))),
        "query": int(query_pos if query_pos is not None else max(0, last_pos - 2)),
        "last": int(last_pos),
    }


def semantic_groups(tokenizer: Any, case: dict[str, Any]) -> tuple[list[int], list[int], list[str], list[str]]:
    target_aliases = [str(x) for x in case.get("target_aliases") or [case.get("target", "")]]
    attr = str(case.get("attribute_type") or case.get("semantic_field") or "unknown")
    distractors = [x for x in p305.DISTRACTORS.get(attr, []) if x not in target_aliases]
    if not distractors:
        distractors = ["fruit", "vegetable", "tool", "red", "yellow", "sweet", "sour"]
    return p305.token_ids(tokenizer, target_aliases), p305.token_ids(tokenizer, distractors), target_aliases, distractors


def decompose_position(
    model_obj: Any,
    final_norm: Any,
    captured: dict[int, dict[str, torch.Tensor]],
    pos: int,
    case: dict[str, Any],
    position_role: str,
    target_ids: list[int],
    distractor_ids: list[int],
    target_aliases: list[str],
    distractors: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": case["model"],
        "case_id": case["case_id"],
        "case_type": case.get("case_type"),
        "object_id": case.get("object_id"),
        "contrast_object_id": case.get("contrast_object_id"),
        "category_id": case.get("category_id"),
        "subclass_id": case.get("subclass_id"),
        "attribute_type": case.get("attribute_type"),
        "prompt_type": case.get("prompt_type"),
        "target": case.get("target"),
        "target_aliases": target_aliases,
        "distractor_aliases": distractors,
        "position_role": position_role,
        "token_position": pos,
    }
    rows: list[dict[str, Any]] = []
    for layer_idx in sorted(captured):
        comp = captured[layer_idx]
        h0 = p268.tensor_at_pos(comp.get("layer_input"), pos)
        attn = p268.tensor_at_pos(comp.get("attn_out"), pos)
        mlp = p268.tensor_at_pos(comp.get("mlp_out"), pos)
        layer_out = p268.tensor_at_pos(comp.get("layer_out"), pos)
        if h0 is None or layer_out is None:
            continue
        h0_read = p305.semantic_readout(p268.project_state(model_obj, final_norm, h0), target_ids, distractor_ids)
        h_attn = h0 + attn if attn is not None else h0
        h_attn_read = p305.semantic_readout(p268.project_state(model_obj, final_norm, h_attn), target_ids, distractor_ids)
        h_mlp = h_attn + mlp if mlp is not None else h_attn
        h_mlp_read = p305.semantic_readout(p268.project_state(model_obj, final_norm, h_mlp), target_ids, distractor_ids)
        h_out_read = p305.semantic_readout(p268.project_state(model_obj, final_norm, layer_out), target_ids, distractor_ids)
        m0 = safe_float(h0_read["semantic_margin"])
        ma = safe_float(h_attn_read["semantic_margin"])
        mm = safe_float(h_mlp_read["semantic_margin"])
        mo = safe_float(h_out_read["semantic_margin"])
        rows.append(
            {
                **base,
                "three_position_component_id": f"phase307:component:{case['model']}:{case['case_id']}:{position_role}:L{layer_idx}",
                "layer_index": layer_idx,
                "input_semantic_margin": round(m0, 6),
                "after_attn_semantic_margin": round(ma, 6),
                "after_mlp_semantic_margin": round(mm, 6),
                "layer_out_semantic_margin": round(mo, 6),
                "delta_attn_semantic_margin": round(ma - m0, 6),
                "delta_mlp_semantic_margin": round(mm - ma, 6),
                "delta_residual_semantic_margin": round(mo - mm, 6),
                "layer_out_semantic_winner": h_out_read["semantic_winner"],
            }
        )
    strongest_attn = max(rows, key=lambda r: safe_float(r["delta_attn_semantic_margin"]), default={})
    strongest_mlp = max(rows, key=lambda r: safe_float(r["delta_mlp_semantic_margin"]), default={})
    strongest_resid = max(rows, key=lambda r: safe_float(r["delta_residual_semantic_margin"]), default={})
    summary = {
        **base,
        "three_position_summary_id": f"phase307:summary:{case['model']}:{case['case_id']}:{position_role}",
        "layers_observed": len(rows),
        "final_layer_out_semantic_margin": rows[-1]["layer_out_semantic_margin"] if rows else None,
        "final_layer_out_semantic_winner": rows[-1]["layer_out_semantic_winner"] if rows else None,
        "sum_positive_attn_semantic_delta": round(sum(max(0.0, safe_float(r["delta_attn_semantic_margin"])) for r in rows), 6),
        "sum_positive_mlp_semantic_delta": round(sum(max(0.0, safe_float(r["delta_mlp_semantic_margin"])) for r in rows), 6),
        "sum_positive_residual_semantic_delta": round(sum(max(0.0, safe_float(r["delta_residual_semantic_margin"])) for r in rows), 6),
        "strongest_attn_layer": strongest_attn.get("layer_index"),
        "strongest_attn_delta": strongest_attn.get("delta_attn_semantic_margin"),
        "strongest_mlp_layer": strongest_mlp.get("layer_index"),
        "strongest_mlp_delta": strongest_mlp.get("delta_mlp_semantic_margin"),
        "strongest_residual_layer": strongest_resid.get("layer_index"),
        "strongest_residual_delta": strongest_resid.get("delta_residual_semantic_margin"),
    }
    positives = {
        "attention": safe_float(summary["sum_positive_attn_semantic_delta"]),
        "mlp": safe_float(summary["sum_positive_mlp_semantic_delta"]),
        "residual": safe_float(summary["sum_positive_residual_semantic_delta"]),
    }
    summary["dominant_positive_semantic_component"] = max(positives.items(), key=lambda kv: kv[1])[0]
    return rows, summary


def run_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    out_dir = OUT / args.round_name
    cases = p305.select_cases(model, args.cases_per_model)
    model_obj = tokenizer = None
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        final_norm = p268.get_final_norm(model_obj)
        for idx, case in enumerate(cases, 1):
            try:
                prompt = str(case["prompt"])
                captured, _final_logits, last_pos = p268.capture_components(model_obj, tokenizer, device, prompt)
                positions = locate_positions(tokenizer, case, prompt, last_pos)
                target_ids, distractor_ids, target_aliases, distractors = semantic_groups(tokenizer, case)
                for role, pos in positions.items():
                    rows, summary = decompose_position(
                        model_obj, final_norm, captured, pos, case, role, target_ids, distractor_ids, target_aliases, distractors
                    )
                    component_rows.extend(rows)
                    summary_rows.append(summary)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append({"schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(), "model": model, "case_id": case.get("case_id"), "reason": repr(exc)})
            print(f"{model}: three-position semantic traced {idx}/{len(cases)}", flush=True)
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
    payload = summarize_model(model, cases, summary_rows, missing_rows)
    write_json(out_dir / f"phase307_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase307_{model}_three_position_component_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase307_{model}_three_position_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase307_{model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_model(model: str, selected: list[dict[str, Any]], summaries: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "model": model,
        "selected_cases": len(selected),
        "three_position_summary_rows": len(summaries),
        "missing_rows": len(missing),
        "position_counts": dict(Counter(str(r.get("position_role")) for r in summaries)),
        "attribute_counts": dict(Counter(str(r.get("attribute_type")) for r in summaries)),
        "final_layer_out_winner_counts": dict(Counter(str(r.get("final_layer_out_semantic_winner")) for r in summaries)),
        "dominant_component_counts": dict(Counter(str(r.get("dominant_positive_semantic_component")) for r in summaries)),
        "mean_sum_positive_attn_semantic_delta": mean_safe([safe_float(r.get("sum_positive_attn_semantic_delta")) for r in summaries]),
        "mean_sum_positive_mlp_semantic_delta": mean_safe([safe_float(r.get("sum_positive_mlp_semantic_delta")) for r in summaries]),
        "mean_sum_positive_residual_semantic_delta": mean_safe([safe_float(r.get("sum_positive_residual_semantic_delta")) for r in summaries]),
    }


def collect(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    component: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    model_summaries = []
    for model in MODELS:
        model_summaries.append(read_json(out_dir / f"phase307_{model}_summary.json"))
        component.extend(read_jsonl(out_dir / f"phase307_{model}_three_position_component_rows.jsonl"))
        summaries.extend(read_jsonl(out_dir / f"phase307_{model}_three_position_summary_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase307_{model}_missing_rows.jsonl"))
    model_summaries = [s for s in model_summaries if s]
    by_position = defaultdict(list)
    for row in summaries:
        by_position[str(row.get("position_role"))].append(row)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "round_name": round_name,
        "model_summaries": model_summaries,
        "three_position_component_rows": len(component),
        "three_position_summary_rows": len(summaries),
        "missing_rows": len(missing),
        "position_counts": dict(Counter(str(r.get("position_role")) for r in summaries)),
        "attribute_counts": dict(Counter(str(r.get("attribute_type")) for r in summaries)),
        "dominant_component_counts": dict(Counter(str(r.get("dominant_positive_semantic_component")) for r in summaries)),
        "position_mean_attn_delta": {k: mean_safe([safe_float(r.get("sum_positive_attn_semantic_delta")) for r in vals]) for k, vals in sorted(by_position.items())},
        "position_mean_mlp_delta": {k: mean_safe([safe_float(r.get("sum_positive_mlp_semantic_delta")) for r in vals]) for k, vals in sorted(by_position.items())},
        "position_mean_residual_delta": {k: mean_safe([safe_float(r.get("sum_positive_residual_semantic_delta")) for r in vals]) for k, vals in sorted(by_position.items())},
    }
    write_json(out_dir / "phase307_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase307_three_position_component_rows.jsonl", component)
    write_jsonl(out_dir / "phase307_three_position_summary_rows.jsonl", summaries)
    write_jsonl(out_dir / "phase307_missing_rows.jsonl", missing)
    write_json(V2 / "phase307_cross_model_summary.json", payload)
    write_jsonl(V2 / "phase307_three_position_component_rows.jsonl", component)
    write_jsonl(V2 / "phase307_three_position_summary_rows.jsonl", summaries)
    write_jsonl(V2 / "phase307_missing_rows.jsonl", missing)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--cases-per-model", type=int, default=12)
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
