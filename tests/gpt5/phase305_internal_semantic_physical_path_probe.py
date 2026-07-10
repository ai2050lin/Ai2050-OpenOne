#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
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
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = "Phase305"
SCHEMA_VERSION = "2.32.0"
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase305_internal_semantic_physical_path_probe"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "internal_semantic_physical_path_probe"

DISTRACTORS = {
    "category": ["vegetable", "furniture", "tool", "mineral", "object"],
    "subclass": ["citrus", "berry", "tropical", "tree fruit", "stone fruit", "vegetable", "tool"],
    "color": ["red", "green", "yellow", "orange", "blue", "purple", "brown", "gray"],
    "shape": ["round", "long", "curved", "oval", "spiky", "irregular", "sharp"],
    "taste": ["sweet", "sour", "bitter", "starchy", "earthy", "inedible"],
    "texture": ["crisp", "soft", "juicy", "hard", "crunchy", "watery"],
    "part": ["peel", "core", "pit", "skin", "blade", "legs", "rind"],
    "use": ["juice", "pie", "smoothie", "seasoning", "sitting", "cutting", "building", "fries"],
    "shared": ["fruit", "citrus", "food", "plant", "tool", "furniture", "mineral"],
    "difference": ["yellow", "curved", "sweet", "food", "tool", "hard", "sour"],
}


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


def token_ids(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        for candidate in [str(text), " " + str(text)]:
            toks = tokenizer.encode(candidate, add_special_tokens=False)
            if toks:
                ids.append(int(toks[0]))
    return sorted(set(ids))


def max_score(logits: torch.Tensor, ids: list[int]) -> tuple[float, int]:
    valid = [int(x) for x in ids if 0 <= int(x) < logits.numel()]
    if not valid:
        return -1e30, -1
    idx = torch.tensor(valid, dtype=torch.long, device=logits.device)
    vals = logits[idx]
    pos = int(torch.argmax(vals).item())
    return float(vals[pos].item()), int(idx[pos].item())


def semantic_readout(logits: torch.Tensor, target_ids: list[int], distractor_ids: list[int]) -> dict[str, Any]:
    target, target_id = max_score(logits.detach().float().cpu(), target_ids)
    distractor, distractor_id = max_score(logits.detach().float().cpu(), distractor_ids)
    return {
        "target_semantic_logit": target,
        "distractor_semantic_logit": distractor,
        "semantic_margin": target - distractor,
        "target_token_id": target_id,
        "distractor_token_id": distractor_id,
        "semantic_winner": "target" if target >= distractor else "distractor",
    }


def case_bank() -> dict[str, dict[str, Any]]:
    return {str(r.get("case_id")): r for r in read_jsonl(V2 / "phase301_semantic_full_test_plan_rows.jsonl")}


def select_cases(model: str, limit: int) -> list[dict[str, Any]]:
    bank = [r for r in read_jsonl(V2 / "phase301_semantic_full_test_plan_rows.jsonl") if r.get("model") == model]
    reuse = [r for r in read_jsonl(V2 / "phase304_semantic_reuse_matrix_v2_rows.jsonl") if r.get("semantic_relation_v2") == "subclass_shared_backbone"]
    controls = [r for r in read_jsonl(V2 / "phase304_semantic_reuse_matrix_v2_rows.jsonl") if r.get("semantic_relation_v2") == "contrast_control"]
    priority_objects: list[str] = []
    for row in sorted(reuse, key=lambda r: -safe_float(r.get("corrected_reuse_score_v2")))[:8]:
        priority_objects.extend([str(row.get("left_object_id")), str(row.get("right_object_id"))])
    for row in sorted(controls, key=lambda r: -safe_float(r.get("corrected_delta_score_v2")))[:4]:
        priority_objects.extend([str(row.get("left_object_id")), str(row.get("right_object_id"))])
    priority = []
    for row in bank:
        score = 0
        if row.get("case_type") == "semantic_contrast":
            score += 5
        if str(row.get("object_id")) in priority_objects:
            score += 3
        if str(row.get("attribute_type")) in {"category", "subclass", "color", "taste", "use", "shared", "difference"}:
            score += 2
        if score > 0:
            priority.append((score, row))
    priority.sort(key=lambda x: (-x[0], str(x[1].get("case_type")), str(x[1].get("object_id")), str(x[1].get("attribute_type"))))
    selected: list[dict[str, Any]] = []
    used_attr: Counter[str] = Counter()
    for _score, row in priority:
        if len(selected) >= limit:
            break
        attr = str(row.get("attribute_type"))
        if used_attr[attr] >= max(2, limit // 6):
            continue
        selected.append(row)
        used_attr[attr] += 1
    for _score, row in priority:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)
    return selected[:limit]


def decompose_case(model_obj: Any, tokenizer: Any, device: torch.device, case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = str(case["prompt"])
    target_aliases = [str(x) for x in case.get("target_aliases") or [case.get("target", "")]]
    attr = str(case.get("attribute_type") or case.get("semantic_field") or "unknown")
    distractors = [x for x in DISTRACTORS.get(attr, []) if x not in target_aliases]
    if not distractors:
        distractors = ["fruit", "vegetable", "tool", "red", "yellow", "sweet", "sour"]
    target_ids = token_ids(tokenizer, target_aliases)
    distractor_ids = token_ids(tokenizer, distractors)
    captured, final_logits, last_pos = p268.capture_components(model_obj, tokenizer, device, prompt)
    final_norm = p268.get_final_norm(model_obj)
    final_readout = semantic_readout(final_logits, target_ids, distractor_ids)
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
        "attribute_type": attr,
        "prompt_type": case.get("prompt_type"),
        "target": case.get("target"),
        "target_aliases": target_aliases,
        "distractor_aliases": distractors,
    }
    rows: list[dict[str, Any]] = []
    for layer_idx in sorted(captured):
        comp = captured[layer_idx]
        h0 = p268.tensor_at_pos(comp.get("layer_input"), last_pos)
        attn = p268.tensor_at_pos(comp.get("attn_out"), last_pos)
        mlp = p268.tensor_at_pos(comp.get("mlp_out"), last_pos)
        layer_out = p268.tensor_at_pos(comp.get("layer_out"), last_pos)
        if h0 is None or layer_out is None:
            continue
        h0_read = semantic_readout(p268.project_state(model_obj, final_norm, h0), target_ids, distractor_ids)
        h_attn = h0 + attn if attn is not None else h0
        h_attn_read = semantic_readout(p268.project_state(model_obj, final_norm, h_attn), target_ids, distractor_ids)
        h_mlp = h_attn + mlp if mlp is not None else h_attn
        h_mlp_read = semantic_readout(p268.project_state(model_obj, final_norm, h_mlp), target_ids, distractor_ids)
        h_out_read = semantic_readout(p268.project_state(model_obj, final_norm, layer_out), target_ids, distractor_ids)
        m0 = safe_float(h0_read["semantic_margin"])
        ma = safe_float(h_attn_read["semantic_margin"])
        mm = safe_float(h_mlp_read["semantic_margin"])
        mo = safe_float(h_out_read["semantic_margin"])
        rows.append(
            {
                **base,
                "semantic_component_path_id": f"phase305:semantic_component:{case['model']}:{case['case_id']}:L{layer_idx}",
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
        "semantic_component_summary_id": f"phase305:summary:{case['model']}:{case['case_id']}",
        "layers_observed": len(rows),
        "final_semantic_margin": round(safe_float(final_readout["semantic_margin"]), 6),
        "final_semantic_winner": final_readout["semantic_winner"],
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
    cases = select_cases(model, args.cases_per_model)
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
                rows, summary = decompose_case(model_obj, tokenizer, device, case)
                component_rows.extend(rows)
                summary_rows.append(summary)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append({"schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(), "model": model, "case_id": case.get("case_id"), "reason": repr(exc)})
            print(f"{model}: semantic component traced {idx}/{len(cases)}", flush=True)
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
    write_json(out_dir / f"phase305_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase305_{model}_semantic_component_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase305_{model}_semantic_component_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase305_{model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_model(model: str, selected: list[dict[str, Any]], summaries: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "model": model,
        "selected_rows": len(selected),
        "semantic_component_summary_rows": len(summaries),
        "missing_rows": len(missing),
        "attribute_counts": dict(Counter(str(r.get("attribute_type")) for r in summaries)),
        "final_semantic_winner_counts": dict(Counter(str(r.get("final_semantic_winner")) for r in summaries)),
        "dominant_positive_semantic_component_counts": dict(Counter(str(r.get("dominant_positive_semantic_component")) for r in summaries)),
        "mean_final_semantic_margin": mean_safe([safe_float(r.get("final_semantic_margin")) for r in summaries]),
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
        model_summaries.append(read_json(out_dir / f"phase305_{model}_summary.json"))
        component.extend(read_jsonl(out_dir / f"phase305_{model}_semantic_component_rows.jsonl"))
        summaries.extend(read_jsonl(out_dir / f"phase305_{model}_semantic_component_summary_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase305_{model}_missing_rows.jsonl"))
    model_summaries = [s for s in model_summaries if s]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "round_name": round_name,
        "model_summaries": model_summaries,
        "semantic_component_rows": len(component),
        "semantic_component_summary_rows": len(summaries),
        "missing_rows": len(missing),
        "attribute_counts": dict(Counter(str(r.get("attribute_type")) for r in summaries)),
        "final_semantic_winner_counts": dict(Counter(str(r.get("final_semantic_winner")) for r in summaries)),
        "dominant_positive_semantic_component_counts": dict(Counter(str(r.get("dominant_positive_semantic_component")) for r in summaries)),
        "mean_final_semantic_margin": mean_safe([safe_float(r.get("final_semantic_margin")) for r in summaries]),
        "mean_sum_positive_attn_semantic_delta": mean_safe([safe_float(r.get("sum_positive_attn_semantic_delta")) for r in summaries]),
        "mean_sum_positive_mlp_semantic_delta": mean_safe([safe_float(r.get("sum_positive_mlp_semantic_delta")) for r in summaries]),
        "mean_sum_positive_residual_semantic_delta": mean_safe([safe_float(r.get("sum_positive_residual_semantic_delta")) for r in summaries]),
    }
    write_json(out_dir / "phase305_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase305_semantic_component_rows.jsonl", component)
    write_jsonl(out_dir / "phase305_semantic_component_summary_rows.jsonl", summaries)
    write_jsonl(out_dir / "phase305_missing_rows.jsonl", missing)
    write_json(V2 / "phase305_cross_model_summary.json", payload)
    write_jsonl(V2 / "phase305_semantic_component_rows.jsonl", component)
    write_jsonl(V2 / "phase305_semantic_component_summary_rows.jsonl", summaries)
    write_jsonl(V2 / "phase305_missing_rows.jsonl", missing)
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
