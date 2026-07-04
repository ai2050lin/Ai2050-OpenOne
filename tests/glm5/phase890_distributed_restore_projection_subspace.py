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

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase888_direction_set_internal_subspace_probe as p888  # noqa: E402


PHASE = 890
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase890_distributed_restore_projection_subspace")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def effective_mode(mode: str) -> str:
    mapping = {
        "proj_out": "zero",
        "proj_half": "half",
        "proj_reflect": "flip",
        "proj_boost": "scale_up",
    }
    return mapping.get(str(mode), str(mode))


def projection_equivalent(mode: str) -> str | None:
    return effective_mode(mode) if str(mode).startswith("proj_") else None


def safe_ids(items: list[Any]) -> list[int]:
    out = []
    for item in items or []:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def records_by_ids(records: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
    wanted = set(int(x) for x in ids)
    return [dict(record) for record in records or [] if int(record.get("token_id", -1)) in wanted]


def role_groups(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records or []:
        if record.get("token_id") is None:
            continue
        groups[str(record.get("role") or "unknown")].append(int(record["token_id"]))
    return {key: safe_ids(vals) for key, vals in groups.items()}


def restore_specs(source: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()

    def add(name: str, ids: list[int], records: list[dict[str, Any]] | None = None) -> None:
        cleaned = safe_ids(ids)[: int(max_tokens)]
        if not cleaned:
            return
        key = tuple(cleaned)
        if key in seen:
            return
        seen.add(key)
        specs.append(
            {
                "restore_set_type": name,
                "restore_token_ids": cleaned,
                "restore_records": records_by_ids(records or source.get("shared_removed_records") or [], cleaned),
            }
        )

    shared_records = source.get("shared_removed_records") or []
    exact_records = source.get("exact_cut_records") or []
    add("exact_cut", safe_ids(source.get("exact_cut_ids") or []), exact_records)
    add("shared_removed", safe_ids(source.get("shared_removed_ids") or []), shared_records)
    add("base_blockers", safe_ids(source.get("base_blocker_ids") or []), shared_records)
    add("candidate_removed", safe_ids(source.get("candidate_removed_ids") or []), shared_records)
    add("opposite_removed", safe_ids(source.get("opposite_removed_ids") or []), shared_records)
    for role, ids in sorted(role_groups(shared_records).items()):
        add(f"role:{role}", ids, shared_records)
    return specs


def token_delta_mean(logits: torch.Tensor, base_logits: torch.Tensor, token_ids: list[int]) -> float | None:
    vals = []
    for token_id in token_ids:
        if 0 <= int(token_id) < int(logits.numel()):
            vals.append(float(logits[int(token_id)].item() - base_logits[int(token_id)].item()))
    return mean(vals)


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = p888.load_phase887_rows(args.model)
    selected = p888.select_probe_rows(all_rows, args)
    cases = p888.case_map()
    modes = parse_csv(args.edit_modes)
    if args.dry_run or not selected:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if selected else "no_probe_rows",
            "selected_rows": selected,
            "edit_modes": modes,
        }
        p846.write_json(out_dir / f"phase890_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase890_{args.model}_rows.jsonl", [])
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cache: dict[tuple[str, str], tuple[torch.Tensor, dict[str, Any], dict[str, Any], list[int]]] = {}
        for idx, source in enumerate(selected, 1):
            case = cases.get(str(source.get("case_id")))
            if not case:
                continue
            prompt_variant = str(source.get("prompt_variant"))
            prompt = p885.prompt_for_case(case, prompt_variant)
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            token_sets = p856.token_sets(tokenizer, case)
            cache_key = (str(source.get("case_id")), prompt_variant)
            if cache_key not in cache:
                base_logits = p862.first_logits_with_scaled_gears(model, device, prompt_ids, [], "original", float(args.scale_up_factor))
                base_metrics = p888.metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens))
                cache[cache_key] = (base_logits, base_metrics, token_sets, prompt_ids)
            base_logits, base_metrics, token_sets, prompt_ids = cache[cache_key]
            specs = restore_specs(source, int(args.max_restore_tokens))
            gears = p885.parse_gears_from_candidate_key(str(source.get("parent_candidate_key")))
            for mode in modes:
                real_mode = effective_mode(mode)
                mode_logits = p862.first_logits_with_scaled_gears(
                    model, device, prompt_ids, gears, real_mode, float(args.scale_up_factor)
                )
                mode_metrics = p888.metrics_for_logits(tokenizer, mode_logits, token_sets, int(args.topk_tokens))
                mode_closed = bool(mode_metrics.get("class_boundary_closed"))
                base_closed = bool(base_metrics.get("class_boundary_closed"))
                for spec in specs:
                    token_ids = safe_ids(spec.get("restore_token_ids") or [])
                    restored_logits = p888.restore_cut_logits(mode_logits, base_logits, token_ids)
                    restored_metrics = p888.metrics_for_logits(tokenizer, restored_logits, token_sets, int(args.topk_tokens))
                    suppressed_base_logits = p888.threshold_suppress_logits(
                        base_logits, token_ids, base_metrics.get("class_best_logit"), float(args.suppress_margin)
                    )
                    suppressed_base_metrics = p888.metrics_for_logits(
                        tokenizer, suppressed_base_logits, token_sets, int(args.topk_tokens)
                    )
                    rows.append(
                        {
                            "phase": PHASE,
                            "row_kind": "phase890_distributed_restore_projection_row",
                            "model": args.model,
                            "parent_candidate_key": source.get("parent_candidate_key"),
                            "case_id": source.get("case_id"),
                            "case_split": source.get("case_split"),
                            "eval_domain": source.get("eval_domain"),
                            "object": source.get("object"),
                            "prompt_variant": prompt_variant,
                            "edit_mode": str(mode),
                            "effective_mode": real_mode,
                            "projection_equivalent_mode": projection_equivalent(str(mode)),
                            "is_projection_style": bool(str(mode).startswith("proj_")),
                            "restore_set_type": spec.get("restore_set_type"),
                            "restore_token_ids": token_ids,
                            "restore_set_size": len(token_ids),
                            "restore_records": spec.get("restore_records") or [],
                            "source_exact_single_blocker_cut": bool(source.get("exact_single_blocker_cut")),
                            "source_shared_complete_topk_cut": bool(source.get("shared_complete_topk_cut")),
                            "base_boundary_closed": base_closed,
                            "base_class_blocker_count": base_metrics.get("class_blocker_count"),
                            "base_class_logit": base_metrics.get("class_best_logit"),
                            "mode_boundary_closed": mode_closed,
                            "mode_closure_from_open": bool((not base_closed) and mode_closed),
                            "mode_class_blocker_count": mode_metrics.get("class_blocker_count"),
                            "mode_class_logit": mode_metrics.get("class_best_logit"),
                            "mode_class_logit_delta": None
                            if base_metrics.get("class_best_logit") is None or mode_metrics.get("class_best_logit") is None
                            else finite(mode_metrics.get("class_best_logit")) - finite(base_metrics.get("class_best_logit")),
                            "mode_blocker_reduction": None
                            if base_metrics.get("class_blocker_count") is None or mode_metrics.get("class_blocker_count") is None
                            else finite(base_metrics.get("class_blocker_count")) - finite(mode_metrics.get("class_blocker_count")),
                            "restore_boundary_closed": bool(restored_metrics.get("class_boundary_closed")),
                            "restore_reopens_boundary": bool(mode_closed and not restored_metrics.get("class_boundary_closed")),
                            "restore_class_blocker_count": restored_metrics.get("class_blocker_count"),
                            "restore_increases_blockers": bool(
                                mode_metrics.get("class_blocker_count") is not None
                                and restored_metrics.get("class_blocker_count") is not None
                                and finite(restored_metrics.get("class_blocker_count")) > finite(mode_metrics.get("class_blocker_count"))
                            ),
                            "base_suppress_boundary_closed": bool(suppressed_base_metrics.get("class_boundary_closed")),
                            "base_suppress_class_blocker_count": suppressed_base_metrics.get("class_blocker_count"),
                            "restore_token_delta_mean": token_delta_mean(mode_logits, base_logits, token_ids),
                        }
                    )
            log(f"{args.model}/{args.round_name}: source={idx}/{len(selected)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_rows(args.model, rows, selected, modes, attn_impl)
    p846.write_json(out_dir / f"phase890_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase890_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_rows(model_name: str, rows: list[dict[str, Any]], selected: list[dict[str, Any]], modes: list[str], attn_impl: str | None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("parent_candidate_key"))].append(row)
    candidate_groups = []
    for key, vals in groups.items():
        closure_rows = [row for row in vals if row.get("mode_closure_from_open")]
        restore_rows = [row for row in vals if row.get("restore_reopens_boundary")]
        distributed_restore_rows = [
            row
            for row in restore_rows
            if row.get("restore_set_type") in {"shared_removed", "base_blockers", "candidate_removed", "opposite_removed"}
            or int(row.get("restore_set_size") or 0) > 1
        ]
        projection_rows = [row for row in vals if row.get("is_projection_style")]
        projection_closure = [row for row in projection_rows if row.get("mode_closure_from_open")]
        set_type_counter = Counter(str(row.get("restore_set_type")) for row in restore_rows)
        closure_mode_counter = Counter(str(row.get("edit_mode")) for row in closure_rows)
        projection_closure_modes = Counter(str(row.get("edit_mode")) for row in projection_closure)
        suppress_set_counter = Counter(str(row.get("restore_set_type")) for row in vals if row.get("base_suppress_boundary_closed"))
        if distributed_restore_rows:
            label = "distributed_restore_reopens_boundary"
        elif projection_closure:
            label = "projection_equivalent_direction_signal"
        elif closure_rows:
            label = "direction_signal_no_distributed_restore"
        elif any(row.get("base_suppress_boundary_closed") for row in vals):
            label = "output_distributed_suppress_only"
        else:
            label = "negative_no_distributed_restore"
        candidate_groups.append(
            {
                "model": model_name,
                "parent_candidate_key": key,
                "evidence_label": label,
                "n_source_cases": len(set((str(row.get("case_id")), str(row.get("prompt_variant"))) for row in vals)),
                "n_rows": len(vals),
                "mode_closure_from_open": len(closure_rows),
                "restore_reopens_boundary": len(restore_rows),
                "distributed_restore_reopens_boundary": len(distributed_restore_rows),
                "projection_style_rows": len(projection_rows),
                "projection_style_closure": len(projection_closure),
                "restore_reopen_set_types": counter_values(set_type_counter),
                "closure_modes": counter_values(closure_mode_counter),
                "projection_closure_modes": counter_values(projection_closure_modes),
                "base_suppress_set_types": counter_values(suppress_set_counter),
                "mean_closure_class_logit_delta": mean([finite(row.get("mode_class_logit_delta")) for row in closure_rows]) or 0.0,
                "mean_closure_restore_token_delta": mean([finite(row.get("restore_token_delta_mean")) for row in closure_rows if row.get("restore_token_delta_mean") is not None]) or 0.0,
                "objects": sorted(set(str(row.get("object")) for row in vals)),
                "prompt_variants": sorted(set(str(row.get("prompt_variant")) for row in vals)),
            }
        )
    candidate_groups.sort(
        key=lambda row: (
            row.get("distributed_restore_reopens_boundary") or 0,
            row.get("projection_style_closure") or 0,
            row.get("mode_closure_from_open") or 0,
        ),
        reverse=True,
    )
    return {
        "phase": PHASE,
        "title": "Distributed Restore and Projection-style Subspace Intervention",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "edit_modes": modes,
        "selected_source_rows": len(selected),
        "output_rows": len(rows),
        "overall": {
            "mode_closure_from_open": sum(1 for row in rows if row.get("mode_closure_from_open")),
            "restore_reopens_boundary": sum(1 for row in rows if row.get("restore_reopens_boundary")),
            "distributed_restore_reopens_boundary": sum(
                1
                for row in rows
                if row.get("restore_reopens_boundary")
                and (
                    row.get("restore_set_type") in {"shared_removed", "base_blockers", "candidate_removed", "opposite_removed"}
                    or int(row.get("restore_set_size") or 0) > 1
                )
            ),
            "projection_style_closure": sum(1 for row in rows if row.get("is_projection_style") and row.get("mode_closure_from_open")),
            "base_suppress_boundary_closed": sum(1 for row in rows if row.get("base_suppress_boundary_closed")),
            "closure_modes": counter_values(Counter(str(row.get("edit_mode")) for row in rows if row.get("mode_closure_from_open"))),
            "restore_reopen_set_types": counter_values(Counter(str(row.get("restore_set_type")) for row in rows if row.get("restore_reopens_boundary"))),
        },
        "candidate_groups": candidate_groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in candidate_groups)),
        "boundary": (
            "Phase890 tests distributed blocker-set restore and single-axis projection-equivalent interventions. "
            "For current mostly single-channel candidates, projection-style modes degenerate to axis projection equivalents."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 890 distributed restore and projection-style subspace intervention",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
        f"- source_rows: {payload.get('source_rows')}",
        f"- output_rows: {payload.get('output_rows')}",
        f"- mode_closure_from_open: {payload.get('overall', {}).get('mode_closure_from_open')}",
        f"- restore_reopens_boundary: {payload.get('overall', {}).get('restore_reopens_boundary')}",
        f"- distributed_restore_reopens_boundary: {payload.get('overall', {}).get('distributed_restore_reopens_boundary')}",
        f"- projection_style_closure: {payload.get('overall', {}).get('projection_style_closure')}",
        f"- unique_source_cases: {payload.get('unique_case_counts', {}).get('source_cases')}",
        f"- unique_closure_cases: {payload.get('unique_case_counts', {}).get('closure_cases')}",
        f"- unique_restore_cases: {payload.get('unique_case_counts', {}).get('restore_reopen_cases')}",
        f"- unique_distributed_restore_cases: {payload.get('unique_case_counts', {}).get('distributed_restore_reopen_cases')}",
        "",
        "## Candidate groups",
        "",
        "| model | candidate | label | closures | restore | distributed restore | projection closure | set types | modes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload.get("candidate_groups", [])[:30]:
        lines.append(
            "| {model} | {key} | {label} | {closures} | {restore} | {distributed} | {projection} | {sets} | {modes} |".format(
                model=row.get("model"),
                key=row.get("parent_candidate_key"),
                label=row.get("evidence_label"),
                closures=row.get("mode_closure_from_open"),
                restore=row.get("restore_reopens_boundary"),
                distributed=row.get("distributed_restore_reopens_boundary"),
                projection=row.get("projection_style_closure"),
                sets=json.dumps(row.get("restore_reopen_set_types") or {}, ensure_ascii=False),
                modes=json.dumps(row.get("closure_modes") or {}, ensure_ascii=False),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unique_case_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    def case_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (str(row.get("parent_candidate_key")), str(row.get("case_id")), str(row.get("prompt_variant")))

    source_cases = {case_key(row) for row in rows}
    closure_cases = {case_key(row) for row in rows if row.get("mode_closure_from_open")}
    projection_closure_cases = {
        case_key(row) for row in rows if row.get("mode_closure_from_open") and row.get("is_projection_style")
    }
    restore_reopen_cases = {case_key(row) for row in rows if row.get("restore_reopens_boundary")}
    distributed_restore_reopen_cases = {
        case_key(row)
        for row in rows
        if row.get("restore_reopens_boundary")
        and (
            row.get("restore_set_type") in {"shared_removed", "base_blockers", "candidate_removed", "opposite_removed"}
            or int(row.get("restore_set_size") or 0) > 1
        )
    }
    return {
        "source_cases": len(source_cases),
        "closure_cases": len(closure_cases),
        "projection_closure_cases": len(projection_closure_cases),
        "restore_reopen_cases": len(restore_reopen_cases),
        "distributed_restore_reopen_cases": len(distributed_restore_reopen_cases),
    }


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase890_{model}_summary.json") for model in MODELS]
    summaries = [item for item in summaries if item and item.get("status") == "complete"]
    overall = Counter()
    groups: list[dict[str, Any]] = []
    source_rows = 0
    output_rows = 0
    for summary in summaries:
        source_rows += int(summary.get("selected_source_rows") or 0)
        output_rows += int(summary.get("output_rows") or 0)
        groups.extend(summary.get("candidate_groups") or [])
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
    all_rows: list[dict[str, Any]] = []
    model_unique_counts: dict[str, dict[str, int]] = {}
    for model in MODELS:
        rows_path = out_dir / f"phase890_{model}_rows.jsonl"
        rows = p846.read_jsonl(rows_path) if rows_path.exists() else []
        all_rows.extend(rows)
        if rows:
            model_unique_counts[model] = unique_case_counts(rows)
    groups.sort(
        key=lambda row: (
            row.get("distributed_restore_reopens_boundary") or 0,
            row.get("projection_style_closure") or 0,
            row.get("mode_closure_from_open") or 0,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "source_rows": source_rows,
        "output_rows": output_rows,
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "unique_case_counts": unique_case_counts(all_rows),
        "model_unique_case_counts": model_unique_counts,
        "candidate_groups": groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in groups)),
    }
    p846.write_json(out_dir / "phase890_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase890_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="distributed_restore_projection")
    parser.add_argument("--edit-modes", default="zero,flip,half,scale_up,proj_out,proj_reflect")
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--suppress-margin", type=float, default=0.05)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--max-exact-rows-per-candidate", type=int, default=10)
    parser.add_argument("--max-cut-rows-per-candidate", type=int, default=18)
    parser.add_argument("--max-control-rows-per-candidate", type=int, default=4)
    parser.add_argument("--max-restore-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
