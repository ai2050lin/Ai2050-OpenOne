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


PHASE = 888
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE887_ROOT = Path("tests/result/phase887_blocker_minimal_cut_subspace_basis_probe/blocker_minimal_cut_probe")
RESULT_ROOT = Path("tests/result/phase888_direction_set_internal_subspace_probe")


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


def case_map() -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): dict(case) for case in p885.extended_cases()}


def boundary_closed(metrics: dict[str, Any]) -> bool:
    return metrics.get("class_blocker_count") == 0 and metrics.get("class_best_rank") == 1


def safe_token_ids(items: list[Any]) -> list[int]:
    out = []
    for item in items or []:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def token_delta_mean(logits: torch.Tensor, base_logits: torch.Tensor, token_ids: list[int]) -> float | None:
    vals = []
    for token_id in token_ids:
        if 0 <= int(token_id) < int(logits.numel()):
            vals.append(float(logits[int(token_id)].item() - base_logits[int(token_id)].item()))
    return mean(vals)


def token_delta_negative_count(logits: torch.Tensor, base_logits: torch.Tensor, token_ids: list[int]) -> int:
    count = 0
    for token_id in token_ids:
        if 0 <= int(token_id) < int(logits.numel()) and float(logits[int(token_id)].item() - base_logits[int(token_id)].item()) < 0:
            count += 1
    return count


def metrics_for_logits(tokenizer, logits: torch.Tensor, token_sets: dict[str, Any], topk: int) -> dict[str, Any]:
    metrics = p856.first_token_metrics(tokenizer, logits, token_sets, int(topk))
    metrics["class_boundary_closed"] = boundary_closed(metrics)
    compact = []
    for item in metrics.get("top_tokens") or []:
        compact.append(
            {
                "token_id": item.get("token_id"),
                "token": item.get("token"),
                "logit": item.get("logit"),
                "role": item.get("role"),
            }
        )
    metrics["top_tokens_compact"] = compact[: int(topk)]
    return metrics


def suppress_logits(logits: torch.Tensor, token_ids: list[int], value: float = -1.0e9) -> torch.Tensor:
    patched = logits.clone()
    for token_id in token_ids:
        if 0 <= int(token_id) < int(patched.numel()):
            patched[int(token_id)] = value
    return patched


def threshold_suppress_logits(logits: torch.Tensor, token_ids: list[int], class_best_logit: float | None, margin: float) -> torch.Tensor:
    patched = logits.clone()
    if class_best_logit is None:
        return suppress_logits(logits, token_ids)
    threshold = float(class_best_logit) - float(margin)
    for token_id in token_ids:
        if 0 <= int(token_id) < int(patched.numel()):
            patched[int(token_id)] = min(float(patched[int(token_id)].item()), threshold)
    return patched


def restore_cut_logits(mode_logits: torch.Tensor, base_logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    patched = mode_logits.clone()
    for token_id in token_ids:
        if 0 <= int(token_id) < int(patched.numel()):
            patched[int(token_id)] = base_logits[int(token_id)]
    return patched


def load_phase887_rows(model: str) -> list[dict[str, Any]]:
    return p846.read_jsonl(PHASE887_ROOT / f"phase887_{model}_cut_rows.jsonl")


def select_probe_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[str(row.get("parent_candidate_key"))].append(row)
    selected: list[dict[str, Any]] = []
    for _candidate, vals in sorted(by_candidate.items()):
        exact = [row for row in vals if row.get("exact_single_blocker_cut")]
        shared = [row for row in vals if row.get("shared_complete_topk_cut") and not row.get("exact_single_blocker_cut")]
        controls = [row for row in vals if not row.get("same_boundary_closure")]
        exact.sort(key=lambda row: (str(row.get("object")), str(row.get("prompt_variant"))))
        shared.sort(key=lambda row: (finite(row.get("base_blocker_count"), 999.0), str(row.get("object")), str(row.get("prompt_variant"))))
        controls.sort(key=lambda row: (str(row.get("case_split")), str(row.get("object")), str(row.get("prompt_variant"))))
        chosen = exact[: int(args.max_exact_rows_per_candidate)]
        chosen += shared[: max(0, int(args.max_cut_rows_per_candidate) - len(chosen))]
        chosen += controls[: int(args.max_control_rows_per_candidate)]
        selected.extend(chosen)
    return selected


def make_row(
    model_name: str,
    source: dict[str, Any],
    mode: str,
    base_metrics: dict[str, Any],
    mode_metrics: dict[str, Any],
    base_mask_metrics: dict[str, Any],
    base_suppress_metrics: dict[str, Any],
    restored_metrics: dict[str, Any],
    cut_token_delta: float | None,
    cut_token_negative_count: int,
) -> dict[str, Any]:
    base_closed = bool(base_metrics.get("class_boundary_closed"))
    mode_closed = bool(mode_metrics.get("class_boundary_closed"))
    restored_closed = bool(restored_metrics.get("class_boundary_closed"))
    return {
        "phase": PHASE,
        "row_kind": "phase888_direction_set_probe_row",
        "model": model_name,
        "parent_candidate_key": source.get("parent_candidate_key"),
        "case_id": source.get("case_id"),
        "case_split": source.get("case_split"),
        "eval_domain": source.get("eval_domain"),
        "object": source.get("object"),
        "prompt_variant": source.get("prompt_variant"),
        "edit_mode": mode,
        "source_exact_single_blocker_cut": bool(source.get("exact_single_blocker_cut")),
        "source_shared_complete_topk_cut": bool(source.get("shared_complete_topk_cut")),
        "source_same_boundary_closure": bool(source.get("same_boundary_closure")),
        "cut_token_ids": safe_token_ids(source.get("exact_cut_ids") or source.get("shared_removed_ids") or []),
        "exact_cut_records": source.get("exact_cut_records") or [],
        "shared_removed_records": source.get("shared_removed_records") or [],
        "base_class_blocker_count": base_metrics.get("class_blocker_count"),
        "base_class_rank": base_metrics.get("class_best_rank"),
        "base_class_logit": base_metrics.get("class_best_logit"),
        "base_boundary_closed": base_closed,
        "mode_class_blocker_count": mode_metrics.get("class_blocker_count"),
        "mode_class_rank": mode_metrics.get("class_best_rank"),
        "mode_class_logit": mode_metrics.get("class_best_logit"),
        "mode_boundary_closed": mode_closed,
        "mode_closure_from_open": bool((not base_closed) and mode_closed),
        "mode_blocker_reduction": None
        if base_metrics.get("class_blocker_count") is None or mode_metrics.get("class_blocker_count") is None
        else finite(base_metrics.get("class_blocker_count")) - finite(mode_metrics.get("class_blocker_count")),
        "mode_rank_improvement": None
        if base_metrics.get("class_best_rank") is None or mode_metrics.get("class_best_rank") is None
        else finite(base_metrics.get("class_best_rank")) - finite(mode_metrics.get("class_best_rank")),
        "mode_class_logit_delta": None
        if base_metrics.get("class_best_logit") is None or mode_metrics.get("class_best_logit") is None
        else finite(mode_metrics.get("class_best_logit")) - finite(base_metrics.get("class_best_logit")),
        "cut_token_delta_mean": cut_token_delta,
        "cut_token_negative_count": cut_token_negative_count,
        "base_mask_boundary_closed": bool(base_mask_metrics.get("class_boundary_closed")),
        "base_mask_class_blocker_count": base_mask_metrics.get("class_blocker_count"),
        "base_suppress_boundary_closed": bool(base_suppress_metrics.get("class_boundary_closed")),
        "base_suppress_class_blocker_count": base_suppress_metrics.get("class_blocker_count"),
        "restored_boundary_closed": restored_closed,
        "restored_class_blocker_count": restored_metrics.get("class_blocker_count"),
        "restored_reopens_boundary": bool(mode_closed and not restored_closed),
        "restored_increases_blockers": bool(
            mode_metrics.get("class_blocker_count") is not None
            and restored_metrics.get("class_blocker_count") is not None
            and finite(restored_metrics.get("class_blocker_count")) > finite(mode_metrics.get("class_blocker_count"))
        ),
        "mode_top_tokens": mode_metrics.get("top_tokens_compact"),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_phase887_rows(args.model)
    selected = select_probe_rows(all_rows, args)
    cases = case_map()
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
        p846.write_json(out_dir / f"phase888_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase888_{args.model}_rows.jsonl", [])
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
                base_metrics = metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens))
                cache[cache_key] = (base_logits, base_metrics, token_sets, prompt_ids)
            base_logits, base_metrics, token_sets, prompt_ids = cache[cache_key]
            cut_token_ids = safe_token_ids(source.get("exact_cut_ids") or source.get("shared_removed_ids") or [])
            base_mask_metrics = metrics_for_logits(
                tokenizer,
                suppress_logits(base_logits, cut_token_ids),
                token_sets,
                int(args.topk_tokens),
            )
            base_suppress_metrics = metrics_for_logits(
                tokenizer,
                threshold_suppress_logits(
                    base_logits, cut_token_ids, base_metrics.get("class_best_logit"), float(args.suppress_margin)
                ),
                token_sets,
                int(args.topk_tokens),
            )
            gears = p885.parse_gears_from_candidate_key(str(source.get("parent_candidate_key")))
            for mode in modes:
                mode_logits = p862.first_logits_with_scaled_gears(
                    model, device, prompt_ids, gears, str(mode), float(args.scale_up_factor)
                )
                mode_metrics = metrics_for_logits(tokenizer, mode_logits, token_sets, int(args.topk_tokens))
                restored_logits = restore_cut_logits(mode_logits, base_logits, cut_token_ids)
                restored_metrics = metrics_for_logits(tokenizer, restored_logits, token_sets, int(args.topk_tokens))
                rows.append(
                    make_row(
                        args.model,
                        source,
                        str(mode),
                        base_metrics,
                        mode_metrics,
                        base_mask_metrics,
                        base_suppress_metrics,
                        restored_metrics,
                        token_delta_mean(mode_logits, base_logits, cut_token_ids),
                        token_delta_negative_count(mode_logits, base_logits, cut_token_ids),
                    )
                )
            log(f"{args.model}/{args.round_name}: probe_row={idx}/{len(selected)} output_rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(args.model, rows, selected, modes, attn_impl)
    p846.write_json(out_dir / f"phase888_{args.model}_summary.json", summary)
    p846.write_jsonl(out_dir / f"phase888_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "overall": summary["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def summarize_rows(
    model_name: str,
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    modes: list[str],
    attn_impl: str | None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("parent_candidate_key"))].append(row)
    candidate_groups = []
    for key, vals in groups.items():
        source_cases = {(row.get("case_id"), row.get("prompt_variant")) for row in vals}
        mode_counter = Counter(str(row.get("edit_mode")) for row in vals if row.get("mode_closure_from_open"))
        restore_counter = Counter(str(row.get("edit_mode")) for row in vals if row.get("restored_reopens_boundary"))
        source_exact = sum(1 for item in selected if str(item.get("parent_candidate_key")) == key and item.get("exact_single_blocker_cut"))
        source_shared = sum(1 for item in selected if str(item.get("parent_candidate_key")) == key and item.get("shared_complete_topk_cut"))
        by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in vals:
            by_case[(str(row.get("case_id")), str(row.get("prompt_variant")))].append(row)
        multi_mode_cases = 0
        stable_restore_cases = 0
        for case_key, case_vals in by_case.items():
            close_modes = [row for row in case_vals if row.get("mode_closure_from_open")]
            restore_modes = [row for row in case_vals if row.get("restored_reopens_boundary")]
            if len(close_modes) >= 2:
                multi_mode_cases += 1
            if close_modes and restore_modes:
                stable_restore_cases += 1
        base_mask_closed_cases = len({(row.get("case_id"), row.get("prompt_variant")) for row in vals if row.get("base_mask_boundary_closed")})
        base_suppress_closed_cases = len({(row.get("case_id"), row.get("prompt_variant")) for row in vals if row.get("base_suppress_boundary_closed")})
        closure_count = sum(1 for row in vals if row.get("mode_closure_from_open"))
        restore_reopen_count = sum(1 for row in vals if row.get("restored_reopens_boundary"))
        if closure_count and restore_reopen_count:
            label = "direction_set_internal_subspace_signal"
        elif closure_count:
            label = "direction_set_boundary_signal_no_restore"
        elif base_mask_closed_cases:
            label = "output_cut_sufficient_without_internal_mode"
        else:
            label = "negative_no_direction_set_signal"
        candidate_groups.append(
            {
                "model": model_name,
                "parent_candidate_key": key,
                "evidence_label": label,
                "n_source_cases": len(source_cases),
                "n_rows": len(vals),
                "source_exact_single_cases": source_exact,
                "source_shared_complete_cases": source_shared,
                "mode_closure_from_open": closure_count,
                "restored_reopens_boundary": restore_reopen_count,
                "restored_increases_blockers": sum(1 for row in vals if row.get("restored_increases_blockers")),
                "base_mask_closed_cases": base_mask_closed_cases,
                "base_suppress_closed_cases": base_suppress_closed_cases,
                "multi_mode_closure_cases": multi_mode_cases,
                "stable_restore_cases": stable_restore_cases,
                "closure_modes": counter_values(mode_counter),
                "restore_reopen_modes": counter_values(restore_counter),
                "mean_cut_token_delta": mean([finite(row.get("cut_token_delta_mean")) for row in vals if row.get("cut_token_delta_mean") is not None]) or 0.0,
                "mean_mode_blocker_reduction": mean([finite(row.get("mode_blocker_reduction")) for row in vals if row.get("mode_blocker_reduction") is not None]) or 0.0,
                "objects": sorted(set(str(row.get("object")) for row in vals)),
                "prompt_variants": sorted(set(str(row.get("prompt_variant")) for row in vals)),
            }
        )
    candidate_groups.sort(key=lambda row: (row["mode_closure_from_open"], row["restored_reopens_boundary"], row["base_mask_closed_cases"]), reverse=True)
    return {
        "phase": PHASE,
        "title": "Direction-set Intervention and Internal Subspace Basis Probe",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "edit_modes": modes,
        "selected_source_rows": len(selected),
        "output_rows": len(rows),
        "overall": {
            "mode_closure_from_open": sum(1 for row in rows if row.get("mode_closure_from_open")),
            "restored_reopens_boundary": sum(1 for row in rows if row.get("restored_reopens_boundary")),
            "restored_increases_blockers": sum(1 for row in rows if row.get("restored_increases_blockers")),
            "base_mask_boundary_closed": sum(1 for row in rows if row.get("base_mask_boundary_closed")),
            "base_suppress_boundary_closed": sum(1 for row in rows if row.get("base_suppress_boundary_closed")),
            "mode_counts": counter_values(Counter(str(row.get("edit_mode")) for row in rows)),
            "closure_modes": counter_values(Counter(str(row.get("edit_mode")) for row in rows if row.get("mode_closure_from_open"))),
            "restore_reopen_modes": counter_values(Counter(str(row.get("edit_mode")) for row in rows if row.get("restored_reopens_boundary"))),
        },
        "candidate_groups": candidate_groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in candidate_groups)),
        "boundary": (
            "Phase888 performs real internal channel direction-set interventions plus output cut-token "
            "mask/suppress/restore counterfactuals. It is internal subspace evidence probing, not closure."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 888 direction-set intervention and internal subspace probe",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
        f"- source_rows: {payload.get('source_rows')}",
        f"- output_rows: {payload.get('output_rows')}",
        f"- mode_closure_from_open: {payload.get('overall', {}).get('mode_closure_from_open')}",
        f"- restored_reopens_boundary: {payload.get('overall', {}).get('restored_reopens_boundary')}",
        f"- base_mask_boundary_closed: {payload.get('overall', {}).get('base_mask_boundary_closed')}",
        f"- unique_base_mask_closed_cases: {payload.get('unique_case_overall', {}).get('base_mask_closed_cases')}",
        f"- unique_multi_mode_closure_cases: {payload.get('unique_case_overall', {}).get('multi_mode_closure_cases')}",
        "",
        "## Candidate groups",
        "",
        "| model | candidate | label | cases | closures | restore reopen | mask closed | multi-mode | modes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("candidate_groups", [])[:30]:
        lines.append(
            "| {model} | {key} | {label} | {cases} | {closures} | {restore} | {mask} | {multi} | {modes} |".format(
                model=row.get("model"),
                key=row.get("parent_candidate_key"),
                label=row.get("evidence_label"),
                cases=row.get("n_source_cases"),
                closures=row.get("mode_closure_from_open"),
                restore=row.get("restored_reopens_boundary"),
                mask=row.get("base_mask_closed_cases"),
                multi=row.get("multi_mode_closure_cases"),
                modes=json.dumps(row.get("closure_modes") or {}, ensure_ascii=False),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase888_{model}_summary.json") for model in MODELS]
    summaries = [item for item in summaries if item and item.get("status") == "complete"]
    overall = Counter()
    groups: list[dict[str, Any]] = []
    source_rows = 0
    output_rows = 0
    unique_group_totals = Counter()
    for summary in summaries:
        source_rows += int(summary.get("selected_source_rows") or 0)
        output_rows += int(summary.get("output_rows") or 0)
        group_rows = summary.get("candidate_groups") or []
        groups.extend(group_rows)
        for group in group_rows:
            unique_group_totals["base_mask_closed_cases"] += int(group.get("base_mask_closed_cases") or 0)
            unique_group_totals["base_suppress_closed_cases"] += int(group.get("base_suppress_closed_cases") or 0)
            unique_group_totals["multi_mode_closure_cases"] += int(group.get("multi_mode_closure_cases") or 0)
            unique_group_totals["stable_restore_cases"] += int(group.get("stable_restore_cases") or 0)
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
    groups.sort(key=lambda row: (row.get("mode_closure_from_open") or 0, row.get("restored_reopens_boundary") or 0), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "source_rows": source_rows,
        "output_rows": output_rows,
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "unique_case_overall": {key: int(value) for key, value in sorted(unique_group_totals.items())},
        "candidate_groups": groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in groups)),
    }
    p846.write_json(out_dir / "phase888_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase888_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="direction_set_probe")
    parser.add_argument("--edit-modes", default="zero,flip,half,scale_up")
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--suppress-margin", type=float, default=0.05)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--max-exact-rows-per-candidate", type=int, default=10)
    parser.add_argument("--max-cut-rows-per-candidate", type=int, default=18)
    parser.add_argument("--max-control-rows-per-candidate", type=int, default=4)
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
                {"phase": PHASE, "round": args.round_name, "status": payload.get("status"), "models": payload.get("models"), "overall": payload.get("overall")},
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
