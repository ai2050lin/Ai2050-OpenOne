#!/usr/bin/env python3
"""
Phase 695: Source-Token Visibility Causal Audit.

Phase 694 mapped which source-token groups are attended by boundary candidate
heads, but attention mass is observational. This phase performs a simple causal
source intervention: keep the text fixed, then use attention_mask to hide
selected source-token groups from the model and measure the first-answer-token
readout effect.

This is not a V-level path patch and not head-specific. It is a conservative
visibility audit that asks which source-token groups are necessary enough to
degrade terse successes or, rarely, repair short failures.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    expected_first_ids,
    expected_for,
    prompt_for,
    route_id_sets,
    select_base_cases,
    value_phrase,
)
from phase685_natural_value_readout_writer_localization import (  # noqa: E402
    SHORT_VARIANT,
    TERSE_VARIANT,
    projection,
    select_paired_cases,
    value_minus_prose_direction,
)
from phase687_l26_l27_value_support_state_decomposition import (  # noqa: E402
    classify,
    get_module,
    model_layers,
)
from phase694_boundary_head_source_token_attention_audit import token_groups  # noqa: E402


OUT_ROOT = Path("results/glm5_phase695_source_token_visibility_causal_audit")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    return tokenizer.encode(prompt, add_special_tokens=False)


def all_nonempty_positions(groups: dict[str, list[int]], key: str, n: int) -> set[int]:
    return {i for i in groups.get(key, []) if 0 <= i < n}


def make_mask_conditions(groups: dict[str, list[int]], n: int) -> dict[str, set[int]]:
    self_last = n - 1
    all_pos = set(range(n))
    record = all_nonempty_positions(groups, "record_line", n)
    question = all_nonempty_positions(groups, "question_line", n)
    instruction = all_nonempty_positions(groups, "instruction_line", n)
    answer = all_nonempty_positions(groups, "answer_line", n)
    answer_context = {i for i in answer if i != self_last}
    target_value = all_nonempty_positions(groups, "target_value", n)
    obj = all_nonempty_positions(groups, "object_name", n)
    relation = all_nonempty_positions(groups, "relation", n)
    record_value_object_relation = (target_value | obj | relation) & record
    record_without_value = record - target_value
    record_without_value_object_relation = record - record_value_object_relation

    keep_record_answer = record | answer | {self_last}
    keep_instruction_answer = instruction | answer | {self_last}
    keep_record_instruction_answer = record | instruction | answer | {self_last}
    keep_question_answer = question | answer | {self_last}

    conds = {
        "mask_record_line": record,
        "mask_question_line": question,
        "mask_instruction_line": instruction,
        "mask_answer_context": answer_context,
        "mask_target_value": target_value,
        "mask_object_name": obj,
        "mask_relation": relation,
        "mask_record_value_object_relation": record_value_object_relation,
        "mask_record_without_target_value": record_without_value,
        "mask_record_without_value_object_relation": record_without_value_object_relation,
        "keep_only_record_answer": all_pos - keep_record_answer,
        "keep_only_instruction_answer": all_pos - keep_instruction_answer,
        "keep_only_record_instruction_answer": all_pos - keep_record_instruction_answer,
        "keep_only_question_answer": all_pos - keep_question_answer,
    }
    return {k: {i for i in v if 0 <= i < n and i != self_last} for k, v in conds.items() if v}


def run_with_visibility_mask(
    model,
    tokenizer,
    device,
    prompt: str,
    target_layer: int,
    direction: torch.Tensor,
    routes,
    expected_ids,
    masked_positions: set[int] | None = None,
) -> dict[str, Any]:
    ids = encode_prompt(tokenizer, prompt)
    attention_mask = torch.ones((1, len(ids)), device=device, dtype=torch.long)
    if masked_positions:
        valid = [i for i in masked_positions if 0 <= i < len(ids) - 1]
        if valid:
            attention_mask[0, torch.tensor(valid, device=device)] = 0
    captured: dict[str, torch.Tensor] = {}
    target_module = get_module(model, target_layer, "layer_input")

    def target_pre_hook(_module, inputs):
        captured["target"] = inputs[0][0, -1].detach()

    handle = target_module.register_forward_pre_hook(target_pre_hook)
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        handle.remove()
    diag["target_proj"] = projection(captured["target"], direction)
    diag["seq_len"] = len(ids)
    diag["masked_count"] = int(attention_mask.numel() - int(attention_mask.sum().detach().cpu().item()))
    diag["masked_fraction"] = diag["masked_count"] / max(1, len(ids))
    return diag


def make_row(
    case: dict[str, Any],
    variant_name: str,
    condition: str,
    baseline: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    is_terse = variant_name == "terse_no_explain"
    if is_terse:
        final_success_change = baseline["expected_top1"] and not patched["expected_top1"]
        rank_effect = patched["expected_rank"] - baseline["expected_rank"]
        pmv_effect = patched["prose_minus_value"] - baseline["prose_minus_value"]
        target_effect = baseline["target_proj"] - patched["target_proj"]
        phase_kind = "degradation"
    else:
        final_success_change = (not baseline["expected_top1"]) and patched["expected_top1"]
        rank_effect = baseline["expected_rank"] - patched["expected_rank"]
        pmv_effect = baseline["prose_minus_value"] - patched["prose_minus_value"]
        target_effect = patched["target_proj"] - baseline["target_proj"]
        phase_kind = "repair"
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "object_name": case.get("object_name"),
        "relation": case.get("relation"),
        "value": value_phrase(case),
        "variant": variant_name,
        "phase_kind": phase_kind,
        "condition": condition,
        "baseline_rank": baseline["expected_rank"],
        "patched_rank": patched["expected_rank"],
        "baseline_top1": baseline["expected_top1"],
        "patched_top1": patched["expected_top1"],
        "final_success_change": final_success_change,
        "rank_effect": rank_effect,
        "baseline_pmv": baseline["prose_minus_value"],
        "patched_pmv": patched["prose_minus_value"],
        "pmv_effect": pmv_effect,
        "baseline_target_proj": baseline["target_proj"],
        "patched_target_proj": patched["target_proj"],
        "target_effect": target_effect,
        "baseline_best_other_route": baseline["best_other_route"],
        "patched_best_other_route": patched["best_other_route"],
        "seq_len": patched["seq_len"],
        "masked_count": patched["masked_count"],
        "masked_fraction": patched["masked_fraction"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "success_change_rate": sum(1 for r in rows if r["final_success_change"]) / n,
        "baseline_top1_rate": sum(1 for r in rows if r["baseline_top1"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_baseline_rank": sum(r["baseline_rank"] for r in rows) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_rank_effect": sum(r["rank_effect"] for r in rows) / n,
        "mean_pmv_effect": sum(r["pmv_effect"] for r in rows) / n,
        "mean_target_effect": sum(r["target_effect"] for r in rows) / n,
        "mean_masked_fraction": sum(r["masked_fraction"] for r in rows) / n,
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["phase_kind"], r["variant"], r["condition"])].append(r)
    by_condition = {f"{k}|{v}|{c}": summarize_group(vals) for (k, v, c), vals in grouped.items()}
    degradation = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("degradation|")),
        key=lambda kv: (kv[1]["success_change_rate"], kv[1]["mean_target_effect"], kv[1]["mean_rank_effect"]),
        reverse=True,
    )
    repair = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("repair|")),
        key=lambda kv: (kv[1]["success_change_rate"], kv[1]["mean_target_effect"], kv[1]["mean_rank_effect"]),
        reverse=True,
    )
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "by_condition": by_condition,
        "best_degradation_conditions": [{"condition": k, **v} for k, v in degradation],
        "best_repair_conditions": [{"condition": k, **v} for k, v in repair],
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    try:
        dtype = next(model.parameters()).dtype
        target_layer = model_layers(args.model, len(get_layers(model)))[0]
        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype)
            for variant_name, variant in [("short_only", SHORT_VARIANT), ("terse_no_explain", TERSE_VARIANT)]:
                prompt = prompt_for(case, variant)
                ids = encode_prompt(tokenizer, prompt)
                groups = token_groups(tokenizer, prompt, case, ids)
                baseline = run_with_visibility_mask(
                    model, tokenizer, device, prompt, target_layer, direction, routes, expected_ids, None
                )
                for condition, masked_positions in make_mask_conditions(groups, len(ids)).items():
                    patched = run_with_visibility_mask(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        target_layer,
                        direction,
                        routes,
                        expected_ids,
                        masked_positions,
                    )
                    rows.append(make_row(case, variant_name, condition, baseline, patched))
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: visibility audited {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase695_{args.model}_source_visibility_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 695,
        "title": "Source-Token Visibility Causal Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "target_layer": target_layer,
        "n_paired_cases": len(paired_ids),
        "summary": summary,
    }
    (OUT_ROOT / f"phase695_{args.model}_source_visibility_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase695_*_source_visibility_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 695,
        "title": "Source-Token Visibility Causal Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase695_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 695 Source-Token Visibility Causal Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | rows | best_degrade | drop | patched_top1 | rank_effect | target_effect | best_repair | repair | patched_top1 | rank_effect | target_effect |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in models:
        bd = item["summary"]["best_degradation_conditions"][0] if item["summary"]["best_degradation_conditions"] else {}
        br = item["summary"]["best_repair_conditions"][0] if item["summary"]["best_repair_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {item['summary']['n_rows']} | "
            f"{bd.get('condition', '')} | {bd.get('success_change_rate', 0.0):.3f} | {bd.get('patched_top1_rate', 0.0):.3f} | "
            f"{bd.get('mean_rank_effect', 0.0):.2f} | {bd.get('mean_target_effect', 0.0):.3f} | "
            f"{br.get('condition', '')} | {br.get('success_change_rate', 0.0):.3f} | {br.get('patched_top1_rate', 0.0):.3f} | "
            f"{br.get('mean_rank_effect', 0.0):.2f} | {br.get('mean_target_effect', 0.0):.3f} |"
        )
    for section, key in [("Best Degradation", "best_degradation_conditions"), ("Best Repair", "best_repair_conditions")]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            lines.append("| condition | change | baseline_top1 | patched_top1 | base_rank | patched_rank | rank_effect | pmv_effect | target_effect | mask_frac | best_other |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in item["summary"][key][:20]:
                lines.append(
                    f"| {row['condition']} | {row['success_change_rate']:.3f} | {row['baseline_top1_rate']:.3f} | "
                    f"{row['patched_top1_rate']:.3f} | {row['mean_baseline_rank']:.2f} | {row['mean_patched_rank']:.2f} | "
                    f"{row['mean_rank_effect']:.2f} | {row['mean_pmv_effect']:.3f} | {row['mean_target_effect']:.3f} | "
                    f"{row['mean_masked_fraction']:.3f} | {row['patched_best_other_route']} |"
                )
            lines.append("")
    (OUT_ROOT / "phase695_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=12)
    args = parser.parse_args()
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
