#!/usr/bin/env python3
"""
Phase 709: Natural Generation and Write-In Closure Audit.

Phase 707 showed that full phrase likelihood usually does not inject donor
identity, while prose often still beats short target values. This phase tests
the harder closure: after the same answer-start source-channel patch, what does
the model naturally generate?

The script also records conservative write-in diagnostics for the selected
channels. These diagnostics are not claimed to fully decompose Q/K/V/MLP; they
separate the measured source contribution into:
  - combo_delta_value: pre-W_O source contribution delta on the selected channel;
  - output_dir_proj: W_O/readout alignment of that channel;
  - direct_effect: their product used for channel ranking.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
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
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase132_source_value_contribution_cuda import get_num_kv_heads  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    expected_first_ids,
    expected_for,
    prompt_for,
    route_id_sets,
    select_base_cases,
)
from phase685_natural_value_readout_writer_localization import SHORT_VARIANT, TERSE_VARIANT, select_paired_cases, value_minus_prose_direction  # noqa: E402
from phase687_l26_l27_value_support_state_decomposition import paired_case_metadata  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase697_answer_last_route_transfer_decomposition import transfer_layers  # noqa: E402
from phase707_full_value_phrase_likelihood_audit import (  # noqa: E402
    COMBO_GROUPS,
    build_channel_scores,
    capture_state,
    choose_same_value_donor,
    choose_unrelated_donor,
    compact_conditions,
    condition_channel_sets,
    install_delta_hooks,
    load_top_head_sets,
    make_delta_patch_sets,
    score_channels_for_case,
    target_prose_phrase,
)


OUT_ROOT = Path("results/glm5_phase709_natural_generation_writein_closure")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def canonical_value_ids(tokenizer, value: str) -> set[int]:
    toks = tokenizer.encode(str(value).strip(), add_special_tokens=False)
    return {int(toks[0])} if toks else set()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def generated_category(generated: str, target_value: str, donor_value: str, target_prose: str) -> str:
    gen = normalize_text(generated)
    first_line = normalize_text(generated.splitlines()[0] if generated.splitlines() else generated)
    target = normalize_value(target_value)
    donor = normalize_value(donor_value)
    prose = normalize_text(target_prose)
    compact_first = normalize_value(first_line)
    compact_gen = normalize_value(gen)
    target_exact = compact_first == target or compact_first.startswith(target + " ")
    donor_exact = compact_first == donor or compact_first.startswith(donor + " ")
    if target and target_exact and len(compact_first.split()) <= max(3, len(target.split()) + 1):
        return "target_value"
    if donor and donor_exact and len(compact_first.split()) <= max(3, len(donor.split()) + 1):
        return "donor_value"
    if target and target in compact_gen:
        return "prose_target"
    if donor and donor in compact_gen:
        return "prose_donor"
    if prose and normalize_value(prose)[:24] in compact_gen:
        return "prose_target"
    if not gen:
        return "continuation_failure"
    if gen.startswith(("q:", "question:", "answer:", "a:", "the answer")):
        return "continuation_failure"
    return "other"


def greedy_generate_once(
    model,
    tokenizer,
    device,
    prompt: str,
    patches: list[dict[str, Any]],
    layer_meta: dict[int, tuple[Any, int, int, int]],
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated: list[int] = []
    step0_logits = None
    for step in range(max_new_tokens):
        handles = install_delta_hooks(model, patches, layer_meta) if step == 0 and patches else []
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=torch.tensor([input_ids + generated], device=device),
                    return_dict=True,
                    use_cache=False,
                )
            logits = out.logits[0, -1].detach().float().cpu()
            if step == 0:
                step0_logits = logits
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                break
        finally:
            for handle in handles:
                handle.remove()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "generated_ids": generated,
        "generated_text": text,
        "first_token_id": generated[0] if generated else None,
        "step0_top_logit": None if step0_logits is None else float(step0_logits.max().item()),
    }


def channel_diagnostics(selected: list[dict[str, Any]]) -> dict[str, Any]:
    if not selected:
        return {
            "n_channels": 0,
            "mean_direct_effect": None,
            "mean_abs_effect": None,
            "mean_combo_delta_value": None,
            "mean_output_dir_proj": None,
        }

    def mean(key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in selected) / len(selected)

    return {
        "n_channels": len(selected),
        "mean_direct_effect": mean("mean_direct_effect"),
        "mean_abs_effect": mean("mean_abs_effect"),
        "mean_combo_delta_value": mean("mean_combo_delta_value"),
        "mean_output_dir_proj": mean("mean_output_dir_proj"),
        "positive_direct_fraction": sum(1 for row in selected if float(row.get("mean_direct_effect", 0.0)) > 0) / len(selected),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    cats = Counter(row["generation_category"] for row in rows)

    def rate(cat: str) -> float:
        return cats.get(cat, 0) / n if n else 0.0

    def mean_optional(key: str) -> float | None:
        vals = [row[key] for row in rows if row.get(key) is not None]
        return None if not vals else sum(vals) / len(vals)

    return {
        "n": n,
        "category_counts": dict(cats.most_common()),
        "target_value_rate": rate("target_value"),
        "donor_value_rate": rate("donor_value"),
        "prose_target_rate": rate("prose_target"),
        "prose_donor_rate": rate("prose_donor"),
        "continuation_failure_rate": rate("continuation_failure"),
        "other_rate": rate("other"),
        "target_or_target_prose_rate": rate("target_value") + rate("prose_target"),
        "donor_or_donor_prose_rate": rate("donor_value") + rate("prose_donor"),
        "mean_n_channels": mean_optional("n_channels"),
        "mean_direct_effect": mean_optional("mean_direct_effect"),
        "mean_combo_delta_value": mean_optional("mean_combo_delta_value"),
        "mean_output_dir_proj": mean_optional("mean_output_dir_proj"),
    }


def summarize_model(model_name: str, rows: list[dict[str, Any]], paired_ids: list[str], donor_pairs: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["donor_type"], row["phase_kind"], row["condition"])].append(row)
    by_condition = {
        f"{donor_type}|{kind}|{cond}": summarize_group(vals)
        for (donor_type, kind, cond), vals in grouped.items()
    }
    restore = sorted(
        [{"condition": key, **vals} for key, vals in by_condition.items() if "|restore|" in key],
        key=lambda r: (r["target_value_rate"], r["target_or_target_prose_rate"], -r["donor_or_donor_prose_rate"]),
        reverse=True,
    )
    degradation = sorted(
        [{"condition": key, **vals} for key, vals in by_condition.items() if "|degradation|" in key],
        key=lambda r: (r["continuation_failure_rate"] + r["other_rate"], r["donor_or_donor_prose_rate"]),
        reverse=True,
    )
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "n_donor_pairs": len(donor_pairs),
        "same_value_pairs": sum(1 for p in donor_pairs if p["donor_type"] == "same_value"),
        "unrelated_pairs": sum(1 for p in donor_pairs if p["donor_type"] == "unrelated"),
        "n_rows": len(rows),
        "best_restore_conditions": restore,
        "best_degradation_conditions": degradation,
        "by_condition": by_condition,
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {case["case_id"]: case for case in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    top_head_rows, head_sets = load_top_head_sets(args.model, args.top_heads)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    case_score_rows: dict[str, list[dict[str, Any]]] = {}
    case_cache: dict[str, dict[str, Any]] = {}
    donor_pairs: list[dict[str, str]] = []
    try:
        dtype = next(model.parameters()).dtype
        scan_layers = transfer_layers(args.model, len(get_layers(model)))
        layer_meta: dict[int, tuple[Any, int, int, int]] = {}
        for li in scan_layers:
            o_proj, n_heads, head_dim = head_meta(model, li)
            attn = get_attention_module(get_layers(model)[li])
            layer_meta[li] = (o_proj, n_heads, head_dim, get_num_kv_heads(model, attn, n_heads))
        head_sets = {li: heads for li, heads in head_sets.items() if li in layer_meta}

        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype).detach().cpu()
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)
            short_state = capture_state(
                model, tokenizer, device, short_prompt, case, scan_layers, direction, routes, expected_ids, layer_meta
            )
            terse_state = capture_state(
                model, tokenizer, device, terse_prompt, case, scan_layers, direction, routes, expected_ids, layer_meta
            )
            case_cache[case_id] = {
                "short_prompt": short_prompt,
                "terse_prompt": terse_prompt,
                "direction": direction,
                "short_state": short_state,
                "terse_state": terse_state,
                "canonical_value_ids": canonical_value_ids(tokenizer, meta[case_id]["value"]),
            }
            case_score_rows[case_id] = score_channels_for_case(layer_meta, head_sets, short_state, terse_state, direction)
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: captured/source-scored {idx}/{len(paired_ids)} paired cases")

        all_score_rows = []
        for case_id in paired_ids:
            all_score_rows.extend(case_score_rows[case_id])
        channel_scores = build_channel_scores(all_score_rows, layer_meta)
        counts = [int(x) for x in args.channel_counts.split(",") if x.strip()]
        conditions = compact_conditions(condition_channel_sets(channel_scores, counts, args.seed))

        for case_id in paired_ids:
            same = choose_same_value_donor(case_id, paired_ids, meta)
            if same is not None:
                donor_pairs.append({"target": case_id, "donor": same, "donor_type": "same_value"})
            unrelated = choose_unrelated_donor(case_id, paired_ids, meta, args.seed)
            if unrelated is not None:
                donor_pairs.append({"target": case_id, "donor": unrelated, "donor_type": "unrelated"})

        for idx, pair in enumerate(donor_pairs, 1):
            case_id = pair["target"]
            donor_id = pair["donor"]
            donor_type = pair["donor_type"]
            cur = case_cache[case_id]
            donor = case_cache[donor_id]
            target_case = case_map[case_id]
            target_prose = target_prose_phrase(target_case)
            for cond_name, selected in conditions.items():
                diag = channel_diagnostics(selected)
                for phase_kind, prompt, src_state, dst_state in [
                    ("restore", cur["short_prompt"], donor["terse_state"], cur["short_state"]),
                    ("degradation", cur["terse_prompt"], donor["short_state"], cur["terse_state"]),
                ]:
                    patches = make_delta_patch_sets(src_state, dst_state, selected)
                    gen = greedy_generate_once(model, tokenizer, device, prompt, patches, layer_meta, args.max_new_tokens)
                    category = generated_category(gen["generated_text"], meta[case_id]["value"], meta[donor_id]["value"], target_prose)
                    rows.append({
                        "case_id": case_id,
                        "donor_case_id": donor_id,
                        "donor_type": donor_type,
                        "family": meta[case_id]["family"],
                        "relation": meta[case_id]["relation"],
                        "value": meta[case_id]["value"],
                        "donor_value": meta[donor_id]["value"],
                        "same_value": meta[case_id]["value"] == meta[donor_id]["value"],
                        "phase_kind": phase_kind,
                        "condition": cond_name,
                        "prompt_variant": "short" if phase_kind == "restore" else "terse",
                        "generated_text": gen["generated_text"],
                        "generated_ids": gen["generated_ids"],
                        "first_token_id": gen["first_token_id"],
                        "generation_category": category,
                        **diag,
                    })
            if idx % args.log_every == 0 or idx == len(donor_pairs):
                log(f"{args.model}: natural generated {idx}/{len(donor_pairs)} donor pairs")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, rows, paired_ids, donor_pairs)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase709_{args.model}_natural_generation_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 709,
        "title": "Natural Generation and Write-In Closure Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "transfer_layers": scan_layers,
        "top_heads": args.top_heads,
        "source_combo": COMBO_GROUPS,
        "channel_counts": counts,
        "phase698_top_heads": top_head_rows,
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
    }
    (OUT_ROOT / f"phase709_{args.model}_natural_generation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase709_*_natural_generation_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 709,
        "title": "Natural Generation and Write-In Closure Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase709_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 709 Natural Generation and Write-In Closure Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | donor_pairs | best_restore | n | target_value | donor_value | prose_target | prose_donor | continuation | other |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        s = item["summary"]
        br = s["best_restore_conditions"][0] if s["best_restore_conditions"] else {}
        lines.append(
            f"| {item['model']} | {s['n_paired_cases']} | {s['n_donor_pairs']} | {br.get('condition','')} | "
            f"{br.get('n',0)} | {br.get('target_value_rate',0.0):.3f} | {br.get('donor_value_rate',0.0):.3f} | "
            f"{br.get('prose_target_rate',0.0):.3f} | {br.get('prose_donor_rate',0.0):.3f} | "
            f"{br.get('continuation_failure_rate',0.0):.3f} | {br.get('other_rate',0.0):.3f} |"
        )
    for item in models:
        lines.extend(["", f"## {item['model']}", ""])
        lines.append("| condition | n | target_value | donor_value | prose_target | prose_donor | continuation | other | mean_direct | mean_combo_delta | mean_output_proj |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        rows = item["summary"]["best_restore_conditions"][:18] + item["summary"]["best_degradation_conditions"][:18]
        seen = set()
        for row in rows:
            key = row["condition"]
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"| {key} | {row['n']} | {row['target_value_rate']:.3f} | {row['donor_value_rate']:.3f} | "
                f"{row['prose_target_rate']:.3f} | {row['prose_donor_rate']:.3f} | {row['continuation_failure_rate']:.3f} | {row['other_rate']:.3f} | "
                f"{(row['mean_direct_effect'] or 0.0):.6f} | {(row['mean_combo_delta_value'] or 0.0):.6f} | {(row['mean_output_dir_proj'] or 0.0):.6f} |"
            )
    (OUT_ROOT / "phase709_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-heads", type=int, default=32)
    parser.add_argument("--channel-counts", default="512")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=709)
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
