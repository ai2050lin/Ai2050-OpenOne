#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, select_evenly  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase741_threshold_candidate_causal_validation import parse_component_site  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase756_cross_domain_writer_control_downstream_carrier import expanded_candidates, run_logits  # noqa: E402
from phase765_commonsense_context_identity_closure_test import (  # noqa: E402
    build_cases,
    capture_state,
    route_ids_for_case,
)


PHASE767_ROOT = Path("tests/result/phase767_commonsense_failure_type_topk_audit")
PHASE770_ROOT = Path("tests/result/phase770_balanced_semantic_clean_fiber_reanalysis")
OUT_ROOT = Path("results/glm5_phase771_matched_causal_intervention_reliability_test")
RESULT_ROOT = Path("tests/result/phase771_matched_causal_intervention_reliability_test")

DEFAULT_SOURCE_GROUPS = ["instruction", "question", "object_tokens", "relation_tokens"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def phase767_path(model: str, round_name: str) -> Path:
    path = PHASE767_ROOT / round_name / f"phase767_{model}_rows.jsonl"
    if path.exists():
        return path
    return Path("results/glm5_phase767_commonsense_failure_type_topk_audit") / round_name / f"phase767_{model}_rows.jsonl"


def phase770_path(round_name: str) -> Path:
    path = PHASE770_ROOT / round_name / "phase770_balanced_semantic_clean_fiber_reanalysis.json"
    if path.exists():
        return path
    return Path("results/glm5_phase770_balanced_semantic_clean_fiber_reanalysis") / round_name / "phase770_balanced_semantic_clean_fiber_reanalysis.json"


def semantic_label(row: dict[str, Any]) -> str:
    if row.get("exact_target_top1"):
        return "exact_clean"
    if row.get("target_top1"):
        return "semantic_only"
    return "semantic_fail"


def stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["domain"]), str(row["relation"]), str(row["context_format"]))


def source_groups_for(args: argparse.Namespace) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def select_matched_case_ids(rows767: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    clean: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    fail: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows767:
        (clean if row.get("target_top1") else fail)[stratum(row)].append(row)
    pairs = []
    for key in sorted(set(clean) & set(fail)):
        n = min(len(clean[key]), len(fail[key]), args.max_per_stratum)
        for idx in range(n):
            pairs.append({"stratum": key, "clean": sorted(clean[key], key=lambda r: r["case_id"])[idx], "fail": sorted(fail[key], key=lambda r: r["case_id"])[idx]})
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = [pairs[i] for i in select_evenly(len(pairs), args.max_pairs)]
    selected = []
    for pair_idx, pair in enumerate(pairs, 1):
        for arm in ("clean", "fail"):
            row = pair[arm]
            selected.append(
                {
                    "pair_index": pair_idx,
                    "matched_arm": arm,
                    "stratum": list(pair["stratum"]),
                    "case_id": row["case_id"],
                    "phase767": row,
                }
            )
    return selected


def pair_info_map(phase770: dict[str, Any], model: str) -> dict[str, dict[str, Any]]:
    out = {}
    paired = phase770["by_model"][model]["paired_context_stability"]
    cos_values = [float(r["context_cosine"]) for r in paired.get("rows", []) if r.get("context_cosine") is not None]
    threshold = safe_mean(cos_values)
    if threshold is None:
        threshold = 0.0
    for row in paired.get("rows", []):
        info = {
            "pair_context_cosine": row.get("context_cosine"),
            "semantic_pair": row.get("semantic_pair"),
            "exact_pair": row.get("exact_pair"),
            "lexical_pair": row.get("lexical_pair"),
            "fiber_bucket": "fiber_high" if float(row.get("context_cosine") or 0.0) >= threshold else "fiber_low",
            "fiber_threshold": threshold,
        }
        out[row["question_case_id"]] = info
        out[row["statement_case_id"]] = info
    return out


def case_map_for(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    relation_filter = set(args.relations.split(",")) if args.relations else {"category", "edible", "grows_on_tree"}
    cases = build_cases(None, relation_filter)
    return {case["case_id"]: case for case in cases}


def margin(logits: torch.Tensor, target_id: int, contrast_id: int) -> float:
    return float(logits[target_id].item() - logits[contrast_id].item())


def audit_case(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    case: dict[str, Any],
    case_label: dict[str, Any],
    pair_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_groups: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    candidate_layers = sorted({parse_component_site(c["site"])[0] for c in candidates})
    state = capture_state(model, tokenizer, device, case, candidate_layers)
    target_id = get_first_token_id(tokenizer, case["answer"])
    contrast_id = get_first_token_id(tokenizer, case["contrast_answer"])
    route_ids = route_ids_for_case(tokenizer, case, target_id)
    base_target_diag = logit_diag(state["logits"], target_id)
    base_contrast_diag = logit_diag(state["logits"], contrast_id)
    base_margin = margin(state["logits"], target_id, contrast_id)
    rows = []
    answer_pos = state["answer_pos"]
    phase767 = case_label["phase767"]
    for cand in candidates:
        site = cand["site"]
        layer, _component = parse_component_site(site)
        head = int(cand["head"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        if not (0 <= head < n_heads):
            continue
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        for source_group in source_groups:
            src_positions = [int(p) for p in state["source_groups"].get(source_group, [])]
            if not src_positions:
                continue
            contribution = compute_source_contribution(
                state["attentions"][layer],
                state["values"][layer],
                [answer_pos],
                [src_positions],
                n_heads,
                num_kv_heads,
            )
            projected = project_source_contribution(model, layer, [head], contribution)
            direct = direct_delta_score(projected, unembed, target_id, route_ids)
            install = install_source_contribution_removal(model, site, [head], contribution)
            after_logits = run_logits(model, device, state["ids"], install)
            after_target_diag = logit_diag(after_logits, target_id)
            after_contrast_diag = logit_diag(after_logits, contrast_id)
            after_margin = margin(after_logits, target_id, contrast_id)
            target_logit_drop = float(state["logits"][target_id].item() - after_logits[target_id].item())
            contrast_logit_gain = float(after_logits[contrast_id].item() - state["logits"][contrast_id].item())
            margin_drop = float(base_margin - after_margin)
            rows.append(
                {
                    "row_kind": "matched_causal_intervention",
                    "case_id": case["case_id"],
                    "pair_index": case_label["pair_index"],
                    "matched_arm": case_label["matched_arm"],
                    "stratum": case_label["stratum"],
                    "object": case["object"],
                    "domain": case["domain"],
                    "relation": case["relation"],
                    "context_format": case["context_format"],
                    "target_answer": case["answer"],
                    "contrast_answer": case["contrast_answer"],
                    "phase767_exact_top1": bool(phase767.get("exact_target_top1")),
                    "phase767_semantic_top1": bool(phase767.get("target_top1")),
                    "semantic_label": semantic_label(phase767),
                    **pair_info,
                    "base_target_rank": base_target_diag["target_rank"],
                    "base_target_top1": base_target_diag["target_top1"],
                    "base_contrast_rank": base_contrast_diag["target_rank"],
                    "after_target_rank": after_target_diag["target_rank"],
                    "after_target_top1": after_target_diag["target_top1"],
                    "after_contrast_rank": after_contrast_diag["target_rank"],
                    "top1_loss": bool(base_target_diag["target_top1"]) and not bool(after_target_diag["target_top1"]),
                    "target_logit_drop": target_logit_drop,
                    "contrast_logit_gain": contrast_logit_gain,
                    "margin_drop_target_vs_contrast": margin_drop,
                    "site": site,
                    "layer": layer,
                    "head": head,
                    "subunit_id": cand["subunit_id"],
                    "candidate_kind": cand["candidate_kind"],
                    "selection": cand["selection"],
                    "control_of": cand.get("control_of"),
                    "source_group": source_group,
                    "source_positions_n": len(src_positions),
                    "attention_mass_to_source": attention_mass_for_group(state["attentions"][layer], head, answer_pos, src_positions),
                    "source_direct_score": direct,
                }
            )
    return rows


def group_summary(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        direct = [v.get("source_direct_score") or {} for v in vals]
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(vals),
                "mean_target_logit_drop": safe_mean([v.get("target_logit_drop") for v in vals]),
                "mean_margin_drop_target_vs_contrast": safe_mean([v.get("margin_drop_target_vs_contrast") for v in vals]),
                "mean_contrast_logit_gain": safe_mean([v.get("contrast_logit_gain") for v in vals]),
                "mean_attention_mass": safe_mean([v.get("attention_mass_to_source") for v in vals]),
                "mean_direct_target_boost": safe_mean([d.get("direct_target_boost") for d in direct]),
                "mean_direct_route_suppression": safe_mean([d.get("direct_total_route_suppression") for d in direct]),
                "top1_loss_rate": sum(1 for v in vals if v.get("top1_loss")) / len(vals) if vals else None,
                "candidate_kind_counts": dict(Counter(v.get("candidate_kind") for v in vals)),
            }
        )
        out.append(payload)
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, model_name: str, attn_impl: str) -> dict[str, Any]:
    return {
        "phase": 771,
        "title": "Matched Causal Intervention Reliability Test",
        "model": model_name,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_pairs": len({r["pair_index"] for r in rows}),
        "source_groups": source_groups_for(args),
        "by_matched_arm": group_summary(rows, ["matched_arm"]),
        "by_semantic_label": group_summary(rows, ["semantic_label"]),
        "by_fiber_bucket": group_summary(rows, ["fiber_bucket"]),
        "by_arm_and_fiber": group_summary(rows, ["matched_arm", "fiber_bucket"]),
        "by_candidate_kind": group_summary(rows, ["candidate_kind"]),
        "by_source_group": group_summary(rows, ["source_group"]),
        "strict_interpretation": "This is a matched direct source-contribution removal test. It is still head/source-level, not neuron/channel-level.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows767 = load_jsonl(phase767_path(args.model, args.phase767_round))
    phase770 = load_json(phase770_path(args.phase770_round))
    selected = select_matched_case_ids(rows767, args)
    cmap = case_map_for(args)
    pinfo = pair_info_map(phase770, args.model)
    source_groups = source_groups_for(args)
    log(f"{args.model}/{args.round_name}: matched cases={len(selected)} pairs={len({x['pair_index'] for x in selected})} sources={source_groups}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        candidates = expanded_candidates(model, args.model, args)
        unembed = get_unembed(model)
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(selected, 1):
            case = cmap[item["case_id"]]
            rows.extend(audit_case(model, tokenizer, device, args, case, item, pinfo.get(case["case_id"], {}), candidates, source_groups, unembed))
            if idx % args.log_every == 0 or idx == len(selected):
                log(f"{args.model}: matched causal cases {idx}/{len(selected)} rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, args.model, attn_impl)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase771_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase771_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "n_cases": summary["n_cases"], "by_matched_arm": summary["by_matched_arm"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 771 Matched Causal Intervention Reliability Test ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: matched semantic-clean vs semantic-fail direct source-contribution removal.",
        "- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.",
        "",
        "## Matched Arm Summary",
        "",
        "| model | arm | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_matched_arm"]:
            lines.append(
                f"| {model} | `{row['matched_arm']}` | {row['n']} | {fmt(row['mean_target_logit_drop'])} | "
                f"{fmt(row['mean_margin_drop_target_vs_contrast'])} | {fmt(row['top1_loss_rate'])} | "
                f"{fmt(row['mean_attention_mass'])} | {fmt(row['mean_direct_target_boost'])} | {fmt(row['mean_direct_route_suppression'])} |"
            )
    lines += [
        "",
        "## Semantic Label Summary",
        "",
        "| model | semantic label | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_semantic_label"]:
            lines.append(
                f"| {model} | `{row['semantic_label']}` | {row['n']} | {fmt(row['mean_target_logit_drop'])} | "
                f"{fmt(row['mean_margin_drop_target_vs_contrast'])} | {fmt(row['top1_loss_rate'])} | "
                f"{fmt(row['mean_attention_mass'])} | {fmt(row['mean_direct_target_boost'])} | {fmt(row['mean_direct_route_suppression'])} |"
            )
    lines += [
        "",
        "## Fiber Bucket Summary",
        "",
        "| model | fiber bucket | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_fiber_bucket"]:
            lines.append(
                f"| {model} | `{row['fiber_bucket']}` | {row['n']} | {fmt(row['mean_target_logit_drop'])} | "
                f"{fmt(row['mean_margin_drop_target_vs_contrast'])} | {fmt(row['top1_loss_rate'])} | "
                f"{fmt(row['mean_attention_mass'])} | {fmt(row['mean_direct_target_boost'])} | {fmt(row['mean_direct_route_suppression'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- If clean rows have larger target/margin drops than fail rows, output-clean states are more causally dependent on the tested source paths.",
        "- If fiber-high rows have larger drops than fiber-low rows, paired fiber stability predicts intervention sensitivity.",
        "- This is still a head/source intervention, not a neuron/channel atlas.",
        "- Because the test uses allowed-value commonsense prompts, it does not prove free-generation closure.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase771_{model}_summary.json"
        if path.exists():
            by_model[model] = load_json(path)
    payload = {
        "phase": 771,
        "title": "Matched Causal Intervention Reliability Test",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase771_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase771_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "models": {}}
    for model in MODELS:
        rows767 = load_jsonl(phase767_path(model, args.phase767_round))
        selected = select_matched_case_ids(rows767, args)
        payload["models"][model] = {
            "selected_cases": len(selected),
            "pairs": len({x["pair_index"] for x in selected}),
            "arms": dict(Counter(x["matched_arm"] for x in selected)),
            "source_groups": source_groups_for(args),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase767-round", default="main")
    parser.add_argument("--phase770-round", default="confirm_x_main")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--max-per-stratum", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--controls-per-candidate", type=int, default=1)
    parser.add_argument("--control-offset", type=int, default=3)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--max-source-groups", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --dry-run or --summarize-only")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
