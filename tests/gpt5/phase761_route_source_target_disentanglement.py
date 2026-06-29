#!/usr/bin/env python3
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
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase741_threshold_candidate_causal_validation import parse_component_site  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    capture_attention_value_state,
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group, norm  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id, select_global_pairs  # noqa: E402
from phase756_cross_domain_writer_control_downstream_carrier import expanded_candidates, run_logits  # noqa: E402
from phase760_route_suppression_matrix_atlas import (  # noqa: E402
    build_explicit_route_groups,
    route_ids_from_groups,
    route_matrix,
)


OUT_ROOT = Path("results/glm5_phase761_route_source_target_disentanglement")

BASE_SOURCE_GROUPS = ["target_record_line", "target_value_tokens", "records_all"]
ROUTE_SOURCE_PRIORITY = [
    "contrast_answer",
    "top_class:recipient_answer",
    "other_record_value",
    "generic_answer",
    "format_schema",
    "object_relation_echo",
    "top_non_target",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def source_groups_for(args: argparse.Namespace) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return BASE_SOURCE_GROUPS[: args.max_base_source_groups]


def source_family(name: str) -> str:
    if name in {"target_record_line", "target_value_tokens"}:
        return "target_source"
    if name == "records_all":
        return "broad_record_source"
    if name == "route_src:any_route":
        return "route_source_union"
    if name.startswith("route_src:format_schema"):
        return "format_source"
    if name.startswith("route_src:object_relation_echo"):
        return "echo_source"
    if name.startswith("route_src:"):
        return "route_token_source"
    return "other_source"


def positions_from_route_group(ids: list[int], answer_pos: int, group: dict[str, Any]) -> list[int]:
    token_ids = {int(tok["token_id"]) for tok in group.get("tokens", [])}
    if not token_ids:
        return []
    return [idx for idx, tid in enumerate(ids[:answer_pos]) if int(tid) in token_ids]


def dynamic_route_source_groups(
    ids: list[int],
    answer_pos: int,
    route_groups: dict[str, dict[str, Any]],
    max_route_source_groups: int,
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    ordered_names = []
    for name in ROUTE_SOURCE_PRIORITY:
        if name in route_groups:
            ordered_names.append(name)
    for name in sorted(route_groups):
        if name not in ordered_names:
            ordered_names.append(name)
    for route_name in ordered_names:
        pos = positions_from_route_group(ids, answer_pos, route_groups[route_name])
        if not pos:
            continue
        out[f"route_src:{route_name}"] = sorted(set(pos))
        if len(out) >= max_route_source_groups:
            break
    union = sorted({p for xs in out.values() for p in xs})
    if union:
        out = {"route_src:any_route": union, **out}
    return out


def merged_source_groups(state: dict[str, Any], route_groups: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name in source_groups_for(args):
        pos = [int(p) for p in state["source_groups"].get(name, [])]
        if pos:
            out[name] = sorted(set(pos))
    out.update(dynamic_route_source_groups(state["ids"], state["answer_pos"], route_groups, args.max_route_source_groups))
    if args.max_total_source_groups and len(out) > args.max_total_source_groups:
        keep: dict[str, list[int]] = {}
        for name in source_groups_for(args):
            if name in out:
                keep[name] = out[name]
        for name in sorted(out):
            if name in keep:
                continue
            keep[name] = out[name]
            if len(keep) >= args.max_total_source_groups:
                break
        out = keep
    return out


def route_success(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return float(row.get("route_release") or 0.0) >= args.min_route_release


def target_drop_success(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return float(row.get("target_logit_drop") or 0.0) >= args.min_target_drop


def target_boost(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return float(row.get("target_logit_drop") or 0.0) <= -args.min_target_boost


def classify_source_role(vals: list[dict[str, Any]], source_group: str, args: argparse.Namespace) -> str:
    if not vals:
        return "empty"
    n = len(vals)
    route_rate = sum(route_success(v, args) for v in vals) / n
    target_drop_rate = sum(target_drop_success(v, args) for v in vals) / n
    target_boost_rate = sum(target_boost(v, args) for v in vals) / n
    mean_route = safe_mean([v["route_release"] for v in vals]) or 0.0
    mean_target_drop = safe_mean([v["target_logit_drop"] for v in vals]) or 0.0
    family = source_family(source_group)
    if target_boost_rate >= args.artifact_rate and route_rate >= args.route_rate:
        return "negative_target_drop_route_artifact"
    if family in {"target_source", "broad_record_source"} and target_drop_rate >= args.target_rate and mean_target_drop >= args.min_target_drop:
        if route_rate >= args.route_rate and mean_route >= args.min_route_release:
            return "target_source_with_route_release"
        return "target_source_writer"
    if family in {"route_source_union", "route_token_source", "format_source", "echo_source"}:
        if route_rate >= args.route_rate and target_drop_rate < args.target_rate:
            return "route_source_release_without_target_drop"
        if route_rate >= args.route_rate and target_drop_rate >= args.target_rate:
            return "mixed_route_and_target_source"
    if route_rate >= args.route_rate and mean_route >= args.min_route_release:
        return "route_release_unclear_source"
    return "weak_or_unclear"


def audit_pair(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    pair: dict[str, Any],
    candidates: list[dict[str, Any]],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    target = pair["explicit_profile"]
    contrast = pair["conflict_profile"]
    candidate_layers = sorted({parse_component_site(c["site"])[0] for c in candidates})
    state = capture_attention_value_state(model, tokenizer, device, target, candidate_layers)
    target_id = get_first_token_id(tokenizer, target["answer"])
    contrast_id = get_first_token_id(tokenizer, contrast["answer"])
    route_groups = build_explicit_route_groups(tokenizer, state["logits"], target, contrast, target_id, contrast_id, args)
    if not route_groups:
        return []
    source_map = merged_source_groups(state, route_groups, args)
    if not source_map:
        return []
    route_ids = route_ids_from_groups(route_groups)
    target_diag = logit_diag(state["logits"], target_id)
    contrast_diag = logit_diag(state["logits"], contrast_id)
    answer_pos = state["answer_pos"]
    rows: list[dict[str, Any]] = [
        {
            "row_kind": "base_route_source_map",
            "pair_id": pair["pair_id"],
            "domain": target["domain"],
            "object": target["object"],
            "relation": target["relation"],
            "target_answer": target["answer"],
            "contrast_answer": contrast["answer"],
            "target_token_id": target_id,
            "contrast_token_id": contrast_id,
            "target_rank": target_diag["target_rank"],
            "target_top1": target_diag["target_top1"],
            "contrast_rank": contrast_diag["target_rank"],
            "route_group_names": sorted(route_groups),
            "source_groups": {k: {"family": source_family(k), "positions_n": len(v)} for k, v in source_map.items()},
            "route_groups": route_groups,
        }
    ]

    for cand in candidates:
        site = cand["site"]
        layer, _component = parse_component_site(site)
        head = int(cand["head"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        if not (0 <= head < n_heads):
            continue
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        for source_group, src_positions in source_map.items():
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
            removal_install = install_source_contribution_removal(model, site, [head], contribution)
            after_logits = run_logits(model, device, state["ids"], removal_install)
            target_drop = float(state["logits"][target_id].item() - after_logits[target_id].item())
            matrix = route_matrix(state["logits"], after_logits, route_groups, target_id)
            rows.append(
                {
                    "row_kind": "source_removal_overview",
                    "pair_id": pair["pair_id"],
                    "domain": target["domain"],
                    "object": target["object"],
                    "relation": target["relation"],
                    "target_answer": target["answer"],
                    "contrast_answer": contrast["answer"],
                    "site": site,
                    "layer": layer,
                    "head": head,
                    "subunit_id": cand["subunit_id"],
                    "candidate_kind": cand["candidate_kind"],
                    "selection": cand["selection"],
                    "control_of": cand.get("control_of"),
                    "source_group": source_group,
                    "source_family": source_family(source_group),
                    "source_positions_n": len(src_positions),
                    "attention_mass_to_source": attention_mass_for_group(state["attentions"][layer], head, answer_pos, src_positions),
                    "source_projected_delta_norm": norm(projected),
                    "source_direct_score": direct,
                    "target_logit_drop": target_drop,
                    "total_positive_route_release": float(sum(max(0.0, float(v["route_release"])) for v in matrix.values())),
                }
            )
            for route_group, cell in matrix.items():
                rows.append(
                    {
                        "row_kind": "route_source_disentangle_cell",
                        "pair_id": pair["pair_id"],
                        "domain": target["domain"],
                        "object": target["object"],
                        "relation": target["relation"],
                        "target_answer": target["answer"],
                        "contrast_answer": contrast["answer"],
                        "site": site,
                        "layer": layer,
                        "head": head,
                        "subunit_id": cand["subunit_id"],
                        "candidate_kind": cand["candidate_kind"],
                        "selection": cand["selection"],
                        "control_of": cand.get("control_of"),
                        "source_group": source_group,
                        "source_family": source_family(source_group),
                        "source_positions_n": len(src_positions),
                        "attention_mass_to_source": attention_mass_for_group(state["attentions"][layer], head, answer_pos, src_positions),
                        "source_projected_delta_norm": norm(projected),
                        "source_direct_score": direct,
                        "route_group": route_group,
                        "target_logit_drop": target_drop,
                        "route_release": cell["route_release"],
                        "margin_drop_target_vs_route": cell["margin_drop_target_vs_route"],
                        "base_route_max_token_text": cell["base_route_max_token_text"],
                        "after_route_max_token_text": cell["after_route_max_token_text"],
                        "base_route_max_token_label": cell["base_route_max_token_label"],
                        "after_route_max_token_label": cell["after_route_max_token_label"],
                        "route_token_count": cell["token_count"],
                    }
                )
    return rows


def summarize_cells(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    cells = [r for r in rows if r["row_kind"] == "route_source_disentangle_cell"]
    groups: dict[tuple[str, int, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        groups[(row["site"], int(row["head"]), row["candidate_kind"], row["source_group"], row["source_family"], row["route_group"])].append(row)
    out: list[dict[str, Any]] = []
    for (site, head, candidate_kind, source_group, family, route_group), vals in groups.items():
        n = len(vals)
        target_drops = [v["target_logit_drop"] for v in vals]
        route_releases = [v["route_release"] for v in vals]
        out.append(
            {
                "site": site,
                "head": head,
                "subunit_id": f"{site}:H{head}",
                "candidate_kind": candidate_kind,
                "source_group": source_group,
                "source_family": family,
                "route_group": route_group,
                "n": n,
                "domains": sorted({v["domain"] for v in vals}),
                "relations": sorted({v["relation"] for v in vals}),
                "mean_attention_mass_to_source": safe_mean([v["attention_mass_to_source"] for v in vals]),
                "mean_source_target_logit_contribution": safe_mean([v["source_direct_score"]["direct_target_boost"] for v in vals]),
                "mean_source_total_route_suppression_contribution": safe_mean([v["source_direct_score"]["direct_total_route_suppression"] for v in vals]),
                "mean_target_logit_drop": safe_mean(target_drops),
                "target_drop_rate": sum(target_drop_success(v, args) for v in vals) / n,
                "target_boost_rate": sum(target_boost(v, args) for v in vals) / n,
                "mean_route_release": safe_mean(route_releases),
                "route_release_rate": sum(route_success(v, args) for v in vals) / n,
                "mean_margin_drop_target_vs_route": safe_mean([v["margin_drop_target_vs_route"] for v in vals]),
                "role_guess": classify_source_role(vals, source_group, args),
                "after_route_token_counts": dict(Counter(v["after_route_max_token_text"] for v in vals).most_common(8)),
            }
        )
    out.sort(
        key=lambda r: (
            r["route_release_rate"],
            r["mean_route_release"] or 0.0,
            -r["target_drop_rate"],
            -(r["target_boost_rate"]),
        ),
        reverse=True,
    )
    return out


def summarize_by_source_family(cell_summary: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_summary:
        groups[(row["source_family"], row["route_group"])].append(row)
    out = {}
    for (family, route_group), vals in sorted(groups.items()):
        out[f"{family}::{route_group}"] = {
            "source_family": family,
            "route_group": route_group,
            "n_groups": len(vals),
            "mean_route_release_rate": safe_mean([v["route_release_rate"] for v in vals]),
            "mean_target_drop_rate": safe_mean([v["target_drop_rate"] for v in vals]),
            "mean_target_boost_rate": safe_mean([v["target_boost_rate"] for v in vals]),
            "mean_route_release": safe_mean([v["mean_route_release"] for v in vals]),
            "mean_target_drop": safe_mean([v["mean_target_logit_drop"] for v in vals]),
            "role_counts": dict(Counter(v["role_guess"] for v in vals)),
        }
    return out


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    attn_impl: str,
) -> dict[str, Any]:
    cell_summary = summarize_cells(rows, args)
    return {
        "phase": 761,
        "title": "Route Source Target Disentanglement",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_base_route_source_maps": sum(1 for r in rows if r["row_kind"] == "base_route_source_map"),
        "n_source_removal_overviews": sum(1 for r in rows if r["row_kind"] == "source_removal_overview"),
        "n_route_source_cells": sum(1 for r in rows if r["row_kind"] == "route_source_disentangle_cell"),
        "candidates": candidates,
        "base_source_groups": source_groups_for(args),
        "source_family_route_baseline": summarize_by_source_family(cell_summary),
        "top_route_source_cells": cell_summary[:128],
        "strict_interpretation": "This phase compares target-record and route-token source removals. A route-source cell is a source-level clue, not a global suppressor or neuron-level mechanism.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_global_pairs(args.max_pairs)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} base_sources={source_groups_for(args)}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        candidates = expanded_candidates(model, args.model, args)
        unembed = get_unembed(model)
        log(f"{args.model}: candidates={len(candidates)} max_route_sources={args.max_route_source_groups}")
        rows: list[dict[str, Any]] = []
        for idx, pair in enumerate(pairs, 1):
            rows.extend(audit_pair(model, tokenizer, device, args, pair, candidates, unembed))
            if idx % args.log_every == 0 or idx == len(pairs):
                log(f"{args.model}: source-target disentangle {idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args, rows, candidates, attn_impl)
    write_jsonl(out_dir / f"phase761_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase761_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "n_rows": summary["n_rows"],
                "top_route_source_cells": summary["top_route_source_cells"][:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model_name in MODELS:
        path = out_dir / f"phase761_{model_name}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 761,
        "title": "Route Source Target Disentanglement",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "Compares source families for route release after attention-head source-contribution removal. It is still head-level source evidence.",
    }
    write_json(out_dir / "phase761_cross_model_summary.json", payload)
    lines = [
        f"# Phase 761 Route Source Target Disentanglement ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Test: remove the same candidate head contribution from target-record sources and route-token sources, then measure target drop and route release separately.",
        "",
        "## Source Family x Route Group",
        "",
        "| model | source family | route group | groups | route rate | target drop rate | target boost rate | route release | target drop | roles |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, summary in payload["by_model"].items():
        rows = list(summary.get("source_family_route_baseline", {}).values())
        rows.sort(
            key=lambda r: (
                r.get("mean_route_release_rate") or 0.0,
                r.get("mean_route_release") or 0.0,
                -(r.get("mean_target_drop_rate") or 0.0),
            ),
            reverse=True,
        )
        for row in rows[:40]:
            lines.append(
                f"| {model_name} | `{row['source_family']}` | `{row['route_group']}` | {row['n_groups']} | "
                f"{(row.get('mean_route_release_rate') or 0):.3f} | "
                f"{(row.get('mean_target_drop_rate') or 0):.3f} | "
                f"{(row.get('mean_target_boost_rate') or 0):.3f} | "
                f"{(row.get('mean_route_release') or 0):.3f} | "
                f"{(row.get('mean_target_drop') or 0):.3f} | `{row.get('role_counts')}` |"
            )
    lines.extend(
        [
            "",
            "## Top Cells",
            "",
            "| model | head | kind | source | family | route | n | route rate | target drop rate | target boost rate | route release | target drop | role |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_route_source_cells", [])[:36]:
            lines.append(
                f"| {model_name} | {row['subunit_id']} | `{row['candidate_kind']}` | `{row['source_group']}` | `{row['source_family']}` | "
                f"`{row['route_group']}` | {row['n']} | {row['route_release_rate']:.3f} | {row['target_drop_rate']:.3f} | "
                f"{row['target_boost_rate']:.3f} | {(row.get('mean_route_release') or 0):.3f} | "
                f"{(row.get('mean_target_logit_drop') or 0):.3f} | `{row['role_guess']}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- If route-token sources release routes without target drop, route competition has source-family evidence distinct from target writer evidence.",
            "- If target sources and route sources both release routes, the result supports a mixed or distributed route field.",
            "- If same-layer control heads match the candidate heads, the effect is not specific enough to call a global suppressor.",
            "",
        ]
    )
    (out_dir / "phase761_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {
        "phase": 761,
        "round": args.round_name,
        "pairs": len(select_global_pairs(args.max_pairs)),
        "max_candidates": args.max_candidates,
        "base_source_groups": source_groups_for(args),
        "route_source_priority": ROUTE_SOURCE_PRIORITY,
        "max_route_source_groups": args.max_route_source_groups,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--include-controls", action="store_true", default=True)
    parser.add_argument("--controls-per-candidate", type=int, default=1)
    parser.add_argument("--control-offset", type=int, default=13)
    parser.add_argument("--max-base-source-groups", type=int, default=3)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--max-route-source-groups", type=int, default=5)
    parser.add_argument("--max-total-source-groups", type=int, default=9)
    parser.add_argument("--top-k-vocab", type=int, default=18)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-dynamic-route-classes", type=int, default=5)
    parser.add_argument("--min-target-drop", type=float, default=0.20)
    parser.add_argument("--min-target-boost", type=float, default=0.10)
    parser.add_argument("--min-route-release", type=float, default=0.10)
    parser.add_argument("--route-rate", type=float, default=0.30)
    parser.add_argument("--target-rate", type=float, default=0.30)
    parser.add_argument("--artifact-rate", type=float, default=0.30)
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only or --dry-run is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
