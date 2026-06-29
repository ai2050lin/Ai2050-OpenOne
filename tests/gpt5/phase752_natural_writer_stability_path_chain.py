#!/usr/bin/env python3
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
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase739_readout_threshold_closure_boundary import choose_donor_recipient, get_unembed  # noqa: E402
from phase740_natural_readout_boost_source_backtrace import load_phase739_audits  # noqa: E402
from phase741_threshold_candidate_causal_validation import parse_component_site  # noqa: E402
from phase743_competitor_format_suppression_audit import taxonomy_context  # noqa: E402
from phase748_natural_route_suppressor_matrix import route_max_logits, selected_distribution, js_divergence  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score, route_token_ids  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    SOURCE_GROUPS,
    build_route_context,
    capture_attention_value_state,
    eval_after_logits,
    install_source_contribution_removal,
    project_source_contribution,
)
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402


OUT_ROOT = Path("results/glm5_phase752_natural_writer_stability_path_chain")
PHASE751_ROOT = Path("results/glm5_phase751_natural_attention_head_mechanism_backtrace")

DEFAULT_SOURCE_GROUPS = [
    "target_record_line",
    "target_value_tokens",
    "records_all",
    "relation_tokens",
    "object_tokens",
    "question",
    "instruction",
    "records_other",
]

FIXED_CANDIDATES = {
    "qwen3": [
        {"site": "L33:attn_out", "head": 15, "source": "phase751_confirm"},
        {"site": "L33:attn_out", "head": 23, "source": "phase751_confirm"},
        {"site": "L32:attn_out", "head": 11, "source": "phase751_confirm"},
        {"site": "L32:attn_out", "head": 0, "source": "phase751_confirm"},
    ],
    "glm4": [
        {"site": "L35:attn_out", "head": 29, "source": "phase751_confirm"},
        {"site": "L34:attn_out", "head": 4, "source": "phase751_focus"},
        {"site": "L34:attn_out", "head": 9, "source": "phase751_confirm"},
    ],
    "deepseek7b": [
        {"site": "L22:attn_out", "head": 24, "source": "phase751_confirm"},
        {"site": "L22:attn_out", "head": 1, "source": "phase750_751_focus"},
        {"site": "L22:attn_out", "head": 7, "source": "phase750_751_focus"},
        {"site": "L23:attn_out", "head": 6, "source": "phase751_confirm"},
    ],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec.float()).item())


def source_groups_for(args) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def candidate_key(site: str, head: int) -> str:
    return f"{site}:H{int(head)}"


def head_from_subunit(subunit_id: str) -> int | None:
    m = re.search(r":H(\d+)$", subunit_id)
    return int(m.group(1)) if m else None


def load_candidates(args) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for cand in FIXED_CANDIDATES[args.model]:
        key = (cand["site"], int(cand["head"]))
        rows.append(
            {
                "site": cand["site"],
                "head": int(cand["head"]),
                "subunit_id": candidate_key(cand["site"], int(cand["head"])),
                "selection": cand["source"],
            }
        )
        seen.add(key)
    path = PHASE751_ROOT / args.phase751_round / f"phase751_{args.model}_summary.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("top_source_mechanism_candidates", []):
            if row.get("subunit_kind") not in {"attn_head", "attn_head_focus"}:
                continue
            if float(row.get("mean_target_logit_drop_after_source_removal") or 0.0) < args.min_phase751_drop:
                continue
            head = head_from_subunit(str(row.get("subunit_id", "")))
            if head is None:
                continue
            key = (row["site"], int(head))
            if key in seen:
                continue
            rows.append(
                {
                    "site": row["site"],
                    "head": int(head),
                    "subunit_id": candidate_key(row["site"], int(head)),
                    "selection": "auto_from_phase751_confirm",
                    "phase751_source_group": row.get("source_group"),
                    "phase751_mean_target_drop": row.get("mean_target_logit_drop_after_source_removal"),
                    "phase751_mean_route_release": row.get("mean_total_positive_route_release_after_source_removal"),
                }
            )
            seen.add(key)
            if len(rows) >= args.max_candidates:
                break
    return rows[: args.max_candidates]


def hidden_layers_for(candidates: list[dict[str, Any]], n_layers: int) -> list[int]:
    layers: set[int] = set()
    for cand in candidates:
        li, _ = parse_component_site(cand["site"])
        for x in [li + 1, li + 2, n_layers - 1]:
            if 0 <= x < n_layers:
                layers.add(x)
    return sorted(layers)


def run_logits_hidden(
    model,
    device,
    ids: list[int],
    layers: list[int],
    install: Callable[[], list[Any]] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    handles = install() if install else []
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_hidden_states=True,
            )
        logits = out.logits[0, -1].detach().float().cpu()
        hidden: dict[int, torch.Tensor] = {}
        for li in layers:
            idx = min(li + 1, len(out.hidden_states) - 1)
            hidden[li] = out.hidden_states[idx][0, -1].detach().float().cpu()
        return logits, hidden
    finally:
        for h in handles:
            h.remove()


def attention_mass_for_group(attn_array, head: int, answer_pos: int, source_positions: list[int]) -> float:
    if not source_positions:
        return 0.0
    row = torch.tensor(attn_array[0, head, answer_pos, :], dtype=torch.float32)
    idxs = [i for i in source_positions if 0 <= i < row.numel()]
    return float(row[idxs].sum().item()) if idxs else 0.0


def audit_context(
    model,
    tokenizer,
    device,
    args,
    context_name: str,
    target_item: dict[str, Any],
    contrast_item: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_groups: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    candidate_layers = sorted({parse_component_site(c["site"])[0] for c in candidates})
    state = capture_attention_value_state(model, tokenizer, device, target_item, candidate_layers)
    ctx = taxonomy_context(tokenizer, target_item, contrast_item)
    target_id = int(ctx["donor_id"])
    contrast_id = int(ctx["recipient_id"])
    route_ctx = build_route_context(state["logits"], tokenizer, ctx, target_id, args.top_k_vocab, args.max_topk_tokens, args.max_route_classes)
    if route_ctx is None:
        return []
    route_ids = route_token_ids(route_ctx["route_max"])
    hidden_layers = hidden_layers_for(candidates, len(get_layers(model)))
    _base_logits2, base_hidden = run_logits_hidden(model, device, state["ids"], hidden_layers)
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        site = cand["site"]
        layer, _component = parse_component_site(site)
        head = int(cand["head"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        if head < 0 or head >= n_heads:
            continue
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        answer_pos = state["answer_pos"]
        for source_group in source_groups:
            src_positions = state["source_groups"].get(source_group, [])
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
            after_logits, after_hidden = run_logits_hidden(model, device, state["ids"], hidden_layers, install)
            metrics = eval_after_logits(state["logits"], after_logits, route_ctx, target_id, contrast_id)
            chain_norms = {f"L{k}": norm(after_hidden[k] - base_hidden[k]) for k in hidden_layers if k in after_hidden and k in base_hidden}
            rows.append(
                {
                    "context_name": context_name,
                    "prompt_type": target_item.get("prompt_type"),
                    "object": target_item["object"],
                    "object_group": target_item.get("object_group"),
                    "relation": target_item["relation"],
                    "answer": target_item["answer"],
                    "contrast_answer": contrast_item["answer"],
                    "site": site,
                    "layer": layer,
                    "head": head,
                    "subunit_id": cand["subunit_id"],
                    "selection": cand["selection"],
                    "source_group": source_group,
                    "source_positions_n": len(src_positions),
                    "attention_mass_to_source": attention_mass_for_group(state["attentions"][layer], head, answer_pos, src_positions),
                    "source_projected_delta_norm": norm(projected),
                    "source_direct_score": direct,
                    "downstream_hidden_delta_norms_after_removal": chain_norms,
                    "final_hidden_delta_norm_after_removal": chain_norms.get(f"L{len(get_layers(model)) - 1}"),
                    **metrics,
                }
            )
    return rows


def audit_pair(
    model,
    tokenizer,
    device,
    args,
    pair: dict[str, Any],
    audit: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_groups: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    donor, recipient = choose_donor_recipient(pair, audit["direction"])
    contexts = [("natural_donor", donor, recipient)]
    if not args.donor_context_only:
        contexts.append(("natural_recipient", recipient, donor))
    rows: list[dict[str, Any]] = []
    for context_name, target_item, contrast_item in contexts:
        for row in audit_context(model, tokenizer, device, args, context_name, target_item, contrast_item, candidates, source_groups, unembed):
            row.update({"pair_id": pair["pair_id"], "direction": audit["direction"], "intervention_label": intervention_label(audit["intervention"])})
            rows.append(row)
    return rows


def breakdown(vals: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in vals:
        groups[str(v.get(key))].append(v)
    for name, xs in groups.items():
        out[name] = {
            "n": len(xs),
            "mean_target_drop": safe_mean([x["target_logit_drop"] for x in xs]),
            "mean_route_release": safe_mean([x["total_positive_route_release"] for x in xs]),
            "top1_loss_rate": sum(1 for x in xs if x["top1_loss"]) / len(xs),
            "support_rate": sum(1 for x in xs if x["target_logit_drop"] > 0.20) / len(xs),
            "route_guard_rate": sum(1 for x in xs if x["total_positive_route_release"] > 0.20) / len(xs),
        }
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["mean_target_drop"] or 0.0), reverse=True))


def classify_stability(vals: list[dict[str, Any]]) -> str:
    n = len(vals)
    if not n:
        return "empty"
    mean_drop = safe_mean([v["target_logit_drop"] for v in vals]) or 0.0
    mean_release = safe_mean([v["total_positive_route_release"] for v in vals]) or 0.0
    support_rate = sum(1 for v in vals if v["target_logit_drop"] > 0.20) / n
    route_rate = sum(1 for v in vals if v["total_positive_route_release"] > 0.20) / n
    rels = breakdown(vals, "relation")
    top_rel = max((x["mean_target_drop"] or 0.0 for x in rels.values()), default=0.0)
    answers = breakdown(vals, "answer")
    top_ans = max((x["mean_target_drop"] or 0.0 for x in answers.values()), default=0.0)
    if support_rate >= 0.50 and mean_drop >= 0.35 and len(rels) >= 3:
        if route_rate >= 0.25 or mean_release >= 0.20:
            return "stable_mixed_writer_guard"
        return "stable_target_writer"
    if top_rel >= 0.45 and top_rel > max(0.10, 2.0 * mean_drop):
        return "relation_conditioned_writer"
    if top_ans >= 0.45 and top_ans > max(0.10, 2.0 * mean_drop):
        return "answer_value_specific_writer"
    if route_rate >= 0.35 and mean_release >= 0.20:
        return "route_guard_without_stable_target_support"
    return "weak_or_unstable"


def summarize_rows(rows: list[dict[str, Any]], args, candidates: list[dict[str, Any]], source_groups: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["context_name"], row["site"], int(row["head"]), row["subunit_id"], row["source_group"])].append(row)
    summary = []
    for (context_name, site, head, subunit_id, source_group), vals in grouped.items():
        n = len(vals)
        rel_break = breakdown(vals, "relation")
        ans_break = breakdown(vals, "answer")
        obj_break = breakdown(vals, "object_group")
        summary.append(
            {
                "context_name": context_name,
                "site": site,
                "head": head,
                "subunit_id": subunit_id,
                "source_group": source_group,
                "n": n,
                "objects": sorted({v["object"] for v in vals}),
                "relations": sorted({v["relation"] for v in vals}),
                "answers": sorted({v["answer"] for v in vals}),
                "mean_attention_mass_to_source": safe_mean([v["attention_mass_to_source"] for v in vals]),
                "mean_source_target_logit_contribution": safe_mean([v["source_direct_score"]["direct_target_boost"] for v in vals]),
                "mean_source_total_route_suppression_contribution": safe_mean([v["source_direct_score"]["direct_total_route_suppression"] for v in vals]),
                "mean_target_logit_drop_after_source_removal": safe_mean([v["target_logit_drop"] for v in vals]),
                "mean_total_positive_route_release_after_source_removal": safe_mean([v["total_positive_route_release"] for v in vals]),
                "mean_route_release_coverage": safe_mean([v["route_release_coverage"] for v in vals]),
                "mean_margin_drop_target_vs_routes": safe_mean([v["mean_margin_drop_target_vs_routes"] for v in vals]),
                "top1_loss_rate": sum(1 for v in vals if v["top1_loss"]) / n,
                "support_rate_drop_gt_0_20": sum(1 for v in vals if v["target_logit_drop"] > 0.20) / n,
                "route_guard_rate_release_gt_0_20": sum(1 for v in vals if v["total_positive_route_release"] > 0.20) / n,
                "mean_final_hidden_delta_norm_after_removal": safe_mean([v["final_hidden_delta_norm_after_removal"] for v in vals]),
                "stability_guess": classify_stability(vals),
                "relation_breakdown": rel_break,
                "answer_breakdown_top": dict(list(ans_break.items())[:8]),
                "object_group_breakdown": obj_break,
            }
        )
    summary.sort(
        key=lambda r: (
            r["support_rate_drop_gt_0_20"],
            r["mean_target_logit_drop_after_source_removal"] or 0.0,
            r["mean_total_positive_route_release_after_source_removal"] or 0.0,
            r["top1_loss_rate"] or 0.0,
        ),
        reverse=True,
    )
    return {
        "phase": 752,
        "title": "Natural Writer Stability and Path Chain Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "candidates": candidates,
        "source_groups": source_groups,
        "summary": summary,
        "top_stability_candidates": summary[:32],
        "strict_interpretation": "This tests fixed head/source stability across cases. Downstream hidden delta is path-propagation evidence, not proof of a closed neuron-level chain.",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args)
    source_groups = source_groups_for(args)
    audits = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} candidates={len(candidates)} sources={source_groups} audits={len(audits['audits'])}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    unembed = get_unembed(model)
    try:
        rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audits["audits"]:
                rows.extend(audit_pair(model, tokenizer, device, args, pair, audit, candidates, source_groups, unembed))
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: stability path chain {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, candidates, source_groups)
    summary["attn_implementation"] = attn_impl
    summary["dtype"] = "bfloat16"
    summary["quantization"] = "off"
    write_jsonl(out_dir / f"phase752_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase752_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "top": summary["top_stability_candidates"][:10]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase752_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 752,
        "title": "Natural Writer Stability and Path Chain Validation",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": "Fixed-head stability test. It is still a head/source-path graph, not a neuron-level global atlas.",
    }
    write_json(out_dir / "phase752_cross_model_summary.json", payload)
    lines = [
        f"# Phase 752 Natural Writer Stability and Path Chain Validation ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence: fixed head/source contribution removal across expanded object-relation-answer cases.",
        "",
        "| model | context | site | head | source | n | relations | support rate | mean drop | route guard rate | mean release | top1 loss | final delta | stability |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, summary in payload["by_model"].items():
        for row in summary.get("top_stability_candidates", [])[:18]:
            lines.append(
                f"| {model_name} | {row['context_name']} | {row['site']} | {row['head']} | {row['source_group']} | {row['n']} | "
                f"{len(row.get('relations') or [])} | "
                f"{(row.get('support_rate_drop_gt_0_20') or 0):.3f} | "
                f"{(row.get('mean_target_logit_drop_after_source_removal') or 0):.3f} | "
                f"{(row.get('route_guard_rate_release_gt_0_20') or 0):.3f} | "
                f"{(row.get('mean_total_positive_route_release_after_source_removal') or 0):.3f} | "
                f"{(row.get('top1_loss_rate') or 0):.3f} | "
                f"{(row.get('mean_final_hidden_delta_norm_after_removal') or 0):.3f} | "
                f"`{row.get('stability_guess')}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Stable target drop supports fixed writer-path necessity.",
            "- Route release supports guard/suppressor participation.",
            "- Downstream hidden delta only says the perturbation propagates; it does not prove a complete chain closure.",
            "- Source groups are still external token-span labels.",
            "",
        ]
    )
    (out_dir / "phase752_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model_name in MODELS:
        args.model = model_name
        audits = load_phase739_audits(model_name, args.phase739_round, args.top_audits)
        payload["models"][model_name] = {
            "candidates": load_candidates(args),
            "source_groups": source_groups_for(args),
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase739-round", default="confirm")
    parser.add_argument("--phase751-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--top-audits", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--min-phase751-drop", type=float, default=0.40)
    parser.add_argument("--top-k-vocab", type=int, default=16)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-route-classes", type=int, default=6)
    parser.add_argument("--max-source-groups", type=int, default=5)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--donor-context-only", action="store_true")
    parser.add_argument("--include-extended-relations", action="store_true")
    parser.add_argument("--log-every", type=int, default=2)
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
