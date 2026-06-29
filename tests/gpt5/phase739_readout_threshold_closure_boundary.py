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
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import release_model  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase727_category_fruit_cluster_intervention import hit_answer, norm_text  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, norm, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import (  # noqa: E402
    install_source_contribution_replacement,
    select_conflict_pairs,
    source_contribution_for_case,
)
from phase737_writer_rewriter_joint_replacement import (  # noqa: E402
    build_interventions,
    install_mlp_group_replacements,
    intervention_label,
    load_phase735_mlp_specs,
    load_phase735_source_specs,
    mlp_group_outputs_for_case,
)
from phase738_readout_margin_continuation_audit import (  # noqa: E402
    OUT_ROOT as PHASE738_ROOT,
    candidate_specs_for_case,
    decode_token,
    rank_candidates,
    top_vocab,
)


OUT_ROOT = Path("results/glm5_phase739_readout_threshold_closure_boundary")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_unembed(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach().float().cpu()
    return model.get_output_embeddings().weight.detach().float().cpu()


def first_token_id(tokenizer, text: str) -> int:
    ids = target_token_ids(tokenizer, text)
    return int(ids[0])


def choose_donor_recipient(pair: dict[str, Any], direction: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if direction == "conflict<-explicit":
        return pair["explicit_profile"], pair["conflict_profile"]
    if direction == "explicit<-conflict":
        return pair["conflict_profile"], pair["explicit_profile"]
    raise ValueError(direction)


def load_phase738_audits(model_name: str, phase738_round: str, top_audits: int) -> dict[str, Any]:
    source_payload = load_phase735_source_specs(model_name, "confirm", 3, None)
    mlp_specs = load_phase735_mlp_specs(model_name, "confirm", 2)
    all_interventions = {intervention_label(x): x for x in build_interventions(source_payload["paths"], mlp_specs, "compact")}
    summary_path = PHASE738_ROOT / phase738_round / f"phase738_{model_name}_summary.json"
    audits: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_payload["target_site"] = summary.get("target_site") or source_payload["target_site"]
        for row in summary.get("top_readout_audits", []):
            label = row.get("intervention_label")
            direction = row.get("direction")
            if not label or not direction or label not in all_interventions:
                continue
            key = (label, direction)
            if key in seen:
                continue
            audits.append({"intervention": all_interventions[label], "direction": direction, "phase738_row": row})
            seen.add(key)
            if len(audits) >= top_audits:
                break
    if len(audits) < top_audits:
        for intervention in build_interventions(source_payload["paths"], mlp_specs, "compact"):
            for direction in ["conflict<-explicit", "explicit<-conflict"]:
                key = (intervention_label(intervention), direction)
                if key in seen:
                    continue
                audits.append({"intervention": intervention, "direction": direction, "phase738_row": None})
                seen.add(key)
                if len(audits) >= top_audits:
                    break
            if len(audits) >= top_audits:
                break
    return {"target_site": source_payload["target_site"], "audits": audits[:top_audits]}


def prepare_joint_install(
    model,
    tokenizer,
    device,
    target_site: str,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    recipient_ids: list[int],
    donor_ids: list[int],
    intervention: dict[str, Any],
) -> tuple[dict[str, Any], Callable[[], list[Any]]]:
    source_spec = intervention.get("source_spec")
    mlp_specs = intervention.get("mlp_specs") or []
    meta: dict[str, Any] = {
        "donor_attention_mass": None,
        "recipient_attention_mass": None,
        "donor_source_token_count": None,
        "recipient_source_token_count": None,
        "source_contribution_delta_norm": None,
        "mlp_delta_norm_total": 0.0,
        "mlp_components": [s["component_id"] for s in mlp_specs],
    }
    donor_contrib = recipient_contrib = None
    if source_spec is not None:
        layer_idx = int(source_spec["layer"])
        head_idx = int(source_spec["head"])
        source_group = source_spec["source_group"]
        _d_vec, _d_logits, donor_contrib, donor_mass, donor_count = source_contribution_for_case(
            model, tokenizer, device, target_site, donor_case, donor_ids, layer_idx, head_idx, source_group
        )
        _r_vec, _r_logits, recipient_contrib, recipient_mass, recipient_count = source_contribution_for_case(
            model, tokenizer, device, target_site, recipient_case, recipient_ids, layer_idx, head_idx, source_group
        )
        meta.update(
            {
                "donor_attention_mass": donor_mass,
                "recipient_attention_mass": recipient_mass,
                "donor_source_token_count": donor_count,
                "recipient_source_token_count": recipient_count,
                "source_contribution_delta_norm": norm(donor_contrib - recipient_contrib),
            }
        )
    donor_mlp = mlp_group_outputs_for_case(model, device, donor_ids, mlp_specs)
    recipient_mlp = mlp_group_outputs_for_case(model, device, recipient_ids, mlp_specs)
    mlp_replacements = []
    for spec in mlp_specs:
        cid = spec["component_id"]
        delta = donor_mlp[cid] - recipient_mlp[cid]
        meta["mlp_delta_norm_total"] += norm(delta)
        mlp_replacements.append({**spec, "donor": donor_mlp[cid], "recipient": recipient_mlp[cid], "delta_norm": norm(delta)})

    def install() -> list[Any]:
        handles = []
        if source_spec is not None and donor_contrib is not None and recipient_contrib is not None:
            handles.extend(
                install_source_contribution_replacement(
                    model,
                    int(source_spec["layer"]),
                    int(source_spec["head"]),
                    recipient_contrib,
                    donor_contrib,
                )
            )
        handles.extend(install_mlp_group_replacements(model, mlp_replacements))
        return handles

    return meta, install


def forward_logits_with_final_boost(
    model,
    device,
    ids: list[int],
    install_joint: Callable[[], list[Any]] | None,
    final_delta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    handles = install_joint() if install_joint else []
    final_vec: torch.Tensor | None = None
    final_norm = get_final_norm(model)
    if final_norm is None:
        raise RuntimeError("final norm not found")

    def final_hook(_module, _inputs, output):
        nonlocal final_vec
        y = extract_tensor(output)
        final_vec = y[0, -1].detach().float().cpu()
        if final_delta is None:
            return output
        y_new = y.clone()
        y_new[0, -1, :] = y_new[0, -1, :] + final_delta.to(device=y_new.device, dtype=y_new.dtype)
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    handles.append(final_norm.register_forward_hook(final_hook))
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu(), final_vec
    finally:
        for h in handles:
            h.remove()


def normalized_direction(unembed: torch.Tensor, donor_id: int, competitor_id: int) -> torch.Tensor | None:
    diff = unembed[int(donor_id)] - unembed[int(competitor_id)]
    n = float(torch.linalg.vector_norm(diff).item())
    if n <= 1e-8:
        return None
    return diff / n


def alpha_needed(logits: torch.Tensor, unembed: torch.Tensor, donor_id: int, competitor_id: int, direction: torch.Tensor | None) -> float | None:
    if int(donor_id) == int(competitor_id):
        return 0.0
    if direction is None:
        return None
    gap = float((logits[int(competitor_id)] - logits[int(donor_id)]).item())
    if gap <= 0:
        return 0.0
    denom = float(torch.dot(unembed[int(donor_id)] - unembed[int(competitor_id)], direction).item())
    if denom <= 1e-8:
        return None
    return gap / denom


def scan_alphas(alpha_star: float | None, max_alpha: float) -> list[float]:
    if alpha_star is None or not math.isfinite(alpha_star) or alpha_star <= 0:
        base = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    else:
        base = [
            0.0,
            0.25 * alpha_star,
            0.5 * alpha_star,
            0.75 * alpha_star,
            alpha_star,
            1.25 * alpha_star,
            1.5 * alpha_star,
            2.0 * alpha_star,
            3.0 * alpha_star,
            4.0 * alpha_star,
            6.0 * alpha_star,
            8.0 * alpha_star,
        ]
    out: list[float] = []
    for alpha in base:
        alpha = min(float(alpha), max_alpha)
        if alpha not in out:
            out.append(alpha)
    return out


def top_token_info(logits: torch.Tensor, tokenizer) -> dict[str, Any]:
    tid = int(torch.argmax(logits).item())
    return {"token_id": tid, "token_text": decode_token(tokenizer, tid), "logit": float(logits[tid].item())}


def candidate_best_label(logits: torch.Tensor, tokenizer, specs: list[dict[str, Any]]) -> str | None:
    ranked = rank_candidates(logits, tokenizer, specs)
    return ranked[0]["label"] if ranked else None


def classify_generated_text(text: str, donor_answer: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    donor = donor_answer.lower().strip()
    if not stripped:
        return "empty"
    if lowered == donor or lowered.startswith(donor + "\n") or lowered.startswith(donor + "."):
        return "answer_stop"
    if lowered.startswith(donor + " is") or lowered.startswith(donor + " of"):
        return "answer_then_prose"
    if hit_answer(stripped, donor_answer):
        return "answer_mentioned"
    return "other"


def greedy_generate_joint_first_boost(
    model,
    tokenizer,
    device,
    target_site: str,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    intervention: dict[str, Any],
    first_delta: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    recipient_ids = tokenizer.encode(prompt_for(recipient_case), add_special_tokens=False)
    donor_ids = tokenizer.encode(prompt_for(donor_case), add_special_tokens=False)
    new_ids: list[int] = []
    for step in range(max_new_tokens):
        _meta, install = prepare_joint_install(model, tokenizer, device, target_site, recipient_case, donor_case, recipient_ids, donor_ids, intervention)
        delta = first_delta if step == 0 else None
        logits, _final = forward_logits_with_final_boost(model, device, recipient_ids, install, delta)
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        recipient_ids.append(tok)
        donor_ids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "token_ids": new_ids,
        "token_texts": [decode_token(tokenizer, tid) for tid in new_ids],
        "donor_hit": hit_answer(text, donor_case["answer"]),
        "recipient_hit": hit_answer(text, recipient_case["answer"]),
        "class": classify_generated_text(text, donor_case["answer"]),
    }


def audit_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    audit: dict[str, Any],
    alpha_max: float,
    max_new_tokens: int,
    do_generation: bool,
) -> dict[str, Any]:
    intervention = audit["intervention"]
    direction_name = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction_name)
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    meta, install = prepare_joint_install(model, tokenizer, device, target_site, recipient, donor, recipient_ids, donor_ids, intervention)
    patched_logits, final_vec = forward_logits_with_final_boost(model, device, recipient_ids, install, None)

    token0_specs = candidate_specs_for_case(tokenizer, donor, recipient)
    patched_candidates = rank_candidates(patched_logits, tokenizer, token0_specs)
    donor_id = first_token_id(tokenizer, donor["answer"])
    recipient_id = first_token_id(tokenizer, recipient["answer"])
    candidate_best = patched_candidates[0]
    vocab_best = top_token_info(patched_logits, tokenizer)
    unembed = get_unembed(model)
    vocab_dir = normalized_direction(unembed, donor_id, int(vocab_best["token_id"]))
    candidate_dir = normalized_direction(unembed, donor_id, int(candidate_best["token_id"]))
    alpha_star_vocab = alpha_needed(patched_logits, unembed, donor_id, int(vocab_best["token_id"]), vocab_dir)
    alpha_star_candidate = alpha_needed(patched_logits, unembed, donor_id, int(candidate_best["token_id"]), candidate_dir)
    vocab_alphas = scan_alphas(alpha_star_vocab, alpha_max)

    scan_rows: list[dict[str, Any]] = []
    first_vocab_alpha = None
    first_candidate_alpha = None
    for alpha in vocab_alphas:
        delta = None if vocab_dir is None else vocab_dir * float(alpha)
        boosted_logits, _ = forward_logits_with_final_boost(model, device, recipient_ids, install, delta)
        boosted_top = top_token_info(boosted_logits, tokenizer)
        donor_diag = logit_diag(boosted_logits, donor_id)
        recip_diag = logit_diag(boosted_logits, recipient_id)
        cand_label = candidate_best_label(boosted_logits, tokenizer, token0_specs)
        donor_vocab_top = int(boosted_top["token_id"]) == donor_id
        donor_candidate_top = cand_label == "donor_answer"
        if donor_vocab_top and first_vocab_alpha is None:
            first_vocab_alpha = alpha
        if donor_candidate_top and first_candidate_alpha is None:
            first_candidate_alpha = alpha
        scan_rows.append(
            {
                "alpha": alpha,
                "alpha_over_star_vocab": (alpha / alpha_star_vocab) if alpha_star_vocab and alpha_star_vocab > 0 else None,
                "top_token_id": boosted_top["token_id"],
                "top_token_text": boosted_top["token_text"],
                "donor_vocab_top": donor_vocab_top,
                "candidate_best_label": cand_label,
                "donor_candidate_top": donor_candidate_top,
                "donor_rank": donor_diag["target_rank"],
                "donor_logprob": donor_diag["target_logprob"],
                "donor_vs_recipient_margin": donor_diag["target_logit"] - recip_diag["target_logit"],
                "vocab_top": top_vocab(boosted_logits, tokenizer, 5),
            }
        )

    generation = None
    if do_generation and first_vocab_alpha is not None and vocab_dir is not None:
        generation = greedy_generate_joint_first_boost(
            model,
            tokenizer,
            device,
            target_site,
            recipient,
            donor,
            intervention,
            vocab_dir * float(first_vocab_alpha),
            max_new_tokens,
        )

    donor_diag0 = logit_diag(patched_logits, donor_id)
    recipient_diag0 = logit_diag(patched_logits, recipient_id)
    return {
        "model": model_name,
        "target_site": target_site,
        "pair_id": pair["pair_id"],
        "direction": direction_name,
        "intervention_mode": intervention["mode"],
        "intervention_label": intervention_label(intervention),
        "source_component_id": (intervention.get("source_spec") or {}).get("component_id"),
        "source_group": (intervention.get("source_spec") or {}).get("source_group"),
        "mlp_components": [m["component_id"] for m in intervention.get("mlp_specs") or []],
        "object": donor["object"],
        "relation": donor["relation"],
        "donor_answer": donor["answer"],
        "recipient_answer": recipient["answer"],
        "donor_token_id": donor_id,
        "donor_token_text": decode_token(tokenizer, donor_id),
        "recipient_token_id": recipient_id,
        "recipient_token_text": decode_token(tokenizer, recipient_id),
        "patched_donor_rank": donor_diag0["target_rank"],
        "patched_donor_logprob": donor_diag0["target_logprob"],
        "patched_margin_donor_vs_recipient": donor_diag0["target_logit"] - recipient_diag0["target_logit"],
        "patched_candidate_best_label": candidate_best["label"],
        "patched_candidate_best_token_id": candidate_best["token_id"],
        "patched_candidate_best_token_text": candidate_best["token_text"],
        "patched_margin_donor_vs_candidate_best": donor_diag0["target_logit"] - candidate_best["target_logit"],
        "patched_vocab_top_token_id": vocab_best["token_id"],
        "patched_vocab_top_token_text": vocab_best["token_text"],
        "patched_margin_donor_vs_vocab_top": donor_diag0["target_logit"] - vocab_best["logit"],
        "alpha_star_vocab_top": alpha_star_vocab,
        "alpha_star_candidate_best": alpha_star_candidate,
        "first_alpha_donor_vocab_top": first_vocab_alpha,
        "first_alpha_donor_candidate_top": first_candidate_alpha,
        "alpha_scan": scan_rows,
        "boosted_generation": generation,
        "final_norm_output_norm": norm(final_vec) if final_vec is not None else None,
        **meta,
    }


def run_audits(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pairs: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    alpha_max: float,
    max_new_tokens: int,
    do_generation: bool,
    log_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_idx, pair in enumerate(pairs, 1):
        for audit in audits:
            rows.append(audit_pair(model, tokenizer, device, model_name, target_site, pair, audit, alpha_max, max_new_tokens, do_generation))
        if pair_idx % log_every == 0 or pair_idx == len(pairs):
            log(f"{model_name}: threshold audit {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["intervention_label"] + " " + row["direction"]].append(row)
    out = []
    for key, vals in groups.items():
        n = len(vals)
        gen_rows = [v for v in vals if v.get("boosted_generation") is not None]
        gen_classes = Counter((v.get("boosted_generation") or {}).get("class") for v in vals if v.get("boosted_generation") is not None)
        out.append(
            {
                "intervention_key": key,
                "intervention_label": vals[0]["intervention_label"],
                "direction": vals[0]["direction"],
                "intervention_mode": vals[0]["intervention_mode"],
                "source_component_id": vals[0]["source_component_id"],
                "source_group": vals[0]["source_group"],
                "mlp_components": vals[0]["mlp_components"],
                "n": n,
                "patched_candidate_best_counts": dict(Counter(v["patched_candidate_best_label"] for v in vals)),
                "patched_vocab_top_counts": dict(Counter(v["patched_vocab_top_token_text"] for v in vals)),
                "mean_patched_margin_donor_vs_recipient": safe_mean([v["patched_margin_donor_vs_recipient"] for v in vals]),
                "mean_patched_margin_donor_vs_candidate_best": safe_mean([v["patched_margin_donor_vs_candidate_best"] for v in vals]),
                "mean_patched_margin_donor_vs_vocab_top": safe_mean([v["patched_margin_donor_vs_vocab_top"] for v in vals]),
                "mean_alpha_star_vocab_top": safe_mean([v["alpha_star_vocab_top"] for v in vals]),
                "mean_alpha_star_candidate_best": safe_mean([v["alpha_star_candidate_best"] for v in vals]),
                "candidate_flip_found_rate": sum(1 for v in vals if v["first_alpha_donor_candidate_top"] is not None) / n,
                "vocab_flip_found_rate": sum(1 for v in vals if v["first_alpha_donor_vocab_top"] is not None) / n,
                "mean_first_alpha_donor_candidate_top": safe_mean([v["first_alpha_donor_candidate_top"] for v in vals]),
                "mean_first_alpha_donor_vocab_top": safe_mean([v["first_alpha_donor_vocab_top"] for v in vals]),
                "boosted_generation_tested": len(gen_rows),
                "boosted_generation_donor_hit_rate": (sum(1 for v in gen_rows if v["boosted_generation"]["donor_hit"]) / len(gen_rows)) if gen_rows else None,
                "boosted_generation_class_counts": dict(gen_classes),
                "mean_final_norm_output_norm": safe_mean([v["final_norm_output_norm"] for v in vals]),
                "mean_source_contribution_delta_norm": safe_mean([v["source_contribution_delta_norm"] for v in vals]),
                "mean_mlp_delta_norm_total": safe_mean([v["mlp_delta_norm_total"] for v in vals]),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["vocab_flip_found_rate"],
            -(r["mean_first_alpha_donor_vocab_top"] or 999999),
            r["candidate_flip_found_rate"],
        ),
        reverse=True,
    )


def build_summary(model_name: str, round_name: str, target_site: str, audits: list[dict[str, Any]], rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": 739,
        "title": "Readout Threshold and Closure Boundary Test",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": "eager",
        "attention_note": "eager attention is required because Phase738-selected joint states can include source contribution replacement",
        "quantization": "off",
        "dtype": "bfloat16",
        "phase738_round": args.phase738_round,
        "target_site": target_site,
        "top_audits": args.top_audits,
        "max_pairs": args.max_pairs,
        "alpha_max": args.alpha_max,
        "max_new_tokens": args.max_new_tokens,
        "n_rows": len(rows),
        "audited_interventions": [
            {"label": intervention_label(a["intervention"]), "direction": a["direction"], "phase738_margin": (a.get("phase738_row") or {}).get("mean_token0_patched_margin_donor_vs_recipient")}
            for a in audits
        ],
        "top_threshold_audits": summarize_rows(rows)[:32],
        "strict_interpretation": "This phase measures artificial final-readout boost needed to flip donor token0; success under this boost is a boundary measurement, not proof of a natural internal path.",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = load_phase738_audits(args.model, args.phase738_round, args.top_audits)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} target={audit_payload['target_site']} audits={len(audit_payload['audits'])}")
    model, tokenizer, device, _attn_impl = load_model_bf16_eager(args.model)
    try:
        rows = run_audits(
            model,
            tokenizer,
            device,
            args.model,
            audit_payload["target_site"],
            pairs,
            audit_payload["audits"],
            args.alpha_max,
            args.max_new_tokens,
            not args.no_generation,
            args.log_every,
        )
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args.model, args.round_name, audit_payload["target_site"], audit_payload["audits"], rows, args)
    write_jsonl(out_dir / f"phase739_{args.model}_threshold_rows.jsonl", rows)
    write_json(out_dir / f"phase739_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "target_site": audit_payload["target_site"], "top_threshold_audits": summary["top_threshold_audits"][:5]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_atlas_graph(payload: dict[str, Any], round_name: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_node(node: dict[str, Any]) -> None:
        if node["id"] in seen:
            return
        seen.add(node["id"])
        nodes.append(node)

    for model_index, model in enumerate(payload.get("models", [])):
        lane_z = (model_index - (len(payload.get("models", [])) - 1) / 2) * 10
        summary = payload["by_model"][model]
        model_node = f"{model}:model"
        threshold_node = f"{model}:readout_threshold:{summary['target_site']}"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-30, 0, lane_z], "role": "tested_model"})
        add_node({"id": threshold_node, "type": "readout_threshold", "label": summary["target_site"], "model": model, "position": [4, 0, lane_z], "role": "readout_threshold_boundary"})
        edges.append({"source": model_node, "target": threshold_node, "relation": "audits", "phase": 739})
        for rec in summary.get("top_threshold_audits", [])[:8]:
            audit_node = f"{model}:threshold_audit:{round_name}:{rec['intervention_key']}"
            add_node(
                {
                    "id": audit_node,
                    "type": "threshold_audit",
                    "label": rec["intervention_mode"],
                    "model": model,
                    "role": "final_readout_boost",
                    "mean_alpha_star_vocab_top": rec["mean_alpha_star_vocab_top"],
                    "vocab_flip_found_rate": rec["vocab_flip_found_rate"],
                }
            )
            edges.append({"source": audit_node, "target": threshold_node, "relation": "measures_minimum_boost", "weight": rec["mean_alpha_star_vocab_top"], "phase": 739})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 739 Readout Threshold and Closure Boundary ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 739, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "joint path state -> final readout threshold -> boosted generation", "y": "threshold strength", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 739},
        "source_files": [str(OUT_ROOT / round_name / "phase739_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase739_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 739,
        "title": "Readout Threshold and Closure Boundary Test",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "minimum final-readout boost needed to flip donor token0 plus boosted generation closure",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase739_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase739_atlas_graph.json", graph)
    lines = [
        f"# Phase 739 Readout Threshold and Closure Boundary Test ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: minimum final-readout boost and boosted generation closure.",
        "",
        "| model | target site | top audit | mean alpha* vocab | vocab flip found | boosted donor hit | generation classes |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for model, summary in payload["by_model"].items():
        rec = (summary.get("top_threshold_audits") or [{}])[0]
        hit = rec.get("boosted_generation_donor_hit_rate")
        lines.append(
            f"| {model} | {summary.get('target_site')} | {rec.get('intervention_key')} | "
            f"{(rec.get('mean_alpha_star_vocab_top') or 0):.3f} | "
            f"{(rec.get('vocab_flip_found_rate') or 0):.3f} | "
            f"{(hit if hit is not None else 0):.3f} | {rec.get('boosted_generation_class_counts')} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- This phase applies an artificial final readout boost along donor-vs-current-top direction.",
            "- If donor token0 flips only after a large alpha, the natural writer/rewriter path is far from closure.",
            "- If boosted generation still leaves the answer route, continuation closure remains a separate bottleneck.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase739_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        audits = load_phase738_audits(model, args.phase738_round, args.top_audits)
        payload["models"][model] = {
            "target_site": audits["target_site"],
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase738-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--top-audits", type=int, default=2)
    parser.add_argument("--include-extended-relations", action="store_true")
    parser.add_argument("--alpha-max", type=float, default=80.0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--no-generation", action="store_true")
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
