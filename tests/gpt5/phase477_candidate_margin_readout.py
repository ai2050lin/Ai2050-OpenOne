#!/usr/bin/env python3
"""Phase477 A/B candidate-margin readout.

Reads identical evidence-first label-mapping pairs and projects role mean hidden
states through the final norm plus lm_head as an external logit-lens observer.
This is a readout diagnostic only, not a causal or internal-mechanism claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase451_glm4_v2_pilot_behavior import load_jsonl, prompt_for, write_jsonl  # noqa: E402
from phase475_mapping_position_scalar_precheck import (  # noqa: E402
    GEN_PATH,
    MAX_PAIR_INDEX,
    ROLE_NAMES,
    SAMPLES_PATH,
    SELECTED_TRANSFORM,
    behavior_truth,
    build_eval_rows,
    locate_role_positions,
)


OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase477_candidate_margin_readout"
PROTOCOL_PATH = OUT_DIR / "phase477_candidate_margin_protocol.json"
ROWS_PATH = OUT_DIR / "phase477_candidate_margin_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase477_candidate_margin_summary.json"


def mapping_sign(label_mapping: str) -> int:
    if label_mapping == "mu_ab":
        return 1
    if label_mapping == "mu_ba":
        return -1
    raise ValueError(label_mapping)


def truth_sign(truth_value: bool) -> int:
    return 1 if truth_value else -1


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"Expected one token for {text!r}, got {ids}")
    return int(ids[0])


def final_norm(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise RuntimeError("Could not locate final normalization module for logit lens")


def logit_lens_scores(model: Any, norm: Any, mean_vec: torch.Tensor, a_id: int, b_id: int) -> dict[str, float]:
    with torch.inference_mode():
        normed = norm(mean_vec)
        logits = model.lm_head(normed)
    score_a = float(logits[a_id].detach().float().item())
    score_b = float(logits[b_id].detach().float().item())
    margin_ab = score_a - score_b
    return {
        "score_A": score_a,
        "score_B": score_b,
        "margin_ab": margin_ab,
    }


def trace_rows(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    a_id = single_token_id(tokenizer, "A")
    b_id = single_token_id(tokenizer, "B")
    norm = final_norm(model)
    out = []
    for idx, row in enumerate(rows, start=1):
        prompt = prompt_for(row["eval_text"])
        role_positions = locate_role_positions(tokenizer, prompt, row["eval_text"])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        seq_len = int(encoded["attention_mask"][0].sum().item())
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        s_mu = mapping_sign(row["label_mapping"])
        s_truth = truth_sign(bool(row["truth_value"]))
        for role, positions in role_positions.items():
            if any(pos < 0 or pos >= seq_len for pos in positions):
                raise RuntimeError(f"Phase477 invalid role positions: {row['sample_id']} {role}")
            for layer_index, hidden in enumerate(outputs.hidden_states):
                mean_vec = hidden[0, positions].mean(dim=0)
                scores = logit_lens_scores(model, norm, mean_vec, a_id, b_id)
                margin_true = s_mu * scores["margin_ab"]
                margin_correct = s_truth * s_mu * scores["margin_ab"]
                out.append({
                    "model": "glm4",
                    "phase": "phase477",
                    "readout": "final_norm_plus_lm_head_logit_lens",
                    "readout_scope": "external_observer_not_internal_mechanism",
                    "token_id_A": a_id,
                    "token_id_B": b_id,
                    "sample_id": row["sample_id"],
                    "source_sample_id": row["source_sample_id"],
                    "source_pair_id": row["source_pair_id"],
                    "pair_index": row["pair_index"],
                    "pair_role": row["pair_role"],
                    "transform": row["transform"],
                    "label_mapping": row["label_mapping"],
                    "mapping_sign": s_mu,
                    "role": role,
                    "expected_label": row["expected_label"],
                    "truth_value": row["truth_value"],
                    "truth_sign": s_truth,
                    "classification": row["classification"],
                    "normalized_generated": row["normalized_generated"],
                    "behavior_truth": row["behavior_truth"],
                    "target_position": row["target_position"],
                    "query_position": row["query_position"],
                    "layer_index": layer_index,
                    "role_token_count": len(positions),
                    **scores,
                    "margin_true": margin_true,
                    "margin_correct": margin_correct,
                })
        if idx % 12 == 0:
            print(f"[phase477] traced {idx}/{len(rows)} prompts", flush=True)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = {(row["sample_id"], row["label_mapping"]) for row in rows if row["layer_index"] == 0 and row["role"] == "terminal_token"}
    behavior_counts = Counter(
        (row["label_mapping"], row["truth_value"], row["classification"], row["behavior_truth"])
        for row in rows
        if row["layer_index"] == 0 and row["role"] == "terminal_token"
    )
    role_counts = Counter(row["role"] for row in rows if row["layer_index"] == 0)
    role_token_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        if row["layer_index"] == 0:
            role_token_counts[row["role"]][int(row["role_token_count"])] += 1
    return {
        "schema_version": "phase477_candidate_margin_readout.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_margin_readout_complete",
        "model": "glm4",
        "transform": SELECTED_TRANSFORM,
        "roles": list(ROLE_NAMES),
        "prompt_count": len(prompts),
        "trace_row_count": len(rows),
        "role_counts_at_layer0": dict(role_counts),
        "role_token_count_distribution": {role: dict(counts) for role, counts in role_token_counts.items()},
        "behavior_counts": {str(key): value for key, value in behavior_counts.items()},
        "readout_definition": {
            "score": "S_v^(ell,r) = [W_U N_f(mean(H_r^(ell)))]_v",
            "margin_ab": "M_AB = S_A - S_B",
            "margin_true": "M_true = s_mu * M_AB",
            "margin_correct": "M_correct = s_truth * s_mu * M_AB",
        },
        "authorization": {
            "internal_mechanism_claim_authorized": False,
            "causal_or_neuron_claim_authorized": False,
            "next_step": "phase478_candidate_margin_paired_analysis",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_eval_rows(load_jsonl(SAMPLES_PATH), load_jsonl(GEN_PATH))
    protocol = {
        "schema_version": "phase477_candidate_margin_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_candidate_margin_readout",
        "model": "glm4",
        "transform": SELECTED_TRANSFORM,
        "roles": list(ROLE_NAMES),
        "pair_index_range": [0, MAX_PAIR_INDEX],
        "prompt_count": len(rows),
        "readout_scope": "external logit-lens readout only; not an internal mechanism or causal claim",
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model, tokenizer, device = load_model("glm4", use_8bit=args.use_8bit)
    try:
        traced = trace_rows(model, tokenizer, device, rows)
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_jsonl(ROWS_PATH, traced)
    SUMMARY_PATH.write_text(json.dumps(summarize(traced), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(ROWS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
