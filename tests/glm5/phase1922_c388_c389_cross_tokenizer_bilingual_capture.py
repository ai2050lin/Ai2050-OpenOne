#!/usr/bin/env python3
"""Supplemental C388-C389 producer for cross-tokenizer bilingual capture.

The frozen C369-C390 campaign assumed that a semantic role string always
appeared as an exact token subsequence. GLM's tokenizer can merge a registered
Chinese relation with an adjacent particle, so this producer keeps the frozen
material and gates but uses an audited local decode-window fallback.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1903_c369_c390_language_operation_graph_campaign as campaign
from model_utils import MODEL_CONFIGS, get_model_info


MODEL_BY_CAMPAIGN = {"C388": "glm4", "C389": "deepseek7b"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_piece(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return "".join(ch for ch in value if not ch.isspace() and ch not in "▁Ġ")


def token_subsequence_spans(ids: list[int], needle: list[int]) -> list[list[int]]:
    if not needle:
        return []
    width = len(needle)
    return [list(range(start, start + width)) for start in range(len(ids) - width + 1) if ids[start:start + width] == needle]


def robust_spans(tokenizer, ids: list[int], value: str) -> tuple[list[list[int]], str]:
    exact = campaign.common.graph_base.name_spans(tokenizer, ids, value)
    if exact:
        return exact, "registered_exact"

    for variant in (value, " " + value):
        needle = tokenizer.encode(variant, add_special_tokens=False)
        spans = token_subsequence_spans(ids, needle)
        if spans:
            return spans, "token_subsequence"

    target = normalize_piece(value)
    candidates: list[tuple[int, int, int, list[int]]] = []
    for width in range(1, min(12, len(ids)) + 1):
        for start in range(0, len(ids) - width + 1):
            decoded = normalize_piece(tokenizer.decode(ids[start:start + width], skip_special_tokens=True))
            if target and target in decoded:
                candidates.append((len(decoded) - len(target), width, start, list(range(start, start + width))))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        best_excess, best_width = candidates[0][0], candidates[0][1]
        spans = [item[3] for item in candidates if item[0] == best_excess and item[1] == best_width]
        return spans, "decoded_local_window"
    return [], "missing"


def compile_rows(tokenizer, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    compiled: list[dict] = []
    span_audit: list[dict] = []
    for row in rows:
        ids, candidates = campaign.common.render_interface(tokenizer, row, "strict_chat")
        positions: dict[str, list[int]] = {}
        methods: dict[str, str] = {}
        for role, value in row["role_values"].items():
            spans, method = robust_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value, method))
            positions[role] = spans[-1] if role == "query" else spans[0]
            methods[role] = method
        positions["boundary"] = [len(ids) - 1]
        methods["boundary"] = "compiled_generation_boundary"
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
        span_audit.append({"case_id": row["case_id"], "methods": methods, "role_positions": positions})
    return compiled, span_audit


def tokenizer_only(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[model_name]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def validate_tokenizer(model_name: str) -> dict:
    rows = campaign.read_rows(campaign.OUTS["C387"] / "material/cases.jsonl")
    tokenizer = tokenizer_only(model_name)
    compiled, audit = compile_rows(tokenizer, rows)
    methods = Counter(method for row in audit for method in row["methods"].values())
    result = {"model": model_name, "rows": len(compiled), "span_method_counts": dict(methods), "all_roles_compiled": len(compiled) == 288}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def begin(campaign_name: str, model_name: str) -> Path:
    out = campaign.OUTS[campaign_name]
    if (out / "analysis/final.json").exists():
        return out
    if out.exists():
        raise RuntimeError(f"partial output exists: {out}")
    parent = "C387" if campaign_name == "C388" else "C388"
    checks = {
        "parent": campaign.final(parent)["all_checks_passed"],
        "registered_model": model_name in campaign.common.MODELS,
        "cuda": torch.cuda.is_available(),
        "frozen_material": (campaign.OUTS["C387"] / "material/cases.jsonl").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": campaign.PHASES[campaign_name][0],
        "campaign": campaign_name,
        "created_at_utc": utc_now(),
        "producer_sha256": producer_hash(),
        "status": "supplemental_cross_tokenizer_bilingual_panel_frozen",
        "model": model_name,
        "rows": "same frozen C387 6-family English/Chinese panel, 288 rows",
        "archive": "all native checkpoints x six roles x all model-native coordinates",
        "interface": "strict_chat frozen before behavior",
        "span_policy": "exact registered span, exact token subsequence, then minimal decoded local window",
        "claim_boundary": "fallback span compilation is an interface repair, not evidence of semantic alignment",
    })
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def close(campaign_name: str, headline: dict, checks: dict, next_authorization: str) -> dict:
    out = campaign.OUTS[campaign_name]
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    protocol = json.loads((out / "protocol/preregistration.json").read_text(encoding="utf-8"))
    final_checks = {
        "contract": json.loads((out / "audit/internal_contract_audit.json").read_text(encoding="utf-8"))["all_checks_passed"],
        "analysis": all(checks.values()),
        "producer_hash": protocol["producer_sha256"] == producer_hash(),
    }
    value = {
        "phase": campaign.PHASES[campaign_name][0],
        "campaign": campaign_name,
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": headline,
        "next_authorization": next_authorization,
    }
    save(out / "analysis/final.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


@torch.inference_mode()
def run(campaign_name: str) -> None:
    model_name = MODEL_BY_CAMPAIGN[campaign_name]
    out = begin(campaign_name, model_name)
    if (out / "analysis/final.json").exists():
        print((out / "analysis/final.json").read_text(encoding="utf-8"), flush=True)
        return
    rows = campaign.read_rows(campaign.OUTS["C387"] / "material/cases.jsonl")
    write_rows(out / "material/cases.jsonl", rows)
    model = None
    try:
        model, tokenizer, device, placement = campaign.common.model_base.load_bf16(model_name)
        compiled, span_audit = compile_rows(tokenizer, rows)
        write_rows(out / "compiled/model_rows.jsonl", compiled)
        write_rows(out / "audit/span_compilation.jsonl", span_audit)
        span_methods = Counter(method for row in span_audit for method in row["methods"].values())
        save(out / "audit/span_compilation_summary.json", {"counts": dict(span_methods), "rows": len(span_audit)})

        info = get_model_info(model, model_name)
        checkpoints = info.n_layers + 1
        states = np.lib.format.open_memmap(
            out / "raw/role_states.float16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(rows), checkpoints, len(campaign.ROLES), info.d_model),
        )
        behavior: list[dict] = []
        index: list[dict] = []
        for row_i, row in enumerate(compiled):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )
            if len(output.hidden_states) != checkpoints:
                raise RuntimeError((model_name, len(output.hidden_states), checkpoints))
            for checkpoint_i, state in enumerate(output.hidden_states):
                for role_i, role in enumerate(campaign.ROLES):
                    states[row_i, checkpoint_i, role_i] = (
                        state[0, row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
                    )
            if all(len(candidate) == 1 for candidate in row["candidate_ids"]):
                scores = [float(output.logits[0, ids.shape[1] - 1, candidate[0]]) for candidate in row["candidate_ids"]]
            else:
                pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                scores = campaign.common.score_prompt_candidates(model, row["prompt_ids"], row["candidate_ids"], device, pad).tolist()
            prediction = int(np.argmax(scores))
            correct = prediction == row["gold_position"]
            behavior.append({
                "case_id": row["case_id"], "family": row["family"], "language": row["language"],
                "partition": row["partition"], "prediction": prediction, "correct": correct, "scores": scores,
            })
            index.append({
                "hidden_index": row_i, "case_id": row["case_id"], "family": row["family"],
                "language": row["language"], "unit": row["unit"], "factor_a": row["factor_a"],
                "factor_b": row["factor_b"], "order": row["order"], "partition": row["partition"],
                "correct": correct,
            })
            states.flush()
            if row_i % 48 == 0 or row_i + 1 == len(rows):
                print(f"[{campaign_name}] {model_name} {row_i + 1}/{len(rows)}", flush=True)

        write_rows(out / "raw/behavior.jsonl", behavior)
        write_rows(out / "raw/hidden_index.jsonl", index)
        lockbox = [row for row in behavior if row["partition"] == "lockbox"]
        by_language = {language: float(np.mean([row["correct"] for row in lockbox if row["language"] == language])) for language in ("en", "zh")}
        by_family = {family: float(np.mean([row["correct"] for row in lockbox if row["family"] == family])) for family in campaign.BILINGUAL_FAMILIES}
        eligible = min(by_language.values()) >= 0.60 and min(by_family.values()) >= 0.50
        headline = {
            "status": "single_model_bilingual_panel_closed",
            "model": model_name,
            "placement": placement,
            "rows": len(rows),
            "role_shape": list(states.shape),
            "span_method_counts": dict(span_methods),
            "lockbox_accuracy": float(np.mean([row["correct"] for row in lockbox])),
            "language_accuracy": by_language,
            "family_accuracy": by_family,
            "abstract_response_eligible": eligible,
            "strict_interpretation": "Eligibility permits role/checkpoint response abstraction only; native coordinate identity and semantic equivalence remain unclaimed.",
        }
        checks = {
            "rows": len(rows) == 288,
            "all_roles_compiled": len(span_audit) == 288,
            "shape": states.shape[0] == 288 and states.shape[2] == len(campaign.ROLES),
            "finite": campaign.finite(headline),
        }
        close(campaign_name, headline, checks, "C389" if campaign_name == "C388" else "C390")
    finally:
        campaign.common.model_base.release(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=tuple(MODEL_BY_CAMPAIGN))
    parser.add_argument("--validate-tokenizer", choices=tuple(MODEL_BY_CAMPAIGN.values()))
    args = parser.parse_args()
    if args.validate_tokenizer:
        validate_tokenizer(args.validate_tokenizer)
    elif args.run:
        run(args.run)
    else:
        parser.error("provide --run or --validate-tokenizer")


if __name__ == "__main__":
    main()
