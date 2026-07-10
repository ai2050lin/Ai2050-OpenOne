#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase311_core_language_physical_atlas as p311  # noqa: E402
import phase312_matched_path_feature_analysis as p312  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = "Phase315"
SCHEMA_VERSION = "3.4.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "template_heldout_path_validation"
OUT = ROOT / "tests/gpt5/result/phase315_template_heldout_path_validation"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def paraphrase(case: dict[str, Any]) -> tuple[str, str]:
    prompt = str(case["prompt"])
    mechanism = str(case["mechanism_id"])
    query_surface = str(case["query_surface"])
    prompt = prompt.replace("Fact:", "Statement:").replace("Complete:", "Fill the blank:")
    prompt = prompt.replace("Answer yes or no. Answer:", "Reply only yes or no. Response:")
    prompt = prompt.replace("Facts and rule:", "Given these conditions:").replace("Question:", "Decide whether")
    prompt = prompt.replace("Answer:", "Response:")
    if mechanism == "subject_role":
        prompt = prompt.replace("In the sentence", "Read the sentence").replace(
            "identify the grammatical subject", "which expression is the grammatical subject"
        )
    elif mechanism == "object_role":
        prompt = prompt.replace("In the sentence", "Read the sentence").replace(
            "identify the direct object", "which expression functions as the direct object"
        )
    elif mechanism in {"singular_agreement", "plural_agreement"}:
        prompt = prompt.replace(
            "Complete with the grammatically agreeing verb:", "Select the verb that agrees with the subject:"
        )
        query_surface = "agrees with"
    elif mechanism == "past_tense":
        prompt = prompt.replace("Complete with the correct past-tense verb:", "Select a past-tense completion for:")
    elif mechanism == "pronoun_number":
        prompt = prompt.replace(
            "Complete with the pronoun that agrees in number:", "Choose a number-matching pronoun for:"
        )
        query_surface = "number-matching"
    elif mechanism == "adjective_attachment":
        prompt = prompt.replace("In the phrase", "Read the phrase").replace("describes the ___", "modifies which noun")
        query_surface = "modifies"
    elif mechanism == "relative_clause_role":
        prompt = prompt.replace("In the sentence", "Read the sentence").replace(
            "what performed the action", "which noun carried out the action"
        )
        query_surface = "carried out"
    elif case["family_id"] == "reasoning_constraint":
        query_surface = "Decide whether"
    return "Give only the requested answer. " + prompt, query_surface


def template_cases(model: str) -> list[dict[str, Any]]:
    rows = []
    for base in p311.build_case_bank():
        if base["split"] != "heldout":
            continue
        prompt, query_surface = paraphrase(base)
        rows.append(
            {
                **base,
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "model": model,
                "case_id": f"phase315:{base['family_id']}:{base['mechanism_id']}:{base['item_index']:02d}:{model}",
                "prompt": prompt,
                "query_surface": query_surface,
                "split": "template_heldout",
                "template_source": "paraphrased_after_phase311_prototype_freeze",
                "measurement_status": "planned",
            }
        )
    return rows


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT / args.round_name
    planned = template_cases(args.model)
    components: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    model_obj = tokenizer = None
    prov: dict[str, Any] = {}
    try:
        model_obj, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prov = p311.provenance(model_obj, tokenizer, args.model, attn_impl)
        prov["run_id"] = prov["run_id"].replace("phase311", "phase315")
        for index, case in enumerate(planned, 1):
            try:
                rows, position_rows, result = p311.trace_case(model_obj, tokenizer, device, case, prov)
                for row in rows:
                    row.update({"schema_version": SCHEMA_VERSION, "phase_id": PHASE, "template_source": case["template_source"]})
                for row in position_rows:
                    row.update({"schema_version": SCHEMA_VERSION, "phase_id": PHASE, "template_source": case["template_source"]})
                result.update({"schema_version": SCHEMA_VERSION, "phase_id": PHASE, "template_source": case["template_source"]})
                components.extend(rows)
                summaries.extend(position_rows)
                measured.append(result)
            except Exception as exc:  # noqa: BLE001
                missing.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": args.model,
                        "case_id": case["case_id"],
                        "reason": repr(exc),
                    }
                )
            print(f"{args.model}: template heldout path {index}/{len(planned)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(measured) + len(missing) == len(planned) else "partial",
        "model": args.model,
        "planned_template_heldout_cases": len(planned),
        "valid_template_heldout_cases": len(measured),
        "component_rows": len(components),
        "position_summary_rows": len(summaries),
        "missing_rows": len(missing),
        "target_winner_rate": mean_safe([1.0 if r["actual_final_semantic_winner"] == "target" else 0.0 for r in measured]),
        "token_match_confidence_mean": mean_safe([safe_float(r["token_match_confidence"]) for r in summaries]),
        "provenance": prov,
    }
    write_json(out_dir / f"phase315_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase315_{args.model}_case_rows.jsonl", measured)
    write_jsonl(out_dir / f"phase315_{args.model}_component_rows.jsonl", components)
    write_jsonl(out_dir / f"phase315_{args.model}_position_rows.jsonl", summaries)
    write_jsonl(out_dir / f"phase315_{args.model}_missing_rows.jsonl", missing)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def validation_predictions(cases: list[dict[str, Any]], components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original_cases = read_jsonl(V2 / "phase311_core_language_case_result_rows.jsonl")
    original_components = read_jsonl(V2 / "phase311_core_language_component_rows.jsonl")
    original_profiles = p312.profile_index(original_components)
    heldout_profiles = p312.profile_index(components)
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        train = [r for r in original_cases if r["model"] == model and r["split"] in {"discovery", "calibration"}]
        family_prototypes: dict[str, list[float]] = {}
        mechanism_prototypes: dict[tuple[str, str], list[float]] = {}
        for family in sorted({str(r["family_id"]) for r in train}):
            family_prototypes[family] = p312.prototype(
                [p312.case_vector(model, str(r["case_id"]), original_profiles) for r in train if r["family_id"] == family]
            )
            for mechanism in sorted({str(r["mechanism_id"]) for r in train if r["family_id"] == family}):
                mechanism_prototypes[(family, mechanism)] = p312.prototype(
                    [
                        p312.case_vector(model, str(r["case_id"]), original_profiles)
                        for r in train
                        if r["family_id"] == family and r["mechanism_id"] == mechanism
                    ]
                )
        for case in [r for r in cases if r["model"] == model]:
            vector = p312.case_vector(model, str(case["case_id"]), heldout_profiles)
            family_scores = {family: p312.cosine(vector, proto) for family, proto in family_prototypes.items()}
            predicted_family = max(family_scores, key=family_scores.get)
            family = str(case["family_id"])
            mechanism_scores = {
                mechanism: p312.cosine(vector, proto)
                for (fam, mechanism), proto in mechanism_prototypes.items()
                if fam == family
            }
            predicted_mechanism = max(mechanism_scores, key=mechanism_scores.get)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "prediction_id": f"phase315:prediction:{model}:{case['case_id']}",
                    "model": model,
                    "case_id": case["case_id"],
                    "true_family_id": family,
                    "predicted_family_id": predicted_family,
                    "family_correct": predicted_family == family,
                    "true_mechanism_id": case["mechanism_id"],
                    "predicted_mechanism_id": predicted_mechanism,
                    "mechanism_correct": predicted_mechanism == case["mechanism_id"],
                    "family_best_cosine": family_scores[predicted_family],
                    "mechanism_best_cosine": mechanism_scores[predicted_mechanism],
                    "family_random_baseline": 1.0 / len(family_scores),
                    "mechanism_random_baseline": 1.0 / len(mechanism_scores),
                    "prototype_source": "phase311_discovery_and_calibration_frozen",
                    "validation_split": "new_template_and_heldout_item",
                }
            )
    return rows


def collect(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    model_summaries = []
    cases: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        path = out_dir / f"phase315_{model}_summary.json"
        if path.exists():
            model_summaries.append(json.loads(path.read_text(encoding="utf-8")))
        cases.extend(read_jsonl(out_dir / f"phase315_{model}_case_rows.jsonl"))
        components.extend(read_jsonl(out_dir / f"phase315_{model}_component_rows.jsonl"))
        positions.extend(read_jsonl(out_dir / f"phase315_{model}_position_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase315_{model}_missing_rows.jsonl"))
    predictions = validation_predictions(cases, components)
    by_model = {}
    for model in MODELS:
        vals = [r for r in predictions if r["model"] == model]
        by_model[model] = {
            "rows": len(vals),
            "family_accuracy": mean_safe([1.0 if r["family_correct"] else 0.0 for r in vals]),
            "mechanism_accuracy": mean_safe([1.0 if r["mechanism_correct"] else 0.0 for r in vals]),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(cases) == 72 and not missing else "partial",
        "planned_template_heldout_model_cases": 72,
        "valid_template_heldout_model_cases": len(cases),
        "component_rows": len(components),
        "position_summary_rows": len(positions),
        "missing_rows": len(missing),
        "prediction_rows": len(predictions),
        "template_heldout_family_accuracy": mean_safe([1.0 if r["family_correct"] else 0.0 for r in predictions]),
        "template_heldout_mechanism_accuracy": mean_safe([1.0 if r["mechanism_correct"] else 0.0 for r in predictions]),
        "family_random_baseline": 1.0 / 3.0,
        "mechanism_random_baseline": 1.0 / 8.0,
        "target_winner_rate": mean_safe([1.0 if r["actual_final_semantic_winner"] == "target" else 0.0 for r in cases]),
        "token_match_confidence_mean": mean_safe([safe_float(r["token_match_confidence"]) for r in positions]),
        "model_prediction_summary": by_model,
        "model_summaries": model_summaries,
        "caution": [
            "Templates are rule-based paraphrases, not independently authored open-set prompts.",
            "Only one lexical/rule item per mechanism is tested.",
            "Prediction is observational and does not establish a causal mechanism.",
        ],
    }
    write_json(out_dir / "phase315_template_heldout_summary.json", payload)
    write_jsonl(out_dir / "phase315_template_heldout_case_rows.jsonl", cases)
    write_jsonl(out_dir / "phase315_template_heldout_component_rows.jsonl", components)
    write_jsonl(out_dir / "phase315_template_heldout_position_rows.jsonl", positions)
    write_jsonl(out_dir / "phase315_template_heldout_prediction_rows.jsonl", predictions)
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase315_template_heldout_summary.json", payload)
        write_jsonl(base / "phase315_template_heldout_case_rows.jsonl", cases)
        write_jsonl(base / "phase315_template_heldout_component_rows.jsonl", components)
        write_jsonl(base / "phase315_template_heldout_position_rows.jsonl", positions)
        write_jsonl(base / "phase315_template_heldout_prediction_rows.jsonl", predictions)
        write_jsonl(base / "phase315_missing_rows.jsonl", missing)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        collect(args.round_name)
    elif args.model:
        run_model(args)
    else:
        for model in MODELS:
            args.model = model
            run_model(args)
        collect(args.round_name)


if __name__ == "__main__":
    main()
