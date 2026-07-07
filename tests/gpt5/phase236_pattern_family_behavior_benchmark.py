#!/usr/bin/env python3
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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = 236
SOURCE_PHASE = 235
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase236_pattern_family_behavior_benchmark")
SCHEMA_VERSION = "1.0.0"


REGIME_TEXTS = {
    "because_reason": [" Because", "Because", " because", "because"],
    "period_stop": [".", ".\n"],
    "comma_repeat": [",", ", "],
    "for_continuation": [" For", "For", " for", "for"],
    "the_continuation": [" The", "The", " the", "the"],
    "be_continuation": [" be", "be", " is", "is", " are", "are"],
    "newline_boundary": ["\n", "\n\n"],
    "answer_boundary": [" Answer", "Answer", " answer", "answer"],
}


EXTRA_CASES = [
    {
        "case_id": "phase236_reason_negation_0001",
        "family_id": "reasoning_constraint",
        "mode_id": "negation",
        "prompt": "If a bird is not a mammal, is the bird a mammal?\nAnswer with yes or no.\nAnswer:",
        "target": "no",
        "expected_pattern": "negation",
    },
    {
        "case_id": "phase236_reason_ifthen_0001",
        "family_id": "reasoning_constraint",
        "mode_id": "condition_if_then",
        "prompt": "If all dax are blue and this object is a dax, what color is this object?\nAnswer with one word.\nAnswer:",
        "target": "blue",
        "expected_pattern": "condition_if_then",
    },
    {
        "case_id": "phase236_syntax_clause_0001",
        "family_id": "syntax_structure",
        "mode_id": "clause_embedding",
        "prompt": "The cup that is on the table is red. What color is the cup?\nAnswer with one word.\nAnswer:",
        "target": "red",
        "expected_pattern": "clause_embedding",
    },
    {
        "case_id": "phase236_syntax_coord_0001",
        "family_id": "syntax_structure",
        "mode_id": "coordination",
        "prompt": "Alice has a red ball and Bob has a blue ball. What color is Bob's ball?\nAnswer with one word.\nAnswer:",
        "target": "blue",
        "expected_pattern": "coordination",
    },
    {
        "case_id": "phase236_action_translate_0001",
        "family_id": "language_action",
        "mode_id": "translate",
        "prompt": "Translate to Chinese: red apple\nAnswer:",
        "target": "红",
        "expected_pattern": "translate",
    },
    {
        "case_id": "phase236_cross_en_zh_0001",
        "family_id": "cross_lingual",
        "mode_id": "en_to_zh",
        "prompt": "Translate the word snow to Chinese.\nAnswer:",
        "target": "雪",
        "expected_pattern": "en_to_zh",
    },
    {
        "case_id": "phase236_protocol_short_0001",
        "family_id": "output_protocol",
        "mode_id": "short_answer",
        "prompt": "What is the color of grass?\nAnswer with exactly one word.\nAnswer:",
        "target": "green",
        "expected_pattern": "short_answer",
    },
    {
        "case_id": "phase236_readout_because_0001",
        "family_id": "readout_competition",
        "mode_id": "because_reason",
        "prompt": "What is the color of banana?\nAnswer with the answer first, then one short reason using because.\nAnswer:",
        "target": "yellow",
        "expected_pattern": "because_reason",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def token_ids_for_texts(tokenizer: Any, texts: list[str]) -> list[int]:
    out: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            out.append(int(ids[0]))
    return sorted(set(out))


def first_token_candidates(tokenizer: Any, text: str) -> list[int]:
    candidates = [text, " " + text]
    ids: list[int] = []
    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def max_group_score(logits: torch.Tensor, ids: list[int]) -> tuple[float, int, int]:
    valid = [int(x) for x in ids if 0 <= int(x) < logits.numel()]
    if not valid:
        return -1e30, -1, -1
    idx = torch.tensor(valid, dtype=torch.long, device=logits.device)
    values = logits[idx]
    pos = int(torch.argmax(values).item())
    token_id = int(idx[pos].item())
    rank = int((logits > values[pos]).sum().item()) + 1
    return float(values[pos].item()), token_id, rank


def next_token_logits(model: Any, tokenizer: Any, device: torch.device, prompt: str) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        out = model(**encoded)
    return out.logits[0, -1, :].detach().float().cpu()


def generate_text(model: Any, tokenizer: Any, device: torch.device, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_len = int(encoded["input_ids"].shape[-1])
    encoded = {k: v.to(device) for k, v in encoded.items()}
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        generated = model.generate(**encoded, **kwargs)
    new_ids = generated[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def classify_output(text: str, target: str, expected_pattern: str) -> dict[str, Any]:
    normalized = text.strip()
    low = normalized.lower()
    target_low = target.strip().lower()
    first_word = low.replace("\n", " ").split()[0].strip(".,:;!?，。；：！？") if low.split() else ""
    contains_target = bool(target_low) and target_low in low
    starts_target = bool(target_low) and low.startswith(target_low)
    exact_first = bool(target_low) and first_word == target_low.strip(".,:;!?，。；：！？")
    token_count = len(normalized.replace("\n", " ").split())
    has_because = "because" in low or "因为" in normalized
    has_repeat_comma = "," in normalized or "，" in normalized
    has_period = normalized.endswith(".") or normalized.endswith("。")
    if not normalized:
        drift_type = "empty"
    elif not contains_target:
        drift_type = "wrong_or_missing_target"
    elif expected_pattern in {"short", "short_answer"} and token_count > 3:
        drift_type = "over_generation"
    elif "because" in expected_pattern and not has_because:
        drift_type = "missing_reason_marker"
    elif "repeat" in expected_pattern and not has_repeat_comma:
        drift_type = "missing_repeat_separator"
    else:
        drift_type = "none"
    pattern_match = drift_type == "none" and (contains_target or starts_target or exact_first)
    behavior_score = 0.0
    if contains_target:
        behavior_score += 0.45
    if starts_target or exact_first:
        behavior_score += 0.25
    if pattern_match:
        behavior_score += 0.30
    return {
        "output_text": normalized,
        "output_token_count": token_count,
        "contains_target": contains_target,
        "starts_target": starts_target,
        "exact_first_token_match": exact_first,
        "pattern_match": pattern_match,
        "behavior_score": round(behavior_score, 4),
        "drift_type": drift_type,
        "has_because": has_because,
        "has_repeat_comma": has_repeat_comma,
        "has_period": has_period,
        "stop_type": "eos_or_limit_unknown" if normalized else "empty",
    }


def readout_metrics(tokenizer: Any, logits: torch.Tensor, target: str) -> dict[str, Any]:
    target_ids = first_token_candidates(tokenizer, target)
    target_logit, target_token_id, target_rank = max_group_score(logits, target_ids)
    regime_scores: dict[str, float] = {}
    for regime, texts in REGIME_TEXTS.items():
        score, _token_id, _rank = max_group_score(logits, token_ids_for_texts(tokenizer, texts))
        regime_scores[regime] = score
    winning_regime = max(regime_scores.items(), key=lambda item: item[1])[0] if regime_scores else "none"
    winning_score = regime_scores.get(winning_regime, -1e30)
    top_id = int(torch.argmax(logits).item())
    return {
        "target_token_id": int(target_token_id),
        "target_logit": float(target_logit),
        "target_rank": int(target_rank),
        "top_token_id": top_id,
        "top_token": tokenizer.decode([top_id]),
        "winning_regime": winning_regime,
        "winning_regime_logit": float(winning_score),
        "target_margin_vs_winner": float(target_logit - winning_score),
        "regime_scores": regime_scores,
    }


def expanded_cases(max_cases: int) -> list[dict[str, Any]]:
    rows = read_jsonl(ATLAS_ROOT / "test_cases.jsonl")
    base = []
    for row in rows:
        base.append(
            {
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "prompt": row["prompt"],
                "target": row["target"],
                "expected_pattern": row.get("expected_pattern", ""),
            }
        )
    cases = base + EXTRA_CASES
    seen = set()
    unique = []
    for case in cases:
        cid = str(case["case_id"])
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(case)
    return unique[: int(max_cases)]


def observation_rows_for_case(
    model_name: str,
    run_id: str,
    case: dict[str, Any],
    generation: dict[str, Any],
    readout: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase236",
        "created_at": utc_now(),
        "run_id": run_id,
        "case_id": case["case_id"],
        "model": model_name,
        "family_id": case["family_id"],
        "mode_id": case["mode_id"],
    }
    metrics = {
        "behavior_score": generation["behavior_score"],
        "pattern_match": float(bool(generation["pattern_match"])),
        "contains_target": float(bool(generation["contains_target"])),
        "starts_target": float(bool(generation["starts_target"])),
        "output_token_count": generation["output_token_count"],
        "target_rank": readout["target_rank"],
        "target_logit": readout["target_logit"],
        "target_margin_vs_winner": readout["target_margin_vs_winner"],
    }
    rows = []
    for name, value in metrics.items():
        rows.append(
            {
                **common,
                "observation_id": f"phase236:{model_name}:{case['case_id']}:{name}",
                "level": "behavior" if name in {"behavior_score", "pattern_match", "contains_target", "starts_target", "output_token_count"} else "readout_competition",
                "metric_name": name,
                "metric_value": finite_float(value),
                "metric_unit": "score",
                "output_text": generation["output_text"],
                "drift_type": generation["drift_type"],
                "winning_regime": readout["winning_regime"],
                "top_token": readout["top_token"],
            }
        )
    return rows


def summarize_observations(case_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_mode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_family[(row["model"], row["family_id"])].append(row)
        by_mode[(row["model"], row["family_id"], row["mode_id"])].append(row)
    now = utc_now()
    metric_rows: list[dict[str, Any]] = []
    graph_updates: list[dict[str, Any]] = []
    for (model, family_id), items in sorted(by_family.items()):
        avg = sum(finite_float(x["behavior_score"]) for x in items) / len(items)
        match_rate = sum(1 for x in items if x["pattern_match"]) / len(items)
        metric_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase236",
                "created_at": now,
                "metric_id": f"phase236:{model}:family:{family_id}:behavior_score",
                "scope": "family",
                "model": model,
                "family_id": family_id,
                "metric_name": "mean_behavior_score",
                "metric_value": round(avg, 4),
                "case_count": len(items),
                "pattern_match_rate": round(match_rate, 4),
                "drift_types": dict(Counter(str(x["drift_type"]) for x in items).most_common()),
            }
        )
    for (model, family_id, mode_id), items in sorted(by_mode.items()):
        avg = sum(finite_float(x["behavior_score"]) for x in items) / len(items)
        match_rate = sum(1 for x in items if x["pattern_match"]) / len(items)
        metric_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase236",
                "created_at": now,
                "metric_id": f"phase236:{model}:mode:{family_id}:{mode_id}:behavior_score",
                "scope": "mode",
                "model": model,
                "family_id": family_id,
                "mode_id": mode_id,
                "metric_name": "mean_behavior_score",
                "metric_value": round(avg, 4),
                "case_count": len(items),
                "pattern_match_rate": round(match_rate, 4),
                "drift_types": dict(Counter(str(x["drift_type"]) for x in items).most_common()),
            }
        )
        graph_updates.append(
            {
                "edge_id": f"phase236:{model}:{family_id}:{mode_id}:behavior_support",
                "source": f"mode:{family_id}:{mode_id}",
                "target": f"model:{model}",
                "edge_type": "behavior_benchmark_support",
                "family_id": family_id,
                "mode_id": mode_id,
                "model": model,
                "evidence_type": "generation_behavior",
                "effect_direction": "positive" if avg >= 0.5 else "weak_or_negative",
                "effect_size": round(avg, 4),
                "confidence": round(0.25 + min(0.45, len(items) / 80.0) + min(0.20, avg * 0.20), 4),
                "supporting_phases": ["Phase236"],
                "status": "behavior_tested",
            }
        )
    return metric_rows, graph_updates


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"phase236:{args.model}:{args.round_name}"
    cases = expanded_cases(int(args.max_cases))
    model = None
    tokenizer = None
    case_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, case in enumerate(cases, start=1):
            prompt = str(case["prompt"])
            target = str(case["target"])
            logits = next_token_logits(model, tokenizer, device, prompt)
            readout = readout_metrics(tokenizer, logits, target)
            output = generate_text(model, tokenizer, device, prompt, int(args.max_new_tokens))
            generation = classify_output(output, target, str(case.get("expected_pattern") or ""))
            row = {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "run_id": run_id,
                "model": args.model,
                "case_index": idx,
                **case,
                **generation,
                **readout,
            }
            case_rows.append(row)
            observation_rows.extend(observation_rows_for_case(args.model, run_id, case, generation, readout))
            if idx % 12 == 0:
                log(f"{args.model}: cases={idx}/{len(cases)}")
            del logits
            if torch.cuda.is_available() and idx % 12 == 0:
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_rows, graph_updates = summarize_observations(case_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Pattern family behavior benchmark",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "run_id": run_id,
        "case_count": len(case_rows),
        "observation_rows": len(observation_rows),
        "metric_rows": len(metric_rows),
        "mean_behavior_score": round(sum(finite_float(x["behavior_score"]) for x in case_rows) / max(1, len(case_rows)), 4),
        "pattern_match_rate": round(sum(1 for x in case_rows if x["pattern_match"]) / max(1, len(case_rows)), 4),
        "drift_types": dict(Counter(str(x["drift_type"]) for x in case_rows).most_common()),
        "top_rows": sorted(case_rows, key=lambda x: finite_float(x["behavior_score"]))[:20],
    }
    write_json(out_dir / f"phase236_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase236_{args.model}_case_rows.jsonl", case_rows)
    write_jsonl(out_dir / f"phase236_{args.model}_observations.jsonl", observation_rows)
    write_jsonl(out_dir / f"phase236_{args.model}_metrics.jsonl", metric_rows)
    write_jsonl(out_dir / f"phase236_{args.model}_graph_edges.jsonl", graph_updates)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "cases": len(case_rows)}, ensure_ascii=False, indent=2))
    return payload


def append_unique_jsonl(path: Path, new_rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + new_rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    case_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    summaries = []
    for model in MODELS:
        summaries.append(read_json(out_dir / f"phase236_{model}_summary.json"))
        case_rows.extend(read_jsonl(out_dir / f"phase236_{model}_case_rows.jsonl"))
        observation_rows.extend(read_jsonl(out_dir / f"phase236_{model}_observations.jsonl"))
        metric_rows.extend(read_jsonl(out_dir / f"phase236_{model}_metrics.jsonl"))
        graph_edges.extend(read_jsonl(out_dir / f"phase236_{model}_graph_edges.jsonl"))
    summaries = [x for x in summaries if x]
    cross = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model pattern family behavior benchmark",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "case_rows": len(case_rows),
        "observation_rows": len(observation_rows),
        "metric_rows": len(metric_rows),
        "mean_behavior_score": round(sum(finite_float(x.get("behavior_score")) for x in case_rows) / max(1, len(case_rows)), 4),
        "pattern_match_rate": round(sum(1 for x in case_rows if x.get("pattern_match")) / max(1, len(case_rows)), 4),
        "drift_types": dict(Counter(str(x.get("drift_type")) for x in case_rows).most_common()),
    }
    write_json(out_dir / "phase236_cross_model_summary.json", cross)
    write_jsonl(out_dir / "phase236_cross_model_case_rows.jsonl", case_rows)
    write_jsonl(out_dir / "phase236_cross_model_observations.jsonl", observation_rows)
    write_jsonl(out_dir / "phase236_cross_model_metrics.jsonl", metric_rows)
    write_jsonl(out_dir / "phase236_cross_model_graph_edges.jsonl", graph_edges)
    update_atlas(round_name, cross, observation_rows, metric_rows, graph_edges)
    write_summary_md(out_dir / "phase236_cross_model_summary.md", cross, summaries)
    print(json.dumps({"phase": PHASE, "status": "complete", "models": cross["models"], "case_rows": len(case_rows)}, ensure_ascii=False, indent=2))
    return cross


def update_atlas(round_name: str, cross: dict[str, Any], observation_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]], graph_edges: list[dict[str, Any]]) -> None:
    run_rows = read_jsonl(ATLAS_ROOT / "runs.jsonl")
    for model in cross.get("models") or []:
        run_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase236",
                "created_at": utc_now(),
                "run_id": f"phase236:{model}:{round_name}",
                "model": model,
                "status": "complete",
                "started_at": cross["timestamp"],
                "finished_at": cross["timestamp"],
                "source_phase": "Phase235",
                "case_count": sum(1 for row in observation_rows if row.get("model") == model and row.get("metric_name") == "behavior_score"),
            }
        )
    append_unique_jsonl(ATLAS_ROOT / "runs.jsonl", run_rows, "run_id")
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observation_rows, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metric_rows, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", graph_edges, "edge_id")
    progress = read_json(ATLAS_ROOT / "progress.json")
    if progress:
        progress["phase_id"] = "Phase236"
        progress["created_at"] = utc_now()
        progress.setdefault("global_progress", {})["pattern_family_atlas"] = 0.41
        progress.setdefault("global_progress", {})["general_language_mechanism_confidence"] = 0.45
        progress.setdefault("levels", {})["behavior"] = 0.46
        progress["next_phase"] = "Phase237_prompt_trigger_family_atlas"
        progress["latest_phase"] = {
            "phase_id": "Phase236",
            "title": "模式族行为基准测试",
            "case_rows": cross["case_rows"],
            "observation_rows": cross["observation_rows"],
            "mean_behavior_score": cross["mean_behavior_score"],
            "pattern_match_rate": cross["pattern_match_rate"],
        }
        write_json(ATLAS_ROOT / "progress.json", progress)
    summary_path = ATLAS_ROOT / "summary.md"
    summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = "## Phase236 Behavior Benchmark Update"
    if marker in summary:
        summary = summary.split(marker, 1)[0].rstrip()
    addition = (
        f"\n{marker}\n\n"
        f"- models: {', '.join(str(x) for x in cross.get('models') or [])}\n"
        f"- case_rows: {cross['case_rows']}\n"
        f"- observation_rows: {cross['observation_rows']}\n"
        f"- mean_behavior_score: {cross['mean_behavior_score']}\n"
        f"- pattern_match_rate: {cross['pattern_match_rate']}\n"
        f"- drift_types: {cross['drift_types']}\n"
    )
    summary_path.write_text(summary.rstrip() + "\n" + addition, encoding="utf-8")


def write_summary_md(path: Path, cross: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lines = ["# Phase 236 Pattern Family Behavior Benchmark", ""]
    lines.append(f"models: {', '.join(str(x.get('model')) for x in summaries if x)}")
    lines.append(f"case_rows: {cross['case_rows']}")
    lines.append(f"observation_rows: {cross['observation_rows']}")
    lines.append(f"mean_behavior_score: {cross['mean_behavior_score']}")
    lines.append(f"pattern_match_rate: {cross['pattern_match_rate']}")
    lines.extend(["", "## Model Summary", "", "| model | cases | mean score | match rate | drift types |", "| --- | ---: | ---: | ---: | --- |"])
    for row in summaries:
        if not row:
            continue
        lines.append(
            f"| {row.get('model')} | {row.get('case_count')} | {finite_float(row.get('mean_behavior_score')):.4f} | "
            f"{finite_float(row.get('pattern_match_rate')):.4f} | {row.get('drift_types')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase236 pattern family behavior benchmark")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="pattern_family_behavior_benchmark")
    parser.add_argument("--max-cases", type=int, default=44)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    args = parser.parse_args()
    if not args.summarize and not args.model:
        parser.error("--model is required unless --summarize is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
