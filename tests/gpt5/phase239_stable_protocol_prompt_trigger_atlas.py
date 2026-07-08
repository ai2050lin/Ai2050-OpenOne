#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
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


PHASE = 239
SOURCE_PHASE = 238
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE236_ROOT = Path("tests/result/phase236_pattern_family_behavior_benchmark/pattern_family_behavior_benchmark")
RESULT_ROOT = Path("tests/result/phase239_stable_protocol_prompt_trigger_atlas")


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def append_unique_jsonl(path: Path, new_rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + new_rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def words(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def contains_alias(output: str, aliases: list[str]) -> tuple[bool, str]:
    low = output.lower()
    word_set = words(output)
    for alias in aliases:
        a = str(alias).lower()
        if re.search(r"[\u4e00-\u9fff]", a):
            if a in output:
                return True, alias
        elif " " in a:
            if a in low:
                return True, alias
        elif a in word_set or a in low:
            return True, alias
    return False, ""


def first_token_candidates(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        for candidate in [str(text), " " + str(text)]:
            toks = tokenizer.encode(candidate, add_special_tokens=False)
            if toks:
                ids.append(int(toks[0]))
    return sorted(set(ids))


def token_ids_for_texts(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
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
    return tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()


def readout_metrics(tokenizer: Any, logits: torch.Tensor, aliases: list[str]) -> dict[str, Any]:
    target_ids = first_token_candidates(tokenizer, aliases)
    target_logit, target_token_id, target_rank = max_group_score(logits, target_ids)
    regime_scores: dict[str, float] = {}
    for regime, texts in REGIME_TEXTS.items():
        score, _token_id, _rank = max_group_score(logits, token_ids_for_texts(tokenizer, texts))
        regime_scores[regime] = score
    ranked = sorted(regime_scores.items(), key=lambda item: item[1], reverse=True)
    winning_regime = ranked[0][0] if ranked else "none"
    second_regime = ranked[1][0] if len(ranked) > 1 else "none"
    winning_score = regime_scores.get(winning_regime, -1e30)
    top_id = int(torch.argmax(logits).item())
    return {
        "target_token_id": int(target_token_id),
        "target_logit": float(target_logit),
        "target_rank": int(target_rank),
        "top_token_id": top_id,
        "top_token": tokenizer.decode([top_id]),
        "winning_regime": winning_regime,
        "second_competitor": second_regime,
        "winning_regime_logit": float(winning_score),
        "target_margin_vs_winner": float(target_logit - winning_score),
        "regime_scores": regime_scores,
    }


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt).strip()


def strip_answer_anchor(prompt: str) -> str:
    return re.sub(r"\n?Answer:\s*$", "", prompt.strip(), flags=re.I)


def prompt_variants(base_prompt: str, target: str) -> dict[str, str]:
    stem = strip_answer_anchor(base_prompt)
    one_word = f"{stem}\nAnswer with exactly one word. Do not explain.\nAnswer:"
    variants = {
        "full": base_prompt,
        "no_answer_anchor": stem,
        "strong_answer_anchor": f"{stem}\nFinal answer only. One word. No explanation.\nAnswer:",
        "one_word_strict": one_word,
        "one_word_no_explain": f"{stem}\nOutput only the answer word, with no sentence and no explanation.\nAnswer:",
        "period_forced": f"{stem}\nOutput exactly one word followed by a period, then stop.\nAnswer:",
        "newline_removed": normalize_prompt(base_prompt),
        "colon_removed": base_prompt.replace("Answer:", "Answer"),
        "short_answer_instruction": f"{stem}\nAnswer with one word.\nAnswer:",
        "explain_instruction": f"{stem}\nAnswer with the answer first, then one short reason using because.\nAnswer:",
        "target_seeded": f"{stem}\nAnswer: {target}",
    }
    return variants


def original_protocol_match(output: str, aliases: list[str], expected_pattern: str) -> bool:
    normalized = output.strip()
    low = normalized.lower()
    alias_hit, _matched_alias = contains_alias(normalized, aliases)
    token_count = len(normalized.replace("\n", " ").split())
    has_because = "because" in low or "因为" in normalized
    has_answer_loop = "answer:" in low
    expected = str(expected_pattern or "").lower()
    if expected in {"short", "short_answer"}:
        return alias_hit and token_count <= 3 and not has_because and not has_answer_loop
    if "explain" in expected or "because" in expected:
        return alias_hit and has_because
    if "repeat" in expected:
        return alias_hit and ("," in normalized or "，" in normalized)
    if "list" in expected:
        return alias_hit and ("\n" in normalized or "," in normalized or "1." in normalized or "-" in normalized)
    return alias_hit and token_count <= 8 and not has_answer_loop


def classify_output(output: str, aliases: list[str], variant: str, expected_pattern: str) -> dict[str, Any]:
    normalized = output.strip()
    low = normalized.lower()
    alias_hit, matched_alias = contains_alias(normalized, aliases)
    token_count = len(normalized.replace("\n", " ").split())
    starts_alias = any(low.startswith(str(a).lower()) for a in aliases if str(a))
    has_because = "because" in low or "因为" in normalized
    has_answer_loop = "answer:" in low
    strict_variant = variant in {"strong_answer_anchor", "one_word_strict", "one_word_no_explain", "short_answer_instruction", "period_forced", "target_seeded"}
    if variant == "explain_instruction":
        protocol_ok = alias_hit and has_because
    elif variant == "period_forced":
        protocol_ok = alias_hit and token_count <= 4 and (normalized.endswith(".") or normalized.endswith("。"))
    elif strict_variant:
        protocol_ok = alias_hit and token_count <= 3 and not has_because and not has_answer_loop
    else:
        protocol_ok = alias_hit and (token_count <= 6 or starts_alias)
    original_ok = original_protocol_match(normalized, aliases, expected_pattern)
    closure_ok = token_count <= 24 and not has_answer_loop
    original_over_generation = alias_hit and not original_ok
    if not normalized:
        drift_type = "empty"
    elif not alias_hit:
        drift_type = "semantic_or_target_failure"
    elif original_over_generation:
        drift_type = "protocol_or_over_generation"
    elif not closure_ok:
        drift_type = "closure_or_rollout_failure"
    else:
        drift_type = "none"
    variant_score = 0.35 * float(alias_hit) + 0.25 * float(protocol_ok) + 0.25 * float(alias_hit) + 0.15 * float(closure_ok)
    objective_score = 0.35 * float(alias_hit) + 0.25 * float(original_ok) + 0.25 * float(alias_hit) + 0.15 * float(closure_ok)
    return {
        "output_text": normalized,
        "output_token_count": token_count,
        "answer_hit": alias_hit,
        "matched_alias": matched_alias,
        "starts_alias": starts_alias,
        "variant_protocol_match": protocol_ok,
        "original_protocol_match": original_ok,
        "protocol_match": original_ok,
        "closure_signal": closure_ok,
        "over_generation": original_over_generation,
        "has_because": has_because,
        "has_answer_loop": has_answer_loop,
        "variant_behavior_score": round(variant_score, 4),
        "calibrated_behavior_score": round(objective_score, 4),
        "drift_type": drift_type,
    }


def load_selected_cases(max_cases: int) -> list[dict[str, Any]]:
    candidates = read_jsonl(ATLAS_ROOT / "stable_failure_candidates.jsonl")
    aliases = {row["case_id"]: row for row in read_jsonl(ATLAS_ROOT / "case_aliases.jsonl")}
    phase236_rows = read_jsonl(PHASE236_ROOT / "phase236_cross_model_case_rows.jsonl")
    by_case: dict[str, dict[str, Any]] = {}
    for row in phase236_rows:
        by_case.setdefault(row["case_id"], row)
    selected = []
    for cand in candidates:
        if cand.get("failure_type") != "stable_protocol_failure":
            continue
        case_id = cand["case_id"]
        source = by_case.get(case_id)
        alias = aliases.get(case_id)
        if not source or not alias:
            continue
        selected.append(
            {
                "case_id": case_id,
                "family_id": cand["family_id"],
                "mode_id": cand["mode_id"],
                "failure_type": cand["failure_type"],
                "prompt": source["prompt"],
                "target": source["target"],
                "expected_pattern": source.get("expected_pattern", ""),
                "target_aliases": alias.get("target_aliases") or [source["target"]],
                "relation_schema": alias.get("relation_schema", ""),
                "answer_policy": alias.get("answer_policy", ""),
            }
        )
    selected.sort(key=lambda row: (row["family_id"] != "output_protocol", row["case_id"]))
    return selected[: int(max_cases)]


def observation_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase239",
        "created_at": utc_now(),
        "run_id": row["run_id"],
        "case_id": row["case_id"],
        "model": row["model"],
        "family_id": row["family_id"],
        "mode_id": row["mode_id"],
    }
    metrics = {
        "calibrated_behavior_score": row["calibrated_behavior_score"],
        "protocol_match": float(row["protocol_match"]),
        "variant_protocol_match": float(row["variant_protocol_match"]),
        "over_generation": float(row["over_generation"]),
        "closure_signal": float(row["closure_signal"]),
        "target_rank": row["target_rank"],
        "target_margin_vs_winner": row["target_margin_vs_winner"],
        "baseline_score_delta": row.get("baseline_score_delta", 0.0),
        "baseline_margin_delta": row.get("baseline_margin_delta", 0.0),
    }
    return [
        {
            **common,
            "observation_id": f"phase239:{row['model']}:{row['case_id']}:{row['variant_id']}:{name}",
            "level": "prompt_trigger",
            "metric_name": name,
            "metric_value": float(value),
            "metric_unit": "score",
            "variant_id": row["variant_id"],
            "winning_regime": row["winning_regime"],
            "second_competitor": row["second_competitor"],
            "drift_type": row["drift_type"],
        }
        for name, value in metrics.items()
    ]


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_selected_cases(int(args.max_cases))
    run_id = f"phase239:{args.model}:{args.round_name}"
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for case_index, case in enumerate(cases, start=1):
            variant_prompts = prompt_variants(str(case["prompt"]), str(case["target"]))
            baseline: dict[str, Any] | None = None
            for variant_id, prompt in variant_prompts.items():
                logits = next_token_logits(model, tokenizer, device, prompt)
                readout = readout_metrics(tokenizer, logits, list(case["target_aliases"]))
                output = generate_text(model, tokenizer, device, prompt, int(args.max_new_tokens))
                behavior = classify_output(output, list(case["target_aliases"]), variant_id, str(case.get("expected_pattern") or ""))
                row = {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "schema_version": SCHEMA_VERSION,
                    "created_at": utc_now(),
                    "run_id": run_id,
                    "model": args.model,
                    "case_index": case_index,
                    **case,
                    "variant_id": variant_id,
                    "prompt_variant": prompt,
                    **readout,
                    **behavior,
                }
                if variant_id == "full":
                    baseline = row
                    row["baseline_score_delta"] = 0.0
                    row["baseline_margin_delta"] = 0.0
                    row["winner_changed_vs_baseline"] = False
                elif baseline is not None:
                    row["baseline_score_delta"] = round(float(row["calibrated_behavior_score"]) - float(baseline["calibrated_behavior_score"]), 4)
                    row["baseline_margin_delta"] = round(float(row["target_margin_vs_winner"]) - float(baseline["target_margin_vs_winner"]), 4)
                    row["winner_changed_vs_baseline"] = row["winning_regime"] != baseline["winning_regime"]
                rows.append(row)
                observations.extend(observation_rows(row))
                del logits
            log(f"{args.model}: case={case_index}/{len(cases)} rows={len(rows)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_rows, edge_rows = summarize_model(args.model, rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Stable protocol prompt trigger atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "case_count": len(cases),
        "variant_rows": len(rows),
        "observation_rows": len(observations),
        "metric_rows": len(metric_rows),
        "mean_score": round(sum(float(x["calibrated_behavior_score"]) for x in rows) / max(1, len(rows)), 4),
        "protocol_match_rate": round(sum(1 for x in rows if x["protocol_match"]) / max(1, len(rows)), 4),
        "best_variants": best_variants(rows)[:20],
    }
    write_json(out_dir / f"phase239_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase239_{args.model}_prompt_trigger_rows.jsonl", rows)
    write_jsonl(out_dir / f"phase239_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase239_{args.model}_metrics.jsonl", metric_rows)
    write_jsonl(out_dir / f"phase239_{args.model}_graph_edges.jsonl", edge_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "variant_rows": len(rows)}, ensure_ascii=False, indent=2))
    return payload


def best_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["variant_id"]].append(row)
    out = []
    for variant, items in buckets.items():
        out.append(
            {
                "variant_id": variant,
                "rows": len(items),
                "mean_score": round(sum(float(x["calibrated_behavior_score"]) for x in items) / len(items), 4),
                "protocol_match_rate": round(sum(1 for x in items if x["protocol_match"]) / len(items), 4),
                "over_generation_rate": round(sum(1 for x in items if x["over_generation"]) / len(items), 4),
                "mean_score_delta": round(sum(float(x.get("baseline_score_delta") or 0.0) for x in items) / len(items), 4),
                "winner_regimes": dict(Counter(str(x["winning_regime"]) for x in items).most_common()),
            }
        )
    out.sort(key=lambda row: (row["mean_score"], row["protocol_match_rate"], row["mean_score_delta"]), reverse=True)
    return out


def summarize_model(model_name: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    now = utc_now()
    metrics = []
    edges = []
    for row in best_variants(rows):
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase239",
                "created_at": now,
                "metric_id": f"phase239:{model_name}:variant:{row['variant_id']}:prompt_trigger",
                "scope": "prompt_variant",
                "model": model_name,
                "family_id": "output_protocol",
                "mode_id": "prompt_trigger",
                "variant_id": row["variant_id"],
                "metric_name": "mean_calibrated_behavior_score",
                "metric_value": row["mean_score"],
                "protocol_match_rate": row["protocol_match_rate"],
                "over_generation_rate": row["over_generation_rate"],
                "mean_score_delta": row["mean_score_delta"],
                "winner_regimes": row["winner_regimes"],
            }
        )
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase239",
                "created_at": now,
                "edge_id": f"phase239:{model_name}:prompt_variant:{row['variant_id']}",
                "source": f"prompt_variant:{row['variant_id']}",
                "target": "mode:output_protocol:short_answer",
                "edge_type": "prompt_anchor_ablation",
                "family_id": "output_protocol",
                "mode_id": "short_answer",
                "model": model_name,
                "evidence_type": "prompt_trigger_behavior_readout",
                "effect_direction": "positive" if row["mean_score_delta"] > 0 else "negative_or_neutral",
                "effect_size": row["mean_score_delta"],
                "confidence": round(0.25 + min(0.35, row["protocol_match_rate"] * 0.35) + min(0.20, max(0.0, row["mean_score_delta"]) * 0.20), 4),
                "supporting_phases": ["Phase239"],
                "status": "trigger_mapped",
            }
        )
    return metrics, edges


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    summaries = []
    for model in MODELS:
        summaries.append(read_json(out_dir / f"phase239_{model}_summary.json"))
        rows.extend(read_jsonl(out_dir / f"phase239_{model}_prompt_trigger_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase239_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase239_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase239_{model}_graph_edges.jsonl"))
    summaries = [x for x in summaries if x]
    cross_best = best_variants(rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model stable protocol prompt trigger atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "variant_rows": len(rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "mean_score": round(sum(float(x["calibrated_behavior_score"]) for x in rows) / max(1, len(rows)), 4),
        "protocol_match_rate": round(sum(1 for x in rows if x["protocol_match"]) / max(1, len(rows)), 4),
        "best_variants": cross_best[:30],
    }
    write_json(out_dir / "phase239_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase239_cross_model_prompt_trigger_rows.jsonl", rows)
    write_jsonl(out_dir / "phase239_cross_model_observations.jsonl", observations)
    write_jsonl(out_dir / "phase239_cross_model_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase239_cross_model_graph_edges.jsonl", edges)
    write_report(out_dir / "phase239_protocol_failure_report.md", payload, summaries)
    write_json(out_dir / "phase239_stable_failure_selection.json", {"phase": PHASE, "selected_cases": load_selected_cases(999)})
    update_atlas(payload, observations, metrics, edges)
    print(json.dumps({"phase": PHASE, "status": "complete", "models": payload["models"], "variant_rows": len(rows)}, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, payload: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lines = ["# Phase239 Stable Protocol Prompt Trigger Atlas", ""]
    lines.append(f"variant_rows: {payload['variant_rows']}")
    lines.append(f"mean_score: {payload['mean_score']}")
    lines.append(f"protocol_match_rate: {payload['protocol_match_rate']}")
    lines.extend(["", "## Model Summary", "", "| model | rows | mean score | protocol match |", "| --- | ---: | ---: | ---: |"])
    for row in summaries:
        lines.append(f"| {row.get('model')} | {row.get('variant_rows')} | {row.get('mean_score')} | {row.get('protocol_match_rate')} |")
    lines.extend(["", "## Best Variants", "", "| variant | rows | mean score | protocol match | over generation | score delta | winners |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in payload["best_variants"][:30]:
        lines.append(
            f"| {row['variant_id']} | {row['rows']} | {row['mean_score']} | {row['protocol_match_rate']} | "
            f"{row['over_generation_rate']} | {row['mean_score_delta']} | {row['winner_regimes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_atlas(payload: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress = read_json(ATLAS_ROOT / "progress.json")
    if progress:
        progress["phase_id"] = "Phase239"
        progress["created_at"] = utc_now()
        progress.setdefault("global_progress", {})["pattern_family_atlas"] = 0.52
        progress.setdefault("global_progress", {})["general_language_mechanism_confidence"] = 0.47
        progress.setdefault("levels", {})["prompt_trigger"] = 0.32
        progress["next_phase"] = "Phase240_gate_product_protocol_trace"
        progress["latest_phase"] = {
            "phase_id": "Phase239",
            "title": "稳定协议失败 prompt trigger / anchor 消融",
            "variant_rows": payload["variant_rows"],
            "observation_rows": payload["observation_rows"],
            "mean_score": payload["mean_score"],
            "protocol_match_rate": payload["protocol_match_rate"],
            "best_variant": payload["best_variants"][0]["variant_id"] if payload["best_variants"] else "",
        }
        write_json(ATLAS_ROOT / "progress.json", progress)
    summary_path = ATLAS_ROOT / "summary.md"
    old = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = "## Phase239 Prompt Trigger Update"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip()
    addition = (
        f"\n{marker}\n\n"
        f"- variant_rows: {payload['variant_rows']}\n"
        f"- observation_rows: {payload['observation_rows']}\n"
        f"- mean_score: {payload['mean_score']}\n"
        f"- protocol_match_rate: {payload['protocol_match_rate']}\n"
        f"- best_variants: {payload['best_variants'][:5]}\n"
    )
    summary_path.write_text(old.rstrip() + "\n" + addition, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase239 stable protocol prompt trigger atlas")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="stable_protocol_prompt_trigger_atlas")
    parser.add_argument("--max-cases", type=int, default=8)
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
