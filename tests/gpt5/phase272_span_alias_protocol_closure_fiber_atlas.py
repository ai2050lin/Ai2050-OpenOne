#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase271_mlp_writer_direction_closure_fiber_audit as p271  # noqa: E402


PHASE = 272
SOURCE_PHASE = 271
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase272_span_alias_protocol_closure_fiber_atlas")
ROUND_DEFAULT = "span_alias_protocol_closure_fiber_atlas"
PATCH_TYPES = p271.PATCH_TYPES


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


def append_unique_jsonl(path: Path, rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def token_ids(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        for variant in [text, " " + text if not text.startswith(" ") else text]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.append(int(encoded[0]))
    return sorted(set(ids))


def max_ids(logits: torch.Tensor, ids: list[int]) -> float:
    if not ids:
        return -1e30
    return float(logits.detach().float().cpu()[ids].max().item())


def completion_logprob(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, completion: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    comp_ids = full_ids[len(prompt_ids) :]
    if not comp_ids:
        alt = tokenizer(completion, add_special_tokens=False)["input_ids"]
        comp_ids = alt
        full_ids = prompt_ids + comp_ids
    if not comp_ids:
        return {"sum_logprob": None, "mean_logprob": None, "n_tokens": 0}
    input_ids = torch.tensor([full_ids], device=device)
    handles = p271.install_mlp_hooks(model_obj, specs)
    try:
        with torch.inference_mode():
            out = model_obj(input_ids=input_ids, use_cache=False, return_dict=True)
        logits = out.logits[0].detach().float()
    finally:
        for handle in handles:
            handle.remove()
    vals: list[float] = []
    start = len(prompt_ids)
    for i, tok in enumerate(comp_ids):
        pos = start + i - 1
        if pos < 0 or pos >= logits.shape[0]:
            continue
        lp = torch.log_softmax(logits[pos], dim=-1)[int(tok)].item()
        vals.append(float(lp))
    return {
        "sum_logprob": round(sum(vals), 6) if vals else None,
        "mean_logprob": round(sum(vals) / len(vals), 6) if vals else None,
        "n_tokens": len(vals),
    }


def next_logits_after(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, completion: str, specs: list[dict[str, Any]]) -> torch.Tensor:
    encoded = tokenizer(prompt + completion, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    handles = p271.install_mlp_hooks(model_obj, specs)
    try:
        with torch.inference_mode():
            out = model_obj(**encoded, use_cache=False, return_dict=True)
        return out.logits[0, last_pos].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def protocol_forms(source: dict[str, Any]) -> list[dict[str, str]]:
    aliases = [str(x) for x in source.get("target_aliases") or [source.get("target") or source.get("expected_answer")]]
    aliases = [a for a in aliases if a and a != "None"]
    protocol = str(source.get("output_protocol", "short"))
    forms: list[dict[str, str]] = []
    for alias in aliases:
        forms.append({"form_type": "alias_plain", "completion": alias})
        forms.append({"form_type": "alias_leading_space", "completion": " " + alias})
        forms.append({"form_type": "alias_period", "completion": alias + "."})
        if "json" in protocol:
            forms.append({"form_type": "protocol_json", "completion": "{\"answer\":\"" + alias + "\"}"})
        if "list" in protocol:
            forms.append({"form_type": "protocol_list", "completion": " " + alias})
        if "explain" in protocol:
            forms.append({"form_type": "protocol_explain", "completion": alias + " because"})
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for form in forms:
        key = form["form_type"] + "\t" + form["completion"]
        if key not in seen:
            seen.add(key)
            out.append(form)
    return out[:8]


def stop_gate_scores(tokenizer: Any, logits: torch.Tensor) -> dict[str, Any]:
    period = max_ids(logits, token_ids(tokenizer, [".", ".\n"]))
    eos = float(logits[int(tokenizer.eos_token_id)].item()) if tokenizer.eos_token_id is not None else -1e30
    newline = max_ids(logits, token_ids(tokenizer, ["\n", "\n\n"]))
    continue_score = max_ids(logits, token_ids(tokenizer, [" because", " and", ",", " the", " is", " therefore", "\n-"]))
    stop_gate = max(period, eos, newline)
    return {
        "period_logit": round(period, 6),
        "eos_logit": round(eos, 6),
        "newline_logit": round(newline, 6),
        "continue_after_answer_logit": round(continue_score, 6),
        "protocol_stop_logit": round(stop_gate, 6),
        "protocol_stop_margin": round(stop_gate - continue_score, 6),
    }


def closure_quality(base: dict[str, Any], patched: dict[str, Any], patch_type: str) -> dict[str, Any]:
    span_delta = safe_float(patched.get("best_span_mean_logprob")) - safe_float(base.get("best_span_mean_logprob"))
    protocol_delta = safe_float(patched.get("best_protocol_mean_logprob")) - safe_float(base.get("best_protocol_mean_logprob"))
    stop_delta = safe_float(patched.get("protocol_stop_margin")) - safe_float(base.get("protocol_stop_margin"))
    continue_delta = safe_float(patched.get("continue_stop_margin")) - safe_float(base.get("continue_stop_margin"))
    side_effect = abs(span_delta) + abs(protocol_delta) + max(0.0, -stop_delta)
    score = span_delta + protocol_delta + stop_delta - side_effect
    strict_clean = bool(
        continue_delta < 0.0
        and side_effect < 2.5
        and stop_delta > -1.0
        and patch_type != "window_mlp_random_same_norm"
    )
    return {
        "span_mean_logprob_delta": round(span_delta, 6),
        "protocol_mean_logprob_delta": round(protocol_delta, 6),
        "protocol_stop_margin_delta": round(stop_delta, 6),
        "continue_margin_delta": round(continue_delta, 6),
        "span_protocol_side_effect": round(side_effect, 6),
        "span_protocol_fiber_score": round(score, 6),
        "strict_protocol_clean": strict_clean,
    }


def evaluate_state(model_obj: Any, tokenizer: Any, device: torch.device, source: dict[str, Any], specs: list[dict[str, Any]], patch_type: str) -> dict[str, Any]:
    prompt = str(source["prompt"])
    aliases = [str(x) for x in source.get("target_aliases") or [source.get("target", "")]]
    logits = p271.with_hooks_logits(model_obj, tokenizer, device, prompt, specs)
    readout = p271.score_logits(tokenizer, logits, aliases)
    forms = protocol_forms(source)
    span_scores: list[dict[str, Any]] = []
    for form in forms:
        lp = completion_logprob(model_obj, tokenizer, device, prompt, form["completion"], specs)
        span_scores.append({**form, **lp})
    alias_scores = [r for r in span_scores if r["form_type"].startswith("alias")]
    protocol_scores = [r for r in span_scores if r["form_type"].startswith("protocol")]
    if not protocol_scores:
        protocol_scores = alias_scores
    best_alias = max(alias_scores or span_scores, key=lambda r: safe_float(r.get("mean_logprob"), -1e30))
    best_protocol = max(protocol_scores or span_scores, key=lambda r: safe_float(r.get("mean_logprob"), -1e30))
    next_logits = next_logits_after(model_obj, tokenizer, device, prompt, best_protocol["completion"], specs)
    stop_scores = stop_gate_scores(tokenizer, next_logits)
    return {
        "patch_type": patch_type,
        "continue_stop_margin": round(safe_float(readout.get("continue_stop_margin")), 6),
        "tri_winner": readout.get("tri_winner"),
        "answer_boundary_margin": round(safe_float(readout.get("answer_boundary_margin")), 6),
        "answer_class_rank": int(readout.get("answer_class_rank", 0)),
        "best_alias_form": best_alias.get("form_type"),
        "best_alias_completion": best_alias.get("completion"),
        "best_alias_mean_logprob": best_alias.get("mean_logprob"),
        "best_span_mean_logprob": max(safe_float(r.get("mean_logprob"), -1e30) for r in span_scores),
        "best_protocol_form": best_protocol.get("form_type"),
        "best_protocol_completion": best_protocol.get("completion"),
        "best_protocol_mean_logprob": best_protocol.get("mean_logprob"),
        "span_form_count": len(span_scores),
        **stop_scores,
    }


def row_base(case: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase272",
        "created_at": utc_now(),
        "model": case["model"],
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "mode_id": source["mode_id"],
        "variant_id": source["variant_id"],
        "path_schema_id": source["path_schema_id"],
        "target": source["target"],
        "output_protocol": source.get("output_protocol"),
        "expected_pattern": source.get("expected_pattern"),
        "boundary_type": source.get("boundary_type"),
        "strongest_mlp_layer_phase268": case.get("strongest_mlp_layer"),
    }


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_bank = {r["case_id"]: r for r in read_jsonl(ATLAS_ROOT / "mode_family_case_bank_v3.jsonl")}
    selected = p271.select_cases(args.model, int(args.cases_per_model))
    model_obj = None
    tokenizer = None
    span_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    fiber_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(p271.get_layers(model_obj))
        prompts: list[str] = []
        layers_needed: list[int] = []
        valid_cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for case in selected:
            source = case_bank.get(str(case["case_id"]))
            if not source:
                missing_rows.append({"schema_version": SCHEMA_VERSION, "phase_id": "Phase272", "missing_id": f"phase272:missing:{args.model}:{case['case_id']}", "model": args.model, "case_id": case["case_id"], "reason": "case not found"})
                continue
            valid_cases.append((case, source))
            prompts.append(str(source["prompt"]))
            layers_needed.extend(p271.make_window(int(case["strongest_mlp_layer"]), n_layers))
        means = p271.collect_mlp_means(model_obj, tokenizer, device, prompts, layers_needed)
        log(f"{args.model}: collected MLP means for {len(means)} layers")
        for idx, (case, source) in enumerate(valid_cases, start=1):
            center = int(case["strongest_mlp_layer"])
            window = p271.make_window(center, n_layers)
            base = row_base(case, source)
            try:
                base_state = evaluate_state(model_obj, tokenizer, device, source, [], "base")
                all_states = [("base", [], base_state)]
                for patch_type in PATCH_TYPES:
                    specs = p271.patch_specs(patch_type, center, n_layers, args.model, str(case["case_id"]), means)
                    all_states.append((patch_type, specs, evaluate_state(model_obj, tokenizer, device, source, specs, patch_type)))
                for patch_type, specs, state in all_states:
                    state_row = {
                        **base,
                        "span_alias_id": f"phase272:span:{args.model}:{case['case_id']}:L{center}:{patch_type}",
                        "patch_type": patch_type,
                        "center_layer": center,
                        "window_layers": window,
                        "patched_component_count": len(specs),
                        **state,
                    }
                    span_rows.append(state_row)
                    protocol_rows.append(
                        {
                            **base,
                            "protocol_gate_id": state_row["span_alias_id"].replace(":span:", ":protocol:"),
                            "patch_type": patch_type,
                            "center_layer": center,
                            "window_layers": window,
                            "best_protocol_form": state["best_protocol_form"],
                            "best_protocol_mean_logprob": state["best_protocol_mean_logprob"],
                            "protocol_stop_margin": state["protocol_stop_margin"],
                            "period_logit": state["period_logit"],
                            "eos_logit": state["eos_logit"],
                            "continue_after_answer_logit": state["continue_after_answer_logit"],
                        }
                    )
                    if patch_type != "base":
                        quality = closure_quality(base_state, state, patch_type)
                        fiber = {
                            **base,
                            "span_protocol_fiber_id": state_row["span_alias_id"].replace(":span:", ":fiber:"),
                            "patch_type": patch_type,
                            "center_layer": center,
                            "window_layers": window,
                            **quality,
                        }
                        fiber_rows.append(fiber)
                        observations.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": "Phase272",
                                "created_at": utc_now(),
                                "observation_id": fiber["span_protocol_fiber_id"].replace(":fiber:", ":obs:"),
                                "case_id": case["case_id"],
                                "model": args.model,
                                "family_id": case["family_id"],
                                "level": "span_alias_protocol_closure_fiber",
                                "component": f"{patch_type}:L{center}:window{window}",
                                "metric_name": "span_protocol_fiber_score",
                                "metric_value": fiber["span_protocol_fiber_score"],
                                "metric_unit": "fiber_score",
                                "strict_protocol_clean": fiber["strict_protocol_clean"],
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase272",
                        "created_at": utc_now(),
                        "missing_id": f"phase272:missing:{args.model}:{case['case_id']}",
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case.get("family_id"),
                        "reason": repr(exc),
                    }
                )
            log(f"{args.model}: span/protocol audited {idx}/{len(valid_cases)} cases")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = make_metrics(args.model, span_rows, protocol_rows, fiber_rows)
    edges = make_edges(args.model, fiber_rows)
    payload = summarize_model(args.model, selected, span_rows, protocol_rows, fiber_rows, observations, metrics, edges, missing_rows)
    write_model_outputs(out_dir, args.model, payload, span_rows, protocol_rows, fiber_rows, observations, metrics, edges, missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_metrics(model: str, span: list[dict[str, Any]], protocol: list[dict[str, Any]], fiber: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_patch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fiber:
        by_patch[str(row["patch_type"])].append(row)
    for patch_type, vals in sorted(by_patch.items()):
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase272",
                "created_at": utc_now(),
                "metric_id": f"phase272:{model}:{patch_type}:span_protocol_quality",
                "scope": "span_alias_protocol_closure_fiber",
                "model": model,
                "patch_type": patch_type,
                "metric_name": "mean_span_protocol_fiber_score",
                "metric_value": mean_safe([safe_float(r["span_protocol_fiber_score"]) for r in vals]),
                "mean_continue_margin_delta": mean_safe([safe_float(r["continue_margin_delta"]) for r in vals]),
                "mean_protocol_stop_margin_delta": mean_safe([safe_float(r["protocol_stop_margin_delta"]) for r in vals]),
                "strict_protocol_clean_rate": round(sum(1 for r in vals if r.get("strict_protocol_clean")) / len(vals), 6) if vals else 0.0,
                "rows": len(vals),
            }
        )
    out.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase272",
            "created_at": utc_now(),
            "metric_id": f"phase272:{model}:strict_protocol_clean_rate",
            "scope": "span_alias_protocol_closure_fiber",
            "model": model,
            "metric_name": "strict_protocol_clean_rate",
            "metric_value": round(sum(1 for r in fiber if r.get("strict_protocol_clean")) / len(fiber), 6) if fiber else 0.0,
            "rows": len(fiber),
        }
    )
    return out


def make_edges(model: str, fiber: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    grouped = Counter((r["family_id"], r["patch_type"], bool(r["strict_protocol_clean"])) for r in fiber)
    for (family, patch_type, clean), count in grouped.items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase272",
                "created_at": utc_now(),
                "edge_id": f"phase272:{model}:{family}:{patch_type}:strict_protocol_clean:{clean}",
                "source": f"node:{family}",
                "target": f"node:span_protocol_fiber:{patch_type}",
                "edge_type": "span_alias_protocol_quality_control",
                "model": model,
                "strict_protocol_clean": clean,
                "effect_size": count,
                "status": "quality_control_not_closure",
            }
        )
    return edges


def summarize_model(model: str, selected: list[dict[str, Any]], span: list[dict[str, Any]], protocol: list[dict[str, Any]], fiber: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Span alias protocol closure-fiber atlas expansion",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "selected_cases": len(selected),
        "span_alias_rows": len(span),
        "protocol_gate_rows": len(protocol),
        "span_protocol_fiber_rows": len(fiber),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in fiber)),
        "strict_protocol_clean_counts": dict(Counter(str(r["strict_protocol_clean"]) for r in fiber)),
        "mean_span_protocol_fiber_score": mean_safe([safe_float(r["span_protocol_fiber_score"]) for r in fiber]),
        "mean_continue_margin_delta": mean_safe([safe_float(r["continue_margin_delta"]) for r in fiber]),
        "mean_protocol_stop_margin_delta": mean_safe([safe_float(r["protocol_stop_margin_delta"]) for r in fiber]),
    }


def write_model_outputs(out_dir: Path, model: str, summary: dict[str, Any], span: list[dict[str, Any]], protocol: list[dict[str, Any]], fiber: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"phase272_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase272_{model}_span_alias_rows.jsonl", span)
    write_jsonl(out_dir / f"phase272_{model}_protocol_gate_rows.jsonl", protocol)
    write_jsonl(out_dir / f"phase272_{model}_span_protocol_fiber_rows.jsonl", fiber)
    write_jsonl(out_dir / f"phase272_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase272_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase272_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase272_{model}_missing_rows.jsonl", missing)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase272_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    span: list[dict[str, Any]] = []
    protocol: list[dict[str, Any]] = []
    fiber: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        span.extend(read_jsonl(out_dir / f"phase272_{model}_span_alias_rows.jsonl"))
        protocol.extend(read_jsonl(out_dir / f"phase272_{model}_protocol_gate_rows.jsonl"))
        fiber.extend(read_jsonl(out_dir / f"phase272_{model}_span_protocol_fiber_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase272_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase272_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase272_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase272_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.92,
        "physical_path_atlas": 0.44,
        "multi_family_case_bank": 0.46,
        "multi_family_baseline_scan": 0.18,
        "state_factor_atlas": 0.40,
        "path_cluster_mining": 0.19,
        "trace_signature_validation": 0.57,
        "readout_competition_trace": 0.82,
        "component_path_atlas": 0.28,
        "closure_fiber_quality_control": 0.25,
        "span_alias_protocol_gate": 0.20,
        "stepwise_rollout_trace": 0.46,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.70,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Span alias protocol closure-fiber atlas expansion",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "span_alias_rows": len(span),
        "protocol_gate_rows": len(protocol),
        "span_protocol_fiber_rows": len(fiber),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in fiber)),
        "strict_protocol_clean_counts": dict(Counter(str(r["strict_protocol_clean"]) for r in fiber)),
        "mean_span_protocol_fiber_score": mean_safe([safe_float(r["span_protocol_fiber_score"]) for r in fiber]),
        "mean_continue_margin_delta": mean_safe([safe_float(r["continue_margin_delta"]) for r in fiber]),
        "mean_protocol_stop_margin_delta": mean_safe([safe_float(r["protocol_stop_margin_delta"]) for r in fiber]),
        "progress": progress,
    }
    write_json(out_dir / "phase272_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase272_span_alias_rows.jsonl", span)
    write_jsonl(out_dir / "phase272_protocol_gate_rows.jsonl", protocol)
    write_jsonl(out_dir / "phase272_span_protocol_fiber_rows.jsonl", fiber)
    write_jsonl(out_dir / "phase272_observations.jsonl", observations)
    write_jsonl(out_dir / "phase272_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase272_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase272_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_jsonl(ATLAS_ROOT / "phase272_span_alias_rows.jsonl", span)
    write_jsonl(ATLAS_ROOT / "phase272_protocol_gate_rows.jsonl", protocol)
    write_jsonl(ATLAS_ROOT / "phase272_span_protocol_fiber_rows.jsonl", fiber)
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase272", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase272 Span Alias Protocol Closure-Fiber Atlas",
        "",
        f"- status: {payload['status']}",
        f"- span_alias_rows: {payload['span_alias_rows']}",
        f"- protocol_gate_rows: {payload['protocol_gate_rows']}",
        f"- span_protocol_fiber_rows: {payload['span_protocol_fiber_rows']}",
        f"- strict_protocol_clean_counts: {json.dumps(payload['strict_protocol_clean_counts'], ensure_ascii=False)}",
        f"- mean_span_protocol_fiber_score: {payload['mean_span_protocol_fiber_score']}",
        f"- mean_continue_margin_delta: {payload['mean_continue_margin_delta']}",
        f"- mean_protocol_stop_margin_delta: {payload['mean_protocol_stop_margin_delta']}",
        "",
        "Note: This expands closure-fiber quality control to alias/span/protocol/stop-gate fields. It is not closure.",
    ]
    (out_dir / "phase272_span_alias_protocol_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--cases-per-model", type=int, default=6)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if args.model:
        evaluate_model(args)
        return
    for model in MODELS:
        args.model = model
        evaluate_model(args)
    summarize_round(args.round_name)


if __name__ == "__main__":
    main()
