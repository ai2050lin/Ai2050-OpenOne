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
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase262_continuation_regime_decomposition_atlas as p262  # noqa: E402


PHASE = 267
SOURCE_PHASE = 266
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE266_ROOT = Path("tests/result/phase266_multi_family_baseline_behavior_readout_scan/multi_family_baseline_behavior_readout_scan")
RESULT_ROOT = Path("tests/result/phase267_multifamily_continuation_physical_path_trace")
ROUND_DEFAULT = "multifamily_continuation_physical_path_trace"
PRIMARY_CHANNELS = {
    "continue_list_item",
    "continue_the",
    "continue_next_sentence",
    "continue_json_structure",
    "continue_format",
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


def append_unique_jsonl(path: Path, rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    if token_id < 0 or token_id >= logits.numel():
        return -1
    return int((logits > logits[token_id]).sum().item()) + 1


def get_final_norm(model_obj: Any) -> Any | None:
    candidates = [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("transformer", "encoder", "final_layernorm"),
        ("transformer", "output_layernorm"),
    ]
    for path in candidates:
        obj = model_obj
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok:
            return obj
    return None


def project_hidden(model_obj: Any, final_norm: Any | None, hidden: torch.Tensor, is_final: bool) -> torch.Tensor:
    h = hidden.detach().clone()
    if final_norm is not None and not is_final:
        h = final_norm(h)
    lm_head = model_obj.get_output_embeddings()
    with torch.no_grad():
        return lm_head(h).detach().float().cpu()


def group_scores(logits: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, tuple[float, int]]:
    scores: dict[str, tuple[float, int]] = {}
    for name, ids in groups.items():
        valid = [int(x) for x in ids if 0 <= int(x) < logits.numel()]
        if not valid:
            scores[name] = (-1e30, -1)
            continue
        idx = torch.tensor(valid, dtype=torch.long)
        vals = logits[idx]
        pos = int(torch.argmax(vals).item())
        scores[name] = (float(vals[pos].item()), int(idx[pos].item()))
    return scores


def layer_readout(tokenizer: Any, logits: torch.Tensor, stop_ids: dict[str, list[int]], cont_ids: dict[str, list[int]], aliases: list[str]) -> dict[str, Any]:
    stop_scores = group_scores(logits, stop_ids)
    cont_scores = group_scores(logits, cont_ids)
    best_stop_name, (best_stop, best_stop_id) = max(stop_scores.items(), key=lambda kv: kv[1][0])
    best_cont_name, (best_cont, best_cont_id) = max(cont_scores.items(), key=lambda kv: kv[1][0])
    target = p239.readout_metrics(tokenizer, logits, aliases)
    top_id = int(torch.argmax(logits).item())
    if target.get("target_logit") is None:
        target_logit = -1e30
        target_rank = -1
        target_id = -1
    else:
        target_logit = safe_float(target.get("target_logit"), -1e30)
        target_rank = int(target.get("target_rank") or -1)
        target_id = int(target.get("target_token_id") or -1)
    winner = "continue" if best_cont >= max(best_stop, target_logit) else ("stop" if best_stop >= target_logit else "target")
    return {
        "r_stop": round(best_stop, 6),
        "r_stop_name": best_stop_name,
        "r_stop_token_id": best_stop_id,
        "r_continue": round(best_cont, 6),
        "r_continue_name": best_cont_name,
        "r_continue_token_id": best_cont_id,
        "top_continue_channel": best_cont_name,
        "stop_continue_margin": round(best_stop - best_cont, 6),
        "continue_stop_margin": round(best_cont - best_stop, 6),
        "target_logit": round(target_logit, 6),
        "target_rank": target_rank,
        "target_token_id": target_id,
        "target_margin_vs_continue": round(target_logit - best_cont, 6),
        "competition_winner": winner,
        "top_token_id": top_id,
        "top_token_rank_continue": rank_of(logits, best_cont_id),
        "top_token_rank_stop": rank_of(logits, best_stop_id),
    }


def select_trace_cases(model: str, max_cases_per_family: int) -> list[dict[str, Any]]:
    behavior = {r["case_id"]: r for r in read_jsonl(PHASE266_ROOT / f"phase266_{model}_behavior_rows.jsonl")}
    readout = {r["case_id"]: r for r in read_jsonl(PHASE266_ROOT / f"phase266_{model}_readout_rows.jsonl")}
    quality = {r["case_id"]: r for r in read_jsonl(PHASE266_ROOT / f"phase266_{model}_quality_calibration_rows.jsonl")}
    cases = {r["case_id"]: r for r in read_jsonl(ATLAS_ROOT / "mode_family_case_bank_v3.jsonl")}
    merged: list[dict[str, Any]] = []
    for case_id, q in quality.items():
        if case_id not in cases or case_id not in readout or case_id not in behavior:
            continue
        row = {**cases[case_id], **behavior[case_id], **readout[case_id], **q}
        top_channel = str(row.get("top_continue_channel"))
        score = 0.0
        if str(row.get("scoring_risk_calibrated")) == "high":
            score += 1000
        elif str(row.get("scoring_risk_calibrated")) == "medium":
            score += 500
        if row.get("answer_correct_proxy") and not row.get("pattern_matched_proxy"):
            score += 220
        if top_channel in PRIMARY_CHANNELS:
            score += 120
        score += abs(safe_float(row.get("stop_continue_margin"))) * 10
        score += max(0.0, -safe_float(row.get("target_margin_vs_winner"))) * 2
        row["trace_selection_score"] = round(score, 6)
        merged.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        grouped[str(row["family_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda r: (-safe_float(r["trace_selection_score"]), str(r["case_id"])))
        selected.extend(ranked[:max_cases_per_family])
    return selected


def capture_case_trace(model_obj: Any, tokenizer: Any, device: torch.device, case: dict[str, Any], stop_ids: dict[str, list[int]], cont_ids: dict[str, list[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompt = str(case["prompt"])
    aliases = [str(x) for x in case.get("target_aliases") or [case.get("target", "")]]
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    final_norm = get_final_norm(model_obj)
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden_states = out.hidden_states
    final_logits = out.logits[0, last_pos].detach().float().cpu()
    layer_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    prev_margin: float | None = None
    first_continue_layer = None
    stable_from_layer = None
    positive_streak = 0
    peak_layer = -1
    peak_margin = -1e30
    for layer_idx, h in enumerate(hidden_states):
        hidden = h[0, last_pos]
        logits = final_logits if layer_idx == len(hidden_states) - 1 else project_hidden(model_obj, final_norm, hidden, False)
        read = layer_readout(tokenizer, logits, stop_ids, cont_ids, aliases)
        margin = safe_float(read["continue_stop_margin"])
        if margin > peak_margin:
            peak_margin = margin
            peak_layer = layer_idx
        if margin > 0 and first_continue_layer is None:
            first_continue_layer = layer_idx
        positive_streak = positive_streak + 1 if margin > 0 else 0
        if positive_streak >= 3 and stable_from_layer is None:
            stable_from_layer = layer_idx - 2
        row = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase267",
            "created_at": utc_now(),
            "layer_readout_id": f"phase267:layer:{case['model']}:{case['case_id']}:L{layer_idx}",
            "model": case["model"],
            "case_id": case["case_id"],
            "family_id": case["family_id"],
            "mode_id": case["mode_id"],
            "variant_id": case["variant_id"],
            "path_schema_id": case["path_schema_id"],
            "layer_index": layer_idx,
            "layer_kind": "embedding" if layer_idx == 0 else ("final" if layer_idx == len(hidden_states) - 1 else "residual_after_layer"),
            "selected_top_continue_channel_phase266": case.get("top_continue_channel"),
            **read,
        }
        layer_rows.append(row)
        if prev_margin is not None:
            component_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase267",
                    "created_at": utc_now(),
                    "component_contribution_id": f"phase267:component:{case['model']}:{case['case_id']}:L{layer_idx}:residual_delta",
                    "model": case["model"],
                    "case_id": case["case_id"],
                    "family_id": case["family_id"],
                    "mode_id": case["mode_id"],
                    "variant_id": case["variant_id"],
                    "layer_index": layer_idx,
                    "component_type": "residual_layer_delta",
                    "delta_continue_stop_margin": round(margin - prev_margin, 6),
                    "continue_stop_margin": round(margin, 6),
                    "note": "Layer-level residual delta; not separated into attention and MLP yet.",
                }
            )
        prev_margin = margin
    signature = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase267",
        "created_at": utc_now(),
        "physical_path_id": f"phase267:path:{case['model']}:{case['case_id']}",
        "model": case["model"],
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "mode_id": case["mode_id"],
        "variant_id": case["variant_id"],
        "path_schema_id": case["path_schema_id"],
        "target": case.get("target"),
        "scoring_risk_calibrated": case.get("scoring_risk_calibrated"),
        "answer_correct_proxy_phase266": case.get("answer_correct_proxy"),
        "pattern_matched_proxy_phase266": case.get("pattern_matched_proxy"),
        "top_continue_channel_phase266": case.get("top_continue_channel"),
        "first_continue_win_layer": first_continue_layer,
        "stable_continue_from_layer": stable_from_layer,
        "peak_continue_margin_layer": peak_layer,
        "peak_continue_stop_margin": round(peak_margin, 6),
        "final_continue_stop_margin": layer_rows[-1]["continue_stop_margin"] if layer_rows else 0.0,
        "final_competition_winner": layer_rows[-1]["competition_winner"] if layer_rows else "missing",
        "num_layers_observed": len(layer_rows),
    }
    return layer_rows, component_rows, signature


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = select_trace_cases(args.model, int(args.cases_per_family))
    for row in cases:
        row["model"] = args.model
    model_obj = None
    tokenizer = None
    layer_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
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
        stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
        cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
        for idx, case in enumerate(cases, start=1):
            try:
                lr, cr, sig = capture_case_trace(model_obj, tokenizer, device, case, stop_ids, cont_ids)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase267",
                        "created_at": utc_now(),
                        "missing_id": f"phase267:missing:{args.model}:{case['case_id']}",
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case["family_id"],
                        "reason": repr(exc),
                    }
                )
                continue
            layer_rows.extend(lr)
            component_rows.extend(cr)
            path_rows.append(sig)
            signature_rows.append({**sig, "family_path_signature_id": sig["physical_path_id"].replace(":path:", ":signature:")})
            channel_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase267",
                    "created_at": utc_now(),
                    "channel_trace_id": f"phase267:channel:{args.model}:{case['case_id']}:{sig['top_continue_channel_phase266']}",
                    "model": args.model,
                    "case_id": case["case_id"],
                    "family_id": case["family_id"],
                    "mode_id": case["mode_id"],
                    "variant_id": case["variant_id"],
                    "channel": sig["top_continue_channel_phase266"],
                    "first_continue_win_layer": sig["first_continue_win_layer"],
                    "stable_continue_from_layer": sig["stable_continue_from_layer"],
                    "peak_continue_margin_layer": sig["peak_continue_margin_layer"],
                    "final_continue_stop_margin": sig["final_continue_stop_margin"],
                }
            )
            observations.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase267",
                    "created_at": utc_now(),
                    "observation_id": f"phase267:obs:{args.model}:{case['case_id']}",
                    "case_id": case["case_id"],
                    "model": args.model,
                    "family_id": case["family_id"],
                    "level": "layerwise_physical_path_trace",
                    "component": sig["top_continue_channel_phase266"],
                    "metric_name": "stable_continue_from_layer",
                    "metric_value": -1 if sig["stable_continue_from_layer"] is None else sig["stable_continue_from_layer"],
                    "metric_unit": "layer_index",
                    "winner": sig["final_competition_winner"],
                }
            )
            if idx % 9 == 0:
                log(f"{args.model}: traced {idx}/{len(cases)} selected cases")
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
    metrics = make_metrics(args.model, path_rows, layer_rows)
    edges = make_edges(args.model, path_rows)
    summary = summarize_model(args.model, cases, path_rows, layer_rows, component_rows, channel_rows, signature_rows, metrics, edges, missing_rows)
    write_model_outputs(out_dir, args.model, summary, path_rows, layer_rows, component_rows, channel_rows, signature_rows, observations, metrics, edges, missing_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def make_metrics(model: str, path_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        by_family[str(row["family_id"])].append(row)
    for family, items in sorted(by_family.items()):
        stable = [safe_float(r["stable_continue_from_layer"], -1) for r in items if r.get("stable_continue_from_layer") is not None]
        final_margin = [safe_float(r["final_continue_stop_margin"]) for r in items]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase267",
                "created_at": utc_now(),
                "metric_id": f"phase267:{model}:{family}:mean_stable_continue_layer",
                "scope": "physical_path_trace",
                "model": model,
                "family_id": family,
                "metric_name": "mean_stable_continue_layer",
                "metric_value": mean_safe(stable),
                "mean_final_continue_stop_margin": mean_safe(final_margin),
                "rows": len(items),
            }
        )
    winners = Counter(str(r.get("competition_winner")) for r in layer_rows if r.get("layer_kind") == "final")
    for winner, count in winners.items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase267",
                "created_at": utc_now(),
                "metric_id": f"phase267:{model}:final_winner:{winner}",
                "scope": "physical_path_trace",
                "model": model,
                "metric_name": "final_layer_winner_count",
                "winner": winner,
                "metric_value": count,
                "rows": len(path_rows),
            }
        )
    return rows


def make_edges(model: str, path_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    grouped = Counter((r["family_id"], str(r.get("top_continue_channel_phase266")), r.get("stable_continue_from_layer")) for r in path_rows)
    for (family, channel, stable_layer), count in grouped.items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase267",
                "created_at": utc_now(),
                "edge_id": f"phase267:{model}:{family}:{channel}:L{stable_layer}",
                "source": f"node:{family}",
                "target": f"node:{channel}:stable_from_L{stable_layer}",
                "edge_type": "family_to_continuation_physical_path",
                "model": model,
                "evidence_type": "layerwise_residual_readout_trace",
                "effect_size": count,
                "status": "residual_path_trace_not_component_closure",
            }
        )
    return edges


def summarize_model(model: str, cases: list[dict[str, Any]], path_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]], signature_rows: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(r["family_id"]) for r in path_rows)
    channel_counts = Counter(str(r.get("top_continue_channel_phase266")) for r in path_rows)
    stable_layers = [int(r["stable_continue_from_layer"]) for r in path_rows if r.get("stable_continue_from_layer") is not None]
    first_layers = [int(r["first_continue_win_layer"]) for r in path_rows if r.get("first_continue_win_layer") is not None]
    final_winners = Counter(str(r.get("final_competition_winner")) for r in path_rows)
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Multi-family continuation channel physical path trace",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "selected_cases": len(cases),
        "physical_path_rows": len(path_rows),
        "layerwise_readout_rows": len(layer_rows),
        "component_contribution_rows": len(component_rows),
        "continue_channel_trace_rows": len(channel_rows),
        "family_path_signature_rows": len(signature_rows),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "family_counts": dict(family_counts),
        "channel_counts": dict(channel_counts),
        "final_winner_counts": dict(final_winners),
        "mean_first_continue_win_layer": mean_safe(first_layers),
        "mean_stable_continue_from_layer": mean_safe(stable_layers),
        "mean_final_continue_stop_margin": mean_safe([safe_float(r.get("final_continue_stop_margin")) for r in path_rows]),
    }


def write_model_outputs(out_dir: Path, model: str, summary: dict[str, Any], path_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]], signature_rows: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"phase267_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase267_{model}_physical_path_rows.jsonl", path_rows)
    write_jsonl(out_dir / f"phase267_{model}_layerwise_readout_rows.jsonl", layer_rows)
    write_jsonl(out_dir / f"phase267_{model}_component_contribution_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase267_{model}_continue_channel_trace_rows.jsonl", channel_rows)
    write_jsonl(out_dir / f"phase267_{model}_family_path_signature_rows.jsonl", signature_rows)
    write_jsonl(out_dir / f"phase267_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase267_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase267_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase267_{model}_missing_rows.jsonl", missing)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase267_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    path_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        path_rows.extend(read_jsonl(out_dir / f"phase267_{model}_physical_path_rows.jsonl"))
        layer_rows.extend(read_jsonl(out_dir / f"phase267_{model}_layerwise_readout_rows.jsonl"))
        component_rows.extend(read_jsonl(out_dir / f"phase267_{model}_component_contribution_rows.jsonl"))
        channel_rows.extend(read_jsonl(out_dir / f"phase267_{model}_continue_channel_trace_rows.jsonl"))
        signature_rows.extend(read_jsonl(out_dir / f"phase267_{model}_family_path_signature_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase267_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase267_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase267_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase267_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.89,
        "physical_path_atlas": 0.34,
        "multi_family_case_bank": 0.45,
        "multi_family_baseline_scan": 0.18,
        "state_factor_atlas": 0.38,
        "path_cluster_mining": 0.16,
        "trace_signature_validation": 0.50,
        "readout_competition_trace": 0.80,
        "stepwise_rollout_trace": 0.44,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.68,
    }
    stable_layers = [int(r["stable_continue_from_layer"]) for r in path_rows if r.get("stable_continue_from_layer") is not None]
    first_layers = [int(r["first_continue_win_layer"]) for r in path_rows if r.get("first_continue_win_layer") is not None]
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Multi-family continuation channel physical path trace",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "physical_path_rows": len(path_rows),
        "layerwise_readout_rows": len(layer_rows),
        "component_contribution_rows": len(component_rows),
        "continue_channel_trace_rows": len(channel_rows),
        "family_path_signature_rows": len(signature_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "family_counts": dict(Counter(str(r["family_id"]) for r in path_rows)),
        "channel_counts": dict(Counter(str(r.get("top_continue_channel_phase266")) for r in path_rows).most_common()),
        "final_winner_counts": dict(Counter(str(r.get("final_competition_winner")) for r in path_rows)),
        "mean_first_continue_win_layer": mean_safe(first_layers),
        "mean_stable_continue_from_layer": mean_safe(stable_layers),
        "mean_final_continue_stop_margin": mean_safe([safe_float(r.get("final_continue_stop_margin")) for r in path_rows]),
        "progress": progress,
    }
    write_json(out_dir / "phase267_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase267_physical_path_rows.jsonl", path_rows)
    write_jsonl(out_dir / "phase267_layerwise_readout_rows.jsonl", layer_rows)
    write_jsonl(out_dir / "phase267_component_contribution_rows.jsonl", component_rows)
    write_jsonl(out_dir / "phase267_continue_channel_trace_rows.jsonl", channel_rows)
    write_jsonl(out_dir / "phase267_family_path_signature_rows.jsonl", signature_rows)
    write_jsonl(out_dir / "phase267_observations.jsonl", observations)
    write_jsonl(out_dir / "phase267_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase267_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase267_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_jsonl(ATLAS_ROOT / "phase267_physical_path_rows.jsonl", path_rows)
    write_jsonl(ATLAS_ROOT / "phase267_layerwise_readout_rows.jsonl", layer_rows)
    write_jsonl(ATLAS_ROOT / "phase267_component_contribution_rows.jsonl", component_rows)
    write_jsonl(ATLAS_ROOT / "phase267_continue_channel_trace_rows.jsonl", channel_rows)
    write_jsonl(ATLAS_ROOT / "phase267_family_path_signature_rows.jsonl", signature_rows)
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase267", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase267 Multi-Family Continuation Channel Physical Path Trace",
        "",
        f"- status: {payload['status']}",
        f"- physical_path_rows: {payload['physical_path_rows']}",
        f"- layerwise_readout_rows: {payload['layerwise_readout_rows']}",
        f"- component_contribution_rows: {payload['component_contribution_rows']}",
        f"- continue_channel_trace_rows: {payload['continue_channel_trace_rows']}",
        f"- family_path_signature_rows: {payload['family_path_signature_rows']}",
        f"- family_counts: {json.dumps(payload['family_counts'], ensure_ascii=False)}",
        f"- channel_counts: {json.dumps(payload['channel_counts'], ensure_ascii=False)}",
        f"- final_winner_counts: {json.dumps(payload['final_winner_counts'], ensure_ascii=False)}",
        f"- mean_first_continue_win_layer: {payload['mean_first_continue_win_layer']}",
        f"- mean_stable_continue_from_layer: {payload['mean_stable_continue_from_layer']}",
        f"- mean_final_continue_stop_margin: {payload['mean_final_continue_stop_margin']}",
        "",
        "Note: component rows are residual layer deltas, not attention/MLP-separated causal attribution.",
    ]
    (out_dir / "phase267_multifamily_continue_path_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--cases-per-family", type=int, default=3)
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
