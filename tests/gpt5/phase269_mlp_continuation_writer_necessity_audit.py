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
from model_utils import get_layers  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_mlp  # noqa: E402


PHASE = 269
SOURCE_PHASE = 268
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE268_ROOT = Path("tests/result/phase268_attention_mlp_continuation_path_attribution/attention_mlp_continuation_path_attribution")
RESULT_ROOT = Path("tests/result/phase269_mlp_continuation_writer_necessity_audit")
ROUND_DEFAULT = "mlp_continuation_writer_necessity_audit"
PATCHES = [
    {"patch_type": "mlp_zero_last_token", "scale": 0.0},
    {"patch_type": "mlp_half_last_token", "scale": 0.5},
]


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


def replace_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple) and output:
        return (tensor, *output[1:])
    return output


def select_cases(model: str, cases_per_model: int) -> list[dict[str, Any]]:
    rows = [r for r in read_jsonl(PHASE268_ROOT / "phase268_component_summary_rows.jsonl") if r.get("model") == model]
    rows.sort(
        key=lambda r: (
            -safe_float(r.get("sum_positive_mlp_delta")),
            -safe_float(r.get("final_continue_stop_margin")),
            str(r.get("family_id")),
            str(r.get("case_id")),
        )
    )
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    for row in rows:
        if len(selected) >= cases_per_model:
            break
        if str(row.get("family_id")) in used_families and len(rows) >= cases_per_model:
            continue
        selected.append(row)
        used_families.add(str(row.get("family_id")))
    for row in rows:
        if len(selected) >= cases_per_model:
            break
        if row not in selected:
            selected.append(row)
    return selected[:cases_per_model]


def score_logits(tokenizer: Any, logits: torch.Tensor, aliases: list[str]) -> dict[str, Any]:
    stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
    cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
    scores = {**p262.score_channels(logits.detach().float().cpu(), stop_ids, cont_ids), **p239.readout_metrics(tokenizer, logits.detach().float().cpu(), aliases)}
    target = safe_float(scores.get("target_logit"), -1e30)
    winner_value = max(safe_float(scores.get("r_stop"), -1e30), safe_float(scores.get("r_continue"), -1e30), target)
    if winner_value == target:
        winner = "target"
    elif winner_value == safe_float(scores.get("r_continue"), -1e30):
        winner = "continue"
    else:
        winner = "stop"
    scores["tri_winner"] = winner
    scores["continue_stop_margin"] = safe_float(scores.get("r_continue")) - safe_float(scores.get("r_stop"))
    return scores


def install_mlp_scale_hook(model_obj: Any, layer_idx: int, scale: float):
    layers = get_layers(model_obj)
    mlp = get_mlp(layers[int(layer_idx)])
    if mlp is None:
        raise ValueError(f"no mlp module at layer {layer_idx}")

    def hook(_module, _inputs, output):
        y = extract_tensor(output)
        if y is None:
            return output
        patched = y.clone()
        if patched.ndim == 3:
            patched[:, -1, :] = patched[:, -1, :] * float(scale)
        elif patched.ndim == 2:
            patched[-1, :] = patched[-1, :] * float(scale)
        elif patched.ndim == 1:
            patched = patched * float(scale)
        return replace_tensor(output, patched)

    return mlp.register_forward_hook(hook)


def forward_logits(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, layer_idx: int | None = None, scale: float | None = None) -> torch.Tensor:
    handle = None
    if layer_idx is not None and scale is not None:
        handle = install_mlp_scale_hook(model_obj, int(layer_idx), float(scale))
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            out = model_obj(**encoded, use_cache=False, return_dict=True)
        return out.logits[0, last_pos].detach().float().cpu()
    finally:
        if handle is not None:
            handle.remove()


def generate_text(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, max_new_tokens: int, layer_idx: int | None = None, scale: float | None = None) -> tuple[str, int]:
    handle = None
    if layer_idx is not None and scale is not None:
        handle = install_mlp_scale_hook(model_obj, int(layer_idx), float(scale))
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    input_len = int(encoded["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            out = model_obj.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0, input_len:].detach().cpu().tolist()
        return tokenizer.decode(new_ids, skip_special_tokens=False), len(new_ids)
    finally:
        if handle is not None:
            handle.remove()


def row_base(case: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase269",
        "created_at": utc_now(),
        "model": case["model"],
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "mode_id": source["mode_id"],
        "variant_id": source["variant_id"],
        "path_schema_id": source["path_schema_id"],
        "target": source["target"],
        "top_continue_channel_phase268": case.get("top_continue_channel_phase267"),
        "strongest_mlp_layer_phase268": case.get("strongest_mlp_layer"),
        "strongest_mlp_delta_phase268": case.get("strongest_mlp_delta"),
    }


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_bank = {r["case_id"]: r for r in read_jsonl(ATLAS_ROOT / "mode_family_case_bank_v3.jsonl")}
    selected = select_cases(args.model, int(args.cases_per_model))
    model_obj = None
    tokenizer = None
    necessity_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
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
        for idx, case in enumerate(selected, start=1):
            source = case_bank.get(str(case["case_id"]))
            if not source:
                missing_rows.append({"schema_version": SCHEMA_VERSION, "phase_id": "Phase269", "missing_id": f"phase269:missing:{args.model}:{case['case_id']}", "model": args.model, "case_id": case["case_id"], "reason": "case not found"})
                continue
            base = row_base(case, source)
            aliases = [str(x) for x in source.get("target_aliases") or [source.get("target", "")]]
            prompt = str(source["prompt"])
            layer_idx = int(case["strongest_mlp_layer"])
            try:
                base_logits = forward_logits(model_obj, tokenizer, device, prompt)
                base_scores = score_logits(tokenizer, base_logits, aliases)
                base_text, base_new_tokens = generate_text(model_obj, tokenizer, device, prompt, int(args.rollout_tokens))
                for patch in PATCHES:
                    patched_logits = forward_logits(model_obj, tokenizer, device, prompt, layer_idx, float(patch["scale"]))
                    patched_scores = score_logits(tokenizer, patched_logits, aliases)
                    patched_text, patched_new_tokens = generate_text(model_obj, tokenizer, device, prompt, int(args.rollout_tokens), layer_idx, float(patch["scale"]))
                    delta_continue_stop = safe_float(patched_scores.get("continue_stop_margin")) - safe_float(base_scores.get("continue_stop_margin"))
                    delta_target = safe_float(patched_scores.get("target_logit")) - safe_float(base_scores.get("target_logit"))
                    row = {
                        **base,
                        "mlp_necessity_id": f"phase269:necessity:{args.model}:{case['case_id']}:L{layer_idx}:{patch['patch_type']}",
                        "patch_type": patch["patch_type"],
                        "patch_scale": patch["scale"],
                        "base_continue_stop_margin": round(safe_float(base_scores.get("continue_stop_margin")), 6),
                        "patched_continue_stop_margin": round(safe_float(patched_scores.get("continue_stop_margin")), 6),
                        "delta_continue_stop_margin": round(delta_continue_stop, 6),
                        "base_winner": base_scores.get("tri_winner"),
                        "patched_winner": patched_scores.get("tri_winner"),
                        "winner_changed": base_scores.get("tri_winner") != patched_scores.get("tri_winner"),
                        "base_target_logit": base_scores.get("target_logit"),
                        "patched_target_logit": patched_scores.get("target_logit"),
                        "delta_target_logit": round(delta_target, 6),
                        "necessity_supported": bool(delta_continue_stop < -1.0 or base_scores.get("tri_winner") != patched_scores.get("tri_winner")),
                    }
                    necessity_rows.append(row)
                    causal_rows.append(
                        {
                            **base,
                            "causal_effect_id": row["mlp_necessity_id"].replace(":necessity:", ":effect:"),
                            "patch_type": patch["patch_type"],
                            "patch_scale": patch["scale"],
                            "effect_metric": "continue_stop_margin",
                            "effect_value": row["delta_continue_stop_margin"],
                            "winner_changed": row["winner_changed"],
                            "necessity_supported": row["necessity_supported"],
                        }
                    )
                    rollout_rows.append(
                        {
                            **base,
                            "rollout_effect_id": row["mlp_necessity_id"].replace(":necessity:", ":rollout:"),
                            "patch_type": patch["patch_type"],
                            "patch_scale": patch["scale"],
                            "base_text": base_text[:300],
                            "patched_text": patched_text[:300],
                            "base_new_tokens": base_new_tokens,
                            "patched_new_tokens": patched_new_tokens,
                            "rollout_changed": base_text != patched_text,
                        }
                    )
                    observations.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase269",
                            "created_at": utc_now(),
                            "observation_id": row["mlp_necessity_id"].replace(":necessity:", ":obs:"),
                            "case_id": case["case_id"],
                            "model": args.model,
                            "family_id": case["family_id"],
                            "level": "mlp_necessity_audit",
                            "component": f"MLP_L{layer_idx}",
                            "metric_name": "delta_continue_stop_margin",
                            "metric_value": row["delta_continue_stop_margin"],
                            "metric_unit": "logit_margin",
                            "winner": row["patched_winner"],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase269",
                        "created_at": utc_now(),
                        "missing_id": f"phase269:missing:{args.model}:{case['case_id']}",
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case["family_id"],
                        "reason": repr(exc),
                    }
                )
            log(f"{args.model}: necessity audited {idx}/{len(selected)} cases")
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
    metrics = make_metrics(args.model, necessity_rows)
    edges = make_edges(args.model, necessity_rows)
    payload = summarize_model(args.model, selected, necessity_rows, causal_rows, rollout_rows, observations, metrics, edges, missing_rows)
    write_model_outputs(out_dir, args.model, payload, necessity_rows, causal_rows, rollout_rows, observations, metrics, edges, missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_metrics(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_patch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patch[str(row["patch_type"])].append(row)
    for patch_type, vals in sorted(by_patch.items()):
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase269",
                "created_at": utc_now(),
                "metric_id": f"phase269:{model}:{patch_type}:mean_effect",
                "scope": "mlp_necessity_audit",
                "model": model,
                "patch_type": patch_type,
                "metric_name": "mean_delta_continue_stop_margin",
                "metric_value": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in vals]),
                "necessity_supported_rate": round(sum(1 for r in vals if r.get("necessity_supported")) / len(vals), 6) if vals else 0.0,
                "winner_change_rate": round(sum(1 for r in vals if r.get("winner_changed")) / len(vals), 6) if vals else 0.0,
                "rows": len(vals),
            }
        )
    return out


def make_edges(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    grouped = Counter((r["family_id"], r["strongest_mlp_layer_phase268"], r["patch_type"], bool(r["necessity_supported"])) for r in rows)
    for (family, layer, patch_type, supported), count in grouped.items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase269",
                "created_at": utc_now(),
                "edge_id": f"phase269:{model}:{family}:L{layer}:{patch_type}:{supported}",
                "source": f"node:{family}",
                "target": f"node:MLP_L{layer}",
                "edge_type": "mlp_writer_necessity_candidate",
                "model": model,
                "patch_type": patch_type,
                "necessity_supported": supported,
                "effect_size": count,
                "status": "small_scale_causal_audit_not_closure",
            }
        )
    return edges


def summarize_model(model: str, selected: list[dict[str, Any]], necessity: list[dict[str, Any]], causal: list[dict[str, Any]], rollout: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "MLP continuation writer necessity audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "selected_cases": len(selected),
        "mlp_necessity_rows": len(necessity),
        "causal_effect_rows": len(causal),
        "rollout_effect_rows": len(rollout),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in necessity)),
        "necessity_supported_counts": dict(Counter(str(r["necessity_supported"]) for r in necessity)),
        "winner_changed_counts": dict(Counter(str(r["winner_changed"]) for r in necessity)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in necessity]),
        "mean_delta_target_logit": mean_safe([safe_float(r["delta_target_logit"]) for r in necessity]),
        "rollout_changed_counts": dict(Counter(str(r["rollout_changed"]) for r in rollout)),
    }


def write_model_outputs(out_dir: Path, model: str, summary: dict[str, Any], necessity: list[dict[str, Any]], causal: list[dict[str, Any]], rollout: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"phase269_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase269_{model}_mlp_necessity_rows.jsonl", necessity)
    write_jsonl(out_dir / f"phase269_{model}_causal_effect_rows.jsonl", causal)
    write_jsonl(out_dir / f"phase269_{model}_rollout_effect_rows.jsonl", rollout)
    write_jsonl(out_dir / f"phase269_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase269_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase269_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase269_{model}_missing_rows.jsonl", missing)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase269_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    necessity: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []
    rollout: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        necessity.extend(read_jsonl(out_dir / f"phase269_{model}_mlp_necessity_rows.jsonl"))
        causal.extend(read_jsonl(out_dir / f"phase269_{model}_causal_effect_rows.jsonl"))
        rollout.extend(read_jsonl(out_dir / f"phase269_{model}_rollout_effect_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase269_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase269_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase269_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase269_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.90,
        "physical_path_atlas": 0.39,
        "multi_family_case_bank": 0.45,
        "multi_family_baseline_scan": 0.18,
        "state_factor_atlas": 0.39,
        "path_cluster_mining": 0.18,
        "trace_signature_validation": 0.54,
        "readout_competition_trace": 0.80,
        "component_path_atlas": 0.20,
        "stepwise_rollout_trace": 0.45,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.69,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "MLP continuation writer necessity audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "mlp_necessity_rows": len(necessity),
        "causal_effect_rows": len(causal),
        "rollout_effect_rows": len(rollout),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in necessity)),
        "necessity_supported_counts": dict(Counter(str(r["necessity_supported"]) for r in necessity)),
        "winner_changed_counts": dict(Counter(str(r["winner_changed"]) for r in necessity)),
        "rollout_changed_counts": dict(Counter(str(r["rollout_changed"]) for r in rollout)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in necessity]),
        "mean_delta_target_logit": mean_safe([safe_float(r["delta_target_logit"]) for r in necessity]),
        "progress": progress,
    }
    write_json(out_dir / "phase269_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase269_mlp_necessity_rows.jsonl", necessity)
    write_jsonl(out_dir / "phase269_causal_effect_rows.jsonl", causal)
    write_jsonl(out_dir / "phase269_rollout_effect_rows.jsonl", rollout)
    write_jsonl(out_dir / "phase269_observations.jsonl", observations)
    write_jsonl(out_dir / "phase269_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase269_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase269_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_jsonl(ATLAS_ROOT / "phase269_mlp_necessity_rows.jsonl", necessity)
    write_jsonl(ATLAS_ROOT / "phase269_causal_effect_rows.jsonl", causal)
    write_jsonl(ATLAS_ROOT / "phase269_rollout_effect_rows.jsonl", rollout)
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase269", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase269 MLP Continuation Writer Necessity Audit",
        "",
        f"- status: {payload['status']}",
        f"- mlp_necessity_rows: {payload['mlp_necessity_rows']}",
        f"- causal_effect_rows: {payload['causal_effect_rows']}",
        f"- rollout_effect_rows: {payload['rollout_effect_rows']}",
        f"- patch_counts: {json.dumps(payload['patch_counts'], ensure_ascii=False)}",
        f"- necessity_supported_counts: {json.dumps(payload['necessity_supported_counts'], ensure_ascii=False)}",
        f"- winner_changed_counts: {json.dumps(payload['winner_changed_counts'], ensure_ascii=False)}",
        f"- rollout_changed_counts: {json.dumps(payload['rollout_changed_counts'], ensure_ascii=False)}",
        f"- mean_delta_continue_stop_margin: {payload['mean_delta_continue_stop_margin']}",
        f"- mean_delta_target_logit: {payload['mean_delta_target_logit']}",
        "",
        "Note: This is a small-scale causal necessity audit, not closure.",
    ]
    (out_dir / "phase269_mlp_necessity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--cases-per-model", type=int, default=2)
    parser.add_argument("--rollout-tokens", type=int, default=8)
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
