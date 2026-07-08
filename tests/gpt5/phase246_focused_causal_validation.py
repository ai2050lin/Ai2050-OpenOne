#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402


PHASE = 246
SOURCE_PHASE = 245
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE245_DIR = Path("tests/result/phase245_trace_signature_validation_and_frozen_audit/trace_signature_validation_and_frozen_audit")
RESULT_ROOT = Path("tests/result/phase246_focused_causal_validation")
ROUND_DEFAULT = "focused_causal_validation"

SPECS = {
    "qwen3": {"source_layer": 29, "observe_layers": [29, 31, 33]},
    "glm4": {"source_layer": 30, "observe_layers": [28, 30, 32]},
    "deepseek7b": {"source_layer": 24, "observe_layers": [24, 26, 27]},
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
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def vector_norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def normalize(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    norm = torch.linalg.vector_norm(vec).item()
    if norm <= 1e-8:
        return torch.zeros_like(vec)
    return vec / norm * float(target_norm)


def load_inputs(max_total_candidates: int) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    candidates = read_jsonl(PHASE245_DIR / "phase245_causal_test_candidate_rows.jsonl")
    if not candidates:
        raise FileNotFoundError(f"missing phase245 causal candidates under {PHASE245_DIR}")
    candidates.sort(key=lambda x: safe_float(x.get("candidate_score")), reverse=True)
    if max_total_candidates > 0:
        candidates = candidates[:max_total_candidates]
    behavior_rows = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    if not behavior_rows:
        raise FileNotFoundError(f"missing Phase241 behavior rows under {PHASE241_DIR}")
    behavior_by_key = {(str(x["model"]), str(x["case_id"]), str(x["variant_id"])): x for x in behavior_rows}
    return candidates, behavior_by_key


def selected_for_model(model_name: str, max_total_candidates: int) -> list[dict[str, Any]]:
    candidates, _behavior = load_inputs(max_total_candidates)
    return [row for row in candidates if str(row.get("model")) == model_name]


def get_output_embedding_weight(model: Any) -> torch.Tensor:
    head = model.get_output_embeddings()
    weight = getattr(head, "weight", None)
    if weight is None:
        raise RuntimeError("model output embeddings do not expose weight")
    return weight.detach().float().cpu()


def replace_last_token(output: Any, delta: torch.Tensor, sign: float = 1.0) -> Any:
    tensor = p228.extract_tensor(output)
    if tensor is None:
        return output
    changed = tensor.clone()
    delta_device = delta.to(device=changed.device, dtype=changed.dtype)
    changed[:, -1, :] = changed[:, -1, :] + float(sign) * delta_device
    if torch.is_tensor(output):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    return output


@contextmanager
def no_hook() -> Iterator[None]:
    yield


@contextmanager
def down_out_delta_hook(model: Any, source_layer: int, delta: torch.Tensor, sign: float) -> Iterator[None]:
    layers = p228.get_layers(model)
    mlp = p228.get_mlp(layers[int(source_layer)])
    down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
    if down_proj is None:
        yield
        return

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_last_token(output, delta, sign=sign)

    handle = down_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def residual_direction_hook(model: Any, layer_idx: int, direction: torch.Tensor, sign: float) -> Iterator[None]:
    layers = p228.get_layers(model)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_last_token(output, direction, sign=sign)

    handle = layers[int(layer_idx)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def forward_logits(model: Any, tokenizer: Any, device: torch.device, prompt: str, hook_ctx: Any) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with hook_ctx:
        with torch.inference_mode():
            outputs = model(**encoded)
    return outputs.logits[0, -1, :].detach().float().cpu()


def rollout_text(model: Any, tokenizer: Any, device: torch.device, prompt: str, max_new_tokens: int, hook_ctx_factory: Any) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with hook_ctx_factory():
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
    return tokenizer.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def intervention_context(
    model: Any,
    intervention: str,
    source_layer: int,
    final_observe_layer: int,
    delta_down_out: torch.Tensor,
    target_direction: torch.Tensor,
    competitor_direction: torch.Tensor,
) -> Any:
    if intervention == "down_out_delta_ablation":
        return down_out_delta_hook(model, source_layer, delta_down_out, sign=-1.0)
    if intervention == "target_unembed_injection":
        return residual_direction_hook(model, final_observe_layer, target_direction, sign=1.0)
    if intervention == "top_competitor_suppression":
        return residual_direction_hook(model, final_observe_layer, competitor_direction, sign=-1.0)
    return no_hook()


def classify_effect(original_margin: float, intervention_margin: float, intervention: str) -> str:
    delta = intervention_margin - original_margin
    if intervention == "down_out_delta_ablation":
        if delta <= -1.0:
            return "necessity_signal_margin_dropped"
        if delta >= 1.0:
            return "ablation_improved_margin_opposite_signal"
        return "weak_or_no_necessity_signal"
    if delta >= 1.0:
        return "sufficiency_or_readout_gain_signal"
    if delta <= -1.0:
        return "intervention_harmed_margin"
    return "weak_or_no_sufficiency_signal"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    raw_dir = out_dir / "raw_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidates, behavior_by_key = load_inputs(int(args.max_total_candidates))
    model_candidates = [row for row in candidates if str(row.get("model")) == args.model]
    spec = SPECS[args.model]
    source_layer = int(spec["source_layer"])
    observe_layers = list(spec["observe_layers"])
    final_observe_layer = int(observe_layers[-1])
    run_id = f"phase246:{args.model}:{args.round_name}"
    validation_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    injection_rows: list[dict[str, Any]] = []
    suppression_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    raw_manifest: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        output_weight = get_output_embedding_weight(model)
        for index, candidate in enumerate(model_candidates, start=1):
            key = (args.model, str(candidate.get("case_id")), str(candidate.get("variant_id")))
            base_key = (args.model, str(candidate.get("case_id")), "full")
            behavior = behavior_by_key.get(key)
            baseline_behavior = behavior_by_key.get(base_key)
            if behavior is None or baseline_behavior is None:
                missing_rows.append({"model": args.model, "case_id": candidate.get("case_id"), "variant_id": candidate.get("variant_id")})
                continue
            prompt = str(behavior["prompt_variant"])
            baseline_prompt = str(baseline_behavior["prompt_variant"])
            internal, hidden, logits = p228.capture_internal(model, tokenizer, device, prompt, [source_layer], observe_layers)
            base_internal, base_hidden, _base_logits = p228.capture_internal(
                model, tokenizer, device, baseline_prompt, [source_layer], observe_layers
            )
            readout_original = p239.readout_metrics(tokenizer, logits, list(behavior.get("target_aliases") or []))
            down_out = internal.get(source_layer, {}).get("down_out")
            base_down_out = base_internal.get(source_layer, {}).get("down_out")
            product = internal.get(source_layer, {}).get("product")
            base_product = base_internal.get(source_layer, {}).get("product")
            residual = hidden.get(final_observe_layer)
            base_residual = base_hidden.get(final_observe_layer)
            if not torch.is_tensor(down_out) or not torch.is_tensor(base_down_out) or not torch.is_tensor(residual) or not torch.is_tensor(base_residual):
                missing_rows.append({"model": args.model, "case_id": candidate.get("case_id"), "variant_id": candidate.get("variant_id"), "reason": "missing_raw_vectors"})
                continue
            delta_down_out = down_out.float() - base_down_out.float()
            delta_product = product.float() - base_product.float() if torch.is_tensor(product) and torch.is_tensor(base_product) else torch.zeros_like(delta_down_out)
            delta_residual = residual.float() - base_residual.float()
            target_token_id = int(readout_original.get("target_token_id"))
            top_token_id = int(readout_original.get("top_token_id"))
            perturb_norm = max(vector_norm(delta_residual), vector_norm(delta_down_out), 1e-6) * float(args.perturb_scale)
            target_direction = normalize(output_weight[target_token_id], perturb_norm)
            competitor_direction = normalize(output_weight[top_token_id], perturb_norm)
            raw_path = raw_dir / f"{safe_slug(args.model + '_' + str(candidate.get('case_id')) + '_' + str(candidate.get('variant_id')))}.pt"
            torch.save(
                {
                    "model": args.model,
                    "case_id": candidate.get("case_id"),
                    "variant_id": candidate.get("variant_id"),
                    "source_layer": source_layer,
                    "final_observe_layer": final_observe_layer,
                    "delta_down_out": delta_down_out.cpu(),
                    "delta_product": delta_product.cpu(),
                    "delta_residual": delta_residual.cpu(),
                    "target_direction": target_direction.cpu(),
                    "competitor_direction": competitor_direction.cpu(),
                    "target_token_id": target_token_id,
                    "top_token_id": top_token_id,
                },
                raw_path,
            )
            raw_manifest.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase246",
                    "created_at": utc_now(),
                    "raw_vector_id": f"phase246:raw:{args.model}:{candidate.get('case_id')}:{candidate.get('variant_id')}",
                    "model": args.model,
                    "case_id": candidate.get("case_id"),
                    "variant_id": candidate.get("variant_id"),
                    "path": str(raw_path),
                    "source_layer": source_layer,
                    "final_observe_layer": final_observe_layer,
                    "delta_down_out_norm": round(vector_norm(delta_down_out), 6),
                    "delta_product_norm": round(vector_norm(delta_product), 6),
                    "delta_residual_norm": round(vector_norm(delta_residual), 6),
                    "perturb_norm": round(perturb_norm, 6),
                }
            )
            interventions = [
                "no_intervention",
                "down_out_delta_ablation",
                "target_unembed_injection",
                "top_competitor_suppression",
            ]
            original_margin = safe_float(readout_original.get("target_margin_vs_winner"))
            for intervention in interventions:
                ctx = intervention_context(
                    model,
                    intervention,
                    source_layer,
                    final_observe_layer,
                    delta_down_out,
                    target_direction,
                    competitor_direction,
                )
                if intervention == "no_intervention":
                    logits_i = logits
                    rollout_ctx_factory = no_hook
                else:
                    logits_i = forward_logits(model, tokenizer, device, prompt, ctx)

                    def rollout_ctx_factory(
                        intervention_name: str = intervention,
                        d_down: torch.Tensor = delta_down_out,
                        t_dir: torch.Tensor = target_direction,
                        c_dir: torch.Tensor = competitor_direction,
                    ) -> Any:
                        return intervention_context(
                            model,
                            intervention_name,
                            source_layer,
                            final_observe_layer,
                            d_down,
                            t_dir,
                            c_dir,
                        )

                readout = p239.readout_metrics(tokenizer, logits_i, list(behavior.get("target_aliases") or []))
                rollout = rollout_text(model, tokenizer, device, prompt, int(args.max_rollout_tokens), rollout_ctx_factory)
                margin = safe_float(readout.get("target_margin_vs_winner"))
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase246",
                    "source_phase": "Phase245",
                    "created_at": utc_now(),
                    "run_id": run_id,
                    "validation_id": f"phase246:{args.model}:{candidate.get('case_id')}:{candidate.get('variant_id')}:{intervention}",
                    "candidate_id": candidate.get("candidate_id"),
                    "model": args.model,
                    "case_id": candidate.get("case_id"),
                    "variant_id": candidate.get("variant_id"),
                    "family_id": candidate.get("family_id"),
                    "mode_id": candidate.get("mode_id"),
                    "data_split": candidate.get("data_split"),
                    "signature_class": candidate.get("signature_class"),
                    "recommended_causal_test": candidate.get("recommended_causal_test"),
                    "intervention": intervention,
                    "source_layer": source_layer,
                    "final_observe_layer": final_observe_layer,
                    "target_token_id": target_token_id,
                    "top_token_id": top_token_id,
                    "target_margin_vs_winner": margin,
                    "target_margin_delta_vs_original": round(margin - original_margin, 6),
                    "target_rank": readout.get("target_rank"),
                    "winning_regime": readout.get("winning_regime"),
                    "second_competitor": readout.get("second_competitor"),
                    "top_token": readout.get("top_token"),
                    "winner_changed_vs_original": readout.get("winning_regime") != readout_original.get("winning_regime"),
                    "rollout_text": rollout,
                    "rollout_token_count": len(rollout.replace("\n", " ").split()) if rollout else 0,
                    "causal_effect_label": classify_effect(original_margin, margin, intervention),
                    "candidate_score": candidate.get("candidate_score"),
                    "selection_reasons": candidate.get("selection_reasons", []),
                }
                validation_rows.append(row)
                if intervention == "down_out_delta_ablation":
                    ablation_rows.append(row)
                elif intervention == "target_unembed_injection":
                    injection_rows.append(row)
                elif intervention == "top_competitor_suppression":
                    suppression_rows.append(row)
                rollout_rows.append(
                    {
                        **row,
                        "rollout_id": f"{row['validation_id']}:rollout",
                        "trace_level": "rollout_closure_perturbation",
                    }
                )
            log(f"{args.model}: candidate={index}/{len(model_candidates)} rows={len(validation_rows)}")
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

    metrics = model_metric_rows(args.model, validation_rows)
    observations = observation_rows(validation_rows)
    edges = graph_edges(validation_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Focused causal validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "candidate_count": len(model_candidates),
        "missing_rows": len(missing_rows),
        "validation_rows": len(validation_rows),
        "component_ablation_rows": len(ablation_rows),
        "target_injection_rows": len(injection_rows),
        "competitor_suppression_rows": len(suppression_rows),
        "rollout_closure_perturbation_rows": len(rollout_rows),
        "raw_delta_vectors": len(raw_manifest),
        "necessity_signal_count": sum(1 for x in ablation_rows if x.get("causal_effect_label") == "necessity_signal_margin_dropped"),
        "target_injection_gain_count": sum(1 for x in injection_rows if x.get("causal_effect_label") == "sufficiency_or_readout_gain_signal"),
        "competitor_suppression_gain_count": sum(1 for x in suppression_rows if x.get("causal_effect_label") == "sufficiency_or_readout_gain_signal"),
        "effect_labels": dict(Counter(str(x.get("causal_effect_label")) for x in validation_rows).most_common()),
    }
    write_json(out_dir / f"phase246_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase246_{args.model}_causal_validation_rows.jsonl", validation_rows)
    write_jsonl(out_dir / f"phase246_{args.model}_component_ablation_rows.jsonl", ablation_rows)
    write_jsonl(out_dir / f"phase246_{args.model}_target_injection_rows.jsonl", injection_rows)
    write_jsonl(out_dir / f"phase246_{args.model}_competitor_suppression_rows.jsonl", suppression_rows)
    write_jsonl(out_dir / f"phase246_{args.model}_rollout_closure_perturbation_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase246_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase246_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase246_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase246_{args.model}_missing_rows.jsonl", missing_rows)
    write_json(out_dir / f"phase246_{args.model}_raw_delta_vector_manifest.json", {"rows": raw_manifest})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def model_metric_rows(model_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    by_intervention: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intervention[str(row.get("intervention"))].append(row)
    for intervention, items in by_intervention.items():
        if not items:
            continue
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase246",
                "created_at": now,
                "metric_id": f"phase246:{model_name}:{intervention}:margin_delta",
                "scope": "focused_causal_validation",
                "model": model_name,
                "intervention": intervention,
                "metric_name": "mean_target_margin_delta_vs_original",
                "metric_value": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in items), 6),
                "winner_changed_rate": round(sum(1 for x in items if x.get("winner_changed_vs_original")) / max(1, len(items)), 4),
                "effect_labels": dict(Counter(str(x.get("causal_effect_label")) for x in items).most_common()),
                "rows": len(items),
            }
        )
    return out


def observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase246",
                "created_at": now,
                "observation_id": f"phase246:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['intervention']}:margin_delta",
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "variant_id": row["variant_id"],
                "level": "focused_causal_validation",
                "component": row["intervention"],
                "metric_name": "target_margin_delta_vs_original",
                "metric_value": safe_float(row.get("target_margin_delta_vs_original")),
                "metric_unit": "logit",
                "signature_class": row.get("signature_class"),
                "causal_effect_label": row.get("causal_effect_label"),
                "data_split": row.get("data_split"),
            }
        )
    return out


def graph_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in rows:
        if row.get("intervention") == "no_intervention":
            continue
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase246",
                "created_at": now,
                "edge_id": f"phase246:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['intervention']}",
                "source": f"intervention:{row['intervention']}",
                "target": "node:ReadoutMargin",
                "edge_type": "focused_causal_validation",
                "model": row["model"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "variant_id": row["variant_id"],
                "evidence_type": "logit_margin_perturbation",
                "effect_direction": row["causal_effect_label"],
                "effect_size": safe_float(row.get("target_margin_delta_vs_original")),
                "confidence": 0.50 if abs(safe_float(row.get("target_margin_delta_vs_original"))) >= 1.0 else 0.36,
                "supporting_phases": ["Phase244", "Phase245", "Phase246"],
                "status": "causal_signal_not_closure",
            }
        )
    return out


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase246_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    validation_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    injection_rows: list[dict[str, Any]] = []
    suppression_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    raw_manifest = []
    for model in MODELS:
        validation_rows.extend(read_jsonl(out_dir / f"phase246_{model}_causal_validation_rows.jsonl"))
        ablation_rows.extend(read_jsonl(out_dir / f"phase246_{model}_component_ablation_rows.jsonl"))
        injection_rows.extend(read_jsonl(out_dir / f"phase246_{model}_target_injection_rows.jsonl"))
        suppression_rows.extend(read_jsonl(out_dir / f"phase246_{model}_competitor_suppression_rows.jsonl"))
        rollout_rows.extend(read_jsonl(out_dir / f"phase246_{model}_rollout_closure_perturbation_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase246_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase246_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase246_{model}_graph_edges.jsonl"))
        raw_manifest.extend(read_json(out_dir / f"phase246_{model}_raw_delta_vector_manifest.json").get("rows", []))
    progress = {
        "pattern_family_atlas": 0.74,
        "candidate_clustering": 0.42,
        "case_bank_calibration": 0.39,
        "high_value_trace_selection": 0.60,
        "first_internal_trace_batch": 0.38,
        "trace_signature_validation": 0.35,
        "focused_causal_validation": 0.20,
        "raw_delta_vector_archive": 0.18,
        "gate_up_product_signature": 0.44,
        "residual_state_signature": 0.41,
        "readout_competition_trace": 0.63,
        "stepwise_rollout_trace": 0.23,
        "proxy_factor_decomposition": 0.18,
        "causal_closure": 0.12,
        "general_language_mechanism_confidence": 0.55,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model focused causal validation",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "model_summaries": {x.get("model"): x for x in summaries},
        "candidate_count": sum(int(x.get("candidate_count") or 0) for x in summaries),
        "validation_rows": len(validation_rows),
        "component_ablation_rows": len(ablation_rows),
        "target_injection_rows": len(injection_rows),
        "competitor_suppression_rows": len(suppression_rows),
        "rollout_closure_perturbation_rows": len(rollout_rows),
        "raw_delta_vectors": len(raw_manifest),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "necessity_signal_count": sum(1 for x in ablation_rows if x.get("causal_effect_label") == "necessity_signal_margin_dropped"),
        "target_injection_gain_count": sum(1 for x in injection_rows if x.get("causal_effect_label") == "sufficiency_or_readout_gain_signal"),
        "competitor_suppression_gain_count": sum(1 for x in suppression_rows if x.get("causal_effect_label") == "sufficiency_or_readout_gain_signal"),
        "effect_labels": dict(Counter(str(x.get("causal_effect_label")) for x in validation_rows).most_common()),
        "mean_ablation_margin_delta": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in ablation_rows), 6)
        if ablation_rows
        else 0.0,
        "mean_target_injection_margin_delta": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in injection_rows), 6)
        if injection_rows
        else 0.0,
        "mean_competitor_suppression_margin_delta": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in suppression_rows), 6)
        if suppression_rows
        else 0.0,
        "pattern_atlas_progress": progress,
        "judgement": "focused_causal_signal_not_closure",
    }
    write_json(out_dir / "phase246_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase246_causal_validation_rows.jsonl", validation_rows)
    write_jsonl(out_dir / "phase246_component_ablation_rows.jsonl", ablation_rows)
    write_jsonl(out_dir / "phase246_target_injection_rows.jsonl", injection_rows)
    write_jsonl(out_dir / "phase246_competitor_suppression_rows.jsonl", suppression_rows)
    write_jsonl(out_dir / "phase246_rollout_closure_perturbation_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / "phase246_observations.jsonl", observations)
    write_jsonl(out_dir / "phase246_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase246_graph_edges.jsonl", edges)
    write_json(out_dir / "phase246_raw_delta_vector_manifest.json", {"rows": raw_manifest})
    write_report(out_dir / "phase246_causal_validation_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase246 focused causal validation",
        "",
        "Phase246 performs small-scale causal-signal tests over Phase245 candidates.",
        "It is not closure validation.",
        "",
        "## Counts",
        "",
        f"- candidates: {summary['candidate_count']}",
        f"- validation_rows: {summary['validation_rows']}",
        f"- raw_delta_vectors: {summary['raw_delta_vectors']}",
        f"- necessity_signal_count: {summary['necessity_signal_count']}",
        f"- target_injection_gain_count: {summary['target_injection_gain_count']}",
        f"- competitor_suppression_gain_count: {summary['competitor_suppression_gain_count']}",
        "",
        "## Mean effects",
        "",
        f"- mean_ablation_margin_delta: {summary['mean_ablation_margin_delta']}",
        f"- mean_target_injection_margin_delta: {summary['mean_target_injection_margin_delta']}",
        f"- mean_competitor_suppression_margin_delta: {summary['mean_competitor_suppression_margin_delta']}",
        "",
        "## Progress",
        "",
        "```json",
        json.dumps(summary["pattern_atlas_progress"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_pattern_atlas(summary: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    ATLAS_ROOT.mkdir(parents=True, exist_ok=True)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress_path = ATLAS_ROOT / "progress.json"
    progress = read_json(progress_path)
    progress.update(
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "latest_phase": "Phase246",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "expand only causal signals that survive focused intervention tests",
            "small_model_bias_warning": "Phase246 uses qwen3/glm4/deepseek7b only; causal signals are small-model evidence and not closure.",
        }
    )
    write_json(progress_path, progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase246 focused causal validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-total-candidates", type=int, default=15)
    parser.add_argument("--max-rollout-tokens", type=int, default=4)
    parser.add_argument("--perturb-scale", type=float, default=0.35)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is set")
    eval_model(args)


if __name__ == "__main__":
    main()
