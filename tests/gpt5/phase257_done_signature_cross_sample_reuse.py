#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
import phase251_orthogonalized_natural_direction_causal_validation as p251  # noqa: E402
import phase252_shared_subspace_coupled_regime_analysis as p252  # noqa: E402
import phase255_modelclose_internal_stop_trace as p255  # noqa: E402
import phase256_done_signature_counterfactual_localization as p256  # noqa: E402


PHASE = 257
SOURCE_PHASE = 256
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
PHASE250_DIR = Path("tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction")
PHASE255_DIR = Path("tests/result/phase255_modelclose_internal_stop_trace/modelclose_internal_stop_trace")
RESULT_ROOT = Path("tests/result/phase257_done_signature_cross_sample_reuse")
ROUND_DEFAULT = "done_signature_cross_sample_reuse"

SPECS = {
    "qwen3": {"final_layer": 33, "observe_layers": [20, 26, 29, 31, 33]},
    "glm4": {"final_layer": 32, "observe_layers": [20, 26, 28, 30, 32]},
    "deepseek7b": {"final_layer": 27, "observe_layers": [16, 22, 24, 26, 27]},
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


def unit(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n


def norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def normalize(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n * float(target_norm)


def dot(vec: torch.Tensor, direction: torch.Tensor) -> float:
    if vec.numel() != direction.numel():
        return 0.0
    return float(torch.dot(vec.float(), unit(direction).float()).item())


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_behavior_rows(model_name: str, limit: int) -> list[dict[str, Any]]:
    rows = [x for x in read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl") if str(x.get("model")) == model_name]
    rows = [
        x
        for x in rows
        if str(x.get("family_id")) in {"output_protocol", "answer_format", "factual_answer"}
        and list(x.get("target_aliases") or [])
        and str(x.get("prompt_variant"))
    ]
    # Stable deterministic spread over cases without random sampling.
    rows.sort(key=lambda x: (str(x.get("family_id")), str(x.get("mode_id")), str(x.get("case_id")), str(x.get("variant_id"))))
    return rows[: int(limit)]


def load_behavior_by_key() -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    return {(str(x["model"]), str(x["case_id"]), str(x["variant_id"])): x for x in rows}


def load_raw_vectors(model_name: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest = read_json(PHASE246_DIR / "phase246_raw_delta_vector_manifest.json").get("rows", [])
    out = {}
    for item in manifest:
        if str(item.get("model")) != model_name:
            continue
        path = Path(str(item.get("path")))
        if path.exists():
            out[key_for(item)] = {**item, "payload": torch.load(path, map_location="cpu")}
    return out


def load_natural_directions(model_name: str) -> dict[str, torch.Tensor]:
    rows = read_jsonl(PHASE250_DIR / "phase250_natural_direction_rows.jsonl")
    out = {}
    for row in rows:
        if str(row.get("model")) != model_name or str(row.get("scope")) != "global":
            continue
        path = Path(str(row.get("path")))
        if path.exists():
            out[str(row["contrast_id"])] = unit(torch.load(path, map_location="cpu")["direction"])
    return out


def replace_last_token(output: Any, delta: torch.Tensor, sign: float = 1.0) -> Any:
    tensor = p228.extract_tensor(output)
    if tensor is None:
        return output
    changed = tensor.clone()
    changed[:, -1, :] = changed[:, -1, :] + float(sign) * delta.to(device=changed.device, dtype=changed.dtype)
    if torch.is_tensor(output):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    return output


@contextmanager
def no_hook() -> Iterator[None]:
    yield


@contextmanager
def residual_hook(model: Any, layer_idx: int, direction: torch.Tensor, sign: float) -> Iterator[None]:
    layers = p228.get_layers(model)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_last_token(output, direction, sign=sign)

    handle = layers[int(layer_idx)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def make_hook_factory(model: Any, layer_idx: int, direction: torch.Tensor | None) -> Any:
    def factory() -> Any:
        if direction is None:
            return no_hook()
        return residual_hook(model, layer_idx, direction, sign=-1.0)

    return factory


def build_done_direction_from_phase255(
    model_obj: Any,
    tokenizer: Any,
    device: torch.device,
    model_name: str,
    final_layer: int,
    observe_layers: list[int],
    control_dir: torch.Tensor,
    readout_dir: torch.Tensor,
    perturb_norm: float,
) -> tuple[torch.Tensor | None, list[dict[str, Any]]]:
    stop_rows = [x for x in read_jsonl(PHASE255_DIR / "phase255_stop_trace_rows.jsonl") if str(x.get("model")) == model_name]
    step_rows = [x for x in read_jsonl(PHASE255_DIR / "phase255_generation_step_rows.jsonl") if str(x.get("model")) == model_name]
    behavior_by_key = load_behavior_by_key()
    vectors = []
    component_rows = []
    for stop in stop_rows:
        if stop.get("stop_type") != "eos_stop" or stop.get("eos_pos") is None or stop.get("answer_first_step") is None:
            continue
        key = key_for(stop)
        behavior = behavior_by_key.get(key)
        if behavior is None:
            continue
        condition = str(stop["condition"])
        generated_ids = p256.reconstruct_generated_ids(step_rows, model_name, key[1], key[2], condition)
        direction = p255.condition_direction(condition, control_dir, readout_dir, perturb_norm)
        hook_factory = make_hook_factory(model_obj, final_layer, direction)
        hidden_map, _logits_map = p256.capture_prefix_hidden(
            model_obj,
            tokenizer,
            device,
            str(behavior["prompt_variant"]),
            generated_ids,
            condition,
            hook_factory,
            observe_layers,
            list(behavior.get("target_aliases") or []),
        )
        h_answer = hidden_map.get((int(stop["answer_first_step"]), final_layer))
        h_eos = hidden_map.get((int(stop["eos_pos"]), final_layer))
        if h_answer is None or h_eos is None:
            continue
        delta = h_eos - h_answer
        vectors.append(delta)
        component_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase257",
                "created_at": utc_now(),
                "done_vector_component_id": f"phase257:seed_component:{model_name}:{key[1]}:{key[2]}:{condition}",
                "model": model_name,
                "case_id": key[1],
                "variant_id": key[2],
                "condition": condition,
                "delta_norm": round(norm(delta), 6),
                "source": "Phase255 eos condition answer_to_eos delta",
            }
        )
    if not vectors:
        return None, component_rows
    return unit(torch.stack(vectors).mean(dim=0)), component_rows


def prefix_variants(row: dict[str, Any]) -> list[tuple[str, str]]:
    prompt = str(row["prompt_variant"]).rstrip()
    aliases = [str(x).strip() for x in row.get("target_aliases") or [] if str(x).strip()]
    answer = aliases[0] if aliases else "blue"
    return [
        ("prompt_only", prompt),
        ("answer_only", f"{prompt}\n{answer}"),
        ("answer_period", f"{prompt}\n{answer}."),
        ("answer_explain_stub", f"{prompt}\n{answer} because"),
        ("answer_done_template", f"{prompt}\nAnswer: {answer}\n\nReason: {answer}."),
    ]


def capture_prefix(
    model_obj: Any,
    tokenizer: Any,
    device: torch.device,
    text: str,
    observe_layers: list[int],
    done_dir: torch.Tensor,
    target_aliases: list[str],
) -> list[dict[str, Any]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    logits = out.logits[0, last_pos].detach().float().cpu()
    closure = p252.closure_scores(tokenizer, logits)
    readout = p239.readout_metrics(tokenizer, logits, target_aliases)
    eos_logit = logits[int(tokenizer.eos_token_id)].item() if tokenizer.eos_token_id is not None else 0.0
    rows = []
    for layer in observe_layers:
        if int(layer) + 1 >= len(out.hidden_states):
            continue
        vec = out.hidden_states[int(layer) + 1][0, last_pos].detach().float().cpu()
        rows.append(
            {
                "layer": int(layer),
                "done_projection": round(dot(vec, done_dir), 6),
                "eos_logit": round(float(eos_logit), 6),
                **{f"closure_{k}": round(v, 6) for k, v in closure.items()},
                **{f"readout_{k}": v for k, v in readout.items()},
            }
        )
    return rows


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_stop_rows = [x for x in read_jsonl(PHASE255_DIR / "phase255_stop_trace_rows.jsonl") if str(x.get("model")) == args.model and str(x.get("stop_type")) == "eos_stop"]
    if not seed_stop_rows:
        payload = {
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "title": "Done signature cross-sample reuse",
            "status": "complete_no_done_seed",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "seed_eos_rows": 0,
            "reuse_rows": 0,
            "case_summary_rows": 0,
            "missing_rows": 0,
        }
        write_json(out_dir / f"phase257_{args.model}_summary.json", payload)
        for name in ["done_vector_component_rows", "reuse_projection_rows", "case_summary_rows", "observations", "metrics", "graph_edges", "missing_rows"]:
            write_jsonl(out_dir / f"phase257_{args.model}_{name}.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    behavior_rows = load_behavior_rows(args.model, int(args.max_cases))
    raw_vectors = load_raw_vectors(args.model)
    natural_dirs = load_natural_directions(args.model)
    component_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    case_summary_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model_obj = None
    tokenizer = None
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenbank_dirs = p252.build_tokenbank_directions(model_obj, tokenizer)
        final_layer = int(SPECS[args.model]["final_layer"])
        observe_layers = list(SPECS[args.model]["observe_layers"])
        # Use the Phase255 seed case to obtain scale and directions. Do not refit on the reuse cases.
        seed_key = key_for(seed_stop_rows[0])
        raw_item = raw_vectors.get(seed_key)
        if raw_item is None:
            missing_rows.append({"model": args.model, "case_id": seed_key[1], "variant_id": seed_key[2], "reason": "missing_seed_raw"})
            done_dir = None
        else:
            raw = raw_item["payload"]
            tokenbank_id, natural_id = p251.ROUTE_TO_DIRECTIONS.get("", ("continuation_regime", "natural_continuation_explain"))
            perturb_norm = max(norm(raw.get("target_direction")), norm(raw.get("competitor_direction")), norm(raw.get("delta_residual")) * float(args.perturb_scale), 1e-6)
            control_dir = normalize(natural_dirs[natural_id], perturb_norm)
            readout_dir = normalize(tokenbank_dirs[tokenbank_id], perturb_norm)
            done_dir, component_rows = build_done_direction_from_phase255(
                model_obj, tokenizer, device, args.model, final_layer, observe_layers, control_dir, readout_dir, perturb_norm
            )
        if done_dir is None:
            missing_rows.append({"model": args.model, "reason": "done_direction_unavailable"})
        else:
            for idx, behavior in enumerate(behavior_rows, start=1):
                aliases = list(behavior.get("target_aliases") or [])
                per_case: dict[str, float] = {}
                for prefix_kind, text in prefix_variants(behavior):
                    rows = capture_prefix(model_obj, tokenizer, device, text, observe_layers, done_dir, aliases)
                    for row in rows:
                        full = {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase257",
                            "created_at": utc_now(),
                            "reuse_projection_id": f"phase257:reuse:{args.model}:{behavior['case_id']}:{behavior['variant_id']}:{prefix_kind}:L{row['layer']}",
                            "model": args.model,
                            "case_id": behavior["case_id"],
                            "variant_id": behavior["variant_id"],
                            "family_id": behavior.get("family_id"),
                            "mode_id": behavior.get("mode_id"),
                            "prefix_kind": prefix_kind,
                            **row,
                        }
                        reuse_rows.append(full)
                        if int(row["layer"]) == final_layer:
                            per_case[prefix_kind] = safe_float(row["done_projection"])
                case_summary_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase257",
                        "created_at": utc_now(),
                        "case_summary_id": f"phase257:case_summary:{args.model}:{behavior['case_id']}:{behavior['variant_id']}",
                        "model": args.model,
                        "case_id": behavior["case_id"],
                        "variant_id": behavior["variant_id"],
                        "family_id": behavior.get("family_id"),
                        "mode_id": behavior.get("mode_id"),
                        "prompt_only_done": per_case.get("prompt_only"),
                        "answer_only_done": per_case.get("answer_only"),
                        "answer_period_done": per_case.get("answer_period"),
                        "answer_explain_stub_done": per_case.get("answer_explain_stub"),
                        "answer_done_template_done": per_case.get("answer_done_template"),
                        "period_minus_answer": round(per_case.get("answer_period", 0.0) - per_case.get("answer_only", 0.0), 6),
                        "done_template_minus_prompt": round(per_case.get("answer_done_template", 0.0) - per_case.get("prompt_only", 0.0), 6),
                        "reuse_match": bool(per_case.get("answer_period", -1e9) > per_case.get("answer_only", 1e9) and per_case.get("answer_done_template", -1e9) > per_case.get("prompt_only", 1e9)),
                    }
                )
                if idx % 10 == 0:
                    log(f"{args.model}: reuse case {idx}/{len(behavior_rows)} rows={len(reuse_rows)}")
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
    observations = make_observations(reuse_rows, case_summary_rows)
    metrics = make_metrics(args.model, reuse_rows, case_summary_rows)
    edges = make_edges(case_summary_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Done signature cross-sample reuse",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "seed_eos_rows": len(seed_stop_rows),
        "done_vector_component_rows": len(component_rows),
        "reuse_rows": len(reuse_rows),
        "case_summary_rows": len(case_summary_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "reuse_match_count": sum(1 for x in case_summary_rows if x.get("reuse_match")),
        "reuse_match_rate": round(sum(1 for x in case_summary_rows if x.get("reuse_match")) / len(case_summary_rows), 6) if case_summary_rows else 0.0,
        "mean_done_projection_by_prefix_kind": mean_by([x for x in reuse_rows if int(x.get("layer", -1)) == SPECS[args.model]["final_layer"]], "prefix_kind", "done_projection"),
        "mean_period_minus_answer": round(mean(safe_float(x.get("period_minus_answer")) for x in case_summary_rows), 6) if case_summary_rows else 0.0,
        "mean_done_template_minus_prompt": round(mean(safe_float(x.get("done_template_minus_prompt")) for x in case_summary_rows), 6) if case_summary_rows else 0.0,
    }
    write_json(out_dir / f"phase257_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase257_{args.model}_done_vector_component_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase257_{args.model}_reuse_projection_rows.jsonl", reuse_rows)
    write_jsonl(out_dir / f"phase257_{args.model}_case_summary_rows.jsonl", case_summary_rows)
    write_jsonl(out_dir / f"phase257_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase257_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase257_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase257_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_observations(reuse_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in reuse_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase257",
                "created_at": now,
                "observation_id": row["reuse_projection_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "done_signature_cross_sample_reuse",
                "component": f"{row['prefix_kind']}:L{row['layer']}",
                "metric_name": "done_projection",
                "metric_value": row["done_projection"],
                "metric_unit": "projection",
            }
        )
    return rows


def make_metrics(model_name: str, reuse_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    final_layer = SPECS[model_name]["final_layer"]
    final_rows = [x for x in reuse_rows if int(x.get("layer", -1)) == final_layer]
    for prefix_kind, value in mean_by(final_rows, "prefix_kind", "done_projection").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase257",
                "created_at": now,
                "metric_id": f"phase257:{model_name}:{prefix_kind}:mean_done_projection",
                "scope": "done_signature_cross_sample_reuse",
                "model": model_name,
                "prefix_kind": prefix_kind,
                "metric_name": "mean_done_projection",
                "metric_value": value,
                "rows": sum(1 for x in final_rows if x.get("prefix_kind") == prefix_kind),
            }
        )
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase257",
            "created_at": now,
            "metric_id": f"phase257:{model_name}:reuse_match_rate",
            "scope": "case_summary",
            "model": model_name,
            "metric_name": "reuse_match_rate",
            "metric_value": round(sum(1 for x in case_rows if x.get("reuse_match")) / len(case_rows), 6) if case_rows else 0.0,
            "rows": len(case_rows),
        }
    )
    return rows


def make_edges(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in case_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase257",
                "created_at": now,
                "edge_id": f"phase257:reuse:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "source": "node:Phase256DoneDirection",
                "target": f"case:{row['case_id']}:{row['variant_id']}",
                "edge_type": "done_signature_cross_sample_reuse",
                "model": row["model"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "evidence_type": "fixed_done_direction_prefix_projection",
                "effect_direction": "reuse_match" if row.get("reuse_match") else "reuse_weak_or_negative",
                "effect_size": row.get("done_template_minus_prompt"),
                "confidence": 0.42 if row.get("reuse_match") else 0.28,
                "supporting_phases": ["Phase256", "Phase257"],
                "status": "cross_sample_probe_not_causal_closure",
            }
        )
    return rows


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase257_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    components: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        components.extend(read_jsonl(out_dir / f"phase257_{model}_done_vector_component_rows.jsonl"))
        reuse_rows.extend(read_jsonl(out_dir / f"phase257_{model}_reuse_projection_rows.jsonl"))
        case_rows.extend(read_jsonl(out_dir / f"phase257_{model}_case_summary_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase257_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase257_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase257_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase257_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.82,
        "high_value_trace_selection": 0.68,
        "trace_signature_validation": 0.43,
        "focused_causal_validation": 0.25,
        "regime_field_direction_bank": 0.35,
        "natural_regime_direction_bank": 0.30,
        "regime_level_causal_validation": 0.26,
        "shared_subspace_analysis": 0.20,
        "coupled_regime_field_analysis": 0.23,
        "control_readout_coupling": 0.21,
        "stop_type_validation": 0.20,
        "semantic_done_signature": 0.15,
        "residual_state_signature": 0.52,
        "readout_competition_trace": 0.73,
        "stepwise_rollout_trace": 0.41,
        "causal_closure": 0.17,
        "general_language_mechanism_confidence": 0.63,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Done signature cross-sample reuse",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "done_vector_component_rows": len(components),
        "reuse_rows": len(reuse_rows),
        "case_summary_rows": len(case_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "reuse_match_count": sum(1 for x in case_rows if x.get("reuse_match")),
        "reuse_match_rate": round(sum(1 for x in case_rows if x.get("reuse_match")) / len(case_rows), 6) if case_rows else 0.0,
        "mean_done_projection_by_prefix_kind": mean_by([x for x in reuse_rows if int(x.get("layer", -1)) == SPECS.get(str(x.get("model")), {}).get("final_layer", -999)], "prefix_kind", "done_projection"),
        "mean_period_minus_answer": round(mean(safe_float(x.get("period_minus_answer")) for x in case_rows), 6) if case_rows else 0.0,
        "mean_done_template_minus_prompt": round(mean(safe_float(x.get("done_template_minus_prompt")) for x in case_rows), 6) if case_rows else 0.0,
        "progress": progress,
    }
    write_json(out_dir / "phase257_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase257_done_vector_component_rows.jsonl", components)
    write_jsonl(out_dir / "phase257_reuse_projection_rows.jsonl", reuse_rows)
    write_jsonl(out_dir / "phase257_case_summary_rows.jsonl", case_rows)
    write_jsonl(out_dir / "phase257_observations.jsonl", observations)
    write_jsonl(out_dir / "phase257_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase257_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase257_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase257", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase257 Done Signature Cross-Sample Reuse",
        "",
        f"- status: {payload['status']}",
        f"- reuse_rows: {payload['reuse_rows']}",
        f"- case_summary_rows: {payload['case_summary_rows']}",
        f"- reuse_match_rate: {payload['reuse_match_rate']}",
        f"- mean_period_minus_answer: {payload['mean_period_minus_answer']}",
        f"- mean_done_template_minus_prompt: {payload['mean_done_template_minus_prompt']}",
        f"- mean_done_projection_by_prefix_kind: {json.dumps(payload['mean_done_projection_by_prefix_kind'], ensure_ascii=False)}",
    ]
    (out_dir / "phase257_done_signature_cross_sample_reuse_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--perturb-scale", type=float, default=0.5)
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
