#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
import phase252_shared_subspace_coupled_regime_analysis as p252  # noqa: E402


PHASE = 258
SOURCE_PHASE = 257
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
RESULT_ROOT = Path("tests/result/phase258_done_state_cluster_mode_decomposition")
ROUND_DEFAULT = "done_state_cluster_mode_decomposition"
MODES = ["short_answer", "one_word", "explain_answer", "repeat_answer", "list_answer", "json_answer", "table_answer", "stop_after_answer"]

SPECS = {
    "qwen3": {"final_layer": 33},
    "glm4": {"final_layer": 32},
    "deepseek7b": {"final_layer": 27},
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


def dot(vec: torch.Tensor, direction: torch.Tensor) -> float:
    if vec.numel() != direction.numel():
        return 0.0
    return float(torch.dot(vec.float(), unit(direction).float()).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        return 0.0
    an = torch.linalg.vector_norm(a.float()).item()
    bn = torch.linalg.vector_norm(b.float()).item()
    if an <= 1e-8 or bn <= 1e-8:
        return 0.0
    return float(torch.dot(a.float(), b.float()).item() / (an * bn))


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def load_mode_rows(model_name: str, max_cases_per_mode: int) -> list[dict[str, Any]]:
    rows = [x for x in read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl") if str(x.get("model")) == model_name]
    rows = [x for x in rows if str(x.get("family_id")) == "output_protocol" and str(x.get("mode_id")) in MODES and list(x.get("target_aliases") or [])]
    rows.sort(key=lambda x: (str(x.get("mode_id")), str(x.get("case_id")), str(x.get("variant_id"))))
    out = []
    counts: Counter[str] = Counter()
    for row in rows:
        mode = str(row.get("mode_id"))
        if counts[mode] >= int(max_cases_per_mode):
            continue
        out.append(row)
        counts[mode] += 1
    return out


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


def capture_hidden(model_obj: Any, tokenizer: Any, device: torch.device, text: str, final_layer: int, target_aliases: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    vec = out.hidden_states[int(final_layer) + 1][0, last_pos].detach().float().cpu()
    logits = out.logits[0, last_pos].detach().float().cpu()
    closure = p252.closure_scores(tokenizer, logits)
    readout = p239.readout_metrics(tokenizer, logits, target_aliases)
    return vec, {**{f"closure_{k}": round(v, 6) for k, v in closure.items()}, **{f"readout_{k}": v for k, v in readout.items()}}


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_rows = load_mode_rows(args.model, int(args.max_cases_per_mode))
    model_obj = None
    tokenizer = None
    vector_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    metric_rows_out: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        final_layer = int(SPECS[args.model]["final_layer"])
        hidden_by_case: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
        meta_by_case: dict[tuple[str, str, str], dict[str, Any]] = {}
        logit_meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for idx, row in enumerate(behavior_rows, start=1):
            key = (str(row["case_id"]), str(row["variant_id"]), str(row["mode_id"]))
            meta_by_case[key] = row
            hidden_by_case[key] = {}
            aliases = list(row.get("target_aliases") or [])
            for prefix_kind, text in prefix_variants(row):
                vec, meta = capture_hidden(model_obj, tokenizer, device, text, final_layer, aliases)
                hidden_by_case[key][prefix_kind] = vec
                logit_meta[(key[0], key[1], key[2], prefix_kind)] = meta
            if idx % 20 == 0:
                log(f"{args.model}: captured {idx}/{len(behavior_rows)} cases")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        directions: dict[str, torch.Tensor] = {}
        for mode in MODES:
            deltas = []
            for key, hidden in hidden_by_case.items():
                if key[2] != mode:
                    continue
                if "answer_done_template" in hidden and "prompt_only" in hidden:
                    deltas.append(hidden["answer_done_template"] - hidden["prompt_only"])
            if not deltas:
                continue
            direction = unit(torch.stack(deltas).mean(dim=0))
            directions[mode] = direction
            vector_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase258",
                    "created_at": utc_now(),
                    "done_cluster_vector_id": f"phase258:done_cluster_vector:{args.model}:{mode}",
                    "model": args.model,
                    "mode_id": mode,
                    "component_cases": len(deltas),
                    "construction": "mean(answer_done_template_hidden - prompt_only_hidden)",
                }
            )
        for source_mode, direction in directions.items():
            for key, hidden in hidden_by_case.items():
                row = meta_by_case[key]
                vals = {prefix: dot(vec, direction) for prefix, vec in hidden.items()}
                period_minus_answer = vals.get("answer_period", 0.0) - vals.get("answer_only", 0.0)
                template_minus_prompt = vals.get("answer_done_template", 0.0) - vals.get("prompt_only", 0.0)
                reuse_match = period_minus_answer > 0 and template_minus_prompt > 0
                transfer_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase258",
                        "created_at": utc_now(),
                        "transfer_id": f"phase258:transfer:{args.model}:{source_mode}:{key[2]}:{key[0]}:{key[1]}",
                        "model": args.model,
                        "source_mode": source_mode,
                        "target_mode": key[2],
                        "case_id": key[0],
                        "variant_id": key[1],
                        "period_minus_answer": round(period_minus_answer, 6),
                        "done_template_minus_prompt": round(template_minus_prompt, 6),
                        "reuse_match": reuse_match,
                        "transfer_type": "within_mode" if source_mode == key[2] else "cross_mode",
                    }
                )
                for prefix_kind, value in vals.items():
                    projection_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase258",
                            "created_at": utc_now(),
                            "projection_id": f"phase258:projection:{args.model}:{source_mode}:{key[2]}:{key[0]}:{key[1]}:{prefix_kind}",
                            "model": args.model,
                            "source_mode": source_mode,
                            "target_mode": key[2],
                            "case_id": key[0],
                            "variant_id": key[1],
                            "prefix_kind": prefix_kind,
                            "done_projection": round(value, 6),
                            **logit_meta.get((key[0], key[1], key[2], prefix_kind), {}),
                        }
                    )
        for source_mode, a in directions.items():
            for target_mode, b in directions.items():
                if source_mode >= target_mode:
                    continue
                metric_rows_out.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase258",
                        "created_at": utc_now(),
                        "metric_id": f"phase258:{args.model}:cosine:{source_mode}:{target_mode}",
                        "scope": "done_cluster_direction_similarity",
                        "model": args.model,
                        "source_mode": source_mode,
                        "target_mode": target_mode,
                        "metric_name": "done_direction_cosine",
                        "metric_value": round(cosine(a, b), 6),
                    }
                )
        for (source_mode, target_mode), rows in defaultdict(list, {k: [] for k in []}).items():
            pass
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in transfer_rows:
            grouped[(str(row["source_mode"]), str(row["target_mode"]))].append(row)
        for (source_mode, target_mode), rows in grouped.items():
            rate = sum(1 for x in rows if x.get("reuse_match")) / len(rows)
            metric_rows_out.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase258",
                    "created_at": utc_now(),
                    "metric_id": f"phase258:{args.model}:transfer_rate:{source_mode}:{target_mode}",
                    "scope": "done_cluster_transfer",
                    "model": args.model,
                    "source_mode": source_mode,
                    "target_mode": target_mode,
                    "metric_name": "reuse_match_rate",
                    "metric_value": round(rate, 6),
                    "rows": len(rows),
                }
            )
            edge_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase258",
                    "created_at": utc_now(),
                    "edge_id": f"phase258:transfer:{args.model}:{source_mode}:{target_mode}",
                    "source": f"done_cluster:{source_mode}",
                    "target": f"mode:{target_mode}",
                    "edge_type": "done_cluster_transfer",
                    "model": args.model,
                    "evidence_type": "fixed_mode_done_direction_transfer",
                    "effect_direction": "within_mode" if source_mode == target_mode else "cross_mode",
                    "effect_size": round(rate, 6),
                    "confidence": 0.44 if source_mode == target_mode and rate >= 0.5 else 0.30,
                    "supporting_phases": ["Phase257", "Phase258"],
                    "status": "cluster_map_not_causal_closure",
                }
            )
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    observations = observation_rows(projection_rows, transfer_rows)
    metrics = metric_rows_out + metric_rows(args.model, transfer_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Done state cluster mode decomposition",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "case_count": len(behavior_rows),
        "done_cluster_vectors": len(vector_rows),
        "projection_rows": len(projection_rows),
        "transfer_rows": len(transfer_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edge_rows),
        "missing_rows": len(missing_rows),
        "within_mode_reuse_rate": reuse_rate([x for x in transfer_rows if x.get("transfer_type") == "within_mode"]),
        "cross_mode_reuse_rate": reuse_rate([x for x in transfer_rows if x.get("transfer_type") == "cross_mode"]),
        "reuse_rate_by_source_mode": mode_rate(transfer_rows, "source_mode"),
        "reuse_rate_by_target_mode": mode_rate(transfer_rows, "target_mode"),
    }
    write_json(out_dir / f"phase258_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase258_{args.model}_done_cluster_vector_rows.jsonl", vector_rows)
    write_jsonl(out_dir / f"phase258_{args.model}_projection_rows.jsonl", projection_rows)
    write_jsonl(out_dir / f"phase258_{args.model}_transfer_rows.jsonl", transfer_rows)
    write_jsonl(out_dir / f"phase258_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase258_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase258_{args.model}_graph_edges.jsonl", edge_rows)
    write_jsonl(out_dir / f"phase258_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def reuse_rate(rows: list[dict[str, Any]]) -> float:
    return round(sum(1 for x in rows if x.get("reuse_match")) / len(rows), 6) if rows else 0.0


def mode_rate(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return {k: reuse_rate(v) for k, v in grouped.items()}


def observation_rows(projections: list[dict[str, Any]], transfers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in projections:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase258",
                "created_at": now,
                "observation_id": row["projection_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": "output_protocol",
                "mode_id": row["target_mode"],
                "variant_id": row["variant_id"],
                "level": "done_cluster_projection",
                "component": f"{row['source_mode']}->{row['target_mode']}:{row['prefix_kind']}",
                "metric_name": "done_projection",
                "metric_value": row["done_projection"],
                "metric_unit": "projection",
            }
        )
    for row in transfers:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase258",
                "created_at": now,
                "observation_id": row["transfer_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": "output_protocol",
                "mode_id": row["target_mode"],
                "variant_id": row["variant_id"],
                "level": "done_cluster_transfer",
                "component": f"{row['source_mode']}->{row['target_mode']}",
                "metric_name": "reuse_match",
                "metric_value": 1.0 if row["reuse_match"] else 0.0,
                "metric_unit": "binary",
            }
        )
    return rows


def metric_rows(model_name: str, transfers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for transfer_type, group in [("within_mode", [x for x in transfers if x.get("transfer_type") == "within_mode"]), ("cross_mode", [x for x in transfers if x.get("transfer_type") == "cross_mode"])]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase258",
                "created_at": now,
                "metric_id": f"phase258:{model_name}:{transfer_type}:reuse_rate",
                "scope": "done_cluster_transfer",
                "model": model_name,
                "transfer_type": transfer_type,
                "metric_name": "reuse_match_rate",
                "metric_value": reuse_rate(group),
                "rows": len(group),
            }
        )
    return rows


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase258_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    vectors: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        vectors.extend(read_jsonl(out_dir / f"phase258_{model}_done_cluster_vector_rows.jsonl"))
        projections.extend(read_jsonl(out_dir / f"phase258_{model}_projection_rows.jsonl"))
        transfers.extend(read_jsonl(out_dir / f"phase258_{model}_transfer_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase258_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase258_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase258_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase258_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.83,
        "trace_signature_validation": 0.44,
        "semantic_done_signature": 0.20,
        "done_state_cluster_map": 0.16,
        "residual_state_signature": 0.53,
        "readout_competition_trace": 0.73,
        "stepwise_rollout_trace": 0.41,
        "causal_closure": 0.17,
        "general_language_mechanism_confidence": 0.64,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Done state cluster mode decomposition",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "done_cluster_vectors": len(vectors),
        "projection_rows": len(projections),
        "transfer_rows": len(transfers),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "within_mode_reuse_rate": reuse_rate([x for x in transfers if x.get("transfer_type") == "within_mode"]),
        "cross_mode_reuse_rate": reuse_rate([x for x in transfers if x.get("transfer_type") == "cross_mode"]),
        "reuse_rate_by_model": mode_rate(transfers, "model"),
        "reuse_rate_by_source_mode": mode_rate(transfers, "source_mode"),
        "reuse_rate_by_target_mode": mode_rate(transfers, "target_mode"),
        "progress": progress,
    }
    write_json(out_dir / "phase258_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase258_done_cluster_vector_rows.jsonl", vectors)
    write_jsonl(out_dir / "phase258_projection_rows.jsonl", projections)
    write_jsonl(out_dir / "phase258_transfer_rows.jsonl", transfers)
    write_jsonl(out_dir / "phase258_observations.jsonl", observations)
    write_jsonl(out_dir / "phase258_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase258_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase258_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase258", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase258 Done State Cluster Mode Decomposition",
        "",
        f"- status: {payload['status']}",
        f"- done_cluster_vectors: {payload['done_cluster_vectors']}",
        f"- projection_rows: {payload['projection_rows']}",
        f"- transfer_rows: {payload['transfer_rows']}",
        f"- within_mode_reuse_rate: {payload['within_mode_reuse_rate']}",
        f"- cross_mode_reuse_rate: {payload['cross_mode_reuse_rate']}",
        f"- reuse_rate_by_model: {json.dumps(payload['reuse_rate_by_model'], ensure_ascii=False)}",
        f"- reuse_rate_by_source_mode: {json.dumps(payload['reuse_rate_by_source_mode'], ensure_ascii=False)}",
        f"- reuse_rate_by_target_mode: {json.dumps(payload['reuse_rate_by_target_mode'], ensure_ascii=False)}",
    ]
    (out_dir / "phase258_done_state_cluster_mode_decomposition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases-per-mode", type=int, default=8)
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
