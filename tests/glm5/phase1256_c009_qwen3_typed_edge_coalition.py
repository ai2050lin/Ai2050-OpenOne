#!/usr/bin/env python3
"""Phase1256: one-shot Qwen3 FP16 typed-edge coalition external validity."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from model_utils import MODEL_CONFIGS, get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1256
CONTRACT_ID = "EXP-C009-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1256_c009_qwen3_typed_edge_coalition_audit.py"
OUT = ROOT / "tests/glm5/result/phase1256_c009_qwen3_typed_edge_coalition"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_worlds.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
DETAILS = OUT / "raw/coalition_result.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/qwen_edge_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

MODEL_PATH = Path(MODEL_CONFIGS["qwen3"]["path"])
LAYERS = 36
NAMES = ("Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry")
VALUES = ("red", "blue", "green", "black", "white", "yellow", "purple", "orange")
WORLD_COUNTS = {"discovery": 32, "selection": 32, "confirmation": 64}
WORLD_SEED = 1_256_900_001
PREFIX_SIZES = (1, 2, 4, 8, 12)
THRESHOLDS = {
    "candidate_finite_fraction_min": 1.0,
    "panel_accuracy_min": 0.90,
    "target_effect_norm_min": 1.0,
    "correct_cosine_min": 0.80,
    "correct_relative_error_max": 0.75,
    "correct_projection_min": 0.45,
    "wrong_cosine_max": 0.65,
    "identity_cosine_margin_min": 0.20,
    "null_effect_fraction_max": 0.20,
    "block_remaining_fraction_max": 0.50,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            output.update(chunk)
    return output.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt(name_a: str, value_a: str, name_b: str, value_b: str) -> str:
    return (
        f"Use only this registry. {name_a} has color {value_a}. "
        f"{name_b} has color {value_b}. Question: What color does {name_a} have? Answer:"
    )


def make_worlds() -> list[dict[str, Any]]:
    rng = np.random.default_rng(WORLD_SEED)
    partitions = [name for name, count in WORLD_COUNTS.items() for _ in range(count)]
    rows: list[dict[str, Any]] = []
    for index, partition in enumerate(partitions):
        names = rng.choice(NAMES, 2, replace=False).tolist()
        values = rng.choice(VALUES, 4, replace=False).tolist()
        base, target, wrong, null_value = values
        panels = {
            "base": prompt(names[0], base, names[1], null_value),
            "target": prompt(names[0], target, names[1], null_value),
            "wrong": prompt(names[0], wrong, names[1], null_value),
            "null": prompt(names[0], base, names[1], target),
        }
        row = {
            "row_id": f"g{index:03d}",
            "partition": partition,
            "names": names,
            "values": {"base": base, "target": target, "wrong": wrong, "null": base},
            "panels": panels,
        }
        row["row_digest"] = digest(row)
        rows.append(row)
    return rows


def component_ids() -> list[str]:
    return [f"L{layer:02d}.{role}" for layer in range(LAYERS) for role in ("q", "ov", "mlp")]


def protocol_payload(rows: list[dict[str, Any]], token_audit: dict[str, Any]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "schema_version": "phase1256.c009.qwen_typed_edge.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "qwen3_single_model_typed_edge_coalition_external_validity",
        "question": "Does one Qwen3 FP16 scope contain a sparse layer-typed Q/OV/MLP answer-boundary coalition that specifically rescues and blocks an unseen object-value counterfactual response?",
        "model": {"name": "qwen3", "path": str(MODEL_PATH), "layers": LAYERS, "precision": "fp16_cuda_no_quantization"},
        "partitions": WORLD_COUNTS,
        "row_count": len(rows),
        "world_digest": digest([{key: row[key] for key in ("row_id", "partition", "row_digest")} for row in rows]),
        "token_audit": token_audit,
        "component_ontology": {
            "q": "replacement of the answer-boundary q_proj output at one layer; query-side address control",
            "ov": "replacement of the answer-boundary attention o_proj input at one layer; aggregated value payload",
            "mlp": "replacement of the answer-boundary MLP output residual write at one layer",
            "component_count": len(component_ids()),
        },
        "selection": {
            "discovery": "single-component correct-rescue relative-error ranking",
            "prefix_sizes": list(PREFIX_SIZES),
            "selection_objective": "correct cosine - correct relative error - null fraction - positive wrong cosine - block remaining - 0.01*size",
            "confirmation": "frozen component identities and prefix size",
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_gpu_hours": 1.0, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "hard_stops": [
            "Behavior is evaluated on base, target, wrong and null panels before any internal result is interpreted.",
            "Behavior failure denies trace capture and every intervention forward pass.",
            "Discovery uses correct-donor rescue only; wrong identity, matched null and reverse blocking enter only at prefix selection and frozen confirmation.",
            "Discovery ranks only registered layer-typed components; confirmation cannot select anything.",
            "Correct rescue, wrong identity, matched null and reverse blocking are conjunctive.",
            "A pass is Qwen3-only, one artificial English scope and teacher-forced candidate scoring.",
            "No GLM4, DS7B, semantic circuit, unique algorithm, cross-model or new-mathematics claim is authorized.",
            "Failure or pass closes this local campaign; no threshold relaxation or component rescan follows automatically.",
        ],
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def tokenizer_audit(tokenizer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    value_ids = {value: tokenizer.encode(" " + value, add_special_tokens=False) for value in VALUES}
    name_ids = {name: tokenizer.encode(" " + name, add_special_tokens=False) for name in NAMES}
    encoded = {
        row["row_id"]: {panel: tokenizer.encode(text, add_special_tokens=False) for panel, text in row["panels"].items()}
        for row in rows
    }
    lengths = {len(ids) for panels in encoded.values() for ids in panels.values()}
    return {
        "value_token_ids": value_ids,
        "name_token_ids": name_ids,
        "all_values_single_token": all(len(ids) == 1 for ids in value_ids.values()),
        "all_names_single_token": all(len(ids) == 1 for ids in name_ids.values()),
        "all_panels_same_length": len(lengths) == 1,
        "input_lengths": sorted(lengths),
        "material_token_digest": digest(encoded),
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    from transformers import AutoTokenizer

    rows = make_worlds()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    audit = tokenizer_audit(tokenizer, rows)
    if not audit["all_values_single_token"] or not audit["all_names_single_token"] or not audit["all_panels_same_length"]:
        raise RuntimeError("tokenizer material qualification failed")
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    atomic_json(PROTOCOL, protocol_payload(rows, audit))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "input_lengths": audit["input_lengths"]}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows, protocol["token_audit"])
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol or source changed")
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("material digest mismatch")
    return protocol, rows


class TraceCapture:
    def __init__(self, layers: list[Any]) -> None:
        self.layers = layers
        self.traces: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    @staticmethod
    def output_tensor(output: Any) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output

    def install(self) -> None:
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(
                lambda _m, _i, output, name=f"L{index:02d}.q": self.traces.__setitem__(
                    name, self.output_tensor(output)[:, -1:, :].detach().cpu()
                )
            ))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda _m, inputs, name=f"L{index:02d}.ov": self.traces.__setitem__(
                    name, inputs[0][:, -1:, :].detach().cpu()
                )
            ))
            self.handles.append(layer.mlp.register_forward_hook(
                lambda _m, _i, output, name=f"L{index:02d}.mlp": self.traces.__setitem__(
                    name, self.output_tensor(output)[:, -1:, :].detach().cpu()
                )
            ))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class CoalitionPatch:
    def __init__(self, layers: list[Any], coalition: list[str], donor: dict[str, torch.Tensor], device: torch.device) -> None:
        self.handles: list[Any] = []
        selected = set(coalition)
        for index, layer in enumerate(layers):
            q_name = f"L{index:02d}.q"
            if q_name in selected:
                value = donor[q_name].to(device)
                self.handles.append(layer.self_attn.q_proj.register_forward_hook(
                    lambda _m, _i, output, value=value: self.replace_last(output, value)
                ))
            ov = f"L{index:02d}.ov"
            if ov in selected:
                value = donor[ov].to(device)
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                    lambda _m, inputs, value=value: (self.replace_last(inputs[0], value),)
                ))
            mlp = f"L{index:02d}.mlp"
            if mlp in selected:
                value = donor[mlp].to(device)
                self.handles.append(layer.mlp.register_forward_hook(
                    lambda _m, _i, output, value=value: self.replace_last(output, value)
                ))

    @staticmethod
    def replace_last(output: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        result = output.clone()
        result[:, -1:, :] = value
        return result

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def centered_scores(logits: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    scores = logits[:, -1].float().index_select(-1, candidate_ids)
    return scores - scores.mean(dim=-1, keepdim=True)


def metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = predicted.reshape(-1).double()
    t = target.reshape(-1).double()
    pn = torch.linalg.vector_norm(p)
    tn = torch.linalg.vector_norm(t).clamp_min(1.0e-12)
    return {
        "cosine": float((torch.dot(p, t) / (pn.clamp_min(1.0e-12) * tn)).item()),
        "relative_error": float((torch.linalg.vector_norm(p - t) / tn).item()),
        "projection": float((torch.dot(p, t) / torch.dot(t, t).clamp_min(1.0e-12)).item()),
    }


def run_forward(model, input_ids: torch.Tensor) -> torch.Tensor:
    return model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        logits_to_keep=1,
        return_dict=True,
    ).logits


def run_patched(model, layers: list[Any], input_ids: torch.Tensor, coalition: list[str], donor: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    patch = CoalitionPatch(layers, coalition, donor, device)
    try:
        return run_forward(model, input_ids)
    finally:
        patch.remove()


def subset_trace(trace: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value.index_select(0, indices.cpu()) for name, value in trace.items()}


def evaluate(
    model,
    layers: list[Any],
    ids: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
    traces: dict[str, dict[str, torch.Tensor]],
    indices: torch.Tensor,
    coalition: list[str],
    device: torch.device,
) -> dict[str, Any]:
    index_device = indices.to(device)
    base = scores["base"].index_select(0, index_device)
    target_effect = scores["target"].index_select(0, index_device) - base

    def patch(receiver: str, donor: str) -> torch.Tensor:
        donor_trace = subset_trace(traces[donor], indices)
        logits = run_patched(model, layers, ids[receiver].index_select(0, index_device), coalition, donor_trace, device)
        return centered_scores(logits, CANDIDATE_IDS.to(device))

    correct = patch("base", "target") - base
    wrong = patch("base", "wrong") - base
    null = patch("base", "null") - base
    blocked = patch("target", "base") - base
    correct_m = metrics(correct, target_effect)
    wrong_m = metrics(wrong, target_effect)
    norm = torch.linalg.vector_norm(target_effect.double()).clamp_min(1.0e-12)
    return {
        "correct": correct_m,
        "wrong": wrong_m,
        "identity_cosine_margin": correct_m["cosine"] - wrong_m["cosine"],
        "null_effect_fraction": float((torch.linalg.vector_norm(null.double()) / norm).item()),
        "block_remaining_fraction": float((torch.linalg.vector_norm(blocked.double()) / norm).item()),
    }


def evaluate_correct_only(
    model,
    layers: list[Any],
    ids: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
    traces: dict[str, dict[str, torch.Tensor]],
    indices: torch.Tensor,
    coalition: list[str],
    device: torch.device,
) -> dict[str, float]:
    index_device = indices.to(device)
    base = scores["base"].index_select(0, index_device)
    target_effect = scores["target"].index_select(0, index_device) - base
    donor_trace = subset_trace(traces["target"], indices)
    logits = run_patched(
        model,
        layers,
        ids["base"].index_select(0, index_device),
        coalition,
        donor_trace,
        device,
    )
    correct = centered_scores(logits, CANDIDATE_IDS.to(device)) - base
    return metrics(correct, target_effect)


def objective(value: dict[str, Any], size: int) -> float:
    return (
        value["correct"]["cosine"] - value["correct"]["relative_error"]
        - value["null_effect_fraction"] - max(0.0, value["wrong"]["cosine"])
        - value["block_remaining_fraction"] - 0.01 * size
    )


def passes(value: dict[str, Any]) -> bool:
    return (
        value["correct"]["cosine"] >= THRESHOLDS["correct_cosine_min"]
        and value["correct"]["relative_error"] <= THRESHOLDS["correct_relative_error_max"]
        and value["correct"]["projection"] >= THRESHOLDS["correct_projection_min"]
        and value["wrong"]["cosine"] <= THRESHOLDS["wrong_cosine_max"]
        and value["identity_cosine_margin"] >= THRESHOLDS["identity_cosine_margin_min"]
        and value["null_effect_fraction"] <= THRESHOLDS["null_effect_fraction_max"]
        and value["block_remaining_fraction"] <= THRESHOLDS["block_remaining_fraction_max"]
    )


CANDIDATE_IDS = torch.empty(0, dtype=torch.long)


def run() -> None:
    global CANDIDATE_IDS
    if COMPLETE.exists():
        raise RuntimeError("formal completion marker exists")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("preaudit not passed")
    protocol, rows = verify_protocol()
    started = time.perf_counter()
    model = None
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if device.type != "cuda" or precision["has_quantized_modules"] or precision["has_bf16_parameters"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError("Qwen3 FP16 qualification failed")
        layers = get_layers(model)
        if len(layers) != LAYERS:
            raise RuntimeError("Qwen3 layer count drift")
        CANDIDATE_IDS = torch.tensor([protocol["token_audit"]["value_token_ids"][value][0] for value in VALUES], dtype=torch.long)
        ids = {
            panel: torch.tensor([tokenizer.encode(row["panels"][panel], add_special_tokens=False) for row in rows], dtype=torch.long, device=device)
            for panel in ("base", "target", "wrong", "null")
        }
        scores: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for panel in ids:
                scores[panel] = centered_scores(run_forward(model, ids[panel]), CANDIDATE_IDS.to(device))
        expected = {
            panel: torch.tensor([VALUES.index(row["values"][panel]) for row in rows], device=device)
            for panel in ids
        }
        panel_accuracy = {
            panel: float((torch.argmax(scores[panel], dim=-1) == expected[panel]).float().mean().item()) for panel in ids
        }
        finite_fraction = float(np.mean([torch.isfinite(scores[panel]).all(dim=-1).float().mean().item() for panel in ids]))
        behavior_passed = finite_fraction >= THRESHOLDS["candidate_finite_fraction_min"] and min(panel_accuracy.values()) >= THRESHOLDS["panel_accuracy_min"]
        result: dict[str, Any] = {
            "behavior": {"panel_accuracy": panel_accuracy, "candidate_finite_fraction": finite_fraction, "passed": behavior_passed},
            "precision_audit": precision,
            "placement": placement,
            "trace_component_count": len(component_ids()),
            "traces_captured": False,
        }
        if behavior_passed:
            traces: dict[str, dict[str, torch.Tensor]] = {}
            capture = TraceCapture(layers)
            capture.install()
            try:
                with torch.inference_mode():
                    for panel in ids:
                        capture.traces = {}
                        run_forward(model, ids[panel])
                        if set(capture.traces) != set(component_ids()):
                            raise RuntimeError("trace component coverage drift")
                        traces[panel] = {name: value.clone() for name, value in capture.traces.items()}
            finally:
                capture.remove()
            result["traces_captured"] = True
            partitions = {
                name: torch.tensor([index for index, row in enumerate(rows) if row["partition"] == name], dtype=torch.long)
                for name in WORLD_COUNTS
            }
            discovery_scores: list[tuple[float, str]] = []
            with torch.inference_mode():
                for number, component in enumerate(component_ids(), start=1):
                    value = evaluate_correct_only(
                        model, layers, ids, scores, traces, partitions["discovery"], [component], device
                    )
                    discovery_scores.append((-value["relative_error"], component))
                    if number % 18 == 0:
                        print(canonical_json({"discovery": number, "total": len(component_ids())}), flush=True)
                ranking = [name for _score, name in sorted(discovery_scores, reverse=True)]
                selection: list[dict[str, Any]] = []
                for size in PREFIX_SIZES:
                    coalition = ranking[:size]
                    value = evaluate(model, layers, ids, scores, traces, partitions["selection"], coalition, device)
                    selection.append({"size": size, "coalition": coalition, "metrics": value, "objective": objective(value, size)})
                chosen = max(selection, key=lambda row: (row["objective"], -row["size"]))
                confirmation = evaluate(model, layers, ids, scores, traces, partitions["confirmation"], chosen["coalition"], device)
            confirmation_index = partitions["confirmation"].to(device)
            target_norm = float(torch.linalg.vector_norm(
                (scores["target"].index_select(0, confirmation_index) - scores["base"].index_select(0, confirmation_index)).double()
            ).item())
            result.update({
                "discovery_ranking": ranking,
                "selection": selection,
                "selected_components": chosen["coalition"],
                "selected_size": chosen["size"],
                "selected_role_counts": {role: sum(name.endswith("." + role) for name in chosen["coalition"]) for role in ("q", "ov", "mlp")},
                "confirmation": confirmation,
                "target_effect_norm": target_norm,
                "passed": target_norm >= THRESHOLDS["target_effect_norm_min"] and passes(confirmation),
            })
        else:
            result["passed"] = False
        result["created_at_utc"] = utc_now()
        atomic_json(DETAILS, result)
        raw = {
            "phase": PHASE,
            "created_at_utc": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_hours": (time.perf_counter() - started) / 3600.0,
            "details_sha256": file_sha256(DETAILS),
            "run_digest": digest(result),
        }
        atomic_json(RAW, raw)
        marker = {"phase": PHASE, "status": "formal_run_complete", "raw_sha256": file_sha256(RAW), "details_sha256": file_sha256(DETAILS), "run_digest": raw["run_digest"]}
        marker["marker_digest"] = digest(marker)
        atomic_json(COMPLETE, marker)
        print(canonical_json({"status": "formal_run_complete", "behavior": behavior_passed, "passed": result["passed"]}))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze() -> None:
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    protocol, _ = verify_protocol()
    raw = read_json(RAW)
    marker = read_json(COMPLETE)
    result = read_json(DETAILS)
    if marker["raw_sha256"] != file_sha256(RAW) or raw["details_sha256"] != file_sha256(DETAILS):
        raise RuntimeError("artifact hash mismatch")
    verdict = "qwen3_typed_edge_coalition_confirmed" if result["passed"] else "qwen3_typed_edge_coalition_not_confirmed"
    analysis = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "verdict": verdict,
        "behavior": result["behavior"],
        "selected_components": result.get("selected_components", []),
        "selected_role_counts": result.get("selected_role_counts", {}),
        "confirmation": result.get("confirmation"),
        "authorization": {"glm4_or_ds7b": False, "semantic_mechanism_claim": False, "cross_model_claim": False, "new_mathematics": False, "automatic_next_phase": False},
        "scope": "Qwen3-4B FP16, one artificial English object-color binding interface, teacher-forced candidate scoring",
    }
    analysis["analysis_digest"] = digest(analysis)
    atomic_json(ANALYSIS, analysis)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "behavior": result["behavior"],
        "confirmation": result.get("confirmation"),
        "selected_components": result.get("selected_components", []),
        "selected_role_counts": result.get("selected_role_counts", {}),
        "authorization": analysis["authorization"],
        "artifact_hashes": {"protocol": file_sha256(PROTOCOL), "material": file_sha256(MATERIAL), "environment": file_sha256(ENVIRONMENT), "preaudit": file_sha256(PREAUDIT), "raw": file_sha256(RAW), "details": file_sha256(DETAILS), "complete": file_sha256(COMPLETE), "analysis": file_sha256(ANALYSIS)},
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"verdict": verdict, "behavior": result["behavior"], "confirmation": result.get("confirmation")}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run", "analyze"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        run()
    else:
        analyze()


if __name__ == "__main__":
    main()
