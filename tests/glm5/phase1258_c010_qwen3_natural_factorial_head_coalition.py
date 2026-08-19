#!/usr/bin/env python3
"""Phase1258: Qwen3 natural-factorial head coalition external validity.

The experiment uses fresh naturalized object-attribute contexts and a frozen
2x2 queried-value x unqueried-value factorial. It screens every Qwen3 layer and
query head at q_proj and o_proj boundaries, plus each layer MLP write, without
using Phase1256's selected layers. Discovery uses target donors only; matched
null, wrong identity, conditional robustness and reverse blocking enter during
selection and frozen confirmation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from model_utils import MODEL_CONFIGS, get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1258
CONTRACT_ID = "EXP-C010-WP02-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1258_c010_qwen3_natural_factorial_head_coalition_audit.py"
PHASE1257_FINAL = ROOT / "tests/glm5/result/phase1257_c010_factorial_head_instrument_calibration/analysis/final.json"
PHASE1257_AUDIT = ROOT / "tests/glm5/result/phase1257_c010_factorial_head_instrument_calibration/audit/independent_final_audit.json"
OUT = ROOT / "tests/glm5/result/phase1258_c010_qwen3_natural_factorial_head_coalition"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_natural_factorial_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
DETAILS = OUT / "raw/head_coalition_result.json"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/natural_factorial_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

MODEL_PATH = Path(MODEL_CONFIGS["qwen3"]["path"])
LAYERS = 36
QUERY_HEADS = 32
HEAD_DIM = 128
VALUES = ("red", "blue", "green", "black", "white", "yellow", "purple", "orange")
NAMES = ("Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry")
PANELS = ("base", "target", "wrong", "null", "joint")
WORLD_COUNTS = {"discovery": 64, "selection": 64, "confirmation": 128}
PARTITION_SEEDS = {"discovery": 1_258_101, "selection": 1_258_201, "confirmation": 1_258_301}
TEMPLATE_FAMILIES = {
    "discovery": (
        "Read the rehearsal report and answer with the stated color. Report: {query}'s scarf was {qv}; {other}'s scarf was {uv}. Question: What color was {query}'s scarf? Answer:",
        "Read the classroom note and answer with the stated color. Note: {query}'s badge was {qv}; {other}'s badge was {uv}. Question: What color was {query}'s badge? Answer:",
        "Read the portrait record and answer with the stated color. Record: {query}'s ribbon was {qv}; {other}'s ribbon was {uv}. Question: What color was {query}'s ribbon? Answer:",
    ),
    "selection": (
        "Read the sorting log and answer with the stated color. Log: {query}'s card was {qv}; {other}'s card was {uv}. Question: What color was {query}'s card? Answer:",
        "Read the workshop entry and answer with the stated color. Entry: {query}'s label was {qv}; {other}'s label was {uv}. Question: What color was {query}'s label? Answer:",
    ),
    "confirmation": (
        "Read the event log and answer with the stated color. Log: {query}'s flag was {qv}; {other}'s flag was {uv}. Question: What color was {query}'s flag? Answer:",
        "Read the design brief and answer with the stated color. Brief: {query}'s icon was {qv}; {other}'s icon was {uv}. Question: What color was {query}'s icon? Answer:",
        "Read the market note and answer with the stated color. Note: {query}'s cup was {qv}; {other}'s cup was {uv}. Question: What color was {query}'s cup? Answer:",
    ),
}
SHORTLIST_QUOTAS = {"q": 12, "ov": 24, "mlp": 12}
PREFIX_SIZES = (1, 2, 4, 8, 12)
THRESHOLDS = {
    "candidate_finite_fraction_min": 1.0,
    "behavior_cell_accuracy_min": 0.85,
    "target_effect_norm_min": 1.0,
    "correct_cosine_min": 0.80,
    "correct_relative_error_max": 0.75,
    "correct_projection_min": 0.45,
    "direct_correct_ratio_min": 0.50,
    "direct_wrong_ratio_max": 0.10,
    "direct_identity_separation_min": 0.60,
    "null_parallel_abs_max": 0.12,
    "null_orthogonal_max": 0.20,
    "null_total_max": 0.23,
    "conditional_cosine_min": 0.75,
    "conditional_relative_error_max": 0.80,
    "block_remaining_fraction_max": 0.50,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def make_worlds(counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    counts = counts or WORLD_COUNTS
    rows: list[dict[str, Any]] = []
    for partition, count in counts.items():
        rng = np.random.default_rng(PARTITION_SEEDS[partition])
        templates = TEMPLATE_FAMILIES[partition]
        for index in range(count):
            query_index = int(rng.integers(0, len(NAMES)))
            other_index = int(rng.integers(0, len(NAMES) - 1))
            if other_index >= query_index:
                other_index += 1
            chosen = rng.choice(len(VALUES), size=5, replace=False).tolist()
            base, target, wrong, null_base, null_alt = [VALUES[value] for value in chosen]
            query = NAMES[query_index]
            other = NAMES[other_index]
            template_slot = index % len(templates)
            template = templates[template_slot]

            def render(qv: str, uv: str) -> str:
                return template.format(query=query, other=other, qv=qv, uv=uv)

            panels = {
                "base": render(base, null_base),
                "target": render(target, null_base),
                "wrong": render(wrong, null_base),
                "null": render(base, null_alt),
                "joint": render(target, null_alt),
            }
            row = {
                "row_id": f"{partition}-{index:04d}",
                "partition": partition,
                "template_slot": template_slot,
                "query_entity": query,
                "unqueried_entity": other,
                "values": {"base": base, "target": target, "wrong": wrong, "null_base": null_base, "null_alt": null_alt},
                "expected": {"base": base, "target": target, "wrong": wrong, "null": base, "joint": target},
                "panels": panels,
            }
            row["row_digest"] = digest(row)
            rows.append(row)
    return rows


def tokenizer_audit(tokenizer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    value_ids = {value: tokenizer.encode(" " + value, add_special_tokens=False) for value in VALUES}
    name_ids = {name: tokenizer.encode(" " + name, add_special_tokens=False) for name in NAMES}
    encoded = {
        row["row_id"]: {panel: tokenizer.encode(text, add_special_tokens=False) for panel, text in row["panels"].items()}
        for row in rows
    }
    row_equal_lengths = all(len({len(value) for value in panels.values()}) == 1 for panels in encoded.values())
    factorial_token_differences = []
    for panels in encoded.values():
        def differences(left: str, right: str) -> int:
            return sum(a != b for a, b in zip(panels[left], panels[right])) if len(panels[left]) == len(panels[right]) else 10**9
        factorial_token_differences.append({
            "base_target": differences("base", "target"),
            "base_wrong": differences("base", "wrong"),
            "base_null": differences("base", "null"),
            "null_joint": differences("null", "joint"),
            "target_joint": differences("target", "joint"),
        })
    differences_exact = all(set(item.values()) == {1} for item in factorial_token_differences)
    template_sets = {partition: list(range(len(TEMPLATE_FAMILIES[partition]))) for partition in WORLD_COUNTS}
    lengths = [len(value) for panels in encoded.values() for value in panels.values()]
    return {
        "value_token_ids": value_ids,
        "name_token_ids": name_ids,
        "all_values_single_token": all(len(value) == 1 for value in value_ids.values()),
        "all_names_single_token": all(len(value) == 1 for value in name_ids.values()),
        "within_world_panel_lengths_equal": row_equal_lengths,
        "factorial_pairs_differ_by_one_token": differences_exact,
        "min_input_length": min(lengths),
        "max_input_length": max(lengths),
        "template_family_counts": {key: len(value) for key, value in TEMPLATE_FAMILIES.items()},
        "template_text_disjoint": len({text for values in TEMPLATE_FAMILIES.values() for text in values}) == sum(len(values) for values in TEMPLATE_FAMILIES.values()),
        "material_token_digest": digest(encoded),
        "template_sets": template_sets,
    }


def all_component_ids() -> list[str]:
    values = []
    for layer in range(LAYERS):
        values.extend(f"L{layer:02d}.qH{head:02d}" for head in range(QUERY_HEADS))
        values.extend(f"L{layer:02d}.ovH{head:02d}" for head in range(QUERY_HEADS))
        values.append(f"L{layer:02d}.mlp")
    return values


def protocol_payload(rows: list[dict[str, Any]], token_audit: dict[str, Any]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "schema_version": "phase1258.c010.qwen_natural_factorial_head.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "qwen3_single_model_natural_factorial_head_coalition_external_validity",
        "question": "Does a newly selected Qwen3 head/MLP coalition specifically transport a queried value across held-out natural templates while rejecting wrong identity and decomposed factorial nulls?",
        "model": {
            "name": "qwen3",
            "path": str(MODEL_PATH),
            "precision": "fp16_cuda_no_quantization",
            "layers": LAYERS,
            "query_heads": QUERY_HEADS,
            "head_dim": HEAD_DIM,
        },
        "dependencies": {
            "phase1257_final": file_sha256(PHASE1257_FINAL),
            "phase1257_audit": file_sha256(PHASE1257_AUDIT),
        },
        "partitions": WORLD_COUNTS,
        "row_count": len(rows),
        "template_families": {key: list(value) for key, value in TEMPLATE_FAMILIES.items()},
        "world_digest": digest([{key: row[key] for key in ("row_id", "partition", "template_slot", "row_digest")} for row in rows]),
        "token_audit": token_audit,
        "component_ontology": {
            "q": "one q_proj query-head slice at the answer boundary before q_norm/RoPE",
            "ov": "one query-head slice of the o_proj input at the answer boundary",
            "mlp": "one layer MLP residual write at the answer boundary",
            "component_count": len(all_component_ids()),
            "not_included": "No Phase1256 selected layer list, no neuron/channel scan, no v-source role chosen by experimenter.",
        },
        "selection": {
            "observational_shortlist": SHORTLIST_QUOTAS,
            "observational_signal": "target-minus-base activation RMS only; null and wrong panels are forbidden",
            "causal_discovery": "single-component target-donor rescue among the frozen balanced shortlist",
            "prefix_sizes": list(PREFIX_SIZES),
            "selection_objective": "correct rescue + direct identity - decomposed null - conditional error - blocking - size",
            "confirmation": "frozen identities and prefix size on disjoint templates/worlds",
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_gpu_hours": 1.0, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "hard_stops": [
            "Behavior is evaluated for every partition x panel cell before hooks or interventions are interpreted.",
            "Discovery and selection templates are disjoint from confirmation templates.",
            "The Phase1256 eight-layer coalition is neither seeded nor rescanned; all 2,340 new head/MLP components enter observational screening symmetrically.",
            "Discovery uses target-minus-base observations and correct donors only; wrong, null, joint and blocking cannot choose candidates.",
            "Wrong identity is adjudicated by direct target-versus-wrong candidate contrast, not shared-base cosine alone.",
            "Matched null is conjunctively tested by target-parallel, target-orthogonal and total response fractions.",
            "A pass is Qwen3-only, naturalized synthetic English, teacher-forced eight-candidate scoring and support-matched activation replacement.",
            "No GLM4, DS7B, full natural-language mechanism, unique algorithm, on-manifold proof, cross-model or new-mathematics claim is authorized.",
            "Pass or failure closes C010; no threshold relaxation, template repair or component rescan follows automatically.",
        ],
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    from transformers import AutoTokenizer

    calibration = read_json(PHASE1257_FINAL)
    audit = read_json(PHASE1257_AUDIT)
    if calibration.get("verdict") != "factorial_head_instrument_calibrated" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1257 authorization missing")
    rows = make_worlds()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    token_audit = tokenizer_audit(tokenizer, rows)
    required = (
        token_audit["all_values_single_token"],
        token_audit["all_names_single_token"],
        token_audit["within_world_panel_lengths_equal"],
        token_audit["factorial_pairs_differ_by_one_token"],
        token_audit["template_text_disjoint"],
    )
    if not all(required):
        raise RuntimeError(f"material token qualification failed: {token_audit}")
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows, token_audit))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "token_lengths": [token_audit["min_input_length"], token_audit["max_input_length"]]}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows, protocol["token_audit"])
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["dependencies"] != expected["dependencies"]:
        raise RuntimeError("source or dependency hash drift")
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest drift")
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("material digest drift")
    return protocol, rows


def encode_panel(tokenizer, rows: list[dict[str, Any]], panel: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [tokenizer.encode(row["panels"][panel], add_special_tokens=False) for row in rows]
    maximum = max(map(len, encoded))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(rows), maximum), int(pad), dtype=torch.long)
    mask = torch.zeros((len(rows), maximum), dtype=torch.long)
    for index, values in enumerate(encoded):
        ids[index, -len(values):] = torch.tensor(values, dtype=torch.long)
        mask[index, -len(values):] = 1
    return ids.to(device), mask.to(device)


def run_forward(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1, return_dict=True).logits


def centered_scores(logits: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    scores = logits[:, -1].float().index_select(-1, candidate_ids)
    return scores - scores.mean(dim=-1, keepdim=True)


class HeadTraceCapture:
    def __init__(self, layers: list[Any]) -> None:
        self.layers = layers
        self.handles: list[Any] = []
        self.reset()

    def reset(self) -> None:
        self.q: list[torch.Tensor | None] = [None] * len(self.layers)
        self.ov: list[torch.Tensor | None] = [None] * len(self.layers)
        self.mlp: list[torch.Tensor | None] = [None] * len(self.layers)

    @staticmethod
    def tensor(output: Any) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output

    def install(self) -> None:
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(
                lambda _module, _inputs, output, index=index: self._capture_q(index, output)
            ))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda _module, inputs, index=index: self._capture_ov(index, inputs[0])
            ))
            self.handles.append(layer.mlp.register_forward_hook(
                lambda _module, _inputs, output, index=index: self._capture_mlp(index, output)
            ))

    def _capture_q(self, index: int, output: Any) -> None:
        value = self.tensor(output)[:, -1, :].reshape(-1, QUERY_HEADS, HEAD_DIM)
        self.q[index] = value.detach().cpu()

    def _capture_ov(self, index: int, value: torch.Tensor) -> None:
        self.ov[index] = value[:, -1, :].reshape(-1, QUERY_HEADS, HEAD_DIM).detach().cpu()

    def _capture_mlp(self, index: int, output: Any) -> None:
        self.mlp[index] = self.tensor(output)[:, -1, :].detach().cpu()

    def result(self) -> dict[str, torch.Tensor]:
        if any(value is None for value in self.q + self.ov + self.mlp):
            raise RuntimeError("trace coverage incomplete")
        return {
            "q": torch.stack([value for value in self.q if value is not None], dim=1),
            "ov": torch.stack([value for value in self.ov if value is not None], dim=1),
            "mlp": torch.stack([value for value in self.mlp if value is not None], dim=1),
        }

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


COMPONENT_RE = re.compile(r"^L(?P<layer>\d{2})\.(?P<role>q|ov)(?:H(?P<head>\d{2}))$|^L(?P<mlp_layer>\d{2})\.mlp$")


def parse_component(name: str) -> tuple[int, str, int | None]:
    match = COMPONENT_RE.match(name)
    if not match:
        raise ValueError(name)
    if match.group("mlp_layer") is not None:
        return int(match.group("mlp_layer")), "mlp", None
    return int(match.group("layer")), str(match.group("role")), int(match.group("head"))


class HeadCoalitionPatch:
    def __init__(self, layers: list[Any], coalition: list[str], donor: dict[str, torch.Tensor], device: torch.device) -> None:
        grouped: dict[tuple[int, str], list[int | None]] = {}
        for name in coalition:
            layer, role, head = parse_component(name)
            grouped.setdefault((layer, role), []).append(head)
        self.handles: list[Any] = []
        for (layer_index, role), heads in grouped.items():
            layer = layers[layer_index]
            if role == "q":
                selected = [int(head) for head in heads if head is not None]
                values = donor["q"][:, layer_index, selected, :].to(device)
                self.handles.append(layer.self_attn.q_proj.register_forward_hook(
                    lambda _module, _inputs, output, selected=selected, values=values: self._replace_heads(output, selected, values)
                ))
            elif role == "ov":
                selected = [int(head) for head in heads if head is not None]
                values = donor["ov"][:, layer_index, selected, :].to(device)
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                    lambda _module, inputs, selected=selected, values=values: (self._replace_heads(inputs[0], selected, values),)
                ))
            else:
                value = donor["mlp"][:, layer_index, :].to(device)
                self.handles.append(layer.mlp.register_forward_hook(
                    lambda _module, _inputs, output, value=value: self._replace_last(output, value)
                ))

    @staticmethod
    def _replace_heads(output: torch.Tensor, heads: list[int], value: torch.Tensor) -> torch.Tensor:
        result = output.clone()
        shaped = result[:, -1, :].reshape(result.shape[0], QUERY_HEADS, HEAD_DIM)
        shaped[:, heads, :] = value
        return result

    @staticmethod
    def _replace_last(output: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        result = output.clone()
        result[:, -1, :] = value
        return result

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def subset_trace(trace: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    cpu = indices.cpu()
    return {role: value.index_select(0, cpu) for role, value in trace.items()}


def run_patched(model, layers: list[Any], ids: torch.Tensor, mask: torch.Tensor, coalition: list[str], donor: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    patch = HeadCoalitionPatch(layers, coalition, donor, device)
    try:
        return run_forward(model, ids, mask)
    finally:
        patch.remove()


def vector_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = predicted.reshape(-1).double()
    t = target.reshape(-1).double()
    pn = torch.linalg.vector_norm(p).clamp_min(1.0e-12)
    tn = torch.linalg.vector_norm(t).clamp_min(1.0e-12)
    return {
        "cosine": float((torch.dot(p, t) / (pn * tn)).item()),
        "relative_error": float((torch.linalg.vector_norm(p - t) / tn).item()),
        "projection": float((torch.dot(p, t) / torch.dot(t, t).clamp_min(1.0e-12)).item()),
    }


def direct_contrast_ratio(response: torch.Tensor, target: torch.Tensor, rows: list[dict[str, Any]], indices: torch.Tensor) -> float:
    index_list = indices.tolist()
    target_slots = torch.tensor([VALUES.index(rows[index]["values"]["target"]) for index in index_list], device=response.device)
    wrong_slots = torch.tensor([VALUES.index(rows[index]["values"]["wrong"]) for index in index_list], device=response.device)
    row_slots = torch.arange(len(index_list), device=response.device)
    numerator = (response[row_slots, target_slots] - response[row_slots, wrong_slots]).double().sum()
    denominator = (target[row_slots, target_slots] - target[row_slots, wrong_slots]).double().sum().clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def null_decomposition(null: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    n = null.reshape(-1).double()
    t = target.reshape(-1).double()
    t2 = torch.dot(t, t).clamp_min(1.0e-12)
    alpha = torch.dot(n, t) / t2
    orthogonal = n - alpha * t
    target_norm = torch.sqrt(t2)
    return {
        "parallel_fraction": float(alpha.item()),
        "orthogonal_fraction": float((torch.linalg.vector_norm(orthogonal) / target_norm).item()),
        "total_fraction": float((torch.linalg.vector_norm(n) / target_norm).item()),
    }


def evaluate(
    model,
    layers: list[Any],
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor]],
    scores: dict[str, torch.Tensor],
    traces: dict[str, dict[str, torch.Tensor]],
    rows: list[dict[str, Any]],
    indices: torch.Tensor,
    coalition: list[str],
    candidate_ids: torch.Tensor,
    device: torch.device,
    store_vectors: bool = False,
) -> dict[str, Any]:
    index_device = indices.to(device)
    base = scores["base"].index_select(0, index_device)
    target_effect = scores["target"].index_select(0, index_device) - base
    conditional_target = scores["joint"].index_select(0, index_device) - scores["null"].index_select(0, index_device)

    def patch(receiver: str, donor: str) -> torch.Tensor:
        ids, mask = encoded[receiver]
        donor_trace = subset_trace(traces[donor], indices)
        logits = run_patched(
            model,
            layers,
            ids.index_select(0, index_device),
            mask.index_select(0, index_device),
            coalition,
            donor_trace,
            device,
        )
        return centered_scores(logits, candidate_ids)

    correct = patch("base", "target") - base
    wrong = patch("base", "wrong") - base
    null = patch("base", "null") - base
    blocked = patch("target", "base") - base
    conditional = patch("null", "joint") - scores["null"].index_select(0, index_device)
    correct_metrics = vector_metrics(correct, target_effect)
    conditional_metrics = vector_metrics(conditional, conditional_target)
    correct_direct = direct_contrast_ratio(correct, target_effect, rows, indices)
    wrong_direct = direct_contrast_ratio(wrong, target_effect, rows, indices)
    decomposition = null_decomposition(null, target_effect)
    target_norm = torch.linalg.vector_norm(target_effect.double()).clamp_min(1.0e-12)
    result: dict[str, Any] = {
        "correct": correct_metrics,
        "wrong_cosine_diagnostic": vector_metrics(wrong, target_effect)["cosine"],
        "direct_correct_ratio": correct_direct,
        "direct_wrong_ratio": wrong_direct,
        "direct_identity_separation": correct_direct - wrong_direct,
        "null": decomposition,
        "conditional": conditional_metrics,
        "block_remaining_fraction": float((torch.linalg.vector_norm(blocked.double()) / target_norm).item()),
    }
    if store_vectors:
        result["response_tensor"] = {
            "row_ids": [rows[index]["row_id"] for index in indices.tolist()],
            "target": target_effect.detach().cpu().tolist(),
            "correct": correct.detach().cpu().tolist(),
            "wrong": wrong.detach().cpu().tolist(),
            "null": null.detach().cpu().tolist(),
            "conditional_target": conditional_target.detach().cpu().tolist(),
            "conditional": conditional.detach().cpu().tolist(),
            "blocked": blocked.detach().cpu().tolist(),
        }
    return result


def evaluate_correct_only(model, layers, encoded, scores, traces, indices, coalition, candidate_ids, device) -> dict[str, float]:
    index_device = indices.to(device)
    base = scores["base"].index_select(0, index_device)
    target = scores["target"].index_select(0, index_device) - base
    ids, mask = encoded["base"]
    logits = run_patched(
        model,
        layers,
        ids.index_select(0, index_device),
        mask.index_select(0, index_device),
        coalition,
        subset_trace(traces["target"], indices),
        device,
    )
    response = centered_scores(logits, candidate_ids) - base
    return vector_metrics(response, target)


def observational_shortlist(traces: dict[str, dict[str, torch.Tensor]], indices: torch.Tensor) -> tuple[list[str], dict[str, float]]:
    selected: list[str] = []
    score_map: dict[str, float] = {}
    cpu = indices.cpu()
    for role in ("q", "ov"):
        delta = traces["target"][role].index_select(0, cpu).float() - traces["base"][role].index_select(0, cpu).float()
        base = traces["base"][role].index_select(0, cpu).float()
        for layer in range(LAYERS):
            for head in range(QUERY_HEADS):
                name = f"L{layer:02d}.{role}H{head:02d}"
                numerator = torch.linalg.vector_norm(delta[:, layer, head, :])
                denominator = torch.linalg.vector_norm(base[:, layer, head, :]).clamp_min(1.0e-12)
                score_map[name] = float((numerator / denominator).item())
    delta_mlp = traces["target"]["mlp"].index_select(0, cpu).float() - traces["base"]["mlp"].index_select(0, cpu).float()
    base_mlp = traces["base"]["mlp"].index_select(0, cpu).float()
    for layer in range(LAYERS):
        name = f"L{layer:02d}.mlp"
        score_map[name] = float((torch.linalg.vector_norm(delta_mlp[:, layer, :]) /
                                 torch.linalg.vector_norm(base_mlp[:, layer, :]).clamp_min(1.0e-12)).item())
    for role, quota in SHORTLIST_QUOTAS.items():
        candidates = [name for name in score_map if ("." + role) in name]
        selected.extend(sorted(candidates, key=lambda name: (score_map[name], name), reverse=True)[:quota])
    return selected, score_map


def objective(value: dict[str, Any], size: int) -> float:
    return (
        value["correct"]["cosine"]
        - value["correct"]["relative_error"]
        + value["direct_identity_separation"]
        - abs(value["null"]["parallel_fraction"])
        - value["null"]["orthogonal_fraction"]
        - value["conditional"]["relative_error"]
        - value["block_remaining_fraction"]
        - 0.01 * size
    )


def passes(value: dict[str, Any]) -> bool:
    null = value["null"]
    return (
        value["correct"]["cosine"] >= THRESHOLDS["correct_cosine_min"]
        and value["correct"]["relative_error"] <= THRESHOLDS["correct_relative_error_max"]
        and value["correct"]["projection"] >= THRESHOLDS["correct_projection_min"]
        and value["direct_correct_ratio"] >= THRESHOLDS["direct_correct_ratio_min"]
        and value["direct_wrong_ratio"] <= THRESHOLDS["direct_wrong_ratio_max"]
        and value["direct_identity_separation"] >= THRESHOLDS["direct_identity_separation_min"]
        and abs(null["parallel_fraction"]) <= THRESHOLDS["null_parallel_abs_max"]
        and null["orthogonal_fraction"] <= THRESHOLDS["null_orthogonal_max"]
        and null["total_fraction"] <= THRESHOLDS["null_total_max"]
        and value["conditional"]["cosine"] >= THRESHOLDS["conditional_cosine_min"]
        and value["conditional"]["relative_error"] <= THRESHOLDS["conditional_relative_error_max"]
        and value["block_remaining_fraction"] <= THRESHOLDS["block_remaining_fraction_max"]
    )


def behavior_probe() -> None:
    model = None
    try:
        model, tokenizer, device, _placement = load_fp16("qwen3")
        rows = make_worlds({"discovery": 8, "selection": 8, "confirmation": 16})
        candidate_ids = torch.tensor([tokenizer.encode(" " + value, add_special_tokens=False)[0] for value in VALUES], device=device)
        result = {}
        with torch.inference_mode():
            for panel in PANELS:
                ids, mask = encode_panel(tokenizer, rows, panel, device)
                scores = centered_scores(run_forward(model, ids, mask), candidate_ids)
                expected = torch.tensor([VALUES.index(row["expected"][panel]) for row in rows], device=device)
                result[panel] = float((scores.argmax(dim=-1) == expected).float().mean().item())
        print(canonical_json({"probe_panel_accuracy": result, "minimum": min(result.values())}))
    finally:
        if model is not None:
            release_fp16(model)


def run() -> None:
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
            raise RuntimeError("layer count drift")
        config = model.config
        if int(config.num_attention_heads) != QUERY_HEADS or int(config.head_dim) != HEAD_DIM or int(config.num_key_value_heads) != 8:
            raise RuntimeError("Qwen3 head geometry drift")
        candidate_ids = torch.tensor([protocol["token_audit"]["value_token_ids"][value][0] for value in VALUES], dtype=torch.long, device=device)
        encoded = {panel: encode_panel(tokenizer, rows, panel, device) for panel in PANELS}
        scores: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for panel in PANELS:
                ids, mask = encoded[panel]
                scores[panel] = centered_scores(run_forward(model, ids, mask), candidate_ids)

        partitions = {
            name: torch.tensor([index for index, row in enumerate(rows) if row["partition"] == name], dtype=torch.long)
            for name in WORLD_COUNTS
        }
        behavior_cells: dict[str, float] = {}
        for partition, indices in partitions.items():
            index_device = indices.to(device)
            for panel in PANELS:
                expected = torch.tensor([VALUES.index(rows[index]["expected"][panel]) for index in indices.tolist()], device=device)
                behavior_cells[f"{partition}.{panel}"] = float((scores[panel].index_select(0, index_device).argmax(dim=-1) == expected).float().mean().item())
        finite_fraction = float(np.mean([torch.isfinite(value).all(dim=-1).float().mean().item() for value in scores.values()]))
        behavior_passed = finite_fraction >= THRESHOLDS["candidate_finite_fraction_min"] and min(behavior_cells.values()) >= THRESHOLDS["behavior_cell_accuracy_min"]
        result: dict[str, Any] = {
            "behavior": {
                "cell_accuracy": behavior_cells,
                "minimum_cell_accuracy": min(behavior_cells.values()),
                "candidate_finite_fraction": finite_fraction,
                "passed": behavior_passed,
            },
            "precision_audit": precision,
            "placement": placement,
            "registered_component_count": len(all_component_ids()),
            "traces_captured": False,
        }
        if behavior_passed:
            traces: dict[str, dict[str, torch.Tensor]] = {}
            capture = HeadTraceCapture(layers)
            capture.install()
            try:
                with torch.inference_mode():
                    for panel in PANELS:
                        capture.reset()
                        ids, mask = encoded[panel]
                        run_forward(model, ids, mask)
                        traces[panel] = capture.result()
            finally:
                capture.remove()
            result["traces_captured"] = True
            result["trace_shapes"] = {role: list(traces["base"][role].shape) for role in ("q", "ov", "mlp")}
            shortlist, observational_scores = observational_shortlist(traces, partitions["discovery"])
            if len(shortlist) != sum(SHORTLIST_QUOTAS.values()) or len(set(shortlist)) != len(shortlist):
                raise RuntimeError("balanced shortlist drift")
            causal_discovery = []
            with torch.inference_mode():
                for number, component in enumerate(shortlist, start=1):
                    value = evaluate_correct_only(model, layers, encoded, scores, traces, partitions["discovery"], [component], candidate_ids, device)
                    causal_discovery.append({"component": component, "correct": value})
                    if number % 12 == 0:
                        print(canonical_json({"causal_discovery": number, "total": len(shortlist)}), flush=True)
                ranking = [item["component"] for item in sorted(causal_discovery, key=lambda item: (-item["correct"]["relative_error"], item["correct"]["cosine"]), reverse=True)]
                selection = []
                for size in PREFIX_SIZES:
                    coalition = ranking[:size]
                    value = evaluate(model, layers, encoded, scores, traces, rows, partitions["selection"], coalition, candidate_ids, device)
                    selection.append({"size": size, "coalition": coalition, "metrics": value, "objective": objective(value, size)})
                chosen = max(selection, key=lambda item: (item["objective"], -item["size"]))
                confirmation = evaluate(
                    model,
                    layers,
                    encoded,
                    scores,
                    traces,
                    rows,
                    partitions["confirmation"],
                    chosen["coalition"],
                    candidate_ids,
                    device,
                    store_vectors=True,
                )
            confirmation_index = partitions["confirmation"].to(device)
            target_norm = float(torch.linalg.vector_norm((scores["target"].index_select(0, confirmation_index) - scores["base"].index_select(0, confirmation_index)).double()).item())
            role_counts = {role: sum(parse_component(name)[1] == role for name in chosen["coalition"]) for role in ("q", "ov", "mlp")}
            result.update({
                "observational_shortlist": shortlist,
                "observational_shortlist_scores": {name: observational_scores[name] for name in shortlist},
                "causal_discovery": causal_discovery,
                "causal_ranking": ranking,
                "selection": selection,
                "selected_components": chosen["coalition"],
                "selected_size": chosen["size"],
                "selected_role_counts": role_counts,
                "confirmation": confirmation,
                "target_effect_norm": target_norm,
                "passed": target_norm >= THRESHOLDS["target_effect_norm_min"] and passes(confirmation),
            })
        else:
            result["passed"] = False
        result["created_at_utc"] = utc_now()
        atomic_json(DETAILS, result)
        elapsed = time.perf_counter() - started
        summary = {
            "phase": PHASE,
            "created_at_utc": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "details_sha256": file_sha256(DETAILS),
            "run_digest": digest(result),
        }
        atomic_json(SUMMARY, summary)
        marker = {
            "phase": PHASE,
            "status": "formal_run_complete",
            "details_sha256": file_sha256(DETAILS),
            "summary_sha256": file_sha256(SUMMARY),
            "run_digest": summary["run_digest"],
        }
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
    protocol, _rows = verify_protocol()
    result = read_json(DETAILS)
    summary = read_json(SUMMARY)
    marker = read_json(COMPLETE)
    if marker["details_sha256"] != file_sha256(DETAILS) or marker["summary_sha256"] != file_sha256(SUMMARY):
        raise RuntimeError("artifact hash mismatch")
    verdict = "qwen3_natural_factorial_head_coalition_confirmed" if result["passed"] else "qwen3_natural_factorial_head_coalition_not_confirmed"
    authorization = {
        "semantic_mechanism_claim": False,
        "naturalized_factorial_local_claim": bool(result["passed"]),
        "glm4_or_ds7b": False,
        "cross_model_claim": False,
        "new_mathematics": False,
        "automatic_next_phase": False,
    }
    analysis = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "verdict": verdict,
        "behavior": result["behavior"],
        "selected_components": result.get("selected_components", []),
        "selected_role_counts": result.get("selected_role_counts", {}),
        "confirmation": result.get("confirmation"),
        "authorization": authorization,
        "scope": "Qwen3-4B FP16; naturalized synthetic English object-color factorial; held-out template families; teacher-forced eight-candidate scoring; head/MLP support-matched activation replacement.",
    }
    analysis["analysis_digest"] = digest(analysis)
    atomic_json(ANALYSIS, analysis)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "behavior": result["behavior"],
        "selected_components": result.get("selected_components", []),
        "selected_role_counts": result.get("selected_role_counts", {}),
        "confirmation_without_tensor": ({key: value for key, value in result["confirmation"].items() if key != "response_tensor"} if result.get("confirmation") else None),
        "authorization": authorization,
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL),
            "environment": file_sha256(ENVIRONMENT),
            "material": file_sha256(MATERIAL),
            "preaudit": file_sha256(PREAUDIT),
            "details": file_sha256(DETAILS),
            "summary": file_sha256(SUMMARY),
            "complete": file_sha256(COMPLETE),
            "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"verdict": verdict, "behavior": result["behavior"], "confirmation": final["confirmation_without_tensor"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "preregister", "run", "analyze"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "probe":
        behavior_probe()
    elif args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        run()
    else:
        analyze()


if __name__ == "__main__":
    main()
