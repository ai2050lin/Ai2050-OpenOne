#!/usr/bin/env python3
"""C638-C640 individual-donor output-identity equivalence campaign."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase2170_c635_c637_fresh_code_lockbox_campaign as prior

PHASES = {
    "C638": (2173, "individual_donor_functional_equivalence"),
    "C639": (2174, "identity_dose_and_coordinate_partition"),
    "C640": (2175, "identity_equivalence_visual_theory_audit"),
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
        for name, (phase, slug) in PHASES.items()}
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c640_output_identity_equivalence_atlas.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"

CODES = prior.CODES
prior.base.CODES = CODES
prior.base.SYSTEM = prior.SYSTEM
FAMILIES = prior.FAMILIES
LANGUAGES = prior.LANGUAGES
SURFACES = prior.SURFACES
ROLES = prior.ROLES
DIM = prior.DIM
MODEL_BASE = prior.MODEL_BASE
KINDS = {
    "code_x": (0, 1, 1),
    "semantic_x": (1, 0, 1),
    "code_y": (0, 2, 2),
    "semantic_y": (2, 0, 2),
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def begin(name: str, protocol: dict, dependencies: dict) -> Path:
    out = OUTS[name]
    for part in ("protocol", "analysis", "raw", "audit"):
        (out / part).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "dependencies": dependencies,
        "camera": "q32 boundary HiddenState and output only; all 2560 signed coordinates; no Top-K/PCA/attention/MLP/weights/gradients",
        "branch_policy": "a failed family or donor class does not stop other registered classes",
    })
    print(f"=== {name} phase={PHASES[name][0]} ===", flush=True)
    return out


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    result = {"phase": PHASES[name][0], "campaign": name, "status": "closed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
              "headline": headline, "checks": checks,
              "next_authorization": authorization}
    save(OUTS[name] / "analysis/final.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def analyze_discrete_and_logit_signatures(records: list[dict],
                                           donor_ledger: list[dict]) -> tuple[list[dict], dict]:
    signatures = []
    for donor in donor_ledger:
        rows = sorted((row for row in records if row["donor_index"] == donor["donor_index"]),
                      key=lambda row: row["receiver_index"])
        signatures.append({"donor_index": donor["donor_index"], "kind": donor["kind"],
                           "expected_prediction": donor["expected_prediction"],
                           "predictions": [row["prediction"] for row in rows],
                           "expected_signature": [row["expected"] for row in rows],
                           "candidate_score_signature": [row["candidate_scores"] for row in rows],
                           "universal": all(row["expected"] for row in rows)})
    discrete_classes = defaultdict(list)
    for row in signatures:
        key = (row["expected_prediction"], tuple(row["predictions"]))
        discrete_classes[str(key)].append(row["donor_index"])
    continuous = {}
    for expected in (1, 2):
        members = [row for row in signatures if row["expected_prediction"] == expected]
        arrays = np.asarray([row["candidate_score_signature"] for row in members], np.float64)
        center = np.mean(arrays, axis=0)
        deviations = np.sqrt(np.mean((arrays - center) ** 2, axis=(1, 2)))
        code = np.mean(np.asarray([row["candidate_score_signature"] for row in members
                                   if row["kind"].startswith("code_")], np.float64), axis=0)
        semantic = np.mean(np.asarray([row["candidate_score_signature"] for row in members
                                       if row["kind"].startswith("semantic_")], np.float64), axis=0)
        continuous[str(expected)] = {
            "donors": len(members),
            "within_class_logit_rms_median": float(np.median(deviations)),
            "within_class_logit_rms_max": float(np.max(deviations)),
            "code_vs_semantic_centroid_rms": float(np.sqrt(np.mean((code - semantic) ** 2))),
            "exact_continuous_equivalence_claimed": False,
        }
    summary = {"discrete_prediction_classes": len(discrete_classes),
               "discrete_class_sizes": sorted((len(value) for value in discrete_classes.values()), reverse=True),
               "continuous_logit_signature_audit": continuous}
    return signatures, summary


def qualified_slices() -> list[tuple[str, str, str]]:
    slices = prior.final("C635")["headline"]["slices"]
    return [tuple(key.split("|")) for key, value in sorted(slices.items())
            if value["qualified"]]


def compiled_rows() -> dict[str, dict]:
    return {row["case_id"]: row for row in prior.read_rows(prior.compiled_path())}


def capture_case_ids() -> list[str]:
    ids = []
    for family, language, surface in qualified_slices():
        for unit in (0, 1, 2, 5):
            for semantic, shift in ((0, 0), (0, 1), (1, 0), (0, 2), (2, 0)):
                ids.append(prior.case_id(family, language, surface, unit,
                                         semantic, shift))
    return ids


def capture_q32(model, device, compiled: dict[str, dict], ids: list[str]) -> tuple[np.ndarray, list[dict]]:
    path = OUTS["C638"] / "raw/q32_boundary_states.float32.npy"
    states = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(len(ids), DIM))
    observed: list[torch.Tensor] = []
    def hook(_module, _args, output):
        observed.append(output[0] if isinstance(output, tuple) else output)
    handle = model.model.layers[31].register_forward_hook(hook)
    ledger = []
    try:
        for row_i, cid in enumerate(ids):
            item = compiled[cid]
            input_ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids)
            positions = torch.arange(input_ids.shape[1], device=device)[None]
            observed.clear()
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention,
                      position_ids=positions, use_cache=False, return_dict=True)
            boundary = int(item["role_positions"]["boundary"][-1])
            states[row_i] = observed[0][0, boundary].float().cpu().numpy()
            ledger.append({"state_index": row_i, "case_id": cid,
                           "family": item["family"], "language": item["language"],
                           "surface": item["surface"], "unit": item["unit"],
                           "semantic": item["semantic"],
                           "code_shift": item["code_shift"],
                           "checkpoint": 32, "role": "boundary"})
            if row_i % 32 == 0 or row_i + 1 == len(ids):
                print(f"[C638 capture] {row_i + 1}/{len(ids)}", flush=True)
    finally:
        handle.remove()
    states.flush()
    write_rows(OUTS["C638"] / "raw/q32_state_ledger.jsonl", ledger)
    return states, ledger


def build_donors(states: np.ndarray, ledger: list[dict]) -> tuple[np.ndarray, list[dict]]:
    lookup = {row["case_id"]: row["state_index"] for row in ledger}
    donors, donor_ledger = [], []
    for family, language, surface in qualified_slices():
        for unit in (0, 1, 2):
            base_id = prior.case_id(family, language, surface, unit, 0, 0)
            base = np.asarray(states[lookup[base_id]], np.float32)
            for kind, (semantic, shift, expected) in KINDS.items():
                target_id = prior.case_id(family, language, surface, unit,
                                          semantic, shift)
                vector = np.asarray(states[lookup[target_id]], np.float32) - base
                donor_ledger.append({"donor_index": len(donors), "kind": kind,
                                     "expected_prediction": expected,
                                     "family": family, "language": language,
                                     "surface": surface, "unit": unit,
                                     "base_case": base_id, "target_case": target_id,
                                     "rms": float(np.sqrt(np.mean(vector ** 2))),
                                     "nonzero_fraction": float(np.mean(vector != 0))})
                donors.append(vector)
    donor_array = np.stack(donors).astype(np.float32)
    np.save(OUTS["C638"] / "raw/individual_donor_vectors.float32.npy",
            donor_array, allow_pickle=False)
    write_rows(OUTS["C638"] / "analysis/individual_donor_ledger.jsonl",
               donor_ledger)
    return donor_array, donor_ledger


def c638() -> None:
    out = begin("C638", {
        "object": "individual q32 output-identity donor response signatures",
        "donors": "3 discovery units x 8 qualified slices x code/semantic routes to X/Y",
        "receivers": "all eight fresh-word WXYZ lockbox slices",
        "test": "each complete 2560-coordinate donor is written into every receiver",
        "equivalence": "donors are grouped only by their complete receiver-output signature",
        "controls": "zero receiver behavior and opposite output identity remain explicit",
    }, {"C637": prior.final("C637")["all_checks_passed"]})
    compiled = compiled_rows()
    ids = capture_case_ids()
    model = None
    try:
        model, tokenizer, device, placement = MODEL_BASE.load_bf16("qwen3")
        states, state_ledger = capture_q32(model, device, compiled, ids)
        donors, donor_ledger = build_donors(states, state_ledger)
        records = []
        receivers = []
        for family, language, surface in qualified_slices():
            cid = prior.case_id(family, language, surface, 5, 0, 0)
            receivers.append({"receiver_index": len(receivers), "case_id": cid,
                              "family": family, "language": language,
                              "surface": surface})
        for receiver_i, receiver in enumerate(receivers):
            item = compiled[receiver["case_id"]]
            boundary = int(item["role_positions"]["boundary"][-1])
            zero = prior.base.patched_generate(model, tokenizer, item, [])
            for donor in donor_ledger:
                output = prior.base.patched_generate(
                    model, tokenizer, item,
                    [{"q": 32, "position": boundary,
                      "vector": donors[donor["donor_index"]]}])
                records.append({"receiver_index": receiver_i,
                                "receiver_case": receiver["case_id"],
                                "receiver_family": receiver["family"],
                                "receiver_language": receiver["language"],
                                "receiver_surface": receiver["surface"],
                                "donor_index": donor["donor_index"],
                                "donor_kind": donor["kind"],
                                "donor_family": donor["family"],
                                "donor_language": donor["language"],
                                "donor_surface": donor["surface"],
                                "donor_unit": donor["unit"],
                                "expected_prediction": donor["expected_prediction"],
                                "prediction": output["prediction"],
                                "expected": output["prediction"] == donor["expected_prediction"],
                                "generated_text": output["text"],
                                "candidate_scores": output["candidate_scores"],
                                "zero_prediction": zero["prediction"]})
            print(f"[C638 transfer] {receiver_i + 1}/{len(receivers)}", flush=True)
    finally:
        MODEL_BASE.release_bf16(model); gc.collect()
    write_rows(out / "analysis/individual_transfer_records.jsonl", records)
    write_rows(out / "analysis/receiver_ledger.jsonl", receivers)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["donor_kind"]].append(row["expected"])
    signatures, signature_audit = analyze_discrete_and_logit_signatures(records, donor_ledger)
    write_rows(out / "analysis/functional_signatures.jsonl", signatures)
    headline = {
        "status": "individual_donor_equivalence_closed",
        "captured_states": len(state_ledger), "state_shape": list(states.shape),
        "donors": len(donor_ledger), "receivers": len(receivers),
        "transfer_tests": len(records),
        "target_rates_by_kind": {kind: float(np.mean(values))
                                 for kind, values in sorted(grouped.items())},
        "universal_donors": sum(row["universal"] for row in signatures),
        **signature_audit,
        "strict_interpretation": (
            "A shared discrete receiver-output signature supports a finite behavioral "
            "response-equivalence candidate at q32. Continuous candidate-logit signatures "
            "remain non-identical; physically different vectors are not declared equal."),
    }
    states.flush(); prior.close_mmap(states)
    close("C638", headline, {
        "capture_complete": headline["state_shape"] == [160, 2560],
        "donor_complete": headline["donors"] == 96,
        "receiver_complete": headline["receivers"] == 8,
        "transfer_complete": headline["transfer_tests"] == 768,
        "finite": finite(headline),
    }, "C639_identity_dose_and_coordinate_partition")


def reanalyze_c638() -> None:
    out = OUTS["C638"]
    records = read_rows(out / "analysis/individual_transfer_records.jsonl")
    donor_ledger = read_rows(out / "analysis/individual_donor_ledger.jsonl")
    state_ledger = read_rows(out / "raw/q32_state_ledger.jsonl")
    signatures, signature_audit = analyze_discrete_and_logit_signatures(records, donor_ledger)
    write_rows(out / "analysis/functional_signatures.jsonl", signatures)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["donor_kind"]].append(row["expected"])
    old = final("C638")
    headline = {
        "status": "individual_donor_equivalence_closed",
        "captured_states": len(state_ledger), "state_shape": [len(state_ledger), DIM],
        "donors": len(donor_ledger), "receivers": len({row["receiver_index"] for row in records}),
        "transfer_tests": len(records),
        "target_rates_by_kind": {kind: float(np.mean(values)) for kind, values in sorted(grouped.items())},
        "universal_donors": sum(row["universal"] for row in signatures),
        **signature_audit,
        "strict_interpretation": (
            "All donors share one of two discrete receiver-output signatures, but their "
            "continuous candidate-logit signatures are not identical. The result is a "
            "finite behavioral response-equivalence candidate, not exact state equivalence."),
    }
    old["timestamp_utc_reanalysis"] = datetime.now(timezone.utc).isoformat()
    old["headline"] = headline
    save(out / "analysis/final.json", old)
    print(json.dumps(old, ensure_ascii=False, indent=2), flush=True)


def reanalyze_c640() -> None:
    result = final("C640")
    result["timestamp_utc_reanalysis"] = datetime.now(timezone.utc).isoformat()
    theory = result["headline"]["theory"]
    theory["updated_object"] = "registered discrete-output response-equivalence candidate for typed finite q32 state changes"
    theory["equivalence_formula"] = "delta_a ~_(T,Q_disc) delta_b iff all registered discrete receiver outputs agree"
    theory["strict_interpretation"] = (
        "The quotient is defined only under the finite discrete readout Q_disc. Candidate-logit "
        "signatures differ, so exact continuous response equivalence is not established.")
    result["headline"]["strict_conclusion"] = (
        "Fresh q32 identity is not an averaging artifact: every individual donor crosses every "
        "registered receiver. The two classes are discrete output-signature classes, while "
        "continuous logits remain donor-specific and upstream semantic closure is absent.")
    save(OUTS["C640"] / "analysis/final.json", result)
    visual = load(VISUAL)
    visual["functional_signatures"] = read_rows(OUTS["C638"] / "analysis/functional_signatures.jsonl")
    visual["claim_boundary"] = (
        "Exact activation coordinates are exposed. Equivalence is established only for "
        "registered discrete outputs; continuous candidate-logit signatures are non-identical.")
    save(VISUAL, visual)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def bf16_effective(base_state: np.ndarray, vector: np.ndarray,
                   scale: float) -> dict:
    base = torch.tensor(base_state, dtype=torch.bfloat16)
    requested = torch.tensor(vector * scale, dtype=torch.bfloat16)
    actual = (base + requested).float() - base.float()
    vector64 = np.asarray(vector, np.float64)
    actual64 = actual.numpy().astype(np.float64)
    return {"actual_rms": float(np.sqrt(np.mean(actual64 ** 2))),
            "effective_scale": float(np.dot(actual64, vector64) /
                                     (np.dot(vector64, vector64) + 1e-12)),
            "actual_nonzero_fraction": float(np.mean(actual64 != 0))}


def c639() -> None:
    out = begin("C639", {
        "object": "BF16-effective dose curves and magnitude-free coordinate partitions",
        "doses": [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "routes": ["code_x", "semantic_x", "code_y", "semantic_y"],
        "partitions": "16 fixed coordinate-index residue classes; each alone and each leave-one-out",
        "claim_boundary": "partition success measures distributed sufficiency under this basis; it is not a minimal causal cut",
    }, {"C638": final("C638")["all_checks_passed"]})
    compiled = compiled_rows()
    states = np.load(OUTS["C638"] / "raw/q32_boundary_states.float32.npy", mmap_mode="r")
    state_ledger = read_rows(OUTS["C638"] / "raw/q32_state_ledger.jsonl")
    state_lookup = {row["case_id"]: row["state_index"] for row in state_ledger}
    donors = np.load(OUTS["C638"] / "raw/individual_donor_vectors.float32.npy", mmap_mode="r")
    donor_ledger = read_rows(OUTS["C638"] / "analysis/individual_donor_ledger.jsonl")
    grouped = defaultdict(list)
    for donor in donor_ledger:
        grouped[(donor["family"], donor["language"], donor["surface"], donor["kind"])].append(
            np.asarray(donors[donor["donor_index"]], np.float32))
    prototypes = {key: np.mean(np.stack(values), axis=0) for key, values in grouped.items()}
    prototype_keys = sorted(prototypes)
    prototype_array = np.stack([prototypes[key] for key in prototype_keys]).astype(np.float32)
    np.save(out / "raw/route_prototypes.float32.npy", prototype_array, allow_pickle=False)
    write_rows(out / "analysis/route_prototype_ledger.jsonl", [
        {"prototype_index": i, "family": key[0], "language": key[1],
         "surface": key[2], "kind": key[3], "discovery_donors": 3}
        for i, key in enumerate(prototype_keys)])

    scales = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    dose_records, partition_records = [], []
    model = None
    try:
        model, tokenizer, device, placement = MODEL_BASE.load_bf16("qwen3")
        for receiver_i, (family, language, surface) in enumerate(qualified_slices()):
            receiver_id = prior.case_id(family, language, surface, 5, 0, 0)
            item = compiled[receiver_id]
            boundary = int(item["role_positions"]["boundary"][-1])
            base_state = np.asarray(states[state_lookup[receiver_id]], np.float32)
            for kind, (_, _, expected) in KINDS.items():
                vector = prototypes[(family, language, surface, kind)]
                for scale in scales:
                    output = prior.base.patched_generate(
                        model, tokenizer, item,
                        [{"q": 32, "position": boundary,
                          "vector": vector * scale}])
                    dose_records.append({"receiver_index": receiver_i,
                                         "receiver_case": receiver_id,
                                         "family": family, "language": language,
                                         "surface": surface, "kind": kind,
                                         "scale": scale, "expected_prediction": expected,
                                         "prediction": output["prediction"],
                                         "expected": output["prediction"] == expected} |
                                        bf16_effective(base_state, vector, scale))
            for kind in ("code_x", "semantic_x"):
                vector = prototypes[(family, language, surface, kind)]
                for part in range(16):
                    mask = np.arange(DIM) % 16 == part
                    for mode in ("alone", "leave_one_out"):
                        patch = np.zeros(DIM, np.float32)
                        if mode == "alone":
                            patch[mask] = vector[mask]
                        else:
                            patch[:] = vector
                            patch[mask] = 0
                        output = prior.base.patched_generate(
                            model, tokenizer, item,
                            [{"q": 32, "position": boundary, "vector": patch}])
                        partition_records.append({"receiver_index": receiver_i,
                                                  "receiver_case": receiver_id,
                                                  "family": family, "language": language,
                                                  "surface": surface, "kind": kind,
                                                  "partition": part, "mode": mode,
                                                  "coordinate_count": int(np.sum(mask) if mode == "alone" else np.sum(~mask)),
                                                  "prediction": output["prediction"],
                                                  "target": output["prediction"] == 1})
            print(f"[C639 receiver] {receiver_i + 1}/{len(qualified_slices())}", flush=True)
    finally:
        MODEL_BASE.release_bf16(model); gc.collect()
    write_rows(out / "analysis/dose_records.jsonl", dose_records)
    write_rows(out / "analysis/partition_records.jsonl", partition_records)
    by_scale = {}
    for kind, scale in itertools.product(KINDS, scales):
        rows = [row for row in dose_records if row["kind"] == kind and row["scale"] == scale]
        by_scale[f"{kind}|{scale:g}"] = {
            "tests": len(rows), "target_rate": float(np.mean([row["expected"] for row in rows])),
            "effective_scale_median": float(np.median([row["effective_scale"] for row in rows])),
            "actual_nonzero_fraction_median": float(np.median([row["actual_nonzero_fraction"] for row in rows])),
        }
    partition_summary = {}
    for kind, mode in itertools.product(("code_x", "semantic_x"),
                                        ("alone", "leave_one_out")):
        rows = [row for row in partition_records if row["kind"] == kind and row["mode"] == mode]
        partition_summary[f"{kind}|{mode}"] = {
            "tests": len(rows), "target_rate": float(np.mean([row["target"] for row in rows])),
            "receivers_with_any_target": len({row["receiver_index"] for row in rows if row["target"]}),
        }
    headline = {
        "status": "dose_partition_closed", "dose_tests": len(dose_records),
        "partition_tests": len(partition_records), "dose_summary": by_scale,
        "partition_summary": partition_summary,
        "strict_interpretation": (
            "Dose thresholds and deterministic coordinate partitions characterize the "
            "finite q32 instrument. They do not identify a unique neural coordinate set, "
            "and BF16-effective doses are reported instead of nominal doses alone."),
    }
    prior.close_mmap(states); prior.close_mmap(donors)
    close("C639", headline, {
        "dose_complete": len(dose_records) == 320,
        "partition_complete": len(partition_records) == 512,
        "partition_coordinate_arithmetic": all(row["coordinate_count"] in (160, 2400)
                                                for row in partition_records),
        "finite": finite(headline),
    }, "C640_identity_equivalence_visual_theory_audit")


def update_catalog() -> None:
    catalog = load(CATALOG)
    datasets = catalog.setdefault("field_datasets", [])
    datasets[:] = [item for item in datasets if item.get("id") != "c640_output_identity_equivalence_atlas"]
    datasets.append({"id": "c640_output_identity_equivalence_atlas",
                     "label": "C640 Output Identity Equivalence Atlas",
                     "path": "/vis_data/research_kernel/c640_output_identity_equivalence_atlas.json",
                     "phase": 2175, "full_coordinate": True})
    save(CATALOG, catalog)


def c640() -> None:
    out = begin("C640", {
        "object": "exact-coordinate identity-equivalence atlas and theory audit",
        "display": "all captured q32 states, all individual donor vectors, receiver signatures, doses and fixed partitions",
        "theory": "functional equivalence is defined by registered future responses, not coordinate equality",
    }, {"C638": final("C638")["all_checks_passed"],
        "C639": final("C639")["all_checks_passed"]})
    states = np.load(OUTS["C638"] / "raw/q32_boundary_states.float32.npy", mmap_mode="r")
    donors = np.load(OUTS["C638"] / "raw/individual_donor_vectors.float32.npy", mmap_mode="r")
    transfer = read_rows(OUTS["C638"] / "analysis/individual_transfer_records.jsonl")
    signatures = read_rows(OUTS["C638"] / "analysis/functional_signatures.jsonl")
    doses = read_rows(OUTS["C639"] / "analysis/dose_records.jsonl")
    partitions = read_rows(OUTS["C639"] / "analysis/partition_records.jsonl")
    visual = {
        "schema": "ai2050.output_identity_equivalence_atlas.v1",
        "phase": 2175, "campaign": "C638-C640", "model": "Qwen3-4B",
        "coordinate_policy": "all 2560 signed q32 coordinates, no Top-K/PCA/cosine compression",
        "state_ledger": read_rows(OUTS["C638"] / "raw/q32_state_ledger.jsonl"),
        "q32_boundary_states": np.asarray(states, np.float32).tolist(),
        "donor_ledger": read_rows(OUTS["C638"] / "analysis/individual_donor_ledger.jsonl"),
        "individual_donor_vectors": np.asarray(donors, np.float32).tolist(),
        "receiver_ledger": read_rows(OUTS["C638"] / "analysis/receiver_ledger.jsonl"),
        "functional_signatures": signatures,
        "transfer_records": transfer,
        "dose_records": doses,
        "partition_records": partitions,
        "claim_boundary": "This atlas exposes exact activation coordinates and registered responses; it is not a weight map or unique causal circuit.",
    }
    save(VISUAL, visual)
    update_catalog()
    prior.close_mmap(states); prior.close_mmap(donors)

    c638_summary = final("C638")["headline"]
    c639_summary = final("C639")["headline"]
    same_identity_routes = all(c638_summary["target_rates_by_kind"].get(kind, 0) >= 0.75
                               for kind in KINDS)
    dose_one = all(c639_summary["dose_summary"][f"{kind}|1"]["target_rate"] >= 0.75
                   for kind in KINDS)
    theory_gates = {
        "fresh_code_identity_replication": prior.final("C637")["headline"]["theory"]["gates"]["fresh_alphabet_discovery_identity"],
        "individual_donor_transfer": c638_summary["universal_donors"] >= 72,
        "operation_route_equivalence": same_identity_routes,
        "bf16_effective_unit_dose": dose_one,
        "fresh_interaction_transfer": prior.final("C637")["headline"]["theory"]["gates"]["fresh_interaction_transfer"],
        "dose_stable_single_coordinate_transport": False,
        "natural_external_human_validation": False,
    }
    theory = {
        "name": "conditional output-field closure theory",
        "organizing_principle": "reuse-difference-conditioning",
        "updated_object": "registered future-response equivalence class of typed finite q32 state changes",
        "equivalence_formula": "delta_a ~_T delta_b iff registered receiver/output responses agree",
        "gates": theory_gates, "passed": sum(theory_gates.values()),
        "required": len(theory_gates),
        "new_foundational_mathematics_authorized": all(theory_gates.values()),
        "strict_interpretation": "A response-equivalence class is an empirical quotient under a finite test set, not a universal semantic state or new foundation of mathematics.",
    }
    headline = {
        "status": "individual_identity_major_stage_closed",
        "visual": str(VISUAL.relative_to(ROOT)), "visual_bytes": VISUAL.stat().st_size,
        "visual_state_shape": list(states.shape), "visual_donor_shape": list(donors.shape),
        "theory": theory,
        "strict_conclusion": (
            "The campaign decides whether fresh q32 identity is an averaged artifact or an "
            "individual-donor response class. Any positive result remains an output-boundary "
            "mechanism unless upstream language operations and independent natural materials close."),
    }
    close("C640", headline, {
        "prior_closed": final("C638")["all_checks_passed"] and final("C639")["all_checks_passed"],
        "visual_exists": VISUAL.exists() and VISUAL.stat().st_size > 0,
        "visual_complete": headline["visual_state_shape"] == [160, 2560] and
                           headline["visual_donor_shape"] == [96, 2560],
        "all_rows_accounted": len(transfer) == 768 and len(doses) == 320 and len(partitions) == 512,
        "finite": finite(headline),
    }, "major_stage_complete_next_stage_must_change_from_output_identity_to_upstream_natural_language_operation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=tuple(PHASES), default="C638")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reanalyze", action="store_true")
    args = parser.parse_args()
    if args.reanalyze:
        reanalyze_c638()
        reanalyze_c640()
        return
    names = list(PHASES)
    for name in names[names.index(args.start):]:
        result_path = OUTS[name] / "analysis/final.json"
        if result_path.exists() and not args.force:
            print(f"[resume] {name} already closed", flush=True)
            continue
        globals()[name.lower()]()


if __name__ == "__main__":
    main()
