#!/usr/bin/env python3
"""C117 observation-first fourth-family campaign for controlled whole-part exceptions."""
from __future__ import annotations

import gc
import itertools
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1635_c117_whole_part_exception_observation_campaign"
C115 = RESULT / "phase1625_c115_fifth_lexicon_prospective_replication"
C116 = RESULT / "phase1630_c116_negation_scope_observation_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base
import phase1608_c110_exact_field_capture as capture_base
import phase1610_c110_frozen_transport_comparison as transport

FAMILY = "whole_part_exception"
PARTITIONS = ("discovery", "confirmation", "lockbox")
ROLES = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary")
PATH_ROLES = ("focus_record", "focus_post", "query_focus", "query_anchor")
STATES, DIM, WIDTH, BATCH = 37, 2560, 224, 8
KEY_STATES = (0, 4, 8, 12, 16, 19, 24, 28, 32, 36)
RAW_STATES = (0, 8, 16, 19, 24, 32, 36)
STEMS = (
    "aks", "bel", "cor", "dun", "eph", "fir", "gan", "hex", "ivor", "jun", "kesh", "lor",
    "mav", "nex", "orin", "pyr", "qas", "ryl", "siv", "tarn", "uve", "vor", "wex", "xyr",
    "yarn", "zef", "bram", "clen", "drax", "frel", "gorn", "hyl", "jasp", "krov", "luth", "mern",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return np.zeros_like(values) if norm <= 1e-12 else np.asarray(values, dtype=np.float32) / norm


def topk(values: np.ndarray, k: int) -> list[int]:
    candidate = np.argpartition(np.abs(values), -k)[-k:]
    return candidate[np.argsort(-np.abs(values[candidate]), kind="stable")].astype(int).tolist()


def med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def inventory() -> list[tuple[str, str, str, str, str]]:
    return [
        (f"Rig{stem}A", f"Rig{stem}B", f"{stem}valve", f"{stem}assembly", f"{stem}tag")
        for stem in STEMS
    ]


def build() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for index, values in enumerate(inventory()):
        partition = PARTITIONS[index // 12]
        unit_row = {
            "arm": "breadth",
            "unit_id": f"c117-exception-{index:02d}",
            "family": FAMILY,
            "world": "controlled_synthetic_default_exception",
            "partition": partition,
            "surface": "factorial",
            "values": list(values),
        }
        units.append(unit_row)
        for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
            prompt, focus, anchor = breadth_base.breadth_prompt(FAMILY, values, truth, surface, distractor, code)
            output_yes = (truth == 1) == (code == 1)
            cases.append({
                **unit_row,
                "case_id": f"c117-{len(cases):04d}",
                "truth_factor": truth,
                "surface_factor": surface,
                "distractor_factor": distractor,
                "code": code,
                "codebook": graph_base.CODEBOOKS[code]["name"],
                "truth": truth == 1,
                "output_yes": output_yes,
                "gold_position": 0 if output_yes else 1,
                "focus": focus,
                "anchor": anchor,
                "prompt": prompt,
            })
    return units, cases


def historical_values() -> set[str]:
    paths = (
        RESULT / "phase1575_c101_dual_arm/material/breadth_units.jsonl",
        RESULT / "phase1581_c102_typed_relation_coordinate_campaign/material/breadth_units.jsonl",
        C115 / "material/units.jsonl",
        C116 / "material/units.jsonl",
    )
    return {
        str(value).casefold()
        for path in paths
        for row in core.rows(path)
        for value in row.get("values", [])
    }


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C117 already exists: {OUT}")
    parent = core.load(C116 / "analysis/closure.json")
    parent_audit = core.load(C116 / "audit/independent_closure_audit.json")
    expected = "C117 observation-first fourth-family whole-part exception campaign"
    if not parent_audit["all_checks_passed"] or not parent["next_authorization"].startswith(expected):
        raise RuntimeError("C117 authorization missing")

    units, cases = build()
    tok = graph_base.tokenizer()
    compiled = breadth_base.compile_breadth(tok, cases)
    zero = breadth_base.zero_models(cases, True)
    occurrences, disjoint = [], True
    for row_index, row in enumerate(compiled):
        occupied = []
        for role in ROLES:
            positions = [int(value) for value in row["role_positions"][role]]
            occupied.extend(positions)
            for subtoken, position in enumerate(positions):
                token_id = int(row["prompt_ids"][position])
                occurrences.append({
                    "occurrence_index": len(occurrences),
                    "row_index": row_index,
                    "case_id": row["case_id"],
                    "unit_id": row["unit_id"],
                    "family": FAMILY,
                    "partition": row["partition"],
                    "truth_factor": row["truth_factor"],
                    "surface_factor": row["surface_factor"],
                    "distractor_factor": row["distractor_factor"],
                    "code": row["code"],
                    "role": role,
                    "subtoken": subtoken,
                    "span_length": len(positions),
                    "token_position": position,
                    "token_id": token_id,
                    "token_text": tok.convert_ids_to_tokens([token_id])[0],
                })
        disjoint = disjoint and len(occupied) == len(set(occupied))

    cells = Counter(
        (row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"])
        for row in cases
    )
    old = historical_values()
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    rng = np.random.default_rng(1635)
    permutations = [rng.permutation(256).astype(int).tolist() for _ in range(8)]
    checks = {
        "authorization": parent_audit["all_checks_passed"],
        "units": len(units) == 36,
        "cases": len(cases) == 576,
        "partitions": Counter(row["partition"] for row in units) == {partition: 12 for partition in PARTITIONS},
        "factorial": cells == {(partition, *cell): 12 for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=4)},
        "zero_models": all(abs(value - 0.5) < 1e-12 for key, value in zero.items() if key != "truth_x_code_oracle") and zero["truth_x_code_oracle"] == 1.0,
        "semantic_uniqueness": len({row["prompt"] for row in cases}) == 576,
        "freshness": not (set(fresh) & old) and len(fresh) == len(set(fresh)),
        "compiled": len(compiled) == 576 and all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled) and disjoint,
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "machine_naturalness": all(
            "The handbook says" in row["prompt"]
            and "The exception log says" in row["prompt"]
            and "currently retain" in row["prompt"]
            and row["prompt"].endswith("Reply exactly yes or no.")
            for row in cases
        ),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})

    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.write_rows(OUT / "protocol/role_occurrence_manifest.jsonl", occurrences)
    protocol = {
        "phase": 1635,
        "campaign": "C117",
        "created_at_utc": now(),
        "status": "whole_part_exception_observation_first_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "fourth-family controlled default-exception field, assignment, route and common-component residual atlas",
        "family": FAMILY,
        "partitions": list(PARTITIONS),
        "units": 36,
        "cases": 576,
        "units_per_partition": 12,
        "roles": list(ROLES),
        "states": STATES,
        "activation_coordinates": DIM,
        "occurrences": len(occurrences),
        "archive": {
            "path": "raw/qwen3_role_subtoken_all_states.uint16.npy",
            "shape": [STATES, len(occurrences), DIM],
            "dtype": "uint16 exact BF16 bit patterns",
            "fixed_width": WIDTH,
            "batch_size": BATCH,
        },
        "discovery_rule": {
            "partitions_allowed": ["discovery"],
            "eligible_states": list(range(1, 31)),
            "eligible_roles": list(ROLES),
            "minimum_half_norm": 0.25,
            "score": "split_half_cosine * min(split_half_norms)",
            "support_k": 256,
            "tie_break": "larger split-half cosine, then smaller state, then role order",
        },
        "frozen_validation_gates": {
            "confirmation_lockbox_cosine_min": 0.85,
            "each_to_discovery_cosine_min": 0.80,
            "each_support_topk_overlap_min": 0.45,
            "correct_movement_gt_permutation_median_cells": 4,
        },
        "role_descriptors": {
            "selected_role_positive_cells": "descriptive",
            "path_gt_query_cells": "descriptive",
            "query_anchor_positive_cells": "descriptive",
        },
        "common_component_residual": {
            "reference_families": ["C115_attribute_binding", "C115_agent_patient", "C116_negation_scope"],
            "definition": "normalize(sum(normalize(reference_family_mean)))",
            "residual": "R - dot(R,G)*G",
            "status": "descriptive_no_gate",
        },
        "movement_permutations": permutations,
        "intervention_modes": ["frozen_support"] + [f"movement_permutation_{index}" for index in range(8)] + ["selected_role", "query_anchor", "record_to_query_path", "all_registered_roles"],
        "observation_first": "capture all registered embedding/HiddenState coordinates; discovery alone nominates role/state/support; confirmation and lockbox remain unread until nomination is frozen",
        "behavior_policy": "standard and reversed code are reported separately; behavior does not erase descriptive full-field observations, and raw truth rescue is not task-aligned output closure",
        "completion_rule": "run all field, residual and intervention routes after nomination; a failed descriptor retires only that descriptor",
        "numeric": {"movement_l2_relative_tolerance": 0.02, "batch_size": BATCH, "fixed_width": WIDTH},
        "typed_missingness": {
            "human_naturalness": "machine-only controlled English",
            "cross_model": "Qwen3 only",
            "natural_language_scope": "default-exception template, not general whole-part semantics",
        },
        "claim_boundary": "controlled synthetic default-exception activation study; no natural-language universality, weights, semantic neurons, attention/MLP, endogenous route, common assignment algorithm, low-dimensional manifold, orthogonal subspaces, topology, algebraic closure, symmetry group, or new-mathematics claim",
        "source_paths": {
            "c115_mean": str(C115 / "analysis/mean_truth_role_state.float32.npy"),
            "c116_closure": str(C116 / "analysis/closure.json"),
            "c116_audit": str(C116 / "audit/independent_closure_audit.json"),
        },
        "source_hashes": {
            "c115_mean": core.sha(C115 / "analysis/mean_truth_role_state.float32.npy"),
            "c116_closure": core.sha(C116 / "analysis/closure.json"),
            "c116_audit": core.sha(C116 / "audit/independent_closure_audit.json"),
        },
        "material_digest": core.digest([*units, *cases]),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1636_c117_exact_field_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {
        "phase": 1635,
        "campaign": "C117",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "zero_models": zero,
        "occurrences": len(occurrences),
        "max_width": max(len(row["prompt_ids"]) for row in compiled),
        "authorization": protocol["authorization"],
    }
    core.save(OUT / "audit/internal_pre_model_audit.json", report)
    print(json.dumps(report, indent=2))


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_pre_model_audit.json")["all_checks_passed"]:
        raise RuntimeError("C117 pre-model audit missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    by_row: dict[int, list[dict]] = defaultdict(list)
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        row_index = int(occurrence["row_index"])
        by_row[row_index].append(occurrence)
        lookup[(row_index, occurrence["role"])].append(int(occurrence["occurrence_index"]))

    raw_path = OUT / protocol["archive"]["path"]
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (raw_path, logits_path, index_path)):
        raise RuntimeError("C117 raw output already exists")
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=tuple(protocol["archive"]["shape"]))
    candidate_logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    behavior, model, first_rows = [], None, None
    repeat_hidden = repeat_logits = 0.0
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            output, logits, ids, mask, positions, lengths = capture_base.forward(model, batch, pad, device, WIDTH)
            for state_index, state in enumerate(output.hidden_states):
                if state.dtype != torch.bfloat16 or not bool(torch.isfinite(state).all()):
                    raise RuntimeError((state_index, state.dtype))
                for local in range(len(batch)):
                    row_index = start + local
                    occurrences = by_row[row_index]
                    indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                    token_positions = [int(item["token_position"]) for item in occurrences]
                    field[state_index, indices] = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
            for local, row in enumerate(batch):
                row_index = start + local
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                candidate_logits[row_index] = scores
                prediction = int(scores[1] > scores[0])
                behavior.append({
                    "row_index": row_index,
                    "case_id": row["case_id"],
                    "unit_id": row["unit_id"],
                    "partition": row["partition"],
                    "truth_factor": row["truth_factor"],
                    "surface_factor": row["surface_factor"],
                    "distractor_factor": row["distractor_factor"],
                    "code": row["code"],
                    "gold_position": row["gold_position"],
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "yes_minus_no": scores[0] - scores[1],
                })
            if start == 0:
                first_rows = batch
            if (start // BATCH + 1) % 12 == 0:
                field.flush()
                candidate_logits.flush()
                print(f"[phase1636] captured {start + len(batch)}/{len(rows)}", flush=True)
            del output, logits, ids, mask, positions
        field.flush()
        candidate_logits.flush()

        output, logits, ids, mask, positions, lengths = capture_base.forward(model, first_rows, pad, device, WIDTH)
        for state_index, state in enumerate(output.hidden_states):
            for local in range(len(first_rows)):
                occurrences = by_row[local]
                indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                token_positions = [int(item["token_position"]) for item in occurrences]
                old = np.asarray(field[state_index, indices])
                new = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
                if not np.array_equal(old, new):
                    repeat_hidden = max(repeat_hidden, float(np.max(np.abs(decode(old) - decode(new)))))
        for local, row in enumerate(first_rows):
            scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
            repeat_logits = max(repeat_logits, float(np.max(np.abs(scores - candidate_logits[local]))))
        del output, logits, ids, mask, positions
    finally:
        field.flush()
        candidate_logits.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()

    core.write_rows(index_path, behavior)
    field = np.load(raw_path, mmap_mode="r")
    candidate_logits = np.load(logits_path, mmap_mode="r")
    by_unit: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_unit[row["unit_id"]].append(row_index)
    causal_prefix = code_previsible = 0.0
    for row_indices in by_unit.values():
        reference = row_indices[0]
        for row_index in row_indices[1:]:
            left = capture_base.role_bits(field, lookup, reference, "focus_pre")
            right = capture_base.role_bits(field, lookup, row_index, "focus_pre")
            if not np.array_equal(left, right):
                causal_prefix = max(causal_prefix, float(np.max(np.abs(decode(left) - decode(right)))))
        unit_rows = [rows[index] for index in row_indices]
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            standard = next(index for index, row in zip(row_indices, unit_rows, strict=True) if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, 1))
            reversed_code = next(index for index, row in zip(row_indices, unit_rows, strict=True) if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, -1))
            for role in ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor"):
                left = capture_base.role_bits(field, lookup, standard, role)
                right = capture_base.role_bits(field, lookup, reversed_code, role)
                if not np.array_equal(left, right):
                    code_previsible = max(code_previsible, float(np.max(np.abs(decode(left) - decode(right)))))

    def accuracy(selected: list[dict]) -> float:
        return float(np.mean([row["correct"] for row in selected]))

    behavior_summary = {
        "global_accuracy": accuracy(behavior),
        "by_partition": {partition: accuracy([row for row in behavior if row["partition"] == partition]) for partition in PARTITIONS},
        "by_code": {str(code): accuracy([row for row in behavior if row["code"] == code]) for code in (1, -1)},
    }
    checks = {
        "shape": list(field.shape) == protocol["archive"]["shape"],
        "dtype": field.dtype == np.uint16,
        "logits": list(candidate_logits.shape) == [576, 2] and bool(np.isfinite(candidate_logits).all()),
        "index": len(behavior) == 576,
        "repeat": repeat_hidden == 0.0 and repeat_logits == 0.0,
        "causal_prefix": causal_prefix == 0.0,
        "code_previsible": code_previsible == 0.0,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1636,
        "campaign": "C117",
        "created_at_utc": now(),
        "status": "whole_part_exception_exact_field_capture_complete",
        "shape": list(field.shape),
        "raw_file_bytes": raw_path.stat().st_size,
        "raw_data_bytes": int(field.nbytes),
        "raw_sha256": core.sha(raw_path),
        "logits_sha256": core.sha(logits_path),
        "index_sha256": core.sha(index_path),
        "behavior": behavior_summary,
        "numeric": {
            "repeat_hidden_max_abs": repeat_hidden,
            "repeat_logits_max_abs": repeat_logits,
            "causal_prefix_max_abs": causal_prefix,
            "code_previsible_max_abs": code_previsible,
        },
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "authorization": "run_phase1637_c117_discovery_freeze",
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def truth_fields(partitions: set[str], path: Path) -> tuple[np.ndarray, list[dict]]:
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = [row for row in core.rows(OUT / "material/units.jsonl") if row["partition"] in partitions]
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    raw = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    role_index = {role: index for index, role in enumerate(ROLES)}
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(int(occurrence["occurrence_index"]))
    fields = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(units), len(ROLES), STATES, DIM))
    fields[:] = 0.0
    for state in range(STATES):
        for row_index, row in enumerate(rows):
            if row["partition"] not in partitions:
                continue
            coefficient = float(row["truth_factor"]) / 16.0
            u = unit_index[row["unit_id"]]
            for role in ROLES:
                values = decode(raw[state, lookup[(row_index, role)]])
                fields[u, role_index[role], state] += coefficient * np.mean(values, axis=0, dtype=np.float32)
        if state % 6 == 0 or state == 36:
            fields.flush()
            print(f"[C117 field] derived state {state}/36 for {sorted(partitions)}", flush=True)
    fields.flush()
    return fields, units


def discover() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"]:
        raise RuntimeError("C117 capture audit missing")
    path = OUT / "analysis/discovery_unit_truth_role_state.float32.npy"
    if path.exists():
        raise RuntimeError("C117 discovery field already exists")
    fields, units = truth_fields({"discovery"}, path)
    candidates = []
    minimum = float(protocol["discovery_rule"]["minimum_half_norm"])
    for role_index, role in enumerate(ROLES):
        for state in protocol["discovery_rule"]["eligible_states"]:
            left = np.mean(fields[:6, role_index, state], axis=0, dtype=np.float32)
            right = np.mean(fields[6:, role_index, state], axis=0, dtype=np.float32)
            left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
            split_cosine = cosine(left, right)
            score = split_cosine * min(left_norm, right_norm) if min(left_norm, right_norm) >= minimum else None
            candidates.append({"role": role, "state": int(state), "split_half_cosine": split_cosine, "left_norm": left_norm, "right_norm": right_norm, "score": score})
    eligible = [row for row in candidates if row["score"] is not None]
    winner = sorted(eligible, key=lambda row: (-row["score"], -row["split_half_cosine"], row["state"], ROLES.index(row["role"])))[0]
    role_index = ROLES.index(winner["role"])
    mean = np.mean(fields[:, role_index, winner["state"]], axis=0, dtype=np.float32)
    support = topk(mean, int(protocol["discovery_rule"]["support_k"]))
    nomination = {
        **winner,
        "support_k": len(support),
        "support": support,
        "discovery_units": [row["unit_id"] for row in units],
        "field_norm": float(np.linalg.norm(mean)),
        "field_sha256": core.sha(path),
        "candidate_table_sha256": "pending",
        "created_at_utc": now(),
    }
    table_path = OUT / "analysis/discovery_candidate_table.jsonl"
    core.write_rows(table_path, candidates)
    nomination["candidate_table_sha256"] = core.sha(table_path)
    nomination_path = OUT / "protocol/frozen_discovery_nomination.json"
    core.save(nomination_path, nomination)
    checks = {
        "units": len(units) == 12,
        "shape": list(fields.shape) == [12, 7, 37, 2560],
        "candidates": len(candidates) == 210,
        "eligible": bool(eligible),
        "support": len(support) == 256 and len(set(support)) == 256,
        "discovery_only": all(row["partition"] == "discovery" for row in units),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1637,
        "campaign": "C117",
        "created_at_utc": now(),
        "status": "discovery_role_state_support_frozen",
        "winner": winner,
        "support_k": 256,
        "checks": checks,
        "discovery_sha256": core.sha(path),
        "nomination_sha256": core.sha(nomination_path),
        "authorization": "execute_phase1638_c117_confirmation_lockbox_validation",
    }
    core.save(OUT / "analysis/discovery_freeze.json", report)
    print(json.dumps(report, indent=2))


def common_component_metrics(role: str, state: int, confirmation: np.ndarray, lockbox: np.ndarray) -> dict:
    role_index = ROLES.index(role)
    c115 = np.load(C115 / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    attribute = np.mean(np.asarray(c115[0, :, role_index, state], dtype=np.float32), axis=0, dtype=np.float32)
    agent = np.mean(np.asarray(c115[1, :, role_index, state], dtype=np.float32), axis=0, dtype=np.float32)
    c116_d = np.load(C116 / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r")
    c116_v = np.load(C116 / "analysis/validation_unit_truth_role_state.float32.npy", mmap_mode="r")
    negation = np.mean(np.concatenate((np.asarray(c116_d[:, role_index, state]), np.asarray(c116_v[:, role_index, state])), axis=0), axis=0, dtype=np.float32)
    references = {"attribute_binding": attribute, "agent_patient": agent, "negation_scope": negation}
    common = unit(sum((unit(value) for value in references.values()), start=np.zeros(DIM, dtype=np.float32)))

    def residual(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) - float(np.dot(value, common)) * common

    c_residual, l_residual = residual(confirmation), residual(lockbox)
    return {
        "reference_pairwise_cosines": {
            "attribute_agent": cosine(attribute, agent),
            "attribute_negation": cosine(attribute, negation),
            "agent_negation": cosine(agent, negation),
        },
        "whole_part_to_references": {
            "confirmation": {name: cosine(confirmation, value) for name, value in references.items()},
            "lockbox": {name: cosine(lockbox, value) for name, value in references.items()},
        },
        "whole_part_to_common": {"confirmation": cosine(confirmation, common), "lockbox": cosine(lockbox, common)},
        "residual_cross_partition_cosine": cosine(c_residual, l_residual),
        "residual_norm_fraction": {
            "confirmation": float(np.linalg.norm(c_residual) / max(np.linalg.norm(confirmation), 1e-12)),
            "lockbox": float(np.linalg.norm(l_residual) / max(np.linalg.norm(lockbox), 1e-12)),
        },
        "definition": "G=normalize(sum_f normalize(R_f)); E=R-dot(R,G)G; descriptive only",
    }


@torch.inference_mode()
def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    if not core.load(OUT / "audit/independent_discovery_audit.json")["all_checks_passed"]:
        raise RuntimeError("C117 discovery audit missing")
    path = OUT / "analysis/validation_unit_truth_role_state.float32.npy"
    if path.exists():
        fields = np.load(path, mmap_mode="r")
        units = [row for row in core.rows(OUT / "material/units.jsonl") if row["partition"] in {"confirmation", "lockbox"}]
        core.save(OUT / "protocol/phase1638_execution_amendment.json", {
            "phase": 1638,
            "campaign": "C117",
            "created_at_utc": now(),
            "reason": "the first two validation attempts referenced the same undefined donor_states alias while constructing the first patch dictionary, before any intervention result, summary, gate or adjudication was written",
            "repair": "replace every residual donor_states alias with the existing don_states variable and reuse the exact already-derived validation field",
            "unchanged": ["materials", "partitions", "nomination", "support", "intervention modes", "permutations", "all gates", "claim boundary"],
            "original_producer_sha256": protocol["producer_sha256"],
            "repaired_producer_sha256": core.sha(Path(__file__)),
            "nomination_sha256": core.sha(OUT / "protocol/frozen_discovery_nomination.json"),
            "validation_field_sha256": core.sha(path),
        })
    else:
        fields, units = truth_fields({"confirmation", "lockbox"}, path)
    role_index, state = ROLES.index(nomination["role"]), int(nomination["state"])
    discovery = np.load(OUT / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r")
    d = np.mean(discovery[:, role_index, state], axis=0, dtype=np.float32)
    c = np.mean(fields[:12, role_index, state], axis=0, dtype=np.float32)
    l = np.mean(fields[12:, role_index, state], axis=0, dtype=np.float32)
    gates = protocol["frozen_validation_gates"]
    support_set = set(nomination["support"])
    field_metrics = {
        "role": nomination["role"],
        "state": state,
        "confirmation_lockbox_cosine": cosine(c, l),
        "confirmation_to_discovery_cosine": cosine(c, d),
        "lockbox_to_discovery_cosine": cosine(l, d),
        "confirmation_support_overlap": len(set(topk(c, 256)) & support_set) / 256,
        "lockbox_support_overlap": len(set(topk(l, 256)) & support_set) / 256,
    }
    field_checks = {
        "confirmation_lockbox": field_metrics["confirmation_lockbox_cosine"] >= gates["confirmation_lockbox_cosine_min"],
        "to_discovery": min(field_metrics["confirmation_to_discovery_cosine"], field_metrics["lockbox_to_discovery_cosine"]) >= gates["each_to_discovery_cosine_min"],
        "support_overlap": min(field_metrics["confirmation_support_overlap"], field_metrics["lockbox_support_overlap"]) >= gates["each_support_topk_overlap_min"],
    }
    residual_metrics = common_component_metrics(nomination["role"], state, c, l)

    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    pairs = [pair for pair in transport.build_pairs(rows, protocol) if pair["partition"] in {"confirmation", "lockbox"}]
    for index, pair in enumerate(pairs):
        pair["pair_id"] = f"c117-pair-{index:04d}"
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        lengths = tuple(len(pair["recipient"]["role_positions"][role]) for role in ROLES)
        grouped[(pair["partition"], pair["code"], lengths)].append(pair)

    results, model, first_repeat = [], None, None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        support = torch.tensor(nomination["support"], dtype=torch.long, device=device)
        permutations = [torch.tensor(values, dtype=torch.long, device=device) for values in protocol["movement_permutations"]]
        for (partition, code, lengths), group in sorted(grouped.items()):
            for start in range(0, len(group), BATCH):
                batch = group[start:start + BATCH]
                recipients = [pair["recipient"] for pair in batch]
                donors = [pair["donor"] for pair in batch]
                rec_logits, rec_states = transport.forward_with_roles(model, recipients, ROLES, state, pad, device, WIDTH)
                don_logits, don_states = transport.forward_with_roles(model, donors, ROLES, state, pad, device, WIDTH)
                if first_repeat is None:
                    first_repeat = (recipients, rec_logits.detach().clone(), {role: value.detach().clone() for role, value in rec_states.items()})
                recipient_state = rec_states[nomination["role"]]
                donor_state = don_states[nomination["role"]]
                delta = donor_state[..., support] - recipient_state[..., support]
                norm = torch.sqrt(torch.sum(delta.float() ** 2, dim=(1, 2))).clamp_min(1e-12)
                patches = {}
                value = recipient_state.clone()
                value[..., support] = donor_state[..., support]
                patches["frozen_support"] = {nomination["role"]: value}
                permutation_norms = []
                for index, permutation in enumerate(permutations):
                    value = recipient_state.clone()
                    value[..., support] = (recipient_state[..., support].float() + delta.float()[..., permutation]).to(recipient_state.dtype)
                    patches[f"movement_permutation_{index}"] = {nomination["role"]: value}
                    permutation_norms.append(torch.sqrt(torch.sum((value[..., support] - recipient_state[..., support]).float() ** 2, dim=(1, 2))).clamp_min(1e-12))
                patches["selected_role"] = {nomination["role"]: don_states[nomination["role"]].clone()}
                patches["query_anchor"] = {"query_anchor": don_states["query_anchor"].clone()}
                patches["record_to_query_path"] = {role: don_states[role].clone() for role in PATH_ROLES}
                patches["all_registered_roles"] = {role: don_states[role].clone() for role in ROLES}
                patched = {mode: transport.forward_patched_roles(model, recipients, values, state, pad, device, WIDTH) for mode, values in patches.items()}
                for local, pair in enumerate(batch):
                    base = transport.margin(rec_logits[local], recipients[local])
                    mode_results = {}
                    for mode, logits in patched.items():
                        margin = transport.margin(logits[local], recipients[local])
                        mode_results[mode] = {"truth_direction_gain": margin - base, "truth_flip": base <= 0 < margin}
                    results.append({
                        "pair_id": pair["pair_id"],
                        "unit_id": pair["unit_id"],
                        "partition": partition,
                        "code": code,
                        "surface_factor": pair["surface_factor"],
                        "distractor_factor": pair["distractor_factor"],
                        "recipient_yes_minus_no": base,
                        "target_movement_l2": float(norm[local]),
                        "permutation_l2_relative_errors": [float(torch.abs(value[local] - norm[local]) / norm[local]) for value in permutation_norms],
                        "modes": mode_results,
                    })
                print(f"[phase1638] {partition}/code={code}/lengths={lengths} {start + len(batch)}/{len(group)}", flush=True)
        repeat_rows, old_logits, old_states = first_repeat
        new_logits, new_states = transport.forward_with_roles(model, repeat_rows, ROLES, state, pad, device, WIDTH)
        repeat_logits = float(torch.max(torch.abs(new_logits - old_logits)))
        repeat_hidden = max(float(torch.max(torch.abs(new_states[role] - old_states[role]))) for role in ROLES)
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()

    result_path = OUT / "analysis/validation_intervention_results.jsonl"
    core.write_rows(result_path, results)
    summaries = []
    for partition in ("confirmation", "lockbox"):
        for code in (1, -1):
            selected = [row for row in results if row["partition"] == partition and row["code"] == code]
            correct = med([row["modes"]["frozen_support"]["truth_direction_gain"] for row in selected])
            permutation_medians = [med([row["modes"][f"movement_permutation_{index}"]["truth_direction_gain"] for row in selected]) for index in range(8)]
            mode_medians = {mode: med([row["modes"][mode]["truth_direction_gain"] for row in selected]) for mode in ("selected_role", "query_anchor", "record_to_query_path", "all_registered_roles")}
            summaries.append({
                "partition": partition,
                "code": code,
                "pairs": len(selected),
                "independent_units": len({row["unit_id"] for row in selected}),
                "frozen_support_median_gain": correct,
                "permutation_median_gains": permutation_medians,
                "frozen_support_gt_permutation_median": correct > med(permutation_medians),
                "frozen_support_gt_all_permutations": all(correct > value for value in permutation_medians),
                "mode_median_gains": mode_medians,
                "path_gt_query": mode_medians["record_to_query_path"] > mode_medians["query_anchor"],
                "selected_role_positive": mode_medians["selected_role"] > 0,
                "query_anchor_positive": mode_medians["query_anchor"] > 0,
                "truth_flip_counts": {mode: sum(item["modes"][mode]["truth_flip"] for item in selected) for mode in selected[0]["modes"]},
            })
    summary_path = OUT / "analysis/validation_summary.jsonl"
    core.write_rows(summary_path, summaries)
    predictions = {
        "field_passed": all(field_checks.values()),
        "correct_movement_gt_permutation_median_cells": sum(row["frozen_support_gt_permutation_median"] for row in summaries),
        "strict_win_cells_descriptive": sum(row["frozen_support_gt_all_permutations"] for row in summaries),
        "path_gt_query_cells_descriptive": sum(row["path_gt_query"] for row in summaries),
        "selected_role_positive_cells_descriptive": sum(row["selected_role_positive"] for row in summaries),
        "query_anchor_positive_cells_descriptive": sum(row["query_anchor_positive"] for row in summaries),
    }
    prediction_checks = {
        "field": predictions["field_passed"],
        "coordinate_assignment": predictions["correct_movement_gt_permutation_median_cells"] == gates["correct_movement_gt_permutation_median_cells"],
    }
    max_error = max(error for row in results for error in row["permutation_l2_relative_errors"])
    checks = {
        "units": len(units) == 24,
        "shape": list(fields.shape) == [24, 7, 37, 2560],
        "pairs": len(results) == 192,
        "summaries": len(summaries) == 4 and all(row["pairs"] == 48 and row["independent_units"] == 12 for row in summaries),
        "l2": max_error <= protocol["numeric"]["movement_l2_relative_tolerance"],
        "finite": all(math.isfinite(row["modes"][mode]["truth_direction_gain"]) for row in results for mode in row["modes"]),
        "repeat": repeat_hidden == 0.0 and repeat_logits == 0.0,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "residual_finite": all(math.isfinite(value) for value in [
            residual_metrics["residual_cross_partition_cosine"],
            *residual_metrics["residual_norm_fraction"].values(),
            *residual_metrics["whole_part_to_common"].values(),
        ]),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1638,
        "campaign": "C117",
        "created_at_utc": now(),
        "status": "whole_part_exception_confirmation_lockbox_validation_complete",
        "nomination": {"role": nomination["role"], "state": state, "support_k": 256},
        "field_metrics": field_metrics,
        "field_checks": field_checks,
        "common_component_residual": residual_metrics,
        "predictions": predictions,
        "prediction_checks": prediction_checks,
        "all_primary_gates_passed": all(prediction_checks.values()),
        "max_l2_relative_error": max_error,
        "checks": checks,
        "runtime": {"placement": placement, "quantization": quant},
        "field_sha256": core.sha(path),
        "results_sha256": core.sha(result_path),
        "summary_sha256": core.sha(summary_path),
        "authorization": "run_phase1639_c117_synthesis_heatmap_and_closure",
    }
    core.save(OUT / "analysis/validation_adjudication.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    discovery = core.load(OUT / "analysis/discovery_freeze.json")
    validation = core.load(OUT / "analysis/validation_adjudication.json")
    if not core.load(OUT / "audit/independent_validation_audit.json")["all_checks_passed"]:
        raise RuntimeError("C117 validation audit missing")
    nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    discovery_fields = np.load(OUT / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r")
    validation_fields = np.load(OUT / "analysis/validation_unit_truth_role_state.float32.npy", mmap_mode="r")
    means = [
        np.mean(discovery_fields, axis=0, dtype=np.float32),
        np.mean(validation_fields[:12], axis=0, dtype=np.float32),
        np.mean(validation_fields[12:], axis=0, dtype=np.float32),
    ]
    payload = core.load(PUBLIC)
    display_states = tuple(sorted(set(KEY_STATES) | {int(nomination["state"])}))
    effect_rows = []
    for partition, mean in zip(PARTITIONS, means, strict=True):
        for role_index, role in enumerate(ROLES):
            for state in display_states:
                effect_rows.append({
                    "dataset": "C117",
                    "family": FAMILY,
                    "partition": partition,
                    "role": role,
                    "state": state,
                    "state_kind": "embedding" if state == 0 else "hidden_state",
                    "effect": "balanced_truth_walsh",
                    "values": np.asarray(mean[role_index, state], dtype=np.float32).tolist(),
                })
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    raw_field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    lookup: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(occurrence)
    raw_rows = []
    for partition in PARTITIONS:
        row_index = next(index for index, row in enumerate(compiled) if row["partition"] == partition)
        row = compiled[row_index]
        query = lookup[(row_index, "query_anchor")][0]
        query_index = int(query["occurrence_index"])
        for state in RAW_STATES:
            raw_rows.append({
                "dataset": "C117",
                "case_id": row["case_id"],
                "family": FAMILY,
                "partition": partition,
                "truth_factor": row["truth_factor"],
                "surface_factor": row["surface_factor"],
                "distractor_factor": row["distractor_factor"],
                "code": row["code"],
                "role": "query_anchor",
                "subtoken": int(query["subtoken"]),
                "token_position": int(query["token_position"]),
                "token_id": int(query["token_id"]),
                "token_text": query["token_text"],
                "state": state,
                "state_kind": "embedding" if state == 0 else "hidden_state",
                "values": decode(raw_field[state, query_index]).tolist(),
            })
        candidate = lookup[(row_index, nomination["role"])][0]
        candidate_index = int(candidate["occurrence_index"])
        state = int(nomination["state"])
        raw_rows.append({
            "dataset": "C117",
            "case_id": row["case_id"],
            "family": FAMILY,
            "partition": partition,
            "truth_factor": row["truth_factor"],
            "surface_factor": row["surface_factor"],
            "distractor_factor": row["distractor_factor"],
            "code": row["code"],
            "role": nomination["role"],
            "subtoken": int(candidate["subtoken"]),
            "token_position": int(candidate["token_position"]),
            "token_id": int(candidate["token_id"]),
            "token_text": candidate["token_text"],
            "state": state,
            "state_kind": "hidden_state",
            "values": decode(raw_field[state, candidate_index]).tolist(),
        })

    payload["effect_rows"] = [row for row in payload["effect_rows"] if row.get("dataset") != "C117"] + effect_rows
    payload["raw_rows"] = [row for row in payload["raw_rows"] if row.get("dataset") != "C117"] + raw_rows
    payload["default_coordinates"] = nomination["support"][:64]
    payload["scale"] = {
        "effect_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["effect_rows"]]), 0.99)),
        "raw_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["raw_rows"]]), 0.99)),
    }
    payload.update({
        "phase": 1639,
        "campaign": "C109-C117",
        "title": "C109-C117 Relation-Role-State Activation Atlas",
        "c117_batch": {
            "discovery": discovery,
            "validation": validation,
            "summaries": core.rows(OUT / "analysis/validation_summary.jsonl"),
            "nomination": {"role": nomination["role"], "state": nomination["state"], "support_k": nomination["support_k"]},
        },
        "claim_boundary": "C117 adds one controlled default-exception family with discovery-frozen role/state/support and independent confirmation/lockbox validation. It is an activation-coordinate observation/intervention atlas, not weights, semantic neurons, attention/MLP, an endogenous route, a common assignment algorithm, orthogonal subspaces, topology, universal relation algebra, or new mathematics.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c117_relation_role_state_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    closure = {
        "phase": 1639,
        "campaign": "C117",
        "created_at_utc": now(),
        "status": "whole_part_exception_fourth_family_observation_validation_complete",
        "headline": {
            "discovery_winner": discovery["winner"],
            "behavior": core.load(OUT / "analysis/capture_summary.json")["behavior"],
            "field_metrics": validation["field_metrics"],
            "field_checks": validation["field_checks"],
            "common_component_residual": validation["common_component_residual"],
            "predictions": validation["predictions"],
            "prediction_checks": validation["prediction_checks"],
            "all_primary_gates_passed": validation["all_primary_gates_passed"],
        },
        "new_puzzles": {
            "K307": "discovery-frozen controlled default-exception relation-role-state candidate and confirmation/lockbox outcome",
            "K308": "fourth-family exact-energy coordinate-assignment result with route descriptors kept separate",
            "K309-BOUNDARY": "raw common truth component and family residual are descriptive decompositions, not proven orthogonal semantic factors",
        },
        "theory_update": "RDC now has four relation-family observations. Shared truth-aligned response, physical coordinate assignment, role-state selection and output protocol remain separately typed; neither a universal route nor a relation-specific topology is assumed.",
        "unified_formula": "y = O_c(L_{S_f,V_f,C_f,P,s}(R[f,r,s](x))); R_f = alpha_f G + E_f (descriptive decomposition only)",
        "problems": [
            "one synthetic default-exception template and one Qwen3 prompt grammar",
            "the handbook statement is background while the exception log directly determines the answer, so the task may reduce to local retain/lack reading",
            "discovery searches only states 1-30 and can select a late decision candidate",
            "simultaneous activation patching can be off-manifold and does not identify an endogenous route",
            "256 coordinates and eight permutations do not establish minimality, semantic specificity or a shared assignment algorithm",
            "common-component subtraction is a descriptive projection chosen from three old family means, not an identified orthogonal semantic basis",
        ],
        "heatmap": {
            "path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"),
            "bytes": PUBLIC.stat().st_size,
            "sha256": core.sha(PUBLIC),
            "activation_coordinates": 2560,
            "includes_embedding": True,
        },
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": "C118 observation-first comparator-family breadth campaign only after a fresh contract separates continuous comparison dimensions from output truth and registers no orthogonality or manifold expectation as a required result",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "validation": core.load(OUT / "audit/independent_validation_audit.json")["all_checks_passed"],
        "effects": len(effect_rows) == 3 * len(ROLES) * len(display_states) and all(len(row["values"]) == 2560 for row in effect_rows),
        "raw": len(raw_rows) == 24 and all(len(row["values"]) == 2560 for row in raw_rows),
        "candidate_visible": any(row["role"] == nomination["role"] and row["state"] == nomination["state"] for row in effect_rows),
        "asset": core.sha(canonical) == core.sha(PUBLIC),
        "batch": all(key in payload for key in ("c115_batch", "c116_batch", "c117_batch")),
        "boundary": "not weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1639,
        "campaign": "C117",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "asset_sha256": core.sha(PUBLIC),
        "authorization": "audit_frontend_append_c117_memo_then_consider_c118",
    }
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"checks": checks, "headline": closure["headline"], "next_authorization": closure["next_authorization"]}, indent=2))


STAGES = {"contract": contract, "capture": capture, "discover": discover, "validate": validate, "synthesize": synthesize}


def main(stage: str) -> None:
    STAGES[stage]()
