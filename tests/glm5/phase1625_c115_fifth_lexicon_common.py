#!/usr/bin/env python3
"""C115 fifth-lexicon prospective replication shared implementation."""
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
OUT = RESULT / "phase1625_c115_fifth_lexicon_prospective_replication"
C110 = RESULT / "phase1607_c110_fresh_readout_control_separation"
C113 = RESULT / "phase1618_c113_fourth_lexicon_role_lattice_replication"
C114 = RESULT / "phase1623_c114_existing_data_structural_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base
import phase1608_c110_exact_field_capture as capture_base
import phase1610_c110_frozen_transport_comparison as transport

FAMILIES = ("attribute_binding", "agent_patient")
PARTITIONS = ("fifth_confirmation", "fifth_lockbox")
ROLES = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary")
PATH_ROLES = ("focus_record", "focus_post", "query_focus", "query_anchor")
KEY_STATES = (0, 4, 8, 12, 16, 19, 24, 28, 32, 36)
RAW_STATES = (0, 8, 16, 19, 24, 32, 36)
STATES = 37
DIM = 2560
WIDTH = 224
BATCH_SIZE = 8
STEMS = (
    "bav", "ced", "dor", "fen", "gal", "hir", "jas", "kel",
    "lum", "mor", "nav", "pel", "qor", "rav", "sel", "tor",
    "ulm", "val", "wen", "xal", "yor", "zel", "bri", "cru",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def inventories() -> dict[str, list[tuple[str, str, str, str, str]]]:
    attribute, agent = [], []
    for index, stem in enumerate(STEMS):
        rotations = [STEMS[(index + offset) % len(STEMS)] for offset in (0, 5, 11, 17, 21)]
        attribute.append((
            f"quora{rotations[0]}meter", f"xena{rotations[1]}hue", f"viro{rotations[2]}node",
            f"pala{rotations[3]}seal", f"mira{rotations[4]}gate",
        ))
        agent.append((
            f"Al{rotations[0]}en", f"Be{rotations[1]}ra", f"Co{rotations[2]}is",
            f"De{rotations[3]}la", f"re{rotations[4]}scribed",
        ))
    return {"attribute_binding": attribute, "agent_patient": agent}


def build_material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for family in FAMILIES:
        for unit_index, values in enumerate(inventories()[family]):
            partition = PARTITIONS[unit_index // 12]
            unit = {
                "arm": "breadth", "unit_id": f"c115-{family}-{unit_index:02d}", "family": family,
                "world": "controlled_synthetic_fifth_lexicon", "partition": partition,
                "surface": "factorial", "values": list(values),
            }
            units.append(unit)
            for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
                prompt, focus, anchor = breadth_base.breadth_prompt(family, values, truth, surface, distractor, code)
                output_yes = (truth == 1) == (code == 1)
                cases.append({
                    **unit, "case_id": f"c115-{len(cases):04d}", "truth_factor": truth,
                    "surface_factor": surface, "distractor_factor": distractor, "code": code,
                    "codebook": graph_base.CODEBOOKS[code]["name"], "truth": truth == 1,
                    "output_yes": output_yes, "gold_position": 0 if output_yes else 1,
                    "focus": focus, "anchor": anchor, "prompt": prompt,
                })
    return units, cases


def historical_values() -> set[str]:
    paths = [
        RESULT / "phase1575_c101_dual_arm/material/breadth_units.jsonl",
        RESULT / "phase1581_c102_typed_relation_coordinate_campaign/material/breadth_units.jsonl",
        RESULT / "phase1589_c104_upstream_candidate_validation/material/units.jsonl",
        RESULT / "phase1600_c108_fresh_coordinate_causality/material/units.jsonl",
        C110 / "material/units.jsonl", C113 / "material/units.jsonl",
    ]
    return {str(value).casefold() for path in paths for row in core.rows(path) for value in row["values"]}


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C115 already exists: {OUT}")
    closure = core.load(C114 / "analysis/closure.json")
    closure_audit = core.load(C114 / "audit/independent_closure_audit.json")
    expected = "C115 fifth-lexicon prospective test using the frozen C114 prediction template"
    if not closure_audit["all_checks_passed"] or not closure["next_authorization"].startswith(expected):
        raise RuntimeError("C115 authorization missing")
    source_protocol = core.load(C110 / "protocol/preregistration.json")
    units, cases = build_material()
    tok = graph_base.tokenizer()
    compiled = breadth_base.compile_breadth(tok, cases)
    zero = breadth_base.zero_models(cases, True)
    old = historical_values()
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    occurrences = []
    disjoint = True
    for row_index, row in enumerate(compiled):
        occupied = []
        for role in ROLES:
            positions = [int(value) for value in row["role_positions"][role]]
            occupied.extend(positions)
            for subtoken, position in enumerate(positions):
                token_id = int(row["prompt_ids"][position])
                occurrences.append({
                    "occurrence_index": len(occurrences), "row_index": row_index, "case_id": row["case_id"],
                    "unit_id": row["unit_id"], "family": row["family"], "partition": row["partition"],
                    "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"],
                    "distractor_factor": row["distractor_factor"], "code": row["code"], "role": role,
                    "subtoken": subtoken, "span_length": len(positions), "token_position": position,
                    "token_id": token_id, "token_text": tok.convert_ids_to_tokens([token_id])[0],
                })
        disjoint = disjoint and len(occupied) == len(set(occupied))
    cells = Counter((row["family"], row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases)
    rng = np.random.default_rng(1625)
    movement_permutations = {
        family: [rng.permutation(k).astype(int).tolist() for _ in range(8)]
        for family, k in (("attribute_binding", 256), ("agent_patient", 128))
    }
    role_coalitions = {
        "record_to_query_path": list(PATH_ROLES),
        "path_plus_code": [*PATH_ROLES, "code_instruction"],
        "path_plus_code_boundary": [*PATH_ROLES, "code_instruction", "boundary"],
        "all_registered_roles": list(ROLES),
        **{f"path_without_{role}": [value for value in PATH_ROLES if value != role] for role in PATH_ROLES},
    }
    supports = source_protocol["supports"]
    checks = {
        "authorization": closure_audit["all_checks_passed"], "units": len(units) == 48,
        "cases": len(cases) == 768,
        "partitions": Counter((row["family"], row["partition"]) for row in units) == {(family, partition): 12 for family in FAMILIES for partition in PARTITIONS},
        "factorial": cells == {(family, partition, *cell): 12 for family in FAMILIES for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=4)},
        "zero_models": all(abs(value - 0.5) < 1e-12 for key, value in zero.items() if key != "truth_x_code_oracle") and zero["truth_x_code_oracle"] == 1.0,
        "semantic_uniqueness": len({row["prompt"] for row in cases}) == 768,
        "freshness": not (set(fresh) & old), "value_uniqueness": len(fresh) == len(set(fresh)),
        "compiled": len(compiled) == 768 and all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled) and disjoint,
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "supports": len(supports["attribute_binding_k256"]) == 256 and len(supports["agent_patient_k128"]) == 128,
        "machine_naturalness": all(row["prompt"].count("Query:") == 1 and row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.write_rows(OUT / "protocol/role_occurrence_manifest.jsonl", occurrences)
    protocol = {
        "phase": 1625, "campaign": "C115", "created_at_utc": now(),
        "status": "fifth_lexicon_large_sample_prospective_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "prospective fifth-lexicon replication of full truth fields, graded coordinate assignment, and query-centered role coalitions",
        "families": list(FAMILIES), "partitions": list(PARTITIONS), "units": 48, "cases": 768,
        "independent_units_per_family_partition": 12,
        "roles": list(ROLES), "states": STATES, "activation_coordinates": DIM, "occurrences": len(occurrences),
        "archive": {"path": "raw/qwen3_role_subtoken_all_states.uint16.npy", "shape": [STATES, len(occurrences), DIM], "dtype": "uint16 exact BF16 bit patterns", "fixed_width": WIDTH, "batch_size": BATCH_SIZE},
        "supports": supports, "movement_permutations": movement_permutations,
        "role_coalitions": role_coalitions,
        "modes": ["frozen_support"] + [f"movement_permutation_{index}" for index in range(8)] + [f"single_{role}" for role in ROLES] + [f"coalition_{name}" for name in role_coalitions],
        "frozen_field_prediction": {"role": "query_anchor", "state": 19, "cross_partition_cosine_min": 0.90, "each_partition_to_c110_reference_cosine_min": 0.85, "each_partition_frozen_support_topk_overlap_min": 0.50},
        "frozen_predictions": closure["c115_frozen_prediction_template"],
        "operational_gates": {
            "both_field_families_pass": True,
            "both_families_correct_movement_gt_permutation_median_cells": 4,
            "agent_record_path_gt_query_cells": 4,
            "agent_query_anchor_positive_cells": 4,
            "agent_query_focus_positive_cells": 4,
            "agent_leave_query_anchor_lowers_cells": 4,
            "agent_leave_query_focus_lowers_cells": 4,
            "agent_leave_focus_post_lowers_cells": 4,
        },
        "descriptive_only": ["strict victory over all eight coordinate permutations", "code_instruction increment", "boundary increment", "focus_record contribution"],
        "observation_first": "capture and adjudicate embedding-to-state36 role fields before interventions; intervention candidates cannot rewrite field gates",
        "behavior_policy": "report standard and reversed code separately; reversed failure is typed missingness for output-task claims and does not erase upstream truth-field observations",
        "completion_rule": "finish every registered observation and intervention route; a failed route retires only that route",
        "numeric": {"movement_permutation_actual_l2_relative_tolerance": 0.02, "fixed_width": WIDTH, "batch_size": BATCH_SIZE},
        "typed_missingness": {"human_naturalness": "no independent blind rating", "cross_model": "Qwen3 only", "natural_route": "simultaneous role patching does not identify endogenous transport order"},
        "claim_boundary": "controlled-English activation-coordinate replication only; activation coordinates are not weights or independent semantic neurons; finite permutations do not establish an equivalence relation, group, gauge symmetry, topology, minimality, natural route, or universal-language mechanism; no attention/MLP analysis",
        "source_paths": {"c110_protocol": str(C110 / "protocol/preregistration.json"), "c110_mean_field": str(C110 / "analysis/mean_truth_role_state.float32.npy"), "c114_closure": str(C114 / "analysis/closure.json"), "c114_audit": str(C114 / "audit/independent_closure_audit.json")},
        "source_hashes": {"c110_protocol": core.sha(C110 / "protocol/preregistration.json"), "c110_mean_field": core.sha(C110 / "analysis/mean_truth_role_state.float32.npy"), "c114_closure": core.sha(C114 / "analysis/closure.json"), "c114_audit": core.sha(C114 / "audit/independent_closure_audit.json")},
        "material_digest": core.digest([*units, *cases]), "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1626_c115_exact_field_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"phase": 1625, "campaign": "C115", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "occurrences": len(occurrences), "max_width": max(len(row["prompt_ids"]) for row in compiled), "material_digest": protocol["material_digest"], "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_pre_model_audit.json", audit)
    print(json.dumps(audit, indent=2))


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/independent_pre_model_audit.json")
    if protocol["authorization"] != "execute_phase1626_c115_exact_field_capture" or not audit["all_checks_passed"]:
        raise RuntimeError("C115 capture authorization missing")
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
    if raw_path.exists() or logits_path.exists() or index_path.exists():
        raise RuntimeError("C115 raw archive already exists")
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=tuple(protocol["archive"]["shape"]))
    candidate_logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    behavior, model, first_rows = [], None, None
    repeat_hidden = repeat_logits = 0.0
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        batch_size = int(protocol["archive"]["batch_size"])
        width = int(protocol["archive"]["fixed_width"])
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            output, logits, ids, mask, positions, lengths = capture_base.forward(model, batch, pad, device, width)
            if len(output.hidden_states) != STATES or output.hidden_states[-1].shape[-1] != DIM:
                raise RuntimeError((len(output.hidden_states), output.hidden_states[-1].shape))
            for state_index, state in enumerate(output.hidden_states):
                if state.dtype != torch.bfloat16 or not bool(torch.isfinite(state).all()):
                    raise RuntimeError((state_index, state.dtype))
                for local in range(len(batch)):
                    row_index = start + local
                    occurrences = by_row[row_index]
                    occurrence_indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                    token_positions = [int(item["token_position"]) for item in occurrences]
                    field[state_index, occurrence_indices] = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
            for local, row in enumerate(batch):
                row_index = start + local
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                candidate_logits[row_index] = scores
                prediction = int(scores[1] > scores[0])
                behavior.append({
                    "row_index": row_index, "case_id": row["case_id"], "unit_id": row["unit_id"],
                    "family": row["family"], "partition": row["partition"], "truth_factor": row["truth_factor"],
                    "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"],
                    "code": row["code"], "gold_position": row["gold_position"], "prediction": prediction,
                    "correct": prediction == row["gold_position"], "yes_minus_no": scores[0] - scores[1],
                })
            if start == 0:
                first_rows = batch
            if (start // batch_size + 1) % 12 == 0:
                field.flush(); candidate_logits.flush()
                print(f"[phase1626] captured {start + len(batch)}/{len(rows)} cases", flush=True)
            del output, logits, ids, mask, positions
        field.flush(); candidate_logits.flush()
        output, logits, ids, mask, positions, lengths = capture_base.forward(model, first_rows, pad, device, width)
        for state_index, state in enumerate(output.hidden_states):
            for local in range(len(first_rows)):
                occurrences = by_row[local]
                indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                token_positions = [int(item["token_position"]) for item in occurrences]
                old_bits = np.asarray(field[state_index, indices], dtype=np.uint16)
                new_bits = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
                if not np.array_equal(old_bits, new_bits):
                    repeat_hidden = max(repeat_hidden, float(np.max(np.abs(decode_bf16(old_bits) - decode_bf16(new_bits)))))
        for local, row in enumerate(first_rows):
            scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
            repeat_logits = max(repeat_logits, float(np.max(np.abs(scores - candidate_logits[local]))))
        del output, logits, ids, mask, positions
    finally:
        field.flush(); candidate_logits.flush()
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
                causal_prefix = max(causal_prefix, float(np.max(np.abs(decode_bf16(left) - decode_bf16(right)))))
        unit_rows = [rows[index] for index in row_indices]
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            standard = next(index for index, row in zip(row_indices, unit_rows, strict=True) if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, 1))
            reversed_code = next(index for index, row in zip(row_indices, unit_rows, strict=True) if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, -1))
            for role in ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor"):
                left = capture_base.role_bits(field, lookup, standard, role)
                right = capture_base.role_bits(field, lookup, reversed_code, role)
                if not np.array_equal(left, right):
                    code_previsible = max(code_previsible, float(np.max(np.abs(decode_bf16(left) - decode_bf16(right)))))
    def acc(selected: list[dict]) -> float:
        return float(np.mean([row["correct"] for row in selected]))
    behavior_summary = {
        "global_accuracy": acc(behavior),
        "by_family": {family: acc([row for row in behavior if row["family"] == family]) for family in protocol["families"]},
        "by_partition": {partition: acc([row for row in behavior if row["partition"] == partition]) for partition in protocol["partitions"]},
        "by_code": {str(code): acc([row for row in behavior if row["code"] == code]) for code in (1, -1)},
    }
    checks = {
        "shape": list(field.shape) == protocol["archive"]["shape"], "dtype": field.dtype == np.uint16,
        "logits": candidate_logits.shape == (len(rows), 2) and candidate_logits.dtype == np.float32 and bool(np.isfinite(candidate_logits).all()),
        "index": len(behavior) == len(rows) and all(row["row_index"] == index for index, row in enumerate(behavior)),
        "repeat_hidden": repeat_hidden == 0.0, "repeat_logits": repeat_logits == 0.0,
        "causal_prefix": causal_prefix == 0.0, "code_previsible": code_previsible == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1626, "campaign": "C115", "created_at_utc": now(), "status": "fifth_lexicon_exact_bf16_field_capture_complete",
        "producer_sha256": core.sha(Path(__file__)), "shape": list(field.shape), "raw_file_bytes": raw_path.stat().st_size,
        "raw_data_bytes": int(field.nbytes), "raw_sha256": core.sha(raw_path), "logits_sha256": core.sha(logits_path),
        "index_sha256": core.sha(index_path), "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logits_max_abs": repeat_logits, "causal_prefix_max_abs": causal_prefix, "code_previsible_max_abs": code_previsible},
        "behavior": behavior_summary, "runtime": {"placement": placement, "quantization": quant}, "checks": checks,
        "authorization": "run_phase1627_c115_field_adjudication",
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def adjudicate_field() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    audit = core.load(OUT / "audit/independent_capture_audit.json")
    if capture_report["authorization"] != "run_phase1627_c115_field_adjudication" or not audit["all_checks_passed"]:
        raise RuntimeError("C115 field authorization missing")
    field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    roles, families, partitions = protocol["roles"], protocol["families"], protocol["partitions"]
    role_index = {role: index for index, role in enumerate(roles)}
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(int(occurrence["occurrence_index"]))
    unit_path = OUT / "analysis/unit_truth_role_state.float32.npy"
    mean_path = OUT / "analysis/mean_truth_role_state.float32.npy"
    if unit_path.exists() or mean_path.exists():
        raise RuntimeError("C115 derived field already exists")
    unit_truth = np.lib.format.open_memmap(unit_path, mode="w+", dtype=np.float32, shape=(len(units), len(roles), STATES, DIM))
    unit_truth[:] = 0.0
    for state in range(STATES):
        for row_index, row in enumerate(rows):
            coefficient = float(row["truth_factor"]) / 16.0
            u = unit_index[row["unit_id"]]
            for role in roles:
                values = decode_bf16(field[state, lookup[(row_index, role)]])
                unit_truth[u, role_index[role], state] += coefficient * np.mean(values, axis=0, dtype=np.float32)
        if state % 6 == 0 or state == 36:
            unit_truth.flush(); print(f"[phase1627] derived state {state}/36", flush=True)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        groups[(unit["family"], unit["partition"])].append(index)
    mean_truth = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=(2, 2, len(roles), STATES, DIM))
    for family_index, family in enumerate(families):
        for partition_index, partition in enumerate(partitions):
            mean_truth[family_index, partition_index] = np.mean(unit_truth[groups[(family, partition)]], axis=0, dtype=np.float32)
    mean_truth.flush()
    old_mean = np.load(C110 / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    prediction = protocol["frozen_field_prediction"]
    state, role_i = int(prediction["state"]), role_index[prediction["role"]]
    results = []
    for family_index, family in enumerate(families):
        vectors = [np.asarray(mean_truth[family_index, partition_index, role_i, state], dtype=np.float32) for partition_index in range(2)]
        reference = np.mean(np.asarray(old_mean[family_index, :, role_i, state], dtype=np.float32), axis=0, dtype=np.float32)
        support = protocol["supports"]["attribute_binding_k256" if family == "attribute_binding" else "agent_patient_k128"]
        support_set = set(support)
        cross_cos = cosine(vectors[0], vectors[1])
        reference_cos = [cosine(vector, reference) for vector in vectors]
        overlaps = [len(topk(vector, len(support)) & support_set) / len(support) for vector in vectors]
        gates = {
            "cross_partition": cross_cos >= float(prediction["cross_partition_cosine_min"]),
            "reference": all(value >= float(prediction["each_partition_to_c110_reference_cosine_min"]) for value in reference_cos),
            "support_overlap": all(value >= float(prediction["each_partition_frozen_support_topk_overlap_min"]) for value in overlaps),
        }
        results.append({
            "family": family, "role": prediction["role"], "state": state, "k": len(support),
            "cross_partition_cosine": cross_cos,
            "partition_to_c110_reference_cosine": dict(zip(partitions, reference_cos, strict=True)),
            "frozen_support_topk_overlap": dict(zip(partitions, overlaps, strict=True)),
            "norms": dict(zip(partitions, [float(np.linalg.norm(vector)) for vector in vectors], strict=True)),
            "gates": gates, "prediction_passed": all(gates.values()),
        })
    result_path = OUT / "analysis/field_prediction_results.jsonl"
    core.write_rows(result_path, results)
    trajectory = []
    for family_index, family in enumerate(families):
        for role, r in role_index.items():
            for state_i in range(STATES):
                left = np.asarray(mean_truth[family_index, 0, r, state_i], dtype=np.float32)
                right = np.asarray(mean_truth[family_index, 1, r, state_i], dtype=np.float32)
                trajectory.append({"family": family, "role": role, "state": state_i, "state_kind": "embedding" if state_i == 0 else "hidden_state", "cross_partition_cosine": cosine(left, right), "confirmation_norm": float(np.linalg.norm(left)), "lockbox_norm": float(np.linalg.norm(right))})
    trajectory_path = OUT / "analysis/role_state_trajectory.jsonl"
    core.write_rows(trajectory_path, trajectory)
    checks = {
        "source": core.sha(OUT / protocol["archive"]["path"]) == capture_report["raw_sha256"],
        "unit_shape": list(unit_truth.shape) == [48, 7, 37, 2560], "mean_shape": list(mean_truth.shape) == [2, 2, 7, 37, 2560],
        "finite": bool(np.isfinite(unit_truth).all() and np.isfinite(mean_truth).all()),
        "results": len(results) == 2, "trajectory": len(trajectory) == 518,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1627, "campaign": "C115", "created_at_utc": now(), "status": "fifth_lexicon_field_prediction_adjudicated",
        "results": results, "passed_families": [row["family"] for row in results if row["prediction_passed"]],
        "interpretation": "field stability is an upstream readout result and remains separate from intervention leverage",
        "checks": checks, "producer_sha256": core.sha(Path(__file__)), "unit_sha256": core.sha(unit_path),
        "mean_sha256": core.sha(mean_path), "results_sha256": core.sha(result_path), "trajectory_sha256": core.sha(trajectory_path),
        "authorization": "execute_phase1628_c115_coordinate_and_role_interventions_regardless_of_field_gate",
    }
    core.save(OUT / "analysis/field_adjudication.json", report)
    print(json.dumps({"checks": checks, "results": results}, indent=2))


@torch.inference_mode()
def intervene() -> None:
    contract_data = core.load(OUT / "protocol/preregistration.json")
    field_report = core.load(OUT / "analysis/field_adjudication.json")
    field_audit = core.load(OUT / "audit/independent_field_adjudication_audit.json")
    if field_report["authorization"] != "execute_phase1628_c115_coordinate_and_role_interventions_regardless_of_field_gate" or not field_audit["all_checks_passed"]:
        raise RuntimeError("C115 intervention authorization missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    pairs = transport.build_pairs(rows, contract_data)
    for index, pair in enumerate(pairs):
        pair["pair_id"] = f"c115-pair-{index:04d}"
    if len(pairs) != 384:
        raise RuntimeError(len(pairs))
    all_roles = tuple(contract_data["roles"])
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        lengths = tuple(len(pair["recipient"]["role_positions"][role]) for role in all_roles)
        grouped[(pair["family"], pair["partition"], pair["code"], lengths)].append(pair)
    results, model, first_repeat = [], None, None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        width = int(contract_data["numeric"]["fixed_width"])
        for (family, partition, code, lengths), group in sorted(grouped.items()):
            support_name = "attribute_binding_k256" if family == "attribute_binding" else "agent_patient_k128"
            support = torch.tensor(contract_data["supports"][support_name], dtype=torch.long, device=device)
            permutations = [torch.tensor(values, dtype=torch.long, device=device) for values in contract_data["movement_permutations"][family]]
            for start in range(0, len(group), int(contract_data["numeric"]["batch_size"])):
                batch = group[start:start + int(contract_data["numeric"]["batch_size"])]
                recipients, donors = [pair["recipient"] for pair in batch], [pair["donor"] for pair in batch]
                recipient_logits, recipient_states = transport.forward_with_roles(model, recipients, all_roles, 19, pad, device, width)
                donor_logits, donor_states = transport.forward_with_roles(model, donors, all_roles, 19, pad, device, width)
                if first_repeat is None:
                    first_repeat = (recipients, recipient_logits.detach().clone(), {role: value.detach().clone() for role, value in recipient_states.items()})
                rq, dq = recipient_states["query_anchor"], donor_states["query_anchor"]
                target_delta = dq[..., support] - rq[..., support]
                target_norm = torch.sqrt(torch.sum(target_delta.float() ** 2, dim=(1, 2))).clamp_min(1e-12)
                patches: dict[str, dict[str, torch.Tensor]] = {}
                value = rq.clone(); value[..., support] = dq[..., support]
                patches["frozen_support"] = {"query_anchor": value}
                permutation_norms = []
                for permutation_index, permutation in enumerate(permutations):
                    value = rq.clone()
                    value[..., support] = (rq[..., support].float() + target_delta.float()[..., permutation]).to(rq.dtype)
                    patches[f"movement_permutation_{permutation_index}"] = {"query_anchor": value}
                    permutation_norms.append(torch.sqrt(torch.sum((value[..., support] - rq[..., support]).float() ** 2, dim=(1, 2))).clamp_min(1e-12))
                for role in all_roles:
                    patches[f"single_{role}"] = {role: donor_states[role].clone()}
                for name, roles in contract_data["role_coalitions"].items():
                    patches[f"coalition_{name}"] = {role: donor_states[role].clone() for role in roles}
                patched_logits = {mode: transport.forward_patched_roles(model, recipients, values, 19, pad, device, width) for mode, values in patches.items()}
                for local, pair in enumerate(batch):
                    base = transport.margin(recipient_logits[local], recipients[local])
                    donor_margin = transport.margin(donor_logits[local], donors[local])
                    mode_results = {}
                    for mode, logits in patched_logits.items():
                        patched = transport.margin(logits[local], recipients[local]); gain = patched - base
                        mode_results[mode] = {"yes_minus_no": patched, "truth_direction_gain": gain, "code_aligned_task_gain": code * gain, "truth_flip": base <= 0.0 < patched, "task_flip": code * base <= 0.0 < code * patched}
                    results.append({
                        "pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "family": family, "partition": partition,
                        "code": code, "surface_factor": pair["surface_factor"], "distractor_factor": pair["distractor_factor"],
                        "recipient_yes_minus_no": base, "donor_yes_minus_no": donor_margin, "target_movement_l2": float(target_norm[local]),
                        "permutation_l2_relative_errors": [float(torch.abs(norm[local] - target_norm[local]) / target_norm[local]) for norm in permutation_norms],
                        "role_l2": {role: float(torch.sqrt(torch.sum((donor_states[role][local] - recipient_states[role][local]).float() ** 2))) for role in all_roles},
                        "modes": mode_results,
                    })
                print(f"[phase1628] {family}/{partition}/code={code}/lengths={lengths} {start + len(batch)}/{len(group)}", flush=True)
        repeat_rows, old_logits, old_states = first_repeat
        new_logits, new_states = transport.forward_with_roles(model, repeat_rows, all_roles, 19, pad, device, width)
        repeat_logits = float(torch.max(torch.abs(new_logits - old_logits)))
        repeat_hidden = max(float(torch.max(torch.abs(new_states[role] - old_states[role]))) for role in all_roles)
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()
    result_path = OUT / "analysis/intervention_results.jsonl"
    core.write_rows(result_path, results)
    summaries = []
    for family in contract_data["families"]:
        for partition in contract_data["partitions"]:
            for code in (1, -1):
                selected = [row for row in results if row["family"] == family and row["partition"] == partition and row["code"] == code]
                correct = med([row["modes"]["frozen_support"]["truth_direction_gain"] for row in selected])
                permutation_medians = [med([row["modes"][f"movement_permutation_{index}"]["truth_direction_gain"] for row in selected]) for index in range(8)]
                role_medians = {role: med([row["modes"][f"single_{role}"]["truth_direction_gain"] for row in selected]) for role in all_roles}
                coalition_medians = {name: med([row["modes"][f"coalition_{name}"]["truth_direction_gain"] for row in selected]) for name in contract_data["role_coalitions"]}
                leave_effects = {role: coalition_medians["record_to_query_path"] - coalition_medians[f"path_without_{role}"] for role in PATH_ROLES}
                summaries.append({
                    "family": family, "partition": partition, "code": code, "pairs": len(selected), "independent_units": len({row["unit_id"] for row in selected}),
                    "frozen_support_median_gain": correct, "movement_permutation_median_gains": permutation_medians,
                    "movement_permutation_median_of_medians": med(permutation_medians),
                    "frozen_support_gt_permutation_median": correct > med(permutation_medians),
                    "frozen_support_gt_all_permutation_medians": all(correct > value for value in permutation_medians),
                    "single_role_median_gains": role_medians, "coalition_median_gains": coalition_medians,
                    "leave_one_path_median_losses": leave_effects,
                    "record_path_gt_query": coalition_medians["record_to_query_path"] > role_medians["query_anchor"],
                    "all_roles_gt_path": coalition_medians["all_registered_roles"] > coalition_medians["record_to_query_path"],
                    "leave_query_anchor_lowers": leave_effects["query_anchor"] > 0,
                    "leave_query_focus_lowers": leave_effects["query_focus"] > 0,
                    "leave_focus_post_lowers": leave_effects["focus_post"] > 0,
                    "staged_increments": {"code_over_path": coalition_medians["path_plus_code"] - coalition_medians["record_to_query_path"], "boundary_over_path_code": coalition_medians["path_plus_code_boundary"] - coalition_medians["path_plus_code"], "all_over_path_code_boundary": coalition_medians["all_registered_roles"] - coalition_medians["path_plus_code_boundary"]},
                    "truth_flip_rates": {mode: float(np.mean([row["modes"][mode]["truth_flip"] for row in selected])) for mode in contract_data["modes"]},
                })
    summary_path = OUT / "analysis/intervention_summary.jsonl"
    core.write_rows(summary_path, summaries)
    attr = [row for row in summaries if row["family"] == "attribute_binding"]
    agent = [row for row in summaries if row["family"] == "agent_patient"]
    predictions = {
        "attribute_median_win_cells": int(sum(row["frozen_support_gt_permutation_median"] for row in attr)),
        "agent_median_win_cells": int(sum(row["frozen_support_gt_permutation_median"] for row in agent)),
        "attribute_strict_win_cells_descriptive": int(sum(row["frozen_support_gt_all_permutation_medians"] for row in attr)),
        "agent_strict_win_cells_descriptive": int(sum(row["frozen_support_gt_all_permutation_medians"] for row in agent)),
        "agent_record_path_gt_query_cells": int(sum(row["record_path_gt_query"] for row in agent)),
        "agent_query_anchor_positive_cells": int(sum(row["single_role_median_gains"]["query_anchor"] > 0 for row in agent)),
        "agent_query_focus_positive_cells": int(sum(row["single_role_median_gains"]["query_focus"] > 0 for row in agent)),
        "agent_leave_query_anchor_lowers_cells": int(sum(row["leave_query_anchor_lowers"] for row in agent)),
        "agent_leave_query_focus_lowers_cells": int(sum(row["leave_query_focus_lowers"] for row in agent)),
        "agent_leave_focus_post_lowers_cells": int(sum(row["leave_focus_post_lowers"] for row in agent)),
    }
    gates = contract_data["operational_gates"]
    prediction_checks = {
        "both_field_families_pass": len(field_report["passed_families"]) == 2,
        "both_families_correct_movement_gt_permutation_median_cells": predictions["attribute_median_win_cells"] == gates["both_families_correct_movement_gt_permutation_median_cells"] and predictions["agent_median_win_cells"] == gates["both_families_correct_movement_gt_permutation_median_cells"],
        **{key: predictions[key] == value for key, value in gates.items() if key not in {"both_field_families_pass", "both_families_correct_movement_gt_permutation_median_cells"}},
    }
    max_l2_error = max(error for row in results for error in row["permutation_l2_relative_errors"])
    checks = {
        "rows": len(results) == 384, "modes": all(set(row["modes"]) == set(contract_data["modes"]) for row in results),
        "summary": len(summaries) == 8 and all(row["pairs"] == 48 and row["independent_units"] == 12 for row in summaries),
        "l2_preserved": max_l2_error <= float(contract_data["numeric"]["movement_permutation_actual_l2_relative_tolerance"]),
        "finite": all(math.isfinite(row["modes"][mode]["truth_direction_gain"]) for row in results for mode in contract_data["modes"]),
        "repeat_hidden": repeat_hidden == 0.0, "repeat_logits": repeat_logits == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "sources_unchanged": all(core.sha(Path(contract_data["source_paths"][name])) == digest for name, digest in contract_data["source_hashes"].items()),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_l2_error": max_l2_error})
    report = {
        "phase": 1628, "campaign": "C115", "created_at_utc": now(), "status": "fifth_lexicon_coordinate_role_interventions_complete",
        "checks": checks, "prediction_checks": prediction_checks, "all_frozen_predictions_passed": all(prediction_checks.values()),
        "max_permutation_l2_relative_error": max_l2_error, "predictions": predictions, "summaries": summaries,
        "runtime": {"placement": placement, "quantization": quant}, "producer_sha256": core.sha(Path(__file__)),
        "results_sha256": core.sha(result_path), "summary_sha256": core.sha(summary_path),
        "claim_boundary": contract_data["claim_boundary"], "authorization": "run_phase1629_c115_synthesis_heatmap_and_closure",
    }
    core.save(OUT / "analysis/intervention_adjudication.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"summaries", "runtime"}}, indent=2))


def family_rollup(rows: list[dict], summaries: list[dict], family: str) -> dict:
    selected = [row for row in rows if row["family"] == family]
    cells = [row for row in summaries if row["family"] == family]
    modes = {"query": "single_query_anchor", "path": "coalition_record_to_query_path", "path_code": "coalition_path_plus_code", "all": "coalition_all_registered_roles"}
    return {
        "pairs": len(selected), "independent_units": len({row["unit_id"] for row in selected}),
        "beats_permutation_median_cells": int(sum(row["frozen_support_gt_permutation_median"] for row in cells)),
        "strictly_beats_all_permutations_cells": int(sum(row["frozen_support_gt_all_permutation_medians"] for row in cells)),
        "frozen_support_median_gain_range": [min(row["frozen_support_median_gain"] for row in cells), max(row["frozen_support_median_gain"] for row in cells)],
        "single_role_median_ranges": {role: [min(row["single_role_median_gains"][role] for row in cells), max(row["single_role_median_gains"][role] for row in cells)] for role in cells[0]["single_role_median_gains"]},
        "coalition_median_ranges": {name: [min(row["coalition_median_gains"][name] for row in cells), max(row["coalition_median_gains"][name] for row in cells)] for name in cells[0]["coalition_median_gains"]},
        "leave_one_path_loss_ranges": {role: [min(row["leave_one_path_median_losses"][role] for row in cells), max(row["leave_one_path_median_losses"][role] for row in cells)] for role in cells[0]["leave_one_path_median_losses"]},
        "staged_increment_ranges": {name: [min(row["staged_increments"][name] for row in cells), max(row["staged_increments"][name] for row in cells)] for name in cells[0]["staged_increments"]},
        **{f"{name}_truth_flips": int(sum(row["modes"][mode]["truth_flip"] for row in selected)) for name, mode in modes.items()},
    }


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    field_report = core.load(OUT / "analysis/field_adjudication.json")
    intervention = core.load(OUT / "analysis/intervention_adjudication.json")
    audit = core.load(OUT / "audit/independent_intervention_audit.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    if intervention["authorization"] != "run_phase1629_c115_synthesis_heatmap_and_closure" or not audit["all_checks_passed"]:
        raise RuntimeError("C115 closure authorization missing")
    rows = core.rows(OUT / "analysis/intervention_results.jsonl")
    summaries = core.rows(OUT / "analysis/intervention_summary.jsonl")
    rollup = {family: family_rollup(rows, summaries, family) for family in protocol["families"]}
    payload = core.load(PUBLIC)
    mean = np.load(OUT / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    role_index = {role: index for index, role in enumerate(protocol["roles"])}
    effect_rows = []
    for family_index, family in enumerate(protocol["families"]):
        for partition_index, partition in enumerate(protocol["partitions"]):
            for role in protocol["roles"]:
                for state in KEY_STATES:
                    effect_rows.append({"dataset": "C115", "family": family, "partition": partition, "role": role, "state": state, "state_kind": "embedding" if state == 0 else "hidden_state", "effect": "balanced_truth_walsh", "values": np.asarray(mean[family_index, partition_index, role_index[role], state], dtype=np.float32).tolist()})
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    raw_field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    lookup: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for occurrence in manifest:
        lookup[(int(occurrence["row_index"]), occurrence["role"])].append(occurrence)
    raw_rows = []
    for family in protocol["families"]:
        for partition in protocol["partitions"]:
            row_index = next(index for index, row in enumerate(compiled) if row["family"] == family and row["partition"] == partition)
            row = compiled[row_index]; occurrence = lookup[(row_index, "query_anchor")][0]
            occurrence_index = int(occurrence["occurrence_index"])
            for state in RAW_STATES:
                raw_rows.append({"dataset": "C115", "case_id": row["case_id"], "family": family, "partition": partition, "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "code": row["code"], "role": "query_anchor", "subtoken": int(occurrence["subtoken"]), "token_position": int(occurrence["token_position"]), "token_id": int(occurrence["token_id"]), "token_text": occurrence["token_text"], "state": state, "state_kind": "embedding" if state == 0 else "hidden_state", "values": decode_bf16(raw_field[state, occurrence_index]).tolist()})
    payload["effect_rows"] = [row for row in payload["effect_rows"] if row.get("dataset") != "C115"] + effect_rows
    payload["raw_rows"] = [row for row in payload["raw_rows"] if row.get("dataset") != "C115"] + raw_rows
    candidate_vectors = [np.asarray(row["values"], dtype=np.float32) for row in effect_rows if row["role"] == "query_anchor" and row["state"] == 19]
    payload["default_coordinates"] = np.argsort(-np.mean(np.stack([np.abs(vector) for vector in candidate_vectors]), axis=0), kind="stable")[:64].astype(int).tolist()
    payload["scale"] = {"effect_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["effect_rows"]]), 0.99)), "raw_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["raw_rows"]]), 0.99))}
    payload.update({
        "phase": 1629, "campaign": "C109-C115", "title": "C109-C115 Coordinate Assignment / Multi-Position Field Atlas",
        "c115_batch": {"field_prediction": field_report, "behavior": capture_report["behavior"], "predictions": intervention["predictions"], "prediction_checks": intervention["prediction_checks"], "all_frozen_predictions_passed": intervention["all_frozen_predictions_passed"], "max_permutation_l2_relative_error": intervention["max_permutation_l2_relative_error"], "summaries": summaries, "family_rollup": rollup},
        "claim_boundary": "C115 is a larger fifth-lexicon Qwen3 controlled-English activation-coordinate replication. Finite exact-energy permutations support graded physical assignment only; they do not establish a mathematical equivalence relation, permutation group, gauge symmetry, topology, unique dictionary, semantic neurons, weights, an endogenous transport route, attention/MLP mechanism, or universal language law.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c115_coordinate_multi_position_atlas.json"
    core.save(canonical, payload); shutil.copyfile(canonical, PUBLIC)
    all_pass = intervention["all_frozen_predictions_passed"]
    closure = {
        "phase": 1629, "campaign": "C115", "created_at_utc": now(),
        "status": "fifth_lexicon_large_sample_prospective_replication_complete",
        "headline": {"field_results": field_report["results"], "predictions": intervention["predictions"], "prediction_checks": intervention["prediction_checks"], "all_frozen_predictions_passed": all_pass, "family_rollup": rollup, "behavior": capture_report["behavior"], "max_permutation_l2_relative_error": intervention["max_permutation_l2_relative_error"]},
        "new_puzzles": {
            "K301": "fifth-lexicon large-sample prospective field and graded-assignment replication; scope and pass/fail are recorded in the frozen prediction checks",
            "K302": "fifth-lexicon prospective agent query-centered coalition test; query, path, leave-one, and protocol-stage results remain separately typed",
            "K303-BOUNDARY": "finite coordinate permutations identify constrained assignment responses, not an equivalence relation, symmetry group, gauge field, topology, or unique semantic coordinate dictionary",
        },
        "theory_update": "RDC retains a readable relation-role-state field and separates physical support/value assignment, a query-centered role coalition, and protocol-stage output leverage. C115 tests repeatability of those descriptors; it does not promote them to a closed algebra.",
        "unified_formula": "y = O_c(L_{S,V,C_q,P,s}(R[f,r,s](x)))",
        "problems": [
            "controlled synthetic fifth lexicon and one fixed English prompt in Qwen3 only",
            "384 intervention pairs arise from 48 independent lexical units and are not 384 independent language replications",
            "simultaneous state19 patching may be off-manifold and does not reveal natural temporal transport",
            "standard/reversed output-code behavior remains separately typed; raw truth rescue is not task-aligned closure",
            "eight permutations sparsely sample assignment alternatives and cannot define an equivalence class or group",
        ],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "activation_coordinates": 2560, "includes_embedding": True, "hidden_states": list(RAW_STATES[1:])},
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": "C116 observation-first third-relation-family campaign: freeze a large orthogonal language-pattern panel, preserve all 2560 embedding/HiddenState coordinates, and test whether the C115 field/assignment/coalition descriptors generalize beyond the two reused relation families" if all_pass else "C116 observation-first third-relation-family campaign remains authorized as a route-level pivot; retire only failed C115 descriptor routes and preserve all 2560 embedding/HiddenState coordinates",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "source": audit["all_checks_passed"], "rows": len(rows) == 384, "summaries": len(summaries) == 8,
        "effects": len(effect_rows) == 280 and all(len(row["values"]) == DIM for row in effect_rows),
        "raw": len(raw_rows) == 28 and all(len(row["values"]) == DIM for row in raw_rows),
        "embedding_hidden": {row["state_kind"] for row in raw_rows} == {"embedding", "hidden_state"},
        "identity": core.sha(canonical) == core.sha(PUBLIC), "typed_prediction": set(intervention["prediction_checks"]) == set(protocol["operational_gates"]),
        "boundary": "do not establish" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1629, "campaign": "C115", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "asset_sha256": core.sha(canonical), "authorization": "audit_frontend_build_append_c115_memo_then_execute_authorized_c116_observation_campaign"}
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"checks": checks, "headline": closure["headline"], "new_puzzles": closure["new_puzzles"], "heatmap": closure["heatmap"], "next_authorization": closure["next_authorization"]}, indent=2))


STAGES = {"contract": contract, "capture": capture, "field": adjudicate_field, "intervene": intervene, "synthesize": synthesize}


def main(stage: str) -> None:
    STAGES[stage]()

