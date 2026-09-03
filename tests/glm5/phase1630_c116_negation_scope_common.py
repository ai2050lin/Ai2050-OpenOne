#!/usr/bin/env python3
"""C116 observation-first third-family campaign for controlled negation scope."""
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
OUT = RESULT / "phase1630_c116_negation_scope_observation_campaign"
C115 = RESULT / "phase1625_c115_fifth_lexicon_prospective_replication"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base
import phase1608_c110_exact_field_capture as capture_base
import phase1610_c110_frozen_transport_comparison as transport

FAMILY = "negation_scope"
PARTITIONS = ("discovery", "confirmation", "lockbox")
ROLES = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary")
PATH_ROLES = ("focus_record", "focus_post", "query_focus", "query_anchor")
STATES, DIM, WIDTH, BATCH = 37, 2560, 224, 8
KEY_STATES = (0, 4, 8, 12, 16, 19, 24, 28, 32, 36)
RAW_STATES = (0, 8, 16, 19, 24, 32, 36)
STEMS = (
    "bav", "ced", "dor", "fen", "gal", "hir", "jas", "kel", "lum", "mor", "nav", "pel",
    "qor", "rav", "sel", "tor", "ulm", "val", "wen", "xal", "yor", "zel", "bri", "cru",
    "drav", "flen", "griv", "hest", "jor", "krav", "lorn", "mest", "nyr", "prax", "riven", "solm",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if den <= 1e-12 else float(np.dot(left, right) / den)


def topk(values: np.ndarray, k: int) -> list[int]:
    return np.argpartition(np.abs(values), -k)[-k:][np.argsort(-np.abs(values[np.argpartition(np.abs(values), -k)[-k:]]), kind="stable")].astype(int).tolist()


def med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def build() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for index, stem in enumerate(STEMS):
        rotated = [STEMS[(index + offset) % len(STEMS)] for offset in (0, 7, 15, 23, 31)]
        values = (f"Or{rotated[0]}el", f"is{rotated[1]}ic", f"Tu{rotated[2]}an", f"is{rotated[3]}al", f"Ve{rotated[4]}or")
        partition = PARTITIONS[index // 12]
        unit = {"arm": "breadth", "unit_id": f"c116-negation-{index:02d}", "family": FAMILY, "world": "controlled_synthetic_negation_scope", "partition": partition, "surface": "factorial", "values": list(values)}
        units.append(unit)
        for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
            prompt, focus, anchor = breadth_base.breadth_prompt(FAMILY, values, truth, surface, distractor, code)
            output_yes = (truth == 1) == (code == 1)
            cases.append({**unit, "case_id": f"c116-{len(cases):04d}", "truth_factor": truth, "surface_factor": surface, "distractor_factor": distractor, "code": code, "codebook": graph_base.CODEBOOKS[code]["name"], "truth": truth == 1, "output_yes": output_yes, "gold_position": 0 if output_yes else 1, "focus": focus, "anchor": anchor, "prompt": prompt})
    return units, cases


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C116 already exists: {OUT}")
    closure = core.load(C115 / "analysis/closure.json")
    audit = core.load(C115 / "audit/independent_closure_audit.json")
    if not audit["all_checks_passed"] or not closure["next_authorization"].startswith("C116 observation-first third-relation-family campaign"):
        raise RuntimeError("C116 authorization missing")
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
                occurrences.append({"occurrence_index": len(occurrences), "row_index": row_index, "case_id": row["case_id"], "unit_id": row["unit_id"], "family": FAMILY, "partition": row["partition"], "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "code": row["code"], "role": role, "subtoken": subtoken, "span_length": len(positions), "token_position": position, "token_id": token_id, "token_text": tok.convert_ids_to_tokens([token_id])[0]})
        disjoint = disjoint and len(occupied) == len(set(occupied))
    cells = Counter((row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases)
    rng = np.random.default_rng(1630)
    permutations = [rng.permutation(256).astype(int).tolist() for _ in range(8)]
    checks = {
        "authorization": audit["all_checks_passed"], "units": len(units) == 36, "cases": len(cases) == 576,
        "partitions": Counter(row["partition"] for row in units) == {partition: 12 for partition in PARTITIONS},
        "factorial": cells == {(partition, *cell): 12 for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=4)},
        "zero_models": all(abs(value - 0.5) < 1e-12 for key, value in zero.items() if key != "truth_x_code_oracle") and zero["truth_x_code_oracle"] == 1.0,
        "semantic_uniqueness": len({row["prompt"] for row in cases}) == 576,
        "lexical_uniqueness": len({value.casefold() for row in units for value in row["values"]}) == 180,
        "compiled": len(compiled) == 576 and all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled) and disjoint,
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "naturalness": all("The note" in row["prompt"] and "Query:" in row["prompt"] and row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.write_rows(OUT / "material/units.jsonl", units); core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled); core.write_rows(OUT / "protocol/role_occurrence_manifest.jsonl", occurrences)
    protocol = {
        "phase": 1630, "campaign": "C116", "created_at_utc": now(), "status": "negation_scope_observation_first_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized", "object": "third-family negation-scope discovery-confirmation-lockbox field and intervention atlas",
        "family": FAMILY, "partitions": list(PARTITIONS), "units": 36, "cases": 576, "units_per_partition": 12,
        "roles": list(ROLES), "states": STATES, "activation_coordinates": DIM, "occurrences": len(occurrences),
        "archive": {"path": "raw/qwen3_role_subtoken_all_states.uint16.npy", "shape": [STATES, len(occurrences), DIM], "dtype": "uint16 exact BF16 bit patterns", "fixed_width": WIDTH, "batch_size": BATCH},
        "discovery_rule": {"partitions_allowed": ["discovery"], "eligible_states": list(range(1, 31)), "eligible_roles": list(ROLES), "minimum_half_norm": 0.25, "score": "split_half_cosine * min(split_half_norms)", "support_k": 256, "tie_break": "larger split-half cosine, then smaller state, then role order"},
        "frozen_validation_gates": {"confirmation_lockbox_cosine_min": 0.85, "each_to_discovery_cosine_min": 0.80, "each_support_topk_overlap_min": 0.45, "correct_movement_gt_permutation_median_cells": 4},
        "role_descriptor_gates": {"path_gt_query_cells": 4, "selected_role_positive_cells": 4},
        "movement_permutations": permutations,
        "intervention_modes": ["frozen_support"] + [f"movement_permutation_{i}" for i in range(8)] + ["selected_role", "query_anchor", "record_to_query_path", "all_registered_roles"],
        "observation_first": "capture all registered embedding/HiddenState coordinates; discovery alone nominates role/state/support; confirmation and lockbox remain unread until nomination is frozen",
        "behavior_policy": "standard and reversed code are reported separately; upstream truth-field observation cannot upgrade failed output-code behavior",
        "completion_rule": "all field and intervention routes run after frozen nomination; failure retires a descriptor, not the entire campaign",
        "numeric": {"movement_l2_relative_tolerance": 0.02, "batch_size": BATCH, "fixed_width": WIDTH},
        "typed_missingness": {"human_naturalness": "machine-only controlled English", "cross_model": "Qwen3 only", "relation_breadth": "one new logical relation family"},
        "claim_boundary": "controlled synthetic negation-scope activation study; no natural-language universality, weights, semantic neurons, attention/MLP, endogenous route, low-dimensional manifold, algebraic closure, symmetry group, or new-mathematics claim",
        "source_paths": {"c115_closure": str(C115 / "analysis/closure.json"), "c115_audit": str(C115 / "audit/independent_closure_audit.json")},
        "source_hashes": {"c115_closure": core.sha(C115 / "analysis/closure.json"), "c115_audit": core.sha(C115 / "audit/independent_closure_audit.json")},
        "material_digest": core.digest([*units, *cases]), "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1631_c116_exact_field_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1630, "campaign": "C116", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "occurrences": len(occurrences), "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_pre_model_audit.json", report); print(json.dumps(report, indent=2))


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_pre_model_audit.json")["all_checks_passed"]:
        raise RuntimeError("C116 capture audit missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl"); manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    by_row: dict[int, list[dict]] = defaultdict(list)
    for occurrence in manifest:
        by_row[int(occurrence["row_index"])].append(occurrence)
    raw_path = OUT / protocol["archive"]["path"]; logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"; index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (raw_path, logits_path, index_path)):
        raise RuntimeError("C116 raw exists")
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=tuple(protocol["archive"]["shape"]))
    candidate_logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    behavior, model, first_rows = [], None, None
    repeat_hidden = repeat_logits = 0.0
    try:
        model, tok, device, placement = load_bf16("qwen3"); quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            output, logits, ids, mask, positions, lengths = capture_base.forward(model, batch, pad, device, WIDTH)
            for state_index, state in enumerate(output.hidden_states):
                if state.dtype != torch.bfloat16 or not bool(torch.isfinite(state).all()):
                    raise RuntimeError((state_index, state.dtype))
                for local in range(len(batch)):
                    row_index = start + local; occurrences = by_row[row_index]
                    indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                    token_positions = [int(item["token_position"]) for item in occurrences]
                    field[state_index, indices] = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
            for local, row in enumerate(batch):
                row_index = start + local; scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                candidate_logits[row_index] = scores; prediction = int(scores[1] > scores[0])
                behavior.append({"row_index": row_index, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "code": row["code"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "yes_minus_no": scores[0] - scores[1]})
            if start == 0:
                first_rows = batch
            if (start // BATCH + 1) % 12 == 0:
                field.flush(); candidate_logits.flush(); print(f"[phase1631] captured {start + len(batch)}/{len(rows)}", flush=True)
            del output, logits, ids, mask, positions
        field.flush(); candidate_logits.flush()
        output, logits, ids, mask, positions, lengths = capture_base.forward(model, first_rows, pad, device, WIDTH)
        for state_index, state in enumerate(output.hidden_states):
            for local in range(len(first_rows)):
                occurrences = by_row[local]; indices = np.asarray([int(item["occurrence_index"]) for item in occurrences]); token_positions = [int(item["token_position"]) for item in occurrences]
                old = np.asarray(field[state_index, indices]); new = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
                if not np.array_equal(old, new): repeat_hidden = max(repeat_hidden, float(np.max(np.abs(decode(old) - decode(new)))))
        for local, row in enumerate(first_rows):
            scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
            repeat_logits = max(repeat_logits, float(np.max(np.abs(scores - candidate_logits[local]))))
        del output, logits, ids, mask, positions
    finally:
        field.flush(); candidate_logits.flush()
        if model is not None: release_bf16(model)
        gc.collect()
    core.write_rows(index_path, behavior)
    def acc(selected: list[dict]) -> float: return float(np.mean([row["correct"] for row in selected]))
    behavior_summary = {"global_accuracy": acc(behavior), "by_partition": {p: acc([row for row in behavior if row["partition"] == p]) for p in PARTITIONS}, "by_code": {str(c): acc([row for row in behavior if row["code"] == c]) for c in (1, -1)}}
    checks = {"shape": list(field.shape) == protocol["archive"]["shape"], "logits": list(candidate_logits.shape) == [576, 2] and bool(np.isfinite(candidate_logits).all()), "index": len(behavior) == 576, "repeat_hidden": repeat_hidden == 0.0, "repeat_logits": repeat_logits == 0.0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()): raise RuntimeError(checks)
    report = {"phase": 1631, "campaign": "C116", "created_at_utc": now(), "status": "negation_scope_exact_field_capture_complete", "shape": list(field.shape), "raw_file_bytes": raw_path.stat().st_size, "raw_data_bytes": int(field.nbytes), "raw_sha256": core.sha(raw_path), "logits_sha256": core.sha(logits_path), "index_sha256": core.sha(index_path), "behavior": behavior_summary, "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logits_max_abs": repeat_logits}, "runtime": {"placement": placement, "quantization": quant}, "checks": checks, "authorization": "run_phase1632_c116_discovery_freeze"}
    core.save(OUT / "analysis/capture_summary.json", report); print(json.dumps({k: v for k, v in report.items() if k != "runtime"}, indent=2))


def truth_fields(partitions: set[str], output_path: Path) -> tuple[np.memmap, list[dict]]:
    protocol = core.load(OUT / "protocol/preregistration.json"); rows = core.rows(OUT / "compiled/qwen3.jsonl"); units = [row for row in core.rows(OUT / "material/units.jsonl") if row["partition"] in partitions]
    field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r"); manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest: lookup[(int(occurrence["row_index"]), occurrence["role"])].append(int(occurrence["occurrence_index"]))
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}; role_index = {role: index for index, role in enumerate(ROLES)}
    result = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(units), len(ROLES), STATES, DIM)); result[:] = 0
    for state in range(STATES):
        for row_index, row in enumerate(rows):
            if row["partition"] not in partitions: continue
            u = unit_index[row["unit_id"]]; coeff = float(row["truth_factor"]) / 16.0
            for role in ROLES:
                result[u, role_index[role], state] += coeff * np.mean(decode(field[state, lookup[(row_index, role)]]), axis=0, dtype=np.float32)
        if state % 8 == 0 or state == 36: result.flush()
    return result, units


def discover() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"]: raise RuntimeError("capture audit missing")
    current_producer = core.sha(Path(__file__))
    if current_producer != protocol["producer_sha256"]:
        core.save(OUT / "protocol/phase1632_execution_amendment.json", {
            "phase": 1632, "campaign": "C116", "created_at_utc": now(),
            "reason": "strict JSON rejected the preregistered ineligible-candidate sentinel -inf before nomination was written",
            "repair": "encode ineligible score as null and reuse the already generated exact discovery field",
            "unchanged": ["materials", "partitions", "eligible roles", "eligible states", "minimum norm", "score formula", "tie break", "support k", "validation gates"],
            "original_producer_sha256": protocol["producer_sha256"], "repaired_producer_sha256": current_producer,
        })
    path = OUT / "analysis/discovery_unit_truth_role_state.float32.npy"
    if path.exists():
        fields = np.load(path, mmap_mode="r")
        units = [row for row in core.rows(OUT / "material/units.jsonl") if row["partition"] == "discovery"]
    else:
        fields, units = truth_fields({"discovery"}, path)
    rule = protocol["discovery_rule"]; candidates = []
    for role_index, role in enumerate(ROLES):
        for state in rule["eligible_states"]:
            left = np.mean(fields[:6, role_index, state], axis=0, dtype=np.float32); right = np.mean(fields[6:, role_index, state], axis=0, dtype=np.float32)
            left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right)); half_cos = cosine(left, right)
            score = half_cos * min(left_norm, right_norm) if min(left_norm, right_norm) >= rule["minimum_half_norm"] else None
            candidates.append({"role": role, "state": state, "split_half_cosine": half_cos, "left_norm": left_norm, "right_norm": right_norm, "score": score})
    eligible = [row for row in candidates if row["score"] is not None]
    if not eligible: raise RuntimeError("no discovery candidate")
    winner = sorted(eligible, key=lambda row: (-row["score"], -row["split_half_cosine"], row["state"], ROLES.index(row["role"])))[0]
    role_index = ROLES.index(winner["role"]); vector = np.mean(fields[:, role_index, winner["state"]], axis=0, dtype=np.float32)
    support = topk(vector, int(rule["support_k"])); nomination = {**winner, "support": support, "support_k": len(support), "discovery_units": [row["unit_id"] for row in units], "field_vector_sha256": core.sha(path), "candidate_table_digest": core.digest(candidates), "frozen_at_utc": now()}
    core.write_rows(OUT / "analysis/discovery_candidate_table.jsonl", candidates); core.save(OUT / "protocol/frozen_discovery_nomination.json", nomination)
    checks = {"units": len(units) == 12, "shape": list(fields.shape) == [12, 7, 37, 2560], "finite": bool(np.isfinite(fields).all()), "candidate_count": len(candidates) == 210, "winner_eligible": winner["score"] is not None and math.isfinite(winner["score"]), "support": len(support) == 256 and len(set(support)) == 256, "no_validation_partition": set(nomination["discovery_units"]) == {row["unit_id"] for row in units}}
    if not all(checks.values()): raise RuntimeError(checks)
    report = {"phase": 1632, "campaign": "C116", "created_at_utc": now(), "status": "discovery_role_state_support_frozen", "winner": {k: v for k, v in winner.items()}, "support_k": 256, "checks": checks, "discovery_sha256": core.sha(path), "nomination_sha256": core.sha(OUT / "protocol/frozen_discovery_nomination.json"), "authorization": "execute_phase1633_c116_confirmation_lockbox_validation"}
    core.save(OUT / "analysis/discovery_freeze.json", report); print(json.dumps(report, indent=2))


@torch.inference_mode()
def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    if not core.load(OUT / "audit/independent_discovery_audit.json")["all_checks_passed"]: raise RuntimeError("discovery audit missing")
    path = OUT / "analysis/validation_unit_truth_role_state.float32.npy"
    core.save(OUT / "protocol/phase1633_execution_amendment.json", {
        "phase": 1633, "campaign": "C116", "created_at_utc": now(),
        "reason": "post-model summary used an undefined loop variable before any intervention result or gate was written",
        "repair": "enumerate modes from the current selected cell and reuse the exact validation field",
        "unchanged": ["materials", "nomination", "support", "partitions", "intervention modes", "permutations", "all gates"],
        "nomination_sha256": core.sha(OUT / "protocol/frozen_discovery_nomination.json"),
    })
    if path.exists():
        fields = np.load(path, mmap_mode="r")
        units = [row for row in core.rows(OUT / "material/units.jsonl") if row["partition"] in {"confirmation", "lockbox"}]
    else:
        fields, units = truth_fields({"confirmation", "lockbox"}, path)
    role_i, state = ROLES.index(nomination["role"]), int(nomination["state"]); support = nomination["support"]
    discovery = np.load(OUT / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r")
    d = np.mean(discovery[:, role_i, state], axis=0, dtype=np.float32); c = np.mean(fields[:12, role_i, state], axis=0, dtype=np.float32); l = np.mean(fields[12:, role_i, state], axis=0, dtype=np.float32)
    gates = protocol["frozen_validation_gates"]
    field_metrics = {"role": nomination["role"], "state": state, "confirmation_lockbox_cosine": cosine(c, l), "confirmation_to_discovery_cosine": cosine(c, d), "lockbox_to_discovery_cosine": cosine(l, d), "confirmation_support_overlap": len(set(topk(c, 256)) & set(support)) / 256, "lockbox_support_overlap": len(set(topk(l, 256)) & set(support)) / 256}
    field_checks = {"confirmation_lockbox": field_metrics["confirmation_lockbox_cosine"] >= gates["confirmation_lockbox_cosine_min"], "to_discovery": min(field_metrics["confirmation_to_discovery_cosine"], field_metrics["lockbox_to_discovery_cosine"]) >= gates["each_to_discovery_cosine_min"], "support_overlap": min(field_metrics["confirmation_support_overlap"], field_metrics["lockbox_support_overlap"]) >= gates["each_support_topk_overlap_min"]}
    rows = core.rows(OUT / "compiled/qwen3.jsonl"); pairs = [pair for pair in transport.build_pairs(rows, {**protocol, "supports": {"negation_scope_k256": support}}) if pair["partition"] in {"confirmation", "lockbox"}]
    for index, pair in enumerate(pairs): pair["pair_id"] = f"c116-pair-{index:04d}"
    all_roles = tuple(ROLES); grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        lengths = tuple(len(pair["recipient"]["role_positions"][role]) for role in all_roles); grouped[(pair["partition"], pair["code"], lengths)].append(pair)
    results, model, first_repeat = [], None, None
    try:
        model, tok, device, placement = load_bf16("qwen3"); quant = quantization_audit(model); pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        support_t = torch.tensor(support, dtype=torch.long, device=device); perms = [torch.tensor(values, dtype=torch.long, device=device) for values in protocol["movement_permutations"]]
        for (partition, code, lengths), group in sorted(grouped.items()):
            for start in range(0, len(group), BATCH):
                batch = group[start:start + BATCH]; recipients = [pair["recipient"] for pair in batch]; donors = [pair["donor"] for pair in batch]
                rec_logits, rec_states = transport.forward_with_roles(model, recipients, all_roles, state, pad, device, WIDTH); don_logits, don_states = transport.forward_with_roles(model, donors, all_roles, state, pad, device, WIDTH)
                if first_repeat is None: first_repeat = (recipients, rec_logits.detach().clone(), {role: value.detach().clone() for role, value in rec_states.items()})
                rr, dr = rec_states[nomination["role"]], don_states[nomination["role"]]; delta = dr[..., support_t] - rr[..., support_t]; norm = torch.sqrt(torch.sum(delta.float() ** 2, dim=(1, 2))).clamp_min(1e-12)
                patches = {}; value = rr.clone(); value[..., support_t] = dr[..., support_t]; patches["frozen_support"] = {nomination["role"]: value}; permutation_norms = []
                for i, permutation in enumerate(perms):
                    value = rr.clone(); value[..., support_t] = (rr[..., support_t].float() + delta.float()[..., permutation]).to(rr.dtype); patches[f"movement_permutation_{i}"] = {nomination["role"]: value}; permutation_norms.append(torch.sqrt(torch.sum((value[..., support_t] - rr[..., support_t]).float() ** 2, dim=(1, 2))).clamp_min(1e-12))
                patches["selected_role"] = {nomination["role"]: don_states[nomination["role"]].clone()}; patches["query_anchor"] = {"query_anchor": don_states["query_anchor"].clone()}; patches["record_to_query_path"] = {role: don_states[role].clone() for role in PATH_ROLES}; patches["all_registered_roles"] = {role: don_states[role].clone() for role in ROLES}
                patched = {mode: transport.forward_patched_roles(model, recipients, values, state, pad, device, WIDTH) for mode, values in patches.items()}
                for local, pair in enumerate(batch):
                    base = transport.margin(rec_logits[local], recipients[local]); mode_results = {}
                    for mode, logits in patched.items():
                        margin = transport.margin(logits[local], recipients[local]); gain = margin - base; mode_results[mode] = {"truth_direction_gain": gain, "truth_flip": base <= 0 < margin}
                    results.append({"pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "partition": partition, "code": code, "surface_factor": pair["surface_factor"], "distractor_factor": pair["distractor_factor"], "recipient_yes_minus_no": base, "target_movement_l2": float(norm[local]), "permutation_l2_relative_errors": [float(torch.abs(value[local] - norm[local]) / norm[local]) for value in permutation_norms], "modes": mode_results})
                print(f"[phase1633] {partition}/code={code}/lengths={lengths} {start + len(batch)}/{len(group)}", flush=True)
        rr, old_logits, old_states = first_repeat; new_logits, new_states = transport.forward_with_roles(model, rr, all_roles, state, pad, device, WIDTH); repeat_logits = float(torch.max(torch.abs(new_logits - old_logits))); repeat_hidden = max(float(torch.max(torch.abs(new_states[role] - old_states[role]))) for role in ROLES)
    finally:
        if model is not None: release_bf16(model)
        gc.collect()
    result_path = OUT / "analysis/validation_intervention_results.jsonl"; core.write_rows(result_path, results)
    summaries = []
    for partition in ("confirmation", "lockbox"):
        for code in (1, -1):
            selected = [row for row in results if row["partition"] == partition and row["code"] == code]
            correct = med([row["modes"]["frozen_support"]["truth_direction_gain"] for row in selected]); permutation_medians = [med([row["modes"][f"movement_permutation_{i}"]["truth_direction_gain"] for row in selected]) for i in range(8)]; mode_medians = {mode: med([row["modes"][mode]["truth_direction_gain"] for row in selected]) for mode in ("selected_role", "query_anchor", "record_to_query_path", "all_registered_roles")}
            summaries.append({"partition": partition, "code": code, "pairs": len(selected), "independent_units": len({row["unit_id"] for row in selected}), "frozen_support_median_gain": correct, "permutation_median_gains": permutation_medians, "frozen_support_gt_permutation_median": correct > med(permutation_medians), "frozen_support_gt_all_permutations": all(correct > value for value in permutation_medians), "mode_median_gains": mode_medians, "path_gt_query": mode_medians["record_to_query_path"] > mode_medians["query_anchor"], "selected_role_positive": mode_medians["selected_role"] > 0, "truth_flip_counts": {mode: sum(item["modes"][mode]["truth_flip"] for item in selected) for mode in selected[0]["modes"]}})
    summary_path = OUT / "analysis/validation_summary.jsonl"; core.write_rows(summary_path, summaries)
    predictions = {"field_passed": all(field_checks.values()), "correct_movement_gt_permutation_median_cells": sum(row["frozen_support_gt_permutation_median"] for row in summaries), "strict_win_cells_descriptive": sum(row["frozen_support_gt_all_permutations"] for row in summaries), "path_gt_query_cells": sum(row["path_gt_query"] for row in summaries), "selected_role_positive_cells": sum(row["selected_role_positive"] for row in summaries)}
    prediction_checks = {"field": predictions["field_passed"], "coordinate_assignment": predictions["correct_movement_gt_permutation_median_cells"] == gates["correct_movement_gt_permutation_median_cells"], "path_descriptor": predictions["path_gt_query_cells"] == protocol["role_descriptor_gates"]["path_gt_query_cells"], "selected_role": predictions["selected_role_positive_cells"] == protocol["role_descriptor_gates"]["selected_role_positive_cells"]}
    max_error = max(error for row in results for error in row["permutation_l2_relative_errors"])
    checks = {"units": len(units) == 24, "shape": list(fields.shape) == [24, 7, 37, 2560], "pairs": len(results) == 192, "summaries": len(summaries) == 4 and all(row["pairs"] == 48 and row["independent_units"] == 12 for row in summaries), "l2": max_error <= protocol["numeric"]["movement_l2_relative_tolerance"], "finite": all(math.isfinite(row["modes"][mode]["truth_direction_gain"]) for row in results for mode in row["modes"]), "repeat": repeat_hidden == 0 and repeat_logits == 0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()): raise RuntimeError(checks)
    report = {"phase": 1633, "campaign": "C116", "created_at_utc": now(), "status": "negation_scope_confirmation_lockbox_validation_complete", "nomination": {"role": nomination["role"], "state": state, "support_k": 256}, "field_metrics": field_metrics, "field_checks": field_checks, "predictions": predictions, "prediction_checks": prediction_checks, "all_descriptor_gates_passed": all(prediction_checks.values()), "max_l2_relative_error": max_error, "checks": checks, "runtime": {"placement": placement, "quantization": quant}, "field_sha256": core.sha(path), "results_sha256": core.sha(result_path), "summary_sha256": core.sha(summary_path), "authorization": "run_phase1634_c116_synthesis_heatmap_and_closure"}
    core.save(OUT / "analysis/validation_adjudication.json", report); print(json.dumps({k: v for k, v in report.items() if k != "runtime"}, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); discovery = core.load(OUT / "analysis/discovery_freeze.json"); validation = core.load(OUT / "analysis/validation_adjudication.json")
    if not core.load(OUT / "audit/independent_validation_audit.json")["all_checks_passed"]: raise RuntimeError("validation audit missing")
    nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    discovery_fields = np.load(OUT / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r"); validation_fields = np.load(OUT / "analysis/validation_unit_truth_role_state.float32.npy", mmap_mode="r")
    means = [np.mean(discovery_fields, axis=0, dtype=np.float32), np.mean(validation_fields[:12], axis=0, dtype=np.float32), np.mean(validation_fields[12:], axis=0, dtype=np.float32)]
    payload = core.load(PUBLIC); effect_rows = []; display_states = tuple(sorted(set(KEY_STATES) | {int(nomination["state"])}))
    for partition, mean in zip(PARTITIONS, means, strict=True):
        for role_i, role in enumerate(ROLES):
            for state in display_states: effect_rows.append({"dataset": "C116", "family": FAMILY, "partition": partition, "role": role, "state": state, "state_kind": "embedding" if state == 0 else "hidden_state", "effect": "balanced_truth_walsh", "values": np.asarray(mean[role_i, state], dtype=np.float32).tolist()})
    compiled = core.rows(OUT / "compiled/qwen3.jsonl"); manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl"); raw_field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r"); lookup: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for occurrence in manifest: lookup[(int(occurrence["row_index"]), occurrence["role"])].append(occurrence)
    raw_rows = []
    for partition in PARTITIONS:
        row_index = next(index for index, row in enumerate(compiled) if row["partition"] == partition); row = compiled[row_index]; occurrence = lookup[(row_index, "query_anchor")][0]; oi = int(occurrence["occurrence_index"])
        for state in RAW_STATES: raw_rows.append({"dataset": "C116", "case_id": row["case_id"], "family": FAMILY, "partition": partition, "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "code": row["code"], "role": "query_anchor", "subtoken": int(occurrence["subtoken"]), "token_position": int(occurrence["token_position"]), "token_id": int(occurrence["token_id"]), "token_text": occurrence["token_text"], "state": state, "state_kind": "embedding" if state == 0 else "hidden_state", "values": decode(raw_field[state, oi]).tolist()})
        boundary = lookup[(row_index, nomination["role"])][0]; boundary_index = int(boundary["occurrence_index"]); state = int(nomination["state"])
        raw_rows.append({"dataset": "C116", "case_id": row["case_id"], "family": FAMILY, "partition": partition, "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "code": row["code"], "role": nomination["role"], "subtoken": int(boundary["subtoken"]), "token_position": int(boundary["token_position"]), "token_id": int(boundary["token_id"]), "token_text": boundary["token_text"], "state": state, "state_kind": "hidden_state", "values": decode(raw_field[state, boundary_index]).tolist()})
    payload["effect_rows"] = [row for row in payload["effect_rows"] if row.get("dataset") != "C116"] + effect_rows; payload["raw_rows"] = [row for row in payload["raw_rows"] if row.get("dataset") != "C116"] + raw_rows
    payload["default_coordinates"] = nomination["support"][:64]
    payload["scale"] = {"effect_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["effect_rows"]]), .99)), "raw_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["raw_rows"]]), .99))}
    payload.update({"phase": 1634, "campaign": "C109-C116", "title": "C109-C116 Relation-Role-State Activation Atlas", "c116_batch": {"discovery": discovery, "validation": validation, "summaries": core.rows(OUT / "analysis/validation_summary.jsonl"), "nomination": {"role": nomination["role"], "state": nomination["state"], "support_k": nomination["support_k"]}}, "claim_boundary": "C116 adds one controlled negation-scope family with discovery-frozen role/state/support and independent confirmation/lockbox validation. It is an activation-coordinate observation/intervention atlas, not weights, semantic neurons, attention/MLP, a universal relation algebra, low-dimensional manifold, symmetry group, or new mathematics.", "created_at_utc": now()})
    canonical = OUT / "visualization/c109_c116_relation_role_state_atlas.json"; core.save(canonical, payload); shutil.copyfile(canonical, PUBLIC)
    closure = {"phase": 1634, "campaign": "C116", "created_at_utc": now(), "status": "negation_scope_third_family_observation_validation_complete", "headline": {"discovery_winner": discovery["winner"], "behavior": core.load(OUT / "analysis/capture_summary.json")["behavior"], "field_metrics": validation["field_metrics"], "field_checks": validation["field_checks"], "predictions": validation["predictions"], "prediction_checks": validation["prediction_checks"], "all_descriptor_gates_passed": validation["all_descriptor_gates_passed"]}, "new_puzzles": {"K304": "discovery-frozen negation-scope relation-role-state candidate and its confirmation/lockbox field outcome", "K305": "third-family coordinate assignment outcome under eight exact-energy permutations", "K306-BOUNDARY": "query/path descriptors are relation-dependent candidates; passing or failing them does not imply one universal role route"}, "theory_update": "RDC now has a legally separated third-family test. The functional object remains a context-indexed relation-role-state response field with separately nominated physical support/value assignment and role coalition; no common algebra is assumed.", "unified_formula": "y = O_c(L_{S_f,V_f,C_f,P,s}(R[f,r,s](x)))", "problems": ["one synthetic negation-scope family and one Qwen3 prompt grammar", "discovery selects a high-leverage role/state/support and therefore requires further fresh-family replication", "simultaneous activation patching can be off-manifold", "standard/reversed code behavior must remain separately typed", "256 coordinates and eight permutations do not establish minimality or equivalence classes"], "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "activation_coordinates": 2560, "includes_embedding": True}, "claim_boundary": payload["claim_boundary"], "next_authorization": "C117 observation-first fourth-family whole-part exception campaign using the frozen C116 discovery-validation protocol; preserve all 2560 embedding/HiddenState coordinates and treat C116 failures as descriptor-level evidence"}
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"validation": core.load(OUT / "audit/independent_validation_audit.json")["all_checks_passed"], "effects": len(effect_rows) == 231 and all(len(row["values"]) == 2560 for row in effect_rows), "raw": len(raw_rows) == 24 and all(len(row["values"]) == 2560 for row in raw_rows), "candidate_visible": any(row["role"] == nomination["role"] and row["state"] == nomination["state"] for row in effect_rows) and sum(row["role"] == nomination["role"] and row["state"] == nomination["state"] for row in raw_rows) == 3, "asset": core.sha(canonical) == core.sha(PUBLIC), "batch": "c115_batch" in payload and "c116_batch" in payload, "boundary": "not weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"]}
    if not all(checks.values()): raise RuntimeError(checks)
    report = {"phase": 1634, "campaign": "C116", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "authorization": "audit_frontend_append_memo_then_consider_c117"}; core.save(OUT / "audit/internal_closure_audit.json", report); print(json.dumps({"checks": checks, "headline": closure["headline"], "next_authorization": closure["next_authorization"]}, indent=2))


STAGES = {"contract": contract, "capture": capture, "discover": discover, "validate": validate, "synthesize": synthesize}


def main(stage: str) -> None:
    STAGES[stage]()
