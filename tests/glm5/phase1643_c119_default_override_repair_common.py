#!/usr/bin/env python3
"""C119: identifiable default inheritance and exception override campaign."""
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
OUT = RESULT / "phase1643_c119_identifiable_default_override_campaign"
C115 = RESULT / "phase1625_c115_fifth_lexicon_prospective_replication"
C116 = RESULT / "phase1630_c116_negation_scope_observation_campaign"
C117 = RESULT / "phase1635_c117_whole_part_exception_observation_campaign"
C118 = RESULT / "phase1640_c118_identifiable_default_override_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base
import phase1608_c110_exact_field_capture as capture_base
import phase1610_c110_frozen_transport_comparison as transport

CAMPAIGN = "C119"
FAMILY = "identifiable_default_override"
PARTITIONS = ("discovery", "confirmation", "lockbox")
ROLES = (
    "focus_pre", "default_kind", "default_value", "exception_target",
    "exception_value", "focus_post", "query_focus", "query_anchor", "boundary",
)
EFFECTS = ("truth", "default_inheritance", "exception_read", "override", "same_hit_control")
STATES, DIM, WIDTH, BATCH = 37, 2560, 256, 8
KEY_STATES = (0, 4, 8, 12, 16, 19, 24, 28, 30, 32, 36)
RAW_STATES = (0, 8, 16, 19, 24, 30, 32, 36)
STEMS = (
    "adex", "brol", "cavi", "drem", "elun", "fesk", "gavi", "hord",
    "isel", "jorv", "kalm", "lure", "mord", "navi", "oxel", "pran",
    "quor", "rusk", "sela", "trov", "ulen", "vesk", "warl", "xyne",
)
SYSTEM = (
    "Use only the local policy and inspection record. A general policy applies to an item "
    "unless an inspection exception for that exact item says otherwise. An exception for "
    "another item does not change the queried item. Reply using exactly the requested vocabulary."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return np.zeros_like(values, dtype=np.float32) if norm <= 1e-12 else np.asarray(values, dtype=np.float32) / norm


def topk(values: np.ndarray, k: int) -> list[int]:
    selected = np.argpartition(np.abs(values), -k)[-k:]
    return selected[np.argsort(-np.abs(values[selected]), kind="stable")].astype(int).tolist()


def med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def inventory() -> list[tuple[str, str, str, str, str]]:
    return [
        (f"Rig{stem}A", f"Rig{stem}B", f"{stem}module", f"{stem}seal", f"{stem}tag")
        for stem in STEMS
    ]


def prompt_for(
    values: tuple[str, str, str, str, str],
    default: int,
    hit: int,
    conflict: int,
    surface: int,
    output_format: int,
) -> tuple[str, dict]:
    focus, other, kind, part, tag = values
    default_word = "retains" if default == 1 else "lacks"
    exception_truth = default * conflict
    exception_word = "retains" if exception_truth == 1 else "lacks"
    exception_subject = focus if hit == 1 else other
    policy = f"General policy: every {kind} normally {default_word} its {part}"
    exception = f"Inspection exception: {exception_subject} {exception_word} its {part}"
    clauses = [policy, exception] if surface == 1 else [exception, policy]
    applicability = (
        "Applicability check: the inspection exception applies to the queried item"
        if hit == 1
        else "Applicability check: the inspection exception does not apply to the queried item"
    )
    clauses.append(applicability)
    clauses.append(f"Inventory note: the spare {tag} is catalogued separately")
    vocabulary = "yes or no" if output_format == 1 else "true or false"
    prompt = (
        f"Focus before record: {focus}. Local record: " + ". ".join(clauses) + ". "
        f"Focus after record: {focus}. Query: Under the policy and inspection exception, "
        f"does {focus} retain its {part}? Reply exactly {vocabulary}."
    )
    truth = default if hit == -1 else exception_truth
    return prompt, {
        "focus": focus,
        "other": other,
        "kind": kind,
        "part": part,
        "tag": tag,
        "default_word": default_word,
        "exception_word": exception_word,
        "exception_subject": exception_subject,
        "truth_factor": truth,
        "output_positive": truth == 1,
        "output_labels": ["yes", "no"] if output_format == 1 else ["true", "false"],
    }


def build() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for index, values in enumerate(inventory()):
        partition = PARTITIONS[index // 8]
        unit = {
            "unit_id": f"c119-override-{index:02d}",
            "family": FAMILY,
            "world": "controlled_synthetic_policy_exception",
            "partition": partition,
            "values": list(values),
        }
        units.append(unit)
        for default, hit, conflict, surface, output_format in itertools.product((1, -1), repeat=5):
            prompt, metadata = prompt_for(values, default, hit, conflict, surface, output_format)
            cases.append({
                **unit,
                **metadata,
                "case_id": f"c119-{len(cases):04d}",
                "default_factor": default,
                "hit_factor": hit,
                "conflict_factor": conflict,
                "surface_factor": surface,
                "output_format": output_format,
                "truth": metadata["truth_factor"] == 1,
                "gold_position": 0 if metadata["output_positive"] else 1,
                "prompt": prompt,
            })
    return units, cases


def find_between(spans: list[list[int]], lower: int, upper: int) -> list[int]:
    candidates = [span for span in spans if min(span) > lower and max(span) < upper]
    if len(candidates) != 1:
        raise RuntimeError(("between", spans, lower, upper, candidates))
    return candidates[0]


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    cache: dict[tuple[str, str], list[int]] = {}
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        labels = tuple(row["output_labels"])
        if labels not in cache:
            cache[labels] = [
                int(value)
                for label in labels
                for value in tok.encode(" " + label, add_special_tokens=False)
            ]
            if len(cache[labels]) != 2:
                raise RuntimeError(("candidate singleton", labels, cache[labels]))
        candidate_ids = [[cache[labels][0]], [cache[labels][1]]]
        focus_spans = breadth_base.find_spans(tok, ids, row["focus"])
        other_spans = breadth_base.find_spans(tok, ids, row["other"])
        kind_spans = breadth_base.find_spans(tok, ids, row["kind"])
        part_spans = breadth_base.find_spans(tok, ids, row["part"])
        if len(focus_spans) < 3 or len(kind_spans) != 1 or len(part_spans) < 3:
            raise RuntimeError((row["case_id"], focus_spans, kind_spans, part_spans))
        focus_pre, focus_post, query_focus = focus_spans[0], focus_spans[-2], focus_spans[-1]
        if row["hit_factor"] == 1:
            exception_target = find_between(focus_spans, max(focus_pre), min(focus_post))
        else:
            exception_target = find_between(other_spans, max(focus_pre), min(focus_post))
        default_candidates = [
            span for span in breadth_base.find_spans(tok, ids, row["default_word"])
            if min(span) > max(kind_spans[0]) and max(span) < min(focus_post)
        ]
        if not default_candidates:
            raise RuntimeError(("default value", row["case_id"], kind_spans, default_candidates))
        default_value = min(default_candidates, key=lambda span: min(span) - max(kind_spans[0]))
        exception_value_candidates = breadth_base.find_spans(tok, ids, row["exception_word"])
        exception_value = [
            span for span in exception_value_candidates
            if min(span) > max(exception_target) and max(span) < min(focus_post)
        ]
        if not exception_value:
            raise RuntimeError(("exception value", row["case_id"], exception_value_candidates, exception_target))
        exception_value = [min(exception_value, key=lambda span: min(span) - max(exception_target))]
        roles = {
            "focus_pre": focus_pre,
            "default_kind": kind_spans[0],
            "default_value": default_value,
            "exception_target": exception_target,
            "exception_value": exception_value[0],
            "focus_post": focus_post,
            "query_focus": query_focus,
            "query_anchor": part_spans[-1],
            "boundary": [len(ids) - 1],
        }
        occupied = [position for span in roles.values() for position in span]
        if len(occupied) != len(set(occupied)):
            raise RuntimeError(("overlapping roles", row["case_id"], roles))
        if not all(max(roles[name]) < min(roles["focus_post"]) for name in ("default_kind", "default_value", "exception_target", "exception_value")):
            raise RuntimeError(("record role order", row["case_id"], roles))
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids, "role_positions": roles})
    return compiled


def zero_models(rows: list[dict]) -> dict[str, float]:
    gold = np.asarray([row["truth_factor"] == 1 for row in rows])
    predictions = {
        "always_positive": np.ones(len(rows), dtype=bool),
        "always_negative": np.zeros(len(rows), dtype=bool),
        "default_only": np.asarray([row["default_factor"] == 1 for row in rows]),
        "exception_only": np.asarray([row["default_factor"] * row["conflict_factor"] == 1 for row in rows]),
        "hit_only": np.asarray([row["hit_factor"] == 1 for row in rows]),
        "surface_only": np.asarray([row["surface_factor"] == 1 for row in rows]),
        "format_only": np.asarray([row["output_format"] == 1 for row in rows]),
    }
    result = {name: float(np.mean(value == gold)) for name, value in predictions.items()}
    result["semantic_oracle"] = 1.0
    return result


def historical_values() -> set[str]:
    paths = (C115 / "material/units.jsonl", C116 / "material/units.jsonl", C117 / "material/units.jsonl")
    return {
        str(value).casefold()
        for path in paths
        for row in core.rows(path)
        for value in row.get("values", [])
    }


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C119 already exists: {OUT}")
    parent = core.load(C118 / "analysis/closure.json")
    parent_audit = core.load(C118 / "audit/independent_closure_audit.json")
    if not parent_audit["all_checks_passed"] or not parent["next_authorization"].startswith("C119"):
        raise RuntimeError("C119 authorization missing")
    units, cases = build()
    tok = graph_base.tokenizer()
    compiled = compile_rows(tok, cases)
    occurrences = []
    for row_index, row in enumerate(compiled):
        for role in ROLES:
            positions = row["role_positions"][role]
            for subtoken, position in enumerate(positions):
                occurrences.append({
                    "occurrence_index": len(occurrences), "row_index": row_index,
                    "case_id": row["case_id"], "unit_id": row["unit_id"],
                    "partition": row["partition"], "truth_factor": row["truth_factor"],
                    "default_factor": row["default_factor"], "hit_factor": row["hit_factor"],
                    "conflict_factor": row["conflict_factor"], "surface_factor": row["surface_factor"],
                    "output_format": row["output_format"], "role": role, "subtoken": subtoken,
                    "span_length": len(positions), "token_position": int(position),
                    "token_id": int(row["prompt_ids"][position]),
                    "token_text": tok.convert_ids_to_tokens([int(row["prompt_ids"][position])])[0],
                })
    cells = Counter(
        (row["partition"], row["default_factor"], row["hit_factor"], row["conflict_factor"], row["surface_factor"], row["output_format"])
        for row in cases
    )
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    old = historical_values()
    zero = zero_models(cases)
    rng = np.random.default_rng(1643)
    permutations = [rng.permutation(256).astype(int).tolist() for _ in range(8)]
    checks = {
        "authorization": True,
        "counts": (len(units), len(cases), len(compiled)) == (24, 768, 768),
        "partitions": Counter(row["partition"] for row in units) == {name: 8 for name in PARTITIONS},
        "factorial": cells == {(partition, *cell): 8 for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=5)},
        "truth_formula": all(row["truth_factor"] == (row["default_factor"] if row["hit_factor"] == -1 else row["default_factor"] * row["conflict_factor"]) for row in cases),
        "default_required": all(row["truth_factor"] == row["default_factor"] for row in cases if row["hit_factor"] == -1),
        "override_required": all(row["truth_factor"] == -row["default_factor"] for row in cases if row["hit_factor"] == 1 and row["conflict_factor"] == -1),
        "zero_models": max(value for key, value in zero.items() if key != "semantic_oracle") <= 0.75 and zero["semantic_oracle"] == 1.0,
        "unique": len({row["prompt"] for row in cases}) == 768,
        "fresh": not (set(fresh) & old) and len(fresh) == len(set(fresh)),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "candidate_singletons": all(len(value) == 1 for row in compiled for value in row["candidate_ids"]),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "machine_naturalness": all("General policy:" in row["prompt"] and "Inspection exception:" in row["prompt"] and "Under the policy and inspection exception" in row["prompt"] for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old), "zero": zero})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.write_rows(OUT / "protocol/role_occurrence_manifest.jsonl", occurrences)
    protocol = {
        "phase": 1643, "campaign": CAMPAIGN, "created_at_utc": now(),
        "status": "identifiable_default_override_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "default inheritance and item-specific conflicting exception override with explicit applicability interface",
        "factors": ["default", "exception_hit", "exception_conflict", "surface", "output_vocabulary"],
        "truth_formula": "t=d if h=-1 else d*k",
        "partitions": list(PARTITIONS), "units": 24, "cases": 768,
        "roles": list(ROLES), "effects": list(EFFECTS), "states": STATES,
        "activation_coordinates": DIM, "occurrences": len(occurrences),
        "archive": {"path": "raw/qwen3_role_subtoken_all_states.uint16.npy", "shape": [STATES, len(occurrences), DIM], "dtype": "uint16 exact BF16 bit patterns", "fixed_width": WIDTH, "batch_size": BATCH},
        "behavior_gates": {"overall_min": 0.80, "each_partition_min": 0.75, "default_inheritance_min": 0.80, "conflicting_override_min": 0.80, "each_output_format_min": 0.75},
        "discovery_rule": {"partition": "discovery", "effect": "override", "eligible_states": list(range(1, 31)), "eligible_roles": list(ROLES), "minimum_half_norm": 0.20, "score": "split_half_cosine * min(split_half_norms)", "support_k": 256},
        "validation_gates": {"confirmation_lockbox_cosine_min": 0.80, "each_to_discovery_cosine_min": 0.75, "each_support_overlap_min": 0.40, "coordinate_assignment_cells_required": 4},
        "movement_permutations": permutations,
        "intervention_modes": ["frozen_support"] + [f"movement_permutation_{i}" for i in range(8)] + ["selected_role", "default_roles", "exception_roles", "query_roles", "all_roles", "boundary_common_only", "boundary_residual_only", "boundary_full"],
        "effect_definitions": {
            "truth": "mean(t*H) over 32 cells",
            "default_inheritance": "mean(d*H | hit=-1)",
            "exception_read": "mean(t*H | hit=+1)",
            "override": "mean((-d)*(H_hit-H_miss) | conflict=-1)",
            "same_hit_control": "mean(d*(H_hit-H_miss) | conflict=+1)",
        },
        "common_residual": {"role": "boundary", "state": 30, "reference_families": ["C115_attribute", "C115_agent", "C116_negation", "C117_explicit_log"], "definition": "leave-C119-out normalized sum of normalized family means", "status": "registered component intervention; no semantic-name gate; wrong-family residual unavailable under matched geometry"},
        "completion_rule": "behavior qualification precedes hidden-state adjudication; after nomination, every registered field and intervention branch runs; a failed branch retires only that branch",
        "typed_missingness": {"human_naturalness": "machine-only controlled English", "cross_model": "Qwen3 only", "natural_paraphrases": "not tested", "wrong_family_residual": "not identifiable with matched prompt geometry", "attention_mlp": "excluded"},
        "claim_boundary": "activation-coordinate study of a controlled default/exception contract; no weights, attention/MLP, semantic neurons, endogenous route, common module, semantic residual, orthogonal subspace, topology, algebra, universal language mechanism, or new mathematics",
        "paired_material_policy": "reuse C118 lexical units and partitions, change only the preregistered applicability interface, and do not inspect C118 hidden states",
        "source_hashes": {"c118_closure": core.sha(C118 / "analysis/closure.json"), "c118_audit": core.sha(C118 / "audit/independent_closure_audit.json")},
        "material_digest": core.digest([*units, *cases]), "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1644_c119_cuda_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1643, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "occurrences": len(occurrences), "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def lookup_manifest() -> tuple[dict[int, list[dict]], dict[tuple[int, str], list[int]]]:
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    by_row: dict[int, list[dict]] = defaultdict(list)
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        row_index = int(occurrence["row_index"])
        by_row[row_index].append(occurrence)
        lookup[(row_index, occurrence["role"])].append(int(occurrence["occurrence_index"]))
    return by_row, lookup


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C119 independent contract audit missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    by_row, lookup = lookup_manifest()
    raw_path = OUT / protocol["archive"]["path"]
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (raw_path, logits_path, index_path)):
        raise RuntimeError("C119 raw output already exists")
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
                    positions_here = [int(item["token_position"]) for item in occurrences]
                    field[state_index, indices] = state[local, positions_here].contiguous().view(torch.uint16).cpu().numpy()
            for local, row in enumerate(batch):
                row_index = start + local
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                candidate_logits[row_index] = scores
                prediction = int(scores[1] > scores[0])
                behavior.append({
                    "row_index": row_index, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"],
                    "default_factor": row["default_factor"], "hit_factor": row["hit_factor"], "conflict_factor": row["conflict_factor"],
                    "surface_factor": row["surface_factor"], "output_format": row["output_format"], "truth_factor": row["truth_factor"],
                    "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"],
                    "positive_minus_negative": scores[0] - scores[1],
                })
            if start == 0:
                first_rows = batch
            if (start // BATCH + 1) % 12 == 0:
                field.flush(); candidate_logits.flush()
                print(f"[phase1644] captured {start + len(batch)}/{len(rows)}", flush=True)
            del output, logits, ids, mask, positions
        field.flush(); candidate_logits.flush()
        output, logits, ids, mask, positions, lengths = capture_base.forward(model, first_rows, pad, device, WIDTH)
        for state_index, state in enumerate(output.hidden_states):
            for local in range(len(first_rows)):
                occurrences = by_row[local]
                indices = np.asarray([int(item["occurrence_index"]) for item in occurrences], dtype=np.int64)
                positions_here = [int(item["token_position"]) for item in occurrences]
                old = np.asarray(field[state_index, indices])
                new = state[local, positions_here].contiguous().view(torch.uint16).cpu().numpy()
                if not np.array_equal(old, new):
                    repeat_hidden = max(repeat_hidden, float(np.max(np.abs(decode(old) - decode(new)))))
        for local, row in enumerate(first_rows):
            scores = np.asarray([float(logits[local, c[0]]) for c in row["candidate_ids"]], dtype=np.float32)
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
    causal_prefix = format_previsible = 0.0
    by_unit: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_unit[row["unit_id"]].append(row_index)
    for indices in by_unit.values():
        reference = indices[0]
        for row_index in indices[1:]:
            left = capture_base.role_bits(field, lookup, reference, "focus_pre")
            right = capture_base.role_bits(field, lookup, row_index, "focus_pre")
            if not np.array_equal(left, right):
                causal_prefix = max(causal_prefix, float(np.max(np.abs(decode(left) - decode(right)))))
        unit_rows = [rows[index] for index in indices]
        for d, h, k, s in itertools.product((1, -1), repeat=4):
            yes_index = next(index for index, row in zip(indices, unit_rows, strict=True) if (row["default_factor"], row["hit_factor"], row["conflict_factor"], row["surface_factor"], row["output_format"]) == (d, h, k, s, 1))
            true_index = next(index for index, row in zip(indices, unit_rows, strict=True) if (row["default_factor"], row["hit_factor"], row["conflict_factor"], row["surface_factor"], row["output_format"]) == (d, h, k, s, -1))
            for role in ROLES[:-1]:
                left = capture_base.role_bits(field, lookup, yes_index, role)
                right = capture_base.role_bits(field, lookup, true_index, role)
                if not np.array_equal(left, right):
                    format_previsible = max(format_previsible, float(np.max(np.abs(decode(left) - decode(right)))))
    def acc(selected: list[dict]) -> float:
        return float(np.mean([row["correct"] for row in selected]))
    summary = {
        "overall": acc(behavior),
        "by_partition": {p: acc([row for row in behavior if row["partition"] == p]) for p in PARTITIONS},
        "default_inheritance": acc([row for row in behavior if row["hit_factor"] == -1]),
        "conflicting_override": acc([row for row in behavior if row["hit_factor"] == 1 and row["conflict_factor"] == -1]),
        "same_exception": acc([row for row in behavior if row["hit_factor"] == 1 and row["conflict_factor"] == 1]),
        "by_output_format": {str(o): acc([row for row in behavior if row["output_format"] == o]) for o in (1, -1)},
    }
    gates = protocol["behavior_gates"]
    gate_checks = {
        "overall": summary["overall"] >= gates["overall_min"],
        "partitions": all(value >= gates["each_partition_min"] for value in summary["by_partition"].values()),
        "default": summary["default_inheritance"] >= gates["default_inheritance_min"],
        "override": summary["conflicting_override"] >= gates["conflicting_override_min"],
        "formats": all(value >= gates["each_output_format_min"] for value in summary["by_output_format"].values()),
    }
    checks = {
        "shape": list(field.shape) == protocol["archive"]["shape"], "dtype": field.dtype == np.uint16,
        "logits": list(candidate_logits.shape) == [768, 2] and bool(np.isfinite(candidate_logits).all()),
        "index": len(behavior) == 768, "repeat": repeat_hidden == 0.0 and repeat_logits == 0.0,
        "causal_prefix": causal_prefix == 0.0, "format_previsible": format_previsible == 0.0,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1644, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "cuda_capture_complete",
        "shape": list(field.shape), "raw_data_bytes": int(field.nbytes), "raw_file_bytes": raw_path.stat().st_size,
        "raw_sha256": core.sha(raw_path), "logits_sha256": core.sha(logits_path), "index_sha256": core.sha(index_path),
        "behavior": summary, "behavior_gate_checks": gate_checks, "behavior_gate_passed": all(gate_checks.values()),
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logits_max_abs": repeat_logits, "causal_prefix_max_abs": causal_prefix, "format_previsible_max_abs": format_previsible},
        "runtime": {"placement": placement, "quantization": quant}, "checks": checks,
        "authorization": "run_phase1645_c119_discovery" if all(gate_checks.values()) else "close_hidden_state_route_and_continue_campaign_missingness_ledger",
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def effect_coefficients(row: dict) -> dict[str, float]:
    d, h, k, t = (float(row[key]) for key in ("default_factor", "hit_factor", "conflict_factor", "truth_factor"))
    return {
        "truth": t / 32.0,
        "default_inheritance": d / 16.0 if h == -1 else 0.0,
        "exception_read": t / 16.0 if h == 1 else 0.0,
        "override": (-d * h) / 8.0 if k == -1 else 0.0,
        "same_hit_control": (d * h) / 8.0 if k == 1 else 0.0,
    }


def derive_fields(partitions: set[str], path: Path) -> tuple[np.ndarray, list[dict]]:
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = [row for row in core.rows(OUT / "material/units.jsonl") if row["partition"] in partitions]
    _, lookup = lookup_manifest()
    raw = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    role_index = {role: index for index, role in enumerate(ROLES)}
    effect_index = {effect: index for index, effect in enumerate(EFFECTS)}
    fields = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(units), len(EFFECTS), len(ROLES), STATES, DIM))
    fields[:] = 0.0
    for state in range(STATES):
        for row_index, row in enumerate(rows):
            if row["partition"] not in partitions:
                continue
            coefficients = effect_coefficients(row)
            u = unit_index[row["unit_id"]]
            for role in ROLES:
                values = np.mean(decode(raw[state, lookup[(row_index, role)]]), axis=0, dtype=np.float32)
                for effect, coefficient in coefficients.items():
                    if coefficient:
                        fields[u, effect_index[effect], role_index[role], state] += coefficient * values
        if state % 6 == 0 or state == 36:
            fields.flush(); print(f"[C119 fields] {sorted(partitions)} state {state}/36", flush=True)
    fields.flush()
    return fields, units


def discover() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    if not core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"] or not capture_report["behavior_gate_passed"]:
        raise RuntimeError("C119 behavior qualification missing")
    path = OUT / "analysis/discovery_unit_effect_role_state.float32.npy"
    fields, units = derive_fields({"discovery"}, path)
    effect_index = EFFECTS.index("override")
    candidates = []
    minimum = protocol["discovery_rule"]["minimum_half_norm"]
    for r, role in enumerate(ROLES):
        for state in protocol["discovery_rule"]["eligible_states"]:
            left = np.mean(fields[:4, effect_index, r, state], axis=0, dtype=np.float32)
            right = np.mean(fields[4:, effect_index, r, state], axis=0, dtype=np.float32)
            left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
            split = cosine(left, right)
            score = split * min(left_norm, right_norm) if min(left_norm, right_norm) >= minimum else None
            candidates.append({"role": role, "state": int(state), "effect": "override", "split_half_cosine": split, "left_norm": left_norm, "right_norm": right_norm, "score": score})
    eligible = [row for row in candidates if row["score"] is not None]
    winner = sorted(eligible, key=lambda row: (-row["score"], -row["split_half_cosine"], row["state"], ROLES.index(row["role"])))[0]
    mean = np.mean(fields[:, effect_index, ROLES.index(winner["role"]), winner["state"]], axis=0, dtype=np.float32)
    support = topk(mean, 256)
    table_path = OUT / "analysis/discovery_candidate_table.jsonl"
    core.write_rows(table_path, candidates)
    nomination = {**winner, "support_k": 256, "support": support, "field_norm": float(np.linalg.norm(mean)), "discovery_units": [row["unit_id"] for row in units], "field_sha256": core.sha(path), "candidate_table_sha256": core.sha(table_path), "created_at_utc": now()}
    nomination_path = OUT / "protocol/frozen_discovery_nomination.json"
    core.save(nomination_path, nomination)
    report = {"phase": 1645, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "override_candidate_frozen", "winner": winner, "support_k": 256, "checks": {"units": len(units) == 8, "shape": list(fields.shape) == [8, 5, 9, 37, 2560], "candidates": len(candidates) == 270, "eligible": bool(eligible), "support": len(set(support)) == 256}, "field_sha256": core.sha(path), "nomination_sha256": core.sha(nomination_path), "authorization": "execute_phase1646_c119_validation"}
    if not all(report["checks"].values()):
        raise RuntimeError(report)
    core.save(OUT / "analysis/discovery_freeze.json", report)
    print(json.dumps(report, indent=2))


def old_common(role: str, state: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if role not in ("focus_pre", "focus_post", "query_focus", "query_anchor", "boundary"):
        raise RuntimeError(("old common role unavailable", role))
    old_roles = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary")
    r = old_roles.index(role)
    c115 = np.load(C115 / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    references = {
        "attribute": np.mean(np.asarray(c115[0, :, r, state], dtype=np.float32), axis=0, dtype=np.float32),
        "agent": np.mean(np.asarray(c115[1, :, r, state], dtype=np.float32), axis=0, dtype=np.float32),
    }
    c116d = np.load(C116 / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r")
    c116v = np.load(C116 / "analysis/validation_unit_truth_role_state.float32.npy", mmap_mode="r")
    references["negation"] = np.mean(np.concatenate((np.asarray(c116d[:, r, state]), np.asarray(c116v[:, r, state])), axis=0), axis=0, dtype=np.float32)
    c117d = np.load(C117 / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r")
    c117v = np.load(C117 / "analysis/validation_unit_truth_role_state.float32.npy", mmap_mode="r")
    references["explicit_log"] = np.mean(np.concatenate((np.asarray(c117d[:, r, state]), np.asarray(c117v[:, r, state])), axis=0), axis=0, dtype=np.float32)
    common = unit(sum((unit(value) for value in references.values()), start=np.zeros(DIM, dtype=np.float32)))
    return common, references


def intervention_pairs(rows: list[dict], partitions: set[str]) -> list[dict]:
    by_key = {(row["unit_id"], row["default_factor"], row["hit_factor"], row["conflict_factor"], row["surface_factor"], row["output_format"]): row for row in rows}
    pairs = []
    units = [row for row in core.rows(OUT / "material/units.jsonl") if row["partition"] in partitions]
    for unit_row in units:
        unit = unit_row["unit_id"]
        for d, s, o in itertools.product((1, -1), repeat=3):
            recipient = by_key[(unit, d, -1, -1, s, o)]
            donor = by_key[(unit, d, 1, -1, s, o)]
            pairs.append({"pair_id": f"{unit}-d{d}-s{s}-o{o}", "unit_id": unit, "partition": unit_row["partition"], "default_factor": d, "surface_factor": s, "output_format": o, "recipient": recipient, "donor": donor, "target_truth": int(donor["truth_factor"])})
    return pairs


@torch.inference_mode()
def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    if not core.load(OUT / "audit/independent_discovery_audit.json")["all_checks_passed"]:
        raise RuntimeError("C119 discovery audit missing")
    path = OUT / "analysis/validation_unit_effect_role_state.float32.npy"
    fields, units = derive_fields({"confirmation", "lockbox"}, path)
    e, r, state = EFFECTS.index("override"), ROLES.index(nomination["role"]), int(nomination["state"])
    discovery = np.load(OUT / "analysis/discovery_unit_effect_role_state.float32.npy", mmap_mode="r")
    d = np.mean(discovery[:, e, r, state], axis=0, dtype=np.float32)
    c = np.mean(fields[:8, e, r, state], axis=0, dtype=np.float32)
    l = np.mean(fields[8:, e, r, state], axis=0, dtype=np.float32)
    support = set(nomination["support"])
    metrics = {
        "role": nomination["role"], "state": state,
        "confirmation_lockbox_cosine": cosine(c, l), "confirmation_to_discovery_cosine": cosine(c, d), "lockbox_to_discovery_cosine": cosine(l, d),
        "confirmation_support_overlap": len(set(topk(c, 256)) & support) / 256, "lockbox_support_overlap": len(set(topk(l, 256)) & support) / 256,
    }
    gates = protocol["validation_gates"]
    field_checks = {
        "confirmation_lockbox": metrics["confirmation_lockbox_cosine"] >= gates["confirmation_lockbox_cosine_min"],
        "to_discovery": min(metrics["confirmation_to_discovery_cosine"], metrics["lockbox_to_discovery_cosine"]) >= gates["each_to_discovery_cosine_min"],
        "support_overlap": min(metrics["confirmation_support_overlap"], metrics["lockbox_support_overlap"]) >= gates["each_support_overlap_min"],
    }
    effect_metrics = {}
    for effect_index, effect in enumerate(EFFECTS):
        c_mean = np.mean(fields[:8, effect_index, r, state], axis=0, dtype=np.float32)
        l_mean = np.mean(fields[8:, effect_index, r, state], axis=0, dtype=np.float32)
        effect_metrics[effect] = {"confirmation_lockbox_cosine": cosine(c_mean, l_mean), "confirmation_norm": float(np.linalg.norm(c_mean)), "lockbox_norm": float(np.linalg.norm(l_mean))}
    common, references = old_common("boundary", 30)
    boundary_r = ROLES.index("boundary")
    c_boundary = np.mean(fields[:8, e, boundary_r, 30], axis=0, dtype=np.float32)
    l_boundary = np.mean(fields[8:, e, boundary_r, 30], axis=0, dtype=np.float32)
    def residual(value: np.ndarray) -> np.ndarray:
        return value - float(np.dot(value, common)) * common
    residual_metrics = {
        "definition": "leave-C119-out G at boundary@state30; E=R-dot(R,G)G; names remain geometric",
        "reference_pairwise_cosines": {f"{a}_{b}": cosine(references[a], references[b]) for i, a in enumerate(references) for b in list(references)[i + 1:]},
        "override_to_common": {"confirmation": cosine(c_boundary, common), "lockbox": cosine(l_boundary, common)},
        "residual_cross_partition_cosine": cosine(residual(c_boundary), residual(l_boundary)),
        "residual_norm_fraction": {"confirmation": float(np.linalg.norm(residual(c_boundary)) / max(np.linalg.norm(c_boundary), 1e-12)), "lockbox": float(np.linalg.norm(residual(l_boundary)) / max(np.linalg.norm(l_boundary), 1e-12))},
    }
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    pairs = intervention_pairs(rows, {"confirmation", "lockbox"})
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        recipient = pair["recipient"]
        grouped[(pair["partition"], pair["output_format"], len(recipient["prompt_ids"]))].append(pair)
    results, model, first_repeat = [], None, None
    repeat_hidden = repeat_logits = 0.0
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        support_tensor = torch.tensor(nomination["support"], dtype=torch.long, device=device)
        permutations = [torch.tensor(values, dtype=torch.long, device=device) for values in protocol["movement_permutations"]]
        common_tensor = torch.tensor(common, dtype=torch.float32, device=device)
        for (partition, output_format, length), group in sorted(grouped.items()):
            for start in range(0, len(group), BATCH):
                batch = group[start:start + BATCH]
                recipients = [pair["recipient"] for pair in batch]
                donors = [pair["donor"] for pair in batch]
                rec_logits, rec_states = transport.forward_with_roles(model, recipients, ROLES, state, pad, device, WIDTH)
                don_logits, don_states = transport.forward_with_roles(model, donors, ROLES, state, pad, device, WIDTH)
                if state == 30:
                    rec_boundary_states, don_boundary_states = rec_states, don_states
                else:
                    _, rec_boundary_states = transport.forward_with_roles(model, recipients, ROLES, 30, pad, device, WIDTH)
                    _, don_boundary_states = transport.forward_with_roles(model, donors, ROLES, 30, pad, device, WIDTH)
                if first_repeat is None:
                    first_repeat = (recipients, rec_logits.detach().clone(), {role: value.detach().clone() for role, value in rec_states.items()})
                rec_selected, don_selected = rec_states[nomination["role"]], don_states[nomination["role"]]
                delta = don_selected[..., support_tensor] - rec_selected[..., support_tensor]
                target_norm = torch.sqrt(torch.sum(delta.float() ** 2, dim=(1, 2))).clamp_min(1e-12)
                patches = {}
                value = rec_selected.clone(); value[..., support_tensor] = don_selected[..., support_tensor]
                patches["frozen_support"] = {nomination["role"]: value}
                permutation_norms = []
                for index, permutation in enumerate(permutations):
                    value = rec_selected.clone(); value[..., support_tensor] = (rec_selected[..., support_tensor].float() + delta.float()[..., permutation]).to(rec_selected.dtype)
                    patches[f"movement_permutation_{index}"] = {nomination["role"]: value}
                    permutation_norms.append(torch.sqrt(torch.sum((value[..., support_tensor] - rec_selected[..., support_tensor]).float() ** 2, dim=(1, 2))).clamp_min(1e-12))
                patches["selected_role"] = {nomination["role"]: don_selected.clone()}
                patches["default_roles"] = {role: don_states[role].clone() for role in ("default_kind", "default_value")}
                patches["exception_roles"] = {role: don_states[role].clone() for role in ("exception_target", "exception_value")}
                patches["query_roles"] = {role: don_states[role].clone() for role in ("query_focus", "query_anchor")}
                patches["all_roles"] = {role: don_states[role].clone() for role in ROLES}
                rec_boundary, don_boundary = rec_boundary_states["boundary"], don_boundary_states["boundary"]
                full_delta = don_boundary.float() - rec_boundary.float()
                coefficient = torch.sum(full_delta * common_tensor, dim=-1, keepdim=True)
                common_delta = coefficient * common_tensor
                residual_delta = full_delta - common_delta
                patches["boundary_common_only"] = {"boundary": (rec_boundary.float() + common_delta).to(rec_boundary.dtype)}
                patches["boundary_residual_only"] = {"boundary": (rec_boundary.float() + residual_delta).to(rec_boundary.dtype)}
                patches["boundary_full"] = {"boundary": don_boundary.clone()}
                patched = {mode: transport.forward_patched_roles(model, recipients, values, state if not mode.startswith("boundary_") else 30, pad, device, WIDTH) for mode, values in patches.items()}
                for local, pair in enumerate(batch):
                    base = transport.margin(rec_logits[local], recipients[local])
                    target = float(pair["target_truth"])
                    modes = {}
                    for mode, logits in patched.items():
                        margin = transport.margin(logits[local], recipients[local])
                        modes[mode] = {"target_direction_gain": target * (margin - base), "target_reached": target * margin > 0}
                    results.append({
                        "pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "partition": partition, "output_format": output_format,
                        "default_factor": pair["default_factor"], "surface_factor": pair["surface_factor"], "target_truth": pair["target_truth"],
                        "recipient_margin": base, "target_movement_l2": float(target_norm[local]),
                        "permutation_l2_relative_errors": [float(torch.abs(value[local] - target_norm[local]) / target_norm[local]) for value in permutation_norms],
                        "modes": modes,
                    })
                print(f"[phase1646] {partition}/format={output_format}/len={length} {start + len(batch)}/{len(group)}", flush=True)
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
        for output_format in (1, -1):
            selected = [row for row in results if row["partition"] == partition and row["output_format"] == output_format]
            correct = med([row["modes"]["frozen_support"]["target_direction_gain"] for row in selected])
            permutation_medians = [med([row["modes"][f"movement_permutation_{i}"]["target_direction_gain"] for row in selected]) for i in range(8)]
            mode_names = ["selected_role", "default_roles", "exception_roles", "query_roles", "all_roles", "boundary_common_only", "boundary_residual_only", "boundary_full"]
            mode_medians = {mode: med([row["modes"][mode]["target_direction_gain"] for row in selected]) for mode in mode_names}
            summaries.append({
                "partition": partition, "output_format": output_format, "pairs": len(selected), "independent_units": len({row["unit_id"] for row in selected}),
                "frozen_support_median_gain": correct, "permutation_median_gains": permutation_medians,
                "frozen_support_gt_permutation_median": correct > med(permutation_medians), "frozen_support_gt_all_permutations": all(correct > value for value in permutation_medians),
                "mode_median_gains": mode_medians,
                "target_reached_counts": {mode: sum(row["modes"][mode]["target_reached"] for row in selected) for mode in results[0]["modes"]},
            })
    summary_path = OUT / "analysis/validation_summary.jsonl"
    core.write_rows(summary_path, summaries)
    predictions = {
        "field_passed": all(field_checks.values()),
        "coordinate_assignment_cells": sum(row["frozen_support_gt_permutation_median"] for row in summaries),
        "strict_assignment_cells": sum(row["frozen_support_gt_all_permutations"] for row in summaries),
        "common_positive_cells": sum(row["mode_median_gains"]["boundary_common_only"] > 0 for row in summaries),
        "residual_positive_cells": sum(row["mode_median_gains"]["boundary_residual_only"] > 0 for row in summaries),
        "full_gt_each_component_cells": sum(row["mode_median_gains"]["boundary_full"] > max(row["mode_median_gains"]["boundary_common_only"], row["mode_median_gains"]["boundary_residual_only"]) for row in summaries),
    }
    max_error = max(error for row in results for error in row["permutation_l2_relative_errors"])
    checks = {"units": len(units) == 16, "shape": list(fields.shape) == [16, 5, 9, 37, 2560], "pairs": len(results) == 128, "summaries": len(summaries) == 4 and all(row["pairs"] == 32 and row["independent_units"] == 8 for row in summaries), "l2": max_error <= 0.02, "finite": all(math.isfinite(row["modes"][mode]["target_direction_gain"]) for row in results for mode in row["modes"]), "repeat": repeat_hidden == 0.0 and repeat_logits == 0.0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1646, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "confirmation_lockbox_validation_complete",
        "nomination": {"effect": "override", "role": nomination["role"], "state": state, "support_k": 256},
        "field_metrics": metrics, "field_checks": field_checks, "effect_metrics": effect_metrics,
        "leave_c119_out_common_residual": residual_metrics, "predictions": predictions,
        "primary_checks": {"field": predictions["field_passed"], "coordinate_assignment": predictions["coordinate_assignment_cells"] == gates["coordinate_assignment_cells_required"]},
        "max_l2_relative_error": max_error, "checks": checks, "runtime": {"placement": placement, "quantization": quant},
        "field_sha256": core.sha(path), "results_sha256": core.sha(result_path), "summary_sha256": core.sha(summary_path),
        "authorization": "run_phase1647_c119_synthesis_visualization_and_closure",
    }
    core.save(OUT / "analysis/validation_adjudication.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_validation_audit.json")["all_checks_passed"]:
        raise RuntimeError("C119 validation audit missing")
    discovery_report = core.load(OUT / "analysis/discovery_freeze.json")
    validation = core.load(OUT / "analysis/validation_adjudication.json")
    nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    discovery = np.load(OUT / "analysis/discovery_unit_effect_role_state.float32.npy", mmap_mode="r")
    heldout = np.load(OUT / "analysis/validation_unit_effect_role_state.float32.npy", mmap_mode="r")
    means = [np.mean(discovery, axis=0, dtype=np.float32), np.mean(heldout[:8], axis=0, dtype=np.float32), np.mean(heldout[8:], axis=0, dtype=np.float32)]
    payload = core.load(PUBLIC)
    display_states = tuple(sorted(set(KEY_STATES) | {int(nomination["state"])}))
    effect_rows = []
    for partition, mean in zip(PARTITIONS, means, strict=True):
        for effect in ("truth", "default_inheritance", "exception_read", "override", "same_hit_control"):
            e = EFFECTS.index(effect)
            for r, role in enumerate(ROLES):
                for state in display_states:
                    effect_rows.append({"dataset": "C119", "family": FAMILY, "partition": partition, "role": role, "state": state, "state_kind": "embedding" if state == 0 else "hidden_state", "effect": effect, "values": np.asarray(mean[e, r, state], dtype=np.float32).tolist()})
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    raw = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    occurrence_lookup: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for occurrence in manifest:
        occurrence_lookup[(int(occurrence["row_index"]), occurrence["role"])].append(occurrence)
    raw_rows = []
    for partition in PARTITIONS:
        row_index = next(i for i, row in enumerate(compiled) if row["partition"] == partition and row["default_factor"] == 1 and row["hit_factor"] == 1 and row["conflict_factor"] == -1 and row["surface_factor"] == 1 and row["output_format"] == 1)
        row = compiled[row_index]
        for role in ("default_value", "exception_target", "exception_value", "query_anchor", "boundary"):
            occurrence = occurrence_lookup[(row_index, role)][0]
            occurrence_index = int(occurrence["occurrence_index"])
            for state in RAW_STATES:
                raw_rows.append({"dataset": "C119", "case_id": row["case_id"], "family": FAMILY, "partition": partition, "default_factor": 1, "hit_factor": 1, "conflict_factor": -1, "surface_factor": 1, "output_format": 1, "role": role, "subtoken": int(occurrence["subtoken"]), "token_position": int(occurrence["token_position"]), "token_id": int(occurrence["token_id"]), "token_text": occurrence["token_text"], "state": state, "state_kind": "embedding" if state == 0 else "hidden_state", "values": decode(raw[state, occurrence_index]).tolist()})
    payload["effect_rows"] = [row for row in payload["effect_rows"] if row.get("dataset") != "C119"] + effect_rows
    payload["raw_rows"] = [row for row in payload["raw_rows"] if row.get("dataset") != "C119"] + raw_rows
    payload["default_coordinates"] = nomination["support"][:64]
    payload["scale"] = {
        "effect_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["effect_rows"]]), 0.99)),
        "raw_symmetric_abs_q99": float(np.quantile(np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["raw_rows"]]), 0.99)),
    }
    payload.update({
        "phase": 1647, "campaign": "C109-C119", "title": "C109-C119 Relation-Role-State Activation Atlas",
        "c119_batch": {"contract": {"factors": protocol["factors"], "truth_formula": protocol["truth_formula"]}, "capture": core.load(OUT / "analysis/capture_summary.json"), "discovery": discovery_report, "validation": validation, "summaries": core.rows(OUT / "analysis/validation_summary.jsonl"), "nomination": {"effect": "override", "role": nomination["role"], "state": nomination["state"], "support_k": 256}},
        "claim_boundary": "C119 tests a controlled identifiable default-inheritance and conflicting item-exception override. It displays all 2560 activation coordinates for registered embedding/HiddenState observations. It does not identify weights, attention/MLP, semantic neurons, endogenous routes, a common module, semantic residual, topology, algebra, a universal mechanism, or new mathematics.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c119_relation_role_state_atlas.json"
    core.save(canonical, payload); shutil.copyfile(canonical, PUBLIC)
    behavior = core.load(OUT / "analysis/capture_summary.json")["behavior"]
    closure = {
        "phase": 1647, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "identifiable_default_override_campaign_complete",
        "headline": {"behavior": behavior, "discovery_winner": discovery_report["winner"], "field_metrics": validation["field_metrics"], "field_checks": validation["field_checks"], "effect_metrics": validation["effect_metrics"], "coordinate_and_component_predictions": validation["predictions"], "leave_c119_out_common_residual": validation["leave_c119_out_common_residual"]},
        "new_puzzles": {
            "K311": "behavioral qualification of default inheritance and conflicting item-specific exception override under two output vocabularies",
            "K312": "discovery-frozen full-coordinate override contrast and held-out field result",
            "K313": "coordinate-assignment and leave-C119-out boundary projection component intervention result, with geometric names only",
        },
        "theory_update": "RDC now separates a natural run, researcher-defined contrasts, candidate coordinate implementations and interventions. C119 repairs the semantic object that C117 did not identify; projection components remain observations until selective deletion, wrong-residual and rescue controls exist under matched geometry.",
        "unified_formula": "H_s=Phi_{theta,<s}(E(x),kappa); R_e=E[w_e(x) H_s(x)]; Gamma=(s,C,S,V); y_I=O(Phi_{theta,>=s}(I_Gamma(H_s)))",
        "problems": ["controlled synthetic English and one Qwen3", "machine naturalness only", "the exception sentence remains explicit when it hits, although miss cases require default inheritance", "discovery searches states1-30 and may prefer late decision states", "activation patching can be off-distribution", "K256 and eight permutations do not prove minimality or semantic specificity", "leave-C119-out G is researcher-defined and output-boundary aligned", "wrong-family residual is not identifiable under matched prompt geometry", "no attention/MLP, weights, cross-model or natural paraphrase test"],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "activation_coordinates": 2560, "includes_embedding": True},
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": "C120 matched-output comparison-family observation campaign, while C119 missingness retains human naturalness, natural paraphrases, cross-model replication and matched wrong-family residual rescue",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"validation": True, "effects": len(effect_rows) == 3 * 5 * 9 * len(display_states), "raw": len(raw_rows) == 3 * 5 * len(RAW_STATES), "coordinates": all(len(row["values"]) == 2560 for row in [*effect_rows, *raw_rows]), "asset": core.sha(canonical) == core.sha(PUBLIC), "batch": "c119_batch" in payload, "boundary": "does not identify weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    audit = {"phase": 1647, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "authorization": "run_independent_closure_and_client_audits_then_append_memo"}
    core.save(OUT / "audit/internal_closure_audit.json", audit)
    print(json.dumps({"checks": checks, "headline": closure["headline"], "next_authorization": closure["next_authorization"]}, indent=2))


STAGES = {"contract": contract, "capture": capture, "discover": discover, "validate": validate, "synthesize": synthesize}


def main(stage: str) -> None:
    STAGES[stage]()


if __name__ == "__main__":
    main(sys.argv[1])
