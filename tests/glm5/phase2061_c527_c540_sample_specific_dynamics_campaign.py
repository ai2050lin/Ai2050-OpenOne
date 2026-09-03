#!/usr/bin/env python3
"""C527-C540 sample-specific HiddenState dynamics campaign.

The campaign observes token embeddings and HiddenState checkpoints only. It
retains all 2560 physical activation coordinates, performs no PCA or Top-K
selection, and never reads Attention, MLP activations, or model weights.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c539_sample_specific_dynamics_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import phase2052_c518_c525_fresh_callable_state_campaign as prior


PHASES = {
    f"C{campaign}": (2061 + campaign - 527, slug)
    for campaign, slug in (
        (527, "evidence_adjudication_and_sample_specific_dynamics_master_contract"),
        (528, "twenty_panel_language_program_graph_and_material_registry"),
        (529, "compiler_semantic_width_behavior_and_naturalness_audit"),
        (530, "qwen_average_exact_and_full_token_all_coordinate_capture"),
        (531, "c523_rollout_strong_baseline_tournament"),
        (532, "centered_sample_residual_and_pair_difference_rollout"),
        (533, "full_coordinate_response_neighborhood_trajectory"),
        (534, "leave_one_family_autonomous_dynamics_lockbox"),
        (535, "exact_token_autonomous_dynamics_and_writable_state_test"),
        (536, "broad_language_operation_response_ecology"),
        (537, "predictive_exact_token_and_causal_eligibility_adjudication"),
        (538, "qualified_exact_token_causal_branch_or_registered_na"),
        (539, "full_coordinate_visual_atlas_and_campaign_synthesis"),
        (540, "raw_field_cleanup_and_next_stage_adjudication"),
    )
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = prior.ROLES
ROLE_INDEX = prior.ROLE_INDEX
READOUTS = (0, 1, 8, 16, 24, 32, 37)
RIDGE = 1e-2
CONTROL_MARGIN = 0.02
COHORTS = ("legacy_program", "lexical_ecology", "fresh_program")
FLAGSHIP = ("nested_composition", "typed_graph_path", "temporal_composition")
LOFO_DOMAINS = (
    "legacy_program:nested_composition",
    "legacy_program:typed_graph_path",
    "legacy_program:temporal_composition",
    "lexical_ecology:lex_noun_taxonomy",
    "lexical_ecology:lex_polysemy",
    "lexical_ecology:lex_verb_event",
)

PARENT_AUDIT = RESULT / "phase2060_c526_fresh_callable_state_campaign_independent_audit/audit/independent_audit.json"
LEGACY_ROWS = prior.previous.OLD_CASES
LEGACY_COMPILED = prior.previous.OLD_COMPILED
LEGACY_BEHAVIOR = prior.previous.OLD_BEHAVIOR
LEXICAL_ROWS = prior.previous.OUTS["C502"] / "material/lexical_cases.jsonl"
LEXICAL_COMPILED = prior.previous.OUTS["C503"] / "compiled/qwen3_lexical.jsonl"
LEXICAL_BEHAVIOR = prior.previous.OUTS["C504"] / "raw/behavior.jsonl"
FRESH_ROWS = prior.OUTS["C518"] / "material/fresh_cases.jsonl"
FRESH_COMPILED = prior.OUTS["C519"] / "compiled/qwen3_fresh.jsonl"
FRESH_BEHAVIOR = prior.OUTS["C519"] / "raw/behavior.jsonl"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def begin(name: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[name]
    if (out / "analysis/final.json").exists():
        return out
    if out.exists():
        raise RuntimeError(f"partial output exists: {out}")
    if not all(checks.values()):
        raise RuntimeError((name, checks))
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": producer_hash(), **protocol,
    })
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    if (out / "analysis/final.json").exists():
        return load(out / "analysis/final.json")
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    protocol = load(out / "protocol/preregistration.json")
    final_checks = {
        "contract": load(out / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": all(checks.values()),
        "producer_hash": protocol["producer_sha256"] == producer_hash(),
    }
    value = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "checks": final_checks, "all_checks_passed": all(final_checks.values()),
        "headline": headline, "next_authorization": authorization,
    }
    save(out / "analysis/final.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def metric_acc() -> dict:
    return prior.metric_acc()


def add_metric(acc: dict, prediction: np.ndarray, truth: np.ndarray) -> None:
    prior.add_metric(acc, prediction, truth)


def finish_metric(acc: dict) -> dict:
    return prior.finish_metric(acc)


def vector_metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    return prior.vector_metric(prediction, truth)


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def model_base():
    return prior.previous.parent.previous.prior.model_base


def source_specs() -> list[tuple[str, Path, Path, Path]]:
    return [
        ("legacy_program", LEGACY_ROWS, LEGACY_COMPILED, LEGACY_BEHAVIOR),
        ("lexical_ecology", LEXICAL_ROWS, LEXICAL_COMPILED, LEXICAL_BEHAVIOR),
        ("fresh_program", FRESH_ROWS, FRESH_COMPILED, FRESH_BEHAVIOR),
    ]


def combined_material() -> tuple[list[dict], list[dict]]:
    rows, compiled = [], []
    for cohort, row_path, compiled_path, _behavior_path in source_specs():
        source_rows = read_rows(row_path)
        compiled_by_id = {row["case_id"]: row for row in read_rows(compiled_path)}
        for row in source_rows:
            domain_id = f"{cohort}:{row['family']}"
            enriched = {**row, "cohort": cohort, "domain_id": domain_id}
            rows.append(enriched)
            compiled.append({**compiled_by_id[row["case_id"]], "cohort": cohort, "domain_id": domain_id})
    return rows, compiled


def capture_paths() -> tuple[Path, Path, Path]:
    raw = OUTS["C530"] / "raw"
    return raw / "role_mean_states.float16.npy", raw / "role_exact_states.float16.npy", raw / "full_token_states.float16.npy"


def capture_index() -> list[dict]:
    return read_rows(OUTS["C530"] / "raw/hidden_index.jsonl")


def fit_bundle(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return prior.fit_bundle(x, y)


def predict_bundle(x: np.ndarray, beta: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return prior.predict_bundle(x, beta, intercept)


def domain_rows(index: list[dict], domain: str, partitions: set[str] | None = None) -> list[int]:
    return [
        int(row["hidden_index"]) for row in index
        if row["domain_id"] == domain and (partitions is None or row["partition"] in partitions)
    ]


def flagship_ids(index: list[dict], cohort: str, family: str) -> list[int]:
    return [int(row["hidden_index"]) for row in index if row["cohort"] == cohort and row["family"] == family]


def shuffled_indices(index_rows: list[dict], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(len(index_rows), dtype=np.int64)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for local, row in enumerate(index_rows):
        groups[(row["construction"], str(row.get("cell", row.get("bits"))))].append(local)
    for ids in groups.values():
        if len(ids) > 1:
            rolled = np.asarray(ids, np.int64)
            shift = int(rng.integers(1, len(ids)))
            order[rolled] = np.roll(rolled, shift)
    return order


def select_full_ids(rows: list[dict]) -> set[str]:
    selected: set[str] = set()
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["partition"] == "lockbox":
            groups[(row["cohort"], row["family"], row["construction"])].append(row)
    for values in groups.values():
        ordered = sorted(values, key=lambda row: row["case_id"])
        selected.add(ordered[0]["case_id"])
        selected.add(ordered[-1]["case_id"])
    return selected


def c527() -> None:
    parent_audit = load(PARENT_AUDIT)
    out = begin("C527", {
        "status": "sample_specific_dynamics_master_contract_frozen",
        "evidence_corrections": [
            "C516/C525 causal eligibility failed but causal result is NA, not a failed intervention",
            "C520-C525 are exploratory under C519 width-contract missingness",
            "tested same-coordinate affine keys failed; fixed-coordinate semantics in general were not disproved",
            "tested six-role same-coordinate linear prediction failed; single-sample prediction in general was not disproved",
            "C523 beat persistence only and is not yet a sample-specific language mechanism",
        ],
        "routes": [
            "strong-baseline rollout", "centered sample residual", "pair differences",
            "response-neighborhood transport", "leave-one-family rollout", "exact-token rollout",
            "broad operation response ecology", "qualified exact-token causality",
        ],
        "primary_gate": "candidate beats every frozen strong control by >=0.02 in every flagship family and surface, and predicts centered and pair-difference fields",
        "route_policy": "failure removes only that route; all observational routes continue",
        "forbidden": ["Attention", "MLP", "weights", "PCA", "Top-K", "post-reveal threshold changes"],
        "coordinates": "all 2560 physical activation coordinates",
    }, {"parent_audit": parent_audit.get("status") == "passed", "cuda": torch.cuda.is_available()})
    close("C527", {
        "status": "contract_closed", "retained_claims": 7, "corrected_overclaims": 4,
        "strict_target": "Determine whether C523 preserves sample-specific language trajectories beyond mean layer-depth evolution, while independently observing broad full-coordinate operation responses.",
    }, {"parent": parent_audit.get("status") == "passed", "routes": 8 == 8}, "C528_program_registry")


def c528() -> None:
    out = begin("C528", {
        "status": "twenty_panel_program_graph_registry_frozen",
        "sources": [str(item[1].relative_to(ROOT)) for item in source_specs()],
        "object": "external language-program metadata separated from internal HiddenState field",
        "natural_knowledge_boundary": "controlled prompt-internal programs and lexical microtasks; no claim of a complete natural-language ontology",
    }, {"parent": final("C527")["all_checks_passed"]})
    rows, compiled = combined_material()
    write_rows(out / "material/combined_cases.jsonl", rows)
    write_rows(out / "compiled/qwen3_combined.jsonl", compiled)
    domains = sorted({row["domain_id"] for row in rows})
    family_counts = {domain: sum(row["domain_id"] == domain for row in rows) for domain in domains}
    operators = defaultdict(int)
    for row in rows:
        for operator in row.get("semantic_graph", {}).get("operators", []):
            operators[str(operator)] += 1
    close("C528", {
        "status": "program_registry_closed", "rows": len(rows), "compiled_rows": len(compiled),
        "cohort_counts": {cohort: sum(row["cohort"] == cohort for row in rows) for cohort in COHORTS},
        "domains": domains, "domain_counts": family_counts, "registered_operator_counts": dict(sorted(operators.items())),
        "strict_interpretation": "Domain labels are researcher metadata for stratification, not variables supplied to Qwen3 or discovered neural guards.",
    }, {
        "rows": len(rows) == 8160, "compiled": len(compiled) == len(rows),
        "unique": len({row["case_id"] for row in rows}) == len(rows), "domains": len(domains) == 20,
    }, "C529_premodel_audit")


def c529() -> None:
    out = begin("C529", {
        "status": "premodel_and_reused_behavior_audit_frozen",
        "width_policy": "C519 129>128 remains a formal missingness label; the broad observational capture uses actual max width without rewriting the old gate",
        "behavior_policy": "reuse frozen source behavior ledgers and re-evaluate behavior during unified capture",
        "naturalness": "machine surface audit only; independent human naturalness remains missing",
    }, {"parent": final("C528")["all_checks_passed"]})
    rows = read_rows(OUTS["C528"] / "material/combined_cases.jsonl")
    compiled = read_rows(OUTS["C528"] / "compiled/qwen3_combined.jsonl")
    by_id = {row["case_id"]: row for row in rows}
    widths = [len(row["prompt_ids"]) for row in compiled]
    role_failures = [row["case_id"] for row in compiled if set(row.get("role_positions", {})) != set(ROLES) or any(not row["role_positions"][role] for role in ROLES)]
    behavior = []
    for cohort, _rows_path, _compiled_path, behavior_path in source_specs():
        for item in read_rows(behavior_path):
            behavior.append({**item, "cohort": cohort})
    behavior_by_id = {row["case_id"]: row for row in behavior}
    domain_accuracy = {}
    for domain in sorted({row["domain_id"] for row in rows}):
        ids = [row["case_id"] for row in rows if row["domain_id"] == domain]
        values = [bool(behavior_by_id[item]["correct"]) for item in ids]
        domain_accuracy[domain] = float(np.mean(values))
    balance = {
        domain: float(np.mean([by_id[row["case_id"]]["gold_position"] == 0 for row in compiled if row["domain_id"] == domain]))
        for domain in domain_accuracy
    }
    close("C529", {
        "status": "premodel_audit_closed", "rows": len(rows), "max_prompt_tokens": max(widths),
        "role_failures": len(role_failures), "domain_accuracy": domain_accuracy,
        "domain_first_position_rate": balance, "formal_missingness": {
            "fresh_program_width_contract": max(len(row["prompt_ids"]) for row in compiled if row["cohort"] == "fresh_program") > 128,
            "human_naturalness": "NA_not_run",
        },
        "field_authorized": True,
        "strict_interpretation": "Behavior and compiler strata qualify observation; C519's old frozen width gate remains failed and is not retroactively changed.",
    }, {
        "rows": len(rows) == 8160, "width": max(widths) <= 144, "roles": not role_failures,
        "behavior_complete": len(behavior_by_id) == len(rows), "finite": finite(domain_accuracy),
    }, "C530_qwen_capture")


@torch.inference_mode()
def c530() -> None:
    out = begin("C530", {
        "status": "qwen_three_view_full_coordinate_capture_frozen",
        "model": "local Qwen3-4B BF16 CUDA, no quantization",
        "views": [
            "all-case six-role span mean at q0-q37",
            "all-case exact last token of each registered role at q0-q37",
            "balanced 120-case all-token q0-q37 field",
        ],
        "coordinate_policy": "all 2560 coordinates retained; no PCA or Top-K",
    }, {"parent": final("C529")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = read_rows(OUTS["C528"] / "material/combined_cases.jsonl")
    compiled = read_rows(OUTS["C528"] / "compiled/qwen3_combined.jsonl")
    selected_ids = select_full_ids(rows)
    selected = [i for i, row in enumerate(rows) if row["case_id"] in selected_ids]
    full_lookup = {source_i: local_i for local_i, source_i in enumerate(selected)}
    n = len(rows)
    width = max(len(row["prompt_ids"]) for row in compiled)
    mean_path, exact_path, full_path = capture_paths()
    mean_states = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    exact_states = np.lib.format.open_memmap(exact_path, mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    full_states = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float16, shape=(len(selected), CHECKPOINTS, width, DIM))
    model = None
    hooks, captured, index = [], [], []
    try:
        model, tokenizer, device, placement = model_base().load_bf16("qwen3")
        quant = model_base().quantization_audit(model)
        base = model.model

        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(hook))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        batch_size = 8
        for start in range(0, n, batch_size):
            batch = compiled[start:start + batch_size]
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            pos = torch.zeros_like(ids)
            lengths = []
            role_weights = torch.zeros((len(batch), len(ROLES), width), dtype=torch.float32, device=device)
            exact_pos = torch.zeros((len(batch), len(ROLES)), dtype=torch.long, device=device)
            for local, row in enumerate(batch):
                values = row["prompt_ids"]
                lengths.append(len(values))
                ids[local, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[local, :len(values)] = 1
                pos[local, :len(values)] = torch.arange(len(values), device=device)
                for role_i, role in enumerate(ROLES):
                    positions = [int(value) for value in row["role_positions"][role]]
                    role_weights[local, role_i, positions] = 1.0 / len(positions)
                    exact_pos[local, role_i] = positions[-1]
            captured.clear()
            output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError(("checkpoint_count", len(captured)))
            for q, state in enumerate(captured):
                state32 = state.float()
                means = torch.einsum("brt,btd->brd", role_weights, state32).cpu().numpy().astype(np.float16)
                gather = exact_pos[:, :, None].expand(-1, -1, DIM)
                exact = torch.gather(state32, 1, gather).cpu().numpy().astype(np.float16)
                mean_states[start:start + len(batch), q] = means
                exact_states[start:start + len(batch), q] = exact
                for local in range(len(batch)):
                    source_i = start + local
                    if source_i in full_lookup:
                        full_states[full_lookup[source_i], q, :lengths[local]] = state[local, :lengths[local]].float().cpu().numpy().astype(np.float16)
            for local, row in enumerate(batch):
                source_i = start + local
                length = lengths[local]
                scores = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                meta = rows[source_i]
                index.append({
                    "hidden_index": source_i, "full_index": full_lookup.get(source_i),
                    "case_id": row["case_id"], "cohort": meta["cohort"], "domain_id": meta["domain_id"],
                    "family": meta["family"], "construction": meta["construction"], "surface": meta.get("surface", meta["construction"]),
                    "unit": int(meta["unit"]), "bits": meta["bits"], "cell": str(meta.get("cell", meta["bits"])),
                    "partition": meta["partition"], "length": length, "role_positions": row["role_positions"],
                    "gold_position": int(meta["gold_position"]), "prediction": prediction, "correct": prediction == int(meta["gold_position"]),
                })
            mean_states.flush(); exact_states.flush(); full_states.flush()
            if start % 128 == 0 or start + len(batch) == n:
                print(f"[C530 capture] {start + len(batch)}/{n}", flush=True)
        write_rows(out / "raw/hidden_index.jsonl", index)
        save(out / "raw/full_field_row_map.json", {"source_indices": selected, "qpoints": list(range(CHECKPOINTS)), "width": width})
        domain_accuracy = {
            domain: float(np.mean([row["correct"] for row in index if row["domain_id"] == domain]))
            for domain in sorted({row["domain_id"] for row in index})
        }
        headline = {
            "status": "capture_closed", "rows": n, "accuracy": float(np.mean([row["correct"] for row in index])),
            "domain_accuracy": domain_accuracy, "mean_shape": list(mean_states.shape), "exact_shape": list(exact_states.shape),
            "full_shape": list(full_states.shape), "full_rows": len(selected), "field_width": width,
            "placement": placement, "quantization": quant,
        }
    finally:
        for item in hooks:
            item.remove()
        close_mmap(mean_states); close_mmap(exact_states); close_mmap(full_states)
        model_base().release_bf16(model)
        gc.collect()
    close("C530", headline, {
        "rows": headline["rows"] == 8160, "mean": headline["mean_shape"] == [8160, 38, 6, 2560],
        "exact": headline["exact_shape"] == [8160, 38, 6, 2560], "full": headline["full_rows"] == 120,
        "bf16": headline["quantization"].get("has_bf16_parameters", False) and not headline["quantization"].get("has_quantized_modules", True),
        "finite": finite(headline),
    }, "C531_strong_rollout")


def run_flagship_rollout(states: np.ndarray, index: list[dict], archive: Path, label: str) -> dict:
    legacy_all = [row for row in index if row["cohort"] == "legacy_program" and row["family"] in FLAGSHIP]
    legacy_ids = [int(row["hidden_index"]) for row in legacy_all]
    test_rows = {
        family: sorted(
            [row for row in index if row["cohort"] == "fresh_program" and row["family"] == family],
            key=lambda row: int(row["hidden_index"]),
        )
        for family in FLAGSHIP
    }
    currents: dict[str, dict[str, np.ndarray]] = {}
    for family_i, family in enumerate(FLAGSHIP):
        ids = [int(row["hidden_index"]) for row in test_rows[family]]
        q0 = np.asarray(states[ids, 0], np.float32)
        order = shuffled_indices(test_rows[family], 527000 + family_i)
        currents[family] = {
            "origin": q0.copy(), "shared": q0.copy(), "family": q0.copy(),
            "sample_shuffle": q0[order].copy(), "role_reverse": q0[:, ::-1].copy(),
            "coordinate_roll": np.roll(q0, 137, axis=2).copy(),
        }
    trajectory: dict[str, dict] = {}
    for q in range(37):
        x_all = np.asarray(states[legacy_ids, q], np.float32)
        y_all = np.asarray(states[legacy_ids, q + 1], np.float32) - x_all
        shared_beta, shared_intercept = fit_bundle(x_all, y_all)
        for family in FLAGSHIP:
            train = [int(row["hidden_index"]) for row in legacy_all if row["family"] == family]
            fx = np.asarray(states[train, q], np.float32)
            fy = np.asarray(states[train, q + 1], np.float32) - fx
            family_beta, family_intercept = fit_bundle(fx, fy)
            values = currents[family]
            values["shared"] += predict_bundle(values["shared"], shared_beta, shared_intercept)
            values["family"] += predict_bundle(values["family"], family_beta, family_intercept)
            for control in ("sample_shuffle", "role_reverse", "coordinate_roll"):
                values[control] += predict_bundle(values[control], shared_beta, shared_intercept)
            if q + 1 in READOUTS:
                ids = [int(row["hidden_index"]) for row in test_rows[family]]
                truth = np.asarray(states[ids, q + 1], np.float32)
                trajectory[f"{family}:q{q + 1}"] = {
                    name: vector_metric(value, truth)
                    for name, value in values.items()
                }
        print(f"[{label} rollout] q{q + 1}/q37", flush=True)

    global_mean = np.asarray(states[legacy_ids, 37], np.float32).mean(axis=0)
    metrics: dict[str, dict[str, dict]] = {}
    surface_metrics: dict[str, dict[str, dict]] = {}
    gates: dict[str, bool] = {}
    prediction_ids, shared_predictions, family_predictions = [], [], []
    control_names = (
        "persistence", "global_mean", "family_mean", "family_surface_mean",
        "sample_shuffle", "role_reverse", "coordinate_roll",
    )
    for family in FLAGSHIP:
        rows = test_rows[family]
        ids = [int(row["hidden_index"]) for row in rows]
        truth = np.asarray(states[ids, 37], np.float32)
        train_family = [int(row["hidden_index"]) for row in legacy_all if row["family"] == family]
        family_mean = np.asarray(states[train_family, 37], np.float32).mean(axis=0)
        predictions = {
            "persistence": currents[family]["origin"],
            "global_mean": np.broadcast_to(global_mean, truth.shape),
            "family_mean": np.broadcast_to(family_mean, truth.shape),
            "shared": currents[family]["shared"],
            "family": currents[family]["family"],
            "sample_shuffle": currents[family]["sample_shuffle"],
            "role_reverse": currents[family]["role_reverse"],
            "coordinate_roll": currents[family]["coordinate_roll"],
        }
        matched = np.empty_like(truth)
        for surface in sorted({row["construction"] for row in rows}):
            test_local = [i for i, row in enumerate(rows) if row["construction"] == surface]
            train_surface = [int(row["hidden_index"]) for row in legacy_all if row["family"] == family and row["construction"] == surface]
            matched[test_local] = np.asarray(states[train_surface, 37], np.float32).mean(axis=0)
        predictions["family_surface_mean"] = matched
        metrics[family] = {name: vector_metric(pred, truth) for name, pred in predictions.items()}
        for surface in sorted({row["construction"] for row in rows}):
            local = [i for i, row in enumerate(rows) if row["construction"] == surface]
            key = f"{family}:{surface}"
            surface_metrics[key] = {name: vector_metric(pred[local], truth[local]) for name, pred in predictions.items()}
            best_control = min(surface_metrics[key][name]["nrmse"] for name in control_names)
            gates[key] = surface_metrics[key]["shared"]["nrmse"] <= best_control - CONTROL_MARGIN
        prediction_ids.extend(row["case_id"] for row in rows)
        shared_predictions.append(currents[family]["shared"].astype(np.float16))
        family_predictions.append(currents[family]["family"].astype(np.float16))
    np.savez_compressed(
        archive,
        case_ids=np.asarray(prediction_ids),
        shared_q37=np.concatenate(shared_predictions).astype(np.float16),
        family_q37=np.concatenate(family_predictions).astype(np.float16),
    )
    return {
        "trajectory": trajectory, "metrics": metrics, "surface_metrics": surface_metrics,
        "surface_gates": gates, "strong_candidate": all(gates.values()),
        "archive": str(archive.relative_to(ROOT)).replace("\\", "/"),
    }


def c531() -> None:
    out = begin("C531", {
        "status": "c523_strong_baseline_tournament_frozen",
        "train": "legacy flagship vocabulary, all samples and 37 adjacent transitions",
        "test": "fresh flagship vocabulary and paraphrases",
        "primary": "shared six-role all-coordinate autonomous rollout from q0",
        "controls": [
            "persistence", "global q37 mean", "family q37 mean", "family-surface q37 mean",
            "matched q0 sample shuffle", "role reversal", "coordinate roll by 137",
        ],
        "gate": "primary NRMSE beats every control by >=0.02 in every family x surface stratum",
    }, {"parent": final("C530")["all_checks_passed"]})
    mean_path, _exact_path, _full_path = capture_paths()
    states = np.load(mean_path, mmap_mode="r")
    result = run_flagship_rollout(states, capture_index(), out / "analysis/q37_predictions.npz", "C531")
    save(out / "analysis/trajectory_metrics.json", result.pop("trajectory"))
    close_mmap(states)
    close("C531", {
        "status": "strong_baseline_tournament_closed", **result,
        "strict_interpretation": "A pass would establish sample-paired trajectory value beyond strong means and corruptions; it would still not identify a unique internal circuit.",
    }, {"finite": finite(result), "strata": len(result["surface_gates"]) == 9, "archive": (out / "analysis/q37_predictions.npz").exists()}, "C532_centered_residual")


def c532() -> None:
    out = begin("C532", {
        "status": "centered_sample_residual_and_pair_difference_frozen",
        "object": "q37 sample residual after subtracting legacy family-surface mean, plus deterministic fresh-unit pair differences",
        "baselines": ["zero centered residual", "zero pair difference"],
        "gate": "shared rollout improves both centered and pair NRMSE by >=0.02 in every family x surface stratum",
    }, {"parent": final("C531")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    by_case = {row["case_id"]: row for row in index}
    archive = np.load(OUTS["C531"] / "analysis/q37_predictions.npz", allow_pickle=False)
    case_ids = [str(value) for value in archive["case_ids"]]
    predictions = np.asarray(archive["shared_q37"], np.float32)
    prediction_by_id = {case_id: predictions[i] for i, case_id in enumerate(case_ids)}
    centered, pairs, gates = {}, {}, {}
    for family in FLAGSHIP:
        for surface in sorted({row["construction"] for row in index if row["cohort"] == "fresh_program" and row["family"] == family}):
            key = f"{family}:{surface}"
            test_rows = sorted(
                [row for row in index if row["cohort"] == "fresh_program" and row["family"] == family and row["construction"] == surface],
                key=lambda row: (str(row["cell"]), int(row["unit"])),
            )
            train_ids = [int(row["hidden_index"]) for row in index if row["cohort"] == "legacy_program" and row["family"] == family and row["construction"] == surface]
            mean = np.asarray(states[train_ids, 37], np.float32).mean(axis=0)
            truth = np.asarray(states[[int(row["hidden_index"]) for row in test_rows], 37], np.float32)
            pred = np.stack([prediction_by_id[row["case_id"]] for row in test_rows])
            centered_truth = truth - mean[None]
            centered_pred = pred - mean[None]
            centered[key] = {
                "zero": vector_metric(np.zeros_like(centered_truth), centered_truth),
                "shared": vector_metric(centered_pred, centered_truth),
            }
            groups: dict[str, list[int]] = defaultdict(list)
            for local, row in enumerate(test_rows):
                groups[str(row["cell"])].append(local)
            truth_diffs, pred_diffs = [], []
            for values in groups.values():
                values = sorted(values, key=lambda i: int(test_rows[i]["unit"]))
                for a, b in zip(values[::2], values[1::2]):
                    truth_diffs.append(truth[a] - truth[b])
                    pred_diffs.append(pred[a] - pred[b])
            truth_diff = np.stack(truth_diffs)
            pred_diff = np.stack(pred_diffs)
            pairs[key] = {
                "pairs": len(truth_diffs),
                "zero": vector_metric(np.zeros_like(truth_diff), truth_diff),
                "shared": vector_metric(pred_diff, truth_diff),
            }
            centered_gain = centered[key]["zero"]["nrmse"] - centered[key]["shared"]["nrmse"]
            pair_gain = pairs[key]["zero"]["nrmse"] - pairs[key]["shared"]["nrmse"]
            gates[key] = centered_gain >= CONTROL_MARGIN and pair_gain >= CONTROL_MARGIN
    close_mmap(states)
    close("C532", {
        "status": "centered_residual_closed", "centered_metrics": centered, "pair_metrics": pairs,
        "stratum_gates": gates, "sample_specific_candidate": all(gates.values()),
        "strict_interpretation": "Absolute-state accuracy is not promoted unless deviations between individual samples are also predicted.",
    }, {"finite": finite(centered) and finite(pairs), "strata": len(gates) == 9, "pairs": all(value["pairs"] > 0 for value in pairs.values())}, "C533_response_neighborhood")


def c533() -> None:
    out = begin("C533", {
        "status": "full_coordinate_response_neighborhood_frozen",
        "algorithm": "within each domain/surface/cell, choose discovery sample with smallest full q0 six-role squared distance; transport its q37 trajectory by the full q0 difference",
        "candidate": "H37(neighbor) + H0(test) - H0(neighbor)",
        "controls": ["matched q37 mean", "nearest q37 without q0 transport", "persistence", "deterministic far neighbor"],
        "gate": "transport beats every control by >=0.02 for >=80% of domains and has positive median margin in every cohort",
    }, {"parent": final("C532")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    metrics, margins = {}, {}
    for domain in sorted({row["domain_id"] for row in index}):
        train = [row for row in index if row["domain_id"] == domain and row["partition"] == "discovery"]
        test = [row for row in index if row["domain_id"] == domain and row["partition"] == "lockbox"]
        truth_all, predictions = [], {name: [] for name in ("transport", "matched_mean", "nearest", "persistence", "far")}
        for row in test:
            matched = [item for item in train if item["construction"] == row["construction"] and str(item["cell"]) == str(row["cell"])]
            if not matched:
                matched = [item for item in train if item["construction"] == row["construction"]]
            test_i = int(row["hidden_index"])
            q0 = np.asarray(states[test_i, 0], np.float32)
            distances = []
            for item in matched:
                candidate_q0 = np.asarray(states[int(item["hidden_index"]), 0], np.float32)
                distances.append(float(np.mean((q0 - candidate_q0) ** 2)))
            nearest = matched[int(np.argmin(distances))]
            far = matched[int(np.argmax(distances))]
            nearest_i, far_i = int(nearest["hidden_index"]), int(far["hidden_index"])
            neighbor_q0 = np.asarray(states[nearest_i, 0], np.float32)
            neighbor_q37 = np.asarray(states[nearest_i, 37], np.float32)
            truth = np.asarray(states[test_i, 37], np.float32)
            matched_q37 = np.asarray(states[[int(item["hidden_index"]) for item in matched], 37], np.float32)
            truth_all.append(truth)
            predictions["transport"].append(neighbor_q37 + q0 - neighbor_q0)
            predictions["matched_mean"].append(matched_q37.mean(axis=0))
            predictions["nearest"].append(neighbor_q37)
            predictions["persistence"].append(q0)
            predictions["far"].append(np.asarray(states[far_i, 37], np.float32))
        truth_array = np.stack(truth_all)
        metrics[domain] = {name: vector_metric(np.stack(values), truth_array) for name, values in predictions.items()}
        best_control = min(metrics[domain][name]["nrmse"] for name in ("matched_mean", "nearest", "persistence", "far"))
        margins[domain] = best_control - metrics[domain]["transport"]["nrmse"]
        print(f"[C533 neighborhood] {domain}", flush=True)
    cohort_median = {
        cohort: float(np.median([value for domain, value in margins.items() if domain.startswith(cohort + ":")]))
        for cohort in COHORTS
    }
    pass_rate = float(np.mean([value >= CONTROL_MARGIN for value in margins.values()]))
    candidate = pass_rate >= 0.80 and all(value > 0 for value in cohort_median.values())
    close_mmap(states)
    close("C533", {
        "status": "response_neighborhood_closed", "domain_metrics": metrics, "domain_margins": margins,
        "pass_rate_at_0_02": pass_rate, "cohort_median_margin": cohort_median,
        "response_neighborhood_candidate": candidate,
        "strict_interpretation": "Nearest-neighbor transport is a full-coordinate empirical dependency rule, not a unique causal circuit or proof of a neural metric manifold.",
    }, {"domains": len(metrics) == 20, "finite": finite(metrics), "all_tested": all(value["transport"]["n"] > 0 for value in metrics.values())}, "C534_lofo")


def c534() -> None:
    out = begin("C534", {
        "status": "leave_one_family_autonomous_dynamics_frozen",
        "holdouts": list(LOFO_DOMAINS),
        "train": "all discovery samples outside the held domain; no held-domain examples",
        "test": "held-domain lockbox samples",
        "primary": "37-step shared six-role all-coordinate rollout",
        "controls": ["persistence", "global q37 mean", "surface-matched q37 mean"],
        "gate": "primary beats every control by >=0.02 for every held domain x surface",
    }, {"parent": final("C533")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    domain_metrics, surface_metrics, gates = {}, {}, {}
    for hold_i, domain in enumerate(LOFO_DOMAINS):
        train_rows = [row for row in index if row["partition"] == "discovery" and row["domain_id"] != domain]
        test_rows = sorted([row for row in index if row["partition"] == "lockbox" and row["domain_id"] == domain], key=lambda row: int(row["hidden_index"]))
        train_ids = [int(row["hidden_index"]) for row in train_rows]
        test_ids = [int(row["hidden_index"]) for row in test_rows]
        current = np.asarray(states[test_ids, 0], np.float32)
        origin = current.copy()
        for q in range(37):
            x = np.asarray(states[train_ids, q], np.float32)
            y = np.asarray(states[train_ids, q + 1], np.float32) - x
            beta, intercept = fit_bundle(x, y)
            current += predict_bundle(current, beta, intercept)
        truth = np.asarray(states[test_ids, 37], np.float32)
        global_mean = np.asarray(states[train_ids, 37], np.float32).mean(axis=0)
        matched = np.empty_like(truth)
        for surface in sorted({row["construction"] for row in test_rows}):
            local = [i for i, row in enumerate(test_rows) if row["construction"] == surface]
            source = [int(row["hidden_index"]) for row in train_rows if row["construction"] == surface]
            matched[local] = np.asarray(states[source, 37], np.float32).mean(axis=0)
        predictions = {
            "rollout": current, "persistence": origin,
            "global_mean": np.broadcast_to(global_mean, truth.shape), "surface_mean": matched,
        }
        domain_metrics[domain] = {name: vector_metric(pred, truth) for name, pred in predictions.items()}
        for surface in sorted({row["construction"] for row in test_rows}):
            local = [i for i, row in enumerate(test_rows) if row["construction"] == surface]
            key = f"{domain}:{surface}"
            surface_metrics[key] = {name: vector_metric(pred[local], truth[local]) for name, pred in predictions.items()}
            best_control = min(surface_metrics[key][name]["nrmse"] for name in ("persistence", "global_mean", "surface_mean"))
            gates[key] = surface_metrics[key]["rollout"]["nrmse"] <= best_control - CONTROL_MARGIN
        print(f"[C534 LOFO] {hold_i + 1}/{len(LOFO_DOMAINS)} {domain}", flush=True)
    close_mmap(states)
    close("C534", {
        "status": "leave_one_family_closed", "domain_metrics": domain_metrics,
        "surface_metrics": surface_metrics, "stratum_gates": gates,
        "cross_family_candidate": all(gates.values()),
        "strict_interpretation": "A success would be cross-domain shared layer dynamics, not evidence that language-family semantics share a fixed physical coordinate dictionary.",
    }, {"domains": len(domain_metrics) == len(LOFO_DOMAINS), "strata": len(gates) == 18, "finite": finite(domain_metrics)}, "C535_exact_token")


def c535() -> None:
    out = begin("C535", {
        "status": "exact_token_autonomous_dynamics_frozen",
        "state": "last physical token of each registered role span at every checkpoint",
        "comparison": "same frozen flagship rollout and strong controls as C531",
        "writable_boundary": "exact token states are physically writable; role-span means are not",
        "gate": "exact-token rollout passes every family x surface strong-control gate and C532 sample-specific gate",
    }, {"parent": final("C534")["all_checks_passed"]})
    states = np.load(capture_paths()[1], mmap_mode="r")
    index = capture_index()
    archive_path = out / "analysis/q37_exact_predictions.npz"
    result = run_flagship_rollout(states, index, archive_path, "C535 exact")
    save(out / "analysis/trajectory_metrics.json", result.pop("trajectory"))
    archive = np.load(archive_path, allow_pickle=False)
    case_ids = [str(value) for value in archive["case_ids"]]
    predictions = np.asarray(archive["shared_q37"], np.float32)
    prediction_by_id = {case_id: predictions[i] for i, case_id in enumerate(case_ids)}
    exact_centered, exact_pairs, exact_sample_gates = {}, {}, {}
    for family in FLAGSHIP:
        for surface in sorted({row["construction"] for row in index if row["cohort"] == "fresh_program" and row["family"] == family}):
            key = f"{family}:{surface}"
            test_rows = sorted(
                [row for row in index if row["cohort"] == "fresh_program" and row["family"] == family and row["construction"] == surface],
                key=lambda row: (str(row["cell"]), int(row["unit"])),
            )
            train_ids = [int(row["hidden_index"]) for row in index if row["cohort"] == "legacy_program" and row["family"] == family and row["construction"] == surface]
            mean = np.asarray(states[train_ids, 37], np.float32).mean(axis=0)
            truth = np.asarray(states[[int(row["hidden_index"]) for row in test_rows], 37], np.float32)
            pred = np.stack([prediction_by_id[row["case_id"]] for row in test_rows])
            truth_centered, pred_centered = truth - mean[None], pred - mean[None]
            exact_centered[key] = {
                "zero": vector_metric(np.zeros_like(truth_centered), truth_centered),
                "shared": vector_metric(pred_centered, truth_centered),
            }
            groups: dict[str, list[int]] = defaultdict(list)
            for local, row in enumerate(test_rows):
                groups[str(row["cell"])].append(local)
            truth_diffs, pred_diffs = [], []
            for values in groups.values():
                values = sorted(values, key=lambda i: int(test_rows[i]["unit"]))
                for a, b in zip(values[::2], values[1::2]):
                    truth_diffs.append(truth[a] - truth[b])
                    pred_diffs.append(pred[a] - pred[b])
            truth_diff, pred_diff = np.stack(truth_diffs), np.stack(pred_diffs)
            exact_pairs[key] = {
                "pairs": len(truth_diffs), "zero": vector_metric(np.zeros_like(truth_diff), truth_diff),
                "shared": vector_metric(pred_diff, truth_diff),
            }
            centered_gain = exact_centered[key]["zero"]["nrmse"] - exact_centered[key]["shared"]["nrmse"]
            pair_gain = exact_pairs[key]["zero"]["nrmse"] - exact_pairs[key]["shared"]["nrmse"]
            exact_sample_gates[key] = centered_gain >= CONTROL_MARGIN and pair_gain >= CONTROL_MARGIN
    close_mmap(states)
    mean_candidate = final("C531")["headline"]["strong_candidate"]
    mean_sample_candidate = final("C532")["headline"]["sample_specific_candidate"]
    exact_sample_candidate = all(exact_sample_gates.values())
    exact_candidate = bool(result["strong_candidate"] and exact_sample_candidate)
    close("C535", {
        "status": "exact_token_rollout_closed", **result,
        "exact_centered_metrics": exact_centered, "exact_pair_metrics": exact_pairs,
        "exact_sample_gates": exact_sample_gates, "exact_sample_specific_candidate": exact_sample_candidate,
        "mean_role_strong_candidate": mean_candidate, "mean_role_sample_specific_candidate": mean_sample_candidate,
        "exact_writable_state_candidate": exact_candidate,
        "strict_interpretation": "Physical writability does not by itself imply semantic specificity; the exact-token trajectory must also retain sample differences.",
    }, {
        "finite": finite(result) and finite(exact_centered) and finite(exact_pairs),
        "strata": len(result["surface_gates"]) == 9 and len(exact_sample_gates) == 9,
        "archive": archive_path.exists(),
    }, "C536_operation_ecology")


def operation_pairs(index: list[dict], domain: str, partition: str, bit_i: int) -> list[tuple[int, int]]:
    lookup: dict[tuple[str, int, tuple[int, ...]], int] = {}
    for row in index:
        if row["domain_id"] == domain and row["partition"] == partition:
            lookup[(row["construction"], int(row["unit"]), tuple(int(value) for value in row["bits"]))] = int(row["hidden_index"])
    pairs = []
    for (surface, unit, bits), left in lookup.items():
        if bit_i >= len(bits) or bits[bit_i] != 0:
            continue
        other = list(bits); other[bit_i] = 1
        right = lookup.get((surface, unit, tuple(other)))
        if right is not None:
            pairs.append((left, right))
    return pairs


def c536() -> None:
    out = begin("C536", {
        "status": "broad_language_operation_response_ecology_frozen",
        "object": "full six-role x 2560-coordinate response to one registered binary material factor at q24 and q37",
        "train": "discovery-unit paired differences",
        "test": "lockbox-unit paired differences",
        "models": ["same-domain same-bit prototype", "zero response", "wrong-bit prototype"],
        "gate": "same-bit prototype beats zero and wrong-bit by >=0.02",
        "boundary": "bit positions are material factors and are not assumed to mean the same operation across domains",
    }, {"parent": final("C535")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    metrics, gates, prototype_archive = {}, {}, {}
    domains = sorted({row["domain_id"] for row in index})
    for domain in domains:
        available = []
        for bit_i in range(4):
            train_pairs = operation_pairs(index, domain, "discovery", bit_i)
            test_pairs = operation_pairs(index, domain, "lockbox", bit_i)
            if train_pairs and test_pairs:
                available.append((bit_i, train_pairs, test_pairs))
        prototypes = {}
        for bit_i, train_pairs, _test_pairs in available:
            prototypes[bit_i] = {}
            for q in (24, 37):
                diffs = np.stack([
                    np.asarray(states[right, q], np.float32) - np.asarray(states[left, q], np.float32)
                    for left, right in train_pairs
                ])
                prototypes[bit_i][q] = diffs.mean(axis=0)
        for bit_i, _train_pairs, test_pairs in available:
            wrong_i = next((candidate for candidate in prototypes if candidate != bit_i), None)
            for q in (24, 37):
                truth = np.stack([
                    np.asarray(states[right, q], np.float32) - np.asarray(states[left, q], np.float32)
                    for left, right in test_pairs
                ])
                same = np.broadcast_to(prototypes[bit_i][q], truth.shape)
                wrong = np.zeros_like(truth) if wrong_i is None else np.broadcast_to(prototypes[wrong_i][q], truth.shape)
                key = f"{domain}:bit{bit_i}:q{q}"
                metrics[key] = {
                    "pairs": len(test_pairs), "same": vector_metric(same, truth),
                    "zero": vector_metric(np.zeros_like(truth), truth), "wrong": vector_metric(wrong, truth),
                }
                best_control = min(metrics[key]["zero"]["nrmse"], metrics[key]["wrong"]["nrmse"])
                gates[key] = metrics[key]["same"]["nrmse"] <= best_control - CONTROL_MARGIN
                if q == 24:
                    prototype_archive[key] = prototypes[bit_i][q].astype(np.float16)
        print(f"[C536 ecology] {domain}", flush=True)
    np.savez_compressed(out / "analysis/q24_operation_prototypes.npz", **prototype_archive)
    pass_rate = float(np.mean(list(gates.values()))) if gates else 0.0
    domain_pass_rate = {
        domain: float(np.mean([value for key, value in gates.items() if key.startswith(domain + ":")]))
        for domain in domains
    }
    broad_candidate = pass_rate >= 0.70 and sum(value >= 0.5 for value in domain_pass_rate.values()) >= 14
    close_mmap(states)
    close("C536", {
        "status": "operation_response_ecology_closed", "metrics": metrics, "gates": gates,
        "gate_pass_rate": pass_rate, "domain_pass_rate": domain_pass_rate,
        "broad_operation_response_candidate": broad_candidate,
        "strict_interpretation": "A prototype pass is a reusable material-factor response in a controlled panel, not yet a universal linguistic operator or composition law.",
    }, {"domains": len(domain_pass_rate) == 20, "comparisons": len(metrics) >= 80, "finite": finite(metrics), "archive": (out / "analysis/q24_operation_prototypes.npz").exists()}, "C537_eligibility")


def c537() -> None:
    out = begin("C537", {
        "status": "predictive_exact_token_causal_eligibility_frozen",
        "requirements": [
            "C531 absolute-state strong controls", "C532 centered residual and pair differences",
            "C535 exact-token strong controls", "C536 broad response candidate",
        ],
        "rule": "causal branch is eligible only if all four independent requirements pass before any patch",
    }, {"parent": final("C536")["all_checks_passed"]})
    requirements = {
        "absolute_strong": bool(final("C531")["headline"]["strong_candidate"]),
        "sample_specific": bool(final("C532")["headline"]["sample_specific_candidate"]),
        "exact_token": bool(final("C535")["headline"]["exact_writable_state_candidate"]),
        "broad_response": bool(final("C536")["headline"]["broad_operation_response_candidate"]),
    }
    authorized = all(requirements.values())
    close("C537", {
        "status": "causal_eligibility_closed", "requirements": requirements,
        "causal_authorized": authorized,
        "causal_result": "NA_not_yet_run" if authorized else "NA_predictive_qualification_failed",
        "strict_interpretation": "A failed eligibility gate is not a failed causal experiment.",
    }, {"complete": len(requirements) == 4}, "C538_causal_or_na")


def c538() -> None:
    out = begin("C538", {
        "status": "qualified_exact_token_causal_branch_frozen",
        "planned_if_eligible": [
            "q24 exact query-token counterfactual replacement", "correct paired donor", "wrong-factor donor",
            "wrong-role donor", "wrong-checkpoint control", "coordinate-roll control", "A/B behavior and free-generation side effects",
        ],
        "rule": "if C537 is false, write an explicit NA ledger and do not load the model",
    }, {"parent": final("C537")["all_checks_passed"]})
    authorized = bool(final("C537")["headline"]["causal_authorized"])
    if not authorized:
        headline = {
            "status": "causal_not_run", "authorized": False, "ran": False,
            "result": "NA_predictive_qualification_failed", "model_loaded": False,
            "strict_interpretation": "No intervention result exists; observational routes remain valid at their registered evidence levels.",
        }
    else:
        # The exact counterfactual executor requires a separately frozen donor-pair
        # compiler. Eligibility alone cannot retroactively define that compiler.
        headline = {
            "status": "causal_not_run", "authorized": True, "ran": False,
            "result": "NA_exact_counterfactual_donor_compiler_not_frozen",
            "model_loaded": False,
            "strict_interpretation": "Passing predictive gates would authorize a new preregistered executor phase, not an improvised post-reveal patch.",
        }
    close("C538", headline, {"no_unregistered_patch": not headline["ran"], "model_not_loaded": not headline["model_loaded"]}, "C539_visual_synthesis")


def register_visual() -> None:
    if REGISTRY.exists():
        registry = load(REGISTRY)
        item = {
            "id": "c539_sample_specific_dynamics_atlas", "title": "C539 Sample-Specific Dynamics Atlas",
            "phase": 2073, "campaign": "C527-C540", "path": "vis_data/research_kernel/c539_sample_specific_dynamics_atlas.json",
            "kind": "sample_specific_exact_token_full_coordinate_atlas", "coordinates": DIM,
        }
        datasets = registry.setdefault("datasets", [])
        datasets[:] = [row for row in datasets if row.get("id") != item["id"]]
        datasets.append(item)
        save(REGISTRY, registry)
    if CATALOG.exists():
        catalog = load(CATALOG)
        item = {
            "id": "c539_sample_specific_dynamics_atlas", "title": "C539 Sample-Specific Dynamics Atlas",
            "url": "/vis_data/research_kernel/c539_sample_specific_dynamics_atlas.json",
            "phase": 2073, "full_coordinate": True,
        }
        datasets = catalog.setdefault("field_datasets", [])
        datasets[:] = [row for row in datasets if row.get("id") != item["id"]]
        datasets.append(item)
        save(CATALOG, catalog)


def c539() -> None:
    out = begin("C539", {
        "status": "full_coordinate_visual_and_synthesis_frozen",
        "visual": "one lockbox sample per domain, all six roles, mean and exact-token q0/q24/q37; one complete full-token trajectory panel",
        "coordinate_policy": "every displayed vector contains all 2560 physical activation coordinates",
        "client_policy": "register the generic research-kernel dataset; no bespoke Markdown documentation",
    }, {"parent": final("C538")["all_checks_passed"]})
    mean_path, exact_path, full_path = capture_paths()
    mean_states = np.load(mean_path, mmap_mode="r")
    exact_states = np.load(exact_path, mmap_mode="r")
    full_states = np.load(full_path, mmap_mode="r")
    index = capture_index()
    rows = []
    representatives = {}
    for domain in sorted({row["domain_id"] for row in index}):
        candidates = sorted([row for row in index if row["domain_id"] == domain and row["partition"] == "lockbox"], key=lambda row: row["case_id"])
        representative = candidates[0]
        representatives[domain] = representative["case_id"]
        i = int(representative["hidden_index"])
        for role_i, role in enumerate(ROLES):
            rows.append({
                "domain_id": domain, "case_id": representative["case_id"], "role": role,
                "embedding_mean_q0": np.asarray(mean_states[i, 0, role_i], np.float32).tolist(),
                "embedding_exact_q0": np.asarray(exact_states[i, 0, role_i], np.float32).tolist(),
                "state_mean_q24": np.asarray(mean_states[i, 24, role_i], np.float32).tolist(),
                "state_exact_q24": np.asarray(exact_states[i, 24, role_i], np.float32).tolist(),
                "state_mean_q37": np.asarray(mean_states[i, 37, role_i], np.float32).tolist(),
                "state_exact_q37": np.asarray(exact_states[i, 37, role_i], np.float32).tolist(),
            })
    full_map = load(OUTS["C530"] / "raw/full_field_row_map.json")
    first_source = int(full_map["source_indices"][0])
    first_meta = next(row for row in index if int(row["hidden_index"]) == first_source)
    length = int(first_meta["length"])
    full_i = int(first_meta["full_index"])
    full_panel = {
        "case_id": first_meta["case_id"], "domain_id": first_meta["domain_id"], "length": length,
        "token_positions": list(range(length)),
        "checkpoint_fields": {
            f"q{q}": np.asarray(full_states[full_i, q, :length], np.float32).tolist()
            for q in (0, 8, 16, 24, 32, 37)
        },
    }
    visual = {
        "schema": "ai2050.sample_specific_dynamics_atlas.v1", "phase": 2073, "campaign": "C527-C540",
        "coordinate_count": DIM, "checkpoint_count": CHECKPOINTS, "roles": list(ROLES),
        "representatives": representatives, "rows": rows, "full_token_panel": full_panel,
        "strong_rollout": final("C531")["headline"],
        "sample_specific": final("C532")["headline"],
        "response_neighborhood": final("C533")["headline"],
        "leave_one_family": final("C534")["headline"],
        "exact_token": final("C535")["headline"],
        "operation_ecology": {
            "gate_pass_rate": final("C536")["headline"]["gate_pass_rate"],
            "domain_pass_rate": final("C536")["headline"]["domain_pass_rate"],
            "broad_candidate": final("C536")["headline"]["broad_operation_response_candidate"],
        },
        "causal": final("C538")["headline"],
    }
    save(VISUAL, visual)
    register_visual()
    close_mmap(mean_states); close_mmap(exact_states); close_mmap(full_states)
    gates = {
        "absolute_strong": final("C531")["headline"]["strong_candidate"],
        "sample_specific": final("C532")["headline"]["sample_specific_candidate"],
        "response_neighborhood": final("C533")["headline"]["response_neighborhood_candidate"],
        "cross_family": final("C534")["headline"]["cross_family_candidate"],
        "exact_token": final("C535")["headline"]["exact_writable_state_candidate"],
        "broad_operation_response": final("C536")["headline"]["broad_operation_response_candidate"],
        "causal_ran": final("C538")["headline"]["ran"],
    }
    close("C539", {
        "status": "visual_and_synthesis_closed", "gates": gates,
        "visual_path": str(VISUAL.relative_to(ROOT)).replace("\\", "/"),
        "visual_rows": len(rows), "visual_role_coordinate_values": len(rows) * 6 * DIM,
        "full_token_case": first_meta["case_id"], "full_token_coordinate_values": 6 * length * DIM,
        "strict_conclusion": "Generic layer-depth prediction, sample-specific residual prediction, exact-token writability, broad operation reuse, and causality remain separate evidence accounts.",
        "new_math_gate": False,
    }, {
        "visual": VISUAL.exists(), "rows": len(rows) == 120,
        "coordinates": all(len(row["state_exact_q37"]) == DIM for row in rows),
        "full_token": all(len(token) == DIM for field in full_panel["checkpoint_fields"].values() for token in field),
        "finite": finite(gates),
    }, "C540_cleanup")


def c540() -> None:
    out = begin("C540", {
        "status": "raw_field_cleanup_and_next_stage_frozen",
        "cleanup": "hash and delete the three C530 raw state fields after C539 registered full-coordinate visual preservation",
        "retained": ["materials", "compiled prompts", "hidden index", "all metrics", "prediction archives", "visual full-coordinate atlas", "hash ledger"],
        "next_rule": "continue automatically only through a separately preregistered route matching the evidence that survives strong controls",
    }, {"parent": final("C539")["all_checks_passed"]})
    cleanup = []
    for path in capture_paths():
        if path.exists():
            cleanup.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    save(out / "audit/raw_field_cleanup_ledger.json", {"files": cleanup, "total_bytes": sum(row["bytes"] for row in cleanup)})
    for row in cleanup:
        (ROOT / row["path"]).unlink()
    gates = final("C539")["headline"]["gates"]
    if gates["exact_token"] and not gates["causal_ran"]:
        next_route = "C542_freeze_exact_counterfactual_donor_compiler"
        same_goal = True
    elif gates["sample_specific"] or gates["response_neighborhood"] or gates["broad_operation_response"]:
        next_route = "C542_state_conditioned_response_equivalence_and_composition_campaign"
        same_goal = True
    else:
        next_route = "new_observational_object_required_before_more_rollout_fitting"
        same_goal = False
    close("C540", {
        "status": "campaign_cleanup_closed", "gates": gates,
        "cleanup_files": len(cleanup), "cleanup_bytes": sum(row["bytes"] for row in cleanup),
        "raw_fields_absent": all(not path.exists() for path in capture_paths()),
        "next_stage_same_goal": same_goal, "next_route": next_route,
        "strict_interpretation": "A route failure does not erase observations from other routes; it only limits the next mechanism claim.",
    }, {
        "cleanup": all(not path.exists() for path in capture_paths()), "ledger": len(cleanup) == 3,
        "visual_retained": VISUAL.exists(),
    }, "C541_independent_audit")


FUNCTIONS = {
    "C527": c527, "C528": c528, "C529": c529, "C530": c530,
    "C531": c531, "C532": c532, "C533": c533, "C534": c534,
    "C535": c535, "C536": c536, "C537": c537, "C538": c538,
    "C539": c539, "C540": c540,
}


def self_test() -> None:
    rows, compiled = combined_material()
    assert len(rows) == 8160 and len(compiled) == 8160
    assert len({row["case_id"] for row in rows}) == 8160
    assert len({row["domain_id"] for row in rows}) == 20
    assert len(select_full_ids(rows)) == 120
    assert all(set(row["role_positions"]) == set(ROLES) for row in compiled)
    print(json.dumps({
        "self_test": "passed", "rows": len(rows), "domains": len({row["domain_id"] for row in rows}),
        "full_rows": len(select_full_ids(rows)), "max_width": max(len(row["prompt_ids"]) for row in compiled),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--start", default="C527")
    parser.add_argument("--stop", default="C540")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    names = list(FUNCTIONS)
    start, stop = names.index(args.start), names.index(args.stop)
    for name in names[start:stop + 1]:
        FUNCTIONS[name]()


if __name__ == "__main__":
    main()
