#!/usr/bin/env python3
"""Prospective first-lexical-divergence residual intervention.

Phase1140 showed that full candidate-path patching rescued exactly the curves
whose two answers share their first token, but that observation was post hoc.
This phase freezes fresh discovery and confirmation cohorts and compares:

1. answer_boundary: the state predicting candidate token zero;
2. first_lexical_divergence: the state predicting the first unequal token;
3. candidate_prediction_span: every state used to score the candidate.

The first-divergence position is a lexical discrimination coordinate, not a
pre-labelled semantic boundary.  Same-answer and cross-item donor controls are
included before any component search is authorized.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import quantization_audit, release_fp16  # noqa: E402
import phase1135_temporal_binding_intervention as source  # noqa: E402
import phase1138_temporal_residual_onset as phase1138  # noqa: E402
import phase1140_temporal_position_scope_sufficiency as prior  # noqa: E402


PHASE = 1141
MODELS = ("qwen3_4b", "qwen3_14b")
SPLITS = ("discovery", "confirmation")
SCOPES = (
    "answer_boundary",
    "first_lexical_divergence",
    "candidate_prediction_span",
)
ALPHAS = (0.0, 1.0)
REQUESTED_FRACTION = 0.7
COHORT_SIZE = 30
MAIN_CURVES_PER_ITEM = 4
TEMPORAL_CURVES_PER_ITEM = 2
CROSS_ITEM_CURVES_PER_ITEM = 4
EXPECTED_RECORDS = COHORT_SIZE * (
    MAIN_CURVES_PER_ITEM * len(SCOPES) * len(ALPHAS)
    + (TEMPORAL_CURVES_PER_ITEM + CROSS_ITEM_CURVES_PER_ITEM) * len(ALPHAS)
)
MAIN_STATES = {"original_pre", "original_post", "swapped_pre", "swapped_post"}
OTHER_PROPERTY_QUOTAS = {"P286": 4, "P6": 3, "P488": 2, "P169": 1}
STRATUM_QUOTAS = {
    "shared_prefix_p54": 10,
    "immediate_p54": 10,
    "immediate_other": 10,
}
OUT_ROOT = ROOT / "tests/glm5/result/phase1141_first_lexical_divergence_boundary"
SOURCE_ITEMS = source.SOURCE
Q4_DECISIONS = prior.Q4_DECISIONS
Q14_DECISIONS = prior.Q14_DECISIONS
TOKEN_CASES = (
    ROOT
    / "tests/glm5/result/phase1137_qwen14b_temporal_binding_endpoint"
    / "protocol/cases.qwen3_14b.jsonl"
)
SOURCE1138 = prior.SOURCE1138
SOURCE1139 = prior.SOURCE1139
SOURCE1140 = prior.OUT_ROOT
EPSILON = 1e-8

THRESHOLDS = {
    "finite_fraction": 0.99,
    "identity_max_abs_margin_drift": 0.005,
    "baseline_valid_fraction": 0.99,
    "main_endpoint_flip_fraction": 0.95,
    "stratum_endpoint_flip_fraction": 0.95,
    "panel_endpoint_flip_fraction": 0.95,
    "main_positive_change_fraction": 0.99,
    "same_answer_control_flip_fraction": 0.10,
    "cross_item_control_flip_fraction": 0.10,
    "main_to_each_control_ratio": 2.0,
    "shared_minus_boundary_min": 0.50,
    "shared_span_noninferiority_margin": 0.05,
    "lcp0_scope_equivalence_max_abs_margin": 0.005,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def median(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def stable_order(label: str, values: Iterable[str]) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"phase1141|{label}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def behavior_all_four_by_split(path: Path) -> dict[str, set[str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        if str(row["state"]) in MAIN_STATES:
            grouped[(str(row["split"]), str(row["item_id"]))].append(row)
    result = {split: set() for split in SPLITS}
    for (split, item_id), rows in grouped.items():
        if (
            split in result
            and len(rows) == 4
            and all(bool(row["finite"]) and row["correct"] is True for row in rows)
        ):
            result[split].add(item_id)
    return result


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        length += 1
    return length


def token_metadata() -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, list[int]]] = defaultdict(dict)
    properties: dict[str, str] = {}
    for row in read_jsonl(TOKEN_CASES):
        item_id = str(row["item_id"])
        key = str(row["candidate_key"])
        ids = [int(value) for value in row["continuation_ids"]]
        prior_ids = candidates[item_id].get(key)
        if prior_ids is not None and prior_ids != ids:
            raise RuntimeError(f"candidate tokenization drift for {item_id} {key}")
        candidates[item_id][key] = ids
        properties[item_id] = str(row["property_id"])
    result = {}
    for item_id, rows in candidates.items():
        if set(rows) != {"old", "new"}:
            raise RuntimeError(f"incomplete candidate tokens for {item_id}")
        lcp = common_prefix_length(rows["old"], rows["new"])
        shortest = min(len(rows["old"]), len(rows["new"]))
        result[item_id] = {
            "old_ids": rows["old"],
            "new_ids": rows["new"],
            "old_count": len(rows["old"]),
            "new_count": len(rows["new"]),
            "common_prefix_length": lcp,
            "prefix_containment": lcp == shortest,
            "strict_shared_prefix": 0 < lcp < shortest,
            "first_token_informative": lcp == 0,
            "property_id": properties[item_id],
        }
    return result


def excluded_hidden_ids() -> set[str]:
    cohort1138 = read_json(SOURCE1138 / "protocol/behavior_conditioned_cohorts.json")
    excluded = set(cohort1138["discovery"]["shared_item_ids"])
    excluded.update(cohort1138["confirmation"]["shared_item_ids"])
    prereg1140 = read_json(SOURCE1140 / "protocol/preregistration.json")
    excluded.update(prereg1140["material"]["cohorts"]["discovery"])
    excluded.update(prereg1140["material"]["cohorts"]["confirmation"])
    excluded.update(prereg1140["material"]["reserve_item_ids"])
    return excluded


def classify_stratum(meta: dict[str, Any]) -> str | None:
    if meta["property_id"] == "P54" and meta["strict_shared_prefix"]:
        return "shared_prefix_p54"
    if meta["property_id"] == "P54" and meta["first_token_informative"]:
        return "immediate_p54"
    if (
        meta["property_id"] in OTHER_PROPERTY_QUOTAS
        and meta["first_token_informative"]
    ):
        return "immediate_other"
    return None


def freeze_cohorts(
    items: list[dict[str, Any]],
    token_meta: dict[str, dict[str, Any]],
    shared: dict[str, set[str]],
    excluded: set[str],
) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    item_index = {str(row["item_id"]): row for row in items}
    cohorts: dict[str, list[str]] = {}
    availability: dict[str, Any] = {}
    all_selected: set[str] = set()
    eligible_by_split: dict[str, set[str]] = {}

    for split in SPLITS:
        eligible = {
            item_id
            for item_id in shared[split]
            if item_id not in excluded
            and item_id in token_meta
            and str(item_index[item_id]["split"]) == split
            and not token_meta[item_id]["prefix_containment"]
            and classify_stratum(token_meta[item_id]) is not None
        }
        eligible_by_split[split] = eligible
        by_stratum = {
            stratum: {
                item_id
                for item_id in eligible
                if classify_stratum(token_meta[item_id]) == stratum
            }
            for stratum in STRATUM_QUOTAS
        }
        selected: list[str] = []
        selected_shared = stable_order(f"{split}|shared", by_stratum["shared_prefix_p54"])[
            : STRATUM_QUOTAS["shared_prefix_p54"]
        ]
        selected_p54 = stable_order(f"{split}|p54", by_stratum["immediate_p54"])[
            : STRATUM_QUOTAS["immediate_p54"]
        ]
        if len(selected_shared) != 10 or len(selected_p54) != 10:
            raise RuntimeError(f"insufficient P54 strata for {split}")
        selected.extend(selected_shared)
        selected.extend(selected_p54)
        property_rows = {}
        immediate_other = by_stratum["immediate_other"]
        for property_id, quota in OTHER_PROPERTY_QUOTAS.items():
            ordered = stable_order(
                f"{split}|other|{property_id}",
                {
                    item_id
                    for item_id in immediate_other
                    if token_meta[item_id]["property_id"] == property_id
                },
            )
            if len(ordered) < quota:
                raise RuntimeError(f"insufficient {property_id} for {split}")
            property_rows[property_id] = ordered[:quota]
            selected.extend(ordered[:quota])
        selected = stable_order(f"{split}|cohort", selected)
        if len(selected) != COHORT_SIZE or len(set(selected)) != COHORT_SIZE:
            raise RuntimeError(f"cohort count drift for {split}")
        cohorts[split] = selected
        all_selected.update(selected)
        availability[split] = {
            "eligible_count": len(eligible),
            "stratum_available": {
                key: len(value) for key, value in by_stratum.items()
            },
            "selected_by_stratum": {
                key: sum(
                    classify_stratum(token_meta[item_id]) == key
                    for item_id in selected
                )
                for key in STRATUM_QUOTAS
            },
            "selected_other_properties": property_rows,
        }

    reserve_pool = set().union(*eligible_by_split.values()) - all_selected
    strict_reserve = stable_order(
        "reserve|shared",
        {
            item_id
            for item_id in reserve_pool
            if classify_stratum(token_meta[item_id]) == "shared_prefix_p54"
        },
    )[:6]
    p54_reserve = stable_order(
        "reserve|p54",
        {
            item_id
            for item_id in reserve_pool
            if classify_stratum(token_meta[item_id]) == "immediate_p54"
        },
    )[:6]
    other_reserve = stable_order(
        "reserve|other",
        {
            item_id
            for item_id in reserve_pool
            if classify_stratum(token_meta[item_id]) == "immediate_other"
        },
    )[:6]
    reserve = stable_order("reserve|all", strict_reserve + p54_reserve + other_reserve)
    if len(reserve) != 18 or len(set(reserve)) != 18:
        raise RuntimeError("insufficient Phase1141 reserve")
    audit = {
        "availability": availability,
        "cohort_sizes": {key: len(value) for key, value in cohorts.items()},
        "cohorts_disjoint": set(cohorts["discovery"]).isdisjoint(
            cohorts["confirmation"]
        ),
        "selected_avoid_prior_hidden": all_selected.isdisjoint(excluded),
        "reserve_count": len(reserve),
        "reserve_strata": {
            key: sum(classify_stratum(token_meta[item_id]) == key for item_id in reserve)
            for key in STRATUM_QUOTAS
        },
        "reserve_disjoint": set(reserve).isdisjoint(all_selected),
    }
    return cohorts, reserve, audit


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1141 after model output exists")
    items = read_jsonl(SOURCE_ITEMS)
    item_index = {str(row["item_id"]): row for row in items}
    q4 = behavior_all_four_by_split(Q4_DECISIONS)
    q14 = behavior_all_four_by_split(Q14_DECISIONS)
    shared = {split: q4[split] & q14[split] for split in SPLITS}
    token_meta = token_metadata()
    excluded = excluded_hidden_ids()
    cohorts, reserve, cohort_audit = freeze_cohorts(
        items, token_meta, shared, excluded
    )
    prior_prereg = read_json(SOURCE1140 / "protocol/preregistration.json")
    prior_audit = read_json(SOURCE1140 / "audit/independent_result_audit.json")
    prior_final = read_json(SOURCE1140 / "analysis/final.json")
    tokenizer4 = ROOT / "models/hf/qwen3-4b/tokenizer.json"
    tokenizer14 = ROOT / "models/hf/Qwen3-14B/tokenizer.json"
    selected = set(cohorts["discovery"]) | set(cohorts["confirmation"])
    selected_meta = {item_id: token_meta[item_id] for item_id in selected}
    property_counts = {
        split: dict(sorted({
            property_id: sum(
                str(item_index[item_id]["property_id"]) == property_id
                for item_id in cohorts[split]
            )
            for property_id in OTHER_PROPERTY_QUOTAS | {"P54": 0}
        }.items()))
        for split in SPLITS
    }
    checks = {
        "phase1140_audit_passed": bool(prior_audit["all_checks_passed"]),
        "phase1140_auto_continue_false": prior_final["auto_continue"] is False,
        "source_count_491": len(items) == 491,
        "token_metadata_complete": len(token_meta) == len(items),
        "qwen_tokenizer_json_identical": sha256_file(tokenizer4) == sha256_file(tokenizer14),
        "cohorts_30_each": all(len(cohorts[split]) == 30 for split in SPLITS),
        "cohorts_disjoint": cohort_audit["cohorts_disjoint"],
        "selected_avoid_prior_hidden": cohort_audit["selected_avoid_prior_hidden"],
        "reserve_18": len(reserve) == 18,
        "reserve_disjoint": cohort_audit["reserve_disjoint"],
        "strata_exact": all(
            cohort_audit["availability"][split]["selected_by_stratum"] == STRATUM_QUOTAS
            for split in SPLITS
        ),
        "no_prefix_containment": all(
            not selected_meta[item_id]["prefix_containment"] for item_id in selected
        ),
        "all_selected_behavior_qualified_both_models": all(
            item_id in shared[split]
            for split in SPLITS
            for item_id in cohorts[split]
        ),
        "frozen_depth_0_7": REQUESTED_FRACTION == 0.7,
        "scopes_frozen": SCOPES == (
            "answer_boundary",
            "first_lexical_divergence",
            "candidate_prediction_span",
        ),
        "expected_records_1080": EXPECTED_RECORDS == 1080,
        "no_output_before_protocol": not (OUT_ROOT / "runs").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1141 protocol checks failed: {checks}")
    core = {
        "schema_version": "phase1141_first_lexical_divergence_preregistration.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "objective": (
            "Prospectively test whether the first unequal candidate token is a minimal "
            "causally sufficient lexical discrimination position, using fresh source splits "
            "and matched temporal and cross-item donor controls."
        ),
        "epistemic_scope": (
            "Same-family FP16 whole-residual sufficiency at one frozen depth. First lexical "
            "divergence is not pre-labelled as a semantic boundary. Materials are deterministic "
            "Wikidata templates with external-machine consensus, not human gold."
        ),
        "source": {
            "items_sha256": sha256_file(SOURCE_ITEMS),
            "qwen4_decisions_sha256": sha256_file(Q4_DECISIONS),
            "qwen14_decisions_sha256": sha256_file(Q14_DECISIONS),
            "token_cases_sha256": sha256_file(TOKEN_CASES),
            "qwen_tokenizer_json_sha256": sha256_file(tokenizer4),
            "phase1140_protocol_digest": prior_prereg["protocol_digest"],
            "phase1140_final_digest": prior_final["final_digest"],
            "phase1140_audit_digest": prior_audit["audit_digest"],
            "script_sha256": sha256_file(Path(__file__)),
        },
        "models": prior_prereg["models"],
        "material": {
            "source_splits": list(SPLITS),
            "shared_behavior_pool_counts_before_exclusion": {
                split: len(shared[split]) for split in SPLITS
            },
            "prior_hidden_exclusion_count": len(excluded),
            "cohorts": cohorts,
            "reserve_item_ids": reserve,
            "cohort_audit": cohort_audit,
            "stratum_quotas": STRATUM_QUOTAS,
            "other_property_quotas": OTHER_PROPERTY_QUOTAS,
            "property_counts": property_counts,
            "token_metadata": selected_meta,
            "hidden_intervention_naive_for_selected_items": True,
        },
        "intervention": {
            "requested_fraction": REQUESTED_FRACTION,
            "position_scopes": list(SCOPES),
            "alphas": list(ALPHAS),
            "component": "whole residual stream after frozen layer",
            "answer_boundary": "patch prompt_length-1",
            "first_lexical_divergence": (
                "patch prompt_length+LCP(old,new)-1; donor uses its own LCP"
            ),
            "candidate_prediction_span": (
                "same-item main curves only; patch the complete candidate scoring path"
            ),
            "main_curves_per_item": MAIN_CURVES_PER_ITEM,
            "same_answer_temporal_curves_per_item": TEMPORAL_CURVES_PER_ITEM,
            "cross_item_wrong_donor_curves_per_item": CROSS_ITEM_CURVES_PER_ITEM,
            "cross_item_pairing": "deterministic cyclic pairing within frozen stratum",
            "expected_records_per_model_split": EXPECTED_RECORDS,
            "full_candidate_and_local_divergence_ledgers": True,
        },
        "thresholds": THRESHOLDS,
        "selection": {
            "discovery_rule": (
                "both models must pass every frozen first-divergence sufficiency, stratum, "
                "minimality, identity, and negative-control gate"
            ),
            "confirmation_rule": "repeat the identical gates on the untouched source confirmation split",
        },
        "hard_stops": [
            "do not consume confirmation unless both discovery models qualify",
            "do not change cohort, depth, scopes, strata, controls, or thresholds after output",
            "do not include prefix-containment pairs whose divergence is EOS versus a token",
            "do not remove failed curves or choose a model-specific scope",
            "do not call first lexical divergence a semantic boundary",
            "do not run head, MLP, neuron, SAE, Jacobian, or component searches before confirmation",
            "a pass is sufficiency at one state position, not necessity or mechanism closure",
        ],
        "auto_continue_rule": (
            "Only independent two-model confirmation authorizes a separately preregistered "
            "component-mediation phase on the 18 untouched reserve items."
        ),
    }
    prereg = dict(core)
    prereg["protocol_digest"] = digest(core)
    audit_core = {
        "schema_version": "phase1141_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol/preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol/audit.json", audit)
    print(json.dumps({
        "phase": PHASE,
        "command": "protocol",
        "checks": f"{audit['passed_count']}/{audit['check_count']}",
        "shared_pool": {split: len(shared[split]) for split in SPLITS},
        "cohorts": {split: len(cohorts[split]) for split in SPLITS},
        "reserve": len(reserve),
        "protocol_digest": prereg["protocol_digest"],
    }, ensure_ascii=False), flush=True)


def build_curves(
    item_ids: list[str],
    token_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for item_id in item_ids:
        stratum = classify_stratum(token_meta[item_id])
        if stratum is None:
            raise RuntimeError(f"unclassified selected item {item_id}")
        by_stratum[stratum].append(item_id)
    partner = {}
    for stratum, ids in by_stratum.items():
        ordered = stable_order(f"wrong-donor|{stratum}", ids)
        if len(ordered) < 2:
            raise RuntimeError(f"insufficient cross-item stratum {stratum}")
        partner.update({item_id: ordered[(index + 1) % len(ordered)] for index, item_id in enumerate(ordered)})

    curves = []

    def add(
        item_id: str,
        kind: str,
        target_state: str,
        source_item_id: str,
        source_state: str,
        base_key: str,
        desired_key: str,
        panel: str,
    ) -> None:
        curves.append({
            "curve_id": (
                f"{item_id}|{kind}|{target_state}|{source_item_id}|{source_state}"
            ),
            "item_id": item_id,
            "source_item_id": source_item_id,
            "curve_kind": kind,
            "panel": panel,
            "stratum": classify_stratum(token_meta[item_id]),
            "property_id": token_meta[item_id]["property_id"],
            "common_prefix_length": token_meta[item_id]["common_prefix_length"],
            "source_common_prefix_length": token_meta[source_item_id]["common_prefix_length"],
            "target_state": target_state,
            "target_case_id": source.state_id(item_id, target_state),
            "source_state": source_state,
            "source_case_id": source.state_id(source_item_id, source_state),
            "base_key": base_key,
            "desired_key": desired_key,
        })

    for item_id in sorted(item_ids):
        specifications = [
            ("original_pre", "original_post", "old", "new", "original"),
            ("original_post", "original_pre", "new", "old", "original"),
            ("swapped_pre", "swapped_post", "new", "old", "swapped"),
            ("swapped_post", "swapped_pre", "old", "new", "swapped"),
        ]
        for target_state, source_state, base_key, desired_key, panel in specifications:
            add(item_id, "main", target_state, item_id, source_state, base_key, desired_key, panel)
            add(
                item_id,
                "cross_item_wrong_donor_control",
                target_state,
                partner[item_id],
                source_state,
                base_key,
                desired_key,
                panel,
            )
        add(
            item_id,
            "same_answer_temporal_control",
            "original_pre",
            item_id,
            "original_pre_early",
            "old",
            "new",
            "original",
        )
        add(
            item_id,
            "same_answer_temporal_control",
            "original_post",
            item_id,
            "original_post_late",
            "new",
            "old",
            "original",
        )
    return curves


def score_token_paths(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_logits = []
    targets = []
    ownership = []
    for index, row in enumerate(rows):
        prompt_length = int(row["prompt_length"])
        for offset, token_id in enumerate(row["continuation_ids"]):
            selected_logits.append(logits[index, prompt_length - 1 + offset, :].float())
            targets.append(int(token_id))
            ownership.append(index)
    matrix = torch.stack(selected_logits)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=matrix.device)
    token_logp = -F.cross_entropy(matrix, target_tensor, reduction="none")
    grouped: list[list[float]] = [[] for _ in rows]
    for owner, value in zip(ownership, token_logp.detach().cpu().tolist()):
        grouped[owner].append(float(value))
    result = []
    for row, values in zip(rows, grouped):
        offset = int(row["decision_offset"])
        if not 0 <= offset < len(values):
            raise RuntimeError("decision offset outside candidate")
        result.append({
            "token_count": len(values),
            "first_token_id": int(row["continuation_ids"][0]),
            "decision_token_id": int(row["continuation_ids"][offset]),
            "first_logp": values[0],
            "decision_logp": values[offset],
            "full_logp_mean": sum(values) / len(values),
            "full_logp_sum": sum(values),
            "finite": all(math.isfinite(value) for value in values),
        })
    del matrix, target_tensor, token_logp
    return result


def score_intervention_batch(
    model,
    layer,
    entries: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    vectors: dict[str, torch.Tensor],
    token_meta: dict[str, dict[str, Any]],
    tokenizer,
    pad_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    expanded = []
    owners = []
    source_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    alpha_rows = []
    for entry in entries:
        target_case = cases[str(entry["target_case_id"])]
        source_case = cases[str(entry["source_case_id"])]
        target_offset = int(token_meta[str(entry["item_id"])]["common_prefix_length"])
        source_offset = int(token_meta[str(entry["source_item_id"])]["common_prefix_length"])
        for candidate_key in ("old", "new"):
            target_row = source.tokenize_case(tokenizer, target_case, candidate_key)
            source_row = source.tokenize_case(tokenizer, source_case, candidate_key)
            expected_target = token_meta[str(entry["item_id"])][f"{candidate_key}_ids"]
            expected_source = token_meta[str(entry["source_item_id"])][f"{candidate_key}_ids"]
            if target_row["continuation_ids"] != expected_target:
                raise RuntimeError("target token metadata drift")
            if source_row["continuation_ids"] != expected_source:
                raise RuntimeError("source token metadata drift")
            target_row["decision_offset"] = target_offset
            all_positions = prior.prediction_positions(target_row, device)
            donor = vectors[prior.vector_key(str(entry["source_case_id"]), candidate_key)]
            scope = str(entry["scope"])
            if scope == "answer_boundary":
                positions = all_positions[:1]
                donor = donor[:1]
            elif scope == "first_lexical_divergence":
                positions = all_positions[target_offset : target_offset + 1]
                donor = donor[source_offset : source_offset + 1]
            elif scope == "candidate_prediction_span":
                if str(entry["item_id"]) != str(entry["source_item_id"]):
                    raise RuntimeError("full span is restricted to same-item main curves")
                if target_row["continuation_ids"] != source_row["continuation_ids"]:
                    raise RuntimeError("same-item candidate path changed across states")
                positions = all_positions
            else:
                raise RuntimeError(f"unknown scope {scope}")
            if positions.numel() == 0 or donor.shape[0] != positions.shape[0]:
                raise RuntimeError("intervention path length mismatch")
            expanded.append(target_row)
            owners.append(entry)
            source_rows.append(donor)
            position_rows.append(positions)
            alpha_rows.append(float(entry["alpha"]))

    input_ids, attention_mask = source.pad_sequences(expanded, pad_id, device)
    alphas = torch.tensor(alpha_rows, dtype=torch.float32, device=device)
    clean_scores = None
    with torch.inference_mode():
        if all(math.isclose(float(entry["alpha"]), 0.0, abs_tol=1e-12) for entry in entries):
            clean_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            clean_scores = score_token_paths(clean_output.logits, expanded)
            del clean_output
        with prior.LivePathInterpolation(layer, position_rows, source_rows, alphas) as patch:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if patch.calls != 1:
            raise RuntimeError(f"interpolation hook called {patch.calls} times")
        patched_scores = score_token_paths(output.logits, expanded)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    clean_grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, (row, entry, score_row) in enumerate(zip(expanded, owners, patched_scores)):
        record_id = str(entry["record_id"])
        grouped[record_id][str(row["candidate_key"])] = score_row
        if clean_scores is not None:
            clean_grouped[record_id][str(row["candidate_key"])] = clean_scores[index]

    result = []
    for entry in entries:
        record_id = str(entry["record_id"])
        scores = grouped[record_id]
        base_key = str(entry["base_key"])
        desired_key = str(entry["desired_key"])
        finite = bool(scores[base_key]["finite"] and scores[desired_key]["finite"])
        full_margin = (
            scores[desired_key]["full_logp_mean"] - scores[base_key]["full_logp_mean"]
            if finite else None
        )
        decision_margin = (
            scores[desired_key]["decision_logp"] - scores[base_key]["decision_logp"]
            if finite else None
        )
        clean_full = clean_decision = full_drift = decision_drift = None
        if record_id in clean_grouped:
            clean = clean_grouped[record_id]
            clean_finite = bool(clean[base_key]["finite"] and clean[desired_key]["finite"])
            if clean_finite:
                clean_full = clean[desired_key]["full_logp_mean"] - clean[base_key]["full_logp_mean"]
                clean_decision = clean[desired_key]["decision_logp"] - clean[base_key]["decision_logp"]
                full_drift = full_margin - clean_full if full_margin is not None else None
                decision_drift = decision_margin - clean_decision if decision_margin is not None else None
        row = dict(entry)
        row.update({
            "finite": finite,
            "candidate_token_counts": {key: int(scores[key]["token_count"]) for key in ("old", "new")},
            "candidate_first_token_ids": {key: int(scores[key]["first_token_id"]) for key in ("old", "new")},
            "candidate_decision_token_ids": {key: int(scores[key]["decision_token_id"]) for key in ("old", "new")},
            "full_oriented_margin": full_margin,
            "decision_oriented_margin": decision_margin,
            "unhooked_full_oriented_margin": clean_full,
            "unhooked_decision_oriented_margin": clean_decision,
            "identity_full_margin_drift": full_drift,
            "identity_decision_margin_drift": decision_drift,
            "scores": scores if finite else None,
        })
        result.append(row)
    del output, input_ids, attention_mask, alphas, patched_scores
    return result


def curve_rows(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["scope"] == scope:
            grouped[str(row["curve_id"])].append(row)
    curves = []
    for curve_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        if [float(row["alpha"]) for row in ordered] != list(ALPHAS):
            raise RuntimeError(f"alpha grid drift for {curve_id}")
        base, endpoint = ordered
        finite = bool(base["finite"] and endpoint["finite"])
        full_base = base["full_oriented_margin"]
        full_endpoint = endpoint["full_oriented_margin"]
        decision_base = base["decision_oriented_margin"]
        decision_endpoint = endpoint["decision_oriented_margin"]
        full_change = (
            float(full_endpoint) - float(full_base)
            if finite and full_base is not None and full_endpoint is not None else None
        )
        decision_change = (
            float(decision_endpoint) - float(decision_base)
            if finite and decision_base is not None and decision_endpoint is not None else None
        )
        curves.append({
            "curve_id": curve_id,
            "model": base["model"],
            "split": base["split"],
            "scope": scope,
            "item_id": base["item_id"],
            "source_item_id": base["source_item_id"],
            "curve_kind": base["curve_kind"],
            "panel": base["panel"],
            "stratum": base["stratum"],
            "property_id": base["property_id"],
            "common_prefix_length": base["common_prefix_length"],
            "target_state": base["target_state"],
            "source_state": base["source_state"],
            "finite": finite,
            "identity_full_margin_drift": base["identity_full_margin_drift"],
            "identity_decision_margin_drift": base["identity_decision_margin_drift"],
            "full_baseline_margin": full_base,
            "full_endpoint_margin": full_endpoint,
            "full_margin_change": full_change,
            "full_baseline_valid": bool(finite and full_base is not None and full_base < 0),
            "full_endpoint_flip": bool(finite and full_endpoint is not None and full_endpoint > 0),
            "full_positive_change": bool(full_change is not None and full_change > 0),
            "decision_baseline_margin": decision_base,
            "decision_endpoint_margin": decision_endpoint,
            "decision_margin_change": decision_change,
            "decision_endpoint_flip": bool(finite and decision_endpoint is not None and decision_endpoint > 0),
        })
    return curves


def fraction(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def scope_metrics(records: list[dict[str, Any]], scope: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curves = curve_rows(records, scope)
    main = [row for row in curves if row["curve_kind"] == "main"]
    temporal = [row for row in curves if row["curve_kind"] == "same_answer_temporal_control"]
    cross = [row for row in curves if row["curve_kind"] == "cross_item_wrong_donor_control"]
    scope_records = [row for row in records if row["scope"] == scope]
    full_identity = [abs(float(row["identity_full_margin_drift"])) for row in curves if row["identity_full_margin_drift"] is not None]
    decision_identity = [abs(float(row["identity_decision_margin_drift"])) for row in curves if row["identity_decision_margin_drift"] is not None]
    main_change = median(row["full_margin_change"] for row in main)
    temporal_change = median(abs(float(row["full_margin_change"])) for row in temporal if row["full_margin_change"] is not None)
    cross_change = median(abs(float(row["full_margin_change"])) for row in cross if row["full_margin_change"] is not None)
    metrics = {
        "scope": scope,
        "record_count": len(scope_records),
        "curve_count": len(curves),
        "main_curve_count": len(main),
        "same_answer_control_count": len(temporal),
        "cross_item_control_count": len(cross),
        "finite_fraction": sum(bool(row["finite"]) for row in scope_records) / max(len(scope_records), 1),
        "identity_full_max_abs_margin_drift": max(full_identity) if full_identity else None,
        "identity_decision_max_abs_margin_drift": max(decision_identity) if decision_identity else None,
        "full_baseline_valid_fraction": fraction(main, "full_baseline_valid"),
        "full_main_endpoint_flip_fraction": fraction(main, "full_endpoint_flip"),
        "full_original_endpoint_flip_fraction": fraction([row for row in main if row["panel"] == "original"], "full_endpoint_flip"),
        "full_swapped_endpoint_flip_fraction": fraction([row for row in main if row["panel"] == "swapped"], "full_endpoint_flip"),
        "full_main_positive_change_fraction": fraction(main, "full_positive_change"),
        "full_main_margin_change_median": main_change,
        "full_same_answer_abs_change_median": temporal_change,
        "full_cross_item_abs_change_median": cross_change,
        "full_main_to_same_answer_ratio": (
            main_change / max(temporal_change, EPSILON)
            if main_change is not None and temporal_change is not None else None
        ),
        "full_main_to_cross_item_ratio": (
            main_change / max(cross_change, EPSILON)
            if main_change is not None and cross_change is not None else None
        ),
        "full_same_answer_endpoint_flip_fraction": fraction(temporal, "full_endpoint_flip"),
        "full_cross_item_endpoint_flip_fraction": fraction(cross, "full_endpoint_flip"),
        "decision_main_endpoint_flip_fraction": fraction(main, "decision_endpoint_flip"),
        "stratum_endpoint_flip_fraction": {
            stratum: fraction([row for row in main if row["stratum"] == stratum], "full_endpoint_flip")
            for stratum in STRATUM_QUOTAS
        },
    }
    metrics["curve_digest"] = digest(curves)
    return metrics, curves


def analyze_records(
    model_name: str,
    split: str,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    metrics = {}
    curves = {}
    for scope in SCOPES:
        metrics[scope], curves[scope] = scope_metrics(records, scope)
    decision = metrics["first_lexical_divergence"]
    boundary = metrics["answer_boundary"]
    span = metrics["candidate_prediction_span"]
    shared_decision = decision["stratum_endpoint_flip_fraction"]["shared_prefix_p54"]
    shared_boundary = boundary["stratum_endpoint_flip_fraction"]["shared_prefix_p54"]
    shared_span = span["stratum_endpoint_flip_fraction"]["shared_prefix_p54"]

    answer_records = {
        (str(row["curve_id"]), float(row["alpha"])): row
        for row in records
        if row["scope"] == "answer_boundary"
        and row["curve_kind"] == "main"
        and int(row["common_prefix_length"]) == 0
    }
    decision_records = {
        (str(row["curve_id"]), float(row["alpha"])): row
        for row in records
        if row["scope"] == "first_lexical_divergence"
        and row["curve_kind"] == "main"
        and int(row["common_prefix_length"]) == 0
    }
    if set(answer_records) != set(decision_records):
        raise RuntimeError("LCP0 scope identity key drift")
    lcp0_diffs = [
        abs(float(answer_records[key]["full_oriented_margin"]) - float(decision_records[key]["full_oriented_margin"]))
        for key in answer_records
        if answer_records[key]["full_oriented_margin"] is not None
        and decision_records[key]["full_oriented_margin"] is not None
    ]
    lcp0_max = max(lcp0_diffs) if lcp0_diffs else None
    gate_checks = {
        "finite": decision["finite_fraction"] >= THRESHOLDS["finite_fraction"],
        "identity_full": decision["identity_full_max_abs_margin_drift"] is not None and decision["identity_full_max_abs_margin_drift"] <= THRESHOLDS["identity_max_abs_margin_drift"],
        "identity_decision": decision["identity_decision_max_abs_margin_drift"] is not None and decision["identity_decision_max_abs_margin_drift"] <= THRESHOLDS["identity_max_abs_margin_drift"],
        "baseline_valid": decision["full_baseline_valid_fraction"] >= THRESHOLDS["baseline_valid_fraction"],
        "main_endpoint": decision["full_main_endpoint_flip_fraction"] >= THRESHOLDS["main_endpoint_flip_fraction"],
        "each_stratum": all(value >= THRESHOLDS["stratum_endpoint_flip_fraction"] for value in decision["stratum_endpoint_flip_fraction"].values()),
        "original_panel": decision["full_original_endpoint_flip_fraction"] >= THRESHOLDS["panel_endpoint_flip_fraction"],
        "swapped_panel": decision["full_swapped_endpoint_flip_fraction"] >= THRESHOLDS["panel_endpoint_flip_fraction"],
        "positive_change": decision["full_main_positive_change_fraction"] >= THRESHOLDS["main_positive_change_fraction"],
        "same_answer_flip": decision["full_same_answer_endpoint_flip_fraction"] <= THRESHOLDS["same_answer_control_flip_fraction"],
        "cross_item_flip": decision["full_cross_item_endpoint_flip_fraction"] <= THRESHOLDS["cross_item_control_flip_fraction"],
        "same_answer_ratio": decision["full_main_to_same_answer_ratio"] is not None and decision["full_main_to_same_answer_ratio"] >= THRESHOLDS["main_to_each_control_ratio"],
        "cross_item_ratio": decision["full_main_to_cross_item_ratio"] is not None and decision["full_main_to_cross_item_ratio"] >= THRESHOLDS["main_to_each_control_ratio"],
        "shared_boundary_improvement": shared_decision - shared_boundary >= THRESHOLDS["shared_minus_boundary_min"],
        "shared_span_noninferiority": shared_decision >= shared_span - THRESHOLDS["shared_span_noninferiority_margin"],
        "lcp0_scope_equivalence": lcp0_max is not None and lcp0_max <= THRESHOLDS["lcp0_scope_equivalence_max_abs_margin"],
    }
    metrics["model"] = model_name
    metrics["split"] = split
    metrics["shared_first_divergence_minus_boundary"] = shared_decision - shared_boundary
    metrics["shared_first_divergence_minus_span"] = shared_decision - shared_span
    metrics["lcp0_boundary_vs_divergence_max_abs_margin"] = lcp0_max
    metrics["gate_checks"] = gate_checks
    metrics["qualified"] = all(gate_checks.values())
    return metrics, curves


def run_command(model_name: str, split: str) -> None:
    if model_name not in MODELS or split not in SPLITS:
        raise RuntimeError("invalid Phase1141 endpoint")
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1141 protocol audit failed")
    if split == "confirmation":
        selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
        if not selection["confirmation_authorized"]:
            raise RuntimeError("Phase1141 discovery denied confirmation")
    output_root = OUT_ROOT / "runs" / split / model_name
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite {output_root}")

    selected_ids = list(prereg["material"]["cohorts"][split])
    items = read_jsonl(SOURCE_ITEMS)
    selected_items, cases = source.causal_cases(items, split, selected_ids)
    all_token_meta = token_metadata()
    selected_token_meta = {item_id: all_token_meta[item_id] for item_id in selected_ids}
    curves = build_curves(selected_ids, selected_token_meta)
    if len(curves) != COHORT_SIZE * (
        MAIN_CURVES_PER_ITEM + TEMPORAL_CURVES_PER_ITEM + CROSS_ITEM_CURVES_PER_ITEM
    ):
        raise RuntimeError("Phase1141 curve count drift")

    model = None
    started = time.time()
    records: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, placement = phase1138.load_model(model_name, prereg)
        precision = quantization_audit(model)
        expected = prereg["models"][model_name]
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != int(expected["expected_parameter_count"]):
            raise RuntimeError(f"{model_name} parameter count mismatch")
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError(f"{model_name} FP16/no-quantization gate failed")
        layers = get_layers(model)
        depth_rows = [
            row for row in phase1138.depth_rows_for_model(len(layers))
            if math.isclose(float(row["requested_fraction"]), REQUESTED_FRACTION, abs_tol=1e-12)
        ]
        if len(depth_rows) != 1:
            raise RuntimeError("frozen depth did not map to one layer")
        depth = int(depth_rows[0]["depth"])
        layer = layers[depth - 1]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        batch_size = int(expected["batch_size"])
        vectors = prior.capture_candidate_paths(
            model, layer, cases, tokenizer, int(pad_id), device, batch_size
        )

        entries = []
        for scope in SCOPES:
            for alpha in ALPHAS:
                for curve in curves:
                    if curve["curve_kind"] != "main" and scope != "first_lexical_divergence":
                        continue
                    entry = dict(curve)
                    entry.update({
                        "schema_version": "phase1141_first_divergence_record.v1",
                        "phase": PHASE,
                        "model": model_name,
                        "split": split,
                        "depth": depth,
                        "relative_depth": depth / len(layers),
                        "requested_fraction": REQUESTED_FRACTION,
                        "scope": scope,
                        "alpha": alpha,
                        "record_id": f"{curve['curve_id']}|scope={scope}|alpha={alpha:.2f}",
                    })
                    entries.append(entry)
        if len(entries) != EXPECTED_RECORDS:
            raise RuntimeError(f"entry count drift: {len(entries)}")

        entries_per_batch = max(1, batch_size // 2)
        for scope in SCOPES:
            for alpha in ALPHAS:
                current = [
                    entry for entry in entries
                    if entry["scope"] == scope and math.isclose(float(entry["alpha"]), alpha, abs_tol=1e-12)
                ]
                for start in range(0, len(current), entries_per_batch):
                    records.extend(score_intervention_batch(
                        model,
                        layer,
                        current[start : start + entries_per_batch],
                        cases,
                        vectors,
                        selected_token_meta,
                        tokenizer,
                        int(pad_id),
                        device,
                    ))
                print(json.dumps({
                    "phase": PHASE,
                    "model": model_name,
                    "split": split,
                    "scope": scope,
                    "alpha": alpha,
                    "records": len(records),
                }), flush=True)
        if len(records) != EXPECTED_RECORDS:
            raise RuntimeError(f"record count drift: {len(records)}")
        metrics, curves_by_scope = analyze_records(model_name, split, records)
        core = {
            "schema_version": "phase1141_first_divergence_run_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "parameter_count": parameter_count,
            "placement": placement,
            "layer_count": len(layers),
            "depth": depth,
            "relative_depth": depth / len(layers),
            "requested_fraction": REQUESTED_FRACTION,
            "item_count": len(selected_items),
            "record_count": len(records),
            "metrics": metrics,
            "elapsed_seconds": time.time() - started,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "record_digest": digest(records),
            "evidence_scope": "same_family_fresh_split_first_lexical_divergence_sufficiency",
        }
        summary = dict(core)
        summary["summary_digest"] = digest(core)
        write_jsonl(output_root / "records.jsonl", records)
        write_json(output_root / "summary.json", summary)
        for scope, rows in curves_by_scope.items():
            write_jsonl(output_root / f"curves.{scope}.jsonl", rows)
        print(json.dumps({
            "phase": PHASE,
            "command": "run",
            "model": model_name,
            "split": split,
            "qualified": metrics["qualified"],
            "boundary_flip": metrics["answer_boundary"]["full_main_endpoint_flip_fraction"],
            "divergence_flip": metrics["first_lexical_divergence"]["full_main_endpoint_flip_fraction"],
            "span_flip": metrics["candidate_prediction_span"]["full_main_endpoint_flip_fraction"],
            "shared_gain": metrics["shared_first_divergence_minus_boundary"],
            "summary_digest": summary["summary_digest"],
        }), flush=True)
    finally:
        if model is not None:
            if model_name == "qwen3_4b":
                release_fp16(model)
            else:
                del model
                gc.collect()
                torch.cuda.empty_cache()


def select_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    summaries = {
        model: read_json(OUT_ROOT / "runs/discovery" / model / "summary.json")
        for model in MODELS
    }
    if any(summary["protocol_digest"] != prereg["protocol_digest"] for summary in summaries.values()):
        raise RuntimeError("Phase1141 discovery protocol digest drift")
    qualified = {model: bool(summaries[model]["metrics"]["qualified"]) for model in MODELS}
    authorized = all(qualified.values())
    core = {
        "schema_version": "phase1141_discovery_selection.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": {model: summaries[model]["metrics"] for model in MODELS},
        "qualified": qualified,
        "both_models_qualified": authorized,
        "selected_scope": "first_lexical_divergence" if authorized else None,
        "confirmation_authorized": authorized,
        "selection_rule_followed": True,
    }
    result = dict(core)
    result["selection_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/discovery_selection.json", result)
    print(json.dumps({
        "phase": PHASE,
        "command": "select",
        "qualified": qualified,
        "confirmation_authorized": authorized,
        "selection_digest": result["selection_digest"],
    }), flush=True)


def finalize_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    confirmation = {}
    if selection["confirmation_authorized"]:
        for model in MODELS:
            path = OUT_ROOT / "runs/confirmation" / model / "summary.json"
            if not path.exists():
                raise RuntimeError(f"missing confirmation output for {model}")
            confirmation[model] = read_json(path)
    confirmed = bool(
        confirmation
        and all(confirmation[model]["metrics"]["qualified"] for model in MODELS)
    )
    if not selection["confirmation_authorized"]:
        outcome = "discovery_failed"
    elif confirmed:
        outcome = "first_lexical_divergence_sufficiency_confirmed"
    else:
        outcome = "confirmation_failed"
    core = {
        "schema_version": "phase1141_first_divergence_final.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "confirmation_run": bool(confirmation),
        "confirmation_models": {model: summary["metrics"] for model, summary in confirmation.items()},
        "first_lexical_divergence_sufficiency_confirmed": confirmed,
        "outcome": outcome,
        "component_mediation_authorized": confirmed,
        "component_mediation_material": "Phase1141 reserve 18 only" if confirmed else None,
        "semantic_boundary_claim_authorized": False,
        "necessity_claim_authorized": False,
        "cross_architecture_claim_authorized": False,
        "auto_continue": confirmed,
        "claim_boundary": (
            "A pass establishes same-family whole-residual sufficiency and negative-control "
            "specificity at the first unequal lexical token, at one frozen depth. It does not "
            "identify a semantic boundary, component, neuron, necessity, or universal coordinate."
        ),
    }
    final = dict(core)
    final["final_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize",
        "outcome": outcome,
        "confirmed": confirmed,
        "auto_continue": confirmed,
        "final_digest": final["final_digest"],
    }), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    run = sub.add_parser("run")
    run.add_argument("model", choices=MODELS)
    run.add_argument("split", choices=SPLITS)
    sub.add_parser("select")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.model, args.split)
    elif args.command == "select":
        select_command()
    elif args.command == "finalize":
        finalize_command()
    else:
        raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
