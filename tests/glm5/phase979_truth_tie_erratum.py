"""Post-hoc, non-admission audit for the Phase979 exact EOS/non-EOS tie.

This script does not alter the frozen protocol, runner, auditor, or rows.  It
authenticates the development artifacts, applies the protocol's finite EOS-gap
formula to an exact zero gap, and writes an explicitly non-authorizing erratum.
It must never create a development admission or unlock replication.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import phase979_truth_audit as frozen


ROLE = "post_hoc_tie_semantics_erratum_non_admission"
OUTPUT_PATH = frozen.OUT / "truth_tie_erratum_development.json"
ORIGINAL_VALIDATOR = frozen._validate_primitive_metrics
TIE_KEYS: list[tuple[str, ...]] = []


def tie_aware_primitive_validator(
    row: dict[str, Any],
    key: tuple[str, ...],
    vocab_n: int,
    eos_ids: set[int],
) -> None:
    """Use the frozen validator except for a finite, algebraic exact-zero gap."""
    gap = frozen._finite(row["gap"], f"{key} gap")
    if gap != 0.0:
        ORIGINAL_VALIDATOR(row, key, vocab_n, eos_ids)
        return

    selected_eos_id = row["selected_eos_id"]
    max_non_eos_id = row["max_non_eos_id"]
    top1_id = row["top1_id"]
    for field, value in (
        ("selected_eos_id", selected_eos_id),
        ("max_non_eos_id", max_non_eos_id),
        ("top1_id", top1_id),
    ):
        frozen.require(
            frozen._is_int(value) and 0 <= int(value) < vocab_n,
            f"{key} invalid vocabulary ID in {field}",
        )
    selected_eos_id = int(selected_eos_id)
    max_non_eos_id = int(max_non_eos_id)
    top1_id = int(top1_id)
    frozen.require(selected_eos_id in eos_ids, f"{key} selected EOS ID is not sealed")
    frozen.require(
        max_non_eos_id not in eos_ids,
        f"{key} max_non_eos_id is an EOS token",
    )

    eos_logit = frozen._finite(row["eos_logit"], f"{key} eos_logit")
    max_non_eos = frozen._finite(
        row["max_non_eos_logit"], f"{key} max_non_eos_logit"
    )
    frozen.require(
        frozen._close(gap, max_non_eos - eos_logit),
        f"{key} gap algebra mismatch",
    )
    frozen.require(
        eos_logit == max_non_eos,
        f"{key} zero gap is not an exact recorded-logit tie",
    )

    rank = row["eos_rank"]
    frozen.require(
        frozen._is_int(rank) and 1 <= int(rank) <= vocab_n,
        f"{key} invalid 1-based EOS rank",
    )
    top1 = row["eos_top1"]
    frozen.require(isinstance(top1, bool), f"{key} eos_top1 is not Boolean")
    frozen.require(
        top1 == (top1_id in eos_ids),
        f"{key} eos_top1/top1_id mismatch",
    )
    # Rank counts logits strictly greater than the best EOS.  At an exact tie,
    # rank 1 and a scalar non-EOS argmax can both be true.
    frozen.require(int(rank) == 1, f"{key} exact tie must have EOS competition rank 1")
    if not top1:
        frozen.require(
            top1_id == max_non_eos_id,
            f"{key} tied non-EOS argmax differs from max_non_eos_id",
        )

    probability = frozen._finite(
        row["eos_probability"], f"{key} eos_probability"
    )
    frozen.require(
        0.0 <= probability <= 1.0,
        f"{key} EOS probability is outside [0,1]",
    )
    if "eos_probability_total" in row:
        probability_total = frozen._finite(
            row["eos_probability_total"], f"{key} eos_probability_total"
        )
        frozen.require(
            probability <= probability_total <= 1.0 + 1e-7,
            f"{key} selected/total EOS probability relationship is invalid",
        )
    TIE_KEYS.append(key)


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "effect_summaries": metrics["effect_summaries"],
        "truth_gates": metrics["truth_gates"],
        "punctuation_gates": metrics["punctuation_gates"],
        "truth_gate_passed": metrics["truth_gate_passed"],
        "punctuation_gate_passed": metrics["punctuation_gate_passed"],
        "both_effect_gates_passed": metrics["both_effect_gates_passed"],
        "pair_metrics_sha256": frozen.core.sha256_json(metrics["pair_metrics"]),
    }


def gate_signature(metrics: dict[str, Any]) -> dict[str, Any]:
    summaries = metrics["effect_summaries"]
    return {
        "truth_gate_passed": metrics["truth_gate_passed"],
        "punctuation_gate_passed": metrics["punctuation_gate_passed"],
        "both_effect_gates_passed": metrics["both_effect_gates_passed"],
        "counts": {
            name: {
                "positive_n": summaries[name]["positive_n"],
                "negative_n": summaries[name]["negative_n"],
                "zero_n": summaries[name]["zero_n"],
            }
            for name in ("D_bare", "D_period", "Q_correct", "Q_wrong")
        },
    }


def run_erratum(*, write: bool = True) -> dict[str, Any]:
    frozen.assert_no_old_holdout_import()
    protocol = frozen.authenticate_protocol()
    frozen.require(
        not frozen.DEVELOPMENT_ADMISSION_PATH.exists(),
        "a formal development admission already exists; erratum must fail closed",
    )
    forbidden = [
        frozen.manifest_path("replication"),
        frozen.rows_path("replication"),
        frozen.REPLICATION_AUDIT_PATH,
    ]
    frozen.require(
        not any(path.exists() for path in forbidden),
        "replication artifacts exist; erratum must not run after replication",
    )

    pairs = frozen.dataset.build_pairs("development")
    dataset_audit = frozen.dataset.audit_pairs(pairs)
    frozen.require(dataset_audit.get("passed") is True, "dataset audit failed")
    identity = frozen.expected_split_identity("development", pairs)
    manifest = frozen.authenticate_manifest("development", protocol, identity)
    records = frozen.read_rows("development", manifest["manifest_sha256"])
    ties = [(key, row) for key, row in records.items() if float(row["gap"]) == 0.0]
    frozen.require(len(ties) == 1, f"expected exactly one exact tie, found {len(ties)}")

    model_vocab_n = frozen.runtime_model_vocab_size()
    eos_ids = {int(value) for value in manifest["eos_token_ids"]}
    original_errors: list[dict[str, Any]] = []
    for key, row in ties:
        try:
            ORIGINAL_VALIDATOR(row, key, model_vocab_n, eos_ids)
        except RuntimeError as exc:
            original_errors.append({"key": list(key), "error": str(exc)})
    frozen.require(
        len(original_errors) == len(ties),
        "frozen auditor unexpectedly accepted an exact tie",
    )

    frozen._validate_primitive_metrics = tie_aware_primitive_validator
    TIE_KEYS.clear()
    tok = frozen.load_tokenizer()
    try:
        integrity = frozen.validate_rows(
            "development", manifest, records, pairs, tok
        )
    finally:
        frozen._validate_primitive_metrics = ORIGINAL_VALIDATOR
        del tok
        gc.collect()
    frozen.require(
        integrity.get("passed") is True and TIE_KEYS == [ties[0][0]],
        "tie-aware reconstruction did not validate exactly the observed tie",
    )

    metrics = frozen.build_split_metrics(records, pairs)
    frozen.require(
        metrics["both_effect_gates_passed"] is False,
        "post-hoc erratum is forbidden from authorizing replication",
    )

    tie_key, tie_row = ties[0]
    exponent = math.floor(math.log2(abs(float(tie_row["eos_logit"]))))
    bf16_ulp = 2.0 ** (exponent - 7)
    sensitivity: dict[str, Any] = {}
    signatures: list[dict[str, Any]] = []
    for delta in (-bf16_ulp, bf16_ulp):
        alternative = {key: dict(row) for key, row in records.items()}
        alternative[tie_key]["gap"] = delta
        alt_metrics = frozen.build_split_metrics(alternative, pairs)
        signature = gate_signature(alt_metrics)
        signatures.append(signature)
        sensitivity[f"delta_{delta:+g}"] = {
            "hypothetical_gap": delta,
            "effect_means": {
                name: alt_metrics["effect_summaries"][name]["mean"]
                for name in (
                    "D_bare", "D_period", "Q_correct", "Q_wrong", "interaction"
                )
            },
            "gate_signature": signature,
        }
    observed_signature = gate_signature(metrics)
    frozen.require(
        all(signature == observed_signature for signature in signatures),
        "one-ULP tie sensitivity changed a gate or sign count",
    )

    payload = {
        "schema_version": 1,
        "phase": 979,
        "experiment": "truth_punctuation_teacher_forcing",
        "split": "development",
        "role": ROLE,
        "formal_phase979_audit_status": "FAILED_TO_CERTIFY_DUE_FROZEN_TIE_BUG",
        "formal_development_admission_written": False,
        "replication_authorized": False,
        "replication_model_evaluated": False,
        "phase977_holdout_authorized": False,
        "holdout_loaded": False,
        "mechanism_authorized": False,
        "natural_rollout_claim_authorized": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": frozen.core.sha256_file(frozen.PROTOCOL_PATH),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": frozen.core.sha256_file(
            frozen.manifest_path("development")
        ),
        "rows_file_sha256": frozen.core.sha256_file(
            frozen.rows_path("development")
        ),
        "frozen_auditor_file_sha256": frozen.core.sha256_file(
            Path(frozen.__file__).resolve()
        ),
        "erratum_script_file_sha256": frozen.core.sha256_file(
            Path(__file__).resolve()
        ),
        "dataset_split_sha256": identity["split_sha256"],
        "integrity_under_protocol_tie_semantics": integrity,
        "exact_tie_n": len(ties),
        "exact_ties": [
            {
                "key": list(key),
                "eos_logit": row["eos_logit"],
                "max_non_eos_logit": row["max_non_eos_logit"],
                "gap": row["gap"],
                "selected_eos_id": row["selected_eos_id"],
                "max_non_eos_id": row["max_non_eos_id"],
                "top1_id": row["top1_id"],
                "eos_rank": row["eos_rank"],
                "eos_top1": row["eos_top1"],
                "row_sha256": row["row_sha256"],
            }
            for key, row in ties
        ],
        "frozen_auditor_rejections": original_errors,
        "tie_semantics": {
            "gap_zero_is_a_finite_primary_metric_value": True,
            "eos_rank_uses_strictly_greater_count": True,
            "scalar_argmax_tie_break_is_secondary": True,
            "bf16_ulp_at_observed_logit": bf16_ulp,
        },
        "post_hoc_metric_diagnostic": metric_summary(metrics),
        "one_ulp_sensitivity": sensitivity,
        "one_ulp_gate_and_sign_counts_unchanged": True,
        "decision": "NO_GO_REPLICATION_REMAINS_CLOSED",
        "interpretation_limit": (
            "This post-hoc report corrects tie semantics only. It is not a frozen "
            "admission, cannot authorize replication, and cannot support natural-"
            "rollout or internal-mechanism claims."
        ),
    }
    report = {
        **payload,
        "erratum_report_sha256": frozen.core.sha256_json(payload),
        "audited_at_utc": frozen.core.utc_now(),
    }
    if write:
        if OUTPUT_PATH.exists():
            prior = frozen.core.load_json(OUTPUT_PATH, "existing truth tie erratum")
            frozen._verify_self_hash(
                prior,
                "erratum_report_sha256",
                "audited_at_utc",
                "truth tie erratum",
            )
            frozen.require(
                prior["erratum_report_sha256"] == report["erratum_report_sha256"],
                "existing truth tie erratum differs from recomputation",
            )
        else:
            frozen.core.atomic_write_json(OUTPUT_PATH, report)
    frozen.assert_no_old_holdout_import()
    return report


def self_test() -> dict[str, Any]:
    tie = {
        "selected_eos_id": 2,
        "max_non_eos_id": 1,
        "top1_id": 1,
        "eos_logit": 27.5,
        "max_non_eos_logit": 27.5,
        "gap": 0.0,
        "eos_rank": 1,
        "eos_top1": False,
        "eos_probability": 0.48,
        "eos_probability_total": 0.48,
    }
    original_rejected = False
    try:
        ORIGINAL_VALIDATOR(tie, ("tie",), 8, {2})
    except RuntimeError:
        original_rejected = True
    frozen.require(original_rejected, "frozen validator did not expose the tie bug")
    TIE_KEYS.clear()
    tie_aware_primitive_validator(tie, ("tie",), 8, {2})
    frozen.require(TIE_KEYS == [("tie",)], "tie-aware validator did not log tie")

    invalid = {**tie, "eos_rank": 2}
    rejected_bad_rank = False
    try:
        tie_aware_primitive_validator(invalid, ("bad_rank",), 8, {2})
    except RuntimeError:
        rejected_bad_rank = True
    frozen.require(rejected_bad_rank, "invalid exact-tie rank was accepted")
    return {
        "passed": True,
        "frozen_validator_rejects_exact_tie": True,
        "tie_aware_validator_accepts_consistent_exact_tie": True,
        "tie_aware_validator_rejects_bad_rank": True,
        "cannot_write_admission": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = self_test() if args.self_test else run_erratum(write=not args.no_write)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
