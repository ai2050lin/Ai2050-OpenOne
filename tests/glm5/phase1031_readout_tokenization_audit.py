#!/usr/bin/env python3
"""Audit Phase1030 readout geometry and endpoint token coverage."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import render_chat, tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1030_composition_replication_protocol as phase1030
from phase1029_multibinding_competition_scan import normalize_rows


PHASE = 1031
SCHEMES = (
    "pooled_leave_surface",
    "within_template_leave_surface",
    "cross_template_leave_surface",
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1031_readout_tokenization_audit"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prototypes(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
    *,
    scheme: str,
    evaluation_template: int,
    held_surface: int,
) -> np.ndarray:
    values = np.empty(
        (8, clean_readout.shape[-1]), dtype=np.float32
    )
    for concept_index in range(8):
        indices = []
        for row in cases:
            if int(row["expected_index"]) != concept_index:
                continue
            if int(row["surface_index"]) == held_surface:
                continue
            template = int(row["template_index"])
            if (
                scheme == "within_template_leave_surface"
                and template != evaluation_template
            ):
                continue
            if (
                scheme == "cross_template_leave_surface"
                and template == evaluation_template
            ):
                continue
            indices.append(int(row["case_index"]))
        if not indices:
            raise RuntimeError(
                f"empty prototype {scheme=} {evaluation_template=} "
                f"{held_surface=} {concept_index=}"
            )
        values[concept_index] = np.asarray(
            clean_readout[indices], dtype=np.float32
        ).mean(axis=0)
    return normalize_rows(values)


def prototype_maps(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[tuple[str, int, int], np.ndarray]:
    return {
        (scheme, template, surface): prototypes(
            clean_readout,
            cases,
            scheme=scheme,
            evaluation_template=template,
            held_surface=surface,
        )
        for scheme in SCHEMES
        for template in range(len(phase1030.TEMPLATES))
        for surface in range(len(phase1030.NONCE_PAIRS))
    }


def classify_rows(
    values: np.ndarray,
    rows: list[dict[str, Any]],
    maps: dict[tuple[str, int, int], np.ndarray],
    scheme: str,
    *,
    base_field: str,
    alternate_field: str | None = None,
) -> dict[str, Any]:
    expected_hits = []
    alternate_hits = []
    margins = []
    for index, row in enumerate(rows):
        proto = maps[
            (
                scheme,
                int(row["template_index"]),
                int(row["surface_index"]),
            )
        ]
        similarity = normalize_rows(
            values[index:index + 1]
        )[0] @ proto.T
        expected = int(row[base_field])
        predicted = int(np.argmax(similarity))
        expected_hits.append(int(predicted == expected))
        wrong = np.delete(similarity, expected)
        margins.append(float(similarity[expected] - np.max(wrong)))
        if alternate_field is not None:
            alternate_hits.append(
                int(predicted == int(row[alternate_field]))
            )
    result = {
        "row_count": len(rows),
        "expected_top1": float(np.mean(expected_hits)),
        "expected_vs_wrong_margin": float(np.mean(margins)),
        "chance": 0.125,
    }
    if alternate_hits:
        result["alternate_top1"] = float(np.mean(alternate_hits))
    return result


def clean_metrics(
    clean_readout: np.ndarray,
    cases: list[dict[str, Any]],
    maps: dict[tuple[str, int, int], np.ndarray],
) -> dict[str, Any]:
    result = {}
    for scheme in SCHEMES:
        scheme_result = {}
        for template in range(len(phase1030.TEMPLATES)):
            template_result = {}
            all_hits = []
            for world in phase1030.WORLD_CODES:
                rows = [
                    row
                    for row in cases
                    if int(row["template_index"]) == template
                    and row["world"] == world
                ]
                values = np.asarray([
                    clean_readout[int(row["case_index"])]
                    for row in rows
                ])
                metrics = classify_rows(
                    values,
                    rows,
                    maps,
                    scheme,
                    base_field="expected_index",
                )
                template_result[world] = metrics
                all_hits.extend(
                    [metrics["expected_top1"]] * metrics["row_count"]
                )
            template_result["all_worlds"] = {
                "case_count": sum(
                    row["row_count"]
                    for world, row in template_result.items()
                    if world in phase1030.WORLD_CODES
                ),
                "expected_top1": float(np.mean(all_hits)),
                "chance": 0.125,
            }
            scheme_result[f"template_{template}"] = template_result
        result[scheme] = scheme_result
    return result


def intervention_metrics(
    outputs: np.ndarray,
    units: list[dict[str, Any]],
    maps: dict[tuple[str, int, int], np.ndarray],
) -> dict[str, Any]:
    metric_units = [
        {
            **row,
            "base_index": int(row["target_index"]),
            "alternate_index": int(row["donor_index"]),
        }
        for row in units
    ]
    result = {}
    for scheme in SCHEMES:
        scheme_rows = {}
        for condition_index, condition in enumerate(
            phase1030.CONDITIONS
        ):
            scopes = {}
            for template in range(len(phase1030.TEMPLATES)):
                indices = [
                    index
                    for index, row in enumerate(metric_units)
                    if int(row["template_index"]) == template
                ]
                values = np.asarray(outputs[condition_index, indices])
                rows = [metric_units[index] for index in indices]
                scopes[f"template_{template}"] = classify_rows(
                    values,
                    rows,
                    maps,
                    scheme,
                    base_field="base_index",
                    alternate_field="alternate_index",
                )
            scheme_rows[condition] = scopes
        result[scheme] = scheme_rows
    return result


def token_span_audit(
    model: str,
    common_cases: list[dict[str, Any]],
    model_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    tokenizer = tokenizer_for(model)
    model_by_index = {
        int(row["case_index"]): row for row in model_cases
    }
    role_counts = {
        role: Counter() for role in phase1030.ROLES if role != "pre_output"
    }
    selected_counts = Counter()
    endpoint_checks = []
    template_selected_counts = {
        0: Counter(),
        1: Counter(),
    }
    for row in common_cases:
        rendered = render_chat(tokenizer, model, row["prompt"])
        spans = offset_token_spans(
            tokenizer,
            rendered,
            row["prompt"],
            row["role_fragments"],
        )
        model_row = model_by_index[int(row["case_index"])]
        for role, (start, end) in spans.items():
            length = int(end) - int(start) + 1
            role_counts[role][length] += 1
            endpoint_checks.append(
                int(end)
                == int(model_row["role_positions"][role])
            )
        if row["world"] == "00":
            selected_role = (
                "concept_a_end"
                if row["q0_slot"] == "a"
                else "concept_b_end"
            )
            start, end = spans[selected_role]
            length = int(end) - int(start) + 1
            selected_counts[length] += 1
            template_selected_counts[
                int(row["template_index"])
            ][length] += 1
    selected_total = sum(selected_counts.values())
    result = {
        "model": model,
        "role_token_span_length_counts": {
            role: dict(sorted(counts.items()))
            for role, counts in role_counts.items()
        },
        "selected_source_span_length_counts": dict(
            sorted(selected_counts.items())
        ),
        "selected_source_single_token_rate": (
            selected_counts[1] / selected_total
        ),
        "selected_source_multitoken_rate": (
            1.0 - selected_counts[1] / selected_total
        ),
        "template_selected_source_span_length_counts": {
            str(template): dict(sorted(counts.items()))
            for template, counts in template_selected_counts.items()
        },
        "stored_endpoint_matches_recomputed_rate": float(
            np.mean(endpoint_checks)
        ),
    }
    del tokenizer
    return result


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def artifact_manifest() -> dict[str, Any]:
    manifest_path = OUT_ROOT / "artifact_manifest.json"
    files = []
    for path in sorted(
        item for item in OUT_ROOT.rglob("*")
        if item.is_file() and item != manifest_path
    ):
        files.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        })
    return {
        "schema_version": "phase1031_artifact_manifest.v1",
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def main() -> None:
    common_cases = phase1030.read_jsonl(
        phase1030.OUT_ROOT / "protocol" / "common_cases.jsonl"
    )
    units = phase1030.read_jsonl(
        phase1030.OUT_ROOT / "protocol" / "units.jsonl"
    )
    model_results = {}
    readout_mismatch_models = []
    endpoint_partial_models = []
    stable_selected_source_models = []
    for model in phase1030.MODELS:
        atlas_dir = phase1030.OUT_ROOT / "atlas" / model
        clean_readout = np.load(
            atlas_dir / "clean_readout.fp16.npy", mmap_mode="r"
        )
        outputs = np.load(
            atlas_dir / "confirmation_conditions.fp16.npy",
            mmap_mode="r",
        )
        model_cases = phase1030.read_jsonl(
            phase1030.OUT_ROOT
            / "protocol"
            / f"cases.{model}.jsonl"
        )
        maps = prototype_maps(clean_readout, model_cases)
        clean = clean_metrics(clean_readout, model_cases, maps)
        interventions = intervention_metrics(outputs, units, maps)
        tokenization = token_span_audit(
            model, common_cases, model_cases
        )
        template_gains = {}
        for template in range(len(phase1030.TEMPLATES)):
            key = f"template_{template}"
            pooled = clean["pooled_leave_surface"][key][
                "all_worlds"
            ]["expected_top1"]
            within = clean["within_template_leave_surface"][key][
                "all_worlds"
            ]["expected_top1"]
            template_gains[key] = float(within - pooled)
        if (
            max(template_gains.values()) >= 0.10
            or min(template_gains.values()) <= -0.10
        ):
            readout_mismatch_models.append(model)
        if tokenization["selected_source_multitoken_rate"] > 0.0:
            endpoint_partial_models.append(model)
        within = interventions["within_template_leave_surface"]
        stable_source = all(
            within["selected_source_b"][f"template_{template}"][
                "alternate_top1"
            ]
            - within["unselected_source_b"][f"template_{template}"][
                "alternate_top1"
            ]
            >= 0.30
            for template in range(len(phase1030.TEMPLATES))
        )
        if stable_source:
            stable_selected_source_models.append(model)
        model_result = {
            "model": model,
            "clean_readout_by_scheme": clean,
            "interventions_by_scheme": interventions,
            "tokenization": tokenization,
            "within_minus_pooled_clean_top1": template_gains,
            "selected_source_stable_within_template": stable_source,
            "source_patch_coverage_limit": (
                "endpoint-only patch covers the full selected concept span"
                if tokenization["selected_source_multitoken_rate"] == 0.0
                else (
                    "endpoint-only patch leaves at least one selected "
                    "concept token unpatched in part of the dataset"
                )
            ),
        }
        write_json(OUT_ROOT / f"{model}.json", model_result)
        model_results[model] = model_result

    summary = {
        "schema_version": "phase1031_summary.v1",
        "phase": PHASE,
        "source_phase": 1030,
        "audit_type": "posthoc_instrument_audit_not_gate_replacement",
        "models": model_results,
        "cross_model": {
            "readout_scheme_sensitive_models": readout_mismatch_models,
            "endpoint_partial_coverage_models": endpoint_partial_models,
            "selected_source_stable_within_template_models": (
                stable_selected_source_models
            ),
        },
        "interpretation": (
            "Phase1031 does not revise Phase1030 gates. It determines "
            "whether template-dependent prototype geometry and partial "
            "multi-token endpoint patches are plausible sources of "
            "replication loss."
        ),
    }
    write_json(OUT_ROOT / "summary.json", summary)
    checks = {
        "all_phase1030_models_present": all(
            (
                phase1030.OUT_ROOT / "atlas" / model / "summary.json"
            ).exists()
            for model in phase1030.MODELS
        ),
        "all_recomputed_endpoints_match": all(
            row["tokenization"][
                "stored_endpoint_matches_recomputed_rate"
            ] == 1.0
            for row in model_results.values()
        ),
        "all_readout_arrays_finite": all(
            np.isfinite(np.load(
                phase1030.OUT_ROOT
                / "atlas"
                / model
                / "clean_readout.fp16.npy",
                mmap_mode="r",
            )).all()
            for model in phase1030.MODELS
        ),
    }
    audit = {
        "schema_version": "phase1031_audit.v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    write_json(OUT_ROOT / "audit.json", audit)
    manifest = artifact_manifest()
    write_json(OUT_ROOT / "artifact_manifest.json", manifest)
    print(json.dumps({
        "cross_model": summary["cross_model"],
        "audit": audit,
        "manifest": {
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
