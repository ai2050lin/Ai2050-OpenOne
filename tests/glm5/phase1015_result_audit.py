#!/usr/bin/env python3
"""Audit Phase1015 data integrity without auditing mechanism validity."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1015_query_surface_chain_atlas"
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
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
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def recompute_consistency(
    sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    squared = np.einsum(
        "...d,...d->...",
        sums.astype(np.float64),
        sums.astype(np.float64),
    )
    valid = counts >= 2
    result[valid] = (
        (squared[valid] - counts[valid])
        / (counts[valid] * (counts[valid] - 1.0))
    )
    return result


def protocol_audit() -> dict[str, Any]:
    protocol = read_json(RESULT_ROOT / "protocol" / "protocol.json")
    require(protocol["phase"] == 1015, "protocol phase drift")
    require(len(protocol["families"]) == 5, "family count drift")
    require(
        protocol["surfaces_by_split"]["discovery"] == [0, 1, 2],
        "discovery surface drift",
    )
    require(
        protocol["surfaces_by_split"]["confirmation"] == [3, 4, 5],
        "confirmation surface drift",
    )
    model_rows = {}
    discovery_names = set()
    confirmation_names = set()
    for model in MODELS:
        summary = read_json(
            RESULT_ROOT / "protocol" / model / "summary.json"
        )
        units = read_jsonl(
            RESULT_ROOT / "protocol" / model / "units.jsonl"
        )
        cases = read_jsonl(
            RESULT_ROOT / "protocol" / model / "cases.jsonl"
        )
        case_by_id = {row["record_id"]: row for row in cases}
        require(summary["unit_count"] == 720, f"{model} unit drift")
        require(summary["case_count"] == 5040, f"{model} case drift")
        require(len(units) == 720, f"{model} unit file drift")
        require(len(cases) == 5040, f"{model} case file drift")
        require(
            summary["single_token_query_edit_unit_count"] == 720,
            f"{model} query edit drift",
        )
        require(
            summary["balanced_inventory_unit_count"] == 240,
            f"{model} balanced inventory drift",
        )
        for unit in units:
            require(
                unit["edit_counts"]["Q"] == 1,
                f"{unit['unit_id']} Q width drift",
            )
            require(
                unit["edit_counts"]["L"] == 1,
                f"{unit['unit_id']} L width drift",
            )
            base = case_by_id[unit["case_ids"]["base"]]
            query = case_by_id[unit["case_ids"]["Q"]]
            query_position = base["role_positions"]["query_operator"]
            require(
                base["input_ids"][:query_position]
                == query["input_ids"][:query_position],
                f"{unit['unit_id']} Q prefix drift",
            )
            require(
                base["gold"] != query["gold"],
                f"{unit['unit_id']} Q answer did not flip",
            )
            fq = case_by_id[unit["case_ids"]["FQ"]]
            require(
                base["gold"] == fq["gold"],
                f"{unit['unit_id']} FQ answer drift",
            )
        for case in cases:
            names = set(case["candidate_labels"])
            if case["split"] == "discovery":
                discovery_names.update(names)
            else:
                confirmation_names.update(names)
        model_rows[model] = {
            "unit_count": len(units),
            "case_count": len(cases),
            "counterbalance_counts": summary[
                "counterbalance_counts"
            ],
        }
    require(
        discovery_names.isdisjoint(confirmation_names),
        "discovery/confirmation name leakage",
    )
    return {
        "protocol_digest": protocol["preregistration_digest"],
        "models": model_rows,
        "discovery_confirmation_name_overlap": 0,
    }


def scan_audit() -> tuple[dict[str, Any], list[Path]]:
    model_rows = {}
    key_files = []
    arithmetic_checks = 0
    for model in MODELS:
        model_root = RESULT_ROOT / "scan" / model
        summary_path = model_root / "summary.json"
        summary = read_json(summary_path)
        key_files.append(summary_path)
        require(
            summary["panel_count"] == 30,
            f"{model} panel count drift",
        )
        require(
            summary["singleton_forward_count"] == 5760,
            f"{model} forward count drift",
        )
        require(
            not summary["model_info"]["loaded_8bit"],
            f"{model} formal scan is not BF16",
        )
        require(
            summary["identity_maximum"] == 0,
            f"{model} identity drift",
        )
        require(
            summary["q_causal_prefix_maximum"] == 0,
            f"{model} causal prefix drift",
        )
        events = read_jsonl(model_root / "events.jsonl")
        key_files.append(model_root / "events.jsonl")
        whole_count = int(summary["whole_event_count"])
        head_count = int(summary["head_event_count"])
        require(
            len(events) == whole_count + head_count,
            f"{model} event coverage drift",
        )
        panel_count = 0
        unit_count = 0
        for family_root in sorted(
            path for path in model_root.iterdir() if path.is_dir()
        ):
            for panel_root in sorted(
                path for path in family_root.iterdir()
                if path.is_dir()
            ):
                panel_summary = read_json(panel_root / "summary.json")
                units = read_jsonl(panel_root / "units.jsonl")
                responses = np.load(
                    panel_root / "response_scalars.npz"
                )
                directions = np.load(
                    panel_root / "direction_consistency.npz"
                )
                answer = np.load(
                    panel_root / "answer_head_direction_sums.npz"
                )
                require(
                    panel_summary["unit_count"] == 24,
                    f"{panel_root} unit count drift",
                )
                require(
                    len(units) == 24,
                    f"{panel_root} unit file drift",
                )
                require(
                    responses["normalized_magnitude"].shape
                    == (
                        24,
                        8,
                        7,
                        whole_count + head_count,
                    ),
                    f"{panel_root} response shape drift",
                )
                require(
                    panel_summary["identity_maximum"] == 0,
                    f"{panel_root} identity drift",
                )
                require(
                    panel_summary["q_causal_prefix_maximum"] == 0,
                    f"{panel_root} Q prefix drift",
                )
                expected = recompute_consistency(
                    answer["canonical_sum"],
                    answer["count"],
                )
                answer_role = list(
                    directions["role_names"]
                ).index("answer_boundary")
                stored = directions["head"][
                    1, :, answer_role, :
                ].astype(np.float64)
                mask = np.isfinite(expected)
                require(
                    np.allclose(
                        expected[mask],
                        stored[mask],
                        atol=1e-6,
                        rtol=1e-6,
                    ),
                    f"{panel_root} direction arithmetic drift",
                )
                arithmetic_checks += 1
                panel_count += 1
                unit_count += len(units)
                key_files.extend([
                    panel_root / "summary.json",
                    panel_root / "response_scalars.npz",
                    panel_root / "direction_consistency.npz",
                    panel_root / "answer_head_direction_sums.npz",
                ])
                responses.close()
                directions.close()
                answer.close()
        require(panel_count == 30, f"{model} panel directory drift")
        require(unit_count == 720, f"{model} panel unit coverage drift")
        model_rows[model] = {
            "panel_count": panel_count,
            "unit_count": unit_count,
            "singleton_forward_count": summary[
                "singleton_forward_count"
            ],
            "event_count": len(events),
            "identity_maximum": summary["identity_maximum"],
            "q_causal_prefix_maximum": summary[
                "q_causal_prefix_maximum"
            ],
        }
    return {
        "models": model_rows,
        "arithmetic_panel_count": arithmetic_checks,
    }, key_files


def analysis_audit() -> tuple[dict[str, Any], list[Path]]:
    analysis_root = RESULT_ROOT / "analysis"
    summary_path = analysis_root / "summary.json"
    summary = read_json(summary_path)
    require(summary["phase"] == 1015, "analysis phase drift")
    require(
        summary["main_scan_precision"] == "BF16",
        "analysis precision drift",
    )
    candidates = read_jsonl(
        analysis_root / "recurrent_answer_heads.jsonl"
    )
    cores = read_jsonl(
        analysis_root / "concentrated_answer_head_cores.jsonl"
    )
    trajectories = read_jsonl(
        analysis_root / "role_trajectories.jsonl"
    )
    behavior_stratified = read_jsonl(
        analysis_root / "behavior_stratified_core_profiles.jsonl"
    )
    require(
        len(candidates) == summary["recurrent_answer_head_count"],
        "candidate count drift",
    )
    require(
        len(trajectories) == len(cores),
        "trajectory/core coverage drift",
    )
    require(
        len(behavior_stratified) == len(cores),
        "behavior/core coverage drift",
    )
    for row in behavior_stratified:
        require(
            row["candidate_panel_pair_hit_count"]
            + row["candidate_panel_pair_miss_count"]
            == 360,
            f"{row['model']}/{row['event_id']} behavior coverage drift",
        )
    require(
        len(summary["threshold_sensitivity"]) == len(MODELS) * 16,
        "member threshold sensitivity grid drift",
    )
    require(
        len(summary["concentration_sensitivity"])
        == len(MODELS) * 9,
        "concentration sensitivity grid drift",
    )
    q_past_roles = (
        "fact_source",
        "fact_relation",
        "fact_target",
        "lexical_control",
        "query_anchor",
    )
    for model in MODELS:
        model_summary = summary["model_summaries"][model]
        model_scan = read_json(
            RESULT_ROOT / "scan" / model / "summary.json"
        )
        require(
            model_summary["event_role_profile_count"]
            == int(model_scan["event_count"]) * 2 * 7,
            f"{model} event-role profile coverage drift",
        )
        whole = model_summary["whole_component_role_recurrence"]
        for component in ("residual", "attention_output", "mlp_output"):
            for role in q_past_roles:
                require(
                    whole[component]["Q"][role][
                        "discovery_recurrent"
                    ]["count"] == 0,
                    f"{model}/{component}/{role} Q causal-order drift",
                )
    for row in candidates:
        discovery = row["splits"]["discovery"]
        expected_recurrent = bool(
            discovery["member_panel_count"] >= 4
            and len(discovery["families"]) >= 2
            and len(discovery["surfaces"]) >= 2
        )
        require(
            row["recurrent_discovery_member"] == expected_recurrent,
            f"{row['model']}/{row['event_id']} discovery leakage",
        )
        expected_lexical = bool(
            expected_recurrent
            and discovery["natural_member_panel_count"] >= 1
            and discovery["balanced_member_panel_count"] >= 1
        )
        require(
            row["surface_diversified_discovery_member"]
            == expected_lexical,
            f"{row['model']}/{row['event_id']} lexical flag drift",
        )
        expected_core = bool(
            discovery["concentrated_member_panel_count"] >= 3
            and len(discovery["concentrated_families"]) >= 2
            and len(discovery["concentrated_surfaces"]) >= 2
            and discovery["natural_concentrated_panel_count"] >= 1
            and discovery["balanced_concentrated_panel_count"] >= 1
        )
        require(
            row["concentrated_discovery_core"] == expected_core,
            f"{row['model']}/{row['event_id']} core flag drift",
        )
    for model in MODELS:
        for operation in ("F", "Q"):
            central = [
                row for row in summary["concentration_sensitivity"]
                if row["model"] == model
                and row["concentrated_percentile_threshold"] == 0.90
                and row["minimum_concentrated_panels"] == 3
            ]
            require(
                len(central) == 1,
                f"{model}/{operation} central concentration row drift",
            )
            require(
                central[0][f"concentrated_{operation}_core_count"]
                == summary["by_model_operation"][model][operation][
                    "concentrated_core_count"
                ],
                f"{model}/{operation} concentration arithmetic drift",
            )
    instrument = summary["instrument_precision_comparison"]
    require(
        instrument["scan_smoke"][
            "q_causal_prefix_maximum"
        ] > 0,
        "8bit contamination control did not register",
    )
    require(
        instrument["scan_smoke_bf16"][
            "q_causal_prefix_maximum"
        ] == 0,
        "BF16 causal prefix control drift",
    )
    key_files = [
        summary_path,
        analysis_root / "recurrent_answer_heads.jsonl",
        analysis_root / "concentrated_answer_head_cores.jsonl",
        analysis_root / "role_trajectories.jsonl",
        analysis_root / "behavior_stratified_core_profiles.jsonl",
    ]
    for model in MODELS:
        key_files.extend([
            analysis_root / model / "summary.json",
            analysis_root / model / "answer_head_candidates.jsonl",
            analysis_root / model / "role_trajectories.jsonl",
            analysis_root
            / model
            / "behavior_stratified_core_profiles.jsonl",
        ])
    return {
        "recurrent_answer_head_count": len(candidates),
        "concentrated_answer_head_core_count": len(cores),
        "trajectory_count": len(trajectories),
        "behavior_stratified_core_count": len(behavior_stratified),
        "discovery_selection_uses_confirmation": False,
        "discovery_selection_uses_behavior": False,
        "instrument_precision_comparison": instrument,
    }, key_files


def main() -> None:
    protocol = protocol_audit()
    scan, scan_files = scan_audit()
    analysis, analysis_files = analysis_audit()
    raw_tensor_suffixes = {".pt", ".pth", ".safetensors"}
    leaked = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in RESULT_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in raw_tensor_suffixes
    ]
    require(not leaked, f"raw tensor artifacts found: {leaked[:5]}")
    key_files = [
        RESULT_ROOT / "protocol" / "protocol.json",
        *scan_files,
        *analysis_files,
    ]
    hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in key_files
    }
    files = [path for path in RESULT_ROOT.rglob("*") if path.is_file()]
    result = {
        "schema_version": "phase1015_result_audit.v1",
        "phase": 1015,
        "passed": True,
        "protocol": protocol,
        "scan": scan,
        "analysis": analysis,
        "raw_hidden_tensor_artifact_count": len(leaked),
        "result_file_count": len(files),
        "result_byte_count": sum(path.stat().st_size for path in files),
        "key_file_sha256": hashes,
        "claim_limit": (
            "audit establishes protocol integrity, causal ordering, "
            "arithmetic reproducibility, and selection isolation; it "
            "does not establish mechanism validity"
        ),
    }
    write_json(RESULT_ROOT / "audit" / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
