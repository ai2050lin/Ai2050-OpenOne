from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import phase983_cross_model_engine as engine


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = ROOT / "tests/glm5/result/phase994_external_failure_localization_protocol"
EXECUTION_ROOT = ROOT / "tests/glm5/result/phase994_external_failure_localization_execution"
ACTIVATION_PATH = PROTOCOL_ROOT / "activation.json"
MANIFEST_PATH = PROTOCOL_ROOT / "dataset/public_manifest.jsonl"
TRUTH_PATH = PROTOCOL_ROOT / "dataset/private_truth.jsonl"
SCORE_PATH = EXECUTION_ROOT / "scores/public_score.json"
ACCESS_PATH = EXECUTION_ROOT / "scores/truth_access_receipt.json"
AUDIT_PATH = EXECUTION_ROOT / "scores/public_independent_audit.json"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
SERIALIZATIONS = ("raw_text", "native_default_chat")
DEPTHS = ("copy_control", "one_hop", "two_hop")
TRANSFORMS = ("original", "value_swap", "binding_swap", "relation_swap")
BUDGETS = (24, 64)
VALUES = ("red", "blue", "green", "black")
MARKER_RE = re.compile(r"(?<![A-Za-z])(red|blue|green|black)(?![A-Za-z])", re.I)
SCAFFOLD_RE = re.compile(
    r"(?<![A-Za-z])The\s+retrieved\s+marker\s+is\s+(red|blue|green|black)\s*\.", re.I
)
STRICT_RE = re.compile(r"^\s*The\s+retrieved\s+marker\s+is\s+(red|blue|green|black)\s*\.\s*$", re.I)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def runtime_identity() -> dict[str, Any]:
    distributions = ("torch", "transformers", "bitsandbytes", "accelerate", "tokenizers")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "distributions": {name: importlib.metadata.version(name) for name in distributions},
    }


def verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = dict(value)
    expected = unsigned.pop(field, None)
    require(isinstance(expected, str) and expected == sha256_json(unsigned), f"{label} self-hash mismatch")


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[field] = sha256_json(result)
    return result


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if path.exists():
            path.unlink()
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            require(isinstance(value, dict), f"non-object row: {path}")
            rows.append(value)
    return rows


def independent_parse(text: str) -> dict[str, Any]:
    markers = [value.casefold() for value in MARKER_RE.findall(text)]
    scaffolds = [value.casefold() for value in SCAFFOLD_RE.findall(text)]
    distinct = set(markers)
    if scaffolds:
        primary = scaffolds[-1]
        source = "last_complete_scaffold"
    elif len(distinct) == 1:
        primary = next(iter(distinct))
        source = "sole_distinct_marker_fallback"
    else:
        primary = None
        source = "unparsed_no_scaffold_and_not_one_distinct_marker"
    strict = STRICT_RE.fullmatch(text)
    return {
        "primary": primary,
        "source": source,
        "first": markers[0] if markers else None,
        "strict": strict.group(1).casefold() if strict else None,
        "multiple": len(distinct) > 1,
        "multiple_scaffold": len(set(scaffolds)) > 1,
    }


def independent_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(cases)
    counts = {
        "primary_parsed": sum(case["parsed"] for case in cases),
        "primary_correct": sum(case["correct"] for case in cases),
        "phase992_first_marker_parsed": sum(case["first_parsed"] for case in cases),
        "phase992_first_marker_correct": sum(case["first_correct"] for case in cases),
        "strict_format": sum(case["strict"] for case in cases),
        "strict_correct": sum(case["strict_correct"] for case in cases),
        "multiple_distinct_markers": sum(case["multiple"] for case in cases),
        "multiple_distinct_scaffold_markers": sum(case["multiple_scaffold"] for case in cases),
        "eos_seen_within_view": sum(case["eos"] for case in cases),
        "view_budget_exhausted": sum(case["exhausted"] for case in cases),
    }
    result: dict[str, Any] = {"denominator": denominator, **counts}
    for key, value in counts.items():
        result[key + "_percent"] = 100.0 * int(value) / denominator if denominator else 0.0
    parsed = int(counts["primary_parsed"])
    result["primary_correct_given_parsed_percent_posthoc"] = (
        100.0 * int(counts["primary_correct"]) / parsed if parsed else 0.0
    )
    result["primary_source_counts"] = dict(sorted(Counter(case["source"] for case in cases).items()))
    return result


def outcome_pattern(values: Sequence[bool]) -> str:
    return "".join("1" if value else "0" for value in values)


def independent_contrasts(cases: Mapping[int, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    by_budget_record = {
        budget: {str(case["record_id"]): case for case in rows}
        for budget, rows in cases.items()
    }
    budget_pairs = Counter(
        outcome_pattern((by_budget_record[24][record_id]["correct"], by_budget_record[64][record_id]["correct"]))
        for record_id in by_budget_record[24]
    )
    serialization_pairs: Counter[str] = Counter()
    depth_patterns: Counter[str] = Counter()
    for budget in BUDGETS:
        paired_serialization: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
        paired_depth: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
        for case in cases[budget]:
            paired_serialization[(str(case["world"]), str(case["transform"]), str(case["depth"]))][
                str(case["serialization"])
            ] = bool(case["correct"])
            paired_depth[(str(case["world"]), str(case["transform"]), str(case["serialization"]))][
                str(case["depth"])
            ] = bool(case["correct"])
        for pair in paired_serialization.values():
            require(set(pair) == set(SERIALIZATIONS), "audit serialization pair drift")
            serialization_pairs[f"budget{budget}:" + outcome_pattern(tuple(pair[value] for value in SERIALIZATIONS))] += 1
        for (world, transform, serialization), values in paired_depth.items():
            del world, transform
            require(set(values) == set(DEPTHS), "audit depth triple drift")
            depth_patterns[f"{serialization}|budget{budget}:" + outcome_pattern(
                tuple(values[value] for value in DEPTHS)
            )] += 1
    return {
        "budget24_then_64_primary_correct_patterns": dict(sorted(budget_pairs.items())),
        "raw_then_native_primary_correct_patterns": dict(sorted(serialization_pairs.items())),
        "copy_onehop_twohop_primary_correct_patterns": dict(sorted(depth_patterns.items())),
        "pattern_order": {
            "budget": [24, 64],
            "serialization": list(SERIALIZATIONS),
            "depth": list(DEPTHS),
        },
    }


def independent_localization(gates: Mapping[str, Mapping[str, bool]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for depth in DEPTHS:
        raw = gates["64"][f"raw_text|{depth}"]
        native = gates["64"][f"native_default_chat|{depth}"]
        if native and not raw:
            findings.append({
                "label": "native_default_interface_bundle_association",
                "depth": depth,
                "not_a_single_chat_template_causal_effect": True,
            })
    for serialization in SERIALIZATIONS:
        copy_pass = gates["64"][f"{serialization}|copy_control"]
        one_pass = gates["64"][f"{serialization}|one_hop"]
        two_pass = gates["64"][f"{serialization}|two_hop"]
        if copy_pass and one_pass and not two_pass:
            findings.append({
                "label": "scaffolded_composition_boundary_candidate",
                "serialization": serialization,
                "not_internal_mechanism_evidence": True,
            })
        if not copy_pass:
            findings.append({
                "label": "copy_format_emission_gate_unqualified_within_64_tokens",
                "serialization": serialization,
                "right_censoring_beyond_64_unresolved": True,
            })
    for serialization in SERIALIZATIONS:
        for depth in DEPTHS:
            if gates["64"][f"{serialization}|{depth}"] and not gates["24"][f"{serialization}|{depth}"]:
                findings.append({
                    "label": "nested_budget_24_bottleneck_candidate",
                    "serialization": serialization,
                    "depth": depth,
                    "same_64_token_trajectory_not_independent_runs": True,
                })
    return findings


def recompute_model(
    model: str, raw_path: Path, manifest: Mapping[str, dict], truth: Mapping[str, dict],
    score_model: Mapping[str, Any], thresholds: Mapping[str, int], activation_sha256: str,
    run_id: str, manifest_sha256: str,
) -> dict[str, Any]:
    bundle = engine._load_inspection_bundle(model)
    tokenizer = bundle.tokenizer
    expected_eos = sorted(int(value) for value in bundle.identity["eos_identity"]["effective_eos_token_ids"])
    cases: dict[int, list[dict[str, Any]]] = {24: [], 64: []}
    seen: set[str] = set()
    batch_groups: dict[tuple[str, str, int], list[tuple[int, int]]] = defaultdict(list)
    serialization_index = {value: index for index, value in enumerate(SERIALIZATIONS)}
    depth_index = {value: index for index, value in enumerate(DEPTHS)}
    transform_index = {value: index for index, value in enumerate(TRANSFORMS)}
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = row["record_id"]
            require(record_id not in seen and record_id in manifest and record_id in truth,
                    f"{model}: record identity drift")
            seen.add(record_id)
            public = manifest[record_id]
            gold = truth[record_id]
            for field in (
                "paired_cell_id", "semantic_world_id", "world_ordinal", "nuisance_id", "split", "semantic_transform",
                "paraphrase_id", "fact_order_id", "horizon_id", "depth", "serialization",
            ):
                require(gold.get(field) == public.get(field), f"{model}: private/public field drift: {field}")
            require(gold.get("gold_value") in VALUES, f"{model}: truth value drift")
            prompt = str(public["prompt"])
            require(public.get("prompt_sha256") == sha256_bytes(prompt.encode("utf-8")),
                    f"{model}: manifest prompt hash drift")
            require(row.get("activation_sha256") == activation_sha256 and row.get("model") == model
                    and row.get("scope") == "public" and row.get("run_id") == run_id
                    and row.get("input_manifest_sha256") == manifest_sha256,
                    f"{model}: raw activation/model drift")
            for field in (
                "semantic_world_id", "semantic_transform", "split", "depth", "serialization", "nuisance_id",
            ):
                require(row.get(field) == public.get(field), f"{model}: public field drift: {field}")
            if public["serialization"] == "raw_text":
                rendered = prompt
                expected_ids = [int(value) for value in tokenizer(
                    rendered, add_special_tokens=False, return_attention_mask=False
                ).input_ids]
            else:
                native = engine.render_native_user(tokenizer, prompt)
                rendered = native.rendered_text
                expected_ids = list(native.input_ids)
            require(
                row.get("rendered_prompt_sha256") == sha256_bytes(rendered.encode("utf-8"))
                and row.get("input_token_ids") == expected_ids
                and row.get("input_token_ids_sha256") == sha256_json(expected_ids)
                and row.get("input_token_count") == len(expected_ids),
                f"{model}: rendered input/token identity drift",
            )
            unit_rank = int(public["world_ordinal"]) * len(TRANSFORMS) + transform_index[
                str(public["semantic_transform"])
            ]
            expected_factor_batch = unit_rank // 8
            expected_position = unit_rank % 8
            expected_execution_batch = (
                serialization_index[str(public["serialization"])] * len(DEPTHS) * 64
                + depth_index[str(public["depth"])] * 64
                + expected_factor_batch
            )
            expected_units = [
                [rank // len(TRANSFORMS), TRANSFORMS[rank % len(TRANSFORMS)]]
                for rank in range(expected_factor_batch * 8, expected_factor_batch * 8 + 8)
            ]
            require(
                row.get("factor_local_batch_index") == expected_factor_batch
                and row.get("execution_batch_index") == expected_execution_batch
                and row.get("batch_position") == expected_position
                and row.get("batch_unit_members_sha256") == sha256_json(expected_units),
                f"{model}: matched public batch identity drift",
            )
            batch_groups[(
                str(public["serialization"]), str(public["depth"]), expected_factor_batch
            )].append((len(expected_ids), int(row.get("padded_input_width", -1))))
            suffix = [int(value) for value in row["generated_suffix_token_ids"]]
            require(bool(suffix) and len(suffix) <= 64, f"{model}: suffix length drift")
            require(sorted(int(value) for value in row.get("effective_eos_token_ids", [])) == expected_eos,
                    f"{model}: effective EOS identity drift")
            eos_index = next((index for index, value in enumerate(suffix) if value in expected_eos), None)
            before = suffix if eos_index is None else suffix[:eos_index]
            require(
                row.get("first_eos_index") == eos_index
                and row.get("first_eos_token_id") == (None if eos_index is None else suffix[eos_index])
                and row.get("generated_token_ids_before_eos") == before
                and row.get("eos_seen") is (eos_index is not None)
                and row.get("budget_exhausted_64") is (eos_index is None)
                and row.get("termination_reason") == (
                    "effective_eos" if eos_index is not None else "max_new_tokens_64"
                )
                and (eos_index is not None or len(suffix) == 64),
                f"{model}: generated trajectory/EOS drift",
            )
            require(
                row.get("generated_text") == tokenizer.decode(
                    before, skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
                f"{model}: generated text/token drift",
            )
            first_by_budget: dict[int, str | None] = {}
            for budget in BUDGETS:
                text = tokenizer.decode(before[:budget], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                parsed = independent_parse(text)
                first_by_budget[budget] = parsed["first"]
                cases[budget].append({
                    "record_id": record_id,
                    "world": public["semantic_world_id"],
                    "transform": public["semantic_transform"],
                    "split": public["split"],
                    "depth": public["depth"],
                    "serialization": public["serialization"],
                    "correct": parsed["primary"] == gold["gold_value"],
                    "parsed": parsed["primary"] is not None,
                    "first_correct": parsed["first"] == gold["gold_value"],
                    "first_parsed": parsed["first"] is not None,
                    "strict": parsed["strict"] is not None,
                    "strict_correct": parsed["strict"] == gold["gold_value"],
                    "source": parsed["source"],
                    "multiple": parsed["multiple"],
                    "multiple_scaffold": parsed["multiple_scaffold"],
                    "eos": eos_index is not None and int(eos_index) < budget,
                    "exhausted": not (eos_index is not None and int(eos_index) < budget) and len(suffix) >= budget,
                })
            if first_by_budget[24] is not None:
                require(first_by_budget[24] == first_by_budget[64], f"{model}: nested first-marker invariant failed")
    require(seen == set(manifest) == set(truth) and len(seen) == 3072, f"{model}: identity set mismatch")
    require(len(batch_groups) == 6 * 64 and all(len(rows) == 8 for rows in batch_groups.values()),
            f"{model}: public batch population drift")
    for key, rows in batch_groups.items():
        widths = {reported for _, reported in rows}
        require(len(widths) == 1 and next(iter(widths)) == max(length for length, _ in rows),
                f"{model}/{key}: padded input width drift")

    checks: dict[str, bool] = {}
    recomputed_gates: dict[str, dict[str, bool]] = {str(budget): {} for budget in BUDGETS}
    for budget in BUDGETS:
        for serialization in SERIALIZATIONS:
            for depth in DEPTHS:
                subset = [
                    case for case in cases[budget]
                    if case["serialization"] == serialization and case["depth"] == depth
                ]
                require(len(subset) == 512, "view denominator drift")
                key = f"{serialization}|{depth}|budget{budget}"
                reported = score_model["views"][key]
                checks[f"{key}|overall"] = reported["overall"] == independent_summary(subset)
                split_transform_correct: dict[str, int] = {}
                for split in ("localization_a", "localization_b"):
                    for transform in TRANSFORMS:
                        cell = [case for case in subset if case["split"] == split and case["transform"] == transform]
                        require(len(cell) == 64, "split/transform denominator drift")
                        cell_key = f"{split}|{transform}"
                        split_transform_correct[cell_key] = sum(case["correct"] for case in cell)
                        checks[f"{key}|{cell_key}"] = (
                            reported["by_split_transform"][cell_key] == independent_summary(cell)
                        )
                transform_correct: dict[str, int] = {}
                for transform in TRANSFORMS:
                    cell = [case for case in subset if case["transform"] == transform]
                    transform_correct[transform] = sum(case["correct"] for case in cell)
                    checks[f"{key}|transform:{transform}"] = (
                        reported["by_transform"][transform] == independent_summary(cell)
                    )
                worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for case in subset:
                    worlds[str(case["world"])].append(case)
                all_four = sum(all(case["correct"] for case in rows) for rows in worlds.values())
                gate = (
                    all(value >= thresholds["each_split_min_correct_count_of_64"]
                        for value in split_transform_correct.values())
                    and all(value >= thresholds["each_semantic_transform_min_correct_count_of_128"]
                            for value in transform_correct.values())
                    and all_four >= thresholds["all_four_transforms_correct_min_world_count_of_128"]
                )
                checks[f"{key}|all_four"] = reported["all_four_transforms_correct_worlds"] == all_four
                checks[f"{key}|gate"] = reported["gate_passed"] == gate
                recomputed_gates[str(budget)][f"{serialization}|{depth}"] = gate
    checks["reported_gate_matrix"] = recomputed_gates == score_model["gates"]
    recomputed_contrasts = independent_contrasts(cases)
    recomputed_findings = independent_localization(recomputed_gates)
    candidate = any(recomputed_gates["64"][f"{serialization}|two_hop"] for serialization in SERIALIZATIONS)
    checks["reported_contrasts"] = recomputed_contrasts == score_model["contrasts"]
    checks["reported_localization_findings"] = recomputed_findings == score_model["localization_findings"]
    checks["reported_two_hop_candidate"] = candidate == score_model["any_two_hop_external_candidate_at_64"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "recomputed_gates": recomputed_gates,
        "recomputed_contrasts": recomputed_contrasts,
        "recomputed_localization_findings": recomputed_findings,
        "recomputed_two_hop_candidate_at_64": candidate,
    }


def audit() -> dict[str, Any]:
    require(not AUDIT_PATH.exists(), "Phase994 independent audit already exists")
    activation = load_json(ACTIVATION_PATH)
    score = load_json(SCORE_PATH)
    access = load_json(ACCESS_PATH)
    stage = load_json(EXECUTION_ROOT / "public_raw_stage.json")
    verify_self_hash(activation, "activation_sha256", "activation")
    verify_self_hash(score, "score_sha256", "score")
    verify_self_hash(access, "access_sha256", "truth access")
    verify_self_hash(stage, "stage_sha256", "public stage")
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    global_checks: dict[str, bool] = {
        "activation_scope": activation.get("internal_trace_authorized") is False
        and activation.get("phase992_holdout_authorized") is False
        and runtime_identity() == activation.get("runtime_identity"),
        "stage_identity": stage.get("passed") is True and stage.get("truth_opened") is False
        and stage.get("activation_sha256") == activation["activation_sha256"]
        and tuple(stage.get("model_order", ())) == MODEL_ORDER,
        "score_identity": score.get("activation_sha256") == activation["activation_sha256"]
        and score.get("run_id") == stage["run_id"]
        and score.get("public_raw_stage_sha256") == stage["stage_sha256"]
        and score.get("truth_access_receipt_sha256") == access["access_sha256"]
        and set(score.get("models", {})) == set(MODEL_ORDER),
        "truth_access_after_stage": access.get("public_raw_stage_sha256") == stage["stage_sha256"]
        and access.get("run_id") == stage["run_id"]
        and access.get("activation_sha256") == activation["activation_sha256"]
        and access.get("truth_artifact") == activation.get("private_truth_commitment"),
        "truth_hash": score.get("truth_artifact_sha256") == sha256_file(TRUTH_PATH)
        and TRUTH_PATH.stat().st_size == activation["private_truth_commitment"]["bytes"]
        and sha256_file(TRUTH_PATH) == activation["private_truth_commitment"]["sha256"],
        "source_audit_does_not_import_scorer": "phase994_external_failure_localization_scorer" not in imported_modules,
    }
    for role, seal in activation["source_seals"].items():
        path = ROOT / seal["path"]
        global_checks[f"source:{role}"] = (
            path.is_file() and path.stat().st_size == seal["bytes"] and sha256_file(path) == seal["sha256"]
        )
    protocol_seal = activation["protocol"]
    protocol_path = PROTOCOL_ROOT / protocol_seal["path"]
    preregistration = load_json(protocol_path)
    verify_self_hash(preregistration, "protocol_sha256", "preregistration")
    global_checks["protocol_binding"] = (
        protocol_path.stat().st_size == protocol_seal["bytes"]
        and sha256_file(protocol_path) == protocol_seal["sha256"]
        and preregistration["protocol_sha256"] == activation["protocol_self_sha256"]
        and preregistration["thresholds"] == activation["thresholds"]
    )
    for name, seal in activation["dataset_seals"].items():
        path = PROTOCOL_ROOT / seal["path"]
        global_checks[f"dataset:{name}"] = (
            path.is_file() and path.stat().st_size == seal["bytes"] and sha256_file(path) == seal["sha256"]
        )
    manifest_rows = read_jsonl(MANIFEST_PATH)
    truth_rows = read_jsonl(TRUTH_PATH)
    manifest = {row["record_id"]: row for row in manifest_rows}
    truth = {row["record_id"]: row for row in truth_rows}
    global_checks["manifest_truth_identity"] = len(manifest) == len(truth) == 3072 and set(manifest) == set(truth)

    model_reports: dict[str, Any] = {}
    previous: str | None = None
    public_manifest_seal = activation["dataset_seals"]["public_manifest.jsonl"]
    for model in MODEL_ORDER:
        receipt = load_json(EXECUTION_ROOT / f"receipts/public_{model}.json")
        cleanup = load_json(EXECUTION_ROOT / f"receipts/cleanup_public_{model}.json")
        verify_self_hash(receipt, "receipt_sha256", f"{model} receipt")
        verify_self_hash(cleanup, "receipt_sha256", f"{model} cleanup")
        status_path = EXECUTION_ROOT / receipt["worker_status_artifact"]["path"]
        status = load_json(status_path)
        verify_self_hash(status, "worker_status_sha256", f"{model} worker status")
        raw_path = EXECUTION_ROOT / receipt["raw_artifact"]["path"]
        identity = status.get("loaded_model_identity", {})
        verified_root = Path(str(status.get("model_artifact_verification", {}).get("resolved_root", ""))).resolve(
            strict=True
        )
        loaded_root = Path(str(identity.get("artifact_identity", {}).get("local_dir", ""))).resolve(strict=True)
        receipt_ok = (
            receipt.get("model") == model and receipt.get("scope") == "public"
            and receipt.get("run_id") == stage["run_id"]
            and receipt.get("activation_sha256") == activation["activation_sha256"]
            and receipt.get("previous_model_receipt_sha256") == previous
            and receipt.get("truth_opened") is False
            and receipt.get("worker_status_sha256") == status.get("worker_status_sha256")
            and status_path.stat().st_size == receipt["worker_status_artifact"]["bytes"]
            and sha256_file(status_path) == receipt["worker_status_artifact"]["sha256"]
            and status.get("model") == model and status.get("scope") == "public"
            and status.get("run_id") == stage["run_id"] and status.get("truth_opened") is False
            and status.get("internal_trace_authorized") is False and status.get("model_released") is True
            and status.get("raw_row_count") == 3072 and receipt.get("row_count") == 3072
            and status.get("model_artifact_verification", {}).get("passed") is True
            and status.get("model_artifact_verification", {}).get("model") == model
            and identity.get("loaded_attn_implementation") == "sdpa"
            and identity.get("cuda_only_no_cpu_or_disk_offload") is True
            and identity.get("loaded_quantization", {}).get("load_in_8bit") is True
            and identity.get("loaded_quantization", {}).get("non_quantized_dtype") == "torch.bfloat16"
            and status.get("input_manifest", {}).get("sha256") == public_manifest_seal["sha256"]
            and status.get("input_manifest", {}).get("bytes") == public_manifest_seal["bytes"]
            and receipt.get("input_manifest", {}).get("sha256") == public_manifest_seal["sha256"]
            and receipt.get("input_manifest", {}).get("bytes") == public_manifest_seal["bytes"]
            and verified_root == loaded_root
            and Path(str(status.get("loaded_artifact_resolved_root", ""))).resolve(strict=True) == loaded_root
            and cleanup.get("model") == model and cleanup.get("scope") == "public"
            and cleanup.get("run_id") == stage["run_id"]
            and cleanup.get("worker_status_sha256") == status.get("worker_status_sha256")
            and cleanup.get("cleanup_pass") is True
            and cleanup.get("baseline_recovered") is True
            and raw_path.stat().st_size == receipt["raw_artifact"]["bytes"]
            and sha256_file(raw_path) == receipt["raw_artifact"]["sha256"]
            and stage["models"][model]["receipt_sha256"] == receipt["receipt_sha256"]
            and stage["models"][model]["cleanup_receipt_sha256"] == cleanup["receipt_sha256"]
        )
        report = recompute_model(
            model, raw_path, manifest, truth, score["models"][model], activation["thresholds"],
            activation["activation_sha256"], stage["run_id"], public_manifest_seal["sha256"],
        )
        report["receipt_chain_passed"] = receipt_ok
        report["passed"] = report["passed"] and receipt_ok
        model_reports[model] = report
        previous = str(receipt["receipt_sha256"])
    global_checks["score_thresholds"] = score.get("thresholds") == activation["thresholds"]
    global_checks["top_level_gate_matrix"] = score.get("all_model_gates") == {
        model: report["recomputed_gates"] for model, report in model_reports.items()
    }
    global_checks["top_level_two_hop_candidate"] = score.get(
        "any_model_native_or_raw_two_hop_external_candidate_at_64"
    ) == any(report["recomputed_two_hop_candidate_at_64"] for report in model_reports.values())
    global_checks["all_models"] = all(report["passed"] for report in model_reports.values())

    result = sealed({
        "schema_version": "phase994_external_failure_localization_independent_audit.v1",
        "phase": 994,
        "experiment": "external_interface_depth_budget_localization",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "independent_raw_truth_parser_integer_gate_recomputation",
        "activation_sha256": activation["activation_sha256"],
        "score_sha256": score["score_sha256"],
        "score_file_sha256": sha256_file(SCORE_PATH),
        "truth_access_receipt_sha256": access["access_sha256"],
        "checks": global_checks,
        "models": model_reports,
        "passed": all(global_checks.values()),
        "independence": {
            "imports_scorer": False,
            "calls_scorer_parser": False,
            "all_rows_redecoded_from_token_ids": True,
            "all_integer_gates_recomputed": True,
        },
        "scientific_scope": {
            "external_failure_localization_only": True,
            "phase992_holdout_authorized": False,
            "internal_structure_evidence": False,
            "causal_mechanism_evidence": False,
        },
    }, "audit_sha256")
    write_exclusive(AUDIT_PATH, canonical_bytes(result))
    if not result["passed"]:
        raise RuntimeError("Phase994 independent audit failed")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(audit(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
