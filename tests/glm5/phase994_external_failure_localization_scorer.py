from __future__ import annotations

import argparse
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
from typing import Any, Iterable, Mapping, Sequence

import phase983_cross_model_engine as engine


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = ROOT / "tests/glm5/result/phase994_external_failure_localization_protocol"
EXECUTION_ROOT = ROOT / "tests/glm5/result/phase994_external_failure_localization_execution"
ACTIVATION_PATH = PROTOCOL_ROOT / "activation.json"
TRUTH_PATH = PROTOCOL_ROOT / "dataset/private_truth.jsonl"
MANIFEST_PATH = PROTOCOL_ROOT / "dataset/public_manifest.jsonl"
SCORE_DIR = EXECUTION_ROOT / "scores"
SCORE_PATH = SCORE_DIR / "public_score.json"
ACCESS_PATH = SCORE_DIR / "truth_access_receipt.json"
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


def verify_activation() -> dict[str, Any]:
    activation = load_json(ACTIVATION_PATH)
    verify_self_hash(activation, "activation_sha256", "activation")
    require(activation.get("phase") == 994 and activation.get("scoring_authorized_after_all_receipts") is True,
            "scoring is not authorized")
    require(activation.get("internal_trace_authorized") is False
            and activation.get("phase992_holdout_authorized") is False, "scope authority drift")
    require(runtime_identity() == activation.get("runtime_identity"), "scoring runtime package identity drift")
    protocol_seal = activation["protocol"]
    protocol_path = PROTOCOL_ROOT / protocol_seal["path"]
    require(
        protocol_path.is_file() and protocol_path.stat().st_size == protocol_seal["bytes"]
        and sha256_file(protocol_path) == protocol_seal["sha256"],
        "protocol preregistration artifact drift",
    )
    preregistration = load_json(protocol_path)
    verify_self_hash(preregistration, "protocol_sha256", "preregistration")
    require(
        preregistration["protocol_sha256"] == activation["protocol_self_sha256"]
        and preregistration["thresholds"] == activation["thresholds"],
        "protocol/activation binding drift",
    )
    for role, seal in activation["source_seals"].items():
        path = ROOT / seal["path"]
        require(path.is_file() and path.stat().st_size == seal["bytes"]
                and sha256_file(path) == seal["sha256"], f"source drift: {role}")
    for name, seal in activation["dataset_seals"].items():
        path = PROTOCOL_ROOT / seal["path"]
        require(path.is_file() and path.stat().st_size == seal["bytes"]
                and sha256_file(path) == seal["sha256"], f"dataset drift: {name}")
    return activation


def verify_public_stage(activation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    stage = load_json(EXECUTION_ROOT / "public_raw_stage.json")
    verify_self_hash(stage, "stage_sha256", "public raw stage")
    require(stage.get("passed") is True and stage.get("truth_opened") is False
            and stage.get("activation_sha256") == activation["activation_sha256"]
            and tuple(stage.get("model_order", ())) == MODEL_ORDER,
            "public stage identity drift")
    previous: str | None = None
    raw_paths: dict[str, Path] = {}
    for model in MODEL_ORDER:
        receipt = load_json(EXECUTION_ROOT / f"receipts/public_{model}.json")
        cleanup = load_json(EXECUTION_ROOT / f"receipts/cleanup_public_{model}.json")
        verify_self_hash(receipt, "receipt_sha256", f"{model} execution receipt")
        verify_self_hash(cleanup, "receipt_sha256", f"{model} cleanup receipt")
        require(receipt.get("model") == model and receipt.get("scope") == "public"
                and receipt.get("run_id") == stage["run_id"]
                and receipt.get("activation_sha256") == activation["activation_sha256"]
                and receipt.get("previous_model_receipt_sha256") == previous
                and receipt.get("truth_opened") is False and receipt.get("row_count") == 3072,
                f"{model} execution receipt drift")
        require(cleanup.get("model") == model and cleanup.get("scope") == "public"
                and cleanup.get("run_id") == stage["run_id"]
                and cleanup.get("activation_sha256") == activation["activation_sha256"]
                and cleanup.get("worker_status_sha256") == receipt.get("worker_status_sha256")
                and cleanup.get("cleanup_pass") is True and cleanup.get("baseline_recovered") is True,
                f"{model} cleanup receipt drift")
        status_path = EXECUTION_ROOT / receipt["worker_status_artifact"]["path"]
        require(status_path.stat().st_size == receipt["worker_status_artifact"]["bytes"]
                and sha256_file(status_path) == receipt["worker_status_artifact"]["sha256"],
                f"{model} status artifact drift")
        status = load_json(status_path)
        verify_self_hash(status, "worker_status_sha256", f"{model} worker status")
        require(status.get("model") == model and status.get("scope") == "public"
                and status.get("run_id") == stage["run_id"]
                and status.get("activation_sha256") == activation["activation_sha256"]
                and status.get("worker_status_sha256") == receipt.get("worker_status_sha256")
                and status.get("truth_opened") is False and status.get("internal_trace_authorized") is False
                and status.get("model_released") is True and status.get("raw_row_count") == 3072,
                f"{model} worker status drift")
        identity = status.get("loaded_model_identity", {})
        quant = identity.get("loaded_quantization", {})
        verified_root = Path(str(status.get("model_artifact_verification", {}).get("resolved_root", ""))).resolve(
            strict=True
        )
        loaded_root = Path(str(identity.get("artifact_identity", {}).get("local_dir", ""))).resolve(strict=True)
        require(status.get("model_artifact_verification", {}).get("passed") is True
                and status.get("model_artifact_verification", {}).get("model") == model
                and identity.get("model_key") == model
                and identity.get("artifact_identity", {}).get("logical_name") == model
                and identity.get("loaded_attn_implementation") == "sdpa"
                and identity.get("cuda_only_no_cpu_or_disk_offload") is True
                and quant.get("load_in_8bit") is True
                and quant.get("non_quantized_dtype") == "torch.bfloat16"
                and verified_root == loaded_root
                and Path(str(status.get("loaded_artifact_resolved_root", ""))).resolve(strict=True) == loaded_root,
                f"{model} loaded identity drift")
        expected_manifest = activation["dataset_seals"]["public_manifest.jsonl"]
        for reported_manifest in (status.get("input_manifest", {}), receipt.get("input_manifest", {})):
            require(
                reported_manifest.get("bytes") == expected_manifest["bytes"]
                and reported_manifest.get("sha256") == expected_manifest["sha256"],
                f"{model} public manifest seal drift",
            )
        raw_path = EXECUTION_ROOT / receipt["raw_artifact"]["path"]
        require(raw_path.stat().st_size == receipt["raw_artifact"]["bytes"]
                and sha256_file(raw_path) == receipt["raw_artifact"]["sha256"],
                f"{model} raw artifact drift")
        require(stage["models"][model]["receipt_sha256"] == receipt["receipt_sha256"]
                and stage["models"][model]["cleanup_receipt_sha256"] == cleanup["receipt_sha256"],
                f"{model} stage binding drift")
        raw_paths[model] = raw_path
        previous = str(receipt["receipt_sha256"])
    return stage, raw_paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            require(isinstance(row, dict), f"non-object row in {path}")
            rows.append(row)
    return rows


def parse_text(text: str) -> dict[str, Any]:
    markers = [match.casefold() for match in MARKER_RE.findall(text)]
    scaffolds = [match.casefold() for match in SCAFFOLD_RE.findall(text)]
    distinct = sorted(set(markers), key=VALUES.index)
    if scaffolds:
        primary = scaffolds[-1]
        source = "last_complete_scaffold"
    elif len(distinct) == 1:
        primary = distinct[0]
        source = "sole_distinct_marker_fallback"
    else:
        primary = None
        source = "unparsed_no_scaffold_and_not_one_distinct_marker"
    strict = STRICT_RE.fullmatch(text)
    return {
        "primary_prediction": primary,
        "primary_source": source,
        "phase992_first_marker": markers[0] if markers else None,
        "last_marker_anywhere": markers[-1] if markers else None,
        "marker_occurrences": len(markers),
        "distinct_markers": distinct,
        "scaffold_occurrences": len(scaffolds),
        "distinct_scaffold_markers": sorted(set(scaffolds), key=VALUES.index),
        "strict_prediction": strict.group(1).casefold() if strict else None,
    }


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def summarize(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(cases)
    counts = {
        "primary_parsed": sum(case["primary_prediction"] is not None for case in cases),
        "primary_correct": sum(case["primary_correct"] for case in cases),
        "phase992_first_marker_parsed": sum(case["phase992_first_marker"] is not None for case in cases),
        "phase992_first_marker_correct": sum(case["phase992_first_marker_correct"] for case in cases),
        "strict_format": sum(case["strict_prediction"] is not None for case in cases),
        "strict_correct": sum(case["strict_correct"] for case in cases),
        "multiple_distinct_markers": sum(len(case["distinct_markers"]) > 1 for case in cases),
        "multiple_distinct_scaffold_markers": sum(len(case["distinct_scaffold_markers"]) > 1 for case in cases),
        "eos_seen_within_view": sum(case["eos_seen_within_view"] for case in cases),
        "view_budget_exhausted": sum(case["view_budget_exhausted"] for case in cases),
    }
    result: dict[str, Any] = {"denominator": denominator, **counts}
    for key, value in counts.items():
        result[key + "_percent"] = pct(int(value), denominator)
    result["primary_correct_given_parsed_percent_posthoc"] = pct(
        counts["primary_correct"], counts["primary_parsed"]
    )
    result["primary_source_counts"] = dict(sorted(Counter(case["primary_source"] for case in cases).items()))
    return result


def case_views(
    model: str, raw_path: Path, manifest: Mapping[str, dict], truth: Mapping[str, dict], activation_sha256: str,
    run_id: str, manifest_sha256: str,
) -> dict[int, dict[str, dict]]:
    bundle = engine._load_inspection_bundle(model)
    tokenizer = bundle.tokenizer
    expected_eos = sorted(int(value) for value in bundle.identity["eos_identity"]["effective_eos_token_ids"])
    views: dict[int, dict[str, dict]] = {24: {}, 64: {}}
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
                    f"{model}: raw record identity drift")
            seen.add(record_id)
            public = manifest[record_id]
            gold = truth[record_id]
            for key in (
                "paired_cell_id", "semantic_world_id", "world_ordinal", "nuisance_id", "split", "semantic_transform",
                "paraphrase_id", "fact_order_id", "horizon_id", "depth", "serialization",
            ):
                require(gold.get(key) == public.get(key), f"{model}/{record_id}: private/public field drift: {key}")
            require(gold.get("gold_value") in VALUES, f"{model}/{record_id}: truth value drift")
            prompt = str(public["prompt"])
            require(public.get("prompt_sha256") == sha256_bytes(prompt.encode("utf-8")),
                    f"{model}/{record_id}: manifest prompt hash drift")
            for key in (
                "paired_cell_id", "semantic_world_id", "world_ordinal", "nuisance_id", "split", "semantic_transform",
                "paraphrase_id", "fact_order_id", "horizon_id", "depth", "serialization", "prompt_sha256",
            ):
                require(row.get(key) == public.get(key), f"{model}/{record_id}: public field drift: {key}")
            require(row.get("model") == model and row.get("scope") == "public"
                    and row.get("activation_sha256") == activation_sha256
                    and row.get("run_id") == run_id
                    and row.get("input_manifest_sha256") == manifest_sha256,
                    f"{model}/{record_id}: raw scope drift")
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
                f"{model}/{record_id}: rendered input/token identity drift",
            )
            unit_rank = int(public["world_ordinal"]) * len(TRANSFORMS) + transform_index[
                str(public["semantic_transform"])
            ]
            expected_factor_batch = unit_rank // 8
            expected_batch_position = unit_rank % 8
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
                and row.get("batch_position") == expected_batch_position
                and row.get("batch_unit_members_sha256") == sha256_json(expected_units),
                f"{model}/{record_id}: matched public batch identity drift",
            )
            batch_groups[(
                str(public["serialization"]), str(public["depth"]), expected_factor_batch
            )].append((len(expected_ids), int(row.get("padded_input_width", -1))))
            suffix = [int(value) for value in row["generated_suffix_token_ids"]]
            require(bool(suffix) and len(suffix) <= 64, f"{model}/{record_id}: suffix length drift")
            require(sorted(int(value) for value in row.get("effective_eos_token_ids", [])) == expected_eos,
                    f"{model}/{record_id}: effective EOS identity drift")
            derived_first_eos = next((index for index, value in enumerate(suffix) if value in expected_eos), None)
            before_eos = suffix if derived_first_eos is None else suffix[:derived_first_eos]
            require(
                row.get("first_eos_index") == derived_first_eos
                and row.get("first_eos_token_id") == (
                    None if derived_first_eos is None else suffix[derived_first_eos]
                )
                and row.get("generated_token_ids_before_eos") == before_eos
                and row.get("eos_seen") is (derived_first_eos is not None)
                and row.get("budget_exhausted_64") is (derived_first_eos is None)
                and row.get("termination_reason") == (
                    "effective_eos" if derived_first_eos is not None else "max_new_tokens_64"
                )
                and (derived_first_eos is not None or len(suffix) == 64),
                f"{model}/{record_id}: generated trajectory/EOS drift",
            )
            decoded_full = tokenizer.decode(
                before_eos, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            require(row.get("generated_text") == decoded_full,
                    f"{model}/{record_id}: generated text/token drift")
            first_eos_index = derived_first_eos
            for budget in BUDGETS:
                ids = before_eos[:budget]
                text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                parsed = parse_text(text)
                eos_within = first_eos_index is not None and int(first_eos_index) < budget
                primary = parsed["primary_prediction"]
                first_marker = parsed["phase992_first_marker"]
                case = {
                    "record_id": record_id,
                    "paired_cell_id": public["paired_cell_id"],
                    "semantic_world_id": public["semantic_world_id"],
                    "world_ordinal": public["world_ordinal"],
                    "nuisance_id": public["nuisance_id"],
                    "split": public["split"],
                    "semantic_transform": public["semantic_transform"],
                    "paraphrase_id": public["paraphrase_id"],
                    "fact_order_id": public["fact_order_id"],
                    "horizon_id": public["horizon_id"],
                    "depth": public["depth"],
                    "serialization": public["serialization"],
                    "budget": budget,
                    "gold_value": gold["gold_value"],
                    **parsed,
                    "primary_correct": primary == gold["gold_value"],
                    "phase992_first_marker_correct": first_marker == gold["gold_value"],
                    "strict_correct": parsed["strict_prediction"] == gold["gold_value"],
                    "eos_seen_within_view": eos_within,
                    "view_budget_exhausted": not eos_within and len(row["generated_suffix_token_ids"]) >= budget,
                }
                views[budget][record_id] = case
    require(seen == set(manifest) == set(truth) and len(seen) == 3072, f"{model}: raw/public/truth set mismatch")
    require(len(batch_groups) == 6 * 64 and all(len(rows) == 8 for rows in batch_groups.values()),
            f"{model}: public batch population drift")
    for key, rows in batch_groups.items():
        widths = {reported for _, reported in rows}
        require(len(widths) == 1 and next(iter(widths)) == max(length for length, _ in rows),
                f"{model}/{key}: padded input width drift")
    for record_id in seen:
        short = views[24][record_id]
        long = views[64][record_id]
        if short["phase992_first_marker"] is not None:
            require(short["phase992_first_marker"] == long["phase992_first_marker"],
                    f"{model}/{record_id}: nested first-marker invariant failed")
    return views


def view_report(cases: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    require(len(cases) == 512, "serialization/depth/budget view must have 512 paired rows")
    overall = summarize(cases)
    by_split_transform: dict[str, Any] = {}
    for split in ("localization_a", "localization_b"):
        for transform in TRANSFORMS:
            subset = [case for case in cases if case["split"] == split and case["semantic_transform"] == transform]
            require(len(subset) == 64, "split/transform cell denominator drift")
            by_split_transform[f"{split}|{transform}"] = summarize(subset)
    by_transform: dict[str, Any] = {}
    for transform in TRANSFORMS:
        subset = [case for case in cases if case["semantic_transform"] == transform]
        require(len(subset) == 128, "transform denominator drift")
        by_transform[transform] = summarize(subset)
    by_world: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_world[str(case["semantic_world_id"])].append(case)
    require(len(by_world) == 128 and all(len(rows) == 4 for rows in by_world.values()), "world grouping drift")
    all_four = sum(all(row["primary_correct"] for row in rows) for rows in by_world.values())
    gate = (
        all(item["primary_correct"] >= thresholds["each_split_min_correct_count_of_64"]
            for item in by_split_transform.values())
        and all(item["primary_correct"] >= thresholds["each_semantic_transform_min_correct_count_of_128"]
                for item in by_transform.values())
        and all_four >= thresholds["all_four_transforms_correct_min_world_count_of_128"]
    )
    return {
        "overall": overall,
        "by_split_transform": by_split_transform,
        "by_transform": by_transform,
        "all_four_transforms_correct_worlds": all_four,
        "all_four_transforms_correct_worlds_percent": pct(all_four, 128),
        "gate_passed": gate,
    }


def outcome_pattern(values: Iterable[bool]) -> str:
    return "".join("1" if value else "0" for value in values)


def contrasts(views: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    budget_pairs = Counter()
    serialization_pairs = Counter()
    depth_patterns = Counter()
    by_identity: dict[tuple, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for budget in BUDGETS:
        for case in views[budget].values():
            key = (
                case["semantic_world_id"], case["semantic_transform"], case["depth"], budget,
            )
            by_identity[key][str(case["serialization"])] = case
    for budget in BUDGETS:
        for record_id, short_or_long in views[budget].items():
            if budget == 24:
                long = views[64][record_id]
                budget_pairs[outcome_pattern((short_or_long["primary_correct"], long["primary_correct"]))] += 1
        for key, pair in by_identity.items():
            if key[-1] == budget:
                require(set(pair) == set(SERIALIZATIONS), "serialization pair drift")
                serialization_pairs[f"budget{budget}:" + outcome_pattern(
                    (pair["raw_text"]["primary_correct"], pair["native_default_chat"]["primary_correct"])
                )] += 1
        for serialization in SERIALIZATIONS:
            grouped: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
            for case in views[budget].values():
                if case["serialization"] == serialization:
                    grouped[(str(case["semantic_world_id"]), str(case["semantic_transform"]))][str(case["depth"])] = bool(
                        case["primary_correct"]
                    )
            for depth_values in grouped.values():
                require(set(depth_values) == set(DEPTHS), "depth pair drift")
                depth_patterns[f"{serialization}|budget{budget}:" + outcome_pattern(
                    depth_values[depth] for depth in DEPTHS
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


def localization(gates: Mapping[str, Mapping[str, bool]]) -> list[dict[str, Any]]:
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


def score() -> dict[str, Any]:
    require(not SCORE_PATH.exists(), "Phase994 score already exists")
    activation = verify_activation()
    stage, raw_paths = verify_public_stage(activation)
    if ACCESS_PATH.exists():
        access = load_json(ACCESS_PATH)
        verify_self_hash(access, "access_sha256", "truth access receipt")
        require(
            access.get("run_id") == stage["run_id"]
            and access.get("activation_sha256") == activation["activation_sha256"]
            and access.get("public_raw_stage_sha256") == stage["stage_sha256"]
            and access.get("truth_artifact") == activation["private_truth_commitment"],
            "existing truth access receipt identity drift",
        )
    else:
        access = sealed({
            "schema_version": "phase994_truth_access_receipt.v1",
            "phase": 994,
            "experiment": "external_interface_depth_budget_localization",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "authorized_after_all_three_raw_status_and_cleanup_receipts",
            "run_id": stage["run_id"],
            "activation_sha256": activation["activation_sha256"],
            "public_raw_stage_sha256": stage["stage_sha256"],
            "truth_artifact": deepcopy(activation["private_truth_commitment"]),
            "phase992_holdout_opened": False,
            "internal_trace_authorized": False,
        }, "access_sha256")
        write_exclusive(ACCESS_PATH, canonical_bytes(access))

    truth_commitment = activation["private_truth_commitment"]
    require(
        TRUTH_PATH.stat().st_size == truth_commitment["bytes"]
        and sha256_file(TRUTH_PATH) == truth_commitment["sha256"],
        "private truth commitment drift after authorized access",
    )

    manifest_rows = read_jsonl(MANIFEST_PATH)
    truth_rows = read_jsonl(TRUTH_PATH)
    manifest = {row["record_id"]: row for row in manifest_rows}
    truth = {row["record_id"]: row for row in truth_rows}
    require(len(manifest) == len(truth) == 3072 and set(manifest) == set(truth), "manifest/truth identity drift")

    model_reports: dict[str, Any] = {}
    all_gates: dict[str, Any] = {}
    for model in MODEL_ORDER:
        views = case_views(
            model, raw_paths[model], manifest, truth, activation["activation_sha256"],
            stage["run_id"], activation["dataset_seals"]["public_manifest.jsonl"]["sha256"],
        )
        reports: dict[str, Any] = {}
        gates: dict[str, dict[str, bool]] = {str(budget): {} for budget in BUDGETS}
        for budget in BUDGETS:
            for serialization in SERIALIZATIONS:
                for depth in DEPTHS:
                    subset = [
                        case for case in views[budget].values()
                        if case["serialization"] == serialization and case["depth"] == depth
                    ]
                    key = f"{serialization}|{depth}|budget{budget}"
                    reports[key] = view_report(subset, activation["thresholds"])
                    gates[str(budget)][f"{serialization}|{depth}"] = bool(reports[key]["gate_passed"])
        model_reports[model] = {
            "model": model,
            "raw_sha256": sha256_file(raw_paths[model]),
            "views": reports,
            "gates": gates,
            "contrasts": contrasts(views),
            "localization_findings": localization(gates),
            "any_two_hop_external_candidate_at_64": any(
                gates["64"][f"{serialization}|two_hop"] for serialization in SERIALIZATIONS
            ),
            "scientific_scope": {
                "external_behavior_only": True,
                "internal_structure_evidence": False,
                "causal_mechanism_evidence": False,
            },
        }
        all_gates[model] = gates

    result = sealed({
        "schema_version": "phase994_external_failure_localization_score.v1",
        "phase": 994,
        "experiment": "external_interface_depth_budget_localization",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "locked_truth_released_external_localization_score",
        "run_id": stage["run_id"],
        "activation_sha256": activation["activation_sha256"],
        "public_raw_stage_sha256": stage["stage_sha256"],
        "truth_access_receipt_sha256": access["access_sha256"],
        "truth_artifact_sha256": sha256_file(TRUTH_PATH),
        "thresholds": deepcopy(activation["thresholds"]),
        "parser_contract": {
            "primary": (
                "last complete generated-only scaffold; otherwise sole distinct marker fallback; "
                "multiple distinct markers without scaffold are unparsed"
            ),
            "scaffold_regex": SCAFFOLD_RE.pattern,
            "phase992_first_marker_reported_separately": True,
        },
        "models": model_reports,
        "all_model_gates": all_gates,
        "any_model_native_or_raw_two_hop_external_candidate_at_64": any(
            report["any_two_hop_external_candidate_at_64"] for report in model_reports.values()
        ),
        "scientific_adjudication": {
            "phase994_data_truth_fully_open_after_scoring": True,
            "phase994_is_diagnostic_not_blind_holdout": True,
            "phase992_holdout_remains_unauthorized": True,
            "internal_trace_authorized": False,
            "causal_intervention_authorized": False,
            "mechanism_formula_authorized": False,
            "native_default_chat_is_a_bundle_not_a_single_variable": True,
            "budget24_is_a_nested_view_of_budget64": True,
            "right_censoring_beyond_64_unresolved_when_no_answer_within_64": True,
        },
    }, "score_sha256")
    write_exclusive(SCORE_PATH, canonical_bytes(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(score(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
