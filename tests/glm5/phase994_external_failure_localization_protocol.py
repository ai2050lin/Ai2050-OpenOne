from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import phase983_cross_model_engine as engine
import phase991_gpu_admission_core as p991


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase994_external_failure_localization_protocol"
EXECUTION_ROOT = ROOT / "tests/glm5/result/phase994_external_failure_localization_execution"
PHASE991_OUT = ROOT / "tests/glm5/result/phase991_delayed_binding_gpu_admission"
PHASE992_PROTOCOL = ROOT / "tests/glm5/result/phase992_delayed_binding_behavior_protocol"
PHASE993_OUT = ROOT / "tests/glm5/result/phase993_delayed_binding_emission_topology"
PHASE991_EXTENSION = PHASE991_OUT / "extension_dataset.json"
PHASE992_ACTIVATION = PHASE992_PROTOCOL / "activation.json"
PHASE993_RESULT = PHASE993_OUT / "phase993_emission_topology.json"
PHASE993_AUDIT = PHASE993_OUT / "phase993_emission_topology_audit.json"

PHASE = 994
EXPERIMENT = "external_interface_depth_budget_localization"
SCHEMA = "phase994_external_failure_localization_protocol.v1"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
SERIALIZATIONS = ("raw_text", "native_default_chat")
DEPTHS = ("copy_control", "one_hop", "two_hop")
SEMANTIC_TRANSFORMS = tuple(p991.p990_core.SEMANTIC_TRANSFORMS)
VALUES = ("red", "blue", "green", "black")
MAX_NEW_TOKENS = 64
DERIVED_BUDGETS = (24, 64)
BATCH_SIZE = 8
SEED_SEARCH_START = 0x0000_0000_03E2_0080
SELECTED_SEED = 0x0000_0000_03E2_0081
RESPONSE_INSTRUCTION = "Reply with one short sentence beginning \"The retrieved marker is\"."
SCAFFOLD_PATTERN = r"(?i)(?<![A-Za-z])The\s+retrieved\s+marker\s+is\s+(red|blue|green|black)\s*\."
NUISANCE_CELLS = (
    ("standard", "order_a", "near"),
    ("standard", "order_a", "far"),
    ("standard", "order_b", "near"),
    ("standard", "order_b", "far"),
    ("paraphrase", "order_a", "near"),
    ("paraphrase", "order_a", "far"),
    ("paraphrase", "order_b", "near"),
    ("paraphrase", "order_b", "far"),
)
THRESHOLDS = {
    "each_split_min_correct_count_of_64": 58,
    "each_semantic_transform_min_correct_count_of_128": 116,
    "all_four_transforms_correct_min_world_count_of_128": 109,
    "gate_budget_tokens": 64,
}
SOURCE_PATHS = {
    "protocol": ROOT / "tests/glm5/phase994_external_failure_localization_protocol.py",
    "runner": ROOT / "tests/glm5/phase994_external_failure_localization_runner.py",
    "scorer": ROOT / "tests/glm5/phase994_external_failure_localization_scorer.py",
    "audit": ROOT / "tests/glm5/phase994_external_failure_localization_audit.py",
    "phase983_engine": ROOT / "tests/glm5/phase983_cross_model_engine.py",
    "phase991_generator": ROOT / "tests/glm5/phase991_gpu_admission_core.py",
    "phase990_core_dependency": ROOT / "tests/glm5/phase990_binding_core.py",
    "phase990_dataset_dependency": ROOT / "tests/glm5/phase990_binding_dataset.py",
    "phase992_runner_helpers": ROOT / "tests/glm5/phase992_delayed_binding_runner.py",
    "model_registry": ROOT / "tests/gpt5/model_registry.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


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


def runtime_identity() -> dict[str, Any]:
    distributions = ("torch", "transformers", "bitsandbytes", "accelerate", "tokenizers")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "distributions": {name: importlib.metadata.version(name) for name in distributions},
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON document is not an object: {path}")
    return value


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    require(field not in result, f"self-hash field already present: {field}")
    result[field] = sha256_json(result)
    return result


def file_seal(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(base)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def opaque(prefix: str, *parts: object) -> str:
    return prefix + sha256_json(list(parts))[:32]


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(dict(row)) for row in rows)


def combined_historical_corpus() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    phase990 = load_json(p991.PHASE990_DATASET)
    phase991 = load_json(PHASE991_EXTENSION)
    merged = deepcopy(phase990)
    merged["worlds"] = list(phase990["worlds"]) + list(phase991["worlds"])
    merged["records"] = list(phase990["records"]) + list(phase991["records"])
    return phase990, phase991, merged


def three_layer_sets(dataset: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    records = list(dataset["records"])
    return (
        {str(row["slot_canonical_semantic_sha256"]) for row in records},
        {str(row["observable_semantic_variant_sha256"]) for row in records},
        {sha256_bytes(p991.normalized_prompt(str(row["prompt"])).encode("utf-8")) for row in records},
    )


def seed_selection() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    phase990, phase991, merged = combined_historical_corpus()
    old_sets = three_layer_sets(merged)
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for seed in range(SEED_SEARCH_START, SELECTED_SEED + 1):
        candidate = p991.generate_extension(seed, merged)
        candidate_sets = three_layer_sets(candidate)
        overlaps = {
            "abstract_semantic": len(candidate_sets[0] & old_sets[0]),
            "observable_semantic": len(candidate_sets[1] & old_sets[1]),
            "normalized_prompt": len(candidate_sets[2] & old_sets[2]),
        }
        passed = all(value == 0 for value in overlaps.values())
        attempts.append({
            "seed_uint64": seed,
            "seed_hex": f"0x{seed:016X}",
            "overlap_counts": overlaps,
            "passed": passed,
            "payload_sha256": candidate["extension_payload_sha256"],
        })
        if passed:
            selected = candidate
            break
    require(selected is not None, "no seed passed the frozen sequential overlap search")
    require(int(selected["generator"]["seed"]) == SELECTED_SEED, "selected seed drift")
    rebuilt = p991.generate_extension(SELECTED_SEED, merged)
    require(rebuilt["extension_payload_sha256"] == selected["extension_payload_sha256"],
            "selected seed exact regeneration drift")
    audit = {
        "selection_rule": "first sequential uint64 seed with zero overlap at all three registered layers",
        "search_start": SEED_SEARCH_START,
        "selected_seed": SELECTED_SEED,
        "outcome_independent_selection": True,
        "model_outputs_read": False,
        "gold_accuracy_used": False,
        "same_seed_exact_regeneration": True,
        "attempts": attempts,
        "historical_counts": {
            "phase990_worlds": len(phase990["worlds"]),
            "phase991_extension_worlds": len(phase991["worlds"]),
            "combined_worlds": len(merged["worlds"]),
        },
    }
    return selected, audit, merged


def replace_question_and_instruction(source_prompt: str, source: Mapping[str, Any], depth: str) -> str:
    lines = source_prompt.splitlines()
    require(len(lines) >= 3 and lines[-2].startswith("Question: "), "source prompt tail drift")
    require(lines[-1].startswith("Answer in one short sentence."), "source answer instruction drift")
    registry = lines[:-2]
    gold = source["gold"]
    state = source["semantic_state"]
    relation = str(state["query"]["relation"])
    entity = str(state["query"]["entity"])
    if depth == "copy_control":
        start_item = str(gold["answer_value"])
        owner_step = "KEEP"
        attribute_step = "KEEP"
    elif depth == "one_hop":
        start_item = str(gold["answer_object"])
        owner_step = "KEEP"
        attribute_step = "FOLLOW"
    elif depth == "two_hop":
        start_item = entity
        owner_step = "FOLLOW"
        attribute_step = "FOLLOW"
    else:
        raise RuntimeError(f"unknown depth: {depth}")
    padding_notes = 1
    if entity in {"Borin", "Celia", "Darin", "Elara", "Faron"} and depth in {"copy_control", "one_hop"}:
        padding_notes = 2
    program = [
        "Follow this fixed lookup program.",
        f"Start item: {start_item}.",
        f"Owner step: {owner_step}.",
        f"Attribute step for relation {relation}: {attribute_step}.",
        "FOLLOW means replace the current item using the matching registry statement.",
        "KEEP means leave the current item unchanged.",
        "Neutral padding: " + " ".join(["note"] * padding_notes) + ".",
        RESPONSE_INSTRUCTION,
        "Answer:",
    ]
    return "\n".join([*registry, *program])


def nuisance_index(world: Mapping[str, Any]) -> int:
    query = int(world["base_query_entity_slot"])
    relation = int(world["base_query_relation_slot"])
    repetition = int(world["local_rep"])
    q0, q1 = query & 1, (query >> 1) & 1
    s0, s1 = repetition & 1, (repetition >> 1) & 1
    n2 = q0 ^ s0
    n1 = q1 ^ s1
    n0 = q0 ^ relation
    return 4 * n2 + 2 * n1 + n0


def build_rows(
    dataset: Mapping[str, Any], historical: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_world: dict[str, dict[str, dict[str, Any]]] = {}
    for source in dataset["records"]:
        world_id = str(source["semantic_world_id"])
        by_world.setdefault(world_id, {})[str(source["variant_id"])] = source
    require(len(by_world) == 128, "fresh world count drift")
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    world_ids = [str(world["semantic_world_id"]) for world in dataset["worlds"]]
    worlds_by_id = {str(world["semantic_world_id"]): world for world in dataset["worlds"]}
    four_distinct_gold_worlds = 0
    for ordinal, source_world_id in enumerate(world_ids):
        nuisance_id = nuisance_index(worlds_by_id[source_world_id])
        paraphrase, fact_order, horizon = NUISANCE_CELLS[nuisance_id]
        split = "localization_a" if ordinal < 64 else "localization_b"
        phase994_world_id = opaque("p994_w_", SELECTED_SEED, ordinal, source_world_id)
        selected_sources: dict[str, dict[str, Any]] = {}
        for semantic in SEMANTIC_TRANSFORMS:
            variant = p991.p990_data.variant_id(semantic, paraphrase, fact_order, horizon)
            selected_sources[semantic] = by_world[source_world_id][variant]
        if len({str(source["gold"]["answer_value"]) for source in selected_sources.values()}) == 4:
            four_distinct_gold_worlds += 1
        for semantic in SEMANTIC_TRANSFORMS:
            source = selected_sources[semantic]
            for depth in DEPTHS:
                prompt = replace_question_and_instruction(str(source["prompt"]), source, depth)
                pair_id = opaque("p994_pair_", phase994_world_id, semantic, depth)
                for serialization in SERIALIZATIONS:
                    record_id = opaque(
                        "p994_i_", phase994_world_id, semantic, paraphrase, fact_order,
                        horizon, depth, serialization,
                    )
                    public.append({
                        "schema_version": "phase994_public_prompt.v1",
                        "phase": PHASE,
                        "experiment": EXPERIMENT,
                        "record_id": record_id,
                        "paired_cell_id": pair_id,
                        "semantic_world_id": phase994_world_id,
                        "world_ordinal": ordinal,
                        "nuisance_id": nuisance_id,
                        "split": split,
                        "semantic_transform": semantic,
                        "paraphrase_id": paraphrase,
                        "fact_order_id": fact_order,
                        "horizon_id": horizon,
                        "depth": depth,
                        "serialization": serialization,
                        "prompt": prompt,
                        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    })
                    truth.append({
                        "schema_version": "phase994_private_truth.v1",
                        "record_id": record_id,
                        "paired_cell_id": pair_id,
                        "semantic_world_id": phase994_world_id,
                        "world_ordinal": ordinal,
                        "nuisance_id": nuisance_id,
                        "split": split,
                        "semantic_transform": semantic,
                        "paraphrase_id": paraphrase,
                        "fact_order_id": fact_order,
                        "horizon_id": horizon,
                        "depth": depth,
                        "serialization": serialization,
                        "gold_object": source["gold"]["answer_object"],
                        "gold_value": source["gold"]["answer_value"],
                        "query_entity": source["semantic_state"]["query"]["entity"],
                        "query_relation": source["semantic_state"]["query"]["relation"],
                        "source_record_id": source["record_id"],
                        "source_abstract_sha256": source["slot_canonical_semantic_sha256"],
                        "source_observable_sha256": source["observable_semantic_variant_sha256"],
                    })
    require(len(public) == 3072 and len(truth) == 3072, "factorial row count drift")
    require(four_distinct_gold_worlds == 128, "four-transform gold distinctness drift")
    require(len({row["record_id"] for row in public}) == 3072, "public record IDs are not unique")
    require([row["record_id"] for row in public] == [row["record_id"] for row in truth], "truth order drift")
    identity_fields = (
        "record_id", "paired_cell_id", "semantic_world_id", "world_ordinal", "nuisance_id", "split",
        "semantic_transform", "paraphrase_id", "fact_order_id", "horizon_id", "depth", "serialization",
    )
    require(
        all(all(left[field] == right[field] for field in identity_fields) for left, right in zip(public, truth, strict=True)),
        "public/private identity field drift",
    )
    require(all(row["gold_value"] in VALUES for row in truth), "private truth value outside frozen marker set")
    counts: dict[str, int] = {}
    for serialization in SERIALIZATIONS:
        for depth in DEPTHS:
            counts[f"{serialization}|{depth}"] = sum(
                row["serialization"] == serialization and row["depth"] == depth for row in public
            )
    require(set(counts.values()) == {512}, "factor cell count drift")
    historical_prompts = {
        sha256_bytes(p991.normalized_prompt(str(row["prompt"])).encode("utf-8"))
        for row in historical["records"]
    }
    new_prompt_hashes = {
        sha256_bytes(p991.normalized_prompt(str(row["prompt"])).encode("utf-8"))
        for row in public
    }
    require(len(new_prompt_hashes) == 1536 and not (new_prompt_hashes & historical_prompts),
            "constructed prompt uniqueness/overlap drift")
    nuisance_counts = Counter(int(row["nuisance_id"]) for row in public if row["serialization"] == "raw_text"
                              and row["depth"] == "two_hop" and row["semantic_transform"] == "original")
    require(nuisance_counts == Counter({index: 16 for index in range(8)}), "nuisance balance drift")
    transform_gold_counts = {
        semantic: dict(sorted(Counter(
            row["gold_value"] for row in truth
            if row["serialization"] == "raw_text" and row["depth"] == "two_hop"
            and row["semantic_transform"] == semantic
        ).items()))
        for semantic in SEMANTIC_TRANSFORMS
    }
    audit = {
        "worlds": 128,
        "independent_unit": "semantic_world_id",
        "rows": 3072,
        "paired_rows_are_not_independent": True,
        "paired_cell_count": len({row["paired_cell_id"] for row in public}),
        "rows_per_paired_cell": 2,
        "rows_per_serialization_depth_cell": counts,
        "splits": {
            split: len({row["semantic_world_id"] for row in public if row["split"] == split})
            for split in ("localization_a", "localization_b")
        },
        "semantic_transforms": list(SEMANTIC_TRANSFORMS),
        "worlds_with_four_distinct_transform_gold_values": four_distinct_gold_worlds,
        "gold_value_counts_by_transform": transform_gold_counts,
        "cross_transform_accuracy_comparisons_are_label_mix_confounded": True,
        "nuisance_schedule": [list(cell) for cell in NUISANCE_CELLS],
        "nuisance_assignment_formula": "n2=q0 XOR s0; n1=q1 XOR s1; n0=q0 XOR relation; id=4*n2+2*n1+n0",
        "nuisance_cell_repeats": 16,
        "constructed_normalized_prompt_count_before_serialization_duplication": len(new_prompt_hashes),
        "constructed_prompt_overlap_with_phase990_plus_phase991": 0,
        "copy_control_start_item_is_answer_value": True,
        "one_hop_start_item_is_resolved_object": True,
        "two_hop_start_item_is_query_entity": True,
        "fixed_lookup_program_scaffold": True,
        "depth_ladder_is_scaffolding_not_a_pure_internal_depth_intervention": True,
        "neutral_padding_rule": (
            "one line with one note token everywhere; copy/one-hop use two note tokens only when query entity is "
            "Borin,Celia,Darin,Elara,or Faron"
        ),
        "neutral_padding_rule_depends_on_query_entity_not_answer": True,
        "same_registry_body_within_source_record_across_depth": True,
        "same_user_text_across_serializations": True,
    }
    return public, truth, audit


def tokenizer_precheck(public: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for model in MODEL_ORDER:
        bundle = engine._load_inspection_bundle(model)
        tokenizer = bundle.tokenizer
        lengths: list[int] = []
        rendered_hashes: list[str] = []
        length_by_case: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
        for row in public:
            prompt = str(row["prompt"])
            if row["serialization"] == "raw_text":
                rendered = prompt
            else:
                prefix = engine.render_native_user(tokenizer, prompt)
                rendered = prefix.rendered_text
                rendered_hashes.append(prefix.rendered_sha256)
            ids = list(tokenizer(rendered, add_special_tokens=False, return_attention_mask=False).input_ids)
            require(ids, f"{model}: empty rendered prompt")
            lengths.append(len(ids))
            length_by_case[(str(row["semantic_world_id"]), str(row["semantic_transform"]),
                            str(row["serialization"]))][str(row["depth"])] = len(ids)
        require(len(length_by_case) == 1024 and all(
            set(depths) == set(DEPTHS) and len(set(depths.values())) == 1 for depths in length_by_case.values()
        ), f"{model}: depth token-length matching drift")
        max_positions = int(getattr(bundle.config, "max_position_embeddings", 0) or 0)
        require(not max_positions or max(lengths) + MAX_NEW_TOKENS <= max_positions, f"{model}: context exceeds model limit")
        report[model] = {
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_length": len(tokenizer),
            "chat_template_sha256": bundle.identity["chat_template_sha256"],
            "effective_eos_token_ids": bundle.identity["eos_identity"]["effective_eos_token_ids"],
            "rendered_prompt_token_min": min(lengths),
            "rendered_prompt_token_max": max(lengths),
            "native_render_count": len(rendered_hashes),
            "native_rendered_identity_sha256": sha256_json(rendered_hashes),
            "depth_triples_with_exact_equal_token_length": len(length_by_case),
            "model_weights_loaded": False,
        }
    return report


def source_seals() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, path in SOURCE_PATHS.items():
        require(path.is_file(), f"required source missing: {path}")
        result[role] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def write_package() -> dict[str, Any]:
    require(not OUT.exists(), f"protocol package already exists: {OUT}")
    require(not EXECUTION_ROOT.exists(), f"execution root already exists before freeze: {EXECUTION_ROOT}")
    phase992 = load_json(PHASE992_ACTIVATION)
    phase993 = load_json(PHASE993_RESULT)
    phase993_audit = load_json(PHASE993_AUDIT)
    require(phase993_audit.get("passed") is True, "Phase993 audit did not pass")
    require(phase993["scope_guards"]["sealed_holdout_opened"] is False, "Phase992 holdout was opened")
    dataset, seed_audit, merged = seed_selection()
    public, truth, dataset_audit = build_rows(dataset, merged)
    precheck = tokenizer_precheck(public)
    runtime = runtime_identity()
    engineering: list[dict[str, Any]] = []
    for serialization in SERIALIZATIONS:
        for depth in DEPTHS:
            discovery = next(
                row for row in public
                if row["serialization"] == serialization and row["depth"] == depth
                and row["split"] == "localization_a" and row["semantic_transform"] == "original"
            )
            confirmation = next(
                row for row in public
                if row["serialization"] == serialization and row["depth"] == depth
                and row["split"] == "localization_b" and row["semantic_transform"] == "value_swap"
            )
            engineering.extend((discovery, confirmation))
    engineering_ids = {row["record_id"] for row in engineering}
    for row in public:
        if row["record_id"] not in engineering_ids:
            engineering.append(row)
            engineering_ids.add(row["record_id"])
        if len(engineering) == 16:
            break
    require(len(engineering) == 16 and len(engineering_ids) == 16, "engineering coverage/batch drift")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".phase994_protocol_pending_", dir=OUT.parent))
    (pending / "dataset").mkdir()
    (pending / "dataset/public_manifest.jsonl").write_bytes(jsonl_bytes(public))
    (pending / "dataset/private_truth.jsonl").write_bytes(jsonl_bytes(truth))
    (pending / "dataset/engineering_manifest.jsonl").write_bytes(jsonl_bytes(engineering))

    sources = source_seals()
    prereg = sealed({
        "schema_version": SCHEMA,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "preregistered_external_failure_localization_before_any_phase994_model_run",
        "upstream": {
            "phase992_activation_sha256": phase992["activation_sha256"],
            "phase992_public_behavior_passed": False,
            "phase993_result_file_sha256": sha256_file(PHASE993_RESULT),
            "phase993_audit_file_sha256": sha256_file(PHASE993_AUDIT),
        },
        "seed_selection": seed_audit,
        "zero_overlap_scope": {
            "compared_against_entire_phase990_cpu_corpus_and_phase991_expanded_confirmation": True,
            "phase992_model_access_gate_changed": False,
            "phase992_holdout_model_access_performed": False,
            "uses_existing_immutable_not_blind_cpu_registry": True,
            "lexical_generalization_tested": False,
        },
        "dataset_audit": dataset_audit,
        "tokenizer_precheck": precheck,
        "runtime_identity": runtime,
        "factor_design": {
            "serializations": list(SERIALIZATIONS),
            "serialization_language": {
                "raw_text": "raw-text serialization",
                "native_default_chat": "model-native default chat-template serialization (no model-specific control passed)",
            },
            "not_a_chat_template_only_effect": True,
            "native_thinking_switch_used": False,
            "native_thinking_state_equated_across_models": False,
            "depths": list(DEPTHS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "derived_prefix_views": list(DERIVED_BUDGETS),
            "budget_24_is_nested_not_an_independent_generation": True,
            "greedy": True,
            "batch_size": BATCH_SIZE,
            "public_batches_match_world_transform_membership_across_depth_and_serialization": True,
            "response_instruction": RESPONSE_INSTRUCTION,
            "teacher_forced_forward_in_phase994": False,
        },
        "parser_contract": {
            "primary": (
                "last complete generated-only scaffold match before effective EOS; if no complete scaffold exists, "
                "fall back only when exactly one distinct marker occurs anywhere in generated text"
            ),
            "primary_regex": SCAFFOLD_PATTERN,
            "phase992_compatible_first_marker_reported_separately": True,
            "multiple_distinct_scaffold_markers_reported_separately": True,
            "multiple_distinct_markers_without_scaffold_are_primary_unparsed": True,
            "strict_format_reported_separately": True,
            "primary_parser_not_selected_after_results": True,
            "nested_budget_invariant_applies_to_phase992_first_marker_not_last_scaffold_primary": True,
        },
        "thresholds": THRESHOLDS,
        "truth_contract": {
            "locally_inspectable_immutable_not_blind": True,
            "runner_truth_access": False,
            "score_only_after_all_three_public_raw_status_and_cleanup_receipts": True,
            "existing_phase992_holdout_access_authorized": False,
        },
        "engineering_contract": {
            "rows": len(engineering),
            "exact_repeat_required_for_each_model": True,
            "must_pass_before_public": True,
        },
        "model_contract": {
            "order": list(MODEL_ORDER),
            "serial_only": True,
            "cuda_only": True,
            "int8_bitsandbytes": True,
            "non_quantized_dtype": "bfloat16",
            "attention": "sdpa",
            "full_phase991_model_file_hash_verification_before_each_load": True,
            "strict_cuda_cleanup_between_models": True,
        },
        "scientific_scope": {
            "diagnostic_external_behavior_only": True,
            "posthoc_phase993_motivated_but_phase994_preregistered_before_run": True,
            "all_phase994_worlds_are_failure_localization_discovery_not_holdout": True,
            "internal_trace_authorized": False,
            "causal_intervention_authorized": False,
            "mechanism_formula_authorized": False,
            "public_failure_does_not_prove_general_model_inability": True,
            "cross_transform_accuracy_differences_are_not_clean_causal_contrasts": True,
        },
        "source_seals": sources,
    }, "protocol_sha256")
    (pending / "protocol_preregistration.json").write_bytes(pretty_bytes(prereg))
    dataset_seals = {
        name: file_seal(pending / f"dataset/{name}", pending)
        for name in ("public_manifest.jsonl", "engineering_manifest.jsonl")
    }
    private_truth_commitment = file_seal(pending / "dataset/private_truth.jsonl", pending)
    phase991_anchors = deepcopy(phase992["phase991_anchors"])
    activation = sealed({
        "schema_version": "phase994_external_failure_localization_activation.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "created_at_utc": prereg["created_at_utc"],
        "role": "formal_gpu_execution_activation",
        "protocol": file_seal(pending / "protocol_preregistration.json", pending),
        "protocol_self_sha256": prereg["protocol_sha256"],
        "dataset_seals": dataset_seals,
        "private_truth_commitment": private_truth_commitment,
        "source_seals": sources,
        "phase991_anchors": phase991_anchors,
        "phase992_holdout_authorized": False,
        "execution_root": str(EXECUTION_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "formal_python": str(Path(os.environ.get("PHASE994_FORMAL_PYTHON", os.sys.executable)).resolve()),
        "runtime_identity": runtime,
        "model_order": list(MODEL_ORDER),
        "serializations": list(SERIALIZATIONS),
        "depths": list(DEPTHS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "derived_budgets": list(DERIVED_BUDGETS),
        "batch_size": BATCH_SIZE,
        "scaffold_pattern": SCAFFOLD_PATTERN,
        "thresholds": THRESHOLDS,
        "engineering_required": True,
        "gpu_execution_authorized": True,
        "scoring_authorized_after_all_receipts": True,
        "internal_trace_authorized": False,
        "causal_intervention_authorized": False,
        "mechanism_formula_authorized": False,
    }, "activation_sha256")
    (pending / "activation.json").write_bytes(pretty_bytes(activation))
    verification = {
        "schema_version": "phase994_protocol_self_test.v1",
        "passed": True,
        "protocol_sha256": prereg["protocol_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "selected_seed": SELECTED_SEED,
        "public_rows": len(public),
        "truth_rows": len(truth),
        "engineering_rows": len(engineering),
        "model_weights_loaded": False,
        "truth_opened_by_model_runner": False,
    }
    (pending / "protocol_self_test.json").write_bytes(pretty_bytes(verification))
    os.replace(pending, OUT)
    return verification


def verify_package() -> dict[str, Any]:
    activation = load_json(OUT / "activation.json")
    unsigned = dict(activation)
    observed = unsigned.pop("activation_sha256")
    require(observed == sha256_json(unsigned), "activation self-hash mismatch")
    protocol_seal = activation["protocol"]
    protocol_path = OUT / protocol_seal["path"]
    require(
        protocol_path.stat().st_size == protocol_seal["bytes"]
        and sha256_file(protocol_path) == protocol_seal["sha256"],
        "protocol preregistration artifact drift",
    )
    preregistration = load_json(protocol_path)
    unsigned_preregistration = dict(preregistration)
    observed_preregistration = unsigned_preregistration.pop("protocol_sha256")
    require(
        observed_preregistration == sha256_json(unsigned_preregistration)
        and observed_preregistration == activation["protocol_self_sha256"],
        "protocol preregistration self-hash/binding drift",
    )
    for role, seal in activation["source_seals"].items():
        path = ROOT / seal["path"]
        require(path.stat().st_size == seal["bytes"] and sha256_file(path) == seal["sha256"], f"source drift: {role}")
    for name, seal in activation["dataset_seals"].items():
        path = OUT / seal["path"]
        require(path.stat().st_size == seal["bytes"] and sha256_file(path) == seal["sha256"], f"dataset drift: {name}")
    return {"passed": True, "activation_sha256": activation["activation_sha256"]}


def main() -> None:
    if OUT.exists():
        print(json.dumps(verify_package(), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_package(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
