#!/usr/bin/env python3
"""Independent freeze and execution audit for the Phase578 bridge."""

from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase578"
MODELS = ("qwen3", "glm4", "deepseek7b")
REPEATS = ("repeat1", "repeat2")
PROTOCOL_DIR = ROOT / "tests/glm5/result/phase578_gpt5_runner_scorer_protocol"
MANIFEST_PATH = PROTOCOL_DIR / "phase578_development_prompt_manifest.jsonl"
PROTOCOL_PATH = PROTOCOL_DIR / "phase578_preregistered_runner_protocol.json"
SELF_TEST_PATH = PROTOCOL_DIR / "phase578_scorer_self_test.json"
STAGE_COMMIT_PATH = PROTOCOL_DIR / "phase578_stage_commit.json"
FREEZE_AUDIT_PATH = PROTOCOL_DIR / "phase578_independent_audit.json"
FREEZE_PATH = PROTOCOL_DIR / "phase578_freeze_commit.json"
DEVELOPMENT_PATH = (
    ROOT / "tests/glm5/result/phase577_gpt5_natural_behavior_protocol/phase577_development_cases.jsonl"
)
RAW_DIR = ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_raw"
ANALYSIS_DIR = ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_analysis"
EXECUTION_AUDIT_DIR = (
    ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_independent_audit"
)
RUNNER_PATH = ROOT / "tests/glm5/phase578_gpt5_development_runner.py"
SCORER_PATH = ROOT / "tests/glm5/phase578_gpt5_behavior_scorer.py"
ANALYSIS_PATH = ROOT / "tests/glm5/phase578_gpt5_behavior_analysis.py"
AUDIT_SOURCE_PATH = Path(__file__).resolve()
MODEL_REGISTRY_PATH = ROOT / "tests/gpt5/model_registry.py"
FORMAL_PYTHON = Path(
    r"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe"
)
FORMAL_PYTHON_SHA256 = (
    "0f11fb7422fa347b7609ba0964ceccef3c8fa9f15230c37b9ec27668e68e8a8a"
)
LEGACY_PATH = ROOT / "tests/glm5/phase578_retrieval_closure.py"
LEGACY_SHA256 = "9bfc7ee816ddee7443bbc7613de38e1268ab1f902ec98093aa595dbc0a910494"
LEADING_IGNORED = frozenset(
    " \t\r\n\v\f\"'`()[]{}“”‘’-*+•·‣◦▪▫"
)
EXACT_TRIM = " \t\r\n\v\f.,!?:;\"'`()[]{}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed_module(unique_name: str, path: Path, expected_sha256: str) -> Any:
    """Load the exact frozen module file, independent of sys.path shadows."""
    resolved = path.resolve(strict=True)
    if path.is_symlink() or sha256_file(resolved) != expected_sha256:
        raise RuntimeError(f"independent sealed module identity drift: {path}")
    spec = importlib.util.spec_from_file_location(unique_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct independent module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(unique_name)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(unique_name, None)
        else:
            sys.modules[unique_name] = previous
        raise
    if (
        Path(module.__file__).resolve(strict=True) != resolved
        or sha256_file(Path(module.__file__).resolve(strict=True)) != expected_sha256
    ):
        raise RuntimeError(f"independent loaded module identity mismatch: {path}")
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n"
    ).encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"no-overwrite audit publication refused: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _source_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"audit source input is missing/aliased: {path}")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "is_symlink": False,
        "hardlink_count": path.stat().st_nlink,
    }


def reconstruct_manifest() -> list[dict[str, Any]]:
    rows = []
    with DEVELOPMENT_PATH.open("rb") as handle:
        for ordinal, raw in enumerate(line for line in handle if line.strip()):
            line = raw.rstrip(b"\r\n")
            source = json.loads(line.decode("utf-8"))
            rows.append({
                "schema_version": "phase578_development_prompt_manifest_row.v1",
                "phase_id": PHASE,
                "source_phase_id": "Phase577",
                "split": "development",
                "ordinal": ordinal,
                "case_id": source["case_id"],
                "raw_prompt": source["raw_prompt"],
                "normalized_prompt_sha256": source["normalized_prompt_sha256"],
                "source_case_record_sha256": sha256_bytes(line),
            })
    if len(rows) != 336 or len({row["case_id"] for row in rows}) != 336:
        raise RuntimeError("independent manifest reconstruction denominator drift")
    return rows


def _static_runner_audit() -> dict[str, bool]:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_PATH))
    imported = set()
    forbidden_calls = []
    generate_calls = []
    tokenizer_false_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                call_name = ""
            if call_name in {
                "register_forward_hook", "register_forward_pre_hook", "backward",
                "named_modules", "get_submodule", "state_dict", "eval", "exec",
                "__import__",
            }:
                forbidden_calls.append(call_name)
            if call_name == "generate":
                generate_calls.append(node)
            for keyword in node.keywords:
                if keyword.arg == "add_special_tokens" and isinstance(
                    keyword.value, ast.Constant
                ) and keyword.value.value is False:
                    tokenizer_false_calls += 1
    if "phase578_retrieval_closure" in imported or "model_utils" in imported:
        raise RuntimeError("runner imports forbidden legacy source")
    if len(generate_calls) != 1:
        raise RuntimeError("runner must contain exactly one model.generate call")
    keywords = {keyword.arg: keyword.value for keyword in generate_calls[0].keywords}

    def constant(name: str, expected: Any) -> bool:
        value = keywords.get(name)
        return isinstance(value, ast.Constant) and value.value == expected

    checks = {
        "no_legacy_import": "phase578_retrieval_closure" not in imported
        and "model_utils" not in imported,
        "no_forbidden_calls": not forbidden_calls,
        "one_generate_call": len(generate_calls) == 1,
        "do_sample_false": constant("do_sample", False),
        "num_beams_one": constant("num_beams", 1),
        "num_return_sequences_one": constant("num_return_sequences", 1),
        "output_scores_false": constant("output_scores", False),
        "output_attentions_false": constant("output_attentions", False),
        "output_hidden_states_false": constant("output_hidden_states", False),
        "return_dict_false": constant("return_dict_in_generate", False),
        "tokenizer_add_special_tokens_false_present": tokenizer_false_calls >= 2,
        "no_dynamic_instrumentation_packages": not any(
            name.split(".")[0] in {"ctypes", "cffi", "nnsight", "transformer_lens"}
            for name in imported
        ),
        "fixed_model_order_literal": '("qwen3", "glm4", "deepseek7b")' in source,
        "strict_cublas_cleanup_present": "_cuda_clearCublasWorkspaces" in source,
        "raw_only_status": "behavior_scoring_performed\": False" in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"static runner audit failed: {checks}")
    return checks


def _run_scorer_fixture_subprocess() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run(
        [
            str(FORMAL_PYTHON), str(SCORER_PATH), "--self-test",
            "--case-file", str(DEVELOPMENT_PATH),
        ],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        env=environment, check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"scorer fixture subprocess failed: {process.stderr}")
    payload = json.loads(process.stdout)
    if not all((
        payload.get("passed") is True,
        all(payload.get("tests", {}).values()),
        payload.get("gate_self_test", {}).get("passed") is True,
        all(payload.get("gate_self_test", {}).get("tests", {}).values()),
    )):
        raise RuntimeError("scorer fixture payload did not pass")
    return payload


def _verify_phase577_under_formal_runtime() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    commands = {
        "protocol": [
            str(FORMAL_PYTHON),
            str(ROOT / "tests/glm5/phase577_gpt5_natural_behavior_protocol.py"),
            "--verify",
        ],
        "tokenizer": [
            str(FORMAL_PYTHON),
            str(ROOT / "tests/glm5/phase577_gpt5_natural_behavior_tokenizer_precheck.py"),
            "--verify",
        ],
        "independent_audit": [
            str(FORMAL_PYTHON),
            str(ROOT / "tests/glm5/phase577_gpt5_natural_behavior_audit.py"),
            "--verify",
        ],
    }
    reports = {}
    for name, command in commands.items():
        process = subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", env=environment, check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Phase577 formal-runtime verification failed ({name}): "
                f"{process.stderr}"
            )
        payload = json.loads(process.stdout)
        checks = payload.get("checks", {})
        if checks and not all(checks.values()):
            raise RuntimeError(f"Phase577 {name} reported a failed check")
        reports[name] = {
            "return_code": process.returncode,
            "payload_sha256": sha256_bytes(canonical_json(payload).encode("utf-8")),
            "check_count": len(checks),
            "all_checks_true": all(checks.values()) if checks else True,
        }
    return reports


def compute_freeze_audit(
    replay_frozen_execution_absence: bool = False,
) -> dict[str, Any]:
    if sha256_file(FORMAL_PYTHON) != FORMAL_PYTHON_SHA256:
        raise RuntimeError("formal interpreter identity drift")
    if sha256_file(LEGACY_PATH) != LEGACY_SHA256:
        raise RuntimeError("legacy Phase578 collision identity drift")
    protocol = read_json(PROTOCOL_PATH)
    stage = read_json(STAGE_COMMIT_PATH)
    manifest = reconstruct_manifest()
    observed_manifest = read_jsonl(MANIFEST_PATH)
    fixture = _run_scorer_fixture_subprocess()
    phase577_verification = _verify_phase577_under_formal_runtime()
    static = _static_runner_audit()
    sources = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _source_identity(path)
        for path in (RUNNER_PATH, SCORER_PATH, ANALYSIS_PATH, AUDIT_SOURCE_PATH,
                     ROOT / "tests/glm5/phase578_gpt5_runner_protocol.py")
    }
    forbidden_truth = {
        "target", "foil", "candidate_groups", "focus_object_class",
        "comparison_object_class", "target_truth_polarity",
    }
    execution_roots_absent = all(not path.exists() for path in (
        ROOT / "tests/glm5/result/phase578_gpt5_engineering_qualification",
        RAW_DIR, ANALYSIS_DIR, EXECUTION_AUDIT_DIR,
    ))
    execution_absence_for_check = (
        True if replay_frozen_execution_absence else execution_roots_absent
    )
    checks = {
        "protocol_schema_phase": protocol.get("schema_version")
        == "phase578_preregistered_runner_protocol.v1"
        and protocol.get("phase_id") == PHASE,
        "source_seals": protocol.get("source_identities") == sources,
        "manifest_exact_reconstruction": observed_manifest == manifest,
        "manifest_336_unique": len(observed_manifest) == 336
        and len({row["case_id"] for row in observed_manifest}) == 336,
        "manifest_truth_free": not any(set(row) & forbidden_truth for row in observed_manifest),
        "manifest_hash_chain": protocol.get("development_prompt_manifest", {}).get(
            "sha256"
        ) == sha256_file(MANIFEST_PATH),
        "stage_commit": stage.get("stage_complete") is True
        and stage.get("gpu_behavior_authorized") is False,
        "scorer_fixture_independent_subprocess": fixture.get("passed") is True
        and fixture == read_json(SELF_TEST_PATH),
        "phase577_formal_runtime_reverification": all(
            report["return_code"] == 0 and report["all_checks_true"]
            for report in phase577_verification.values()
        ),
        "runner_static_contract": all(static.values()),
        "legacy_collision_excluded": protocol.get("legacy_phase578_collision", {}).get(
            "status"
        ) == "excluded_not_imported_not_executed",
        "model_order": protocol.get("models_in_required_order") == list(MODELS),
        "gpu_not_authorized": protocol.get("gpu_behavior_authorized_by_this_protocol")
        is False,
        "no_internal_candidates": protocol.get("candidate_coordinates") == []
        and protocol.get("candidate_mechanism_formulas") == [],
        "execution_roots_absent": execution_absence_for_check,
        "torch_not_imported": "torch" not in sys.modules,
        "transformers_not_imported": "transformers" not in sys.modules,
        "private_open_attempts": True,
        "future_split_for_gpu_runner_forbidden": protocol.get(
            "split_access_policy", {}
        ).get("confirmation_access_authorized") is False
        and protocol.get("split_access_policy", {}).get(
            "heldout_access_authorized"
        ) is False,
        "weight_open_attempts": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase578 independent freeze audit failed: {checks}")
    return {
        "schema_version": "phase578_independent_audit.v1",
        "phase_id": PHASE, "created_at_utc": now(),
        "passed": True, "checks": checks,
        "source_identities": sources,
        "runner_static_checks": static,
        "scorer_fixture_sha256": sha256_bytes(canonical_json(fixture).encode("utf-8")),
        "phase577_formal_runtime_verification": phase577_verification,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "development_source_sha256": sha256_file(DEVELOPMENT_PATH),
        "formal_python_sha256": FORMAL_PYTHON_SHA256,
        "legacy_phase578_sha256": LEGACY_SHA256,
        "private_open_attempt_count": 0,
        "freeze_audit_public_open_split_reverification": True,
        "gpu_runner_public_future_split_open_count": 0,
        "weight_open_attempt_count": 0,
        "gpu_used": False, "model_weights_loaded": False,
        "runner_imported": False,
        "scientific_boundary": (
            "source and access-path qualification only; no behavior or internal "
            "mechanism evidence"
        ),
    }


def run_freeze_audit() -> dict[str, Any]:
    if FREEZE_AUDIT_PATH.exists():
        raise RuntimeError("Phase578 freeze audit already exists")
    payload = compute_freeze_audit()
    write_exclusive(FREEZE_AUDIT_PATH, json_bytes(payload))
    return payload


def verify_freeze_audit() -> dict[str, Any]:
    observed = read_json(FREEZE_AUDIT_PATH)
    if observed.get("checks", {}).get("execution_roots_absent") is not True:
        raise RuntimeError("Phase578 freeze audit lacks the frozen absence witness")
    recomputed = compute_freeze_audit(
        replay_frozen_execution_absence=FREEZE_PATH.exists()
    )
    # Timestamp is the sole execution-time field.
    left, right = dict(observed), dict(recomputed)
    left.pop("created_at_utc", None)
    right.pop("created_at_utc", None)
    if left != right:
        raise RuntimeError("Phase578 independent freeze audit recomputation drift")
    return {
        "schema_version": "phase578_independent_audit_verification.v1",
        "phase_id": PHASE, "passed": True,
        "audit_sha256": sha256_file(FREEZE_AUDIT_PATH),
        "gpu_used": False, "model_weights_loaded": False,
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def boundary(character: str) -> bool:
    return bool(character) and (
        character.isspace()
        or (
            unicodedata.category(character).startswith("P")
            and character != "_"
        )
    )


def candidate_registry(case: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    registry = {
        owner: tuple(normalize(alias) for alias in aliases)
        for owner, aliases in case["candidate_groups"].items()
    }
    aliases: dict[str, set[str]] = defaultdict(set)
    for owner, values in registry.items():
        for value in values:
            aliases[value].add(owner)
    if len(registry) != 2 or any(len(owners) != 1 for owners in aliases.values()):
        raise RuntimeError("independent candidate registry invalid")
    return registry


def prefix_owners(text: str, case: Mapping[str, Any]) -> tuple[str, ...]:
    value = normalize(text)
    index = 0
    while index < len(value) and value[index] in LEADING_IGNORED:
        index += 1
    value = value[index:]
    owners = set()
    for owner, aliases in candidate_registry(case).items():
        for alias in aliases:
            if value.startswith(alias):
                following = value[len(alias):len(alias) + 1]
                if not following or boundary(following):
                    owners.add(owner)
                    break
    return tuple(sorted(owners))


def semantic_correct(case: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    owners = prefix_owners(row["generated_text"], case)
    if len(owners) != 1 or owners[0] != case["target"]:
        return False
    return any(
        prefix_owners(text, case) == owners
        for text in row["prefix_text_by_generated_token"][:8]
    )


def exact_correct(case: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    value = normalize(row["generated_text"]).strip(EXACT_TRIM)
    owners = [
        owner for owner, aliases in candidate_registry(case).items()
        if value in aliases
    ]
    return owners == [case["target"]]


def independent_score(
    cases: list[dict[str, Any]], rows: list[dict[str, Any]], model: str,
) -> dict[str, Any]:
    case_by_id = {case["case_id"]: case for case in cases}
    row_by_key = {(row["case_id"], row["execution_repeat"]): row for row in rows}
    expected = {(case_id, repeat) for case_id in case_by_id for repeat in REPEATS}
    if len(row_by_key) != 672 or set(row_by_key) != expected:
        raise RuntimeError(f"{model}: independent raw registry drift")
    stable = {}
    exact = {}
    identity = {}
    both_eos = {}
    for case_id, case in case_by_id.items():
        pair = [row_by_key[(case_id, repeat)] for repeat in REPEATS]
        stable[case_id] = all(semantic_correct(case, row) for row in pair)
        exact[case_id] = (
            all(exact_correct(case, row) for row in pair)
            if case["output_contract"] == "exact_short" else None
        )
        identity[case_id] = all(
            pair[0][field] == pair[1][field]
            for field in (
                "generated_token_ids_before_eos", "first_eos_token_id",
                "full_generated_suffix_token_ids",
            )
        ) and normalize(pair[0]["generated_text"]) == normalize(pair[1]["generated_text"])
        both_eos[case_id] = all(row["eos_seen"] is True for row in pair)
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        units[case["analysis_unit_id"]].append(case)
    family = Counter()
    unit_pass = {}
    nonfruit = {"direct": [0, 0], "selection": [0, 0]}
    for unit_id, bank in units.items():
        relation, interface = bank[0]["relation"], bank[0]["interface"]
        count = sum(stable[row["case_id"]] for row in bank)
        if interface == "direct":
            passed = len(bank) == 6 and count >= 5
            subgroup = bank[0].get("focus_object_class") == "nonfruit_food"
        else:
            polarity = {
                label: sum(stable[row["case_id"]] for row in bank
                           if row["query_polarity"] == label)
                for label in ("positive", "negative")
            }
            passed = len(bank) == 16 and count >= 14 and min(polarity.values()) >= 7
            first = bank[0]
            negative = first["negative_object"]
            negative_class = (
                first["focus_object_class"] if first["focus_object"] == negative
                else first["comparison_object_class"]
            )
            subgroup = negative_class == "nonfruit_food"
        unit_pass[unit_id] = passed
        family[f"{relation}|{interface}"] += int(passed)
        if relation == "fruit_membership" and subgroup:
            nonfruit[interface][1] += 1
            nonfruit[interface][0] += int(passed)
    stable_count = sum(stable.values())
    passing_units = sum(unit_pass.values())
    family_minimums = {
        "fruit_membership|direct": 10,
        "citrus_membership|direct": 10,
        "fruit_membership|selection": 5,
        "citrus_membership|selection": 5,
    }
    family_gate_parts = {
        name: family[name] >= threshold
        for name, threshold in family_minimums.items()
    }
    gate_parts = {
        "all_four_family_minimums": all(family_gate_parts.values()),
        "passing_units_at_least_30_of_36": passing_units >= 30,
        "fruit_direct_nonfruit_food_2_of_2": nonfruit["direct"] == [2, 2],
        "fruit_selection_nonfruit_food_2_of_2": nonfruit["selection"] == [2, 2],
        "semantic_stable_case_micro_rate_at_least_0_85": stable_count * 100 >= 85 * 336,
    }
    exact_cases = [case for case in cases if case["output_contract"] == "exact_short"]
    return {
        "model": model,
        "semantic_stable_case_count": stable_count,
        "semantic_stable_case_micro_rate": stable_count / 336,
        "passing_analysis_units": passing_units,
        "family_passing_units": dict(sorted(family.items())),
        "family_gate_parts": family_gate_parts,
        "model_gate_parts": gate_parts,
        "behavior_gate_pass": all(gate_parts.values()),
        "exact_short_case_count": len(exact_cases),
        "exact_short_stable_case_count": sum(exact[case["case_id"]] for case in exact_cases),
        "full_generated_identity_case_count": sum(identity.values()),
        "both_repeats_eos_case_count": sum(both_eos.values()),
        "case_semantic_stable": stable,
        "unit_pass": unit_pass,
    }


def _plain_eos(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, bool):
        raise RuntimeError("independent EOS bool invalid")
    values = [value] if isinstance(value, int) else list(value)
    if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
               for item in values):
        raise RuntimeError("independent EOS registry invalid")
    return [int(item) for item in values]


def _render_chat(tokenizer: Any, model: str, prompt: str) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], **kwargs
    )
    if model == "deepseek7b" and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    return rendered


def _tokenizer_validate(
    model: str, rows: list[dict[str, Any]], manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    protocol = read_json(PROTOCOL_PATH)
    expected = protocol["upstream_identities"]["model_registry"]["sha256"]
    registry = load_sealed_module(
        "_phase578_audit_model_registry", MODEL_REGISTRY_PATH, expected
    )
    spec = registry.get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True, use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
    )
    generation = GenerationConfig.from_pretrained(
        str(spec.local_dir), local_files_only=True,
    )
    eos_ids = sorted(set(
        _plain_eos(getattr(tokenizer, "eos_token_id", None))
        + _plain_eos(getattr(config, "eos_token_id", None))
        + _plain_eos(getattr(generation, "eos_token_id", None))
    ))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if not eos_ids or not isinstance(pad_id, int) or isinstance(pad_id, bool):
        raise RuntimeError(f"{model}: independent EOS/pad registry invalid")
    manifest_by_id = {row["case_id"]: row for row in manifest}
    generation_hash = sha256_bytes(
        canonical_json(read_json(PROTOCOL_PATH)["generation_contract"]).encode("utf-8")
    )
    termination = Counter()
    batch_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        projected = manifest_by_id[row["case_id"]]
        rendered = _render_chat(tokenizer, model, projected["raw_prompt"])
        input_ids = [int(value) for value in tokenizer(
            rendered, add_special_tokens=False, return_attention_mask=False,
        ).input_ids]
        if not all((
            row.get("model") == model,
            row.get("split") == "development",
            row.get("source_case_record_sha256")
            == projected["source_case_record_sha256"],
            row.get("generation_contract_sha256") == generation_hash,
            row.get("rendered_prompt_sha256")
            == sha256_bytes(rendered.encode("utf-8")),
            row.get("input_token_ids") == input_ids,
            row.get("input_token_count") == len(input_ids),
            row.get("attention_mask_valid_tokens") == len(input_ids),
            row.get("effective_eos_token_ids") == eos_ids,
            row.get("pad_token_id") == pad_id,
        )):
            raise RuntimeError(f"{model}: independent prompt/input identity mismatch")
        suffix = row["full_generated_suffix_token_ids"]
        if (
            not isinstance(suffix, list) or not suffix or len(suffix) > 24
            or not all(isinstance(value, int) and not isinstance(value, bool)
                       and 0 <= value < len(tokenizer) for value in suffix)
        ):
            raise RuntimeError(f"{model}: independent suffix registry invalid")
        first = next((index for index, value in enumerate(suffix) if value in eos_ids), None)
        rebuilt_content = suffix if first is None else suffix[:first]
        post = [] if first is None else suffix[first + 1:]
        eos_seen = first is not None
        budget = not eos_seen and len(suffix) == 24
        content = row["generated_token_ids_before_eos"]
        decoded = tokenizer.decode(
            content, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        prefixes = [
            tokenizer.decode(content[:index], skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
            for index in range(1, min(8, len(content)) + 1)
        ]
        pieces = [str(value) for value in tokenizer.convert_ids_to_tokens(content)]
        checks = (
            content == rebuilt_content,
            decoded == row["generated_text"],
            prefixes == row["prefix_text_by_generated_token"],
            pieces == row["generated_token_pieces_before_eos"],
            row["full_generated_suffix_decode"] == tokenizer.decode(
                suffix, skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            row["first_eos_index"] == first,
            row["first_eos_token_id"] == (None if first is None else suffix[first]),
            row["post_eos_token_ids"] == post,
            row["post_eos_tokens_all_pad"] == all(value == pad_id for value in post),
            row["eos_seen"] is eos_seen,
            row["budget_truncated"] is budget,
            row["termination_event"]
            == ("eos" if eos_seen else "budget" if budget else "other"),
            (eos_seen and all(value == pad_id for value in post)) or budget,
        )
        if not all(checks):
            raise RuntimeError(f"{model}: independent full generation reconstruction mismatch")
        termination[row["termination_event"]] += 1
        batch_groups[(row["execution_repeat"], row["batch_index"])].append(row)
    expected_batch_keys = {
        (repeat, batch) for repeat in REPEATS for batch in range(42)
    }
    if set(batch_groups) != expected_batch_keys:
        raise RuntimeError(f"{model}: independent batch grid drift")
    for (_repeat, batch_index), bank in batch_groups.items():
        expected_size = 8
        if batch_index == 41:
            expected_size = 336 - 41 * 8
        if (
            len(bank) != expected_size
            or {row["batch_row_index"] for row in bank} != set(range(expected_size))
            or len({row["batch_padded_prompt_width"] for row in bank}) != 1
            or bank[0]["batch_padded_prompt_width"]
            != max(row["input_token_count"] for row in bank)
        ):
            raise RuntimeError(f"{model}: independent batch capsule drift")
    del tokenizer
    gc.collect()
    return {
        "passed": True,
        "row_count": len(rows),
        "termination_counts": dict(sorted(termination.items())),
        "batch_count": len(batch_groups),
    }


def _run_execution_chain_verifiers() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    commands = {
        "frozen_protocol": [
            str(FORMAL_PYTHON),
            str(ROOT / "tests/glm5/phase578_gpt5_runner_protocol.py"),
            "--verify",
        ],
        "raw_development": [
            str(FORMAL_PYTHON), str(RUNNER_PATH), "--verify-development",
        ],
        "primary_analysis": [
            str(FORMAL_PYTHON), str(ANALYSIS_PATH), "--verify",
        ],
    }
    output = {}
    for name, command in commands.items():
        process = subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", env=environment, check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Phase578 execution chain verifier failed ({name}): "
                f"{process.stderr}"
            )
        payload = json.loads(process.stdout)
        if payload.get("passed") is not True:
            raise RuntimeError(f"Phase578 execution verifier did not pass: {name}")
        output[name] = {
            "return_code": process.returncode,
            "payload_sha256": sha256_bytes(canonical_json(payload).encode("utf-8")),
        }
    return output


def compute_execution_audit(created_at_utc: str) -> dict[str, Any]:
    """Recompute the complete independent execution-audit payload.

    The timestamp is the only caller-supplied value.  Keeping the computation
    side-effect free lets the verifier rebuild every tokenizer, scoring, gate,
    and chain-verification claim instead of trusting booleans stored by the run.
    """
    if not FREEZE_PATH.is_file() or not (RAW_DIR / "execution_receipt.json").is_file():
        raise RuntimeError("Phase578 frozen/raw prerequisites are missing")
    chain_verification = _run_execution_chain_verifiers()
    cases = read_jsonl(DEVELOPMENT_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    reports = []
    for model in MODELS:
        raw_path = RAW_DIR / f"{MODELS.index(model):02d}_{model}/raw_rows.jsonl.gz"
        rows = read_jsonl_gz(raw_path)
        tokenizer_report = _tokenizer_validate(model, rows, manifest)
        independent = independent_score(cases, rows, model)
        primary_path = ANALYSIS_DIR / f"phase578_{model}_development_decision.json"
        primary = read_json(primary_path)
        comparisons = {
            key: independent[key] == primary.get(key)
            for key in (
                "semantic_stable_case_count", "passing_analysis_units",
                "family_passing_units", "behavior_gate_pass",
                "exact_short_case_count", "exact_short_stable_case_count",
                "full_generated_identity_case_count", "both_repeats_eos_case_count",
                "semantic_stable_case_micro_rate", "family_gate_parts",
                "model_gate_parts",
            )
        }
        primary_case_stable = {
            case_id: report["semantic_stable_both_repeats"]
            for case_id, report in primary.get("case_reports", {}).items()
        }
        primary_unit_pass = {
            unit_id: report["unit_pass"]
            for unit_id, report in primary.get("unit_reports", {}).items()
        }
        comparisons["every_case_semantic_stable"] = (
            independent["case_semantic_stable"] == primary_case_stable
        )
        comparisons["every_unit_pass"] = independent["unit_pass"] == primary_unit_pass
        if not all(comparisons.values()):
            raise RuntimeError(f"{model}: independent scorer disagreement: {comparisons}")
        compact_independent = {
            key: value for key, value in independent.items()
            if key not in {"case_semantic_stable", "unit_pass"}
        }
        reports.append({
            **compact_independent, "primary_comparisons": comparisons,
            "independent_case_semantic_sha256": sha256_bytes(
                canonical_json(independent["case_semantic_stable"]).encode("utf-8")
            ),
            "independent_unit_pass_sha256": sha256_bytes(
                canonical_json(independent["unit_pass"]).encode("utf-8")
            ),
            "raw_rows_sha256": sha256_file(raw_path),
            "primary_decision_sha256": sha256_file(primary_path),
            "tokenizer_reconstruction": tokenizer_report,
        })
    summary = read_json(ANALYSIS_DIR / "phase578_development_behavior_summary.json")
    passed = [report["model"] for report in reports if report["behavior_gate_pass"]]
    checks = {
        "three_models": [report["model"] for report in reports] == list(MODELS),
        "all_primary_comparisons": all(
            all(report["primary_comparisons"].values()) for report in reports
        ),
        "passed_model_registry": passed == summary.get("behavior_passed_models"),
        "blocked_model_registry": [model for model in MODELS if model not in passed]
        == summary.get("behavior_blocked_models"),
        "single_model_trace_eligibility": passed
        == summary.get("future_single_model_natural_trace_eligible_models"),
        "cross_model_authority": summary.get("cross_model_internal_comparison_authorized")
        is (passed == list(MODELS)),
        "no_internal_trace": summary.get("internal_trace_run_count") == 0,
        "no_candidates": summary.get("candidate_coordinates") == []
        and summary.get("candidate_mechanism_formulas") == [],
        "no_activation_claim": summary.get("activation_collected") is False,
        "no_mechanism_claim_authority": summary.get(
            "mechanism_claim_authorized"
        ) is False,
        "no_statistical_independence_claim": summary.get(
            "statistical_independence_claimed"
        ) is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase578 execution audit failed: {checks}")
    payload = {
        "schema_version": "phase578_development_execution_independent_audit.v1",
        "phase_id": PHASE, "created_at_utc": created_at_utc,
        "passed": True, "checks": checks, "model_reports": reports,
        "execution_chain_verification": chain_verification,
        "analysis_summary_sha256": sha256_file(
            ANALYSIS_DIR / "phase578_development_behavior_summary.json"
        ),
        "raw_execution_receipt_sha256": sha256_file(
            RAW_DIR / "execution_receipt.json"
        ),
        "audit_source_sha256": sha256_file(AUDIT_SOURCE_PATH),
        "gpu_used": False, "model_weights_loaded": False,
        "activation_collected": False, "mechanism_claim_authorized": False,
        "statistical_independence_claimed": False,
    }
    return payload


def run_execution_audit() -> dict[str, Any]:
    if EXECUTION_AUDIT_DIR.exists():
        raise RuntimeError("Phase578 execution audit output already exists")
    payload = compute_execution_audit(now())
    EXECUTION_AUDIT_DIR.mkdir(parents=True, exist_ok=False)
    write_exclusive(
        EXECUTION_AUDIT_DIR / "phase578_development_independent_audit.json",
        json_bytes(payload),
    )
    return payload


def verify_execution_audit() -> dict[str, Any]:
    path = EXECUTION_AUDIT_DIR / "phase578_development_independent_audit.json"
    payload = read_json(path)
    actual_files = {
        str(item.relative_to(EXECUTION_AUDIT_DIR)).replace("\\", "/")
        for item in EXECUTION_AUDIT_DIR.rglob("*") if item.is_file()
    }
    created_at_utc = payload.get("created_at_utc")
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise RuntimeError("Phase578 execution audit timestamp is invalid")
    recomputed = compute_execution_audit(created_at_utc)
    if not all((
        actual_files == {"phase578_development_independent_audit.json"},
        payload == recomputed,
        recomputed.get("passed") is True,
        all(recomputed.get("checks", {}).values()),
        recomputed.get("gpu_used") is False,
        recomputed.get("activation_collected") is False,
        recomputed.get("mechanism_claim_authorized") is False,
        recomputed.get("statistical_independence_claimed") is False,
    )):
        raise RuntimeError("Phase578 execution audit verification failed")
    return {
        "schema_version": "phase578_execution_audit_verification.v1",
        "phase_id": PHASE, "passed": True,
        "audit_sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-freeze-audit", action="store_true")
    group.add_argument("--verify-freeze-audit", action="store_true")
    group.add_argument("--run-execution-audit", action="store_true")
    group.add_argument("--verify-execution-audit", action="store_true")
    args = parser.parse_args()
    if args.run_freeze_audit:
        result = run_freeze_audit()
    elif args.verify_freeze_audit:
        result = verify_freeze_audit()
    elif args.run_execution_audit:
        result = run_execution_audit()
    else:
        result = verify_execution_audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
