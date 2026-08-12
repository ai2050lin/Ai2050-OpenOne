#!/usr/bin/env python3
"""Freeze the Phase1117 Pythia contextual-modulation training protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1117_pythia_training_dynamics_verified_safetensors_v4"
MODEL_ROOT = ROOT / "models" / "hf" / "pythia-1.4b-deduped"
TOKENIZER_ROOT = ROOT / "models" / "hf" / "pythia-1.4b-deduped" / "tokenizer"
INVALID_INSTRUMENT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1117_pythia_training_dynamics"
SOURCE_1114 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1114_wordnet_contextual_hypernym"
)
SOURCE_1115 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1115_wordnet_context_modulation_confirmation"
)

PHASE = 1117
PROTOCOL_REVISION = 4
MODEL_REPO = "EleutherAI/pythia-1.4b-deduped"
PRECISION = "fp16"
QUANTIZATION = "none"
WEIGHT_FORMAT = "model.safetensors"
FINAL_QUALIFICATION_CHECKPOINT = "step143000"
CHECKPOINTS = (
    "step0",
    "step1",
    "step4",
    "step64",
    "step256",
    "step512",
    "step1000",
    "step4000",
    "step16000",
    "step64000",
    "step143000",
)
SPLITS = ("discovery", "independent_confirmation", "heldout")

TEMPLATES = (
    'Sentence: {sentence}\nThe noun "{term}" in this sentence is a kind of',
    'Context: {sentence}\nHere, the noun "{term}" means a type of',
    'Example: {sentence}\nThe contextual category of "{term}" is',
    'Usage: {sentence}\nIn this usage, "{term}" is an example of',
    'Read: {sentence}\nThe intended sense of the noun "{term}" belongs under',
    'English sentence: {sentence}\nThe broader noun category for "{term}" is',
)

THRESHOLDS = {
    "minimum_finite_fraction": 0.99,
    "minimum_overall_direction_accuracy": 0.80,
    "minimum_split_direction_accuracy": 0.75,
    "minimum_template_direction_accuracy": 0.65,
    "minimum_overall_control_advantage": 0.15,
    "minimum_split_control_advantage": 0.10,
    "minimum_template_control_advantage": 0.05,
    "minimum_concept_positive_fraction": 0.80,
    "minimum_positive_concepts_per_split": 15,
    "trajectory_onset_direction_accuracy": 0.75,
    "trajectory_onset_control_advantage": 0.10,
    "trajectory_onset_concept_fraction": 0.70,
    "minimum_final_minus_step0_direction_gain": 0.15,
    "minimum_final_minus_step0_advantage_gain": 0.10,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "Both source result audits, the shared Pythia tokenizer, the 57-concept "
        "inventory, lexical isolation, continuation-token boundaries, matched-control "
        "derangements, split balance, counts, and digests pass before model weights are read. "
        "Before any behavior case is run, every frozen checkpoint must use the declared "
        "weight carrier and pass a tensor-content collision audit."
    ),
    "P2": (
        "The final step143000 checkpoint is finite in FP16 without quantization and "
        "passes every frozen direction, split, template, concept, and mismatched-candidate "
        "control-advantage gate."
    ),
    "P3": (
        "Only P2 authorizes the other eleven frozen checkpoints. No prompt, item, "
        "checkpoint, or threshold may be changed after step143000 is observed."
    ),
    "P4": (
        "If authorized, all eleven valid checkpoints run, and the observed onset is the first "
        "of two consecutive sampled checkpoints meeting the frozen onset gate."
    ),
    "P5": (
        "At step143000, direction accuracy improves over step0 by at least 0.15 and "
        "true-minus-control direction advantage improves by at least 0.10."
    ),
    "P6": (
        "This phase measures formation of an output-margin phenomenon only. It never "
        "authorizes hidden-state, component, neuron, or causal claims."
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_sources() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_specs = ((1114, SOURCE_1114), (1115, SOURCE_1115))
    items: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for phase, root in source_specs:
        audit_path = root / "audit" / "result_audit.json"
        selected_path = root / "protocol" / "selected_concepts.json"
        audit = read_json(audit_path)
        selected = read_json(selected_path)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"Phase{phase} source audit failed")
        for item in selected["selected"]:
            copied = dict(item)
            copied["source_phase"] = phase
            items.append(copied)
        manifests[str(phase)] = {
            "audit_digest": audit["audit_digest"],
            "audit_sha256": file_sha256(audit_path),
            "selected_count": selected["selected_count"],
            "selected_digest": selected["selected_digest"],
            "selected_sha256": file_sha256(selected_path),
        }
    return items, manifests


def continuation_token_id(tokenizer: Any, prompt: str, token: str) -> int:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + " " + token, add_special_tokens=False)
    if full[: len(prefix)] != prefix or len(full) != len(prefix) + 1:
        raise RuntimeError(f"candidate is not one stable continuation token: {token!r}")
    return int(full[-1])


def contains_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, re.I) is not None


def choose_control_maps(items: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, int]]:
    mapping: dict[str, str] = {}
    shifts: dict[str, int] = {}
    for split in SPLITS:
        panel = sorted((item for item in items if item["split"] == split), key=lambda x: x["concept_id"])
        if len(panel) != 19:
            raise RuntimeError(f"expected 19 concepts in {split}, found {len(panel)}")
        chosen_shift = None
        for shift in range(1, len(panel)):
            valid = True
            for index, item in enumerate(panel):
                donor = panel[(index + shift) % len(panel)]
                prompts = [
                    template.format(sentence=sentence, term=item["base"])
                    for template in TEMPLATES
                    for sentence in item["examples"]
                ]
                if any(contains_word(prompt, candidate) for prompt in prompts for candidate in donor["hypernyms"]):
                    valid = False
                    break
            if valid:
                chosen_shift = shift
                break
        if chosen_shift is None:
            raise RuntimeError(f"could not construct nonleaking control derangement for {split}")
        shifts[split] = chosen_shift
        for index, item in enumerate(panel):
            donor = panel[(index + chosen_shift) % len(panel)]
            mapping[item["concept_id"]] = donor["concept_id"]
    return mapping, shifts


def build_protocol() -> dict[str, Any]:
    if not TOKENIZER_ROOT.exists():
        raise RuntimeError(f"missing tokenizer at {TOKENIZER_ROOT}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ROOT, local_files_only=True)
    items, source_manifests = load_sources()
    if len(items) != 57 or len({item["concept_id"] for item in items}) != 57:
        raise RuntimeError("expected 57 unique source concepts")
    split_counts = Counter(item["split"] for item in items)
    if split_counts != Counter({split: 19 for split in SPLITS}):
        raise RuntimeError(f"unexpected split counts: {split_counts}")

    lexical_pool: list[str] = []
    for item in items:
        lexical_pool.extend([item["base"], *item["hypernyms"]])
    if len(lexical_pool) != len(set(lexical_pool)):
        raise RuntimeError("source lexical identities are not globally isolated")

    control_map, control_shifts = choose_control_maps(items)
    by_id = {item["concept_id"]: item for item in items}
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    for item in sorted(items, key=lambda x: (SPLITS.index(x["split"]), x["concept_id"])):
        control = by_id[control_map[item["concept_id"]]]
        selected.append(
            {
                "concept_id": item["concept_id"],
                "source_phase": item["source_phase"],
                "split": item["split"],
                "base": item["base"],
                "examples": item["examples"],
                "hypernyms": item["hypernyms"],
                "control_concept_id": control["concept_id"],
                "control_hypernyms": control["hypernyms"],
            }
        )
        for template_index, template in enumerate(TEMPLATES):
            pair_id = f"phase1117.{item['split']}.{item['concept_id']}.t{template_index}"
            for sense, sentence in enumerate(item["examples"]):
                prompt = template.format(sentence=sentence, term=item["base"])
                for candidate in [*item["hypernyms"], *control["hypernyms"]]:
                    if contains_word(prompt, candidate):
                        raise RuntimeError(f"candidate leakage: {item['concept_id']} {candidate}")
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                true_ids = [continuation_token_id(tokenizer, prompt, value) for value in item["hypernyms"]]
                control_ids = [continuation_token_id(tokenizer, prompt, value) for value in control["hypernyms"]]
                if len(set(true_ids + control_ids)) != 4:
                    raise RuntimeError(f"candidate token collision for {item['concept_id']}")
                prompt_lengths.append(len(input_ids))
                rows.append(
                    {
                        "schema_version": "phase1117_pythia_training_case.v1",
                        "phase": PHASE,
                        "case_index": len(rows),
                        "record_id": f"{pair_id}.s{sense}",
                        "pair_id": pair_id,
                        "concept_id": item["concept_id"],
                        "source_phase": item["source_phase"],
                        "split": item["split"],
                        "template": template_index,
                        "sense": sense,
                        "base": item["base"],
                        "native_example": sentence,
                        "true_candidate_labels": item["hypernyms"],
                        "true_candidate_ids": true_ids,
                        "control_concept_id": control["concept_id"],
                        "control_candidate_labels": control["hypernyms"],
                        "control_candidate_ids": control_ids,
                        "input_ids": input_ids,
                        "query_position": len(input_ids) - 1,
                        "raw_prompt": prompt,
                        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    }
                )

    if len(rows) != 57 * len(TEMPLATES) * 2:
        raise RuntimeError(f"unexpected case count {len(rows)}")
    if len({row["pair_id"] for row in rows}) != 57 * len(TEMPLATES):
        raise RuntimeError("unexpected pair count")

    tokenizer_files = {
        str(path.relative_to(TOKENIZER_ROOT)): {
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(TOKENIZER_ROOT.iterdir())
        if path.is_file() and not path.name.endswith(".metadata") and path.name != ".gitignore"
    }
    selected_payload = {
        "selected": selected,
        "selected_count": len(selected),
        "selected_digest": digest(selected),
        "control_shifts": control_shifts,
    }
    case_digest = digest(rows)
    preregistration_core = {
        "schema_version": "phase1117_pythia_training_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model_repo": MODEL_REPO,
        "weight_format": WEIGHT_FORMAT,
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "final_qualification_checkpoint": FINAL_QUALIFICATION_CHECKPOINT,
        "checkpoints": list(CHECKPOINTS),
        "splits": list(SPLITS),
        "templates": list(TEMPLATES),
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "case_count": len(rows),
        "pair_count": len(rows) // 2,
        "concept_count": len(selected),
        "cases_per_checkpoint": len(rows),
        "case_digest": case_digest,
        "selected_digest": selected_payload["selected_digest"],
        "source_manifests": source_manifests,
        "tokenizer_files": tokenizer_files,
        "control_shifts": control_shifts,
        "model_outputs_read_during_protocol": False,
        "instrument_correction": {
            "scope": "verified weight carrier and invalid-checkpoint exclusion only; materials, thresholds, and outcome gates remain frozen",
            "invalid_revision_output": str(INVALID_INSTRUMENT_ROOT.relative_to(ROOT)),
            "revision2_output": "tests/glm5/result/phase1117_pythia_training_dynamics_pytorch_bin",
            "revision3_output": "tests/glm5/result/phase1117_pythia_training_dynamics_pytorch_bin_v3",
            "observed_collision": "safetensors step16/step32 matched final; pytorch_model.bin step16/step32 and every tested step64-or-later checkpoint matched final",
            "replacement": "exclude corrupted step16/step32, retain ten other original safetensors trajectory points plus final, and require all-checkpoint tensor preflight before behavior",
        },
    }
    preregistration = dict(preregistration_core)
    preregistration["protocol_digest"] = digest(preregistration_core)

    audit_checks = {
        "source_audits_passed": all(read_json(root / "audit" / "result_audit.json")["all_checks_passed"] for root in (SOURCE_1114, SOURCE_1115)),
        "concept_count_57": len(selected) == 57,
        "split_balance_19_each": split_counts == Counter({split: 19 for split in SPLITS}),
        "global_lexical_isolation": len(lexical_pool) == len(set(lexical_pool)),
        "case_count_684": len(rows) == 684,
        "pair_count_342": len({row["pair_id"] for row in rows}) == 342,
        "candidate_nonleakage": all(
            not any(contains_word(row["raw_prompt"], value) for value in [*row["true_candidate_labels"], *row["control_candidate_labels"]])
            for row in rows
        ),
        "single_token_boundaries": all(
            len(row["true_candidate_ids"]) == 2
            and len(row["control_candidate_ids"]) == 2
            and len(set(row["true_candidate_ids"] + row["control_candidate_ids"])) == 4
            for row in rows
        ),
        "control_derangement": all(row["concept_id"] != row["control_concept_id"] for row in rows),
        "case_digest": digest(rows) == case_digest,
        "finite_prompt_lengths": bool(prompt_lengths) and all(length > 0 and math.isfinite(length) for length in prompt_lengths),
        "no_model_outputs_read": True,
    }
    audit = {
        "schema_version": "phase1117_pythia_training_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "checks": audit_checks,
        "all_checks_passed": all(audit_checks.values()),
        "counts": {
            "concepts": len(selected),
            "cases": len(rows),
            "pairs": len(rows) // 2,
            "checkpoints": len(CHECKPOINTS),
            "split_concepts": dict(split_counts),
            "prompt_length_min": min(prompt_lengths),
            "prompt_length_max": max(prompt_lengths),
        },
    }
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"protocol audit failed: {audit_checks}")

    write_json(OUT_ROOT / "protocol" / "selected_concepts.json", selected_payload)
    write_jsonl(OUT_ROOT / "protocol" / "cases.jsonl", rows)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", preregistration)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    return {
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "selected_digest": selected_payload["selected_digest"],
        "case_digest": case_digest,
        "audit": audit,
    }


if __name__ == "__main__":
    print(json.dumps(build_protocol(), ensure_ascii=False, indent=2))
