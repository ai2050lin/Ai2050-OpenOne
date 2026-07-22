#!/usr/bin/env python3
"""Fail-closed, standard-library audit of Phase577 public preregistration data.

The audit deliberately duplicates the task truth registry and prompt templates.
It never imports the generator and installs path guards before reading artifacts.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import io
import itertools
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/glm5/result/phase577_gpt5_natural_behavior_protocol"
PROTOCOL_PATH = OUT_DIR / "phase577_preregistered_protocol.json"
DATASET_AUDIT_PATH = OUT_DIR / "phase577_dataset_audit.json"
STAGE_COMMIT_PATH = OUT_DIR / "phase577_stage_commit.json"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase577_sealed_commitment.json"
TOKENIZER_PATH = OUT_DIR / "phase577_tokenizer_precheck.json"
OUTPUT_PATH = OUT_DIR / "phase577_independent_audit.json"

OPEN_SPLITS = ("development", "confirmation", "heldout_novel_entities")
ALL_SPLITS = OPEN_SPLITS + ("sealed",)
MODELS = ("qwen3", "glm4", "deepseek7b")
RELATIONS = ("fruit_membership", "citrus_membership")
DESIGN_SEED = "phase577-gpt5-natural-behavior-denominator-v2"
EXPECTED_SOURCE_RELATIVES = {
    "tests/glm5/phase577_gpt5_natural_behavior_protocol.py",
    "tests/glm5/phase577_gpt5_natural_behavior_tokenizer_precheck.py",
    "tests/glm5/phase577_gpt5_natural_behavior_audit.py",
}
STAGE_PUBLIC_RELATIVES = (
    "phase577_development_cases.jsonl",
    "phase577_confirmation_cases.jsonl",
    "phase577_heldout_novel_entities_cases.jsonl",
    "phase577_preregistered_protocol.json",
    "phase577_dataset_audit.json",
    "phase577_sealed_commitment.json",
)
BASE_RESULT_RELATIVES = set(STAGE_PUBLIC_RELATIVES) | {
    "phase577_stage_commit.json",
    "protocol/private/phase577_sealed_cases.jsonl",
}
PRE_AUDIT_RELATIVES = BASE_RESULT_RELATIVES | {
    "phase577_tokenizer_precheck.json",
}
ALLOWED_VERIFICATION_RELATIVES = PRE_AUDIT_RELATIVES | {
    "phase577_independent_audit.json",
    "phase577_freeze_commit.json",
}
PRIVATE_FORBIDDEN_ROOTS = (
    OUT_DIR / "protocol/private",
    ROOT / "tests/glm5/result/phase576_gpt5_fruit_structure/protocol/private",
    ROOT / "tests/glm5/result/phase576r1_gpt5_fruit_structure/protocol/private",
    ROOT / "tests/glm5/result/phase576r2_gpt5_fruit_structure/protocol/private",
)
MODEL_DIRS = {
    "qwen3": ROOT / "models/hf/qwen3-4b",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b",
}

RELATION_SPECS = {
    "fruit_membership": {
        "label": "fruit",
        "contract_id": "culinary_fruit_membership.v2",
        "positive_classes": ["citrus", "noncitrus_fruit"],
        "negative_classes": [
            "nonfruit_food", "nonfruit_animal", "nonfruit_object",
        ],
    },
    "citrus_membership": {
        "label": "citrus fruit",
        "contract_id": "citrus_membership.v1",
        "positive_classes": ["citrus"],
        "negative_classes": ["noncitrus_fruit"],
    },
}

# This registry is intentionally independent of the generator source.  The sealed
# part is used only to check a public truth-registry commitment; private cases are
# never opened by this script.
TRUTH_REGISTRY: dict[str, dict[str, str]] = {
    "development": {
        "eureka lemon": "citrus", "persian lime": "citrus",
        "grapefruit": "citrus", "clementine": "citrus", "pomelo": "citrus",
        "satsuma": "citrus", "apple": "noncitrus_fruit",
        "pear": "noncitrus_fruit", "peach": "noncitrus_fruit",
        "pineapple": "noncitrus_fruit", "watermelon": "noncitrus_fruit",
        "papaya": "noncitrus_fruit", "potato": "nonfruit_food",
        "onion": "nonfruit_food", "sparrow": "nonfruit_animal",
        "eagle": "nonfruit_animal", "hammer": "nonfruit_object",
        "copper": "nonfruit_object",
    },
    "confirmation": {
        "navel orange": "citrus", "key lime": "citrus",
        "blood orange": "citrus", "yuzu": "citrus", "kumquat": "citrus",
        "dancy tangerine": "citrus", "raspberry": "noncitrus_fruit",
        "blackberry": "noncitrus_fruit", "plum": "noncitrus_fruit",
        "apricot": "noncitrus_fruit", "nectarine": "noncitrus_fruit",
        "lychee": "noncitrus_fruit", "celery": "nonfruit_food",
        "cabbage": "nonfruit_food", "robin": "nonfruit_animal",
        "falcon": "nonfruit_animal", "table": "nonfruit_object",
        "granite": "nonfruit_object",
    },
    "heldout_novel_entities": {
        "mandarin orange": "citrus", "meyer lemon": "citrus",
        "finger lime": "citrus", "calamansi": "citrus", "kabosu": "citrus",
        "sudachi": "citrus", "gooseberry": "noncitrus_fruit",
        "granadilla": "noncitrus_fruit", "black sapote": "noncitrus_fruit",
        "dragon fruit": "noncitrus_fruit", "passion fruit": "noncitrus_fruit",
        "jackfruit": "noncitrus_fruit", "spinach": "nonfruit_food",
        "lettuce": "nonfruit_food", "owl": "nonfruit_animal",
        "dolphin": "nonfruit_animal", "violin": "nonfruit_object",
        "silver": "nonfruit_object",
    },
    "sealed": {
        "valencia orange": "citrus", "rangpur lime": "citrus",
        "makrut lime": "citrus", "tangelo": "citrus",
        "minneola tangelo": "citrus", "oro blanco": "citrus",
        "honeydew melon": "noncitrus_fruit", "custard apple": "noncitrus_fruit",
        "breadfruit": "noncitrus_fruit", "rambutan": "noncitrus_fruit",
        "carambola": "noncitrus_fruit", "white currant": "noncitrus_fruit",
        "cauliflower": "nonfruit_food", "asparagus": "nonfruit_food",
        "rhinoceros": "nonfruit_animal", "salamander": "nonfruit_animal",
        "screwdriver": "nonfruit_object", "telescope": "nonfruit_object",
    },
}

DIRECT_TEMPLATES = (
    (0, "semantic_label_first", "Taxonomy task. Item: {focus}. Question: Is this item a {relation_label}? Begin the first line with YES or NO; a brief explanation may follow."),
    (0, "exact_short", "Taxonomy task. Item: {focus}. Question: Is this item a {relation_label}? Return exactly YES or NO and nothing else."),
    (1, "semantic_label_first", "Classify {focus} using everyday food taxonomy. Does it count as a {relation_label}? Put YES or NO first; an explanation is optional."),
    (1, "exact_short", "Classify {focus} using everyday food taxonomy. Does it count as a {relation_label}? Return exactly YES or NO and nothing else."),
    (2, "semantic_label_first", "For the named item {focus}, use everyday food taxonomy to decide whether it belongs to the class {relation_label}. Start with YES or NO; reasoning may follow."),
    (2, "exact_short", "For the named item {focus}, use everyday food taxonomy to decide whether it belongs to the class {relation_label}. Return exactly YES or NO and nothing else."),
)
SELECTION_TEMPLATES = (
    (0, "semantic_label_first", "Taxonomy comparison. Options: {left} | {right}. Which option is {polarity_phrase}? Begin with the option text; a brief explanation may follow."),
    (0, "exact_short", "Taxonomy comparison. Options: {left} | {right}. Which option is {polarity_phrase}? Return exactly one option and nothing else."),
    (1, "semantic_label_first", "Choose between {left} and {right}. Identify the one that is {polarity_phrase}. Put the chosen option first; explanation is optional."),
    (1, "exact_short", "Choose between {left} and {right}. Identify the one that is {polarity_phrase}. Return exactly one option and nothing else."),
)

EXPECTED_CASE_KEYS = {
    "schema_version", "phase_id", "case_id", "split", "sealed", "interface",
    "relation", "relation_contract_id", "output_contract", "surface_id",
    "paraphrase_id", "order", "query_polarity", "target_truth_polarity",
    "analysis_unit_id", "focus_object", "focus_object_class",
    "comparison_object", "comparison_object_class", "positive_object",
    "negative_object", "left_option", "right_option", "target", "foil",
    "candidate_groups", "raw_prompt", "normalized_prompt_sha256",
    "raw_role_char_spans", "candidate_layer", "candidate_head",
    "candidate_neuron", "candidate_direction", "candidate_mechanism_formula",
}

PRIOR_OPEN_SOURCES = (
    ("tests/gpt5/result/phase556_fruit_encoding/phase556_open_cases.jsonl", "jsonl", ("raw_prompt", "prompt")),
    ("tests/gpt5/result/phase557_fruit_composite/phase557_open_cases.jsonl", "jsonl", ("raw_prompt", "prompt")),
    ("tests/glm5/result/phase576r2_gpt5_fruit_structure/phase576_open_cases.jsonl", "jsonl", ("raw_prompt",)),
    ("tests/glm5/result/phase990_delayed_binding_protocol/dataset.json", "json_records", ("prompt",)),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_new(path: Path, payload: Any) -> None:
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if path.exists():
        raise RuntimeError(f"refusing to overwrite independent audit: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def confined_path(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be a non-empty string")
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != relative
    ):
        raise ValueError(f"unsafe artifact path: {relative!r}")
    base_resolved = base.resolve(strict=True)
    candidate = base.joinpath(*pure.parts)
    candidate.resolve(strict=True).relative_to(base_resolved)
    cursor = candidate
    while cursor != base:
        if cursor.is_symlink():
            raise ValueError(f"symlink forbidden in artifact path: {relative!r}")
        cursor = cursor.parent
    return candidate


def result_file_relatives() -> set[str]:
    if not OUT_DIR.is_dir() or OUT_DIR.is_symlink():
        raise RuntimeError("Phase577 result root is missing or is a symlink")
    result = set()
    allowed_directories = {"protocol", "protocol/private"}
    for path in OUT_DIR.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Phase577 result symlink is forbidden: {path}")
        if path.is_file():
            result.add(str(path.relative_to(OUT_DIR)).replace("\\", "/"))
        elif path.is_dir():
            relative = str(path.relative_to(OUT_DIR)).replace("\\", "/")
            if relative not in allowed_directories:
                raise RuntimeError(f"unexpected Phase577 result directory: {path}")
        else:
            raise RuntimeError(f"unsupported Phase577 result entry: {path}")
    return result


def verify_public_identity(identity: Any, expected_relative: str) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "path", "size_bytes", "sha256",
    } or identity.get("path") != expected_relative:
        return False
    try:
        path = confined_path(OUT_DIR, expected_relative)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        path.stat().st_size == identity.get("size_bytes")
        and sha256_file(path) == identity.get("sha256")
    )


def verify_root_identity(identity: Any, expected_relative: str) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "path", "size_bytes", "sha256", "is_symlink",
    } or identity.get("path") != expected_relative:
        return False
    try:
        path = confined_path(ROOT, expected_relative)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        identity.get("is_symlink") is False
        and path.stat().st_size == identity.get("size_bytes")
        and sha256_file(path) == identity.get("sha256")
    )


def source_import_roots(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.append((node.module or "").split(".", 1)[0])
    return sorted(set(roots))


@contextmanager
def forbid_private_content_opens() -> Iterator[list[str]]:
    attempts: list[str] = []
    originals = (builtins.open, io.open, os.open)

    def is_forbidden(file: Any) -> bool:
        if isinstance(file, int):
            return False
        try:
            candidate = Path(file).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return False
        return any(
            candidate == forbidden.resolve(strict=False)
            or forbidden.resolve(strict=False) in candidate.parents
            for forbidden in PRIVATE_FORBIDDEN_ROOTS
        )

    def wrap(original: Any) -> Any:
        def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
            if is_forbidden(file):
                label = os.fspath(file) if isinstance(file, (str, os.PathLike)) else ""
                attempts.append(label)
                raise RuntimeError(f"private sealed content open forbidden: {label}")
            return original(file, *args, **kwargs)
        return guarded

    builtins.open = wrap(originals[0])
    io.open = wrap(originals[1])
    os.open = wrap(originals[2])
    try:
        yield attempts
    finally:
        builtins.open, io.open, os.open = originals


def expected_spans(prompt: str, values: dict[str, str | None]) -> dict[str, Any]:
    spans = {}
    for key, value in values.items():
        if value is None:
            spans[key] = None
            continue
        starts = [match.start() for match in re.finditer(re.escape(value), prompt)]
        if not starts:
            raise RuntimeError(f"independent span source missing: {key}/{value}")
        start = starts[-1] if key == "query_anchor" else starts[0]
        spans[key] = {"start": start, "end": start + len(value), "text": value}
    return spans


def seeded_rotation(
    values: Iterable[str], split: str, relation: str, role: str
) -> list[str]:
    bank = list(values)
    digest = hashlib.sha256(
        f"{DESIGN_SEED}|{split}|{relation}|{role}".encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:4], "big") % len(bank)
    return bank[offset:] + bank[:offset]


def length_balanced_pairs(
    positives: Iterable[str], negatives: Iterable[str], split: str, relation: str
) -> list[tuple[str, str]]:
    positive_bank = seeded_rotation(positives, split, relation, "positive_objects")
    negative_bank = list(negatives)
    candidates = []
    for permutation in itertools.permutations(negative_bank):
        positive_longer = sum(
            len(positive) > len(negative)
            for positive, negative in zip(positive_bank, permutation)
        )
        negative_longer = sum(
            len(negative) > len(positive)
            for positive, negative in zip(positive_bank, permutation)
        )
        ties = 6 - positive_longer - negative_longer
        tie_break = hashlib.sha256(
            (
                f"{DESIGN_SEED}|{split}|{relation}|length_pairing|"
                + "|".join(permutation)
            ).encode("utf-8")
        ).hexdigest()
        candidates.append((
            (abs(positive_longer - negative_longer), -ties, tie_break),
            permutation,
        ))
    _, best = min(candidates, key=lambda item: item[0])
    return list(zip(positive_bank, best))


def validate_open_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    truth = TRUTH_REGISTRY[split]
    if len(rows) != 336 or len({row.get("case_id") for row in rows}) != 336:
        raise RuntimeError(f"{split}: case denominator or uniqueness drift")
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_objects = set()
    prompt_hashes = set()
    for row in rows:
        if set(row) != EXPECTED_CASE_KEYS:
            raise RuntimeError(f"{split}: exact case schema mismatch")
        if (
            row["schema_version"] != "phase577_gpt5_natural_behavior_case.v2"
            or row["phase_id"] != "Phase577"
            or row["split"] != split
            or row["sealed"] is not False
            or row["interface"] not in {"direct", "selection"}
            or row["relation"] not in RELATIONS
            or any(row[key] is not None for key in (
                "candidate_layer", "candidate_head", "candidate_neuron",
                "candidate_direction", "candidate_mechanism_formula",
            ))
        ):
            raise RuntimeError(f"{split}/{row['case_id']}: base identity drift")
        relation = row["relation"]
        spec = RELATION_SPECS[relation]
        if row["relation_contract_id"] != spec["contract_id"]:
            raise RuntimeError(f"{split}/{row['case_id']}: relation contract drift")

        if row["interface"] == "direct":
            focus = row["focus_object"]
            if focus not in truth:
                raise RuntimeError(f"{split}/{row['case_id']}: unknown direct object")
            object_class = truth[focus]
            positive = object_class in spec["positive_classes"]
            target = "yes" if positive else "no"
            surface = row["surface_id"]
            if not isinstance(surface, int) or surface not in range(6):
                raise RuntimeError(f"{split}/{row['case_id']}: direct surface drift")
            paraphrase, contract, template = DIRECT_TEMPLATES[surface]
            prompt = template.format(focus=focus, relation_label=spec["label"])
            unit = f"phase577_{split}_direct_{relation}_{focus.replace(' ', '_')}"
            expected = {
                "output_contract": contract,
                "paraphrase_id": paraphrase,
                "order": None,
                "query_polarity": "affirmative",
                "target_truth_polarity": "positive" if positive else "negative",
                "analysis_unit_id": unit,
                "focus_object_class": object_class,
                "comparison_object": None,
                "comparison_object_class": None,
                "positive_object": focus if positive else None,
                "negative_object": None if positive else focus,
                "left_option": None,
                "right_option": None,
                "target": target,
                "foil": "no" if target == "yes" else "yes",
                "candidate_groups": {"yes": ["yes"], "no": ["no"]},
                "raw_prompt": prompt,
                "case_id": f"{unit}_s{surface}",
            }
            spans = expected_spans(prompt, {
                "focus": focus, "comparison": None, "query_anchor": spec["label"],
            })
            observed_objects.add(focus)
        else:
            positive_object = row["positive_object"]
            negative_object = row["negative_object"]
            if positive_object not in truth or negative_object not in truth:
                raise RuntimeError(f"{split}/{row['case_id']}: unknown selection object")
            positive_class = truth[positive_object]
            negative_class = truth[negative_object]
            if (
                positive_class not in spec["positive_classes"]
                or negative_class not in spec["negative_classes"]
                or positive_object == negative_object
            ):
                raise RuntimeError(f"{split}/{row['case_id']}: selection truth drift")
            surface = row["surface_id"]
            order = row["order"]
            polarity = row["query_polarity"]
            if (
                not isinstance(surface, int) or surface not in range(4)
                or order not in {0, 1} or polarity not in {"positive", "negative"}
            ):
                raise RuntimeError(f"{split}/{row['case_id']}: selection grid drift")
            paraphrase, contract, template = SELECTION_TEMPLATES[surface]
            left, right = (
                (positive_object, negative_object)
                if order == 0 else (negative_object, positive_object)
            )
            target = positive_object if polarity == "positive" else negative_object
            foil = negative_object if polarity == "positive" else positive_object
            phrase = (
                f"a {spec['label']}" if polarity == "positive"
                else f"not a {spec['label']}"
            )
            prompt = template.format(left=left, right=right, polarity_phrase=phrase)
            unit = row["analysis_unit_id"]
            pattern = rf"phase577_{re.escape(split)}_selection_{relation}_pair0[0-5]"
            if not isinstance(unit, str) or re.fullmatch(pattern, unit) is None:
                raise RuntimeError(f"{split}/{row['case_id']}: selection unit ID drift")
            expected = {
                "output_contract": contract,
                "paraphrase_id": paraphrase,
                "target_truth_polarity": polarity,
                "focus_object": target,
                "focus_object_class": truth[target],
                "comparison_object": foil,
                "comparison_object_class": truth[foil],
                "left_option": left,
                "right_option": right,
                "target": target,
                "foil": foil,
                "candidate_groups": {
                    positive_object: [positive_object],
                    negative_object: [negative_object],
                },
                "raw_prompt": prompt,
                "case_id": f"{unit}_s{surface}_o{order}_q{polarity[0]}",
            }
            spans = expected_spans(prompt, {
                "focus": target, "comparison": foil, "query_anchor": phrase,
            })
            observed_objects.update((positive_object, negative_object))
        for key, value in expected.items():
            if row.get(key) != value:
                raise RuntimeError(
                    f"{split}/{row['case_id']}: independently reconstructed {key} drift"
                )
        if row["raw_role_char_spans"] != spans:
            raise RuntimeError(f"{split}/{row['case_id']}: role span drift")
        expected_hash = hashlib.sha256(
            normalize_prompt(row["raw_prompt"]).encode("utf-8")
        ).hexdigest()
        if row["normalized_prompt_sha256"] != expected_hash:
            raise RuntimeError(f"{split}/{row['case_id']}: prompt hash drift")
        prompt_hashes.add(expected_hash)
        by_unit[row["analysis_unit_id"]].append(row)

    if observed_objects != set(truth) or len(prompt_hashes) != 336 or len(by_unit) != 36:
        raise RuntimeError(f"{split}: object, prompt, or unit denominator drift")
    strata = Counter((row["interface"], row["relation"]) for row in rows)
    expected_strata = {
        ("direct", "fruit_membership"): 72,
        ("direct", "citrus_membership"): 72,
        ("selection", "fruit_membership"): 96,
        ("selection", "citrus_membership"): 96,
    }
    if strata != expected_strata:
        raise RuntimeError(f"{split}: stratum drift")
    family_units = Counter(
        (bank[0]["interface"], bank[0]["relation"]) for bank in by_unit.values()
    )
    if family_units != {
        ("direct", "fruit_membership"): 12,
        ("direct", "citrus_membership"): 12,
        ("selection", "fruit_membership"): 6,
        ("selection", "citrus_membership"): 6,
    }:
        raise RuntimeError(f"{split}: family unit drift")
    for bank in by_unit.values():
        interface = bank[0]["interface"]
        relation = bank[0]["relation"]
        if any(
            row["interface"] != interface or row["relation"] != relation
            for row in bank
        ):
            raise RuntimeError(f"{split}: mixed analysis unit")
        if interface == "direct":
            grid = Counter(
                (row["paraphrase_id"], row["output_contract"]) for row in bank
            )
            if len(bank) != 6 or grid != {
                (paraphrase, contract): 1
                for paraphrase in range(3)
                for contract in ("semantic_label_first", "exact_short")
            }:
                raise RuntimeError(f"{split}: direct factorial grid drift")
        else:
            positive_pairs = {(row["positive_object"], row["negative_object"]) for row in bank}
            grid = Counter((
                row["paraphrase_id"], row["output_contract"], row["order"],
                row["query_polarity"],
            ) for row in bank)
            if len(bank) != 16 or len(positive_pairs) != 1 or grid != {
                (paraphrase, contract, order, polarity): 1
                for paraphrase in range(2)
                for contract in ("semantic_label_first", "exact_short")
                for order in (0, 1)
                for polarity in ("positive", "negative")
            }:
                raise RuntimeError(f"{split}: selection factorial grid drift")
    objects_by_class = {
        object_class: [
            label for label, registered_class in truth.items()
            if registered_class == object_class
        ]
        for object_class in {
            "citrus", "noncitrus_fruit", "nonfruit_food",
            "nonfruit_animal", "nonfruit_object",
        }
    }
    expected_pair_inputs = {
        "fruit_membership": (
            objects_by_class["citrus"][3:]
            + objects_by_class["noncitrus_fruit"][3:],
            objects_by_class["nonfruit_food"]
            + objects_by_class["nonfruit_animal"]
            + objects_by_class["nonfruit_object"],
        ),
        "citrus_membership": (
            objects_by_class["citrus"], objects_by_class["noncitrus_fruit"],
        ),
    }
    for relation, (positives, negatives) in expected_pair_inputs.items():
        expected_pairs = set(length_balanced_pairs(
            positives, negatives, split, relation
        ))
        representatives = {
            unit_id: bank[0]
            for unit_id, bank in by_unit.items()
            if bank[0]["interface"] == "selection"
            and bank[0]["relation"] == relation
        }
        observed_pairs = {
            (row["positive_object"], row["negative_object"])
            for row in representatives.values()
        }
        if (
            len(representatives) != 6
            or len({row["positive_object"] for row in representatives.values()}) != 6
            or len({row["negative_object"] for row in representatives.values()}) != 6
            or observed_pairs != expected_pairs
        ):
            raise RuntimeError(f"{split}/{relation}: selection pairing drift")
    for relation in RELATIONS:
        direct = [r for r in rows if r["interface"] == "direct" and r["relation"] == relation]
        selection = [r for r in rows if r["interface"] == "selection" and r["relation"] == relation]
        if Counter(r["target"] for r in direct) != {"yes": 36, "no": 36}:
            raise RuntimeError(f"{split}/{relation}: direct label imbalance")
        if Counter(r["query_polarity"] for r in selection) != {
            "positive": 48, "negative": 48,
        }:
            raise RuntimeError(f"{split}/{relation}: selection polarity imbalance")
        if Counter(
            "left" if r["target"] == r["left_option"] else "right"
            for r in selection
        ) != {"left": 48, "right": 48}:
            raise RuntimeError(f"{split}/{relation}: target-position imbalance")
    return {
        "case_count": 336,
        "analysis_unit_count": 36,
        "unique_prompt_count": 336,
        "object_count": 18,
        "object_labels": sorted(observed_objects),
        "strata": {"|".join(key): value for key, value in sorted(strata.items())},
        "truth_target_template_span_error_count": 0,
    }


def shortcut_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    markers = {"orange", "lemon", "lime", "grapefruit"}

    def marked(label: str) -> bool:
        return bool(markers & set(re.findall(r"[a-z0-9]+", label.casefold())))

    direct_citrus = {
        row["analysis_unit_id"]: row for row in rows
        if row["interface"] == "direct" and row["relation"] == "citrus_membership"
    }
    marker_direct = sum(
        ("yes" if marked(row["focus_object"]) else "no") == row["target"]
        for row in direct_citrus.values()
    )
    direct_fruit = {
        row["analysis_unit_id"]: row for row in rows
        if row["interface"] == "direct" and row["relation"] == "fruit_membership"
    }
    foodness = sum(
        ("no" if row["focus_object_class"] in {
            "nonfruit_animal", "nonfruit_object",
        } else "yes") == row["target"]
        for row in direct_fruit.values()
    )
    foodness_food = sum(
        "yes" == row["target"] for row in direct_fruit.values()
        if row["focus_object_class"] == "nonfruit_food"
    )
    length = {}
    marker_selection_resolved = 0
    marker_selection_correct = 0
    for relation in RELATIONS:
        units = {
            row["analysis_unit_id"]: row for row in rows
            if row["interface"] == "selection" and row["relation"] == relation
        }
        positive_longer = sum(
            len(row["positive_object"]) > len(row["negative_object"])
            for row in units.values()
        )
        negative_longer = sum(
            len(row["negative_object"]) > len(row["positive_object"])
            for row in units.values()
        )
        length[relation] = {
            "positive_longer_correct_units": positive_longer,
            "negative_longer_incorrect_units": negative_longer,
            "equal_length_unresolved_units": 6 - positive_longer - negative_longer,
            "can_meet_selection_gate_of_5": positive_longer >= 5,
        }
        if relation == "citrus_membership":
            for row in units.values():
                positive_marked = marked(row["positive_object"])
                negative_marked = marked(row["negative_object"])
                if positive_marked != negative_marked:
                    marker_selection_resolved += 1
                    marker_selection_correct += positive_marked
    any_pass = (
        marker_direct >= 10
        or marker_selection_correct >= 5
        or any(item["can_meet_selection_gate_of_5"] for item in length.values())
        or (foodness >= 10 and foodness_food >= 2)
    )
    if any_pass:
        raise RuntimeError("an independently audited deterministic shortcut can pass")
    return {
        "citrus_marker_direct_correct_units_of_12": marker_direct,
        "citrus_marker_selection_resolved_units_of_6": marker_selection_resolved,
        "citrus_marker_selection_correct_units_of_6": marker_selection_correct,
        "fruit_foodness_correct_units_of_12": foodness,
        "fruit_foodness_nonfruit_food_correct_units_of_2": foodness_food,
        "selection_character_length": length,
        "any_audited_shortcut_can_pass_relevant_gate_chain": False,
    }


def iter_prior(path: Path, kind: str, fields: tuple[str, ...]) -> Iterable[str]:
    if kind == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in fields:
                    if isinstance(row.get(field), str) and row[field]:
                        yield row[field]
                        break
    elif kind == "json_records":
        for row in read_json(path).get("records", []):
            for field in fields:
                if isinstance(row.get(field), str) and row[field]:
                    yield row[field]
                    break
    else:
        raise RuntimeError(f"unsupported prior source: {kind}")


def verify_tokenizer_registry(protocol: dict[str, Any]) -> bool:
    registry = protocol.get("tokenizer_input_identities")
    if not isinstance(registry, dict) or set(registry) != set(MODELS):
        return False
    for model in MODELS:
        entry = registry.get(model)
        directory = MODEL_DIRS[model]
        if not isinstance(entry, dict) or set(entry) != {
            "entry_path", "entry_is_symlink", "resolved_directory", "files",
        }:
            return False
        if (
            entry["entry_path"] != str(directory.relative_to(ROOT)).replace("\\", "/")
            or entry["entry_is_symlink"] is not directory.is_symlink()
            or entry["resolved_directory"]
            != str(directory.resolve(strict=True)).replace("\\", "/")
            or not isinstance(entry["files"], dict)
            or not {"tokenizer_config.json", "tokenizer.json"}.issubset(entry["files"])
        ):
            return False
        for name, identity in entry["files"].items():
            path = directory / name
            if (
                not isinstance(identity, dict)
                or set(identity) != {
                    "path", "resolved_path", "size_bytes", "sha256", "leaf_is_symlink",
                }
                or identity["path"] != f"{entry['entry_path']}/{name}"
                or identity["resolved_path"]
                != str(path.resolve(strict=True)).replace("\\", "/")
                or identity["leaf_is_symlink"] is not path.is_symlink()
                or identity["size_bytes"] != path.stat().st_size
                or identity["sha256"] != sha256_file(path)
            ):
                return False
    return protocol.get("tokenizer_input_registry_sha256") == stable_hash(registry)


def validate_tokenizer_payload(
    payload: dict[str, Any], protocol: dict[str, Any], stage: dict[str, Any]
) -> bool:
    reports = payload.get("reports")
    public = stage.get("public_artifact_identities", {})
    expected_inputs = {
        split: public.get(f"phase577_{split}_cases.jsonl") for split in OPEN_SPLITS
    }
    rows = [
        row for split in OPEN_SPLITS
        for row in read_jsonl(OUT_DIR / f"phase577_{split}_cases.jsonl")
    ]
    return (
        payload.get("schema_version") == "phase577_tokenizer_precheck.v2"
        and payload.get("phase_id") == "Phase577"
        and payload.get("passed") is True
        and payload.get("models_in_observed_order") == list(MODELS)
        and payload.get("open_case_count") == 1008
        and payload.get("input_case_artifact_identities") == expected_inputs
        and payload.get("input_case_id_registry_sha256") == stable_hash(
            [row["case_id"] for row in rows]
        )
        and isinstance(reports, list) and len(reports) == 3
        and [report.get("model") for report in reports] == list(MODELS)
        and all(
            report.get("model_order_index") == index
            and report.get("case_count") == 1008
            and report.get("split_counts") == {split: 336 for split in OPEN_SPLITS}
            and 0 < report.get("prompt_token_min", 0)
            <= report.get("prompt_token_max", 0) <= 512
            and 0 < report.get("candidate_token_min", 0)
            <= report.get("candidate_token_max", 0) <= 8
            and report.get("weight_file_open_attempt_count") == 0
            for index, report in enumerate(reports)
        )
        and payload.get("protocol_sha256") == sha256_file(PROTOCOL_PATH)
        and payload.get("stage_commit_sha256") == sha256_file(STAGE_COMMIT_PATH)
        and payload.get("tokenizer_source_sha256") == sha256_file(
            ROOT / "tests/glm5/phase577_gpt5_natural_behavior_tokenizer_precheck.py"
        )
        and payload.get("tokenizer_input_registry_sha256")
        == protocol.get("tokenizer_input_registry_sha256")
        and payload.get("tokenizer_inputs_verified") is True
        and payload.get("cuda_visible_devices") == ""
        and payload.get("cuda_used") is False
        and payload.get("model_weights_loaded") is False
        and payload.get("weight_file_open_attempts") == []
        and payload.get("unregistered_model_input_attempts") == []
        and payload.get("private_content_open_attempts") == []
        and payload.get("sealed_payload_read") is False
        and payload.get("old_phase576_sealed_payload_read") is False
    )


def recompute_open_evidence() -> dict[str, Any]:
    split_reports = {}
    shortcut_reports = {}
    all_rows = []
    object_sets = {}
    lexical_sets = {}
    for split in OPEN_SPLITS:
        rows = read_jsonl(OUT_DIR / f"phase577_{split}_cases.jsonl")
        report = validate_open_rows(rows, split)
        split_reports[split] = report
        shortcut_reports[split] = shortcut_report(rows)
        all_rows.extend(rows)
        object_sets[split] = set(report["object_labels"])
        lexical_sets[split] = {
            atom for label in object_sets[split]
            for atom in re.findall(r"[a-z0-9]+", label.casefold())
        }
    if len({row["case_id"] for row in all_rows}) != 1008:
        raise RuntimeError("Phase577 open case IDs are not globally unique")
    if len({row["normalized_prompt_sha256"] for row in all_rows}) != 1008:
        raise RuntimeError("Phase577 open prompts are not globally unique")
    exact_intersections = {}
    lexical_intersections = {}
    for index, left in enumerate(OPEN_SPLITS):
        for right in OPEN_SPLITS[index + 1:]:
            exact = sorted(object_sets[left] & object_sets[right])
            if exact:
                raise RuntimeError(f"Phase577 open object leakage: {left}/{right}")
            key = f"{left}|{right}"
            exact_intersections[key] = exact
            lexical_intersections[key] = sorted(lexical_sets[left] & lexical_sets[right])
    return {
        "rows": all_rows,
        "split_reports": split_reports,
        "shortcut_reports": shortcut_reports,
        "exact_intersections": exact_intersections,
        "lexical_intersections": lexical_intersections,
    }


def build_audit_payload() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    stage = read_json(STAGE_COMMIT_PATH)
    dataset_audit = read_json(DATASET_AUDIT_PATH)
    sealed_commitment = read_json(SEALED_COMMITMENT_PATH)
    tokenizer = read_json(TOKENIZER_PATH)
    source_registry = protocol.get("source_identities")
    if not isinstance(source_registry, dict) or set(source_registry) != EXPECTED_SOURCE_RELATIVES:
        raise RuntimeError("Phase577 source registry path set drift")
    source_checks = {
        relative: verify_root_identity(source_registry.get(relative), relative)
        for relative in EXPECTED_SOURCE_RELATIVES
    }
    if not all(source_checks.values()):
        raise RuntimeError("Phase577 source identity drift")
    source_imports = {
        relative: source_import_roots(ROOT / relative)
        for relative in EXPECTED_SOURCE_RELATIVES
    }
    for relative in (
        "tests/glm5/phase577_gpt5_natural_behavior_protocol.py",
        "tests/glm5/phase577_gpt5_natural_behavior_audit.py",
    ):
        if {"torch", "transformers"} & set(source_imports[relative]):
            raise RuntimeError(f"Phase577 CPU source imports model libraries: {relative}")
    tokenizer_source = (
        ROOT / "tests/glm5/phase577_gpt5_natural_behavior_tokenizer_precheck.py"
    ).read_text(encoding="utf-8")
    if "AutoModel" in tokenizer_source or "import torch" in tokenizer_source:
        raise RuntimeError("Phase577 tokenizer source contains model-loading code")

    public = stage.get("public_artifact_identities")
    if (
        stage.get("schema_version") != "phase577_stage_commit.v2"
        or stage.get("phase_id") != "Phase577"
        or stage.get("stage_complete") is not True
        or not isinstance(public, dict)
        or set(public) != set(STAGE_PUBLIC_RELATIVES)
        or not all(
            verify_public_identity(public.get(relative), relative)
            for relative in STAGE_PUBLIC_RELATIVES
        )
        or stage.get("source_identities") != source_registry
        or stage.get("candidate_coordinates") != []
        or stage.get("candidate_mechanism_formulas") != []
        or stage.get("gpu_used") is not False
        or stage.get("model_weights_loaded") is not False
        or stage.get("sealed_payload_path")
        != "protocol/private/phase577_sealed_cases.jsonl"
    ):
        raise RuntimeError("Phase577 public stage chain drift")

    evidence = recompute_open_evidence()
    all_rows = evidence["rows"]
    new_prompts = {normalize_prompt(row["raw_prompt"]) for row in all_rows}
    prior_reports = []
    total_prompt_overlap = set()
    for relative, kind, fields in PRIOR_OPEN_SOURCES:
        path = ROOT / relative
        prior = {normalize_prompt(value) for value in iter_prior(path, kind, fields)}
        overlap = new_prompts & prior
        total_prompt_overlap.update(overlap)
        prior_reports.append({
            "path": relative,
            "sha256": sha256_file(path),
            "normalized_prompt_count": len(prior),
            "overlap_count": len(overlap),
        })
    if total_prompt_overlap:
        raise RuntimeError("Phase577 prior-prompt overlap found")

    phase576_path = ROOT / (
        "tests/glm5/result/phase576r2_gpt5_fruit_structure/phase576_open_cases.jsonl"
    )
    phase576_objects = set()
    for row in read_jsonl(phase576_path):
        for key in ("focus_object_label", "comparison_object_label"):
            if isinstance(row.get(key), str) and row[key]:
                phase576_objects.add(row[key].casefold())
    new_objects = {
        label.casefold() for split in OPEN_SPLITS for label in TRUTH_REGISTRY[split]
    }
    exact_phase576_overlap = sorted(phase576_objects & new_objects)
    if exact_phase576_overlap:
        raise RuntimeError("Phase577 exact objects overlap Phase576 open objects")
    old_atoms = {
        atom for label in phase576_objects for atom in re.findall(r"[a-z0-9]+", label)
    }
    new_atoms = {
        atom for label in new_objects for atom in re.findall(r"[a-z0-9]+", label)
    }
    phase576_object_report = {
        "phase576_open_path": str(phase576_path.relative_to(ROOT)).replace("\\", "/"),
        "phase576_open_sha256": sha256_file(phase576_path),
        "phase576_open_object_count": len(phase576_objects),
        "phase577_open_object_count": len(new_objects),
        "exact_casefold_overlap": exact_phase576_overlap,
        "whole_token_lexical_atom_overlap": sorted(old_atoms & new_atoms),
        "exact_disjoint_does_not_mean_lexically_disjoint": True,
    }

    template_registry = {
        "direct": [
            {"paraphrase_id": p, "contract": c, "text": t}
            for p, c, t in DIRECT_TEMPLATES
        ],
        "selection": [
            {"paraphrase_id": p, "contract": c, "text": t}
            for p, c, t in SELECTION_TEMPLATES
        ],
    }
    expected_gate = {
        "primary_metric": "both_repeats_semantic_prefix_correct",
        "direct_unit_minimum_stable_surfaces_of_6": 5,
        "selection_unit_minimum_stable_cases_of_16": 14,
        "selection_unit_minimum_each_query_polarity_of_8": 7,
        "minimum_stable_fruit_direct_units_of_12": 10,
        "minimum_stable_citrus_direct_units_of_12": 10,
        "minimum_stable_fruit_selection_units_of_6": 5,
        "minimum_stable_citrus_selection_units_of_6": 5,
        "fruit_direct_nonfruit_food_units_required_of_2": 2,
        "fruit_selection_nonfruit_food_pairs_required_of_2": 2,
        "minimum_passing_analysis_units_of_36": 30,
        "case_level_semantic_stable_micro_rate_secondary_floor": 0.85,
        "semantic_correctness_required_in_both_repeats": True,
        "full_generated_identity_required_for_semantic_gate": False,
    }
    gate = protocol.get("behavior_gate", {})
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise RuntimeError("Phase577 behavior gate drift")
    if protocol.get("template_registry") != template_registry:
        raise RuntimeError("Phase577 template registry differs from independent copy")
    if protocol.get("relation_specs") != RELATION_SPECS:
        raise RuntimeError("Phase577 relation contracts differ from independent copy")

    sealed_truth_hash = stable_hash(TRUTH_REGISTRY["sealed"])
    if (
        sealed_commitment.get("schema_version") != "phase577_sealed_commitment.v2"
        or sealed_commitment.get("phase_id") != "Phase577"
        or sealed_commitment.get("sealed_case_count") != 336
        or sealed_commitment.get("sealed_truth_registry_entry_count") != 18
        or sealed_commitment.get("sealed_truth_registry_sha256") != sealed_truth_hash
        or sealed_commitment.get("sealed_payload_path")
        != "protocol/private/phase577_sealed_cases.jsonl"
        or sealed_commitment.get("not_external_blind") is not True
        or sealed_commitment.get("candidate_coordinates_frozen") != []
        or sealed_commitment.get("candidate_mechanism_formulas_frozen") != []
        or sealed_commitment.get("prior_phase576_sealed_payload_read") is not False
    ):
        raise RuntimeError("Phase577 sealed commitment/truth registry drift")
    private_path = OUT_DIR / "protocol/private/phase577_sealed_cases.jsonl"
    if private_path.stat().st_size != sealed_commitment.get("sealed_payload_size_bytes"):
        raise RuntimeError("Phase577 private sealed metadata size drift")

    attachment = protocol.get("attachment_identity", {})
    previous = protocol.get("attachment_comparison", {}).get("previous_identity", {})
    attachment_path = Path(attachment.get("path", "__missing__"))
    previous_path = Path(previous.get("path", "__missing__"))
    attachment_valid = (
        attachment_path.is_file() and previous_path.is_file()
        and attachment_path.stat().st_size == attachment.get("size_bytes")
        and previous_path.stat().st_size == previous.get("size_bytes")
        and sha256_file(attachment_path) == attachment.get("sha256")
        and sha256_file(previous_path) == previous.get("sha256")
        and attachment.get("sha256") == previous.get("sha256")
    )
    legacy = protocol.get("legacy_phase577_collision", {})
    pilot = protocol.get("phase576_pilot_artifact_identities", {})
    tokenizer_valid = validate_tokenizer_payload(tokenizer, protocol, stage)
    tokenizer_registry_valid = verify_tokenizer_registry(protocol)
    checks = {
        "protocol_schema_phase": protocol.get("schema_version")
        == "phase577_gpt5_natural_behavior_protocol.v2"
        and protocol.get("phase_id") == "Phase577",
        "model_order": protocol.get("models_in_required_order") == list(MODELS),
        "split_contract": protocol.get("open_splits") == list(OPEN_SPLITS)
        and protocol.get("splits") == list(ALL_SPLITS),
        "templates_independently_reconstructed": protocol.get("template_registry")
        == template_registry,
        "truth_targets_prompts_spans_independently_reconstructed": all(
            report["truth_target_template_span_error_count"] == 0
            for report in evidence["split_reports"].values()
        ),
        "open_denominator_exact": len(all_rows) == 1008,
        "exact_object_split_isolation": all(
            not overlap for overlap in evidence["exact_intersections"].values()
        ),
        "lexical_overlap_reported_not_hidden": isinstance(
            evidence["lexical_intersections"], dict
        ),
        "prior_prompt_overlap_zero": not total_prompt_overlap,
        "phase576_exact_object_overlap_zero": not exact_phase576_overlap,
        "deterministic_shortcuts_fail_gates": all(
            report["any_audited_shortcut_can_pass_relevant_gate_chain"] is False
            for report in evidence["shortcut_reports"].values()
        ),
        "source_seal": all(source_checks.values()),
        "protocol_and_audit_no_model_imports": all(
            not ({"torch", "transformers"} & set(source_imports[relative]))
            for relative in (
                "tests/glm5/phase577_gpt5_natural_behavior_protocol.py",
                "tests/glm5/phase577_gpt5_natural_behavior_audit.py",
            )
        ),
        "tokenizer_inputs_frozen": tokenizer_registry_valid,
        "tokenizer_precheck_qualified": tokenizer_valid,
        "dataset_audit_consistent": dataset_audit.get("valid") is True
        and dataset_audit.get("global_case_count") == 1344
        and dataset_audit.get("global_unique_case_id_count") == 1344
        and dataset_audit.get("global_unique_normalized_prompt_count") == 1344
        and dataset_audit.get("sealed_rows_generated_and_validated_in_memory") is True
        and dataset_audit.get("private_sealed_file_reopened_during_bundle_build") is False
        and set(dataset_audit.get("deterministic_shortcut_audits", {}))
        == set(ALL_SPLITS)
        and all(
            report.get(
                "shortcut_can_pass_all_relevant_family_and_subgroup_gates"
            ) is False
            for report in dataset_audit.get(
                "deterministic_shortcut_audits", {}
            ).values()
        ),
        "sealed_truth_registry_independently_committed": (
            sealed_commitment.get("sealed_truth_registry_sha256") == sealed_truth_hash
        ),
        "sealed_private_file_content_not_opened": True,
        "old_phase576_sealed_content_not_opened": True,
        "attachment_duplicate_has_no_new_evidence": attachment_valid
        and protocol.get("attachment_has_new_evidence") is False,
        "phase576_is_pilot_only": protocol.get("phase576_role")
        == "pilot_and_hypothesis_generation_only_not_phase577_evidence",
        "legacy_phase577_excluded": legacy.get("status")
        == "excluded_not_executed_not_imported"
        and verify_root_identity(legacy.get("identity"), "tests/glm5/phase577_retrieval_circuit.py"),
        "pilot_artifact_identities": isinstance(pilot, dict) and len(pilot) == 3
        and all(verify_root_identity(identity, identity.get("path", "")) for identity in pilot.values()),
        "model_registry_identity": verify_root_identity(
            protocol.get("model_registry_identity"), "tests/gpt5/model_registry.py"
        ),
        "no_internal_candidates": protocol.get("candidate_coordinates_before_trace") == []
        and protocol.get("candidate_mechanism_formulas_before_trace") == []
        and protocol.get("internal_activation_access_authorized") is False
        and protocol.get("causal_intervention_authorized") is False,
        "gpu_requires_future_runner_scorer_freeze": protocol.get(
            "gpu_behavior_authorized_by_phase577_final_freeze"
        ) is False and protocol.get(
            "future_gpu_behavior_requires_separate_runner_and_scorer_freeze"
        ) is True,
        "no_combined_open_case_file": "phase577_open_cases.jsonl"
        not in result_file_relatives(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase577 independent audit checks failed: {checks}")
    input_identities = {
        split: public[f"phase577_{split}_cases.jsonl"] for split in OPEN_SPLITS
    }
    return {
        "schema_version": "phase577_independent_audit.v2",
        "phase_id": "Phase577",
        "created_at_utc": now(),
        "passed": True,
        "checks": checks,
        "split_reports": evidence["split_reports"],
        "deterministic_shortcut_reports": evidence["shortcut_reports"],
        "open_split_object_intersections": evidence["exact_intersections"],
        "open_split_lexical_atom_intersections": evidence["lexical_intersections"],
        "phase576_object_report": phase576_object_report,
        "prior_prompt_reports": prior_reports,
        "source_import_roots": source_imports,
        "input_case_artifact_identities": input_identities,
        "input_case_id_registry_sha256": stable_hash(
            [row["case_id"] for row in all_rows]
        ),
        "sealed_truth_registry_sha256": sealed_truth_hash,
        "tokenizer_precheck_sha256": sha256_file(TOKENIZER_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "dataset_audit_sha256": sha256_file(DATASET_AUDIT_PATH),
        "stage_commit_sha256": sha256_file(STAGE_COMMIT_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "audit_source_sha256": sha256_file(Path(__file__).resolve()),
        "files_written_before_output": False,
        "cuda_used": False,
        "model_weights_loaded": False,
        "private_content_open_attempts": [],
        "sealed_payload_read": False,
        "old_phase576_sealed_payload_read": False,
    }


def run() -> dict[str, Any]:
    if result_file_relatives() != PRE_AUDIT_RELATIVES:
        raise RuntimeError("Phase577 independent audit requires exact pre-audit files")
    with forbid_private_content_opens() as attempts:
        payload = build_audit_payload()
        if attempts:
            raise RuntimeError("Phase577 audit attempted to open sealed content")
        write_new(OUTPUT_PATH, payload)
    return payload


def verify() -> dict[str, Any]:
    observed = result_file_relatives()
    if (
        not (PRE_AUDIT_RELATIVES | {"phase577_independent_audit.json"}).issubset(observed)
        or not observed.issubset(ALLOWED_VERIFICATION_RELATIVES)
    ):
        raise RuntimeError("Phase577 stored audit result-file allowlist drift")
    with forbid_private_content_opens() as attempts:
        stored = read_json(OUTPUT_PATH)
        recomputed = build_audit_payload()
        evidence = recompute_open_evidence()
        expected_keys = {
            "schema_version", "phase_id", "created_at_utc", "passed", "checks",
            "split_reports", "deterministic_shortcut_reports",
            "open_split_object_intersections", "open_split_lexical_atom_intersections",
            "phase576_object_report", "prior_prompt_reports", "source_import_roots",
            "input_case_artifact_identities", "input_case_id_registry_sha256",
            "sealed_truth_registry_sha256", "tokenizer_precheck_sha256",
            "protocol_sha256", "dataset_audit_sha256", "stage_commit_sha256",
            "sealed_commitment_sha256", "audit_source_sha256",
            "files_written_before_output", "cuda_used", "model_weights_loaded",
            "private_content_open_attempts", "sealed_payload_read",
            "old_phase576_sealed_payload_read",
        }
        checks = {
            "exact_schema": set(stored) == expected_keys,
            "schema_phase_pass": stored.get("schema_version")
            == "phase577_independent_audit.v2"
            and stored.get("phase_id") == "Phase577"
            and stored.get("passed") is True,
            "all_original_checks": isinstance(stored.get("checks"), dict)
            and stored.get("checks") == recomputed.get("checks")
            and bool(stored["checks"]) and all(stored["checks"].values()),
            "all_deterministic_fields_recomputed": all(
                stored.get(key) == recomputed.get(key)
                for key in expected_keys - {"created_at_utc"}
            ),
            "created_at_present": isinstance(stored.get("created_at_utc"), str)
            and bool(stored.get("created_at_utc")),
            "split_reports_recomputed": stored.get("split_reports")
            == evidence["split_reports"],
            "shortcut_reports_recomputed": stored.get("deterministic_shortcut_reports")
            == evidence["shortcut_reports"],
            "object_intersections_recomputed": stored.get(
                "open_split_object_intersections"
            ) == evidence["exact_intersections"],
            "lexical_intersections_recomputed": stored.get(
                "open_split_lexical_atom_intersections"
            ) == evidence["lexical_intersections"],
            "protocol_chain": stored.get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
            "dataset_chain": stored.get("dataset_audit_sha256")
            == sha256_file(DATASET_AUDIT_PATH),
            "stage_chain": stored.get("stage_commit_sha256")
            == sha256_file(STAGE_COMMIT_PATH),
            "commitment_chain": stored.get("sealed_commitment_sha256")
            == sha256_file(SEALED_COMMITMENT_PATH),
            "tokenizer_chain": stored.get("tokenizer_precheck_sha256")
            == sha256_file(TOKENIZER_PATH),
            "source_chain": stored.get("audit_source_sha256")
            == sha256_file(Path(__file__).resolve()),
            "sealed_truth_chain": stored.get("sealed_truth_registry_sha256")
            == stable_hash(TRUTH_REGISTRY["sealed"]),
            "no_gpu_weights": stored.get("cuda_used") is False
            and stored.get("model_weights_loaded") is False,
            "sealed_unread": stored.get("private_content_open_attempts") == []
            and stored.get("sealed_payload_read") is False
            and stored.get("old_phase576_sealed_payload_read") is False,
        }
        if attempts or not all(checks.values()):
            raise RuntimeError(f"Phase577 stored audit verification failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "files_written": False,
        "cuda_used": False,
        "model_weights_loaded": False,
        "sealed_payload_read": False,
        "independent_audit_sha256": sha256_file(OUTPUT_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run() if args.run else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
