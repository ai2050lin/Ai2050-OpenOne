#!/usr/bin/env python3
"""Freeze the Phase1024 lexical-semantic orthogonal atlas protocol.

The protocol deliberately avoids a mechanism formula.  It creates three
independent observational panels:

1. a balanced nonce-surface x assigned-concept factorial,
2. same-surface / different-sense polysemy contexts, and
3. different-surface / related-concept synonym groups.

Discovery selects physical regions.  Confirmation uses new templates and,
where possible, new concepts, nonce surfaces, and lexical groups.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import render_chat, tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans


PHASE = 1024
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
SPLITS = ("discovery", "confirmation")
ROLES = ("anchor_end", "focus_end", "pre_output")
FAMILIES = (
    "fruit",
    "animal",
    "vehicle",
    "profession",
    "place",
    "object",
    "color",
    "body_part",
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1024_lexical_semantic_orthogonal_atlas"
)


CONCEPTS = {
    "discovery": (
        ("apple", "fruit"),
        ("wolf", "animal"),
        ("bicycle", "vehicle"),
        ("doctor", "profession"),
        ("library", "place"),
        ("chair", "object"),
        ("red", "color"),
        ("hand", "body_part"),
    ),
    "confirmation": (
        ("pear", "fruit"),
        ("tiger", "animal"),
        ("airplane", "vehicle"),
        ("teacher", "profession"),
        ("hospital", "place"),
        ("mirror", "object"),
        ("blue", "color"),
        ("shoulder", "body_part"),
    ),
}

NONCES = {
    "discovery": ("dax", "wug", "zorp", "blicket"),
    "confirmation": ("fep", "nist", "koba", "tave"),
}

NONCE_TEMPLATES = {
    "discovery": (
        (
            'In this temporary glossary, "{nonce}" means the concept '
            '{concept}. Later, the code "{nonce}" appears again. '
            "Its broad category is"
        ),
        (
            'For this passage only, assign "{nonce}" to {concept}. '
            'When you encounter "{nonce}" again, classify it as'
        ),
    ),
    "confirmation": (
        (
            'Use this one-time codebook: "{nonce}" refers to {concept}. '
            'The repeated label "{nonce}" belongs to the category'
        ),
        (
            'Within this example, let "{nonce}" stand for {concept}. '
            'On its second mention, "{nonce}" has the broad class'
        ),
    ),
}


POLYSEMY = (
    {
        "word": "bank",
        "partition": "calibration",
        "senses": (
            ("finance", "financial institution",
             "a teller accepting a cash deposit",
             "a customer using a savings account and an ATM"),
            ("river", "river shoreline",
             "a canoe resting beside a muddy river edge",
             "fishermen walking along the water's shore"),
        ),
    },
    {
        "word": "bat",
        "partition": "calibration",
        "senses": (
            ("animal", "flying mammal",
             "a nocturnal flying mammal leaving a cave",
             "an echolocating animal hunting insects at night"),
            ("sports", "sports club",
             "a hitter swinging at a baseball",
             "a cricket player striking the ball"),
        ),
    },
    {
        "word": "crane",
        "partition": "calibration",
        "senses": (
            ("bird", "wading bird",
             "a long-legged bird standing in a marsh",
             "a feathered animal nesting near wetlands"),
            ("machine", "lifting machine",
             "a construction machine lifting steel beams",
             "a tall machine hoisting cargo above a building"),
        ),
    },
    {
        "word": "bark",
        "partition": "calibration",
        "senses": (
            ("tree", "tree covering",
             "the rough outer covering of a tree trunk",
             "the protective surface around a woody stem"),
            ("dog", "dog sound",
             "the sharp sound made by an excited dog",
             "a puppy making a loud warning noise"),
        ),
    },
    {
        "word": "jam",
        "partition": "calibration",
        "senses": (
            ("food", "fruit preserve",
             "a sweet strawberry spread on toast",
             "fruit cooked with sugar inside a jar"),
            ("traffic", "traffic congestion",
             "cars stopped in a crowded highway queue",
             "vehicles unable to move on a blocked road"),
        ),
    },
    {
        "word": "mole",
        "partition": "calibration",
        "senses": (
            ("animal", "burrowing animal",
             "a small mammal digging tunnels underground",
             "a dark-furred creature pushing soil into mounds"),
            ("skin", "skin mark",
             "a small dark spot on a person's skin",
             "a dermatologist examining a pigmented mark"),
        ),
    },
    {
        "word": "pitcher",
        "partition": "calibration",
        "senses": (
            ("container", "liquid container",
             "a handled vessel filled with cold water",
             "a ceramic jug used to pour lemonade"),
            ("baseball", "baseball player",
             "the player throwing the ball from the mound",
             "an athlete delivering a fastball to the batter"),
        ),
    },
    {
        "word": "ring",
        "partition": "calibration",
        "senses": (
            ("jewelry", "finger jewelry",
             "a gold band worn on a finger",
             "a jeweler setting a diamond in a circular band"),
            ("sound", "resonant sound",
             "the clear sound produced by a telephone bell",
             "a metallic chime continuing through the hall"),
        ),
    },
    {
        "word": "light",
        "partition": "heldout",
        "senses": (
            ("illumination", "illumination",
             "brightness from a lamp filling a dark room",
             "sunshine making the window glow"),
            ("weight", "low weight",
             "a suitcase easy to lift with one hand",
             "a package weighing almost nothing"),
        ),
    },
    {
        "word": "wave",
        "partition": "heldout",
        "senses": (
            ("ocean", "ocean swell",
             "a moving ridge of seawater approaching the beach",
             "surf rising and breaking near the shore"),
            ("gesture", "hand gesture",
             "a friendly hand motion used to say hello",
             "someone moving an arm to greet a neighbor"),
        ),
    },
    {
        "word": "mouse",
        "partition": "heldout",
        "senses": (
            ("animal", "small rodent",
             "a tiny rodent searching for crumbs",
             "a whiskered animal hiding under the floor"),
            ("computer", "computer pointing device",
             "a hand device moving the cursor on a screen",
             "clicking a desktop pointer beside a keyboard"),
        ),
    },
    {
        "word": "file",
        "partition": "heldout",
        "senses": (
            ("document", "document record",
             "a digital document stored in a folder",
             "a record opened and saved on a computer"),
            ("tool", "metal shaping tool",
             "a rough metal tool smoothing an edge",
             "a workshop tool scraping metal into shape"),
        ),
    },
    {
        "word": "spring",
        "partition": "heldout",
        "senses": (
            ("season", "season after winter",
             "the season when flowers return after winter",
             "warmer months when new leaves begin growing"),
            ("coil", "metal coil",
             "a compressed metal coil inside a mechanism",
             "a flexible coil pushing a latch back outward"),
        ),
    },
    {
        "word": "club",
        "partition": "heldout",
        "senses": (
            ("group", "social association",
             "an organized group meeting around a shared hobby",
             "members gathering for their weekly association"),
            ("weapon", "heavy striking weapon",
             "a heavy blunt weapon held in one hand",
             "a thick stick used as a striking weapon"),
        ),
    },
    {
        "word": "orange",
        "partition": "heldout",
        "senses": (
            ("fruit", "citrus fruit",
             "a juicy citrus fruit being peeled for lunch",
             "a round fruit divided into sweet segments"),
            ("color", "orange color",
             "a warm hue between red and yellow",
             "paint with the color of a sunset"),
        ),
    },
    {
        "word": "seal",
        "partition": "heldout",
        "senses": (
            ("animal", "marine animal",
             "a marine mammal resting on an icy shore",
             "a flippered animal swimming beside the rocks"),
            ("stamp", "official stamp",
             "an official mark pressed onto a legal document",
             "a stamped emblem certifying an envelope"),
        ),
    },
)

POLYSEMY_TEMPLATES = {
    "discovery": (
        'The scene involved {clue}. The ambiguous final word was "{word}".\nSense:',
        'People discussed {clue}; interpret the final word "{word}".\nSense:',
    ),
    "confirmation": (
        'This passage describes {clue}. Its last ambiguous term is "{word}".\nSense:',
        'Given {clue}, choose the intended meaning of "{word}".\nSense:',
    ),
}


SYNONYMS = (
    ("sofa", "calibration", ("sofa", "couch", "settee")),
    ("car", "calibration", ("car", "automobile", "motorcar")),
    ("child", "calibration", ("child", "kid", "youngster")),
    ("doctor", "calibration", ("doctor", "physician", "medic")),
    ("begin", "calibration", ("begin", "start", "commence")),
    ("buy", "calibration", ("buy", "purchase", "acquire")),
    ("fast", "calibration", ("fast", "quick", "rapid")),
    ("angry", "calibration", ("angry", "mad", "irate")),
    ("small", "heldout", ("small", "tiny", "little")),
    ("large", "heldout", ("large", "big", "huge")),
    ("road", "heldout", ("road", "street", "roadway")),
    ("job", "heldout", ("job", "occupation", "profession")),
    ("home", "heldout", ("home", "house", "residence")),
    ("error", "heldout", ("error", "mistake", "blunder")),
    ("answer", "heldout", ("answer", "response", "reply")),
    ("help", "heldout", ("help", "assist", "aid")),
)

SYNONYM_TEMPLATES = {
    "discovery": (
        'Read the final English vocabulary item. Word: "{alias}".\nMeaning:',
        'Focus on the lexical item at the end. Item: "{alias}".\nMeaning:',
    ),
    "confirmation": (
        'Consider the final term in ordinary English. Term: "{alias}".\nMeaning:',
        'Identify the concept evoked by this final word: "{alias}".\nMeaning:',
    ),
}


QUALIFICATION_CASES = (
    {
        "case_key": "rare.taotie",
        "family": "rare_definition",
        "prompt": "用一句简短的话解释“饕餮”的核心含义：",
        "accepted": ("凶兽", "神兽", "贪食", "贪吃", "暴食"),
        "max_new_tokens": 32,
    },
    {
        "case_key": "rare.xiezhi",
        "family": "rare_definition",
        "prompt": "“獬豸”通常表示什么？请给出简明释义：",
        "accepted": ("神兽", "司法", "公正", "辨别是非"),
        "max_new_tokens": 32,
    },
    {
        "case_key": "translation.apple",
        "family": "translation",
        "prompt": "Translate the English word 'apple' into French. Return the word only.",
        "accepted": ("pomme",),
        "max_new_tokens": 8,
    },
    {
        "case_key": "translation.wolf",
        "family": "translation",
        "prompt": "Translate the English word 'wolf' into Chinese. Return the word only.",
        "accepted": ("狼",),
        "max_new_tokens": 8,
    },
    {
        "case_key": "punctuation.question",
        "family": "punctuation",
        "prompt": "Return only the missing punctuation mark: Where did the train stop",
        "accepted": ("?", "？"),
        "max_new_tokens": 4,
    },
    {
        "case_key": "punctuation.exclamation",
        "family": "punctuation",
        "prompt": "Return only the missing punctuation mark: Watch out for the falling box",
        "accepted": ("!", "！"),
        "max_new_tokens": 4,
    },
    {
        "case_key": "connector.contrast",
        "family": "connector",
        "prompt": (
            "Fill the blank with one connector only: The fruit looked ripe. "
            "___, it was still sour."
        ),
        "accepted": ("however", "yet", "nevertheless"),
        "max_new_tokens": 8,
    },
    {
        "case_key": "connector.result",
        "family": "connector",
        "prompt": (
            "Fill the blank with one connector only: The alarm rang. "
            "___, everyone left."
        ),
        "accepted": ("therefore", "thus", "consequently"),
        "max_new_tokens": 8,
    },
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fragment(prompt: str, value: str, *, last: bool) -> tuple[int, int, str]:
    start = prompt.rfind(value) if last else prompt.find(value)
    if start < 0:
        raise RuntimeError(f"fragment {value!r} missing from prompt")
    return start, start + len(value), value


def build_nonce_cases() -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        for template_index, template in enumerate(NONCE_TEMPLATES[split]):
            for surface_index, nonce in enumerate(NONCES[split]):
                for concept_index, (concept, family) in enumerate(
                    CONCEPTS[split]
                ):
                    prompt = template.format(nonce=nonce, concept=concept)
                    rows.append({
                        "schema_version": "phase1024_common_case.v1",
                        "phase": PHASE,
                        "panel": "nonce_binding",
                        "case_key": (
                            f"nonce.{split}.t{template_index}."
                            f"s{surface_index}.c{concept_index}"
                        ),
                        "split": split,
                        "partition": (
                            "calibration" if split == "discovery"
                            else "heldout"
                        ),
                        "template_index": template_index,
                        "surface_index": surface_index,
                        "surface": nonce,
                        "concept_index": concept_index,
                        "concept": concept,
                        "family": family,
                        "prompt": prompt,
                        "role_fragments": {
                            "anchor_end": fragment(
                                prompt, concept, last=False
                            ),
                            "focus_end": fragment(prompt, nonce, last=True),
                        },
                        "accepted_outputs": (family,),
                        "candidate_outputs": FAMILIES,
                    })
    return rows


def build_polysemy_cases() -> list[dict[str, Any]]:
    rows = []
    for item_index, item in enumerate(POLYSEMY):
        for split in SPLITS:
            for template_index, template in enumerate(
                POLYSEMY_TEMPLATES[split]
            ):
                for sense_index, (
                    sense_id,
                    sense_label,
                    discovery_clue,
                    confirmation_clue,
                ) in enumerate(item["senses"]):
                    clue = (
                        discovery_clue
                        if split == "discovery"
                        else confirmation_clue
                    )
                    prompt = template.format(clue=clue, word=item["word"])
                    rows.append({
                        "schema_version": "phase1024_common_case.v1",
                        "phase": PHASE,
                        "panel": "polysemy",
                        "case_key": (
                            f"poly.{item['partition']}.{item_index}."
                            f"{split}.t{template_index}.s{sense_index}"
                        ),
                        "split": split,
                        "partition": item["partition"],
                        "template_index": template_index,
                        "item_index": item_index,
                        "word": item["word"],
                        "surface": item["word"],
                        "sense_index": sense_index,
                        "sense_id": sense_id,
                        "sense_label": sense_label,
                        "clue": clue,
                        "prompt": prompt,
                        "role_fragments": {
                            "anchor_end": fragment(prompt, clue, last=False),
                            "focus_end": fragment(
                                prompt, item["word"], last=True
                            ),
                        },
                        "accepted_outputs": (sense_label,),
                        "candidate_outputs": tuple(
                            value[1] for value in item["senses"]
                        ),
                    })
    return rows


def build_synonym_cases() -> list[dict[str, Any]]:
    rows = []
    for group_index, (group, partition, aliases) in enumerate(SYNONYMS):
        for split in SPLITS:
            for template_index, template in enumerate(
                SYNONYM_TEMPLATES[split]
            ):
                for alias_index, alias in enumerate(aliases):
                    prompt = template.format(alias=alias)
                    rows.append({
                        "schema_version": "phase1024_common_case.v1",
                        "phase": PHASE,
                        "panel": "synonym",
                        "case_key": (
                            f"syn.{partition}.{group_index}.{split}."
                            f"t{template_index}.a{alias_index}"
                        ),
                        "split": split,
                        "partition": partition,
                        "template_index": template_index,
                        "group_index": group_index,
                        "group": group,
                        "alias_index": alias_index,
                        "surface": alias,
                        "prompt": prompt,
                        "role_fragments": {
                            "anchor_end": fragment(prompt, alias, last=True),
                            "focus_end": fragment(prompt, alias, last=True),
                        },
                        "accepted_outputs": (),
                        "candidate_outputs": (),
                    })
    return rows


def build_common_cases() -> list[dict[str, Any]]:
    return (
        build_nonce_cases()
        + build_polysemy_cases()
        + build_synonym_cases()
    )


def model_case(
    tokenizer,
    model_name: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    rendered = render_chat(tokenizer, model_name, row["prompt"])
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    spans = offset_token_spans(
        tokenizer,
        rendered,
        row["prompt"],
        row["role_fragments"],
    )
    result = dict(row)
    result.pop("role_fragments", None)
    result.update({
        "schema_version": "phase1024_model_case.v1",
        "model": model_name,
        "record_id": f"{model_name}.{row['case_key']}",
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            "anchor_end": int(spans["anchor_end"][1]),
            "focus_end": int(spans["focus_end"][1]),
            "pre_output": len(input_ids) - 1,
        },
        "prompt_token_count": len(input_ids),
        "surface_token_count": len(
            tokenizer.encode(row["surface"], add_special_tokens=False)
        ),
    })
    return result


def audit_common(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_panel = Counter(row["panel"] for row in rows)
    nonce_cells = Counter(
        (row["split"], row["template_index"], row["surface_index"])
        for row in rows
        if row["panel"] == "nonce_binding"
    )
    poly_cells = Counter(
        (row["partition"], row["split"], row["sense_index"])
        for row in rows
        if row["panel"] == "polysemy"
    )
    synonym_cells = Counter(
        (row["partition"], row["split"], row["alias_index"])
        for row in rows
        if row["panel"] == "synonym"
    )
    checks = {
        "case_keys_unique": len({row["case_key"] for row in rows}) == len(rows),
        "panel_counts": by_panel == {
            "nonce_binding": 128,
            "polysemy": 128,
            "synonym": 192,
        },
        "nonce_cells_balanced": set(nonce_cells.values()) == {8},
        "polysemy_cells_balanced": set(poly_cells.values()) == {16},
        "synonym_cells_balanced": set(synonym_cells.values()) == {16},
        "nonce_family_balance": all(
            Counter(
                row["family"]
                for row in rows
                if row["panel"] == "nonce_binding"
                and row["split"] == split
            ) == {family: 8 for family in FAMILIES}
            for split in SPLITS
        ),
    }
    return {
        "schema_version": "phase1024_common_audit.v1",
        "case_count": len(rows),
        "panel_counts": dict(by_panel),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def audit_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "all_roles_present": all(
            set(row["role_positions"]) == set(ROLES) for row in rows
        ),
        "positions_in_range": all(
            all(0 <= value < len(row["input_ids"])
                for value in row["role_positions"].values())
            for row in rows
        ),
        "input_ids_nonempty": all(row["input_ids"] for row in rows),
        "surface_tokens_nonzero": all(
            row["surface_token_count"] > 0 for row in rows
        ),
        "record_ids_unique": len(
            {row["record_id"] for row in rows}
        ) == len(rows),
    }
    return {
        "schema_version": "phase1024_model_audit.v1",
        "model": rows[0]["model"],
        "case_count": len(rows),
        "surface_token_count": dict(Counter(
            row["surface_token_count"] for row in rows
        )),
        "prompt_token_count": {
            "minimum": min(row["prompt_token_count"] for row in rows),
            "maximum": max(row["prompt_token_count"] for row in rows),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def qualification_for_model(tokenizer, model_name: str) -> list[dict[str, Any]]:
    rows = []
    for row in QUALIFICATION_CASES:
        rendered = render_chat(tokenizer, model_name, row["prompt"])
        rows.append({
            "schema_version": "phase1024_qualification_case.v1",
            "phase": PHASE,
            "model": model_name,
            **row,
            "accepted": list(row["accepted"]),
            "rendered_prompt": rendered,
            "input_ids": [
                int(value)
                for value in tokenizer.encode(
                    rendered, add_special_tokens=False
                )
            ],
        })
    return rows


def main() -> None:
    common = build_common_cases()
    common_audit = audit_common(common)
    if not common_audit["all_checks_passed"]:
        raise RuntimeError(json.dumps(common_audit, ensure_ascii=False))

    protocol_payload = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "models": MODELS,
        "roles": ROLES,
        "panels": {
            "nonce_binding": (
                "surface x assigned-concept factorial; no fixed semantic "
                "vector or mechanism formula is assumed"
            ),
            "polysemy": (
                "same surface under two controlled senses; calibration "
                "lexemes select depth, heldout lexemes confirm"
            ),
            "synonym": (
                "different surfaces in related lexical groups; alias 0/1 "
                "form prototypes and alias 2 is the cross-surface query"
            ),
        },
        "selection_policy": (
            "discovery metrics select layers/components; confirmation never "
            "changes a frozen candidate"
        ),
        "claim_limits": (
            "retrieval and repeated response profiles are observational; "
            "they do not prove storage cells, causal necessity, brain "
            "homology, optimality, or a closed language mechanism"
        ),
        "common_case_digest": digest(common),
    }
    protocol_payload["protocol_digest"] = digest(protocol_payload)
    protocol_dir = OUT_ROOT / "protocol"
    write_json(protocol_dir / "preregistration.json", protocol_payload)
    write_json(protocol_dir / "audit.common.json", common_audit)
    write_jsonl(protocol_dir / "common_cases.jsonl", common)

    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        rows = [model_case(tokenizer, model_name, row) for row in common]
        audit = audit_model(rows)
        if not audit["all_checks_passed"]:
            raise RuntimeError(json.dumps(audit, ensure_ascii=False))
        write_jsonl(protocol_dir / f"cases.{model_name}.jsonl", rows)
        write_jsonl(
            protocol_dir / f"qualification.{model_name}.jsonl",
            qualification_for_model(tokenizer, model_name),
        )
        write_json(protocol_dir / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        del tokenizer

    write_json(
        protocol_dir / "audit.models.json",
        {
            "schema_version": "phase1024_model_audits.v1",
            "protocol_digest": protocol_payload["protocol_digest"],
            "models": model_audits,
            "all_checks_passed": all(
                row["all_checks_passed"] for row in model_audits.values()
            ),
        },
    )
    print(json.dumps({
        "protocol_digest": protocol_payload["protocol_digest"],
        "common_audit": common_audit,
        "models": {
            key: value["prompt_token_count"]
            for key, value in model_audits.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
