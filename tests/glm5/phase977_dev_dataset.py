#!/usr/bin/env python3
"""Frozen Phase 977 development corpus: 64 items across eight tasks.

The data are ASCII-only and do not reuse Phase 973--976 question/answer facts.
``alias_groups`` uses AND between groups and OR within each group.
"""
from __future__ import annotations

from collections import Counter
import unicodedata


EXPECTED_COUNTS = {
    "direct_fact": 8,
    "classification": 8,
    "arithmetic": 8,
    "translation_format": 8,
    "definition": 8,
    "causal": 8,
    "multistep_math": 8,
    "logic": 8,
}


# Tuple layout: prompt, canonical answer, alias groups, exact-match flag.
_GROUPS = {
    "direct_fact": [
        ("What city is the capital of Norway?", "Oslo", (("Oslo",),), False),
        ("Reply only with the chemical symbol for potassium.", "K", (("K",),), True),
        ("Who wrote the novel The Hobbit?", "J. R. R. Tolkien", (("J. R. R. Tolkien", "J.R.R. Tolkien", "Tolkien"),), False),
        ("Which moon is the largest in the Solar System?", "Ganymede", (("Ganymede",),), False),
        ("What is the SI unit used for frequency?", "the hertz", (("hertz", "Hz"),), False),
        ("Name the longest bone in the human body.", "the femur", (("femur",),), False),
        ("Which atmospheric layer contains most ordinary weather?", "the troposphere", (("troposphere",),), False),
        ("What instrument measures wind speed?", "an anemometer", (("anemometer",),), False),
    ],
    "classification": [
        ("Choose the animal class for a penguin: bird, mammal, reptile, or fish.", "bird", (("bird",),), False),
        ("Place obsidian in a rock class: igneous, sedimentary, or metamorphic.", "igneous", (("igneous",),), False),
        ("Determine whether 49 is prime or composite.", "composite", (("composite",),), False),
        ("Assign a lobster to one group: crustacean, insect, arachnid, or mollusk.", "crustacean", (("crustacean",),), False),
        ("Categorize Ceres as a planet, dwarf planet, moon, or star.", "dwarf planet", (("dwarf planet",),), False),
        ("Classify brass as an element, compound, alloy, or polymer.", "alloy", (("alloy",),), False),
        ("Choose the broad kingdom label for moss: plant, animal, fungus, or mineral.", "plant", (("plant",),), False),
        ("Is gypsum best categorized as a mineral, organism, alloy, or gas?", "mineral", (("mineral",),), False),
    ],
    "arithmetic": [
        ("What number results from 28 plus 47?", "75", (("75", "seventy-five", "seventy five"),), False),
        ("Calculate 103 minus 58.", "45", (("45", "forty-five", "forty five"),), False),
        ("Find the product of 16 and 7.", "112", (("112", "one hundred twelve", "one hundred and twelve"),), False),
        ("What is 225 divided by 15?", "15", (("15", "fifteen"),), False),
        ("Evaluate 9 cubed.", "729", (("729", "seven hundred twenty-nine", "seven hundred and twenty-nine"),), False),
        ("Find 18 percent of 250.", "45", (("45", "forty-five", "forty five"),), False),
        ("Compute the arithmetic mean of 9, 17, and 28.", "18", (("18", "eighteen"),), False),
        ("A rectangle has sides 11 and 6. What is its area?", "66", (("66", "sixty-six", "sixty six"),), False),
    ],
    "translation_format": [
        ("Translate the Spanish word mesa into English.", "table", (("table",),), False),
        ("Give the English meaning of the German word Baum.", "tree", (("tree",),), False),
        ("Render the French word neige in English.", "snow", (("snow",),), False),
        ("What is the English translation of the Italian word scarpa?", "shoe", (("shoe",),), False),
        ("Replace every space in deep blue sea with an underscore.", "deep_blue_sea", (("deep_blue_sea",),), True),
        ("Write the integer 7 as exactly three decimal digits with leading zeros.", "007", (("007",),), True),
        ("Convert the date 2026/11/04 from YYYY/MM/DD to DD-MM-YYYY.", "04-11-2026", (("04-11-2026",),), True),
        ("Join amber, cedar, and frost with vertical bars and no spaces.", "amber|cedar|frost", (("amber|cedar|frost",),), True),
    ],
    "definition": [
        ("Define latitude in one concise sentence.", "angular distance north or south of the equator", (("angular distance", "angle"), ("north or south",), ("equator",)), False),
        ("What is an aquifer?", "a permeable underground layer that stores and transmits groundwater", (("permeable", "porous"), ("underground",), ("groundwater", "water")), False),
        ("In computing, define an operating system.", "software that manages hardware resources and provides services for programs", (("software",), ("manage", "controls"), ("hardware", "resources"), ("program", "application")), False),
        ("What does symbiosis mean in biology?", "a close interaction between organisms of different species", (("interaction", "relationship", "living together"), ("different species", "organisms")), False),
        ("Give a geographic definition of a watershed.", "a land area whose water drains to a common outlet", (("land area", "area of land"), ("drain", "flows"), ("common outlet", "same outlet")), False),
        ("What is a cache in computing?", "storage that keeps reusable data so later access is faster", (("storage", "stores data"), ("reusable", "frequently used", "recently used"), ("faster", "speed")), False),
        ("Define an antibody at a basic biological level.", "an immune protein that binds a specific target called an antigen", (("immune",), ("protein",), ("bind", "recognize"), ("antigen", "target")), False),
        ("What is an archipelago?", "a group or chain of islands", (("group", "chain", "cluster"), ("islands",)), False),
    ],
    "causal": [
        ("How does a slightly stretching seat belt reduce peak force in a sudden stop?", "It lengthens the stopping time, which lowers the peak force.", (("longer", "lengthens", "increase"), ("stopping time", "deceleration time"), ("lower", "reduce", "less force")), False),
        ("Why does liquid rise into an eyedropper when its bulb is released?", "The pressure inside drops, so outside air pressure pushes the liquid upward.", (("pressure inside", "lower pressure", "pressure drops"), ("outside air", "atmospheric pressure"), ("push", "upward", "rise")), False),
        ("Why can a metal doorknob give a small shock after someone walks across carpet?", "Built-up static charge suddenly discharges through the metal.", (("static", "charge", "electron"), ("discharge", "flows", "equalize")), False),
        ("Why does a sealed syringe resist more as its trapped air is compressed?", "Reducing the gas volume raises its pressure.", (("volume", "compress"), ("pressure", "resist")), False),
        ("Why does a fuse open a circuit when excessive current flows?", "Resistive heating melts the fuse element and breaks the circuit.", (("heat", "heating"), ("melt",), ("open", "break")), False),
        ("Why may a laptop fan spin faster during a heavy computation?", "The processor produces more heat, so faster airflow is needed to remove it.", (("heat", "hot"), ("airflow", "cool", "remove")), False),
        ("Why does tightening a guitar string usually raise its pitch?", "Greater tension increases the string's vibration frequency.", (("tension", "tight"), ("frequency", "vibrat"), ("increase", "higher")), False),
        ("Why can a banana ripen faster inside a closed paper bag?", "The bag retains ethylene gas that promotes ripening.", (("ethylene",), ("retain", "trap", "accumulate"), ("ripen",)), False),
    ],
    "multistep_math": [
        ("A depot begins with 53 crates, receives 16, then ships 24. How many crates remain?", "45 crates", (("45", "forty-five", "forty five"),), False),
        ("Six racks hold 14 jars each. Nineteen jars are removed. How many remain?", "65 jars", (("65", "sixty-five", "sixty five"),), False),
        ("A 12 by 9 garden contains a 4 by 4 pond. What planted area remains?", "92 square units", (("92", "ninety-two", "ninety two"), ("square",)), False),
        ("An account starts with 35 dollars, receives 12 dollars each week for five weeks, then pays 18 dollars. What is the balance?", "77 dollars", (("77", "seventy-seven", "seventy seven"),), False),
        ("Three batches contain 18 rolls each. Six are discarded, and the rest are divided among four trays. How many rolls per tray?", "12 rolls", (("12", "twelve"),), False),
        ("A cyclist rides 18 kilometers per hour for two hours, then rides 12 more kilometers. What total distance was traveled?", "48 kilometers", (("48", "forty-eight", "forty eight"),), False),
        ("A theater has 12 rows of 15 seats, and 17 seats are empty. How many seats are occupied?", "163 seats", (("163", "one hundred sixty-three", "one hundred and sixty-three"),), False),
        ("A rectangular yard is 18 by 9 meters. A 3-meter gate needs no fence. How many meters of fence are needed?", "51 meters", (("51", "fifty-one", "fifty one"),), False),
    ],
    "logic": [
        ("Some painters are sailors, and every sailor is a traveler. What must hold for at least some painters?", "Some painters are travelers.", (("some painters",), ("travelers", "traveller")), False),
        ("A sensor is reliable only if both indicator lamps are on. The left lamp is off. Can it be reliable under that rule?", "No, it cannot be reliable.", (("no", "not reliable", "cannot be reliable"),), False),
        ("Every blue card has a star. A card has a star. Does that fact force the card to be blue?", "No, its color cannot be concluded.", (("no", "not necessarily", "cannot", "does not"),), False),
        ("Set A is {2, 6, 10} and set B is {4, 6, 12}. What is their intersection?", "{6}", (("{6}", "6"),), False),
        ("P is false and Q is true. What is the truth value of (not P) AND Q?", "true", (("true",),), False),
        ("Exactly two of switches X, Y, and Z are on. X is off and Y is on. What is Z's state?", "Z is on.", (("Z is on", "switch Z is on", "on"),), False),
        ("Every orchid is a flower, and no flower is a metal object. What follows about orchids?", "No orchid is a metal object.", (("orchid",), ("not", "no"), ("metal",)), False),
        ("Whenever a parcel is scanned, it is logged. This parcel was not logged. What follows?", "The parcel was not scanned.", (("not scanned",),), False),
    ],
}


def build_dataset():
    """Return fresh dictionaries so callers cannot mutate the frozen constants."""
    rows = []
    for task, specs in _GROUPS.items():
        for index, (prompt, answer, groups, exact) in enumerate(specs, start=1):
            rows.append({
                "id": f"p977_dev_{task}_{index:02d}",
                "task": task,
                "prompt": prompt,
                "answer": answer,
                "alias_groups": [list(group) for group in groups],
                "exact": exact,
            })
    return rows


def _prompt_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _previous_keys(previous_prompts) -> set[str]:
    if previous_prompts is None:
        return set()
    keys = set()
    for value in previous_prompts:
        prompt = value.get("prompt", "") if isinstance(value, dict) else str(value)
        if prompt:
            keys.add(_prompt_key(prompt))
    return keys


def audit_dataset(previous_prompts=None):
    """Audit schema, encoding, balance, duplicates, and cross-set overlap."""
    rows = build_dataset()
    errors = []
    ids = [row["id"] for row in rows]
    prompt_keys = [_prompt_key(row["prompt"]) for row in rows]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    duplicate_prompts = sorted(key for key, count in Counter(prompt_keys).items() if count > 1)
    previous = _previous_keys(previous_prompts)
    cross_set_overlap = sorted(set(prompt_keys) & previous)
    counts = dict(sorted(Counter(row["task"] for row in rows).items()))
    encoding_issues = []
    schema_issues = []

    for row in rows:
        required = {"id", "task", "prompt", "answer", "alias_groups"}
        if not required.issubset(row):
            schema_issues.append(f"{row.get('id', '<missing-id>')}: missing fields")
            continue
        if row["task"] not in EXPECTED_COUNTS:
            schema_issues.append(f"{row['id']}: unknown task {row['task']}")
        groups = row["alias_groups"]
        if not groups or any(not isinstance(group, list) or not group for group in groups):
            schema_issues.append(f"{row['id']}: empty or invalid alias group")
        if row.get("exact", False) and len(groups) != 1:
            schema_issues.append(f"{row['id']}: exact item must have exactly one alias group")
        answer_key = row["answer"].casefold()
        for group in groups:
            if not any(alias.casefold() in answer_key for alias in group):
                schema_issues.append(f"{row['id']}: canonical answer misses alias group {group}")
        strings = [row["id"], row["task"], row["prompt"], row["answer"]]
        strings.extend(alias for group in groups for alias in group)
        for value in strings:
            if unicodedata.normalize("NFC", value) != value:
                encoding_issues.append(f"{row['id']}: non-NFC text")
            if "\ufffd" in value:
                encoding_issues.append(f"{row['id']}: U+FFFD present")
            if any(0x80 <= ord(char) <= 0x9F for char in value):
                encoding_issues.append(f"{row['id']}: C1 control present")
            if not value.isascii():
                encoding_issues.append(f"{row['id']}: non-ASCII text")

    if len(rows) != 64:
        errors.append(f"expected 64 rows, found {len(rows)}")
    if counts != EXPECTED_COUNTS:
        errors.append(f"task counts differ: {counts}")
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids}")
    if duplicate_prompts:
        errors.append(f"duplicate prompts: {duplicate_prompts}")
    if cross_set_overlap:
        errors.append(f"cross-set prompt overlap: {cross_set_overlap}")
    errors.extend(schema_issues)
    errors.extend(sorted(set(encoding_issues)))
    passed = not errors
    return {
        "ok": passed,
        "passed": passed,
        "n_items": len(rows),
        "task_counts": counts,
        "duplicate_ids": duplicate_ids,
        "duplicate_prompts": duplicate_prompts,
        "cross_set_overlap": cross_set_overlap,
        "schema_issues": schema_issues,
        "encoding_issues": sorted(set(encoding_issues)),
        "errors": errors,
    }


if __name__ == "__main__":
    result = audit_dataset()
    print(result)
    raise SystemExit(0 if result["ok"] else 1)
