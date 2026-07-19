#!/usr/bin/env python3
"""Frozen Phase 977 holdout corpus: 128 unseen items across eight tasks.

The corpus is intentionally ASCII-only. ``alias_groups`` uses AND between
groups and OR within a group. Items marked ``exact`` must have one group of
acceptable full answers.
"""
from __future__ import annotations

from collections import Counter
import unicodedata


EXPECTED_COUNTS = {
    "direct_fact": 16,
    "classification": 16,
    "arithmetic": 16,
    "translation_format": 16,
    "definition": 16,
    "causal": 16,
    "multistep_math": 16,
    "logic": 16,
}


# Tuple layout: prompt, canonical answer, alias groups, exact-match flag.
_GROUPS = {
    "direct_fact": [
        ("Which city serves as Australia's federal capital?", "Canberra", (("Canberra",),), False),
        ("Reply only with the element symbol used for sodium.", "Na", (("Na",),), True),
        ("Identify the novelist who wrote Nineteen Eighty-Four.", "George Orwell", (("George Orwell", "Orwell"),), False),
        ("What natural mineral ranks highest on the Mohs hardness scale?", "diamond", (("diamond",),), False),
        ("What instrument is used to measure atmospheric pressure?", "a barometer", (("barometer",),), False),
        ("State the currency used in Switzerland.", "the Swiss franc", (("Swiss franc", "franc"),), False),
        ("What is the capital city of New Zealand?", "Wellington", (("Wellington",),), False),
        ("Which element is first in the periodic table?", "hydrogen", (("hydrogen",),), False),
        ("Name the largest organ of the human body.", "the skin", (("skin",),), False),
        ("Who composed the concertos known as The Four Seasons?", "Antonio Vivaldi", (("Antonio Vivaldi", "Vivaldi"),), False),
        ("How many continents are commonly recognized?", "seven", (("seven", "7"),), False),
        ("Who is widely credited with patenting the first practical telephone?", "Alexander Graham Bell", (("Alexander Graham Bell", "Bell"),), False),
        ("Which ocean is the smallest by surface area?", "the Arctic Ocean", (("Arctic Ocean", "Arctic"),), False),
        ("What is the largest living bird by mass?", "the ostrich", (("ostrich",),), False),
        ("Name the largest internal organ in an adult human.", "the liver", (("liver",),), False),
        ("Which language is the nationwide official language of Austria?", "German", (("German",),), False),
    ],
    "classification": [
        ("Choose the rock class for basalt: igneous, sedimentary, or metamorphic.", "igneous", (("igneous",),), True),
        ("Label a dolphin as a mammal, fish, bird, or reptile.", "mammal", (("mammal",),), True),
        ("Decide whether 51 is prime or composite.", "composite", (("composite",),), True),
        ("Place a spider in one group: insect, arachnid, crustacean, or mollusk.", "arachnid", (("arachnid",),), True),
        ("Select the vertebrate class of a turtle: reptile, amphibian, fish, or mammal.", "reptile", (("reptile",),), True),
        ("Assign salmon to one class: fish, amphibian, reptile, or bird.", "fish", (("fish",),), True),
        ("Classify a mushroom as plant, animal, fungus, or mineral.", "fungus", (("fungus",),), True),
        ("Choose the category for quartz: mineral, organism, alloy, or polymer.", "mineral", (("mineral",),), True),
        ("Is helium an element, compound, or mixture? Give the category only.", "element", (("element",),), True),
        ("Put sandstone in its rock class: igneous, sedimentary, or metamorphic.", "sedimentary", (("sedimentary",),), True),
        ("Determine whether 37 is prime or composite.", "prime", (("prime",),), True),
        ("Categorize methane as an element, compound, or mixture.", "compound", (("compound",),), True),
        ("Label an octagon as polygon, circle, solid, or angle.", "polygon", (("polygon",),), True),
        ("Choose the animal class for an eagle: bird, mammal, reptile, or fish.", "bird", (("bird",),), True),
        ("Classify sulfur as metal, nonmetal, or alloy.", "nonmetal", (("nonmetal", "non-metal"),), True),
        ("Place an earthworm in one group: vertebrate or invertebrate.", "invertebrate", (("invertebrate",),), True),
    ],
    "arithmetic": [
        ("Write only the numeral obtained by adding 23 and 19.", "42", (("42",),), True),
        ("Compute 96 minus 37; respond with digits only.", "59", (("59",),), True),
        ("Give the numeric product of 13 and 9, with no words.", "117", (("117",),), True),
        ("Divide 156 by 13 and supply only the quotient.", "12", (("12",),), True),
        ("What is 7 cubed? Use a numeral alone.", "343", (("343",),), True),
        ("Find 35 percent of 200 and write just the number.", "70", (("70",),), True),
        ("Calculate the mean of 14, 22, and 30. Give one numeral.", "22", (("22",),), True),
        ("A rectangle is 8 units by 3 units. State its perimeter as a numeral.", "22", (("22",),), True),
        ("A triangle has base 10 and height 7. Give its area as a number.", "35", (("35",),), True),
        ("Evaluate five eighths of 64. Output the number only.", "40", (("40",),), True),
        ("A temperature of -4 rises by 11 degrees. State the new value only.", "7", (("7",),), True),
        ("Four notebooks cost 18 dollars each. Give the total number of dollars.", "72", (("72",),), True),
        ("Two shares are in the ratio 3:5 and total 64. Give the smaller share.", "24", (("24",),), True),
        ("Add 50 minutes to 2 hours 45 minutes. Write the total minutes only.", "215", (("215",),), True),
        ("Continue the arithmetic sequence 11, 15, 19 with the next numeral.", "23", (("23",),), True),
        ("Subtract 20 from 2 raised to the seventh power. Give one numeral.", "108", (("108",),), True),
    ],
    "translation_format": [
        ("Give the Spanish word for horse, using one word only.", "caballo", (("caballo",),), True),
        ("Supply the French word meaning lake. Use one word.", "lac", (("lac",),), True),
        ("Write the German translation of bird as a single word.", "Vogel", (("Vogel",),), True),
        ("Convert the English noun rain to Italian; give one word.", "pioggia", (("pioggia",),), True),
        ("State white in Spanish with no surrounding text.", "blanco", (("blanco",),), True),
        ("Render the English word morning in French as one word.", "matin", (("matin",),), True),
        ("Provide the German noun for sister and nothing more.", "Schwester", (("Schwester",),), True),
        ("Translate kitchen from English to Italian in one word.", "cucina", (("cucina",),), True),
        ("Change delta to all capital letters.", "DELTA", (("DELTA",),), True),
        ("Change ROBOT to all lowercase letters.", "robot", (("robot",),), True),
        ("Give the plural form of goose as one word.", "geese", (("geese",),), True),
        ("Give the simple past form of teach as one word.", "taught", (("taught",),), True),
        ("Put these words in dictionary order: kiwi, banana.", "banana, kiwi", (("banana, kiwi", "banana kiwi"),), True),
        ("Express decimal 13 in base two, using digits only.", "1101", (("1101",),), True),
        ("Express 14 as a Roman numeral.", "XIV", (("XIV",),), True),
        ("Write the standard initials for North Atlantic Treaty Organization.", "NATO", (("NATO",),), True),
    ],
    "definition": [
        ("Explain what a pronoun is in one concise clause.", "a word that substitutes for a noun", (("word",), ("substitute", "replace"), ("noun",)), False),
        ("Define a tectonic plate briefly.", "a rigid slab of Earth's outer shell that moves", (("rigid", "solid"), ("Earth",), ("moves", "motion")), False),
        ("In a short clause, define an orbit.", "the path an object follows around another body", (("path",), ("around", "revolves")), False),
        ("What is a neuron? Give a compact definition.", "a nerve cell that transmits signals", (("nerve cell", "cell"), ("signal",)), False),
        ("Define a renewable resource without an example.", "a resource replenished naturally on a human timescale", (("resource",), ("replenish", "replace"), ("natural",)), False),
        ("State the geometric meaning of perimeter.", "the total distance around a shape's boundary", (("distance", "length"), ("around", "boundary")), False),
        ("Describe a compiler in one sentence.", "software that translates source code into executable machine code", (("software", "program"), ("source code",), ("machine code", "executable")), False),
        ("Give a concise geographic definition of a peninsula.", "land surrounded by water on three sides", (("land",), ("water",), ("three sides", "3 sides")), False),
        ("What is a catalyst in chemistry?", "a substance that speeds a reaction without being consumed", (("reaction",), ("speed", "rate"), ("not consumed", "without being consumed", "unchanged")), False),
        ("Define a vaccine at a basic biological level.", "a preparation that trains the immune system to recognize a pathogen", (("immune system",), ("recognize", "response"), ("pathogen", "disease")), False),
        ("Give the basic definition of wavelength.", "the distance between corresponding points of successive waves", (("distance",), ("successive waves", "adjacent waves", "wave")), False),
        ("Define an economic recession in a short clause.", "a broad decline in economic activity lasting for a period", (("decline", "decrease"), ("economic activity", "economy")), False),
        ("What is an integer? Answer with a definition.", "a whole number that may be positive, negative, or zero", (("whole number",), ("negative",), ("zero",)), False),
        ("Define biodiversity concisely.", "the variety of living organisms in an area", (("variety",), ("living organisms", "life", "species")), False),
        ("What does conjunction mean in grammar?", "a word that connects words, phrases, or clauses", (("word",), ("connect", "join"), ("clause", "phrase", "words")), False),
        ("Explain encryption in one short sentence.", "the conversion of data into coded form to restrict access", (("data", "information"), ("coded", "cipher"), ("access", "unauthorized")), False),
    ],
    "causal": [
        ("Why does a popcorn kernel burst when heated? Give the physical cause.", "Water inside becomes steam, raising pressure until the shell ruptures.", (("water", "moisture"), ("steam", "vapor"), ("pressure",)), False),
        ("Why does a cut apple surface turn brown in air?", "Its exposed compounds react with oxygen in an oxidation process.", (("oxygen",), ("oxidation", "oxidize", "react")), False),
        ("Why do Earth's seasons change during the year?", "Earth's tilted axis changes the sunlight each region receives as Earth orbits the Sun.", (("tilt", "tilted axis"), ("orbit", "around the Sun")), False),
        ("What mainly causes the regular ocean tides on Earth?", "The Moon's gravity produces most of the regular tidal pull.", (("Moon", "lunar"), ("gravity", "gravitational")), False),
        ("Why can a straight straw look bent where it enters water?", "Light changes direction by refraction at the air-water boundary.", (("light",), ("refract", "changes direction", "bends")), False),
        ("Why can soap help water remove greasy dirt?", "Soap molecules connect with grease and water so the grease can be carried away.", (("soap", "surfactant"), ("grease", "oil"), ("water",)), False),
        ("Why does food cook faster in a pressure cooker?", "Higher pressure raises water's boiling temperature, allowing hotter cooking.", (("pressure",), ("boiling",), ("temperature", "hotter")), False),
        ("Why does spreading salt help melt ice on a road?", "Dissolved salt lowers water's freezing point.", (("lower", "depress"), ("freezing point", "freezing temperature")), False),
        ("Why does dark fabric usually get hotter than light fabric in sunshine?", "Dark fabric absorbs more incoming light energy.", (("absorb",), ("light", "radiation", "energy")), False),
        ("Why can a person's ears pop during an airplane climb?", "Air pressure changes create a pressure difference that the ear equalizes.", (("air pressure", "pressure"), ("difference", "equalize")), False),
        ("Why does a carbonated drink fizz after its container is opened?", "The pressure drop lets dissolved carbon dioxide leave the liquid as bubbles.", (("pressure",), ("carbon dioxide", "CO2"), ("bubble", "gas", "leave")), False),
        ("Why do many plant roots grow downward even in darkness?", "Gravity directs their growth through gravitropism.", (("gravity",), ("gravitropism", "growth")), False),
        ("Why does a paper towel draw water upward above the water line?", "Capillary action pulls water through narrow spaces between its fibers.", (("capillary",), ("fiber", "narrow spaces", "pores")), False),
        ("Why does dry rice become soft while boiling in water?", "Its starch absorbs water and gelatinizes as it heats.", (("starch",), ("water",), ("absorb", "gelatin")), False),
        ("Why can an old copper roof develop a green surface?", "Copper reacts with air and moisture to form a green patina.", (("copper",), ("react", "oxid"), ("patina", "green")), False),
        ("Why does adding water help a dormant seed begin germination?", "Water activates enzymes and metabolic processes needed for growth.", (("water",), ("enzyme", "metabolism", "metabolic"), ("growth", "germinat")), False),
    ],
    "multistep_math": [
        ("Three boxes hold 8 pens each. Five pens are given away. How many remain? Give a numeral.", "19", (("19",),), True),
        ("A bus starts with 42 riders. Seventeen leave and 9 board. Give the final count.", "34", (("34",),), True),
        ("A 120-liter tank loses 15 liters each hour for 3 hours. State the liters left.", "75", (("75",),), True),
        ("A 12 by 7 rectangle has a 10-square-unit piece removed. Give the remaining area.", "74", (("74",),), True),
        ("Someone saves 25 dollars weekly for 6 weeks, then spends 40. Give the balance.", "110", (("110",),), True),
        ("A recipe uses 3 cups for 12 servings. At the same rate, how many cups serve 20?", "5", (("5",),), True),
        ("A farm gathers 18 eggs daily for 4 days and sells 27. How many eggs remain?", "45", (("45",),), True),
        ("Two positive numbers total 48, and the larger is five times the smaller. Give the smaller.", "8", (("8",),), True),
        ("A class has 40 students and one quarter are absent. How many are present?", "30", (("30",),), True),
        ("A 240-page book is read at 35 pages per day for 4 days. How many pages remain?", "100", (("100",),), True),
        ("Six rows contain 9 plants each; 8 plants fail. How many healthy plants remain?", "46", (("46",),), True),
        ("An 80-dollar item is discounted by 25 percent, then 5 dollars shipping is added. Give the final cost.", "65", (("65",),), True),
        ("A show starts at 2:20 and lasts 95 minutes. Give the ending time in h:mm form.", "3:55", (("3:55",),), True),
        ("Red and blue tokens are in a 2:3 ratio, with 35 total. How many are red?", "14", (("14",),), True),
        ("Four quarters and six dimes have what total dollar value? Give a decimal number.", "1.60", (("1.60", "1.6"),), True),
        ("A machine makes 14 parts per hour for 5 hours; 7 are rejected. How many pass?", "63", (("63",),), True),
    ],
    "logic": [
        ("Every tulip is a plant, and no plant is a machine. What follows about tulips and machines?", "No tulip is a machine.", (("no tulip", "tulips are not"), ("machine",)), False),
        ("If an alarm is armed, its red light is on. The red light is off. What follows?", "The alarm is not armed.", (("not armed", "unarmed"),), False),
        ("A file is either in the drawer or in the cabinet, but it is not in the drawer. Where is it?", "It is in the cabinet.", (("cabinet",),), False),
        ("Every violin is a string instrument, and no string instrument is a wind instrument. What follows about a violin?", "A violin is not a wind instrument.", (("not", "no"), ("wind instrument",)), False),
        ("Ada arrived before Ben, and Ben arrived before Cara. Who arrived before Cara among Ada and Ben?", "Both Ada and Ben arrived before Cara.", (("Ada",), ("Ben",)), False),
        ("A valid code always has a matching checksum. This code's checksum does not match. What follows?", "The code is not valid.", (("not valid", "invalid"),), False),
        ("No even integer is odd. Fourteen is even. State the forced conclusion about fourteen.", "Fourteen is not odd.", (("not odd",),), False),
        ("All cedar trees are evergreen. A shrub is not evergreen. What follows about that shrub being a cedar?", "The shrub is not a cedar.", (("not a cedar", "isn't a cedar"),), False),
        ("Exactly one of a red flag and a blue flag is raised. The red flag is raised. What follows?", "The blue flag is not raised.", (("blue",), ("not raised", "lowered")), False),
        ("If the meeting is on Tuesday, a notice is sent Monday. No notice was sent Monday. What follows?", "The meeting is not on Tuesday.", (("not",), ("Tuesday",)), False),
        ("A key is in exactly one of boxes A, B, and C. It is in neither A nor C. Where is it?", "The key is in box B.", (("box B",),), False),
        ("Every hexagon is a polygon. This figure is not a polygon. What follows about it being a hexagon?", "It is not a hexagon.", (("not a hexagon",),), False),
        ("If Nora enters, Omar enters; if Omar enters, Pia enters. Nora enters. Who else must enter?", "Omar and Pia must enter.", (("Omar",), ("Pia",)), False),
        ("All animals in group N rest at noon. An owl is in group N. What must the owl do at noon?", "The owl must rest at noon.", (("rest",), ("noon",)), False),
        ("Only registered members may enter the archive. Lee entered the archive. What follows about Lee?", "Lee is a registered member.", (("Lee",), ("member", "registered")), False),
        ("If a battery is charged, a device starts. The device starts. Is a charged battery logically forced?", "No, the charged battery is not logically forced.", (("no", "not"), ("not logically forced", "does not follow", "cannot conclude")), False),
    ],
}


def build_dataset():
    """Return fresh dictionaries so callers cannot mutate the frozen constants."""
    rows = []
    for task, specs in _GROUPS.items():
        for index, (prompt, answer, groups, exact) in enumerate(specs, start=1):
            rows.append({
                "id": f"p977_holdout_{task}_{index:02d}",
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

    if len(rows) != 128:
        errors.append(f"expected 128 rows, found {len(rows)}")
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
