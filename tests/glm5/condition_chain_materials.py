"""Authored factorial families. No corpus passages or model responses enter the generator."""
import hashlib
import itertools
import random

SPLITS = {"pilot": 8, "discover": 32, "words": 96, "frames": 96}
MODELS = {"q4": "qwen3-4b", "q14": "Qwen3-14B", "glm": "glm4-9b-chat-hf"}
CANDIDATES = {"boundary": (13, 9571, "G"), "payment": (16, 9414, "G"),
              "assessment": (31, 2619, "G")}
VOCAB = {
    "pilot": "amber birch cobalt dune elm flax granite hazel ivory jade",
    "discover": "acorn beech cedar dahlia elder fir ginger heather iron juniper kelp linen maple nettle oak palm",
    "words": "quartz rose silk thyme umber velvet willow xylem yam zinc apple basil clay daisy ebony feather",
    "frames": "agate bronze coral driftwood eucalyptus fern gold hickory indigo jasmine lemon moss nickel olive pearl reed",
}
SCENARIOS = {
    "boundary": (
        ("reservoir", "kiln", "lift", "greenhouse", "workshop", "basin"),
        "threshold", "quantity reached its limit", "quantity stayed below its limit",
        "At {name}, the safe limit is {b} units. The actual reading is {a} units.",
        "The operator at {name} saw a reading of {a} units and can accept at most {b} units.",
    ),
    "payment": (
        ("clinic", "venue", "hostel", "studio", "depot", "museum"),
        "payment", "visitor owes a fee", "visitor owes nothing",
        "At {name} the visitor used room {a}. The agreement charges for room {b} only. All other rooms are free.",
        "Room {b} alone incurs a charge at {name}. The visitor stayed in room {a}. No other room costs anything.",
    ),
    "assessment": (
        ("headset", "engine", "relay", "sensor", "radio", "pump"),
        "testing", "verdict was based on a measurement", "verdict was based on a guess",
        "The inspector rated the {name} after {source}. This was the only basis for the verdict.",
        "For the {name}, the verdict from the inspector rested entirely on {source}.",
    ),
}


def key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]


def rng(*parts):
    return random.Random(key(2026111501, *parts))


def report_options(options):
    return " | ".join(f"{letter}: {option}" for letter, option in zip("ABCD", options))


def chain(split, domain, i, factors):
    select, update, distract, report = factors
    # The same four values and answer permutation are shared by the full family.
    randomizer = rng(split, domain, i)
    values = randomizer.sample(VOCAB[split].split(), 4)
    order = randomizer.sample(range(4), 4)
    routing = randomizer.sample(range(4), 4)
    if report:
        order = order[1:] + order[:1]
    picked = [0, 2][select]
    terminal = picked + update
    decoy = 0 if distract == 0 else 3
    if split == "frames":
        if domain == "stock":
            pieces = [
                f"Stock clerk's job. Copper bin starts with {values[0]}; glass bin starts with {values[2]}. "
                f"For this ticket, copy the {'copper' if select == 0 else 'glass'} bin's starting label.",
                f"Possible replacement labels: copper {values[1]}, glass {values[3]}. "
                f"On this ticket, {'retain your copy' if update == 0 else 'use the selected bin’s replacement'}.",
                f"The sample sticker says {values[decoy]}. Sample stickers are supplied for illustration only.",
            ]
        else:
            pieces = [
                f"Calendar dispatch. The first calendar starts with {values[0]}; the second starts with {values[2]}. "
                f"This dispatch concerns the {'first' if select == 0 else 'second'} calendar. Copy its initial venue.",
                f"Alternative venues are {values[1]} for the first calendar and {values[3]} for the second. "
                f"Today's order is to {'keep the copy' if update == 0 else 'take the selected calendar’s alternative'}.",
                f"A sample invitation names {values[decoy]}; it is not part of today's dispatch.",
            ]
    else:
        first, second = ("north", "south") if domain == "stock" else ("morning", "evening")
        noun = "label" if domain == "stock" else "venue"
        pieces = [
            f"Track the {noun} for the {'selected stock bin' if domain == 'stock' else 'selected calendar'}.\n"
            f"Initial {noun}: {first} = {values[0]}, {second} = {values[2]}.\n"
            f"Select {'the ' + (first, second)[select]}. Copy its initial {noun}.",
            f"Replacement {noun}: {first} = {values[1]}, {second} = {values[3]}.\n"
            f"Instruction: {'KEEP the copied value' if update == 0 else 'REPLACE with the selected entry’s replacement'}.",
            f"Unrelated example {noun}: {values[decoy]}. Ignore the example.",
        ]
    ordered = [values[j] for j in order]
    codes = ["code-" + str(j + 1) for j in range(4)]
    mapped = [codes[routing[j]] for j in order]
    ending = (f"\nCodebook: {'; '.join(v + ' = ' + codes[routing[j]] for j, v in enumerate(values))}.\n"
              + ("Which option is the final value?" if report == 0 else "Which option is the code for the final value?")
              + "\n" + report_options(ordered if report == 0 else mapped))
    family = key("chain", split, domain, i)
    return dict(id=family + "".join(map(str, factors)), family=family, task=domain, split=split,
                factors=factors, pieces=pieces, ending=ending, truth=[picked, terminal, terminal],
                values=values, option_order=order, routing=routing, answer=order.index(terminal),
                diagnostic_index=decoy)


def candidate(split, task, i, factors):
    target, keyword = factors
    randomizer = rng(split, task, i)
    contexts, cue, yes, no, usual, heldout = SCENARIOS[task]
    # Term present in a separate glossary, independently of actual evidence.
    name = randomizer.choice(contexts) + " " + randomizer.choice(VOCAB[split].split())
    a, gap = randomizer.sample(range(21, 90), 2)
    source = ["a personal guess", "a logged instrument reading"][target]
    if split == "words":
        source = ["an unverified hunch", "a calibrated gauge reading"][target]
    elif split == "frames":
        source = ["their intuition alone", "a repeatable check of the device"][target]
    b = (a + (gap % 9 + 1)*(1 if target == 0 else -1)) if task == "boundary" else (a if target else a+(gap % 9 + 1))
    evidence = (heldout if split == "frames" else usual).format(name=name, a=a, b=b, source=source)
    if task == "boundary":
        if split == "words":
            evidence = f"The ceiling for {name} is {b} units. The dial currently reads {a} units."
        yes = "reading exceeded the bound"
    elif task == "payment" and split == "words":
        evidence = f"The guest at {name} had suite {a}. Suite {b} is the sole suite with a bill. Every other suite is complimentary."
    glossary = f"Unrelated glossary entry: {cue if keyword else 'reference'}."
    pieces = [glossary + "\n" + evidence]
    polarity = randomizer.randrange(2)
    options = [no, yes] if polarity == 0 else [yes, no]
    ending = "\nWhich conclusion follows from the passage?\n" + report_options(options)
    family = key("candidate", split, task, i)
    return dict(id=family + "".join(map(str, factors)), family=family, task=task, split=split,
                factors=factors, pieces=pieces, ending=ending, truth=[target],
                answer=options.index(yes if target else no), options=options, evidence=dict(a=a, b=b, source=source))


def rows(split, panel):
    tasks = ("stock", "calendar") if panel == "chain" else CANDIDATES
    build = chain if panel == "chain" else candidate
    factors = list(itertools.product((0, 1), repeat=4 if panel == "chain" else 2))
    return [dict(build(split, task, i, arm), index=i, panel=panel)
            for task in tasks for i in range(SPLITS[split]) for arm in factors]
