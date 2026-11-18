"""Original paired polarity and register materials for the weighted-write study."""

from hashlib import sha256
from itertools import combinations, product
import random


N = 192
SEED = 29392026
PANELS = ("polarity", "update")
WORDS = {
    "polarity": "trumpet piano flute violin drum harp guitar organ horn bass accordion clarinet bell whistle cymbal microphone stereo speaker radio cassette record tape keyboard conductor orchestra",
    "update": "tiger lion zebra giraffe elephant rabbit fox wolf deer camel turtle dolphin shark whale eagle hawk sparrow crow penguin robin goose duck swan turkey squirrel bear",
}
PILOT = {
    "polarity": "soup stew salad toast sandwich porridge yogurt oatmeal curry pasta pizza noodles",
    "update": "violet crimson scarlet navy pink purple beige amber coral magenta turquoise indigo",
}
NOTES = "cloud rain snow wind storm fog frost mist".split()


def make(panel, split, words=None):
    """Define a family by the unordered pair, never by the four experimental views."""
    is_pilot = split == "pilot"
    words = sorted(words or (PILOT if is_pilot else WORDS)[panel].split())
    rng = random.Random(f"{SEED}/{panel}/{split}")
    pairs = list(combinations(words, 2))
    rng.shuffle(pairs)
    rows = []
    for i, pair in enumerate(pairs[:12 if is_pilot else N]):
        old, new = pair if rng.getrandbits(1) else pair[::-1]
        framing = i % 3
        reverse = i % 2
        note = NOTES[(i//6 + i%6) % len(NOTES)]
        family = sha256(f"weighted-value/{panel}/{'/'.join(pair)}".encode()).hexdigest()[:18]
        for change, swap in product((0,1), repeat=2):
            state, alt = (new,old) if change else (old,new)
            codes = dict(zip((old,new),("Q","P") if swap else ("P","Q")))
            if panel == "polarity":
                yes, no = (
                    (f"The shipment contains {state}.", f"The shipment does not contain {alt}."),
                    (f"The audio archive has {state}.", f"The audio archive does not have {alt}."),
                    (f"The sign mentions {state}.", f"The sign does not mention {alt}."),
                )[framing]
                report = "\n".join((no,yes) if reverse else (yes,no))
                goal = ("Identify the entry described by the affirmative sentence. Use the entry that is present, "
                        "rather than the entry that the report explicitly excludes.")
            else:
                action = "Adopt the proposed label." if change else "Retain the original label."
                report = (f"{('Shelter','Survey','Dispatch')[framing]} register\n"
                          f"Original label: {old}.\nProposed label: {new}.\nDecision: {action}")
                goal = "Determine the live label by applying the stated decision to this register."
            text = (f"{goal} Write the live entry itself as State, then copy the independent reference word into Note. "
                    "Finally translate the State using the current Codebook.\n\n"
                    "An unrelated example of the response format:\nState: bronze\nNote: wood\nCode: Q\n\n"
                    f"Record for this question:\n{report}\nReference word: {note}\n"
                    f"Codebook:\n{old} -> {codes[old]}\n{new} -> {codes[new]}\n\n"
                    "Give the actual result with exactly three lines and no extra text:\n"
                    "State: <entry word>\nNote: <reference word>\nCode: <code>\n"
                    "Use P and Q only in the Code line.")
            rows.append(dict(id=f"{family}-{change}{swap}",family=family,suite=panel,split=split,index=i,
                             frame=framing,old=old,new=new,notes=[note],update=change,swap=swap,
                             state=state,alt=alt,code=codes[state],alt_code=codes[alt],
                             second_keeps=None,voice=f"f{framing}-o{reverse}" if panel=="polarity" else f"f{framing}",text=text))
    assert len(rows)==len({r["text"] for r in rows})==(48 if is_pilot else 4*N)
    return rows
