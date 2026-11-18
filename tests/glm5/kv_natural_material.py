"""Phase 2945 material. Cross competing codes, generated distance and field order."""

from hashlib import sha256
from itertools import product
import random


SEED = 29452026
N = {"pilot": 12, "discover": 32, "confirm": 128}
FIELDS = ("State", "Note", "Spare", "Archive")
DEFAULT_BREAKS = ("  \n",)*5
VALID_BREAKS = ("\n", "  \n", "\r\n", "\n\n", " \n", "  \n\n", "\r\n\r\n")
RECEIPTS = (
    "Archive check complete.",
    "Archive check complete. This receipt confirms that all fields were copied into the record and that the record was filed in the active register for the current review. The review is now ready for the final entry.",
)
VOCAB = {
    "pilot": "furnace tablet lantern sofa bucket cabinet envelope scarf shelf ladder fence pearl cloak tie uniform gauge paddle switch monitor hopper socket screen freezer fridge chimney chimney antenna antenna console buckle parcel barrel dispenser lock latch cushion mattress rug mat hallway corridor hallway walkway gate pulley zipper garage warehouse cloister crate scaffold",
    "discover": "doctor lawyer judge sailor pilot baker butcher farmer painter actor dancer singer porter barber mason tailor nurse clerk chef captain coach guard driver plumber banker broker ranger scout cobbler carpenter hunter florist mechanic teacher editor poet writer porter boxer mayor senator soldier monk prince queen king wizard witch director clerk steward servant smith miller tailor miner usher priest officer merchant artist dancer waiter valet cook janitor cashier dancer accountant",
    "confirm": "quartz granite basalt marble copper nickel silver platinum zinc sodium cobalt iron lithium silicon sulfur neon argon radon diamond sapphire emerald ruby graphite aluminum uranium titanium chlorine hydrogen oxygen silver oxide crystal opal zircon iodine lead calcite shale slate clay sand gypsum obsidian mercury bronze steel carbon oxygen calcium potassium magnesium phosphorus fluoride sulfur bromine alloy metal amber onyx coral pearl lava flint topaz garnet jasper stone gold tin rock pebble chalk mineral iodine mica coal crystal bronze sand iron aluminium selenium chromium helium xenon indium osmium iridium plastic rubber wax resin ether alcohol acid syrup tar bitumen glycerin silicon sodium plaster cement concrete asphalt graphite porcelain glass ceramic glaze copper steel brick gravel limestone sandstone quartzite mud chalk salt fossil pigment dye ink silicon",  # tokenizer and history filter exact lexical identities
}


def generate(split, tok, exclude, revision=1, families=None):
    words = [w for w in dict.fromkeys(VOCAB[split].split())
             if w not in exclude and len(tok.encode(" "+w, add_special_tokens=False)) == 1]
    assert len(words) >= 24, (split, words)
    if families is None:
        families = N[split]
    rng = random.Random(f"{SEED}/{split}/{revision}")
    used = set()
    rows = []
    for j in range(families):
        while True:
            choice = rng.sample(words, 5)
            key = tuple(sorted(choice))
            if key not in used:
                used.add(key)
                break
        state, note, spare, archive, rejected = choice
        code, wrong = ("P", "Q") if j % 2 == 0 else ("Q", "P")
        context = (j//2) % 2
        adopt = bool(rng.getrandbits(1))
        order = rng.sample([*choice], len(choice))
        fid = sha256((split+"/"+"/".join(choice)).encode()).hexdigest()[:16]
        for compete, distance, field_order in product((0, 1), repeat=3):
            values = dict(zip(FIELDS, choice[:4], strict=True))
            fields = (("Note", "State") if field_order else ("State", "Note"))+FIELDS[2:]
            if context:
                initial, proposed = (rejected, state) if adopt else (state, rejected)
                decision = "Adopt the proposal." if adopt else "Keep the original."
                facts = f"Original entry: {initial}\nProposed entry: {proposed}\nDecision: {decision}"
                rule = "State is the active entry after applying the decision."
            else:
                positive = f"The inventory includes {state}."
                negative = f"The inventory excludes {rejected}."
                facts = "\n".join((positive, negative) if adopt else (negative, positive))
                rule = "State is the entry the inventory includes."
            alternatives = {note, rejected} | ({spare, archive} if compete else set())
            codebook = "\n".join(f"{w} -> {wrong if w in alternatives else code}" for w in order)
            receipt = RECEIPTS[distance]
            layout = "\n".join([*(f"{x}: <word>" for x in fields), f"Receipt: {receipt}", "Code: <P or Q>"])
            exact = "\n".join([*(f"{x}: {values[x]}" for x in fields), f"Receipt: {receipt}", "Code:"])
            prompt = (f"Complete the record below. {rule} Copy Note, Spare and Archive from their entries. "
                      "Look up State in the codebook to find its code.\n\n"
                      f"{facts}\nNote entry: {note}\nSpare entry: {spare}\nArchive entry: {archive}\n"
                      f"Codebook:\n{codebook}\n\n"
                      "Return exactly these six plain-text lines in order. Replace the angle-bracket placeholders. "
                      "Copy the Receipt line exactly. Use the code letter by itself after Code:.\n"
                      f"{layout}")
            rows.append(dict(id=f"{fid}-{compete}{distance}{field_order}", family=fid, split=split, prompt=prompt,
                             prefix=exact, fields=fields, values=values, rejected=rejected, code=code,
                             competitors=[x for x in FIELDS[1:] if values[x] in alternatives], distance=distance,
                             compete=compete, order=field_order, context=context))
    assert len({r["prompt"] for r in rows}) == len(rows) == families*8
    return rows


def token_plan(tok, row, breaks=DEFAULT_BREAKS):
    chat = tok.apply_chat_template([dict(role="user", content=row["prompt"])], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    lines = row["prefix"].split("\n")
    assert len(lines) == len(breaks)+1
    suffix = "".join(x+y for x,y in zip(lines,[*breaks,""],strict=True))
    text = chat+suffix
    inside = tok(chat, add_special_tokens=False).input_ids
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded.input_ids
    assert ids[:len(inside)] == inside and len(inside) < len(ids)-1, row["id"]
    letters = []
    for code in "PQ":
        tail = tok(text+" "+code, add_special_tokens=False).input_ids
        assert tail[:-1] == ids and tok.decode(tail[-1]) == " "+code, (row["id"], code)
        letters.append(tail[-1])
    gen, prompt = [], []
    for label in FIELDS:
        word = " "+row["values"][label]
        span = len(chat)+suffix.index(label+":"+word)+len(label)+1
        places = [j for j, (x, y) in enumerate(encoded.offset_mapping) if x == span and y == span+len(word)]
        assert len(places) == 1 and tok.decode(ids[places[0]]) == word
        gen.extend(places)
        prompt.append([idx for idx, token in enumerate(ids[:len(inside)]) if token == ids[places[0]]])
    assert max(gen) < len(ids)-1
    ends = []
    cursor = len(chat)
    for line,sep in zip(lines[:-1],breaks,strict=True):
        cursor += len(line)
        matches = [j for j,(x,y) in enumerate(encoded.offset_mapping) if x<=cursor and y==cursor+len(sep)]
        assert len(matches)==1 and tok.decode(ids[matches[0]]).endswith(sep),(row["id"],sep)
        ends.extend(matches)
        cursor += len(sep)
    return dict(ids=ids, start=len(inside), slots=gen, prompt_slots=prompt, letters=letters,
                gap=[len(ids)-1-j for j in gen], breaks=list(breaks), break_positions=ends)
