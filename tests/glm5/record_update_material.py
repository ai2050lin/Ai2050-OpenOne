"""Deterministic record-update families and a natural, uncoerced output contract."""

from collections import Counter
from hashlib import sha256
from itertools import product
import json
import random
import re


RUN = 'RECORD-UPDATE-20261117-v1'
SEED = 29572026
SIZES = dict(pilot=28, discovery=192, words=384, frames=384, combos=384, lockbox=384)
READ_LAYER = 20
SOURCE_LAYER = 25
SOURCE_HEAD = 18
WIDTH = 320
MAX_NEW = 32
BEHAVIOR_FLOOR = .85
VOCABULARY = {
    'pilot': 'Onyx Terra Finch Auburn River Canon Spark Silver Quill Mercury Cove Ladder Polar Crown Millet Clover',
    'known': 'Velvet Haze Lake Lantern Grove Spruce Parrot Falcon Emerald Acorn Compass Tide Ribbon Almond Laurel Hornet Marin Shadow Lime Orchard Coral Cotton Boulder Saturn Veil Heather Meteor Banner Cricket Marble Ginger Cargo Dove Mint Arrow Morgan Tiger Mesa Delta Violet Carrot Rabbit Salmon Gecko Jade Buffalo Coconut Honey Riverstone Sky Echo Arbor Planet Cannon Grovewood Pebble Silverline Robin Violetwood',
    'words': 'Quartz Chess Maple Walnut Glacier Vapor Carbon Willow Lemon Juniper Granite Cypress Atlas Nimble Candle Orchid Aspen Amber Drift Moss Harbor Grain Silverstone Tango Fjord Copper Eagle Ivory Cypress Topaz Basil Crimson Prism Falconer',
    'lockbox': 'Archer Cobalt Frost Lotus Nimbus Ocean Panda Raven Winter Forest Iron Petal Saber Sable Shell Stone Clove Talon Thorn Velvetine Weaver Dawn Ember Summit Birch Denim Falconry Cedar Pepper Onyxia Comet Canvas Hazel Tiger',
}
POOL = dict(pilot='pilot', discovery='known', words='words', frames='known', combos='known', lockbox='lockbox')
FRAMES = dict(pilot=(0, 1), discovery=(0, 1), words=(0, 1), frames=(2, 3), combos=(0, 1), lockbox=(2, 3))
LABELS = (
    ('North', 'South', 'warehouse bins'),
    ('East', 'West', 'meeting rooms'),
    ('Upper', 'Lower', 'dispatch shelves'),
    ('Morning', 'Evening', 'calendar entries'),
)
ACTION = (
    ('selected', 'approved'),
    ('selected', 'pending'),
    ('other', 'approved'),
    ('other', 'pending'),
)
# The (0,0) composition applies two endorsed replacements to the same record.
# (2,2) updates the competing record twice, which must leave the selection alone.
NOVEL = ((0, 0), (2, 2))
KNOWN = tuple(p for p in product(range(4), repeat=2) if p not in NOVEL)
OUTPUT = re.compile(r'[ \t]*([A-Za-z][A-Za-z-]*)[ \t]*\r?\n[ \t]*Code:[ \t]*([PQ])[ \t]*(?:\r?\n)?[ \t]*')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def vocabulary(tokenizer):
    pools = {}
    used = set()
    for stage, raw in VOCABULARY.items():
        pool = []
        for word in raw.split():
            ids = tokenizer.encode(' '+word, add_special_tokens=False)
            if len(ids) == 1 and tokenizer.decode(ids) == ' '+word and ids[0] not in used:
                used.add(ids[0])
                pool.append(word)
        if len(pool) < 12:
            raise ValueError(f'{stage}: only {len(pool)} intact single-token labels')
        pools[stage] = tuple(pool)
    return pools


def truth(names, active, actions):
    state = list(names[:2])
    history = [state[active]]
    for i, action in enumerate(actions):
        scope, approval = ACTION[action]
        if approval == 'approved':
            addressed = active if scope == 'selected' else 1-active
            state[addressed] = names[i+2]
        history.append(state[active])
    return history


def render(frame, labels, active, actions, codebook, order):
    left, right, domain = LABELS[frame]
    slots = (left, right)
    actor = slots[active]
    messages = [
        f'You maintain two {domain}. Initial labels: {left} = {labels[0]}; {right} = {labels[1]}. Audit only {actor}.',
        f'A record supervisor maintains {domain}. Start with label {labels[0]} in {left} and {labels[1]} in {right}. The ticket is for {actor}.',
        f'This ticket requests the last saved label in {actor}. The current labels of these {domain} are {labels[0]} in {left} and {labels[1]} in {right}.',
        f'Find the final label for {actor} in this set of {domain}. At the start {right} holds {labels[1]}, while {left} holds {labels[0]}.',
    ]
    rules = (
        'Read both entries in order. Apply an approved entry to its named record only. Pending entries leave the record unchanged.',
        'Process the two notices chronologically. A notice changes the label of the stated record only if it is marked approved.',
        'Changes marked pending have no effect. Use each approved notice to revise its own record when that notice occurs.',
        'Go from the earlier to the later notice. For the named record, adopt its new label if approval was granted. Skip pending notices.',
    )
    notices = []
    for i, action in enumerate(actions):
        scope, approval = ACTION[action]
        chosen = slots[active if scope == 'selected' else 1-active]
        value = labels[i+2]
        layout = (
            f'Entry {i+1}: {chosen} requests label {value}. Status: {approval}.',
            f'Notice {i+1}: proposed label {value} for {chosen}, marked {approval}.',
            f'{("Earlier", "Later")[i]} notice: approval is {approval}. Its destination is {chosen} and its label is {value}.',
            f'{("First", "Then")[i]}, a {approval} proposal assigns {value} to {chosen}.',
        )
        notices.append(layout[frame])
    table = ', '.join(f'{labels[j]} -> {codebook[j]}' for j in order)
    instructions = ('Reply in exactly two lines. Put the final label of the audited record on the State line. '
                    'Use the key to put its letter on the Code line. End after the letter. Format:\nState: <label>\nCode: <P or Q>')
    return messages[frame], rules[frame], notices, f'Letter key: {table}.', instructions


def families(tokenizer, stage):
    words = vocabulary(tokenizer)[POOL[stage]]
    frames = FRAMES[stage]
    known_signatures = set()
    # Enforce a distinct ordered initial-label pair across familiar-vocabulary splits.
    initial_pairs = list(product(range(len(words)), repeat=2))
    initial_pairs = [p for p in initial_pairs if p[0] != p[1]]
    random.Random(digest([SEED, POOL[stage], 'initial-pair'])).shuffle(initial_pairs)
    raw_offset = dict(pilot=0, discovery=0, words=0, frames=192, combos=576, lockbox=0)[stage]
    if POOL[stage] == 'known' and raw_offset+SIZES[stage] > len(initial_pairs):
        raise ValueError('Familiar-lexicon initial pairs must stay disjoint')
    for i in range(SIZES[stage]):
        rng = random.Random(digest([SEED, stage, i]))
        j, k = initial_pairs[(raw_offset+i) % len(initial_pairs)]
        tail = [x for x in words if x not in (words[j],words[k])]
        while True:
            names = (words[j], words[k], *rng.sample(tail, 2))
            if names not in known_signatures:
                known_signatures.add(names)
                break
        family = digest([RUN, stage, i, names])[:20]
        order = rng.sample(range(4),4)
        codes = rng.sample('PPQQ',4)
        frame = frames[i % len(frames)]
        active = (i//2) % 2
        if stage in ('combos', 'lockbox'):
            operations = NOVEL
        else:
            m = (i//4) % len(KNOWN)
            operations = (KNOWN[m],KNOWN[(m+len(KNOWN)//2) % len(KNOWN)])
        for branch, actions in enumerate(operations):
            history = truth(names, active, actions)
            answer = codes[names.index(history[-1])]
            intro, rule, notices, key, request = render(frame,names,active,actions,codes,order)
            yield dict(id=f'{family}-{branch}',family=family,split=stage,frame=frame,active=active,
                       names=names,codes=codes,actions=actions,truth=history,answer=answer,
                       sections=(intro,rule,*notices,key,request))


def encode(tokenizer, row):
    rendered = '\n'.join(row['sections'])
    text = tokenizer.apply_chat_template([dict(role='user',content=rendered)], tokenize=False,
                                         add_generation_prompt=True,enable_thinking=False)+'State:'
    offset = text.index(rendered)
    encoded = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    tokens = encoded['input_ids']
    limit = offset
    spans = []
    for section in row['sections']:
        stop = limit+len(section)
        spans.append([i for i,(lo,hi) in enumerate(encoded['offset_mapping']) if lo < stop and hi > limit])
        limit = stop+1
    assert all(spans) and len({idx for group in spans for idx in group})==sum(map(len,spans))
    assert max(spans[2]) < min(spans[3])
    message = f" {row['truth'][-1]}\nCode: {row['answer']}"
    eos = tokenizer.eos_token
    all_ids = tokenizer.encode(text+message+eos,add_special_tokens=False)
    if all_ids[:len(tokens)] != tokens:
        raise ValueError(f"Unstable State boundary for {row['id']}")
    draft = all_ids[len(tokens):]
    code_start = tokenizer.encode(text+message.rsplit(' ',1)[0],add_special_tokens=False)
    assert code_start == all_ids[:len(code_start)]
    if len(tokens)+MAX_NEW > WIDTH:
        raise ValueError(f"context budget exceeds width for {row['id']}")
    return dict(row,token=dict(prompt=tokens,sections=spans,e=len(tokens)-1,stage1=max(spans[2]),
                               draft=draft,expected_state=draft[0],expected_code=all_ids[len(code_start)],
                               code_after=len(code_start)-len(tokens)))


def inspect(tokenizer, sizes=SIZES):
    """Verify causal prefix identity, lexical cutoffs and composition support in memory."""
    p, sets, pairs, initials = {}, {}, {}, {}
    acquired = {}
    for stage in sizes:
        rows = [encode(tokenizer,r) for r in families(tokenizer,stage)]
        assert len(rows)==2*sizes[stage]
        sets[stage] = {tuple(r['names']) for r in rows}
        pairs[stage] = {tuple(r['actions']) for r in rows}
        initials[stage] = {tuple(r['names'][:2]) for r in rows}
        assert all(not sets[stage] & a for k,a in sets.items() if k != stage)
        assert all(not initials[stage] & a for k,a in initials.items() if k != stage)
        cells = Counter((r['frame'],r['active'],tuple(r['actions'])) for r in rows)
        if stage in ('discovery','words','frames'):
            assert pairs[stage] == set(KNOWN)
        elif stage in ('combos','lockbox'):
            assert pairs[stage] == set(NOVEL)
        assert all(r['token']['stage1']<r['token']['e'] for r in rows)
        assert all(r['token']['draft'][-1]==tokenizer.eos_token_id for r in rows)
        p[stage]=dict(families=sizes[stage],views=len(rows),sha256=digest(rows),cells=len(cells),
                      min_cell=min(cells.values()),min_prompt=min(len(r['token']['prompt']) for r in rows),
                      max_prompt=max(len(r['token']['prompt']) for r in rows),
                      vocab_pool=POOL[stage],frames=list(FRAMES[stage]))
        if stage=='discovery':
            acquired={'positions':{j:sorted({x[j] for x in pairs[stage]}) for j in range(2)}}
    assert all(v==list(range(4)) for v in acquired['positions'].values())
    return p
