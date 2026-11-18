"""Equal-token movable slots in independent templates for suffix localization."""

from itertools import product
from random import Random

import paradigm_marker_material as previous
from paradigm_four_material import LEVELS, WIDTH, encode
from paradigm_routes_material import digest

RUN = 'PARADIGM-WINDOW-20261118-v1'
SEED = 29782026
SIZES = dict(pilot=24, discovery=32, confirm=128, lockbox=128)
CAUSAL = 64
GENERATE = 16
LAYOUTS = (0, 2, 3)
MARKERS = ('| No entry was kept.', '| No sample is given.')
EMPTY_REQUEST = 'No form is given.'
FORMAT = ('Write in academic form.', 'Write in Python form.')
ARCHIVES = (('| We spotted nine boats.', '| print("nine boats")', '| print("five tents")'),
            ('| We recorded seven maps.', '| print("seven maps")', '| print("eight flags")'))
NOUNS = dict(pilot='banjos castanets faucets brooches shingles sandals buttons aprons',
             discovery='armbands booklets camcorders gluers manacles earmuffs paddocks fences hallways obelisks sideboards satchels',
             confirm='maracas parkas brochures accordions sketches stickers handkerchiefs scarves ocarinas styli mittens pommels trowels pavers dentils candelabras',
             lockbox='pickaxes quivers headboards caltrops notebooks purses bobbles clocks gloves crucibles goblets hourglasses portholes benches hammocks sledges')

FRAMES = (
    ('Stored example: ', 'Response style: ', 'The first tally gave {a} {noun}.', 'The amended tally gave {b} {noun}.',
     'Reply in ', 'Use the ', ' tally. Enclose the quantity in ',
     ' brackets.\nThe example is unrelated. Follow the requested style. In academic style give one formal sentence. In Python style output one print call with one message string. Omit Markdown.'),
    ('Saved reference: ', 'Style instruction: ', 'Before checking, staff listed {a} {noun}.', 'After checking, staff listed {b} {noun}.',
     'Response language: ', 'Report the ', ' count, with the numeral inside ',
     ' brackets.\nUse the stated format. The saved reference belongs to a different task. For academic format give one sentence. For Python format begin with a print call containing one string. No code fences.'),
    ('Prior record: ', 'Output form: ', 'An earlier estimate put the total at {a} {noun}.', 'A second survey set the total at {b} {noun}.',
     'Write the message in ', 'Select the ', ' number and place it between ',
     ' brackets.\nIgnore the old record when deciding the answer format. Academic form means one formal sentence. Python form means a single print call with one string. Give the answer directly.'),
    ('Archive snippet: ', 'Presentation rule: ', 'Initially the register showed {a} {noun}.', 'The correction changes the register to {b} {noun}.',
     'Answer in ', 'Take the ', ' quantity. Surround the numeral with ',
     ' brackets.\nThe archive snippet is from a separate task. Use the specified presentation. Academic means a formal sentence. Python means one plain print call with one string. Skip headings and fences.'),
    ('Earlier sample: ', 'Style slot: ', 'A provisional list included {a} {noun}.', 'The corrected list now includes {b} {noun}.',
     'Message language: ', 'Choose the ', ' tally. Wrap the digits in ',
     ' brackets.\nThe earlier sample is illustrative. The style slot governs the response. Give a formal sentence for academic, or a single print call with one quoted message for Python.'),
)


def families(split):
    tuples = [(noun, a, b) for noun, a, b in product(NOUNS[split].split(), range(2, 10), range(2, 10)) if a != b]
    Random(SEED + list(SIZES).index(split)).shuffle(tuples)
    start = dict(pilot=0, discovery=0, confirm=2, lockbox=4)[split]
    return [dict(family=digest([RUN, split, noun, a, b])[:20], noun=noun, a=a, b=b, bank=(i//2)%2,
                 frame=start + (i//4)%2 if split != 'lockbox' else start)
            for i, (noun, a, b) in enumerate(tuples[:SIZES[split]])]


def case(tok, family, marker, layout, archive, cell=15):
    ast, rst, fact0, fact1, language, rule, bracket, tail = FRAMES[family['frame']]
    a, r = divmod(layout, 2)
    lang, style, logic, grammar = LEVELS[cell]
    text = ARCHIVES[family['bank']][archive]
    values = dict(a0=MARKERS[marker] if a else text, a1=text if a else MARKERS[marker],
                  r0=EMPTY_REQUEST if r else FORMAT[style], r1=FORMAT[style] if r else EMPTY_REQUEST,
                  fact0=fact0.format(**family), fact1=fact1.format(**family),
                  language=('English', 'Chinese')[lang], rule=('initial', 'revised')[logic],
                  grammar=('round', 'square')[grammar])
    pieces = ((ast, 'a0'), ('\n'+rst, 'r0'), ('\n', 'fact0'), ('\n', 'fact1'), ('\n'+ast, 'a1'),
              ('\n'+rst, 'r1'), ('\n'+language, 'language'), ('.\n'+rule, 'rule'), (bracket, 'grammar'))
    body = ''
    spans = {}
    for lead, name in pieces:
        body += lead
        spans[name] = (len(body), len(body) + len(values[name]))
        body += values[name]
    body += tail
    prompt = tok.apply_chat_template([dict(role='user', content=body)], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    start = prompt.index(body)
    spans = {name: [i for i, (x, y) in enumerate(enc.offset_mapping) if x < start+hi and y > start+lo]
             for name, (lo, hi) in spans.items()}
    assert [len(spans[k]) for k in ('a0', 'a1', 'r0', 'r1')] == [6, 6, 5, 5], (family['frame'], archive, spans)
    ids = list(enc.input_ids)
    prefix = prompt + ('print("' if style else '')
    code = encode(tok, prefix)
    prefix += ('The reported count is', '报告的计数为')[lang]
    core = encode(tok, prefix)
    ends = [encode(tok, prefix + b) for b in (' (', ' [')]
    gold, foil = (family['b'], family['a']) if logic else (family['a'], family['b'])
    labels = [encode(tok, prefix + (' (', ' [')[grammar] + str(x)) for x in (gold, foil)]
    assert code[:len(ids)] == ids and core[:len(code)] == code and all(z[:-1] == core for z in ends)
    assert all(z[:-1] == ends[grammar] for z in labels) and len(ends[grammar]) <= WIDTH
    sites = [spans[k][-1]+1 if k in ('a0','a1','r0','r1') else spans[k][-1]
             for k in ('a0','r0','fact0','fact1','a1','r1','language','rule','grammar')]
    sites.append(len(ids)-4)
    assert len(set(sites)) == len(sites) and max(sites) < len(ids)-1
    return dict(family, id=f"{family['family']}-m{marker}-l{layout}-a{archive}-{cell:04b}", marker=marker, layout=layout,
                cell=cell, dims=[lang,style,logic,grammar], archive=archive, prompt=prompt, spans=spans, sites=sites,
                source=spans[f'a{a}'], token=dict(ids=ends[grammar], slots=[len(ends[grammar])-1]),
                query=[len(ids)-1,len(code)-1,len(core)-1,len(ends[grammar])-1], delimiters=[z[-1] for z in ends],
                number=[z[-1] for z in labels], gold=gold, foil=foil, free_ids=ids)


def verify(rows):
    table = {(x['marker'], x['layout'], x['archive']): x for x in rows}
    for (marker, layout, archive), row in table.items():
        a, r = divmod(layout, 2)
        def diff(other):
            assert all(row[k] == other[k] for k in ('query', 'sites', 'spans'))
            return {i for i, (x, y) in enumerate(zip(row['free_ids'], other['free_ids'], strict=True)) if x != y}
        d = diff(table[marker^1, layout, archive])
        assert d and d <= set(row['spans'][f'a{1-a}'])
        for v in range(3):
            if v != archive:
                d = diff(table[marker, layout, v])
                assert d and d <= set(row['source'])
        neighbors = [(layout^2, 'a0', 'a1'), (layout^1, 'r0', 'r1')]
        for target, left, right in neighbors:
            if target not in LAYOUTS:
                continue
            other = table[marker, target, archive]
            l, r = row['spans'][left], row['spans'][right]
            d = diff(other)
            assert d <= set(l+r) and d & set(l) and d & set(r)
            assert [row['free_ids'][i] for i in l] == [other['free_ids'][i] for i in r]
            assert [row['free_ids'][i] for i in r] == [other['free_ids'][i] for i in l]


def material(tok, split):
    sweep = []
    factors = []
    style = []
    for i, family in enumerate(families(split)):
        views = [case(tok, family, m, l, a) for m, l, a in product(range(2), LAYOUTS, range(3))]
        verify(views)
        sweep.extend(views)
        if split == 'pilot':
            factors.extend(case(tok, family, i%2, LAYOUTS[(i//2)%3], (i//6)%2, c) for c in range(16))
            style.extend(case(tok, family, m, l, a, 11) for m, l, a in product(range(2), LAYOUTS, range(2)))
    return sweep, factors, style


def inventory(tok):
    old = previous.P
    past = ' '.join([*old.F.OBJECTS.values(), *old.P.NOUNS.values(), *old.A.NOUNS.values(), *old.NOUNS.values(),
                    *previous.O.NOUNS.values(), *previous.NOUNS.values()]).split()
    words = ' '.join(NOUNS.values()).split()
    assert len(words) == len(set(words)) and not set(words).intersection(past), set(words).intersection(past)
    return {s: dict(families=n, views=list(map(len, pack)), sha256=digest(pack))
            for s, n in SIZES.items() for pack in (material(tok, s),)}
