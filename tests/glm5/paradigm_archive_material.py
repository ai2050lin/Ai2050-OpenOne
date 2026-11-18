"""Tokenizer-frozen archive-format, wording and punctuation crosses."""

from contextlib import contextmanager
from itertools import batched, product

import paradigm_four_material as F
import paradigm_path_material as P
from paradigm_routes_material import digest


RUN='PARADIGM-ARCHIVE-20261118-v1'
SEED=29732026
SIZES=dict(pilot=16,discovery=24,confirm=128,lockbox=128)
NOUNS=dict(
    pilot='cylinders clasps spouts lanterns hoops vents sheets stoppers',
    discovery='spindles panels barrels bushings bins lockers pistons cranks bells rails reams cleats',
    confirm='staples rulers grommets covers sprockets hampers casings dowels switches axles stands carriers leashes casters inserts sleeves',
    lockbox='zippers cushions curtains bands blankets bonnets carts brooms baskets mirrors placards boards combs shutters awls valves')
FRAMES=dict(
    pilot=('Target language: {language}. Response form: {format}.\nStore entry: {archive}\n{facts}\nPlace the {rule} tally in {grammar} brackets.',),
    discovery=('Recorded example {archive}\nAnswer language: {language}. Requested form: {format}.\n{facts}\nRender the {rule} count in {grammar} brackets.',),
    confirm=('Historical entry {archive}\n{facts}\nRender the {rule} tally in {grammar} brackets. Use {language} in {format} form.',
             'Select a {format} response in {language}.\nSaved item {archive}\n{facts}\nWrite the {rule} count within {grammar} brackets.'),
    lockbox=('Output mode: {format}.\nEarlier item {archive}\n{facts}\nPut the {rule} tally within {grammar} brackets. The answer language is {language}.',))
# The first two banks are discovery. Confirmation uses new payloads, two prose
# carriers and one punctuation-identical lexical challenge.
BANKS=(
    ('We noted {}.','print("{}")',('four charts','four graphs')),
    ('We found {}.','print("{}")',('seven records','seven listings')),
    ('We confirm {}.','print("{}")',('six images','six pictures')),
    ('We noted {}.','print("{}")',('nine graphs','nine plots')),
    ('Report ("{}")','print ("{}")',('five objects','five items')),
    ('We found {}.','print("{}")',('three reports','three summaries')),
)
SPLIT_BANKS=dict(pilot=(0,1),discovery=(0,1),confirm=(2,3,4),lockbox=(5,))
TARGETS=(11,15)
ARCV=tuple(product(range(2),repeat=3))
CAUSAL=64
CROSS=32


def archive(bank,variant):
    a,w,p=variant
    return ('|',';')[p]+' '+BANKS[bank][a].format(BANKS[bank][2][w])


@contextmanager
def scope(split,frame):
    previous=(F.RUN,F.SEED,F.SIZES,F.OBJECTS,F.FRAMES,F.ARCHIVES)
    seed_pair=tuple(archive(0,(a,0,0)) for a in range(2))
    F.RUN,F.SEED,F.SIZES,F.OBJECTS=RUN,SEED,SIZES,NOUNS
    F.FRAMES=(frame,)*4
    F.ARCHIVES=(seed_pair,)*4
    try:
        yield
    finally:
        F.RUN,F.SEED,F.SIZES,F.OBJECTS,F.FRAMES,F.ARCHIVES=previous


def with_archive(tokenizer,row,frame,bank,variant):
    old=archive(0,(row['archive'],0,0))
    new=archive(bank,variant)
    prompt=row['prompt'].replace(old,new,1)
    current=F.encode(tokenizer,prompt)
    e=row['query'][0]+1
    ref=row['token']['ids']
    assert len(current)==e and len(row['spans']['archive'])==6,(old,new,e,len(current))
    assert all(a==b or i in row['spans']['archive'] for i,(a,b) in enumerate(zip(current,ref[:e],strict=True)))
    cell=''.join(map(str,row['dims']))
    return dict(row,id=f"{row['family']}-{cell}-r{bank}-{''.join(map(str,variant))}",cell=int(cell,2),
                frame=frame,bank=bank,variant=variant,archive=variant[0],prompt=prompt,
                token=dict(ids=current+ref[e:],slots=row['token']['slots']))


def material(tokenizer,split):
    variants=[]
    for frame in FRAMES[split]:
        with scope(split,frame):
            variants.append(list(batched(F.validate(tokenizer,split),16)))
    natural=[];control=[];paired=[]
    for i in range(SIZES[split]):
        f=i%len(variants)
        group=variants[f][i]
        bank=SPLIT_BANKS[split][(i//2)%len(SPLIT_BANKS[split])]
        a,w,p=(i//len(variants))%2,(i//4)%2,(i//8)%2
        convert=lambda row,v:with_archive(tokenizer,row,f,bank,v)
        natural.extend(convert(row,(a,w,p)) for row in group)
        if split=='discovery' or (split=='confirm' and i<CROSS):
            for c in TARGETS:
                control.extend(convert(group[c],v) for v in ARCV)
        if split=='confirm' and i<CAUSAL:
            for c in TARGETS:
                paired.extend(convert(group[c],(s,w,p)) for s in range(2))
    for group in batched(control,8):
        ids=[row['token']['ids'] for row in group]
        assert all(len(x)==len(ids[0]) for x in ids)
        span=set(group[0]['spans']['archive'])
        assert all(a==b or i in span for x in ids for i,(a,b) in enumerate(zip(ids[0],x,strict=True)))
        assert len(set(tuple(x) for x in ids))==8
    return natural,control,paired


def inventory(tokenizer):
    prior=' '.join([*F.OBJECTS.values(),*P.NOUNS.values()]).split()
    words=' '.join(NOUNS.values()).split()
    assert len(set(words))==len(words) and not (set(words)&set(prior)),set(words)&set(prior)
    counts={b:{len(F.encode(tokenizer,' '+archive(b,v)+'\n')) for v in ARCV} for b in range(len(BANKS))}
    assert all(value=={6} for value in counts.values()),counts
    result={}
    for split in SIZES:
        rows=material(tokenizer,split)
        result[split]=dict(families=SIZES[split],views=list(map(len,rows)),identity=digest(rows))
    return result
