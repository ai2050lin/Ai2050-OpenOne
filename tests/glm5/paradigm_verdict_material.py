"""Independent token-balanced combinations of the Phase 2978 sign-reversing frames."""

from contextlib import contextmanager
from itertools import product
from random import Random

import paradigm_window_material as W
from paradigm_four_material import GUIDE

RUN='PARADIGM-VERDICT-20261118-v1'
SEED=29792026
SIZES=dict(pilot=24, discovery=32, confirm=128, lockbox=128)
ANCHOR=64
CAUSAL=64
LAYOUTS=(2,3)
NOUNS=dict(pilot='trombones laddlebags pennants canoes saxophones tambourines cupcakes rafts',
           discovery='necklaces frostings understudies rowboats bouquets flashcards microphones pendants hairclips sidewalks snorkels pillowslips',
           confirm='bicycles kettledrums magnifiers coasters pinwheels dandelions skates hairbands barstools worksheets sandcastles quillpens slidewhistles doorbells lampshades saddles',
           lockbox='overcoats starfishes speedboats turntables jellybeans lawnmowers campfires keyrings bookcases birdhouses streetlights tablecloths handrails staircases schoolbags windsails')

# The three independently swapped components are paired headers, paired fact
# sentences, and the language/count/grammar instruction. The final guide is fixed.
RAW=tuple((*f[:-1],' brackets.\n'+GUIDE) for f in W.FRAMES[2:4])
FAMILIES=(
 (RAW[0],(*RAW[1][:2],
         'Initially the official warehouse paper register showed {a} {noun}.',
         'The latest correction changes the register to {b} {noun}.',
         'Please now answer in ',RAW[1][5],' quantity. Surround numeral with ',RAW[1][-1])),
 (('Earlier record: ','Requested form: ',
    'An earlier estimate gave the total as {a} {noun}.',
    'A second survey gave the total as {b} {noun}.',
    'Put the message in ','Choose the ',' count and place it inside ',RAW[0][-1]),
  ('History excerpt: ','Presentation instruction: ',
    'Initially the official warehouse stock register showed {a} {noun}.',
    'The latest revision changes the register to {b} {noun}.',
    'Please now respond in ','Pick the ',' quantity. Put numeral inside ',RAW[0][-1])),
)


def mix(two,corner):
    header,facts,rule=(corner>>b&1 for b in (2,1,0))
    return (*two[header][:2],*two[facts][2:4],*two[rule][4:7],two[0][-1])


FRAMES=tuple(mix(two,corner) for two in FAMILIES for corner in range(8))+RAW


@contextmanager
def scope():
    old=W.FRAMES
    W.FRAMES=FRAMES
    try:
        yield
    finally:
        W.FRAMES=old


def families(split):
    items=[(name,a,b) for name,a,b in product(NOUNS[split].split(),range(2,10),range(2,10)) if a!=b]
    Random(SEED+list(SIZES).index(split)).shuffle(items)
    return [dict(family=W.digest([RUN,split,n,a,b])[:20],noun=n,a=a,b=b,bank=(i//4)%2,marker=i%2,
                 lex=0 if split=='discovery' else (i//2)%2,
                 n=i) for i,(n,a,b) in enumerate(items[:SIZES[split]])]


def case(tok,family,corner,layout,archive,cell=15):
    frame=family['lex']*8+corner if corner<8 else 16+(corner-8)
    row=W.case(tok,dict(family,frame=frame),family['marker'],layout,archive,cell)
    return dict(row,corner=corner,id=row['id']+f'-c{corner}')


def verify(rows,both):
    table={(r['corner'],r['layout'],r['archive'],r['cell']):r for r in rows}
    reference=rows[0]
    for (c,l,a,k),row in table.items():
        if c<8:
            assert row['query'][0]==reference['query'][0] and row['spans']==reference['spans'],(
                row['frame'],row['id'],reference['spans'],row['spans'],reference['query'],row['query'])
        here=row['free_ids']
        for idx,allowed in [((c,l^1,a,k),row['spans']['r0']+row['spans']['r1']),
                            ((c,l,1-a,k),row['source'])]:
            other=table[idx]
            change={j for j,(x,y) in enumerate(zip(here,other['free_ids'],strict=True)) if x!=y}
            assert change and change<=set(allowed),idx
        if both and c<8:
            opposite=table[c,l,a,26-k]
            change={j for j,(x,y) in enumerate(zip(here,opposite['free_ids'],strict=True)) if x!=y}
            assert change and change<=set(row['spans']['r0']+row['spans']['r1'])
    iso=[r for r in rows if r['corner']<8 and r['layout']==2 and r['archive']==0 and r['cell']==15]
    if iso:
        domains=[set() for _ in range(3)]
        for bit in (4,2,1):
            first=iso[0]['free_ids']; other=table[bit,2,0,15]['free_ids']
            delta={j for j,(x,y) in enumerate(zip(first,other,strict=True)) if x!=y}
            assert delta
            domains[(4,2,1).index(bit)]=delta
        assert sum(map(len,domains))==len(set.union(*domains))
        for r in iso:
            allowed=set.union(set(),*(domains[i] for i,bit in enumerate((4,2,1)) if r['corner']&bit))
            change={j for j,(x,y) in enumerate(zip(iso[0]['free_ids'],r['free_ids'],strict=True)) if x!=y}
            assert change==allowed,(r['id'],change,allowed)


def build(tok,split):
    panel=[]; anchors=[]; foils=[]; axes=[]
    with scope():
        for f in families(split):
            sweep=[case(tok,f,c,l,a) for c,l,a in product(range(8),LAYOUTS,range(2))]
            fac=[case(tok,f,c,l,a,11) for c,l,a in product(range(8),LAYOUTS,range(2))] if split=='pilot' else []
            alt=[case(tok,f,c,l,a) for c,l,a in product((8,9),LAYOUTS,range(2))] if split=='confirm' and f['n']<ANCHOR else []
            verify(sweep+fac+alt,bool(fac))
            panel.extend(sweep);anchors.extend(alt);foils.extend(fac)
            if split=='pilot':
                layout=LAYOUTS[(f['n']//2)%2]
                axes.extend(case(tok,f,f['n']%8,layout,(f['n']//4)%2,cell) for cell in range(16))
    return panel,anchors,foils,axes


def inventory(tok):
    existing=' '.join([*W.NOUNS.values(),*W.previous.NOUNS.values(),*W.previous.O.NOUNS.values(),
                      *W.previous.P.NOUNS.values(),*W.previous.P.F.OBJECTS.values()]).split()
    words=' '.join(NOUNS.values()).split()
    assert len(words)==len(set(words)) and not set(existing).intersection(words),set(existing).intersection(words)
    return {split:dict(families=count,views=list(map(len,rows)),sha256=W.digest(rows))
            for split,count in SIZES.items() for rows in (build(tok,split),)}
