"""Independent position families with new equal-token inactive slot markers."""

from contextlib import contextmanager
from itertools import product

import paradigm_position_material as P
from paradigm_routes_material import digest


RUN='PARADIGM-ORDER-20261118-v1'
SEED=29752026
SIZES=dict(pilot=16,discovery=32,confirm=128,lockbox=128)
NOUNS=dict(pilot='calipers bolsters buckles casements chisels tweezers cutters sifters',
           discovery='heaters fasteners welts paddings ducts gavels lucites pliers scoops molds valances gongs',
           confirm='cribs stilettos latches grafts barges bungs opals dividers stalls spigots hoods floats cots kerbs slips ledges',
           lockbox='totes vials crocks trestles abacuses keels vises bunks sabers basins stools braces wedges arches trusses spades')
EMPTY_A='| No sample is given.'
EMPTY_R='Nothing is specified here.'
LAYOUTS=(0,2,3)
EDGES=(('archive',0,2),('request',2,3))
CAUSAL=64


@contextmanager
def scope():
    backup=(P.RUN,P.SEED,P.SIZES,P.NOUNS,P.EMPTY_A,P.EMPTY_R)
    P.RUN,P.SEED,P.SIZES,P.NOUNS,P.EMPTY_A,P.EMPTY_R=RUN,SEED,SIZES,NOUNS,EMPTY_A,EMPTY_R
    try:
        yield
    finally:
        P.RUN,P.SEED,P.SIZES,P.NOUNS,P.EMPTY_A,P.EMPTY_R=backup


def build(tokenizer,split):
    sweep=[];factors=[]
    with scope():
        for n,family in enumerate(P.rows_for(tokenizer,split)):
            for cell in ((11,15) if split=='pilot' else (15,)):
                rows=[P.compile_case(tokenizer,family,cell,layout,arc) for layout,arc in product(range(4),range(2))]
                P.verify_group(rows)
                for row in rows:
                    if row['layout'] in LAYOUTS:
                        sweep.append(row)
            if split=='pilot':
                layout=LAYOUTS[n%3]
                factors.extend(P.compile_case(tokenizer,family,c,layout,(n//3)%2) for c in range(16))
    return sweep,factors


def inventory(tokenizer):
    prior=' '.join([*P.NOUNS.values(),*P.F.OBJECTS.values(),*P.P.NOUNS.values(),*P.A.NOUNS.values()]).split()
    names=' '.join(NOUNS.values()).split()
    assert len(set(names))==len(names) and not set(names)&set(prior),set(names)&set(prior)
    result={}
    for split,size in SIZES.items():
        rows=build(tokenizer,split)
        result[split]=dict(families=size,views=list(map(len,rows)),identity=digest(rows))
    return result
