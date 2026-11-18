"""Independent archive- and request-placeholder swaps at identical token geometry."""

from contextlib import contextmanager
from itertools import product

import paradigm_order_material as O


P=O.P
RUN='PARADIGM-MARKER-20261118-v1'
SEED=29762026
SIZES=dict(pilot=16,discovery=32,confirm=128,lockbox=128)
NOUNS=dict(pilot='mallets hatches sashes coffers spatulas scrapers chippers tripods',
    discovery='stanchions cleavers lancets trunnions looms rakes taps bladders grilles tackles timbers hawsers',
    confirm='ladles rivets tongs lugs jockeys hubs winches turnstiles clevises visors seals cuffs markers finials rivulets trinkets',
    lockbox='bollards mortars fillets vessels caskets canisters drains pouches cradles tiaras tampers strings grips sills pastels bobbins')
OLD=(P.EMPTY_A,P.EMPTY_R)
NEW=(O.EMPTY_A,O.EMPTY_R)
LAYOUTS=O.LAYOUTS
EDGES=O.EDGES
MARKERS=tuple(product(range(2),repeat=2))
CAUSAL=64


@contextmanager
def scope():
    prior=(P.RUN,P.SEED,P.SIZES,P.NOUNS,P.EMPTY_A,P.EMPTY_R)
    P.RUN,P.SEED,P.SIZES,P.NOUNS=RUN,SEED,SIZES,NOUNS
    try:
        yield
    finally:
        P.RUN,P.SEED,P.SIZES,P.NOUNS,P.EMPTY_A,P.EMPTY_R=prior


def use(index):
    a,r=MARKERS[index]
    P.EMPTY_A=(OLD[0],NEW[0])[a]
    P.EMPTY_R=(OLD[1],NEW[1])[r]


def label(row,marker):
    return dict(row,id=f"m{marker}-{row['id']}",marker=marker)


def verify_markers(views):
    indexed={(row['marker'],row['layout'],row['cell'],row['archive']):row for row in views}
    for row in views:
        m,l,c,v=(row[k] for k in ('marker','layout','cell','archive'))
        a,r=P.LAYOUTS[l]
        for bit,field in ((2,f'a{1-a}'),(1,f'r{1-r}')):
            other=indexed[m^bit,l,c,v]
            assert (row['spans'],row['query'],row['sites'])==(other['spans'],other['query'],other['sites'])
            x,y=row['token']['ids'],other['token']['ids']
            changed={i for i,(u,z) in enumerate(zip(x,y,strict=True)) if u!=z}
            assert changed and changed<=set(row['spans'][field])


def build(tokenizer,split):
    sweep=[];factor=[]
    with scope():
        for n,family in enumerate(P.rows_for(tokenizer,split)):
            group=[]
            for marker in range(4):
                use(marker)
                for cell in ((11,15) if split=='pilot' else (15,)):
                    generated=[P.compile_case(tokenizer,family,cell,l,a) for l,a in product(range(4),range(2))]
                    P.verify_group(generated)
                    group.extend(label(row,marker) for row in generated if row['layout'] in LAYOUTS)
            verify_markers(group)
            sweep.extend(group)
            if split=='pilot':
                m=n%4
                use(m)
                factor.extend(label(P.compile_case(tokenizer,family,c,LAYOUTS[(n//4)%3],(n//3)%2),m)
                              for c in range(16))
    return sweep,factor


def inventory(tokenizer):
    old_words=' '.join([*P.F.OBJECTS.values(),*P.P.NOUNS.values(),*P.A.NOUNS.values(),
                        *P.NOUNS.values(),*O.NOUNS.values()]).split()
    words=' '.join(NOUNS.values()).split()
    assert len(set(words))==len(words) and not set(words)&set(old_words),set(words)&set(old_words)
    return {split:dict(families=N,views=list(map(len,rows)),identity=O.digest(rows)) for split,N in SIZES.items()
            for rows in (build(tokenizer,split),)}
