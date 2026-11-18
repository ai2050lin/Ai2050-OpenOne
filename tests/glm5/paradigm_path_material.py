"""Independent noun/frame cohorts and explicit archive swaps for the two target cells."""

from contextlib import contextmanager
from itertools import batched

import paradigm_four_material as F
from paradigm_routes_material import digest


RUN='PARADIGM-PATH-20261117-v1'
SEED=29722026
SIZES=dict(pilot=16,confirm=128,lockbox=128)
NOUNS={
    'pilot':'badges tablets pedals straps sockets stamps decals packages levers pencils sliders ribbons beacons cables flasks adapters',
    'confirm':'batteries washers gaskets shims brackets pallets rollers screws clamps hinges handles rings spools plugs fuses coils',
    'lockbox':'binders envelopes folders straws ferrules spacers magnets washers plates pockets saddles screens studs probes cassettes buckets',
}
NOUNS['confirm']=NOUNS['confirm'].replace('rings','pulleys')
NOUNS['lockbox']=NOUNS['lockbox'].replace('washers','nozzles').replace('plates','stirrups').replace('buckets','labels')
FRAMES=(
    'Message language: {language}. Rendering mode: {format}.\nUnrelated sample: {archive}\n{facts}\nUse the {rule} count and enclose its number in {grammar} brackets.',
    'Export mode: {format}. Text language: {language}.\nOld example: {archive}\n{facts}\nEnclose the {rule} count in {grammar} brackets.',
    'Answer language: {language}. Output style: {format}.\nPast example: {archive}\n{facts}\nDisplay the {rule} count in {grammar} brackets.',
)
TARGET=(11,15)
PAIRED_FUSION=32
CAUSAL=64


@contextmanager
def fresh_scope():
    originals=(F.SEED,F.OBJECTS,F.FRAMES,F.SIZES,F.RUN)
    F.SEED,F.OBJECTS,F.FRAMES,F.SIZES,F.RUN=SEED,NOUNS,FRAMES,SIZES,RUN
    try:
        yield
    finally:
        F.SEED,F.OBJECTS,F.FRAMES,F.SIZES,F.RUN=originals


def swap_archive(row,tokenizer,references):
    original=row['archive']
    prompt=row['prompt'].replace(references[original],references[1-original],1)
    prior_n=row['query'][0]+1
    first=F.encode(tokenizer,prompt)
    assert len(first)==prior_n
    ids=row['token']['ids']
    different={i for i,(u,v) in enumerate(zip(ids[:prior_n],first,strict=True)) if u!=v}
    assert different and different<=set(row['spans']['archive'])
    return dict(row,archive=1-original,prompt=prompt,token=dict(ids=[*first,*ids[prior_n:]],slots=row['token']['slots']))


def cohort(tokenizer,split):
    with fresh_scope():
        base=F.validate(tokenizer,split)
        reference=F.selected_reference(tokenizer,*F.ARCHIVES[list(SIZES).index(split)])
    assert len({r['family'] for r in base})==SIZES[split]
    clean=[];intervention=[]
    for index,group in enumerate(batched(base,16)):
        natural=sorted((dict(row,id=row['id']+f"-a{row['archive']}") for row in group),key=lambda row:row['id'])
        alternate=[]
        if split=='confirm' and index<CAUSAL:
            alternate=[swap_archive(row,tokenizer,reference) for row in group]
            alternate=[dict(row,id=row['id']+f"-a{row['archive']}") for row in alternate]
            intervention.extend(row for i in TARGET for row in sorted((natural[i],alternate[i]),key=lambda r:r['archive']))
        clean.extend(natural)
        if split=='confirm' and index<PAIRED_FUSION:
            clean.extend(alternate)
    assert len({r['id'] for r in clean})==len(clean)
    return clean,intervention


def identities(tokenizer):
    assert not any(set(names.split())&set(old.split()) for names in NOUNS.values() for old in F.OBJECTS.values())
    assert len(set(' '.join(NOUNS.values()).split()))==sum(len(x.split()) for x in NOUNS.values())
    result={}
    for split in SIZES:
        clean,causal=cohort(tokenizer,split)
        result[split]=dict(clean_views=len(clean),causal_views=len(causal),families=SIZES[split],
                           identity=digest([clean,causal]),nouns=sorted({r['noun'] for r in clean}))
    return result
