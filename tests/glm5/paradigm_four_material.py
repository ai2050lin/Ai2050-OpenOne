"""Matched language, format, factual selection and delimiter-grammar families."""

from itertools import batched, product
from random import Random

from paradigm_routes_material import digest, selected_reference


RUN = 'PARADIGM-FOUR-20261117-v1'
SEED = 29702026
WIDTH = 160
LEVELS = tuple(product((0,1),repeat=4))
SIZES = dict(pilot=16, confirm=128, lockbox=128)
OBJECTS = {
    'pilot': 'beads seeds coins keys pins caps clips cards tiles disks drums tubes rings rods shells lamps',
    'confirm': 'springs relays plates lenses tanks gears wires chips poles reels trays bags tags hooks bolts posts',
    'lockbox': 'mats nets frames boxes crates jars cups pots pans bowls lids racks bundles packs cans buckets',
}
REGIONS = ('language','format','archive','rule','grammar','facts')
ARCHIVES = (('Results indicate six observations.', 'print("six observations")'),
            ('The sample contains eight records.', 'print("eight distinct records")'),
            ('Evidence indicates five events.', 'print("five events")'))
FRAMES = (
    'Report language: {language}. Requested format: {format}.\nArchive example: {archive}\n{facts}\nReport the {rule} count. Put the number inside {grammar} brackets.',
    'Output format: {format}. Message language: {language}.\nPast archive: {archive}\n{facts}\nPlace the {rule} count inside {grammar} brackets.',
    'Use {language}. Output style: {format}.\nUnrelated archive: {archive}\n{facts}\nUse the {rule} count, enclosed in {grammar} brackets.',
)
GUIDE = ('Use only the requested format. The archive is from another task. For academic format, write one formal sentence. '
         'For Python format, start directly with a print call containing one message string in the requested language. '
         'Do not add labels or Markdown fences.')


def encode(tokenizer,text):
    return tokenizer.encode(text,add_special_tokens=False)


def expanded(tokenizer,split):
    corpus_index=list(SIZES).index(split)
    archive=selected_reference(tokenizer,*ARCHIVES[corpus_index])
    entries=[(word,a,b) for word,a,b in product(OBJECTS[split].split(),range(2,10),range(2,10)) if a!=b]
    Random(SEED+corpus_index).shuffle(entries)
    for n,(noun,early,late) in enumerate(entries[:SIZES[split]]):
        example=n%2
        fact_frame=(n//2)%2
        family=digest([RUN,split,noun,early,late])[:20]
        for lang,style,logic,grammar in LEVELS:
            parts={
                'language':('English','Chinese')[lang],
                'format':('academic','Python')[style],
                'archive':archive[example],
                'rule':('initial','revised')[logic],
                'grammar':('round','square')[grammar],
                'facts':(f'An initial review found {early} {noun}. A revised review found {late} {noun}.',
                         f'The first count was {early} {noun}. The corrected count was {late} {noun}.')[fact_frame],
            }
            body=FRAMES[corpus_index].format(**parts)+'\n'+GUIDE
            prompt=tokenizer.apply_chat_template([dict(role='user',content=body)],tokenize=False,
                                                 add_generation_prompt=True,enable_thinking=False)
            prompt_ids=encode(tokenizer,prompt)
            continued=prompt+('print("' if style else '')
            code_ids=encode(tokenizer,continued)
            continued+=('The reported count is','报告的计数为')[lang]
            phrase=encode(tokenizer,continued)
            openings=[encode(tokenizer,continued+bracket) for bracket in (' (',' [')]
            assert all(z[:-1]==phrase for z in openings)
            gold=(early,late)[logic];foil=(early,late)[1-logic]
            final_text=continued+(' (',' [')[grammar]
            ids=openings[grammar]
            good,bad=[encode(tokenizer,final_text+str(value)) for value in (gold,foil)]
            assert good[:-1]==bad[:-1]==ids and good[-1]!=bad[-1]
            assert code_ids[:len(prompt_ids)]==prompt_ids and phrase[:len(code_ids)]==code_ids
            mapping=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True).offset_mapping
            start=prompt.index(body)
            offsets={}
            for key,word in parts.items():
                begin=prompt.index(word,start if key!='rule' else prompt.index(parts['facts'],start)+len(parts['facts']))
                end=begin+len(word)
                offsets[key]=[i for i,(left,right) in enumerate(mapping) if left<end and right>begin]
            assert len(set(sum(offsets.values(),[])))==sum(map(len,offsets.values()))
            yield dict(id=f'{family}-{lang}{style}{logic}{grammar}',family=family,dims=[lang,style,logic,grammar],
                       archive=example,noun=noun,prompt=prompt,spans=offsets,query=[len(prompt_ids)-1,len(code_ids)-1,len(phrase)-1,len(ids)-1],
                       token=dict(ids=ids,slots=[len(ids)-1]),number=[good[-1],bad[-1]],
                       delimiters=[x[-1] for x in openings],gold=gold)


def validate(tokenizer,split):
    rows=list(expanded(tokenizer,split))
    assert len(rows)==SIZES[split]*16
    for group in batched(rows,16):
        base=group[0]
        assert all(r['spans']==base['spans'] and r['query'][0]==base['query'][0] for r in group)
        assert len({r['token']['ids'][r['query'][0]] for r in group})==1
        assert max(len(r['token']['ids']) for r in group)<=WIDTH
        for row in group:
            source=('language','format','rule','grammar')
            now=row['token']['ids'][:row['query'][0]+1]
            before=base['token']['ids'][:base['query'][0]+1]
            changed={i for i,(a,b) in enumerate(zip(before,now,strict=True)) if a!=b}
            allowed=set().union(*(set(row['spans'][k]) for k,on in zip(source,row['dims'],strict=True) if on))
            assert changed<=allowed and all(changed.intersection(row['spans'][k]) for k,on in zip(source,row['dims'],strict=True) if on)
    return rows
