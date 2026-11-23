"""Use explicit physical top/bottom rows, fresh families, and equal-length questions."""
import os
from pathlib import Path
import re
import sys

import clt_b26_conflict_v3 as D
import clt_b26_conflict as B
from source_binding_design import sha

B.OUT=B.HERE/'result/clt_b26_conflict_v4'
B.TMP=B.ROOT/'tests/glm5_temp/clt_b26_conflict_v4'
B.SEED=202611181349
B.shared.OUT=B.OUT
os.environ.update(TEMP=str(B.TMP),TMP=str(B.TMP),TMPDIR=str(B.TMP))
PRIOR=(*D.PRIOR,'clt_b26_conflict_v3')
D.PRIOR=PRIOR


def make_family(tok,split,family,names,words):
    source=B.make_family(tok,split,family,names,words)
    prefix=('The archives below each have a TOP row and a BOTTOM row, in that order. '
            'The scratch note is from a different lookup. Answer the final question with one lowercase code word.\n')
    for row in source:
        old=row['prompt']
        begin=old.index('Two archives'); start=old.index('Archive ',begin); end=old.index('Archive to use:',start)
        title=re.search(r'Archive to use: (\w+)',old)[1]; name=re.search(r'Requested name: (\w+)',old)[1]
        rid=('TOP','BOTTOM')[int(re.search(r'Requested row: (\d+)',old)[1])-1]
        table=old[start:end].replace('Row 1:','TOP row:').replace('Row 2:','BOTTOM row:')
        spans=list(re.finditer(r'(?:TOP|BOTTOM) row: \w+ has code \w+\.',table))
        assert len(spans)==4
        shared=old[:begin]+prefix+table+f'Scratch note: name {name}, {rid} row.\nQuestion: In Archive {title}, give only the code'
        final=f' for {name}' if row['rule']==0 else f' from the {rid} row'
        row['prompt']=shared+final+'.'+old[old.index('<|im_end|>'):]
        enc=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        row['tokens']=enc['input_ids'];p=begin+len(prefix)
        row['records']=[[i for i,(x,y) in enumerate(enc['offset_mapping']) if x<p+m.end() and y>p+m.start()] for m in spans]
        row['endpoint']=max(i for i,(a,b) in enumerate(enc['offset_mapping']) if a<b<=len(shared))
        assert all(row['records']) and len(set(sum(row['records'],[])))==len(sum(row['records'],[]))
    for a,b in zip(source[::2],source[1::2]):
        p=a['endpoint']+1
        if len(a['tokens'])!=len(b['tokens']) or a['endpoint']!=b['endpoint'] or a['tokens'][:p]!=b['tokens'][:p]:return None
    return source


D.make_family=make_family


def freeze():
    original=B.commit

    def bind(name,data):
        if name=='prereg.json':
            data['code'][Path(__file__).relative_to(B.ROOT).as_posix()]=sha(Path(__file__))
            data['previous_failures']['v3']=sha(B.OUT.parent/'clt_b26_conflict_v3/pilot-collection.json')
            data['protocol_change']='Explicit TOP/BOTTOM labels and one lowercase word output. Direct terminal questions share prefix and exact token count, with new families. Original admission thresholds retained.'
        original(name,data)

    B.commit=bind
    try:D.freeze()
    finally:B.commit=original


if __name__=='__main__':
    command=sys.argv[1]
    if command in ('pilot','discovery','confirm'):B.collect(command)
    elif command in ('freeze','select','evaluate'):{'freeze':freeze,'select':B.fit_model,'evaluate':B.evaluate}[command]()
    else:
        import clt_b26_conflict_intervene as causal
        if command=='freeze-intervention':causal.freeze()
        else:causal.run(command)
