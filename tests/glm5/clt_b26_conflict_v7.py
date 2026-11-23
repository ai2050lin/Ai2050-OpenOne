"""Controlled conflict between printed ROW and NAME fields in one active table."""
import os
from pathlib import Path
import re
import sys

import clt_b26_conflict_v6 as V6
import clt_b26_conflict_v3 as D
import clt_b26_conflict as B
from source_binding_design import read,sha

B.OUT=B.HERE/'result/clt_b26_conflict_v7'
B.TMP=B.ROOT/'tests/glm5_temp/clt_b26_conflict_v7'
B.SEED=202611181459
B.shared.OUT=B.OUT
os.environ.update(TEMP=str(B.TMP),TMP=str(B.TMP),TMPDIR=str(B.TMP))
PRIOR=(*V6.PRIOR,'clt_b26_conflict_v6');D.PRIOR=PRIOR


def make_family(tok,split,family,names,words):
    rows=V6.make_family(tok,split,family,names,words)
    if rows is None:return None
    for row in rows:
        old=row['prompt']; note=re.search(r'Old lookup notes[^\n]+: name (\w+), (TOP|BOTTOM) row\.',old)
        title=re.search(r'Active table, Archive (\w+):',old)[1]
        before=old.split('Question:',1)[0]
        before=re.sub(r'(TOP|BOTTOM) row: (\w+) has code (\w+)\.',r'ROW \1 | NAME \2 | CODE \3',before)
        stem=before+f'In the active Archive {title}, read the CODE by matching'
        tail=f' NAME {note[1]}' if row['rule']==0 else f' the printed ROW {note[2]}'
        row['prompt']=stem+tail+'. Return one lowercase word, without punctuation.'+old[old.index('<|im_end|>'):]
        enc=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        row['tokens']=enc['input_ids']
        found=list(re.finditer(r'ROW (?:TOP|BOTTOM) \| NAME \w+ \| CODE \w+',before));assert len(found)==4
        if row['name_slot']//2==0:found=found[2:]+found[:2]
        row['records']=[[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<m.end() and b>m.start()] for m in found]
        row['endpoint']=max(i for i,(a,b) in enumerate(enc['offset_mapping']) if a<b<=len(stem))
        assert all(row['records']) and len(set(sum(row['records'],[])))==len(sum(row['records'],[]))
    for a,b in zip(rows[::2],rows[1::2]):
        endpoint=a['endpoint']+1
        if a['endpoint']!=b['endpoint'] or len(a['tokens'])!=len(b['tokens']) or a['tokens'][:endpoint]!=b['tokens'][:endpoint]:return None
    return rows


D.make_family=make_family


def freeze():
    earlier=B.OUT.parent/'clt_b26_conflict_v6/pilot-collection.json'
    assert not read(earlier)['passed']
    original=B.commit

    def bind(name,data):
        if name=='prereg.json':
            data['code'][Path(__file__).relative_to(B.ROOT).as_posix()]=sha(Path(__file__))
            data['previous_failures']['v6']=sha(earlier)
            data['protocol_change']='Typed ROW and NAME conflict inside a table with explicit CODE field. Old notes before tables and active archive last. Labels coincide with physical position, so row label versus ordinal is not separately identified. Final instruction matches one printed field. New length-matched families, original admission rules.'
        original(name,data)

    B.commit=bind
    try:V6.freeze()
    finally:B.commit=original


if __name__=='__main__':
    command=sys.argv[1]
    if command in ('pilot','discovery','confirm'):B.collect(command)
    elif command in ('freeze','select','evaluate'):{'freeze':freeze,'select':B.fit_model,'evaluate':B.evaluate}[command]()
    else:
        import clt_b26_conflict_intervene as causal
        if command=='freeze-intervention':causal.freeze()
        else:causal.run(command)
