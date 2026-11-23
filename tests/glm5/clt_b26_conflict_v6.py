"""Keep the active archive adjacent to the final question, with balanced titles."""
import os
from pathlib import Path
import re
import sys

import clt_b26_conflict_v5 as V5
import clt_b26_conflict_v3 as D
import clt_b26_conflict as B
from source_binding_design import read,sha

B.OUT=B.HERE/'result/clt_b26_conflict_v6'
B.TMP=B.ROOT/'tests/glm5_temp/clt_b26_conflict_v6'
B.SEED=202611181445
B.SIZES['pilot']=32
B.shared.OUT=B.OUT
os.environ.update(TEMP=str(B.TMP),TMP=str(B.TMP),TMPDIR=str(B.TMP))
PRIOR=(*V5.PRIOR,'clt_b26_conflict_v5')
D.PRIOR=PRIOR


def make_family(tok,split,family,names,words):
    rows=B.make_family(tok,split,family,names,words)
    intro=('Each table has a TOP row and a BOTTOM row. Use the active table for the question. '
           'Reply with a single lowercase code word.\n')
    for row in rows:
        old=row['prompt']; begin=old.index('Two archives'); start=old.index('Archive ',begin); end=old.index('Archive to use:',start)
        a=re.search(r'Archive to use: (\w+)',old)[1]; name=re.search(r'Requested name: (\w+)',old)[1]
        rid=('TOP','BOTTOM')[int(re.search(r'Requested row: (\d+)',old)[1])-1]
        original=old[start:end].splitlines(keepends=True)
        order=[0,1] if row['name_slot']//2==1 else [1,0]
        header=old[:begin]+intro+f'Old lookup notes (unrelated to the question): name {name}, {rid} row.\n'
        spans={}; text=header
        for b in order:
            text+=('Active table, ' if b==row['name_slot']//2 else 'Other table, ')+original[3*b]
            for line in range(2):
                record=original[3*b+1+line].replace('Row 1:','TOP row:').replace('Row 2:','BOTTOM row:')
                spans[2*b+line]=len(text),len(text)+len(record.rstrip('\n'))
                text+=record
        shared=text+f'Question: In Archive {a}, what is the code'
        tail=f' for {name}' if row['rule']==0 else f' from the {rid} row'
        row['prompt']=shared+tail+'?'+old[old.index('<|im_end|>'):]
        e=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        row['tokens']=e['input_ids'];row['records']=[[i for i,(x,y) in enumerate(e['offset_mapping']) if x<b and y>a] for a,b in [spans[j] for j in range(4)]]
        row['endpoint']=max(i for i,(x,y) in enumerate(e['offset_mapping']) if x<y<=len(shared))
        assert all(row['records']) and len(set(sum(row['records'],[])))==len(sum(row['records'],[]))
    for a,b in zip(rows[::2],rows[1::2]):
        p=a['endpoint']+1
        if len(a['tokens'])!=len(b['tokens']) or a['endpoint']!=b['endpoint'] or a['tokens'][:p]!=b['tokens'][:p]:return None
    return rows


D.make_family=make_family


def freeze():
    earlier=B.OUT.parent/'clt_b26_conflict_v5/pilot-collection.json'
    assert not read(earlier)['passed']
    original=B.commit

    def bind(name,data):
        if name=='prereg.json':
            data['code'][Path(__file__).relative_to(B.ROOT).as_posix()]=sha(Path(__file__))
            data['previous_failures']['v5']=sha(earlier)
            data['protocol_change']='Both cue hints occur earlier as old notes. The queried archive is labeled active and printed nearest the direct terminal question. The logical archive number is balanced but is no longer an intervention on physical ordering. Within-pair exact length and common prefix, original thresholds, 32 fresh pilot families.'
        original(name,data)

    B.commit=bind
    try:V5.freeze()
    finally:B.commit=original


if __name__=='__main__':
    command=sys.argv[1]
    if command in ('pilot','discovery','confirm'):B.collect(command)
    elif command in ('freeze','select','evaluate'):{'freeze':freeze,'select':B.fit_model,'evaluate':B.evaluate}[command]()
    else:
        import clt_b26_conflict_intervene as causal
        if command=='freeze-intervention':causal.freeze()
        else:causal.run(command)
