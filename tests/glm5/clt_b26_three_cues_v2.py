"""Use explicit table boundaries to make physical first/last queries unambiguous."""
import re
from pathlib import Path
import sys

import clt_b26_three_cues as B
from source_binding_design import read,sha

BASE_FAMILY=B.make_family
B.OUT=B.HERE/'result/clt_b26_three_cues_v2'
B.TMP=B.ROOT/'tests/glm5_temp/clt_b26_three_cues_v2'
B.SEED=202611181616
B.configure()


def family(tok,split,i,names,words,rng):
    rows=BASE_FAMILY(tok,split,i,names,words,rng)
    if rows is None:return None
    for row in rows:
        older=row['prompt'];tab=older.index('Active Archive'); begin=older.index('\n',tab)+1; end=older.index('Question:',begin)
        front=older[:tab].replace('The old note supplies reference keys.', 'The old note is only background. The table is bounded by BEGIN and END lines. Read the CODE on the specified data line.')
        body=front+older[tab:begin]+'BEGIN\n'+older[begin:end]+'END\n'
        stem=body+'Question: In the active archive, find the CODE'
        name,letter=names[0],re.search(r'Old note: NAME \w+; printed ROW ([A-E]);',older)[1]
        probe=(f' in the record named {name}',f' in the record with printed ROW {letter}',
               ' on the data line just below BEGIN' if row['first'] else ' on the data line just above END')[row['rule']]
        row['prompt']=stem+probe+'. Return the code alone, with no punctuation.'+older[older.index('<|im_end|>'):]
        enc=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        spans=list(re.finditer(r'ROW [A-E] \| NAME \w+ \| CODE \w+',body))
        assert len(spans)==5
        by_id={i:spans[j] for j,i in enumerate(row['physical_order'])}
        row['tokens']=enc['input_ids'];row['records']=[[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<by_id[j].end() and b>by_id[j].start()] for j in range(5)]
        row['endpoint']=max(i for i,(a,b) in enumerate(enc['offset_mapping']) if a<b<=len(stem))
        assert all(row['records']) and len(set(sum(row['records'],[])))==len(sum(row['records'],[]))
    for a,b,c in zip(rows[::3],rows[1::3],rows[2::3]):
        cut=a['endpoint']+1
        if not (len(a['tokens'])==len(b['tokens'])==len(c['tokens']) and a['endpoint']==b['endpoint']==c['endpoint'] and a['tokens'][:cut]==b['tokens'][:cut]==c['tokens'][:cut]):return None
    return rows


def freeze():
    old=B.OUT.parent/'clt_b26_three_cues_v1/pilot-collection.json'
    assert not read(old)['passed']
    original=B.common.commit; B.make_family=family
    def bind(name,data):
        if name=='prereg.json':
            data['code'][Path(__file__).relative_to(B.ROOT).as_posix()]=sha(Path(__file__))
            data['previous_pilot']=sha(old)
            data['protocol_change']='Explicit BEGIN and END rows mark table boundaries. Query physical position via the data line just below BEGIN or just above END, retaining independent swaps, length matching and all gates. Fresh disjoint names.'
        original(name,data)
    B.common.commit=bind
    try:B.freeze()
    finally:B.common.commit=original


if __name__=='__main__':
    stage=sys.argv[1]
    if stage in ('pilot','discovery','confirm'):B.collect(stage)
    elif stage in ('freeze','select','evaluate'):{'freeze':freeze,'select':B.select,'evaluate':B.evaluate}[stage]()
    else:
        import clt_b26_three_cues_intervene as interventions
        interventions.freeze() if stage=='freeze-causal' else interventions.run(stage)
