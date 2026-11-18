"""Independent readback of retained, updated and misselected record outputs."""

from collections import Counter, defaultdict
import sys

from transformers import AutoTokenizer

import record_update_material as M
import record_update_run as R


def run(version):
    if version=='2':
        from record_update_v2 import configure
        configure()
    elif version=='3':
        from record_update_v3 import configure
        configure()
    elif version!='1':
        raise ValueError('Use 1, 2 or 3')
    p=R.verified(); raw=R.read('pilot-behavior.json')
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    rows=[M.encode(tok,row) for row in M.families(tok,'pilot')]
    assert M.digest(rows)==raw['material']==p['material']['pilot']['sha256']
    assert raw['plan']==R.C.A.digest(R.OUT/'plan.json')
    assert not raw['gate']['passed']
    groups=defaultdict(Counter)
    examples=[]
    for row,item in zip(rows,raw['records'],strict=True):
        labels=row['names']; steps=[i for i,x in enumerate(row['actions']) if x==0]
        expected=labels[2+steps[-1]] if steps else labels[row['active']]
        assert expected==row['truth'][-1] and row['id']==item['id']
        table=dict(zip(labels,row['codes'],strict=True))
        counts=groups[str(len(steps))]
        for key in ('joint','state_ok','code_ok','format','stop'):counts[key]+=item[key]
        counts['views']+=1
        actual=item['state']
        if actual in table:
            counts['recognized_label']+=1
            counts['code_follows_emitted']+=table[actual]==item['code']
        if actual!=expected:
            kind=('target_initial' if actual==labels[row['active']] else
                  'other_initial' if actual==labels[1-row['active']] else
                  'notice1' if actual==labels[2] else 'notice2' if actual==labels[3] else 'outside_labels')
            counts['wrong_'+kind]+=1
            if len(examples)<5:
                examples.append(dict(id=item['id'],frame=row['frame'],actions=row['actions'],
                                     expected=expected,expected_code=table[expected],actual=actual,
                                     actual_code=item['code'],actual_label_code=table.get(actual),kind=kind))
    for name in ('discovery','words','frames','combos','lockbox'):
        assert not (R.OUT/(name+'-behavior.json')).exists()
    digest=R.write('behavior-audit.json',dict(at=R.C.stamp(),run=p['run'],provenance=p['provenance'],groups=groups,examples=examples,
           links={name:R.C.A.digest(R.OUT/name) for name in ('plan.json','pilot-behavior.json','progress.jsonl')}))
    print(version,{k:dict(v) for k,v in groups.items()},digest,flush=True)


if __name__=='__main__':
    run(sys.argv[1])
