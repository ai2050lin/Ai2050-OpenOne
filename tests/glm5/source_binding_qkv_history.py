"""Check full Phase 2921 Answer: panel against the QKV follow-up gap decomposition."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import source_binding_qkv_followup as exp
import source_binding_transfer_cue as cue
import source_binding_transfer_factors as io
from source_binding_design import sha
from source_binding_fidelity2 import predict
from source_binding_qkv_followup_analysis import gap


def main():
    io.OUT = exp.OUT
    root=cue.OUT
    manifest=json.loads((root/'items.json').read_bytes())
    assert manifest['passed']
    rows=json.loads((root/'materials.json').read_bytes())
    design=json.loads((root/'design.json').read_bytes())
    assert sha(root/'materials.json')==design['materials_sha256']
    metas={r['id']:r for r in rows if r['cue']=='00'}
    lookup={r['original_id']:r for r in metas.values()}
    stats=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:[0,0])))
    position=defaultdict(lambda:defaultdict(lambda:[0,0]))
    for entry in manifest['shards']:
        p=root/entry['name']
        assert sha(p)==entry['sha256']
        for block in json.loads(p.read_bytes())['items']:
            for i,base in enumerate(block['base']):
                if base['id'] not in metas:
                    continue
                r=metas.pop(base['id'])
                for k,v in block.items():
                    if not k.startswith('swap|'):
                        continue
                    y=v[i]['pred_record']
                    if y==base['pred_record']:
                        continue
                    site,desc=k[5:].split('@')
                    layer,rep=map(int,desc.split('#'))
                    donor=lookup[r['original_donors'][rep]]
                    wanted=2*(donor['source'] if site!='qent' else r['source'])+(donor['entity'] if site!='qsrc' else r['entity'])
                    key=f'{site}@{layer}/'+('donor' if y==wanted else 'other')
                    hit=block['H16|'+k[5:]][i]['pred_record']==y
                    c=stats[r['task']][r['family']][key]
                    c[0]+=1
                    c[1]+=hit
                    c=position[r['task']][str(y==predict(r,donor,site,layer,'position'))]
                    c[0]+=1
                    c[1]+=hit
    assert not metas
    keys=[f'{s}@{l}/{k}' for s in exp.SITES for l in exp.LAYERS for k in ('donor','other')]
    matrix={t:dict(answer_only_strata_family_counts={k:[stats[t][f][k] for f in sorted(stats[t])] for k in keys}) for t in exp.TASKS}
    io.save('phase2921-gap-review.json',dict(gap=gap(matrix),counts=matrix,position=position,
        source_manifest_sha256=sha(root/'items.json'),analysis_sha256=sha(__file__),
        sample='Phase 2921 Answer: panel, all four donor draws; independent families relative to follow-up'))


if __name__=='__main__':
    main()
