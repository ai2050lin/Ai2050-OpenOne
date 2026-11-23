"""Use a direct final instruction after the first conflict pilot failed."""
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys

import numpy as np
import clt_b26_conflict as B
from deploy.storage_policy import checked_dir
from source_binding_design import MODELS, WORDS, git, read, sha
from tools.power_analysis import check_phase

B.OUT = B.HERE/'result/clt_b26_conflict_v2'
B.TMP = B.ROOT/'tests/glm5_temp/clt_b26_conflict_v2'
B.SEED = 202611181107
B.shared.OUT = B.OUT
os.environ.update(TEMP=str(B.TMP),TMP=str(B.TMP),TMPDIR=str(B.TMP))


def make_family(tok, split, i, names, words):
    rows=B.make_family(tok,split,i,names,words)
    for r in rows:
        left,old=r['prompt'].split('Lookup key to use:',1)
        title=re.search(r'Archive to use: (\w+)',left)[1]
        name=re.search(r'Requested name: (\w+)',left)[1]
        line=re.search(r'Requested row: (\d+)',left)[1]
        if not r['rule']:
            instruction=f'Final instruction: Ignore the requested row number. In Archive {title}, find the record named {name} and return only its code.'
        else:
            instruction=f'Final instruction: Ignore the requested name. In Archive {title}, find the record at row {line} and return only its code.'
        suffix=old[old.index('<|im_end|>'):]
        old_tokens=r['tokens']; old_max=max(x for sites in r['records'] for x in sites)
        r['prompt']=left+instruction+suffix
        r['tokens']=tok.encode(r['prompt'],add_special_tokens=False)
        assert r['tokens'][:old_max+1]==old_tokens[:old_max+1]
        r['endpoint']=min(r['endpoint'],len(tok.encode(left.rstrip(),add_special_tokens=False))-1)
    for a,b in zip(rows[::2],rows[1::2]):
        p=a['endpoint']+1
        assert a['endpoint']==b['endpoint'] and a['tokens'][:p]==b['tokens'][:p]
        assert a['name_slot']==b['name_slot'] and a['row_slot']==b['row_slot']
    return rows


def freeze():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    checked_dir(B.OUT,240<<20,operation='CLT explicit conflict campaign')
    checked_dir(B.TMP)
    prior=B.OUT.parent/'clt_b26_conflict_v1'; rejected=read(prior/'pilot-collection.json')
    assert not rejected['passed']
    model=MODELS['qwen14']; revision=snapshot(model)
    assert revision==rejected['provenance']['model_revision']
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    used={n for folder in ('conditional_lookup_native_v1','clt_b26_native_v1','clt_b26_compact_v1','clt_b26_conflict_v1')
          for f in (B.OUT.parent/folder).glob('*-materials.json') for row in read(f) for n in row['names']}
    rng=np.random.default_rng(B.SEED); splits={}
    for split,count in B.SIZES.items():
        rows=[]
        for i in range(count):
            while True:
                names=[''.join(rng.choice(list('bdfgklmnprstvz'))+rng.choice(list('aeiou')) for _ in range(3)).capitalize() for _ in range(2)]
                if len(set(names))==2 and not used.intersection(names) and len({len(tok.encode(' '+n,add_special_tokens=False)) for n in names})==1:
                    break
            used.update(names)
            rows.extend(make_family(tok,split,i,names,list(map(str,rng.choice(WORDS,4,replace=False)))))
        file=f'{split}-materials.json'; B.commit(file,rows)
        splits[split]=dict(families=count,inputs=len(rows),sha=sha(B.OUT/file))
    base=read(prior/'prereg.json')
    sources={**base['code'],Path(__file__).relative_to(B.ROOT).as_posix():sha(Path(__file__))}
    power=check_phase(.15,128,p_d=.25,alpha=.025); assert power['pass']
    B.commit('prereg.json',dict(utc=datetime.now(timezone.utc).isoformat(),materials=splits,code=sources,power=power,
             pilot_failure_sha=sha(prior/'pilot-collection.json'),
             protocol_change='Direct final instruction with explicit ignore sentence and active cue repeated. Conclusions concern this instruction package. Rest of CLT conflict design and gates unchanged.',
             provenance=dict(model_path=str(model),model_revision=revision,seed=B.SEED,git_sha=git('rev-parse','HEAD'),git_dirty=True)))
    print('Frozen new explicit-instruction families',flush=True)


if __name__=='__main__':
    command=sys.argv[1]
    if command in ('pilot','discovery','confirm'):B.collect(command)
    elif command in ('freeze','select','evaluate'):{'freeze':freeze,'select':B.fit_model,'evaluate':B.evaluate}[command]()
    else:
        import clt_b26_conflict_intervene as causal
        if command=='freeze-intervention':causal.freeze()
        else:causal.run(command)
