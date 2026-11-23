"""Length-matched direct questions with competing keys in the preceding note."""
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

B.OUT=B.HERE/'result/clt_b26_conflict_v3'
B.TMP=B.ROOT/'tests/glm5_temp/clt_b26_conflict_v3'
B.SEED=202611181228
B.shared.OUT=B.OUT
os.environ.update(TEMP=str(B.TMP),TMP=str(B.TMP),TMPDIR=str(B.TMP))
PRIOR=('conditional_lookup_native_v1','clt_b26_native_v1','clt_b26_compact_v1','clt_b26_conflict_v1','clt_b26_conflict_v2')


def make_family(tok,split,family,names,words):
    source=B.make_family(tok,split,family,names,words)
    prefix='Each archive has two numbered rows. The scratch note is background. Read the final question and answer it using the specified archive.\n'
    for row in source:
        old=row['prompt']
        begin=old.index('Two archives'); start=old.index('Archive ',begin); end=old.index('Archive to use:',start)
        title=re.search(r'Archive to use: (\w+)',old)[1]; name=re.search(r'Requested name: (\w+)',old)[1]; rid=re.search(r'Requested row: (\d+)',old)[1]
        records=list(re.finditer(r'Row [12]: \w+ has code \w+\.',old[start:end]))
        assert len(records)==4
        shared=old[:begin]+prefix+old[start:end]+f'Scratch note: name {name}, row {rid}.\nQuestion: In Archive {title}, what code is listed'
        final=f' for {name}' if row['rule']==0 else f' on row {rid}'
        row['prompt']=shared+final+'?\nReturn only the code word.'+old[old.index('<|im_end|>'):]
        enc=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        row['tokens']=enc['input_ids']
        p=begin+len(prefix)
        row['records']=[[i for i,(x,y) in enumerate(enc['offset_mapping']) if x<p+m.end() and y>p+m.start()] for m in records]
        row['endpoint']=max(i for i,(a,b) in enumerate(enc['offset_mapping']) if a<b<=len(shared))
        assert all(row['records']) and len(set(sum(row['records'],[])))==len(sum(row['records'],[]))
    for a,b in zip(source[::2],source[1::2]):
        p=a['endpoint']+1
        if len(a['tokens'])!=len(b['tokens']) or a['endpoint']!=b['endpoint'] or a['tokens'][:p]!=b['tokens'][:p]:return None
    return source


def freeze():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    checked_dir(B.OUT,230<<20,operation='CLT length-matched campaign');checked_dir(B.TMP)
    prior=B.OUT.parent/'clt_b26_conflict_v1'; base=read(prior/'prereg.json')
    model=MODELS['qwen14']; revision=snapshot(model)
    assert revision==base['provenance']['model_revision']
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    used={n for directory in PRIOR for f in (B.OUT.parent/directory).glob('*-materials.json') for r in read(f) for n in r['names']}
    rng=np.random.default_rng(B.SEED); splits={}
    for split,count in B.SIZES.items():
        rows=[]
        for i in range(count):
            while True:
                names=[''.join(rng.choice(list('bdfgklmnprstvz'))+rng.choice(list('aeiou')) for _ in range(3)).capitalize() for _ in range(2)]
                if len(set(names))!=2 or used.intersection(names):continue
                if any(len(tok.encode(' '+n,add_special_tokens=False))!=3 for n in names):continue
                batch=make_family(tok,split,i,names,list(map(str,rng.choice(WORDS,4,replace=False))))
                if batch is not None:break
            used.update(names);rows.extend(batch)
        file=f'{split}-materials.json'; B.commit(file,rows)
        splits[split]=dict(families=count,inputs=len(rows),sha=sha(B.OUT/file))
        print(f'Frozen {split} length-matched families',flush=True)
    deps={**base['code'],Path(__file__).relative_to(B.ROOT).as_posix():sha(Path(__file__))}
    power=check_phase(.15,128,p_d=.25,alpha=.025);assert power['pass']
    B.commit('prereg.json',dict(utc=datetime.now(timezone.utc).isoformat(),materials=splits,code=deps,power=power,
             previous_failures={'v1':sha(prior/'pilot-collection.json'),'v2':sha(B.OUT.parent/'clt_b26_conflict_v2/pilot-00.json')},
             protocol_change='Both candidate cues appear in a shared scratch note. A direct final question repeats the active cue. Length and prefix token equality in every pair. Behavioral and mechanism gates unchanged. Claims concern this prompt package.',
             provenance=dict(model_path=str(model),model_revision=revision,seed=B.SEED,git_sha=git('rev-parse','HEAD'),git_dirty=True)))


if __name__=='__main__':
    command=sys.argv[1]
    if command in ('pilot','discovery','confirm'):B.collect(command)
    elif command in ('freeze','select','evaluate'):{'freeze':freeze,'select':B.fit_model,'evaluate':B.evaluate}[command]()
    else:
        import clt_b26_conflict_intervene as causal
        if command=='freeze-intervention':causal.freeze()
        else:causal.run(command)
