"""Test which non-name token regions can supplement H16 in the original Answer: task."""
from collections import defaultdict
from datetime import datetime
import argparse
import itertools
import json
from pathlib import Path

import source_binding_qkv_followup as exp
import source_binding_transfer_factors as io

OUT = exp.OUT/'suffix'
CELLS = tuple(itertools.product(exp.SITES,exp.LAYERS))
FAMILIES = 12
KINDS = ('between','after','question','tail','all_suffix')


def save_design():
    from source_binding_design import sha

    assert not (OUT/'design.json').exists()
    base=json.loads((exp.OUT/'design.json').read_bytes())
    files={}
    for task in exp.TASKS:
        name=f'holdout-{task}-materials.json'
        assert sha(exp.OUT/name)==base['files'][name]
        rows=[r for r in json.loads((exp.OUT/name).read_bytes())[:FAMILIES*32] if r['cue']==0]
        assert len(rows)==16*FAMILIES
        io.save(f'{task}-materials.json',rows)
        files[task]=sha(OUT/f'{task}-materials.json')
    io.save('design.json',dict(created=datetime.now().astimezone().isoformat(),code=sha(__file__),snapshot=base['snapshot'],
                               files=files,main_design_sha256=sha(exp.OUT/'design.json'), families=FAMILIES, cells=CELLS,
                               analysis='One added holder region per arm. Target is main wide choice for same original prompt, donor and block. Compare against same-run H.'))


def question_end(row, tok):
    intro = row['prompt'].index(row['content'])
    end = intro+len(row['content'])-len('\nReply with the colour only.' if row['task']=='colours' else '\n只回答动物名称，不要解释。')
    enc = tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
    assert enc['input_ids']==row['tokens'] and end>intro
    return next(i for i,(a,_) in enumerate(enc['offset_mapping']) if a>=end)


def run():
    import source_binding_transfer_factor_engine as engine
    import source_binding_sites as sites
    from source_binding_readheads import spec_heads,H16
    from source_binding_relay import tokenizer
    from source_binding_design import sha

    tok=tokenizer()
    frozen=io.read('design.json')
    assert frozen['code']==sha(__file__) and frozen['main_design_sha256']==sha(exp.OUT/'design.json')
    frozen_main=json.loads((exp.OUT/'design.json').read_bytes())
    assert frozen_main['snapshot']==frozen['snapshot']
    bounds={}
    for task in exp.TASKS:
        assert sha(OUT/f'{task}-materials.json')==frozen['files'][task]
        for r in io.read(f'{task}-materials.json'):
            names=sorted(r['up']['qsrc']+r['up']['qent'])
            stop=r['tokens'].index(tok.convert_tokens_to_ids('<|im_end|>'))
            end=question_end(r,tok)
            lower,upper=sorted([r['up']['qsrc'],r['up']['qent']])
            bounds[r['id']]=dict(between=list(range(lower[-1]+1,upper[0])),after=list(range(names[-1]+1,end)),
                                 tail=list(range(end,stop)),question=[i for i in range(names[0],end) if i not in names],
                                 all_suffix=[i for i in range(names[0],stop) if i not in names])
    for kind in KINDS:
        sites.GROUPS[kind] = lambda r,k=kind: bounds[r['id']][k]

    def arms(rows):
        idx={r['original_id']:i for i,r in enumerate(rows)}
        donors=[idx[r['donor']] for r in rows]
        a={'self':dict(donor=list(range(len(rows))),don=spec_heads('names',23,H16))}
        for s,b in CELLS:
            spec=spec_heads(s,b,H16)
            for region in KINDS:
                for row,j in zip(rows,donors):
                    assert len(bounds[row['id']][region])==len(bounds[rows[j]['id']][region])>0, (row['id'],region)
            cell=f'{s}@{b}'
            a['H|'+cell]=dict(donor=donors,don=spec)
            for region in KINDS:
                a[region+'|'+cell]=dict(donor=donors,don={l:[[[s,region],hs]] for l,((_,hs),) in spec.items()})
        return a

    manifest=[]
    for task in exp.TASKS:
        rows=io.read(f'{task}-materials.json')
        chunks=[rows[i:i+32] for i in range(0,len(rows),32)]
        for start in range(0,len(chunks),2):
            name=f'{task}-{start//2:02d}.json'
            if not (OUT/name).exists():
                result,meta=engine.run(chunks[start:start+2],engine.S.Runtime,arms,f'suffix-{task}-{start+1}/{len(chunks)}',OUT,
                                       lambda:io.check_write(OUT,1<<28,operation='qkv-suffix'))
                for block in result:
                    assert all(b['pred_record']==s['pred_record'] and max(abs(x-y) for x,y in zip(b['scores'],s['scores']))<=1
                               for k in ('self','noop') for b,s in zip(block['base'],block[k]))
                io.save(name,dict(meta=meta,items=result))
            manifest.append(dict(name=name,sha256=sha(OUT/name)))
    io.save('manifest.json',dict(passed=True,shards=manifest))


def analysis():
    import numpy as np
    from source_binding_design import sha

    source=json.loads((exp.OUT/'holdout-manifest.json').read_bytes())
    assert source['passed']
    old={}
    for item in source['shards']:
        path=exp.OUT/item['file']
        assert sha(path)==item['sha256']
        for block in json.loads(path.read_bytes())['groups']:
            for i,id_ in enumerate(block['ids']):
                if id_.endswith('-c0'):
                    old[id_]={k:v[i][0] for k,v in block['items'].items() if k=='base' or k.startswith(('W|','H|','Hback|'))}
    mats={r['id']:r for task in exp.TASKS for r in io.read(f'{task}-materials.json')}
    counters=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:[0,0,defaultdict(lambda:[0,0])])))
    replays=defaultdict(lambda:[0,0,0])
    for shard in io.read('manifest.json')['shards']:
        assert sha(OUT/shard['name'])==shard['sha256']
        for block in io.read(shard['name'])['items']:
            for i,base in enumerate(block['base']):
                r=mats[base['id']]; ref=old[base['id']]
                assert base['pred_record']==ref['base']
                for site,b in CELLS:
                    cell=f'{site}@{b}'
                    if ref['W|'+cell]==ref['base']:
                        continue
                    for kind in ('H',*KINDS):
                        hit=block[kind+'|'+cell][i]['pred_record']==ref['W|'+cell]
                        for scope in ('all',site,cell):
                            data=counters[r['task']][scope][kind]
                            data[0]+=1
                            data[1]+=hit
                            data[2][r['family']][0]+=1
                            data[2][r['family']][1]+=hit
                    replay=replays[r['task']];replay[0]+=1
                    replay[1]+=block['H|'+cell][i]['pred_record']!=ref['H|'+cell]
                    replay[2]+=block['all_suffix|'+cell][i]['pred_record']!=ref['Hback|'+cell]
    rng=np.random.default_rng(exp.SEED)
    for task,scopes in counters.items():
        for scope,kinds in scopes.items():
            ref=kinds['H'][2]
            families=sorted(ref)
            draws=rng.multinomial(len(families),np.full(len(families),1/len(families)),8000)
            n=draws@np.array([ref[f][0] for f in families]); valid=n>0
            for k,(cnt,hit,fam) in list(kinds.items()):
                dif=np.array([fam[f][1]-ref[f][1] for f in families]); bs=(draws@dif)[valid]/n[valid]
                kinds[k]=dict(n=cnt,hit=hit,rate=hit/cnt,delta=float(dif.sum()/cnt),ci95=np.quantile(bs,[.025,.975]).tolist())
    io.save('report.json',dict(results=counters,comparison_replay=replays, design_sha256=sha(OUT/'design.json'),
                               main_manifest_sha256=sha(exp.OUT/'holdout-manifest.json'),manifest_sha256=sha(OUT/'manifest.json')))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('design','run','analysis'))
    mode=ap.parse_args().mode
    io.OUT=OUT
    io.setup()
    {'design':save_design,'run':run,'analysis':analysis}[mode]()


if __name__=='__main__':
    main()
