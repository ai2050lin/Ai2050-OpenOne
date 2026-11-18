"""Separate the answer-slot state from the cue-bearing prefix after block 27."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import source_binding_qkv_followup as exp
import source_binding_transfer_factors as io

OUT = exp.OUT / 'readout'
FAMILIES = 12
CELLS = [('names',23),('qsrc',24),('qent',25)]


def write_design():
    from source_binding_design import sha

    assert not (OUT/'readout-design.json').exists()
    source = json.loads((exp.OUT/'design.json').read_bytes())
    files = {}
    for task in exp.TASKS:
        name = f'holdout-{task}-materials.json'
        assert sha(exp.OUT/name)==source['files'][name]
        rows = json.loads((exp.OUT/name).read_bytes())[:FAMILIES*32]
        assert len({r['family'] for r in rows})==FAMILIES
        io.save(f'{task}-materials.json',rows)
        files[task] = sha(OUT/f'{task}-materials.json')
    io.save('readout-design.json',dict(created=datetime.now().astimezone().isoformat(), source_design=sha(exp.OUT/'design.json'),
        code=sha(__file__), materials=files, model_snapshot=source['snapshot'], cells=CELLS, families=FAMILIES,
        primary='At the input of block 28, cross same complete final-token baseline or edited hidden state with opposite cue prefix. '
        'Compare edit-induced new-minus-old margin holding the final-token baseline and edited states both fixed. Earlier token slots differ only at the cue word. '
        'Separately compare the same edit delta on the opposite final-token baseline, as in main experiment.'))


def run():
    import torch
    from deploy.storage_policy import check_write
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    import source_binding_design as d
    import source_binding_sites as sites
    import source_binding_qkv_followup_engine as eng

    design = io.read('readout-design.json')
    assert design['code']==d.sha(__file__) and design['source_design']==d.sha(exp.OUT/'design.json')
    assert snapshot(d.MODELS['qwen14'])==design['model_snapshot']
    native_kinds,native_window,native_names = eng.KINDS,eng.WINDOW,eng.NAMES
    sites.OUT = OUT
    import time
    manifest = []
    for task in exp.TASKS:
        assert d.sha(OUT/f'{task}-materials.json')==design['materials'][task]
        rows = io.read(f'{task}-materials.json')
        chunks = [rows[i:i+exp.PANEL_ROWS] for i in range(0,len(rows),exp.PANEL_ROWS)]
        for j in range(0,len(chunks),2):
            fname = f'{task}-{j//2:02d}.json'
            if (OUT/fname).exists():
                manifest.append(dict(file=fname,sha256=d.sha(OUT/fname)))
                continue
            started = time.monotonic()
            with GpuLease(),torch.inference_mode():
                rt = eng.Runtime('qwen14')
                batch_rows = chunks[j:j+2]
                batches = [eng.make_batch(rt,b) for b in batch_rows]
                total = 40*sum(map(len,batch_rows))
                stage = f'readout-{task}-{j+1}/{len(chunks)}'
                sites.log(stage,0,total,started)
                for block in range(40):
                    layer = rt.block(block)
                    for batch in batches:
                        eng.KINDS,eng.WINDOW,eng.NAMES = ('W','H'), ((block,) if block in native_window else ()), tuple(s for s,b in CELLS if b==block)
                        if block==28:
                            original = batch['h'][:,-1:].clone()
                            cross = batch['cross'].cpu()
                            batch['states']['Xbase'] = original[cross]
                            for site,b in CELLS:
                                cell=f'{site}@{b}'
                                edited=batch['states']['H|'+cell]
                                batch['states']['X|'+cell]=edited[cross]
                                delta=edited.float()-original.float()
                                batch['states']['T|'+cell]=(original.float()+delta[cross]).to(original.dtype)
                        eng.layer_step(rt,layer,block,batch)
                    del layer
                    if (block+1)%2==0:
                        check_write(OUT,1<<28,operation=stage)
                        sites.log(stage,(block+1)*total//40,total,started,block=block,arms=len(batches[0]['states']))
                eng.KINDS,eng.WINDOW,eng.NAMES=native_kinds,native_window,native_names
                norm = rt.Norm(rt.dim,eps=rt.cfg.rms_norm_eps).to(device='cuda',dtype=torch.bfloat16)
                norm.weight = torch.nn.Parameter(rt.weight('model.norm.weight'),requires_grad=False)
                weight = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
                output=[]
                for batch in batches:
                    selected = {k:h for k,h in batch['states'].items() if k.split('|')[0] in ('W','H','X','Xbase','T','noop','self')}
                    items = {k:eng.select_scores((norm(h.cuda())@weight.T).float(),batch['rows']) for k,h in {'base':batch['h'],**selected}.items()}
                    assert all(r['pred_record']==u['pred_record'] for k in ('noop','self') for r,u in zip(items[k],items['base']))
                    assert all(max(abs(a-b) for a,b in zip(r['scores'],u['scores']))<=1 for k in ('noop','self') for r,u in zip(items[k],items['base']))
                    assert max(max(k,v) for _,k,v in batch['name_invariance'])==0
                    output.append(items)
            io.save(fname,output)
            manifest.append(dict(file=fname,sha256=d.sha(OUT/fname)))
    io.save('readout-manifest.json',dict(shards=manifest,passed=True))


def analysis():
    from collections import defaultdict
    import numpy as np
    from source_binding_design import sha

    design = io.read('readout-design.json')
    manifest = io.read('readout-manifest.json')
    assert manifest['passed']
    saved = defaultdict(dict)
    for entry in manifest['shards']:
        f=entry['file']
        assert sha(OUT/f)==entry['sha256']
        for items in io.read(f):
            for i,r in enumerate(items['base']):
                saved[f.split('-')[0]][r['id']]={k:rows[i] for k,rows in items.items()}
    rng=np.random.default_rng(exp.SEED)
    out={}
    for task in exp.TASKS:
        assert sha(OUT/f'{task}-materials.json')==design['materials'][task]
        rows = io.read(f'{task}-materials.json')
        meta = {r['original_id']:r for r in rows if r['cue']==0}
        by_family = defaultdict(list)
        for id_,r in meta.items():
            a,b=[saved[task][f'{id_}-c{i}'] for i in (0,1)]
            old=a['base']['pred_record']
            if old!=b['base']['pred_record'] or old!=r['target_record']:
                continue
            for site,block in CELLS:
                c=f'{site}@{block}'
                new=a['W|'+c]['pred_record']
                if new==old or new!=b['W|'+c]['pred_record']:
                    continue
                def margin(row):
                    return row['scores'][new]-row['scores'][old]
                ha = margin(a['H|'+c])-margin(a['base'])
                hb = margin(b['H|'+c])-margin(b['base'])
                xa = margin(b['X|'+c])-margin(b['Xbase'])
                xb = margin(a['X|'+c])-margin(a['Xbase'])
                da = margin(b['T|'+c])-margin(b['base'])
                db = margin(a['T|'+c])-margin(a['base'])
                by_family[r['family']].append([hb-ha,xa-ha,hb-xb,da-ha,hb-db])
        ids=sorted({r['family'] for r in rows})
        counts=np.array([len(by_family[f]) for f in ids]); sums=np.array([np.asarray(by_family[f]).sum(0) if by_family[f] else np.zeros(5) for f in ids])
        draws=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),8000)
        n=draws@counts
        estimates=(draws@sums)[n>0]/n[n>0,None]
        stats={k:dict(mean=float(v),ci95=np.quantile(estimates[:,i],[.025,.975]).tolist())
               for i,(k,v) in enumerate(zip(('natural_H_gap','prefix_only_A','prefix_only_Z','fixed_delta_A','fixed_delta_Z'),sums.sum(0)/counts.sum()))}
        out[task]=dict(n=int(counts.sum()),families=len(ids),stats=stats)
    io.save('readout-report.json',dict(tasks=out,design_sha256=sha(OUT/'readout-design.json'),manifest_sha256=sha(OUT/'readout-manifest.json')))
    print(out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('design','run','analysis'))
    mode=parser.parse_args().mode
    io.OUT=OUT
    io.setup()
    {'design':write_design,'run':run,'analysis':analysis}[mode]()


if __name__=='__main__':
    main()
