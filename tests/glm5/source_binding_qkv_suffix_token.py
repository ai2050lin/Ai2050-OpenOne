"""Locate single-token H16 recovery and test natural reading with mask and restoration."""
from collections import defaultdict
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path

import source_binding_qkv_suffix_sites as prior
import source_binding_transfer_factors as io
from source_binding_design import sha
from source_binding_readheads import H16


OUT = prior.exp.OUT / 'suffix-token'
CELLS = prior.CELLS
SEED = 202611130430
CHUNK = 32
N_FAMILIES = 12


def original_targets():
    root = prior.exp.OUT
    manifest = json.loads((root/'holdout-manifest.json').read_bytes())
    assert manifest['passed']
    targets = {}
    for item in manifest['shards']:
        if not item['file'].startswith('holdout-zh-'):
            continue
        path = root/item['file']
        assert sha(path) == item['sha256']
        for group in json.loads(path.read_bytes())['groups']:
            for i,id_ in enumerate(group['ids']):
                if id_.endswith('-c0'):
                    targets[id_] = {k:v[i][0] for k,v in group['items'].items() if k=='base' or k.startswith('W|')}
    return targets


def design():
    from source_binding_relay import tokenizer

    assert not (OUT/'design.json').exists()
    root = prior.exp.OUT
    source = json.loads((root/'design.json').read_bytes())
    name = 'holdout-zh-materials.json'
    assert sha(root/name) == source['files'][name]
    rows = [r for r in json.loads((root/name).read_bytes()) if r['cue']==0]
    assert len(rows)==2*N_FAMILIES*16
    targets = original_targets()
    tok = tokenizer()
    for row in rows:
        assert row['prompt'].endswith('Answer:\n')
        start = max(row['up']['qsrc']+row['up']['qent'])+1
        end = prior.question_end(row,tok)
        suffix = list(range(start,end))
        parts = [tok.decode([row['tokens'][i]]) for i in suffix]
        anchors = [j for j,x in enumerate(parts) if x=='动物']
        assert len(anchors)==1 and 5<=len(parts)<=7
        row['suffix'] = suffix
        row['parts'] = parts
        row['animal_offset'] = anchors[0]
        row['wide'] = targets[row['id']]
    first,second = rows[:N_FAMILIES*16],rows[N_FAMILIES*16:]
    assert {r['family'] for r in first}.isdisjoint(r['family'] for r in second)
    files={}
    for split,material in (('discover',first),('confirm',second)):
        file=f'{split}-materials.json'
        io.save(file,material)
        files[file]=sha(OUT/file)
    scripts = ('source_binding_qkv_suffix_token.py','source_binding_qkv_suffix_token_engine.py','source_binding_qkv_suffix_token_analysis.py',
               'source_binding_transfer_factor_engine.py','source_binding_runtime.py','source_binding_readheads.py')
    io.save('design.json',dict(time=datetime.now().astimezone().isoformat(),seed=SEED,snapshot=source['snapshot'],files=files,
        source_sha256=sha(root/'holdout-manifest.json'),heads=H16,cells=CELLS,code={s:sha(Path(__file__).with_name(s)) for s in scripts},
        panels='Two 12-family panels from the original Phase 2922 material. First is the earlier suffix region panel. Second is disjoint by family, both already exposed to other experiments. One donor per prompt reused.',
        scan='Nine site x start-block settings. H = names-only K/V donor edit. Add one suffix token per arm; compare on the same frozen wide-changed targets. Slot absent in shorter frames is never counted. Whole suffix positive control.',
        selection='Within each template select single highest paired hit gain in discovery, tie by candidate-margin gain then earliest index. Compare frozen location against all remaining single slots and full region in confirmation. Also report fixed lexical animal anchor and slots from end.',
        natural='Original prompt, no name patch. For each frozen and other suffix slot, block direct H16 attention from the answer position at layers 23–27. Compare best matched control by cyclic next slot. Also mask entire suffix, reopen frozen slot or matched slot while others stay blocked. Independent value zero/restore diagnostic leaves attention normalization fixed. Behavior accuracy, next-token top1 and four-choice logit margins; paired family CIs.',
        gates='Original accuracy >=95%; no-op and self-swap choices equal and max candidate score difference <=1; exact masked-value restoration compared with self replay; report all strata. No formal evidence promotion.'))


def headmap(start, heads=H16):
    return {b:h for b,h in heads.items() if b>=start}


def arms_for(rows, stage):
    idx = {r['original_id']:i for i,r in enumerate(rows)}
    own = list(range(len(rows)))
    for r in rows:
        r['groups'] = {site:sorted(r['up'][site]) for site in ('qsrc','qent')}
        r['groups']['names'] = sorted(r['up']['qsrc']+r['up']['qent'])
        r['groups']['full'] = r['suffix']
        for j in range(7):
            r['groups'][f't{j}'] = r['suffix'][j:j+1]
    arms = {'self':dict(donor=own,heads=H16,actions=[('kv','names')])}
    if stage=='scan':
        donors = [idx[r['donor']] for r in rows]
        for i,j in enumerate(donors):
            assert rows[i]['parts']==rows[j]['parts']
        for site,b in CELLS:
            cell=f'{site}@{b}'
            for key in ('H','full',*(f't{i}' for i in range(7))):
                if key.startswith('t') and not any(r['groups'][key] for r in rows):
                    continue
                acts = [('kv',site)]
                if key!='H':
                    acts.append(('kv',key))
                arms[f'{key}|{cell}'] = dict(donor=donors,heads=headmap(b),actions=acts)
        return arms

    selected = io.read('selected.json')['templates']
    for row in rows:
        frozen = selected[str(row['frame'])]['offset']
        rpos = row['suffix'][frozen]
        cpos = row['suffix'][(frozen+1)%len(row['suffix'])]
        group = row['groups']
        group.update(best=[rpos],control=[cpos],except_best=[j for j in row['suffix'] if j!=rpos],
                     except_control=[j for j in row['suffix'] if j!=cpos])
        bit = hashlib.sha256(f"{SEED}:{row['id']}".encode()).digest()[0]
        earlier = sorted(set(range(min(group['names'])-len(row['suffix']),min(group['names']))) - set(group['names']))
        group['prior_control'] = [earlier[bit%len(earlier)]]
    specs = {'mask_best':[('mask','best')], 'mask_control':[('mask','control')], 'mask_prior':[('mask','prior_control')],
             'mask_all':[('mask','full')], 'reopen_best':[('mask','except_best')], 'reopen_control':[('mask','except_control')],
             'zero_best':[('zero','best')], 'zero_control':[('zero','control')], 'zero_all':[('zero','full')],
             'restore_best':[('zero','full'),('own_v','best')], 'restore_control':[('zero','full'),('own_v','control')],
             'restore_all':[('zero','full'),('own_v','full')], 'self_full':[('own_v','full')]}
    specs.update({f'mask_t{j}':[('mask',f't{j}')] for j in range(7)})
    arms.update({k:dict(donor=own,heads=H16,actions=v) for k,v in specs.items()})
    return arms


def run(stage, split):
    import torch
    import transformers
    from source_binding_qkv_suffix_token_engine import run as execute
    from shared.dist_protocol.natural_register_runner import snapshot
    import source_binding_design as d

    frozen = io.read('design.json')
    for file,digest in frozen['code'].items():
        assert sha(Path(__file__).with_name(file))==digest,file
    assert frozen['snapshot']==d.read(d.OUT/'qwen14-identity.json')['model_snapshot']
    material = f'{split}-materials.json'
    assert sha(OUT/material)==frozen['files'][material]
    rows = io.read(material)
    if split=='confirm' or stage=='natural':
        selection = io.read('selected.json')
        assert selection['design_sha256']==sha(OUT/'design.json') and selection['discovery_manifest_sha256']==sha(OUT/'scan-discover-manifest.json')
    file = f'{stage}-{split}-manifest.json'
    paths = []
    print(stage,split,torch.__version__,transformers.__version__,flush=True)
    for i in range(0,len(rows),2*CHUNK):
        name = f'{stage}-{split}-{i//(2*CHUNK):02d}.json'
        if not (OUT/name).exists():
            chunks=[rows[j:j+CHUNK] for j in (i,i+CHUNK)]
            results,meta = execute(chunks,lambda rs:arms_for(rs,stage),name,OUT,lambda:io.check_write(OUT,1<<28,operation='qkv-suffix-token'))
            checks=[]
            for chunk,block in zip(chunks,results,strict=True):
                assert [r['id'] for r in chunk]==[r['id'] for r in block['base']]
                max_diff=0
                for base,noop,self_ in zip(block['base'],block['noop'],block['self'],strict=True):
                    for x in (noop,self_):
                        max_diff=max(max_diff,max(abs(a-b) for a,b in zip(x['scores'],base['scores'],strict=True)))
                        assert x['pred_record']==base['pred_record']
                assert max_diff<=1 and sum(r['forced_correct'] for r in block['base'])>=.95*len(chunk)
                if stage=='natural':
                    assert block['restore_all']==block['self_full']
                checks.append(dict(max_diff=max_diff,behavior=sum(r['forced_correct'] for r in block['base'])/len(chunk),
                    wide_base_replay=sum(row['wide']['base']!=pred['pred_record'] for row,pred in zip(chunk,block['base']))))
            io.save(name,dict(meta=meta,checks=checks,items=results))
            print(name,checks,flush=True)
        paths.append(dict(name=name,sha256=sha(OUT/name)))
    io.save(file,dict(passed=True,shards=paths,design_sha256=sha(OUT/'design.json'),
                      selection_sha256=sha(OUT/'selected.json') if split=='confirm' or stage=='natural' else None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('design','scan','natural'))
    parser.add_argument('--split',choices=('discover','confirm'),default='discover')
    args=parser.parse_args()
    io.OUT=OUT
    io.setup()
    if args.action=='design':
        design()
    else:
        run(args.action,args.split)


if __name__=='__main__':
    main()
