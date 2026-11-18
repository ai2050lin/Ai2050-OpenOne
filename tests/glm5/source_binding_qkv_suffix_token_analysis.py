"""Freeze and evaluate a family-paired token localization and natural-use experiment."""
from collections import defaultdict
import argparse
import itertools
from pathlib import Path

import numpy as np

import source_binding_qkv_suffix_token as exp
import source_binding_transfer_factors as io
from source_binding_design import sha


def data(stage, split):
    manifest = io.read(f'{stage}-{split}-manifest.json')
    design = io.read('design.json')
    assert manifest['passed'] and manifest['design_sha256']==sha(exp.OUT/'design.json')
    if manifest['selection_sha256']:
        assert manifest['selection_sha256']==sha(exp.OUT/'selected.json')
    file = f'{split}-materials.json'
    assert design['files'][file]==sha(exp.OUT/file)
    rows = {r['id']:r for r in io.read(file)}
    for part in manifest['shards']:
        assert sha(exp.OUT/part['name'])==part['sha256']
        payload = io.read(part['name'])
        assert all(c['wide_base_replay']==0 for c in payload['checks'])
        for block in payload['items']:
            for i,base in enumerate(block['base']):
                r = rows.pop(base['id'])
                result = {k:v[i] for k,v in block.items()}
                assert all(x['id']==r['id'] for x in result.values())
                yield r,result
    assert not rows


def margin(row, correct):
    scores = row['scores']
    return scores[correct]-max(x for i,x in enumerate(scores) if i!=correct)


def paired(observations, level=.95):
    families=defaultdict(lambda: np.zeros(5))
    for family,bh,ah,bm,am in observations:
        families[family]+= (1,bh,ah,bm,am)
    count = np.array([families[k] for k in sorted(families)])
    if not len(count):
        return dict(n=0)
    sample = np.random.default_rng(exp.SEED).multinomial(len(count),np.full(len(count),1/len(count)),size=8000)@count
    n,bh,ah,bm,am = count.sum(0)
    alpha=(1-level)/2
    ci=lambda v:np.quantile(v,[alpha,1-alpha]).tolist()
    return dict(n=int(n),families=len(count),before=int(bh),after=int(ah),rate=float(ah/n),gain=float((ah-bh)/n),
                margin_before=float(bm/n),margin_after=float(am/n),margin_gain=float((am-bm)/n),
                gain_ci=ci((sample[:,2]-sample[:,1])/sample[:,0]),margin_gain_ci=ci((sample[:,4]-sample[:,3])/sample[:,0]),interval=level)


def event(row, base, altered, target):
    return row['family'], int(base['pred_record']==target), int(altered['pred_record']==target), margin(base,target), margin(altered,target)


def scan(split):
    slots=defaultdict(lambda:defaultdict(list))
    pieces=defaultdict(set)
    frozen=io.read('selected.json')['templates'] if split=='confirm' else None
    pairs=defaultdict(list)
    for row,outcomes in data('scan',split):
        frame=str(row['frame'])
        for site,b in exp.CELLS:
            cell=f'{site}@{b}'
            key='W|'+cell
            gold=row['wide'][key]
            if gold==row['wide']['base']:
                continue
            base=outcomes['H|'+cell]
            for j,token in enumerate(row['parts']):
                v=outcomes[f't{j}|{cell}']
                pair=event(row,base,v,gold)
                slots[frame][str(j)].append(pair)
                pieces[frame,str(j)].add(token)
                slots['end'][str(j-len(row['parts']))].append(pair)
                if j==row['animal_offset']:
                    slots['lexical']['动物'].append(pair)
            full=event(row,base,outcomes['full|'+cell],gold)
            slots[frame]['full'].append(full)
            slots['all']['full'].append(full)
            if frozen:
                best=outcomes[f't{frozen[frame]["offset"]}|{cell}']
                e=event(row,base,best,gold)
                slots['all']['selected'].append(e)
                slots['all_by_site'][site].append(e)
                pairs[frame].append(event(row,best,outcomes['full|'+cell],gold))
                pairs['all'].append(event(row,best,outcomes['full|'+cell],gold))
    summary={frame:{k:paired(events) for k,events in stats.items()} for frame,stats in slots.items()}
    for (f,k),v in pieces.items():
        summary[f][k]['pieces']=sorted(v)
    record=dict(groups=summary,analysis_sha256=sha(Path(__file__)),manifest_sha256=sha(exp.OUT/f'scan-{split}-manifest.json'),
                whole_minus_selected={k:paired(v) for k,v in pairs.items()})
    io.save(f'scan-{split}-report.json',record)
    if split=='discover':
        candidates={}
        for frame in ('0','1','2','3'):
            j,v=max(((int(j),v) for j,v in summary[frame].items() if j.isdigit()),
                    key=lambda x:(x[1]['gain'],x[1]['margin_gain'],-x[0]))
            candidates[frame]=dict(offset=j,token=v['pieces'],discovery=v)
        selection=dict(templates=candidates,design_sha256=sha(exp.OUT/'design.json'),discovery_manifest_sha256=record['manifest_sha256'])
        if (exp.OUT/'selected.json').exists():
            assert io.read('selected.json')==selection
        else:
            io.save('selected.json',selection)
        print(candidates,flush=True)
    else:
        print('confirm',summary['all'],summary['lexical'],flush=True)


def natural(split):
    frozen=io.read('selected.json')['templates']
    groups=defaultdict(lambda:defaultdict(list))
    top1=defaultdict(lambda:defaultdict(list))
    rescues=defaultdict(lambda:defaultdict(list))
    recovery_pairs=(('reopen_best','mask_all'),('reopen_control','mask_all'),('reopen_best','reopen_control'),
                    ('restore_best','zero_all'),('restore_control','zero_all'),('restore_best','restore_control'),
                    ('restore_all','zero_all'),('mask_best','mask_control'),('mask_best','mask_prior'),('zero_best','zero_control'))
    for row,arms in data('natural',split):
        target=row['target_record']
        frame=str(row['frame'])
        arms['mask_animal']=arms[f'mask_t{row["animal_offset"]}']
        for scope in ('all',frame):
            for k,out in arms.items():
                if k.startswith('mask_t') and int(k[6:])>=len(row['suffix']):
                    continue
                groups[scope][k].append(event(row,arms['base'],out,target))
                top1[scope][k].append((row['family'],int(arms['base']['top1_correct']),int(out['top1_correct']),0.,0.))
            for a,b in recovery_pairs:
                rescues[scope][a+'_minus_'+b].append(event(row,arms[b],arms[a],target))
    summary={scope:{k:paired(v) for k,v in arms.items()} for scope,arms in groups.items()}
    rep=dict(arms=summary,top1={scope:{k:paired(v) for k,v in arms.items()} for scope,arms in top1.items()},
             comparisons={scope:{k:paired(v) for k,v in arms.items()} for scope,arms in rescues.items()},
             analysis_sha256=sha(Path(__file__)),manifest_sha256=sha(exp.OUT/f'natural-{split}-manifest.json'))
    io.save(f'natural-{split}-report.json',rep)
    for k in ('base','mask_best','mask_control','mask_animal','mask_all','reopen_best','zero_best','zero_all','restore_best','restore_all'):
        print(k,summary['all'][k],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('scan','natural'))
    ap.add_argument('--split',choices=('discover','confirm'),default='discover')
    args=ap.parse_args()
    io.OUT=exp.OUT
    (scan if args.stage=='scan' else natural)(args.split)


if __name__=='__main__':
    main()
