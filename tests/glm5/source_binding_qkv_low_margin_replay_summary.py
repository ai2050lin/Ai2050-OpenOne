"""Audit physical-neighbor comparisons, reversible flips and full-suffix rescue."""
from collections import Counter
from pathlib import Path

import source_binding_qkv_low_margin as exp
import source_binding_qkv_suffix_token_analysis as stats
import source_binding_transfer_factors as io
from source_binding_design import sha


def results(rows):
    rules=(('repeat_mask_key','mask_near'),('restore','repeat_mask_key'),
           ('reopen_key','mask_all'),('reopen_near','mask_all'),('reopen_key','reopen_near'),
           ('restore_key','zero_all'),('restore_near','zero_all'),('restore_key','restore_near'))
    pairs={}
    rate=Counter()
    top1=Counter()
    exact=0
    for r in rows:
        arms=r['outputs'] | {'original':r['original'],'restore':r['restore']}
        exact+=arms['restore']['scores']==arms['original']['scores']
        rate.update({k:int(v['forced_correct']) for k,v in arms.items()})
        top1.update({k:int(v['top1_correct']) for k,v in arms.items()})
    for after,before in rules:
        observations=[]
        ups,downs=[],[]
        for r in rows:
            arms=r['outputs'] | {'restore':r['restore']}
            a,b=arms[after],arms[before]
            observations.append((r['panel']+':'+r['family'],int(b['forced_correct']),int(a['forced_correct']),
                                 stats.margin(b,r['target_record']),stats.margin(a,r['target_record'])))
            if a['forced_correct'] and not b['forced_correct']:
                ups.append([r['panel'],r['id']])
            if b['forced_correct'] and not a['forced_correct']:
                downs.append([r['panel'],r['id']])
        pairs[f'{after}_minus_{before}']=dict(stats=stats.paired(observations),recovered=ups,regressed=downs)
    return dict(n=len(rows),families=len({(r['panel'],r['family']) for r in rows}),accuracy=rate,top1=top1,
                restore_original_scores_exact=exact,contrasts=pairs)


def main():
    root=exp.OUT.parent
    reports={}
    rows=[]
    for panel in ('low-margin','low-margin-expanded','low-margin-conflict'):
        io.OUT=root/panel
        if not (io.OUT/'single-edge-replay.json').exists():
            continue
        replay=io.read('single-edge-replay.json')
        selection=io.read('selected.json')
        assert replay['passed'] and replay['complete']==len(selection['rows'])
        assert replay['manifest_sha256']==sha(io.OUT/'test-manifest.json')
        assert replay['script_sha256']==sha(Path(__file__).with_name('source_binding_qkv_low_margin_flip_replay.py'))
        assert all(sha(io.OUT/k)==v for k,v in replay['files'].items())
        for r in replay['records']:
            r['panel']=panel
        rows.extend(replay['records'])
        reports[panel]=dict(summary=results(replay['records']),sha256=sha(io.OUT/'single-edge-replay.json'))
    assert rows
    stable=[]
    for r in rows:
        m=stats.margin(r['original'],r['target_record'])
        if all(r['outputs'][k]['forced_correct'] and abs(stats.margin(r['outputs'][k],r['target_record'])-m)<m
               for k in ('noop','self','self_all')):
            stable.append(r)
    io.OUT=root/'low-margin-expanded'
    obj=dict(panels=reports,all=results(rows),stable=results(stable),script_sha256=sha(__file__),
             flip_cases=[r for r in rows if not r['outputs']['repeat_mask_key']['forced_correct']])
    io.save('physical-neighbor-review.json',obj)
    print('full',obj['all']['n'],obj['all']['accuracy'],flush=True)
    print('stable',obj['stable']['n'],obj['stable']['accuracy'],flush=True)


if __name__=='__main__':
    main()
