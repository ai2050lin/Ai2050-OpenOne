"""Freeze the low-margin cohort and audit paired accuracy, errors and recoveries."""
import argparse
from collections import Counter, defaultdict
import math

import source_binding_qkv_low_margin as exp
import source_binding_qkv_suffix_token_analysis as stats
import source_binding_transfer_factors as io
from source_binding_design import sha


def read_run(mode):
    manifest=io.read(f'{mode}-manifest.json')
    assert manifest['passed'] and manifest['design_sha256']==sha(exp.OUT/'design.json')
    if mode=='test':
        assert manifest['selected_sha256']==sha(exp.OUT/'selected.json')
    rows=io.read('materials.json')
    chunks=exp.batches(rows)
    results={}
    for part in manifest['shards']:
        assert sha(exp.OUT/part['name'])==part['sha256']
        obj=io.read(part['name'])
        for i,group in zip(obj['indices'],obj['items'],strict=True):
            for j,(r,b) in enumerate(zip(chunks[i],group['base'],strict=True)):
                assert r['id']==b['id'] and r['id'] not in results
                outputs={arm:items[j] for arm,items in group.items()}
                assert all(o['id']==r['id'] for o in outputs.values())
                results[r['id']]=(r,outputs,i)
    return results


def freeze():
    results=read_run('screen')
    design=io.read('design.json')
    assert len(results)==design['prompts']
    selected, summaries = [], defaultdict(Counter)
    for id_,(r,g,chunk) in results.items():
        b=g['base']; correct=r['target_record']
        m=stats.margin(b,correct)
        rival=max((j for j in range(4) if j!=correct),key=lambda j:b['scores'][j])
        for k in ('all',f'frame{r["frame"]}',f'variant{r["variant"]}'):
            v=summaries[k]
            v['n']+=1
            v['correct']+=b['forced_correct']
            v['top1']+=b['top1_correct']
            v['intended_foil_runner_up']+=rival==r['foil']
            v['zero_margin']+=m==0
            v['selected']+=0<m<=exp.MAX_MARGIN and b['forced_correct']
        if b['forced_correct'] and 0<m<=exp.MAX_MARGIN:
            selected.append(dict(id=id_,margin=m,best_rival=rival,chunk=chunk,frame=r['frame'],variant=r['variant'],
                                 foil=r['foil'],top1=b['top1_correct'],scores=b['scores']))
    assert selected
    selected.sort(key=lambda x:x['id'])
    obj=dict(rows=selected,chunks=sorted({r['chunk'] for r in selected}),
             manifest_sha256=sha(exp.OUT/'screen-manifest.json'),rule='0 < margin <= 1.0 and unedited four-choice correct',
             screen_summary=summaries,families=len({results[r['id']][0]['family'] for r in selected}))
    if (exp.OUT/'selected.json').exists():
        assert io.read('selected.json')==obj
    else:
        io.save('selected.json',obj)
    print('screen',len(results),'selected',len(selected),'families',obj['families'],dict(summaries),flush=True)


def analyze():
    selected=io.read('selected.json')
    assert selected['manifest_sha256']==sha(exp.OUT/'screen-manifest.json')
    test=read_run('test')
    groups=defaultdict(list)
    raw=[]
    for r in selected['rows']:
        meta, arms, chunk=test[r['id']]
        base=arms['base']; target=meta['target_record']
        assert chunk==r['chunk'] and base['scores']==r['scores'] and base['forced_correct']
        assert set(arms)=={'base',*exp.ARMS}
        assert arms['restore_all']==arms['self_all']
        sham_names=('noop','self','self_all')
        sham_margin=max(abs(stats.margin(arms[k],target)-r['margin']) for k in sham_names)
        sham_choice=all(arms[k]['pred_record']==target for k in sham_names)
        for name in ('all',f'frame{meta["frame"]}',f'variant{meta["variant"]}',
                     'tight' if r['margin']<=exp.TIGHT_MARGIN else 'upper'):
            groups[name].append((meta,arms))
        if r['margin']<=.25:
            groups['very_tight'].append((meta,arms))
        if sham_choice:
            groups['sham_stable'].append((meta,arms))
        if sham_choice and sham_margin<r['margin']:
            groups['sham_stable_margin'].append((meta,arms))
        if base['top1_correct']:
            groups['top1_base_correct'].append((meta,arms))
        raw.append(dict(id=meta['id'],frame=meta['frame'],target=target,foil=meta['foil'],runner_up=r['best_rival'],baseline_margin=r['margin'],
                        sham_stable=sham_choice,sham_margin_change=sham_margin,
                        choices={k:x['pred_record'] for k,x in arms.items()},top1={k:x['top1_correct'] for k,x in arms.items()},
                        margins={k:stats.margin(x,target) for k,x in arms.items()}))
    pairs=(('mask_key','mask_adjacent'),('mask_key','noop'),('mask_adjacent','noop'),('mask_all','noop'),
           ('reopen_key','mask_all'),('reopen_adjacent','mask_all'),('reopen_key','reopen_adjacent'),
           ('restore_key','zero_all'),('restore_adjacent','zero_all'),('restore_key','restore_adjacent'),
           ('restore_all','zero_all'),('restore_all','self_all'),('reopen_key','mask_key'))
    output={}
    for k,rows in groups.items():
        accuracy=Counter();top1=Counter(); recovered={}; comparisons={}
        for _,out in rows:
            accuracy.update({name:int(x['forced_correct']) for name,x in out.items()})
            top1.update({name:int(x['top1_correct']) for name,x in out.items()})
        for after,before in pairs:
            events=[stats.event(r,x[before],x[after],r['target_record']) for r,x in rows]
            key=f'{after}_minus_{before}'
            comparisons[key]=stats.paired(events)
            recovered[key]=dict(from_wrong_to_correct=sum(not x[1] and x[2] for x in events),
                                from_correct_to_wrong=sum(x[1] and not x[2] for x in events),
                                wrong_before=sum(not x[1] for x in events))
        error_types=Counter()
        for meta,out in rows:
            if not out['mask_key']['forced_correct']:
                error_types['all']+=1
                error_types['to_planted_foil']+=out['mask_key']['pred_record']==meta['foil']
        output[k]=dict(n=len(rows),families=len({row['family'] for row,_ in rows}),correct=dict(accuracy),top1_correct=dict(top1),
                       contrasts=comparisons,recovery=recovered,key_error_types=error_types)
    io.save('report.json',dict(groups=output,items=raw,selection_sha256=sha(exp.OUT/'selected.json'),
        test_manifest_sha256=sha(exp.OUT/'test-manifest.json'),analysis_sha256=sha(__file__),
        screening=selected['screen_summary'],note='Conditioned on intact success and a small four-candidate margin. Family bootstrap. The single-mask removal is represented by intact/sham replays. Selective rescue is a separate full-suffix intervention.'))
    for k in ('all','sham_stable','sham_stable_margin','top1_base_correct'):
        if k in output:
            x=output[k]
            print(k,x['n'],x['families'],x['correct'],x['top1_correct'],flush=True)
    print('primary recoveries',output['all']['recovery'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('freeze','analyze'))
    mode=p.parse_args().mode
    io.OUT=exp.OUT
    (freeze if mode=='freeze' else analyze)()


if __name__=='__main__':
    main()
