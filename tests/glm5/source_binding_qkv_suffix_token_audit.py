"""Audit both token panels and pool the fixed natural-use contrasts."""
from collections import defaultdict
import json
from pathlib import Path

import source_binding_qkv_suffix_token as exp
import source_binding_qkv_suffix_token_analysis as analysis
import source_binding_transfer_factors as io
from source_binding_design import sha


def main():
    io.OUT = exp.OUT
    design = io.read('design.json')
    for name, expected in design['code'].items():
        assert sha(Path(__file__).with_name(name))==expected
    assert design['source_sha256']==sha(exp.OUT.parent/'holdout-manifest.json')
    controls = defaultdict(lambda:dict(prompts=0,behavior_hits=0,replay_score_max=0,restored_exact=0))
    comparisons = defaultdict(list)
    top1 = defaultdict(list)
    errors = defaultdict(list)
    hashes = {name:sha(exp.OUT/name) for name in ('design.json','selected.json','discover-materials.json','confirm-materials.json')}
    keys = (('base','mask_best'),('base','mask_control'),('base','mask_prior'),('base','mask_animal'),('base','mask_all'),
            ('mask_all','reopen_best'),('mask_all','reopen_control'),('reopen_control','reopen_best'),
            ('base','zero_best'),('base','zero_control'),('base','zero_all'),('zero_all','restore_best'),
            ('zero_all','restore_control'),('restore_control','restore_best'),('zero_all','restore_all'),('self_full','restore_all'))
    for split in ('discover','confirm'):
        files = [f'{stage}-{split}-{end}.json' for stage in ('scan','natural') for end in ('manifest','report')]
        hashes.update({f:sha(exp.OUT/f) for f in files})
        for stage in ('scan','natural'):
            stats = controls[stage]
            for row,outcomes in analysis.data(stage,split):
                natural = outcomes['base']
                stats['prompts']+=1
                stats['behavior_hits']+=natural['forced_correct']
                for k in ('self','noop'):
                    out=outcomes[k]
                    assert natural['pred_record']==out['pred_record']
                    stats['replay_score_max']=max(stats['replay_score_max'],max(abs(p-q) for p,q in zip(natural['scores'],out['scores'],strict=True)))
                if stage=='natural':
                    assert outcomes['restore_all']==outcomes['self_full']
                    stats['restored_exact']+=1
                    outcomes['mask_animal']=outcomes[f'mask_t{row["animal_offset"]}']
                    for before, after in keys:
                        name = f'{after}_minus_{before}'
                        a,b=outcomes[before],outcomes[after]
                        comparisons[name].append(analysis.event(row,a,b,row['target_record']))
                        top1[name].append((row['family'],int(a['top1_correct']),int(b['top1_correct']),0,0))
                        if a['pred_record']!=b['pred_record'] or a['top1']!=b['top1']:
                            errors[name].append(dict(id=row['id'],frame=row['frame'],before=a,after=b))
    anchor_rows=0
    for shard in io.read('scan-discover-manifest.json')['shards']:
        number=shard['name'][-7:-5]
        previous=json.loads((exp.OUT.parent/'suffix'/f'zh-{number}.json').read_bytes())['items']
        current=io.read(shard['name'])['items']
        for old,new in zip(previous,current,strict=True):
            for k,v in old.items():
                new_key = k.replace('after|','full|')
                if new_key in new:
                    assert v==new[new_key],(number,k)
                    anchor_rows+=len(v)
    report=dict(engine_anchor_records_exact=anchor_rows,controls=controls,
                pooled_natural={k:analysis.paired(v) for k,v in comparisons.items()},
                pooled_top1={k:analysis.paired(v) for k,v in top1.items()},relevant_changed_choices=errors,
                hashes=hashes,analysis_sha256=sha(__file__))
    io.save('audit.json',report)
    print(controls,anchor_rows,flush=True)
    for k,v in report['pooled_natural'].items():
        print(k,v['before'],v['after'],round(v['margin_gain'],3),[round(x,3) for x in v['margin_gain_ci']],flush=True)


if __name__=='__main__':
    main()
