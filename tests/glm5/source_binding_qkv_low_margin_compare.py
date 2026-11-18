"""Compare both independently frozen low-margin panels and retain case-level evidence."""
from collections import Counter
from pathlib import Path

import source_binding_qkv_low_margin as exp
import source_binding_qkv_low_margin_analysis as analysis
import source_binding_qkv_suffix_token_analysis as stats
import source_binding_transfer_factors as io
from source_binding_design import sha


CONTRASTS = (('mask_key','mask_adjacent'),('mask_key','noop'),('reopen_key','mask_all'),
             ('reopen_key','reopen_adjacent'),('restore_key','zero_all'),('restore_key','restore_adjacent'))


def load(root, chunk):
    exp.OUT,exp.CHUNK,io.OUT=root,chunk,root
    design=io.read('design.json')
    assert design['materials_sha256']==sha(root/'materials.json')
    for k,v in design['code'].items():
        assert sha(Path(__file__).with_name(k))==v
    assert design['selected_sha256']==sha(exp.token_test.OUT/'selected.json')
    raw=io.read('report.json')
    assert raw['selection_sha256']==sha(root/'selected.json') and raw['test_manifest_sha256']==sha(root/'test-manifest.json')
    selection=io.read('selected.json')
    assert selection['manifest_sha256']==sha(root/'screen-manifest.json')
    screen=analysis.read_run('screen')
    test=analysis.read_run('test')
    assert len(screen)==design['prompts']
    selected_ids={x['id'] for x in selection['rows']}
    assert selected_ids=={id_ for id_,(r,o,_) in screen.items() if o['base']['forced_correct'] and 0<stats.margin(o['base'],r['target_record'])<=exp.MAX_MARGIN}
    observed=[]
    exact_ids=[]
    for name,(meta,arms,i) in test.items():
        assert arms['base']==screen[name][1]['base'] and i==screen[name][2]
        assert arms['restore_all']==arms['self_all']
        if name not in selected_ids:
            continue
        meta = dict(meta,family=root.name+'-'+meta['family'])
        observed.append((meta,arms))
        if all(arms[a]['scores']==arms['base']['scores'] for a in ('noop','self','self_all')):
            exact_ids.append(name)
    assert len(observed)==len(selected_ids)
    return dict(screen=selection['screen_summary']['all'],results=raw['groups'],rows=observed,exact_ids=exact_ids,
                report_sha256=sha(root/'report.json'))


def summarize(rows):
    accuracy=Counter(); top1=Counter()
    examples=[]
    for r,out in rows:
        accuracy.update({k:int(v['forced_correct']) for k,v in out.items()})
        top1.update({k:int(v['top1_correct']) for k,v in out.items()})
        if not out['mask_key']['forced_correct'] and out['mask_adjacent']['forced_correct']:
            examples.append(dict(id=r['id'],family=r['family'],content=r['content'],key_token=r['parts'][r['suffix'].index(r['key'])],
                                 values=r['values'],correct=r['target_record'],outputs={k:v for k,v in out.items() if k in ('base','mask_key','mask_adjacent','mask_all','reopen_key','reopen_adjacent','restore_all')}))
    pair = {}
    for after,before in CONTRASTS:
        v=[stats.event(r,o[before],o[after],r['target_record']) for r,o in rows]
        pair[after+'_minus_'+before]=dict(stats=stats.paired(v),wrong_to_correct=sum(not b and a for _,b,a,_,_ in v),
                                        correct_to_wrong=sum(b and not a for _,b,a,_,_ in v))
    return dict(n=len(rows),families=len({r['family'] for r,_ in rows}),accuracy=accuracy,top1=top1,contrasts=pair,examples=examples)


def main():
    import argparse

    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--with-conflict',action='store_true')
    args=ap.parse_args()
    parent=exp.OUT.parent
    cohorts={'primary':load(parent/'low-margin',32),'expanded':load(parent/'low-margin-expanded',64)}
    if args.with_conflict:
        cohorts['conflict']=load(parent/'low-margin-conflict',64)
    rows=[row for panel in cohorts.values() for row in panel.pop('rows')]
    stable=[]
    for r,o in rows:
        target=r['target_record']
        margin=stats.margin(o['base'],target)
        controls=[o[k] for k in ('noop','self','self_all')]
        if all(x['forced_correct'] and abs(stats.margin(x,target)-margin)<margin for x in controls):
            stable.append((r,o))
    io.OUT=parent/'low-margin-expanded'
    report=dict(panels=cohorts,combined=summarize(rows),stable=summarize(stable),
                code_sha256=sha(__file__),combined_scope='Descriptive pooling. Separate distractor distributions and independent name pools.',
                stable_rule='All sham choices correct and maximal sham margin deviation smaller than original margin.')
    io.save('three-cohort-review.json' if args.with_conflict else 'paired-cohort-review.json',report)
    print('pooled',report['combined']['n'],report['combined']['accuracy'],flush=True)
    print('stable',report['stable']['n'],report['stable']['accuracy'],flush=True)


if __name__=='__main__':
    main()
