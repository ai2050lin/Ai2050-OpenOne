"""Freeze candidate selection and score the preregistered B31→B32 endpoints."""
import argparse
from datetime import datetime

import numpy as np
from scipy.stats import binomtest

import mlp_b31_authority_design as D
import mlp_b34_confirm_analyze as A
from shared.dist_protocol.mechanism_trace import unbits

OUT, READ, SAVE, HASH = D.OUT,D.READ,D.SAVE,D.HASH


def chunks(plan,split):
    items=D.items(plan,split)
    plan_sha=HASH(OUT/'prereg.json')
    selection_sha=HASH(OUT/'selection.json') if split=='holdout' else None
    for offset in range(0,len(items),16):
        path=OUT/f'{split}-{offset:04d}.npz'
        meta=READ(path.with_suffix('.json'))
        sample=items[offset:offset+16]
        assert meta['raw_sha256']==HASH(path) and meta['plan_sha256']==plan_sha
        assert meta['selection_sha256']==selection_sha and meta['provenance']==plan['provenance']
        assert [r['id'] for r in sample]==[r['id'] for r in meta['rows']]
        yield sample,meta['rows'],path


def capture(plan,split):
    rows,items,files=[],[],[]
    data={b:[] for b in (31,32)}
    for sample,records,path in chunks(plan,split):
        rows.extend(records)
        items.extend(sample)
        files.extend({'path':p.name,'sha256':HASH(p),'bytes':p.stat().st_size} for p in (path,path.with_suffix('.json')))
        with np.load(path) as raw:
            for b in data:
                tensor=unbits(raw[f'A{b}'])
                data[b].append(tensor[:,1]-tensor[:,0])
    return items,rows,{b:np.concatenate(v) for b,v in data.items()},files


def both(items,rows):
    return np.all(np.asarray([r['base_scores'] for r in rows]).argmax(-1)==[s['correct'] for s in items],axis=1)


def select(plan):
    assert not (OUT/'selection.json').exists() and not list(OUT.glob('holdout-*.npz'))
    items,rows,data,files=capture(plan,'discovery')
    rate=float(both(items,rows).mean())
    out={'provenance':plan['provenance'],'plan_sha256':HASH(OUT/'prereg.json'),'discovery_chunks':files,
         'discovery_both_correct':rate,'baseline_gate':rate>=.85,'frozen_at':datetime.now().astimezone().isoformat()}
    rng=np.random.default_rng(D.SEED)
    for b,diff in data.items():
        mean=diff.mean(0,dtype=np.float64)
        scale=np.maximum(np.sqrt(np.square(diff,dtype=np.float64).mean(0)),1e-6)
        units=np.argsort(-abs(mean)/scale,kind='stable')[:D.GROUP]
        control=rng.choice(np.setdiff1d(np.arange(D.WIDTH),units),D.GROUP,replace=False)
        out[f'B{b}']={'units':units.tolist(),'write':mean[units].tolist(),'orientation':np.sign(mean[units]).tolist(),
                      'scale':scale[units].tolist(),'control':control.tolist(),
                      'control_orientation':np.sign(mean[control]).tolist(),'control_scale':scale[control].tolist()}
    SAVE(OUT/'selection.json',out)
    print({'selection_sha256':HASH(OUT/'selection.json'),'pair_accuracy':rate,'gate':out['baseline_gate']},flush=True)


def semantic(data,items,selection,plan):
    result={}
    code=np.array([s['reverse_code'] for s in items])
    frame=np.array([s['frame'] for s in items])

    def breakdown(hit):
        return [float(hit[code==i].mean()) for i in (False,True)],{k:float(hit[frame==k].mean()) for k in sorted(set(frame))}

    for b,diff in data.items():
        card=selection[f'B{b}']
        z=diff[:,card['units']]*card['orientation']/card['scale']
        null=diff[:,card['control']]*card['control_orientation']/card['control_scale']
        hit=z.mean(1)>0
        group=A.paired(hit,null.mean(1)>0)
        group['by_code'],group['by_frame']=breakdown(hit)
        group['passed']=group['p']<=group['alpha'] and group['gain']>=.05 and group['edited_rate']>=.62 and min(group['by_code'])>=.55
        units=[]
        for i,u in enumerate(card['units']):
            hit_unit=z[:,i]>0
            hits=int(hit_unit.sum())
            p=binomtest(hits,len(items),.5,alternative='greater').pvalue
            codes,frames=breakdown(hit_unit)
            units.append({'unit':u,'n':len(items),'hits':hits,'rate':hits/len(items),'p':p,'by_code':codes,'by_frame':frames,
                          'orientation':card['orientation'][i],'ties':int((z[:,i]==0).sum()),
                          'passed':p<=plan['alpha_unit'] and hits/len(items)>=.62 and min(codes)>=.55})
        result[f'B{b}']={'group':group,'units':units,'failed_group_ids':[r['id'] for r,ok in zip(items,hit) if not ok]}
    return result


def report(plan):
    selected=READ(OUT/'selection.json')
    assert selected['plan_sha256']==HASH(OUT/'prereg.json') and selected['baseline_gate']
    items,rows,diff,raw=capture(plan,'holdout')
    assert all(r['restore_bitwise'] for r in rows)
    pair=both(items,rows)
    baseline=float(pair.mean())
    baseline_gate=baseline>=.85
    semantics=semantic(diff,items,selected,plan)
    doses,errors=A.causal_test(items,rows,{'B35':selected['B32']},baseline_gate)
    for i,dose in enumerate(doses):
        source=[]
        for row,record in zip(items,rows):
            condition=record['doses'][i]
            if not condition['norm_valid']:
                continue
            recipient=row['recipient_index']
            gold=plan['choice_ids'][row['correct'][recipient]]
            desired=plan['choice_ids'][row['correct'][1-recipient]]
            was_correct=record['base_top1'][recipient]==gold
            source.append((was_correct,was_correct and condition['edit']['top1']==desired,
                           was_correct and condition['control']['top1']==desired))
        dose['full_vocabulary']={'n':len(source),'base_gold':sum(row[0] for row in source),
                                 'edit_target':sum(row[1] for row in source),'control_target':sum(row[2] for row in source)}
    A.OUT=OUT
    cap=A.capability_test(plan)
    primary=next(row for row in doses if row['dose']==D.PRIMARY)
    switch=bool(primary['switch']['passed'])
    mediation=bool(switch and primary['rescue']['passed'])
    error_ids={r['sample_id'] for r in errors}
    kw=READ(OUT/'keyword-audit.json')
    raw.extend(selected['discovery_chunks'])
    for name in ('keyword-audit.json','calibration.json','capability.json','capability-materials.json','discovery-materials.json','holdout-materials.json','selection.json','prereg.json'):
        p=OUT/name
        raw.append({'path':name,'bytes':p.stat().st_size,'sha256':HASH(p)})
    result={'schema':'mlp_b31_authority_report.v1','kind':'real','run_id':D.RUN,'model_snapshot':D.SNAPSHOT,
            'provenance':plan['provenance'],'preregistration':plan,'plan_sha256':HASH(OUT/'prereg.json'),
            'selection_sha256':HASH(OUT/'selection.json'),'discovery_pairs':D.DISCOVERY,'holdout_pairs':len(items),
            'holdout_both_correct':baseline,'baseline_gate':baseline_gate,
            'by_frame_baseline':{k:float(pair[np.array([r['frame']==k for r in items])].mean()) for k in sorted({r['frame'] for r in items})},
            'semantic':semantics,'dose_curve':doses,'controlled_behavior_passed':switch,'mediation_passed':mediation,
            'restored_pairs':len(rows),'capability':cap,'keyword_audit':kw,
            'counterexamples':errors,'counterexample_materials':{r['id']:{'frame':r['frame'],'prompts':r['prompts']}
                                                   for r in items if r['id'] in error_ids},
            'raw_manifest':raw,'scope':'一次物理节点的预注册测试。新文本为作者编制的英文权限绑定题。正式翻转端点是 A/B，对词表首词另列。能力门为历史选择题池。'}
    SAVE(OUT/'report.json',result)
    print({'baseline':baseline,'semantic':{k:v['group'] for k,v in semantics.items()},'primary':{k:primary[k] for k in ('switch','rescue','full_vocabulary')},
           'capability':{k:{'n':v['n'],'lost':v['control_only'],'upper':v['loss_upper'],'passed':v['passed']} for k,v in cap.items()}},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['select','report'])
    command=parser.parse_args().stage
    globals()[command](D.checked())
