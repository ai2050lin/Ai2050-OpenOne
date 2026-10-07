"""Group-level uncertainty, full-coordinate views, and queryable evidence updates."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path
import numpy as np
from phase2751_trusted_rebuild import ROOT,OUT,OLD,write,sha,load_bank


def boot(records,value,seed=2751011):
    # Resample semantic source groups, including all wording/condition rows together.
    groups={}
    for r in records:groups.setdefault(r.get('source_family',r.get('group')),[]).append(value(r))
    groups=[np.asarray(v,float) for v in groups.values()]
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(2000):
        draw=rng.integers(0,len(groups),len(groups))
        vals.append(float(np.median(np.concatenate([groups[i] for i in draw]))))
    return dict(median=float(np.median(np.concatenate(groups))),ci95=np.quantile(vals,[.025,.975]).tolist(),
                source_groups=len(groups),method='percentile bootstrap over semantic source groups; descriptive, fixed material/model')


def summarize(side):
    dest=OUT/side
    if not (dest/'measurement_repair.json').exists():return None
    comp=json.loads((dest/'composition.json').read_text(encoding='utf-8'))['groups']
    rep=json.loads((dest/'measurement_repair.json').read_text(encoding='utf-8'))
    material={r['id']:r for r in json.loads((OUT/'material.json').read_text(encoding='utf-8'))['rows']}
    summary={}
    for split in sorted(set(r['split'] for r in comp)):
        rs=[r for r in comp if r['split']==split]
        keys=rs[0]['relative_errors'].keys()
        summary[split]=dict(n_groups=len(rs),n_prompts=4*len(rs),
            errors={k:boot(rs,lambda r:r['relative_errors'][k][-1]) for k in keys},
            paired_additive_minus_negation=boot(rs,lambda r:r['relative_errors']['additive'][-1]-r['relative_errors']['negation'][-1]),
            paired_calibrated_minus_additive=boot(rs,lambda r:r['relative_errors']['dev_mean_interaction'][-1]-r['relative_errors']['additive'][-1]),
            first_token_correct=sum(b['exact_yes_no'] for r in rs for b in r['first_token']),
            denominator_first_token=4*len(rs),
            note='Development fit is in-sample. Other split predictions frozen from development; no refitting.')
    for row in rep['rows']:
        row['source_family']=material[row['id']].get('source_family',row['group'])
    repair_ci={}
    for split in sorted(set(r['split'] for r in rep['rows'])):
        rs=[r for r in rep['rows'] if r['split']==split]
        repair_ci[split]=dict(corrected_error=boot(rs,lambda r:r['corrected']['relative_error']),
             paired_corrected_minus_old=boot(rs,lambda r:r['corrected']['relative_error']-r['legacy_wrong']['relative_error']),
             projected_attention=boot(rs,lambda r:r['channel_energy']['projected_a']))
    # Recompute old incorrect statistic as a calibration only, never a valid derivative.
    p=next((OLD/'phase3100').glob('*/*.npz'));anchors=[]
    with np.load(p) as z:
        for r in rep['rows']:
            if r['split']!='legacy':continue
            _,fa,bi,ci=r['id'].split('_');k=(int(ci)-1)*8+int(bi)
            anchors.append(abs(r['legacy_wrong']['cos']-float(z[f'PRED_COS_{side}_{fa}'][k])))
    result=dict(side=side,composition=summary,repair_uncertainty=repair_ci,
        old_wrong_statistic_replay_max_abs=float(max(anchors)) if anchors else None,old_input_sha256=sha(p) if anchors else None,
        heads=rep['head_correspondence'],
        limitations=['No universal law or text-only mechanism extractor.',
            f'{len(set(r["source_family"] for r in comp))} source groups, 2 wordings; one synthetic protocol, English only.',
            ('Matched-length diagnostic uses disjoint participant entities; it is post-primary and changes pragmatics too.' if OUT.name=='length_control' else
             'Primary entity_holdout holds out focal configurations; role/chain partner names overlap development. See material_audit.json. Length control repairs this.'),
            'First-token yes/no only; full generation, stopping and deeper compositions not tested.',
            'Known last-block weights in JVP validate measurement/local approximation, not mechanism discovery.',
            'Bootstrap conditions/wordings remain clustered; no cross-model population significance.'])
    write(dest/'summary.json',result)
    return result


def plots(side):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/side;data,rows=load_bank(dest)
    groups={}
    for i,r in enumerate(rows):
        if r['split']=='joint_holdout':groups.setdefault(r['group'],{})[r['cond']]=i
    signals=[];residuals=[]
    for idx in groups.values():
        a,b,c,d=[data['hidden'][idx[k]].astype(np.float64) for k in range(4)]
        signals.append(d-a);residuals.append(d-c-b+a)
    raw=np.mean(signals,axis=0);interaction=np.mean(residuals,axis=0)
    # Exact native coordinate order, all coordinates; row scaling explicitly separate.
    fig,axes=plt.subplots(2,2,figsize=(15,8),constrained_layout=True)
    vmax=max(float(np.abs(raw).max()),float(np.abs(interaction).max()))
    for col,(name,field) in enumerate([('double-condition change',raw),('interaction remainder',interaction)]):
        im=axes[0,col].imshow(field,aspect='auto',cmap='coolwarm',vmin=-vmax,vmax=vmax,interpolation='nearest')
        axes[0,col].set_title(name+' / raw mean');fig.colorbar(im,ax=axes[0,col])
        scale=np.sqrt(np.mean(field**2,axis=1,keepdims=True))
        norm=np.divide(field,scale,out=np.zeros_like(field),where=scale>0)
        im=axes[1,col].imshow(norm,aspect='auto',cmap='coolwarm',vmin=-5,vmax=5,interpolation='nearest')
        axes[1,col].set_title(name+' / row RMS normalized; color clipped at +/-5');fig.colorbar(im,ax=axes[1,col])
    for ax in axes.flat:ax.set_xlabel('Native coordinate (unsorted, all included)');ax.set_ylabel('Hidden-state output index')
    fig.suptitle(f'{side}: joint holdout, {len(groups)} groups. Last row post-final norm; index0 embedding. Mean can cancel signs.')
    fig.savefig(dest/'full_coordinate_interactions.png',dpi=160);plt.close(fig)
    np.savez(dest/'heatmap_values.npz',raw_mean=raw,interaction_mean=interaction)
    write(dest/'heatmap_metadata.json',dict(aggregation=f'mean over {len(groups)} joint-holdout groups',coordinate_order='native unchanged',
        raw_scale=[-vmax,vmax],normalized_scale=[-5,5],normalized='each displayed row divided by its RMS, color saturation only',
        boundary='Last row post-final norm; raw last-block output retained separately in chunk h2; not all token positions.'))


def material_audit():
    names=set(['Arin','Bela','Ciro','Dena','Eron','Fara','Galen','Hana','Iven','Jora','Kelan','Luma','Miro','Nera','Orin','Pela'])
    audit={}
    for label,path in [('primary',OUT/'material.json'),('length_control',OUT/'length_control/material.json')]:
        if not path.exists():continue
        rows=json.loads(path.read_text(encoding='utf-8'))['rows'];parts={}
        for family in ['category','role','chain']:
            found={}
            for split in ['development','entity_holdout']:
                words=set(word for r in rows if r['family']==family and r['split']==split for word in re.findall(r'\b\w+\b',r['text']))
                found[split]=words&names
            shared=found['development']&found['entity_holdout']
            parts[family]=dict(development=sorted(found['development']),heldout=sorted(found['entity_holdout']),
                              overlap=sorted(shared),strict_named_entity_disjoint=not shared)
        audit[label]=dict(material_sha256=sha(path),families=parts,unique_texts=len(set(r['text'] for r in rows)),n_rows=len(rows))
    audit['correction']='Primary role/chain split is unseen focal configuration, NOT strict unseen vocabulary. Preserved original material. Diagnostic control repairs participant split before its model run.'
    write(OUT/'material_audit.json',audit)


def main():
    material_audit()
    results={}
    for side in ['4B','14B']:
        result=summarize(side)
        if result:results[side]=result
    ledger_path=OUT/'evidence/claims.json'
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
        for claim in ledger['claims']:
            if claim['id'] in ['RDC-M01','RDC-M02','RDC-M03']:
                claim['replacement_results']={side:dict(path=f'{side}/measurement_repair.json',sha256=sha(OUT/side/'measurement_repair.json')) for side in results}
                claim['replacement_status']='corrected_measurement_available; original claim status retained'
        ledger['claims'].append(dict(id='RDC-N01',title='未见实体/表述的条件组合预测',status='heldout_measurement_available',
            reason='逐模型看误差与基线，不能用局部成功声称整个语言理论完成。',
            results={s:f'{s}/summary.json' for s in results},downstream_claim_ids=['RDC-T01'])) if not any(c['id']=='RDC-N01' for c in ledger['claims']) else None
        write(ledger_path,ledger)
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
    status=('completed_bounded_phase' if (OUT/'length_control/4B/summary.json').exists() else 'primary_models_complete_controls_pending') if len(results)==2 else 'partial_models'
    index.update(status=status,
        models={s:{'capture':True,'measurement':True,'composition':True,'summary':f'{s}/summary.json'} for s in results},
        claims='evidence/claims.json')
    write(OUT/'index.json',index)
    print(json.dumps({s:dict(replay=r['old_wrong_statistic_replay_max_abs'],
        composition={k:{m:round(v['median'],4) for m,v in z['errors'].items()} for k,z in r['composition'].items()}) for s,r in results.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plots',choices=['4B','14B']);p.add_argument('--control',action='store_true');args=p.parse_args()
    if args.control:OUT=OUT/'length_control'
    if args.plots:plots(args.plots)
    elif args.control:
        result=summarize('4B');print(json.dumps(result['composition']))
    else:main()
