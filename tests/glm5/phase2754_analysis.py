"""World-level approximate regularities and actual native-source accounting."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','8');os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse,json
from pathlib import Path
import numpy as np
from phase2754_relation_stability import ROOT,OUT,FAMILIES,decode,write,sha,now,snapshot

FACTORS=('constant','orientation','query_role','negation','predicate','orientation_negation','role_negation','truth')
PHI=np.array([[1,t,r,e,t*r,t*e,r*e,t*r*e] for t in (-1,1) for r in (-1,1) for e in (-1,1)],dtype=float)

def source():return {r['id']:r for r in json.loads((OUT/'material.json').read_text(encoding='utf-8'))['rows']}
def metadata(side,stage):return [r for p in sorted((OUT/side/stage).glob('chunk_*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]

def bootstrap(records,key,seed=2754003):
    rng=np.random.default_rng(seed);draws=[];allmeans=[]
    for fam in FAMILIES:
        rr=[r for r in records if r['family']==fam]
        if not rr:continue
        worlds=sorted({r['world'] for r in rr});means=np.array([np.mean([r[key] for r in rr if r['world']==w]) for w in worlds])
        draws.append(means[rng.integers(len(means),size=(2000,len(means)))].mean(1));allmeans.extend(means)
    return dict(mean=float(np.mean(allmeans)),ci95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist(),worlds=len(allmeans))

def analyze(side):
    src=source();stages=['discovery','confirmation'] if side=='4B' else ['confirmation']
    meta=[r for stage in stages for r in metadata(side,stage)]
    groups={}
    for m in meta:groups.setdefault(m['group'],[]).append(m)
    records=[];attribution={};max_identity=0
    for gid,rr in groups.items():
        rr=sorted(rr,key=lambda r:r['cell']);assert [r['cell'] for r in rr]==list(range(8))
        annotations=[src[r['id']] for r in rr];r0=annotations[0]
        margin=np.array([r['margin'] for r in rr]);coeff=PHI.T@margin/8
        assert np.allclose(PHI@coeff,margin,atol=1e-9)
        truth=PHI[:,7];native_ok=np.array([m['prediction_text'].strip().lower()==a['expected'] for m,a in zip(rr,annotations)])
        signed_pairs=np.array([PHI[4+i,7]*(margin[4+i]-margin[i])/2 for i in range(4)])
        record=dict(group=gid,world=r0['world'],family=r0['family'],split=r0['split'],template=r0['template'],
            native_accuracy=float(native_ok.mean()),canonical_accuracy=float(np.mean(np.sign(margin)==truth)),
            restricted_mass_accuracy=float(np.mean([np.sign(m['yesno_logmass']['yes']-m['yesno_logmass']['no'])==a['truth_sign'] for m,a in zip(rr,annotations)])),
            paired_ranking=float(np.mean((signed_pairs>0)+.5*(signed_pairs==0))),truth_coefficient=float(coeff[7]),predicate_coefficient=float(coeff[4]),bias=float(coeff[0]),
            affirmed_gain=float(coeff[7]+coeff[4]),negated_gain=float(coeff[7]-coeff[4]),truth_effect_positive=float(coeff[7]>0),all_cells_correct=float(native_ok.all()),
            coefficients=coeff.tolist())
        records.append(record)
        rms=np.array([m['rms'] for m in rr]);parts={}
        for kind in ('attention','mlp','rounding'):
            parts[kind]=np.array([m['source_numerator'][kind] for m in rr])/rms[:,None]
        initial=np.array([m['source_numerator']['initial'] for m in rr])/rms
        normerr=np.array([m['norm_rounding_margin'] for m in rr]);logiterr=margin-np.array([m['fp32_margin'] for m in rr])
        rebuilt=initial+sum(v.sum(1) for v in parts.values())+normerr+logiterr
        max_identity=max(max_identity,float(np.max(abs(rebuilt-margin))))
        contribution={k:truth@v/8 for k,v in parts.items()}
        contribution.update(initial=np.array(truth@initial/8),norm_rounding=np.array(truth@normerr/8),readout_rounding=np.array(truth@logiterr/8))
        attribution[gid]=contribution
    assert max_identity<1e-7
    report={}
    for split in sorted({r['split'] for r in records}):
        rr=[r for r in records if r['split']==split]
        entry={k:bootstrap(rr,k) for k in ('native_accuracy','canonical_accuracy','restricted_mass_accuracy','paired_ranking','truth_coefficient','truth_effect_positive','affirmed_gain','negated_gain','all_cells_correct')}
        entry['mean_bias']=float(np.mean([r['bias'] for r in rr]));entry['mean_absolute_bias']=float(np.mean([abs(r['bias']) for r in rr]))
        entry['families']={f:{k:float(np.mean([r[k] for r in rr if r['family']==f])) for k in ('native_accuracy','paired_ranking','truth_coefficient','affirmed_gain','negated_gain')} for f in FAMILIES}
        pairs=[]
        for wid in sorted({r['world'] for r in rr}):
            ww=sorted([r for r in rr if r['world']==wid],key=lambda r:r['template'])
            if len(ww)==2:pairs.append([ww[0]['truth_coefficient'],ww[1]['truth_coefficient']])
        if pairs:
            vv=np.array(pairs);entry['within_world_template_truth_effect']=dict(worlds=len(vv),sign_agreement=float(np.mean(np.sign(vv[:,0])==np.sign(vv[:,1]))),both_positive=float(np.mean(np.all(vv>0,axis=1))),
                pearson=float(np.corrcoef(vv.T)[0,1]),mean_absolute_difference=float(np.mean(abs(vv[:,0]-vv[:,1]))))
        report[split]=entry
    write(OUT/side/'regularity_groups.json',records)
    write(OUT/side/'regularity_summary.json',dict(created_utc=now(),source=snapshot(Path(__file__)),splits=report,accounting_max_abs_error=max_identity,
        scope='All material outcomes retained. Directional effects and template stability do not require perfect model outputs. Native source projections are attribution, not isolated causal interventions.'))
    # Layer-resolved truth contributions averaged within split/family, including rounding.
    arrays={};keys=[]
    for split in report:
        for fam in FAMILIES:
            gg=[r['group'] for r in records if r['split']==split and r['family']==fam]
            if not gg:continue
            keys.append(dict(split=split,family=fam))
            for k in next(iter(attribution.values())):arrays.setdefault(k,[]).append(np.mean([attribution[g][k] for g in gg],axis=0))
    np.savez(OUT/side/'source_profiles.npz',**{k:np.stack(v) for k,v in arrays.items()});write(OUT/side/'source_profile_rows.json',keys)
    print(side,{s:{k:report[s][k]['mean'] for k in ('native_accuracy','paired_ranking','truth_coefficient')} for s in report},flush=True)

def fields(side):
    src=source();stages=['discovery','confirmation'] if side=='4B' else ['confirmation'];sums={};counts={};cosrows=[];train_sum={};train_count={}
    dest=OUT/side/'factor_fields';dest.mkdir(exist_ok=True)
    for stage in stages:
        for p in sorted((OUT/side/stage).glob('chunk_*.npz')):
            meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(p) as z:h=decode(z['hidden'])
            assert len(h)%8==0
            labels=[]
            hh=h.reshape(-1,8,*h.shape[1:]);coef=np.einsum('ck,gcld->gkld',PHI,hh,optimize=True)/8
            for g in range(len(hh)):
                rr=[src[m['id']] for m in meta[g*8:g*8+8]];assert [r['cell'] for r in rr]==list(range(8)) and len({r['group'] for r in rr})==1
                r=rr[0];key=(r['split'],r['family']);labels.append(dict(group=r['group'],world=r['world'],split=r['split'],family=r['family']))
                if key not in sums:sums[key]=np.zeros_like(coef[g]);counts[key]=0
                sums[key]+=coef[g];counts[key]+=1
                if r['split']=='train':
                    f=r['family']
                    if f not in train_sum:train_sum[f]=np.zeros_like(coef[g]);train_count[f]=0
                    train_sum[f]+=coef[g];train_count[f]+=1
                elif side=='4B':
                    mu=train_sum[r['family']]/train_count[r['family']]
                    for k in (4,7):
                        x=coef[g,k];y=mu[k];den=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)
                        cos=np.divide(np.sum(x*y,axis=-1),den,out=np.zeros_like(den),where=den>1e-10)
                        err=np.divide(np.linalg.norm(x-y,axis=-1),np.linalg.norm(x,axis=-1),out=np.zeros_like(den),where=np.linalg.norm(x,axis=-1)>1e-10)
                        cosrows.append(dict(**labels[-1],factor=FACTORS[k],cos=cos.tolist(),relative_error=err.tolist(),nonzero=(den>1e-10).tolist()))
            np.savez(dest/f'{stage}_{p.name}',predicate=coef[:,4].astype(np.float32),truth=coef[:,7].astype(np.float32));write(dest/f'{stage}_{p.stem}.json',labels)
    keys=sorted(sums);np.savez(OUT/side/'native_factor_means.npz',means=np.stack([sums[k]/counts[k] for k in keys]).astype(np.float32))
    write(OUT/side/'native_factor_metadata.json',dict(rows=[dict(split=k[0],family=k[1],groups=counts[k]) for k in keys],factors=FACTORS,
        coordinates='All original coordinates, unsorted; eight-condition Walsh coefficients are definitions.',boundary='0 embedding;1..L-1 residual;L finalnorm;L+1 raw final',storage='Per-group predicate/truth fields retained; full eight factor mean fields retained; original native fields permit all others to be recomputed.'))
    if side=='4B':write(OUT/side/'field_generalization.json',cosrows)

def mechanism_summary():
    rr=json.loads((OUT/'mechanism_rows.json').read_text(encoding='utf-8'));modes=('baseline','erase_probe','equal_norm_random');summary={}
    worlds=sorted({r['world'] for r in rr});world_records=[]
    for w in worlds:
        ww=[r for r in rr if r['world']==w];r0=ww[0];d=dict(world=w,family=r0['family'],split=r0['split'])
        for mode in modes:
            xx=[r for r in ww if r['mode']==mode]
            d[mode+'_truth_effect']=float(np.mean([r['truth_sign']*r['margin'] for r in xx]));d[mode+'_accuracy']=float(np.mean([r['correct'] for r in xx]))
        for mode in modes[1:]:
            d[mode+'_effect_change']=d[mode+'_truth_effect']-d['baseline_truth_effect'];d[mode+'_accuracy_change']=d[mode+'_accuracy']-d['baseline_accuracy']
        d['erase_minus_random_effect']=d['erase_probe_truth_effect']-d['equal_norm_random_truth_effect'];world_records.append(d)
    for split in ['all','entity','surface','depth']:
        ww=world_records if split=='all' else [r for r in world_records if r['split']==split]
        summary[split]={k:bootstrap(ww,k) for k in ww[0] if k not in ('world','family','split')}
    checks={}
    for mode in modes:
        xx=[r for r in rr if r['mode']==mode]
        checks[mode]=dict(mean_intended_norm=float(np.mean([r['edit']['intended_norm'] for r in xx])),mean_realized_norm=float(np.mean([r['edit']['realized_norm'] for r in xx])),
            mean_relative_norm=float(np.mean([r['edit']['relative_norm'] for r in xx])),mean_absolute_probe_after=float(np.mean([abs(r['edit']['probe_after']) for r in xx])))
    baseline=[r for r in rr if r['mode']=='baseline'];layers=json.loads((OUT/'mechanism_selection.json').read_text(encoding='utf-8'))['unit_layers_zero_based'];unitfields={};unitmetrics={}
    with np.load(OUT/'unit_activations.npz') as u,np.load(OUT/'unit_parameter_coefficients.npz') as c:
        for layer in layers:
            k=str(layer);contrib=u[k].astype(float)*c[k][None,:]/np.array([r['rms'] for r in baseline])[:,None]
            means=[]
            for split in ('entity','surface','depth'):
                for fam in FAMILIES:
                    ix=[i for i,r in enumerate(baseline) if r['split']==split and r['family']==fam];yy=np.array([baseline[i]['truth_sign'] for i in ix])
                    means.append(np.mean(contrib[ix]*yy[:,None],axis=0))
            unitfields[k]=np.stack(means)
            delta=np.array([r['unit_projection_check'][k]['difference']/r['rms'] for r in baseline]);norm=np.array([r['unit_projection_check'][k]['native_mlp_numerator']/r['rms'] for r in baseline])
            unitmetrics[k]=dict(units=contrib.shape[1],native_vs_fp32_projection_max=float(abs(delta).max()),native_vs_fp32_projection_rms=float(np.sqrt(np.mean(delta**2))),native_projection_rms=float(np.sqrt(np.mean(norm**2))))
    np.savez(OUT/'all_unit_truth_contributions.npz',**unitfields)
    write(OUT/'mechanism_summary.json',dict(created_utc=now(),source=snapshot(Path(__file__)),splits=summary,perturbation_checks=checks,units=unitmetrics,
        unit_rows=[dict(split=s,family=f) for s in ('entity','surface','depth') for f in FAMILIES],
        scope='Matched24worlds; label-blind edits of frozen semantic probe, one fixed norm-matched orthogonal random direction. Not a unique or minimal circuit proof.'))
    print(json.dumps(summary['all'],indent=2),flush=True)

def compare():
    src=source();m4={r['id']:r for r in metadata('4B','confirmation')};m14=metadata('14B','confirmation')
    records=[]
    for m in m14:
        a=src[m['id']];b=m4[m['id']]
        records.append(dict(world=a['world'],family=a['family'],accuracy_diff=float(m['prediction_text'].strip().lower()==a['expected'])-float(b['prediction_text'].strip().lower()==a['expected']),
            canonical_accuracy_diff=float(np.sign(m['margin'])==a['truth_sign'])-float(np.sign(b['margin'])==a['truth_sign']),
            restricted_mass_accuracy_diff=float(np.sign(m['yesno_logmass']['yes']-m['yesno_logmass']['no'])==a['truth_sign'])-float(np.sign(b['yesno_logmass']['yes']-b['yesno_logmass']['no'])==a['truth_sign'])))
    a4={r['group']:r for r in json.loads((OUT/'4B/regularity_groups.json').read_text(encoding='utf-8'))}
    groups14=json.loads((OUT/'14B/regularity_groups.json').read_text(encoding='utf-8'));gr=[]
    for a in groups14:
        b=a4[a['group']];gr.append(dict(world=a['world'],family=a['family'],ranking_diff=a['paired_ranking']-b['paired_ranking'],coefficient_sign_agreement=float(np.sign(a['truth_coefficient'])==np.sign(b['truth_coefficient']))))
    write(OUT/'matched_model_comparison.json',dict(created_utc=now(),worlds=24,prompts=192,native_accuracy_14B_minus4B=bootstrap(records,'accuracy_diff'),
        canonical_accuracy_14B_minus4B=bootstrap(records,'canonical_accuracy_diff'),restricted_mass_accuracy_14B_minus4B=bootstrap(records,'restricted_mass_accuracy_diff'),
        paired_ranking_14B_minus4B=bootstrap(gr,'ranking_diff'),truth_coefficient_sign_agreement=bootstrap(gr,'coefficient_sign_agreement'),
        limits='Same inputs and BF16 eager protocol, but different trained checkpoints, scales and CPU offload. Small24world subset; cannot attribute differences solely to parameter count or infer a perfect large-model limit.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['regularity','fields','mechanism','compare']);p.add_argument('--model',choices=['4B','14B'],default='4B');a=p.parse_args()
    {'regularity':lambda:analyze(a.model),'fields':lambda:fields(a.model),'mechanism':mechanism_summary,'compare':compare}[a.mode]()
