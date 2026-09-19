"""Whole-axis descriptive runtime accounting, never top-k or semantic labels."""
from collections import Counter
from rdc_construction_common import *

OUT=BASE/'phase2746/runtime_analysis'


def moments(shape):
    return np.zeros((2,4,*shape),dtype=np.float64)


def add(acc,value,weight):
    # Horizons: first emitted-token prediction; within-row first three steps.
    # Four moments: raw mean/raw second; whole-vector RMS mean/RMS second.
    normalized=value/np.sqrt(np.mean(value*value,-1,keepdims=True)).clip(1e-12)
    for h,t in enumerate([1,3]):
        for m,a in enumerate([value[:t],value[:t]**2,normalized[:t],normalized[:t]**2]):
            acc[h,m]+=weight*a.mean(0)


def main():
    start=time.monotonic();guard(900*1024**2)
    protocol={'timestamp':stamp(),'source':snapshot(__file__),
        'input':{'path':'phase2746/runtime/records.json.gz','sha256':sha(BASE/'phase2746/runtime/records.json.gz')},
        'scope':'Exploratory observation after capture. Source/corpus/family/outcome labels do not establish internal semantic modules or causal effects.',
        'moments':'Every native coordinate/unit. Raw mean/second moment and whole-vector-RMS mean/second moment; float64 accumulation and persistence. No sorting, compression of axes, PCA or top-k.',
        'horizons':['first_prediction','within_row_first3_predictions'],
        'weighting':'Equal source_group within each family; equal windows within each source. Each row first3average has equal row mass, not three independent documents.',
        'time_scope':'Every own-history step has all-H energy/RMS/cosine statistics; all retained up-to8steps have write/cross/rounding and full-head/source attention summaries. All896rows actually have at least3steps.',
        'identity':'Hnext=H+attention_write+MLP_write+qround; full energy includes every cross term. Stored original BF16 operands decoded then contracted in FP64.',
        'comparisons':'First0 and first3 horizons are identical for natural/controlled. Beyond3 controlled histories ended, so no nonexistent controlled field is filled with zero.',
        'reuse_boundary':'Same unit index compared only within one layer; same-index cross-layer vectors are not asserted functionally aligned.'}
    if (OUT/'protocol.json').exists():
        previous=read(OUT/'protocol.json')
        assert {k:v for k,v in previous.items() if k not in ['timestamp','source']}=={k:v for k,v in protocol.items() if k not in ['timestamp','source']}
        protocol=previous
    else:immutable(OUT/'protocol.json',protocol)
    records=gzread(BASE/'phase2746/runtime/records.json.gz')
    assert len(records)==896 and all(len(r['generated_ids'])>=3 for r in records)
    families=sorted({r['family'] for r in records});row_index=[];summaries=[];all_reports=[]
    for family in families:
        selected=[r for r in records if r['family']==family]
        docs=Counter(r['source_group'] for r in selected)
        unit=moments((36,3,9728));coord=moments((36,5,2560))
        units_lag=np.zeros((5,36,3,9728),dtype=np.float64);lagmass=0.0
        for r in selected:
            weight=1/(len(docs)*docs[r['source_group']]);file=BASE/r['field_path']
            assert sha(file)==r['field_sha256']
            with np.load(file) as z:
                u=unbits(z['units']).astype(float);c=unbits(z['coordinates']).astype(float)
                h=unbits(z['hidden']).astype(float);post=unbits(z['postnorm']).astype(float)
                assert u.shape[0]==r['full_field_steps'] and len(h)==len(r['generated_ids'])
                assert z['generated_ids'].tolist()==r['generated_ids']
                add(unit,u,weight);add(coord,c,weight)
                # Equal row weight over available adjacent pairs; no claim that
                # chronological steps are independent language samples.
                a,b=u[:-1],u[1:]
                for j,term in enumerate([a,b,a*a,b*b,a*b]):
                    units_lag[j]+=weight*term.mean(0)
                lagmass+=weight
                steps=[]
                for t in range(len(h)):
                    hs=h[t];entry={'step':t,'emitted_token_id':r['generated_ids'][t],
                        'hidden_RMS':np.sqrt(np.mean(hs*hs,-1)).tolist(),
                        'postnorm_RMS':float(np.sqrt(np.mean(post[t]**2))),
                        'native_entropy':float(z['statistics'][t,0]),'chosen_probability':float(z['statistics'][t,1])}
                    if t:
                        prev=h[t-1];entry['same_layer_lag_cosine']=(np.sum(hs*prev,-1)/np.sqrt(np.sum(hs*hs,-1)*np.sum(prev*prev,-1)).clip(1e-30)).tolist()
                    if t<len(u):
                        original=hs[:-1];attention=c[t,:,1];mlp=c[t,:,4];native=hs[1:]
                        rounded=native-original-attention-mlp
                        terms=np.stack([original,attention,mlp,rounded])
                        gram=np.einsum('ald,bld->lab',terms,terms)/2560
                        direct=np.mean(native*native,-1)
                        identity_error=float(abs(gram.sum((1,2))-direct).max())
                        assert identity_error<1e-8*max(1.,float(direct.max()))
                        entry.update(full_write_energy_gram=gram.tolist(),rounding_max_abs=float(abs(rounded).max()),
                            energy_identity_max_error=identity_error,
                            write_relative_RMS=np.sqrt(np.mean(mlp*mlp,-1)/np.mean(original*original,-1).clip(1e-30)).tolist())
                        p=unbits(z['attention_step'+str(t)]).astype(float)
                        mass=p.sum(-1);pn=p/mass[...,None].clip(1e-30)
                        length=len(r['actual_prompt_ids']);assert p.shape[-1]==length+t
                        distances=np.arange(p.shape[-1]-1,-1,-1,dtype=float)
                        entry['attention']={'raw_mass':mass.tolist(),
                            'entropy_after_mass_normalization':(-np.sum(pn*np.log(pn.clip(1e-300)),-1)).tolist(),
                            'generated_history_mass':pn[...,length:].sum(-1).tolist(),
                            'expected_token_distance':(pn*distances).sum(-1).tolist(),
                            'all_source_positions':p.shape[-1]}
                    steps.append(entry)
                summary={k:r[k] for k in ['sample_id','source_group','kind','family','split','EOS','censored']}
                summary.update(weight_within_family=weight,steps=steps)
                if r['kind']=='controlled':summary['correct_and_stopped']=r['answer_scoring']['parsed_and_stopped_correct']
                summaries.append(summary)
                row_index.append({'sample_id':r['sample_id'],'source_group':r['source_group'],'family':family,
                    'field_path':r['field_path'],'field_sha256':r['field_sha256'],'weight_within_family':weight})
        assert abs(lagmass-1)<1e-10
        am,bm,aa,bb,ab=units_lag
        variance_a=np.maximum(aa-am*am,0);variance_b=np.maximum(bb-bm*bm,0)
        denom=np.sqrt(variance_a*variance_b);valid=denom>1e-30
        correlation=np.zeros_like(denom);np.divide(ab-am*bm,denom,out=correlation,where=valid)
        assert np.max(abs(correlation[valid]))<=1+1e-9
        # Zero means undefined only where accompanied by this explicit mask.
        path=OUT/'all_axes'/(family+'.npz')
        npz(path,unit_moments=unit,coordinate_moments=coord,unit_lag_moments=units_lag,
            unit_lag_correlation=correlation,unit_lag_correlation_valid=valid)
        report={'family':family,'rows':len(selected),'source_groups':len(docs),'all_axes_path':path.relative_to(BASE).as_posix(),
            'all_axes_sha256':sha(path),'unit_fields':['gate','up','product'],
            'coordinate_fields':['attention_input','attention_write','pre_MLP','MLP_input','MLP_write'],
            'moment_axes':['horizon','raw_mean/raw_second/RMS_mean/RMS_second','block','field','native_coordinate_or_unit'],
            'product_lag_median_by_layer':np.median(correlation[:,2],-1).tolist(),
            'fraction_product_units_positive_lag_by_layer':np.mean((correlation[:,2]>0)&valid[:,2],-1).tolist(),
            'undefined_lag_counts':np.sum(~valid,axis=(-1,-2)).tolist()}
        all_reports.append(report)
        print('RUNTIME_ANALYSIS_FAMILY',family,len(selected),round(time.monotonic()-start,1),flush=True)
        del unit,coord,units_lag,u,c,h,post
    compressed(OUT/'trajectory_statistics.json.gz',summaries)
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'protocol':protocol,
        'families':all_reports,'row_index':row_index,'rows':len(records),
        'actual_generated_steps':sum(len(r['steps']) for r in summaries),
        'maximum_energy_identity_error':max(s.get('energy_identity_max_error',0) for r in summaries for s in r['steps']),
        'statistics_sha256':sha(OUT/'trajectory_statistics.json.gz'),'seconds':time.monotonic()-start,
        'scope':'Full-axis observed reuse/cross terms; association and dynamics summaries, not a extracted mechanism or evidence of a semantic cause.'}
    save(OUT/'result.json',result);ledger('phase2746_runtime_analysis',result['seconds'])
    print('RUNTIME_ANALYSIS_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    start=time.monotonic()
    try:main()
    except Exception as exc:
        failure(OUT,start,exc);raise
