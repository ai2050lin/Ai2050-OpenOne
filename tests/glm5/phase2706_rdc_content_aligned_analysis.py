"""Function-boundary model comparisons, full-coordinate maps, and non-isomorphism caveats."""
import gc
from rdc_conditional_common import *
from rdc_conditional_estimators import FullKernel,binary_report,group_ids
from phase2704_rdc_predictive_gates import gate_fit,errors
OUT=CAMPAIGN/'l_aligned'


def features(key):
    out=OUT/key;rows=read(out/'prefixes.json');assert len(list((out/'prefix_commits').glob('*.json')))==256
    path=out/'features.npz';behavior=[read(out/f'behavior/{r["sample_id"]}.json') for r in rows]
    assert all(b['content_state'] is not None for b in behavior),'Incomplete functional alignment: report missing content before fitting'
    if path.exists():
        with np.load(path) as z:return rows,behavior,{k:z[k] for k in z.files}
    data={}
    for r,b in zip(rows,behavior):
        for stage,sid in [('prefill',r['sample_id']+'-s0'),('content',b['content_state'])]:
            with np.load(out/f'fields/{sid}.npz') as z:
                data.setdefault(stage+'_h',[]).append(z['h_c'])
                for k in z.files:
                    if not k.startswith('L'):continue
                    a=z[k]
                    if k.endswith('_gate_up'):
                        gate,up=np.split(a,2)
                        data.setdefault(stage+'_'+k.replace('gate_up','gate'),[]).append(gate)
                        data.setdefault(stage+'_'+k.replace('gate_up','up'),[]).append(up)
                    else:data.setdefault(stage+'_'+k,[]).append(a)
    data={k:np.stack(v) for k,v in data.items()};npz(path,**data)
    return rows,behavior,data


def main():
    immutable(OUT/'analysis_protocol.json',{'phase':2706,'source_sha':sha(Path(__file__)),'estimator_sha':sha(ROOT/'tests/glm5/rdc_conditional_estimators.py'),
      'reading':'AllnativeCcoordinates at allcheckpoints, separate prefill/content. Training64 validation64 test128 byentity0/8/12,13. Nativebehavior paired over256sharedrecords, no filtering bycorrectness.',
      'native':'Weighted global/lang/familylang16 all-unit reconstruction permodel at declarednativecheckpoints. Compare frozenprefill coefficients appliedatcontent, alongside content-fitted coefficients; same-block observedg/up, not forecasts.',
      'crossmodel':'Qwen4 H24 -> Qwen14/GLM H27 fullcoordinates, linear/quadratic. Compare prefill-to-prefill, content-to-content and frozenprefill-fit-to-content testing with unseenentities. Globalmean and16familylangmean controls. Nativeindices are never directly equated.',
      'limits':['Models need not share a coordinate basis.','Four sharedentitygroups,onlytwoheldout; no broadpopulationconfidence claim.','Crossmodel maps have atmost64training-sample kernelrank, though allinput/output nativecoordinates used.','First lexicaltoken is not completeanswer orEOS.','Content-boundary alignment is assigned after each model generates; aligned mappings are conditional observational maps, not an independently deployable advance selection of the target-model timestep. Fixed-prefill mapping has no such timestep-selection issue.','Functional stateprediction does not establish computation homomorphism or uniqueprogram equivalence.']})
    allresults={};reference_behavior=None
    for key in ('qwen4','qwen14','glm4'):
        out=OUT/key;rows,b,data=features(key);tr,va,te=splits(rows)
        assert tuple(map(len,(tr,va,te)))==(64,64,128)
        y=np.array([[r['fact_truth'],r['expected_yes']] for r in rows],np.float32);results=[];gates=[]
        for stage in ('prefill','content'):
            h=unbits(data[stage+'_h'])
            for l in range(h.shape[1]):
                f=FullKernel(h[:,l],tr,va,te);p,m=f.fit(y,out/f'models/{stage}_H{l}.npz')
                results.append({'stage':stage,'H':l,**m,**binary_report(y,p,te,rows)})
                npz(out/f'predictions/{stage}_H{l}.npz',prediction=p,test=te)
            layers=sorted({int(k.split('_')[1][1:]) for k in data if k.startswith(stage+'_L')})
            for l in layers:
                g,u,a=[unbits(data[f'{stage}_L{l}_{k}']).astype(np.float64) for k in ('gate','up','a')]
                energy=np.mean(a[tr]*a[tr],0)
                for mode in ('global','language','family_language'):
                    group=group_ids(rows,mode);p,c,zero=gate_fit(g,u,a,tr,group,True)
                    m,arr=errors(a[te],p[te],energy);gates.append({'stage':stage,'layer':l,'mode':mode,'kind':'same_stage_fit',**m})
                    npz(out/f'unit_errors/{stage}_L{l}_{mode}.npz',**arr,coefficient=c,groups=np.unique(group),test=te)
                    if stage=='content':
                        with np.load(out/f'unit_errors/prefill_L{l}_{mode}.npz') as z:old=z['coefficient']
                        transferred=g[te]*u[te]*old[group[te]];m,arr=errors(a[te],transferred,energy)
                        gates.append({'stage':stage,'layer':l,'mode':mode,'kind':'frozen_prefill_fit',**m})
                        npz(out/f'unit_errors/content_L{l}_{mode}_frozen_prefill.npz',**arr,test=te)
                del g,u,a,p;gc.collect()
        summary={'model':key,'prefixes':256,'unique_states':len(read(out/'material.json')),
          'first_token_correct':sum(bb['first_token_correct'] for bb in b),'first_content_correct':sum(bb['content_token_correct'] for bb in b),
          'content_candidate_pair_correct':sum(bb['content_candidate_pair_correct'] for bb in b),
          'content_step_counts':{str(s):sum(bb['content_step']==s for bb in b) for s in sorted({bb['content_step'] for bb in b})},
          'behavior_by_family':{family:{'n':sum(r['family']==family for r in rows),
            'first_correct':sum(bb['first_token_correct'] for r,bb in zip(rows,b) if r['family']==family),
            'content_correct':sum(bb['content_token_correct'] for r,bb in zip(rows,b) if r['family']==family)} for family in sorted({r['family'] for r in rows})},
          'readers':results,'gates':gates,'limits':read(OUT/'analysis_protocol.json')['limits']}
        save(out/'result.json',summary);allresults[key]=summary
        print('ALIGNED_ANALYSIS',key,summary['first_content_correct'],flush=True)
        announce('l_aligned_'+key,state='analysis_complete',completed=256,total=256)
        del data;gc.collect()
    with np.load(OUT/'qwen4/features.npz') as z:q4={s:unbits(z[s+'_h'][:,24]) for s in ('prefill','content')}
    rows=read(OUT/'qwen4/prefixes.json');tr,va,te=splits(rows);group=group_ids(rows,'family_language');maps=[]
    for key in ('qwen14','glm4'):
        with np.load(OUT/key/'features.npz') as z:target={s:unbits(z[s+'_h'][:,27]) for s in ('prefill','content')}
        for fitstage,teststage in [('prefill','prefill'),('content','content'),('prefill','content')]:
            x=np.concatenate([q4[fitstage],q4[teststage]],0);y=np.concatenate([target[fitstage],target[teststage]],0);test=te+256
            for kind in ('linear','quadratic'):
                f=FullKernel(x,tr,va,test,kind);mid=f'{key}_{fitstage}_to_{teststage}_{kind}'
                p,m=f.fit(y,OUT/f'models/{mid}.npz');report,arr=errors(y[test],p,np.mean(y[tr].astype(np.float64)**2,0))
                maps.append({'target_model':key,'fit_stage':fitstage,'test_stage':teststage,'algorithm':kind,**m,**report})
                npz(OUT/f'predictions/{mid}.npz',prediction=p,target=y[test],test=te,**arr)
            for mode in ('global_mean','family_language_mean'):
                means={v:y[tr[group[tr]==v]].mean(0) for v in range(16)}
                p=np.repeat(y[tr].mean(0)[None],len(te),0) if mode=='global_mean' else np.stack([means[int(group[i])] for i in te])
                report,arr=errors(y[test],p,np.mean(y[tr].astype(np.float64)**2,0));maps.append({'target_model':key,'fit_stage':fitstage,'test_stage':teststage,'algorithm':mode,**report})
        print('CROSSMODEL_MAP',key,flush=True)
    save(OUT/'result.json',{'phase':2706,'timestamp':stamp(),'models':allresults,'crossmodel_maps':maps,'limits':read(OUT/'analysis_protocol.json')['limits']})


if __name__=='__main__':main()
