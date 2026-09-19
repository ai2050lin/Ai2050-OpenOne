"""Model-specific full-coordinate readers on identical grouped cases; no native index matching."""
from rdc_continuity_common import *
from rdc_feature_extractors import fit_predict,metrics
OUT=CAMPAIGN/'h_scale'

def main():
    immutable(OUT/'analysis_protocol.json',{'source_sha':sha(Path(__file__)),'split':'For everymodel same unit0 train64,8validation64,12/13 test128; all8families andallfactorcells. Limited4units/family.',
      'reading':'Every checkpoint C full native hiddenwidth, A1 targets support and requestedanswer; extra directionC/norm/length at relative1/3,2/3,last checkpoints; no family-naming alignment of neurons.',
      'native_factor':'global/ language / family-language training mean sigmoid versus actuala, everycollectedMLPunit; same-block observedg/up inputs, not earlier-layer forecast.',
      'comparison_scope':'Same experimental cells, native modeltokenization/prompting and computational precision. No fullsequence generation in this scale phase.',
      'form_scope':'Trainunit0 andvalidation8 useform0; test12 form0 and13 form1. Form1 also changes voice or chain depth in somefamilies, so this is a joint shift, not independently crossed factors.'})
    all_results={}
    base=read(CAMPAIGN/'e_confirmation/material.json');selected=[i for i,r in enumerate(base) if r['unit'] in (0,8,12,13)]
    for model in ('qwen4','qwen14','glm4'):
      out=OUT/model
      if model=='qwen4':
        rows=[base[i] for i in selected]
        with np.load(CAMPAIGN/'e_confirmation/features/roles.npz') as z:h=z['matched'][selected,:,2]
        with np.load(CAMPAIGN/'e_confirmation/features/native.npz') as z:native={k:z[k][selected] for k in z.files if k.endswith(('_gate','_up','_a'))}
        behavior=[read(CAMPAIGN/f'e_confirmation/behavior/{r["sample_id"]}.json') for r in rows]
        correct_flags=[bool(b['matched_correct']) for b in behavior]
        pair_flags=[bool((int(np.argmax(b['matched_answer_logits']))==0)==r['expected_yes']) for b,r in zip(behavior,rows)]
        correct=sum(correct_flags);pair_correct=sum(pair_flags)
      else:
        rows=read(out/'material.json');assert len(list((out/'commits').glob('*.json')))==256
        hh=[];native={};behavior=[]
        for r in rows:
          with np.load(out/f'fields/{r["sample_id"]}.npz') as z:
            hh.append(unbits(z['h'][:,-1]))
            for k in z.files:
              if k.startswith('L') and k.endswith(('_gate','_up','_a')):
                a=unbits(z[k][0])
                if k.endswith('_gate_up'):
                    gate,up=np.split(a,2);native.setdefault(k.replace('gate_up','gate'),[]).append(gate);native.setdefault(k.replace('gate_up','up'),[]).append(up)
                else:native.setdefault(k,[]).append(a)
          behavior.append(read(out/f'behavior/{r["sample_id"]}.json'))
        h=np.stack(hh);native={k:np.stack(v) for k,v in native.items()}
        correct_flags=[bool(b['argmax_answer_correct']) for b in behavior];pair_flags=[bool(b['pair_correct']) for b in behavior]
        correct=sum(correct_flags);pair_correct=sum(pair_flags)
      tr,va,te=[np.array([i for i,r in enumerate(rows) if r['word_split']==s]) for s in ('train','validation','test')]
      assert (len(tr),len(va),len(te))==(64,64,128)
      results=[];ys={'support':np.eye(2)[[int(r['fact_truth']) for r in rows]],'answer':np.eye(2)[[int(r['expected_yes']) for r in rows]]}
      def fit(rep,target,blocks):
        m,p,params=fit_predict(blocks,ys[target],tr,va,te,'A1_linear',True)
        hit=p.argmax(1)==ys[target][te].argmax(1)
        by_form={str(form):{'n':int(sum(rows[i]['form']==form for i in te)), 'correct':int(sum(h for i,h in zip(te,hit) if rows[i]['form']==form))} for form in (0,1)}
        mid=f'{rep}_{target}';results.append(dict(split='same_base_heldout',target=target,representation=rep,algorithm='A1_linear',correct=int(np.sum(hit)),by_form=by_form,**m))
        npz(out/f'predictions/{mid}.npz',prediction=p,target=ys[target][te],test_indices=te)
        npz(out/f'models/{mid}.npz',**{k:v for k,v in params.items() if isinstance(v,np.ndarray)})
      for t in ys:
       for l in range(h.shape[1]):fit(f'H{l}_C',t,[h[:,l]])
       for l in (round((h.shape[1]-1)/3),round(2*(h.shape[1]-1)/3),h.shape[1]-1):
        norms=np.linalg.norm(h[:,l],axis=1)
        fit(f'H{l}_direction',t,[h[:,l]/np.maximum(norms[:,None],1e-12)])
        fit(f'H{l}_norm',t,[norms[:,None]])
       fit('length',t,[np.array([[len(r['prompt_ids'])] for r in rows])])
      factors=[]
      for l in sorted({int(k.split('_')[0][1:]) for k in native}):
        if f'L{l}_gate' not in native:continue
        g=native[f'L{l}_gate'].astype(np.float64);u=native[f'L{l}_up'].astype(np.float64);a=native[f'L{l}_a'].astype(np.float64);s=1/(1+np.exp(-g));preds={};unit_errors={}
        for mode in ('global','language','family_language'):
          key=lambda r:'all' if mode=='global' else r['language'] if mode=='language' else r['family']+'_'+r['language']
          means={v:s[[i for i in tr if key(rows[i])==v]].mean(0) for v in {key(r) for r in rows}}
          predicted=g[te]*u[te]*np.stack([means[key(rows[i])] for i in te]);err=predicted-a[te]
          preds[mode]={'all_unit_mse':float(np.mean(err**2)),'relative_to_actual_energy':float(np.sum(err**2)/np.sum(a[te]**2))}
          unit_errors[mode]=np.mean(err**2,axis=0)
        energy=np.mean(a[te]**2,axis=0)
        all_unit_audit={'family_language_better_than_global_units':int(np.sum(unit_errors['family_language']<unit_errors['global'])),
          'equal_error_units':int(np.sum(unit_errors['family_language']==unit_errors['global'])),
          'zero_energy_units':int(np.sum(energy==0)),
          'per_unit_relative_mse_quantiles':{mode:np.quantile(err[energy>0]/energy[energy>0],[0,.25,.5,.75,1]).tolist() for mode,err in unit_errors.items()}}
        npz(out/f'unit_errors/L{l}.npz',energy=energy,**unit_errors)
        factors.append({'layer':l,'units':a.shape[1],'comparisons':preds,'all_unit_audit':all_unit_audit})
      npz(out/'features.npz',h_C=h,**native)
      summary={'model':model,'cases':256,'hidden_width':h.shape[2],'layers':h.shape[1]-1,'first_argmax_correct':correct,'candidate_pair_correct':int(pair_correct),'results':results,'native_factors':factors,
        'behavior_cases':[{'sample_id':r['sample_id'],'base_id':r['base_id'],'unit':r['unit'],'family':r['family'],'first_correct':c,'pair_correct':p} for r,c,p in zip(rows,correct_flags,pair_flags)],
        'first_token_id_counts':{str(k):sum(b[('matched_argmax' if model=='qwen4' else 'first_argmax')]==k for b in behavior) for k in {b[('matched_argmax' if model=='qwen4' else 'first_argmax')] for b in behavior}},
        'initial_whitespace_argmax':None if model=='qwen4' else sum(b['actual_token'].isspace() for b in behavior),
        'limits':['One train baseunit per family; two heldout units.','Train/validation form0; test hasform0/form1, joint wording/voice/depth shifts rather than isolated form effect.','Model-specific probe fits; successful classification does not establish same physical/native algorithm.','No complete naturalanswer scoring; firstargmax andcandidatepair distinguished.','A whitespace firsttoken is a format prefix, not a demonstrated wrong semantic answer. Candidatepair preference at that step is not a measurement of the later semantic-answer position.']}
      save(out/'result.json',summary);all_results[model]=summary;print('SCALE_ANALYZED',model,flush=True)
      if model!='qwen4':announce('h_scale_'+model,state='analyzed',completed=256,total=256)
    save(OUT/'result.json',all_results)
    rng=np.random.default_rng(2702);reference=all_results['qwen4']['behavior_cases'];groups=sorted({r['base_id'] for r in reference})
    comparisons=[]
    for model in ('qwen14','glm4'):
      rows2=all_results[model]['behavior_cases'];assert [r['sample_id'] for r in rows2]==[r['sample_id'] for r in reference]
      for metric in ('first_correct','pair_correct'):
        delta=np.array([float(b[metric])-float(a[metric]) for a,b in zip(reference,rows2)])
        grouped=np.array([delta[[i for i,r in enumerate(reference) if r['base_id']==g]].mean() for g in groups])
        boot=grouped[rng.integers(0,len(groups),(4096,len(groups)))].mean(1)
        sensitivity={}
        for grouping in ('unit','family'):
          labels=sorted({r[grouping] for r in reference})
          means=np.array([delta[[i for i,r in enumerate(reference) if r[grouping]==g]].mean() for g in labels])
          b=means[rng.integers(0,len(labels),(4096,len(labels)))].mean(1)
          sensitivity[grouping]={'blocks':len(labels),'percentile_95_interval':np.quantile(b,[.025,.975]).tolist()}
        comparisons.append({'model_minus_qwen4':model,'metric':metric,'paired_case_difference':float(delta.mean()),'base_groups':len(groups),'cluster_percentile_95_interval':np.quantile(boot,[.025,.975]).tolist(),'shared_entity_and_family_sensitivity':sensitivity})
    save(OUT/'paired_behavior.json',{'comparisons':comparisons,'scope':'Exploratory paired behavior summary on all256 native forwards; bootstrap resamples32 family-unit basegroups, keeping8 language/factor variants together. Shared-entity dependence additionally checked using4 globalunit blocks and8 family blocks; such small block counts do not establish population confidence. Not independent population sampling or evidence of identical algorithms. No test-based model/layer selection.'})

if __name__=='__main__':main()
