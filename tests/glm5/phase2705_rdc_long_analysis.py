"""Full-vocabulary held-out long-output forecasting and prospective gate-language tests."""
import gc
from rdc_conditional_common import *
from rdc_conditional_estimators import FullKernel
from phase2704_rdc_predictive_gates import errors
OUT=CAMPAIGN/'k_long'


def extract():
    rows=[r for r in read(OUT/'material.json') if r['analysis_selected']]
    p=OUT/'features/selected.npz'
    if p.exists():
        with np.load(p) as z:return rows,{k:z[k] for k in z.files}
    data={k:[] for k in ('H0','H12','H36','previous_H36','L23_gate','L23_up','L23_a')}
    for i,r in enumerate(rows):
        with np.load(OUT/f'fields/{r["sample_id"]}.npz') as z:
            h=z['h_c']
            for l in (0,12,36):data[f'H{l}'].append(h[l])
            for k in ('L23_gate','L23_up','L23_a'):data[k].append(z[k])
        if r['generation_step']==0:prev=np.zeros(2560,np.uint16)
        else:
            with np.load(OUT/f'fields/{r["prefix_id"]}-s{r["generation_step"]-1}.npz') as z:prev=z['h_c'][36]
        data['previous_H36'].append(prev)
        if i%256==0:print('LONG_EXTRACT',i,len(rows),flush=True)
    data={k:np.stack(v) for k,v in data.items()};npz(p,**data);save(OUT/'features/selected_rows.json',rows)
    return rows,data


def decode_full_vocab(rows,predictions,truth_h,test):
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    config=read(ROOT/'models/hf/qwen3-4b/config.json');index=read(ROOT/'models/hf/qwen3-4b/model.safetensors.index.json')['weight_map']
    weight_key='lm_head.weight'
    if weight_key not in index:
        assert config['tie_word_embeddings'],'Missing independent output head without embedding tie'
        weight_key='model.embed_tokens.weight'
    weight=checkpoint(weight_key).float().cuda();gamma=checkpoint('model.norm.weight').float().cuda()
    eps=config['rms_norm_eps'];reports=[]
    save(OUT/'vocabulary/readout_contract.json',{'timestamp':stamp(),'native_head_tied_to_embedding':config['tie_word_embeddings'],
      'physical_weight_key':weight_key,'checkpoint_index_sha':sha(ROOT/'models/hf/qwen3-4b/model.safetensors.index.json'),
      'precision':'BF16 checkpoint weights cast toFP32, predicted-state ownRMS, TF32disabled. Compare actualBF16savedlogits includingallvocab entries.'})
    for name,pred in predictions.items():
        per=[]
        with torch.inference_mode():
          for start in range(0,len(test),16):
            indices=test[start:start+16];p=torch.tensor(pred[start:start+16],device='cuda',dtype=torch.float32)
            normalized=p*torch.rsqrt(p.square().mean(-1,keepdim=True)+eps)*gamma
            logp=torch.log_softmax(normalized@weight.T,dim=-1)
            actual=[]
            for ix in indices:
                with np.load(OUT/f'fields/{rows[ix]["sample_id"]}.npz') as z:actual.append(unbits(z['logits']))
            logq=torch.log_softmax(torch.tensor(np.stack(actual),device='cuda'),dim=-1);q=logq.exp()
            kl=(q*(logq-logp)).sum(-1);hit=logp.argmax(-1)==logq.argmax(-1)
            entropy=-(q*logq).sum(-1)
            eos_id=read(ROOT/'models/hf/qwen3-4b/config.json')['eos_token_id']
            if isinstance(eos_id,list):eos_id=eos_id[0]
            eq=q[:,eos_id];ep=logp[:,eos_id].exp()
            for j,ix in enumerate(indices):
                per.append({'sample_id':rows[ix]['sample_id'],'prefix_id':rows[ix]['prefix_id'],'generation_step':rows[ix]['generation_step'],
                  'kl_native_to_prediction':float(kl[j]),'native_argmax_match':bool(hit[j]),'native_entropy':float(entropy[j]),
                  'native_eos_probability':float(eq[j]),'predicted_eos_probability':float(ep[j]),'family':rows[ix]['family'],'language':rows[ix]['language']})
        report={'model':name,'states':len(per),'prefixes':len({r['prefix_id'] for r in per}),'full_vocab_size':weight.shape[0],
          'mean_KL':float(np.mean([r['kl_native_to_prediction'] for r in per])),'median_KL':float(np.median([r['kl_native_to_prediction'] for r in per])),
          'p90_KL':float(np.quantile([r['kl_native_to_prediction'] for r in per],.9)),
          'argmax_matches':sum(r['native_argmax_match'] for r in per),'eos_probability_mse':float(np.mean([(r['native_eos_probability']-r['predicted_eos_probability'])**2 for r in per])),
          'by_family':{f:{'states':len(rr),'mean_KL':float(np.mean([r['kl_native_to_prediction'] for r in rr])),'argmax_match_fraction':float(np.mean([r['native_argmax_match'] for r in rr]))} for f in sorted({r['family'] for r in per}) if (rr:=[r for r in per if r['family']==f])},
          'prefix_mean_KL':{p:float(np.mean([r['kl_native_to_prediction'] for r in per if r['prefix_id']==p])) for p in sorted({r['prefix_id'] for r in per})}}
        reports.append(report);save(OUT/f'vocabulary/{name}.json',{'summary':report,'states':per})
        print('FULL_VOCAB',name,report['mean_KL'],report['argmax_matches'],len(per),flush=True)
    del weight,gamma;torch.cuda.empty_cache()
    return reports


def gate_transfer(rows,data):
    g,u,a=[unbits(data[k]).astype(np.float64) for k in ('L23_gate','L23_up','L23_a')]
    root=CAMPAIGN/'j_predictive_gates/unit_errors'
    with np.load(root/'L23_weighted_global.npz') as z:cg=z['coefficient'][0]
    with np.load(root/'L23_weighted_language.npz') as z:cl=z['coefficient']
    inp=np.array([r['language']=='zh' for r in rows],int)
    out=np.array([(r['language']!='zh') if r['family']=='translation' else (r['language']=='zh') for r in rows],int)
    reports=[];b=g*u;train=np.array([r['word_split']=='train' for r in rows]);energy=np.mean(a[train]*a[train],0)
    for name,c in [('old_global',np.repeat(cg[None],len(rows),0)),('old_input_language',cl[inp]),('old_requested_output_language',cl[out])]:
        pred=b*c;report,arr=errors(a,pred,energy)
        report.update(model=name,scope='prospective new operations/stages; coefficients frozen on factorial training data',
          by_family={f:{'n':int(mask.sum()),'all_unit_mse':float(np.mean((a[mask]-pred[mask])**2)),'relative_to_energy':float(np.sum((a[mask]-pred[mask])**2)/max(np.sum(a[mask]**2),1e-30))} for f in sorted({r['family'] for r in rows}) if (mask:=np.array([r['family']==f for r in rows])).any()},
          by_stage={str(s):{'n':int(mask.sum()),'all_unit_mse':float(np.mean((a[mask]-pred[mask])**2))} for s in ('prefill','early','later') if (mask:=np.array([('prefill' if r['generation_step']==0 else 'early' if r['generation_step']<16 else 'later')==s for r in rows])).any()})
        reports.append(report);npz(OUT/f'unit_errors/transfer_{name}.npz',**arr)
    return reports


def main():
    assert len(list((OUT/'prefix_commits').glob('*.json')))==128
    immutable(OUT/'analysis_protocol_v2.json',{'phase':2705,'source_sha':sha(Path(__file__)),'estimator_sha':sha(ROOT/'tests/glm5/rdc_conditional_estimators.py'),
      'implementation_correction':'Initial sixfull-state fits completed, but vocabulary decoding requested a nonexistent standalone lm_head.weight. Qwen4 ties it to model.embed_tokens.weight. Original script archived under tests/glm5_temp/phase2705_rdc_long_analysis_initial.py and initialprotocol retained. Correct physicalreadout alias; no model/rawoutput/labels changed; rerun fits deterministically.',
      'sampling':'Use exactly capture-declared analysissteps, no output-dependent selection; unit0..3train,4..5val,6..7test. Prefixes rather than tokensteps define splits.',
      'prediction':'H12current fullcoordinates -> H36current fullcoordinates; compare H0current embedding+knownstep, H12 plus previous-step H36. Previous H36 is available after previous token decision, zero before firststep. Linear/quadratic fullinput kernels; no future token/answer/finalnorm input.',
      'fullvocab':'Every vocabulary entry of observed savedBF16logits defines reference. Predicted H36 uses its own RMS and actual fixed finalnorm/unembedding weights, FP32 matrix arithmetic with TF32 disabled. ActualH36 FP32decode provided as arithmetic oracle, not prediction.',
      'gate_transfer':'Old factorial L23 weightedglobal and language coefficients frozen; compare conditioning by input language versus requestedoutput language on translation directions, and observe allotherfamilies. Uses current observedg/u, reconstruction not forecast.',
      'limits':['Temporal analysis sample is not every generatedstep; allqueryH fields retained.','Only2heldoutentitygroups;many tokensteps are correlated.','Full-vocab argmax agreement measures reproduction of native model, not task correctness.','Translation/style lexicalchecklists do not establish complete meaning preservation.','Arithmetic oracle includes originalBF16versusFP32 norm/readout differences.']})
    prefixes=read(OUT/'prefixes.json');behavior=[read(OUT/f'behavior_scored/{r["sample_id"]}.json') for r in prefixes]
    assert all(b['scores']['score_version']==2 for b in behavior)
    rows,data=extract();tr,va,te=splits(rows);y=unbits(data['H36']);forecasts=[];predictions={}
    fields={k:unbits(data[k]) for k in ('H0','H12','previous_H36')};step=np.array([[r['generation_step']] for r in rows],np.float32)
    modes={'H0_step':[fields['H0'],step],'H12':[fields['H12']],'H12_previousH36':[fields['H12'],fields['previous_H36']]}
    for name,blocks in modes.items():
        scales=[max(float(np.sqrt(np.mean(np.sum(np.asarray(b[tr],np.float64)**2,1)))),1e-12) for b in blocks]
        x=np.concatenate([b/s for b,s in zip(blocks,scales)],1)
        for kind in ('linear','quadratic'):
            mid=name+'_'+kind;f=FullKernel(x,tr,va,te,kind);p,m=f.fit(y,OUT/f'models/{mid}.npz')
            report,arr=errors(y[te],p,np.mean(y[tr].astype(np.float64)**2,0))
            forecasts.append({'model':mid,'block_scales':scales,**m,**report});predictions[mid]=p
            npz(OUT/f'predictions/{mid}.npz',prediction=p,target=y[te],test=te,**arr)
            print('LONG_FORECAST',mid,m['mse'],flush=True);del f;gc.collect()
    predictions['mean_H36']=np.repeat(y[tr].mean(0)[None],len(te),0)
    predictions['identity_H12']=fields['H12'][te]
    predictions['actual_H36_arithmetic_oracle']=y[te]
    vocabulary=decode_full_vocab(rows,predictions,y,te)
    gates=gate_transfer(rows,data)
    scores={f:{'prefixes':len(rr),'generated_steps':sum(b['steps'] for b in rr),
      **{key:sum(b['scores'][key] for b in rr) for key in ('exact_reference','declared_content_constraints','order_constraints','format_structure','eos','truncated_at_limit')}} for f in sorted({r['family'] for r in prefixes}) if (rr:=[b for r,b in zip(prefixes,behavior) if r['family']==f])}
    save(OUT/'result.json',{'phase':2705,'timestamp':stamp(),'prefixes':128,'all_states':len(read(OUT/'material.json')),'analyzed_states':len(rows),
      'split_states':[len(tr),len(va),len(te)],'scores_by_family':scores,'forecasts':forecasts,'full_vocabulary':vocabulary,'prospective_gate_transfer':gates,
      'limits':read(OUT/'analysis_protocol_v2.json')['limits']})
    announce('k_long',state='analysis_complete',completed=128,total=128,states=len(read(OUT/'material.json')))


if __name__=='__main__':main()
