"""Frozen same-history heldout forecasts, complete-vocabulary output and pairs."""
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_native_tail import loaded_block,checkpoint_tensor,final_norm,config,cuda_singleton,CUDA_TASKS
from rdc_history_prediction import past_cache,tensor,position_factors,weights
from phase2746_rdc_history_prediction_contract import OUT,freeze
from phase2746_rdc_history_validation import INPUTS,CONTROLS,ROUTES,compile_routes


def predict_targets(features,fit,control,lam):
    import torch
    t=lambda x:torch.tensor(x,device='cuda',dtype=torch.float64)
    train=t(fit['train_Z']);sw=t(fit['sqrt_train_weights']);u=t(fit['dual_eigenvectors']);ev=t(fit['dual_eigenvalues'])
    x=(t(features)-t(fit['feature_mean']))/t(fit['feature_scale'])
    cross=(x@(train*sw[:,None]).T/train.shape[1])@u
    result=cross@(t(fit['projected_targets'][control])/(ev+lam)[:,None])+t(fit['target_means'][control])
    return result.float().cpu().numpy()


def main():
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    cuda_singleton(CUDA_TASKS|{'phase2746_rdc_history_features.py','phase2746_rdc_history_fit.py',
        'phase2746_rdc_history_validation.py',Path(__file__).name})
    verify_storage(1024**3);start=time.monotonic();protocol,rows=freeze();frozen=read(OUT/'frozen.json');directory=OUT/'test'
    selected=[i for i,r in enumerate(rows) if r['split']=='test'];testrows=[rows[i] for i in selected]
    train=[i for i,r in enumerate(rows) if r['split']=='train'];tw=weights([rows[i] for i in train])
    names=[r['name'] for r in frozen['routes']]+['no_fit_native_block35_on_H12','no_fit_source_value_permuted',
        'training_global_H36_mean','training_current_input_token_H36_mean']
    assert len(names)==20
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf'/MODELS['qwen4'],local_files_only=True)
    answer_ids={lang:[tok.encode(v,add_special_tokens=False) for v in values] for lang,values in [('en',['Yes','No']),('zh',['是','否'])]}
    assert all(len(v)==1 for values in answer_ids.values() for v in values)
    answer_ids={k:[v[0] for v in values] for k,values in answer_ids.items()}
    immutable(directory/'protocol.json',{'timestamp':stamp(),'source':snapshot(__file__),'frozen_sha256':sha(OUT/'frozen.json'),
        'points':len(selected),'routes':names,'answer_first_token_ids':answer_ids,
        'fixed_readout_batch':21,'primary_readout':'Original tiedBF16weight, all20predictedpostnorm vectors roundedBF16 plus1originalBF16postnorm reference projected together with the same B21 shape.',
        'metrics':'StateMSE, complete-vocabulary KL(reference||prediction), TV, referenceargmax agreement, original native emittedtoken agreement/logprob, first-position controlled Yes/No conditional probabilities and whole-coordinate pair changes.',
        'semantics':'Original-native-token logprob is teacher fidelity, not gold natural NLL; complete answers require independent own-history deployment. Exposed source/wording boundaries remain explicit.'})
    feature=read(OUT/'features/result.json');assert sha(BASE/feature['field_path'])==feature['field_sha256']
    with np.load(BASE/feature['field_path']) as z:
        h0=unbits(z['H0'][selected]).astype(float);h12=unbits(z['H12'][selected]).astype(float)
        candidates=[z['candidate_native'][selected],z['candidate_source_value_shuffled'][selected]]
        reference=z['target_postnorm'][selected]
        trainH36=unbits(z['target_H36'][train]).astype(float)
    means={};denominators={}
    globalmean=(trainH36*tw[:,None]).sum(0)
    for h,w,i in zip(trainH36,tw,train):
        token=rows[i]['prompt_ids'][-1]
        means[token]=means.get(token,np.zeros(2560))+w*h;denominators[token]=denominators.get(token,0)+w
    means={k:v/denominators[k] for k,v in means.items()}
    target_predictions={};target_keys=[];route_keys=[]
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    for input_id,name in enumerate(INPUTS):
        meta=read(OUT/'fit'/(name+'.json'));assert sha(BASE/meta['field_path'])==meta['field_sha256']
        with np.load(BASE/meta['field_path']) as z:
            data={k:z[k] for k in ['train_Z','sqrt_train_weights','dual_eigenvectors','dual_eigenvalues',
                'feature_mean','feature_scale','projected_targets','target_means']}
        x=np.concatenate([h0,h12,candidates[input_id]],-1)
        for route in frozen['routes']:
            if route['input_variant']!=name:continue
            key=(input_id,route['control_index'],route['lambda_index'])
            if key not in target_predictions:
                target_predictions[key]=predict_targets(x,data,route['control_index'],route['lambda'])
                target_keys.append(key)
        del data,x;gc.collect();torch.cuda.empty_cache()
    for route in frozen['routes']:route_keys.append((route['input_index'],route['control_index'],route['lambda_index']))
    c=config();rotary=Qwen3RotaryEmbedding(c,device='cuda');layer=loaded_block(35,torch.float32)
    norm=checkpoint_tensor('model.norm.weight',torch.float32)
    wu=checkpoint_tensor('model.embed_tokens.weight' if c.tie_word_embeddings else 'lm_head.weight',torch.bfloat16)
    fields=np.empty((20,len(selected),2560),np.float32)
    compiled=np.empty((21,len(selected),2560),np.uint16);records=[];references=[]
    with torch.no_grad():
        for begin in range(0,len(testrows),3):
            group=testrows[begin:begin+3];assert len({r['sample_id'] for r in group})==1
            source=BASE/group[0]['field_path'];assert sha(source)==group[0]['field_sha256']
            with np.load(source) as z:
                for offset,row in enumerate(group):
                    index=begin+offset;kk,vv=past_cache(z,row['step']);key,value=tensor(kk)[None],tensor(vv)[None]
                    cos,sin=position_factors(rotary,len(row['prompt_ids'])-1)
                    cache={k:compile_routes(pred[index],layer,norm,key,value,cos,sin) for k,pred in target_predictions.items()}
                    forecast=[cache[k][route['route_index']] for k,route in zip(route_keys,frozen['routes'])]
                    for h in [candidates[0][index],candidates[1][index],globalmean,means.get(row['prompt_ids'][-1],globalmean)]:
                        forecast.append(final_norm(tensor(h)[None,None],norm,c.rms_norm_eps)[0,0].cpu().numpy())
                    forecast=np.stack(forecast);fields[:,index]=forecast
                    readout=torch.cat([tensor(reference[index])[None],tensor(forecast)],0).bfloat16()
                    assert readout.shape==(21,2560)
                    logits=torch.nn.functional.linear(readout,wu).float();lp=logits.double().log_softmax(-1);p=lp.exp()
                    argmax=logits.argmax(-1);ref_id=int(argmax[0]);native_id=row['next_native_token_id']
                    kl=(p[0:1]*(lp[0:1]-lp[1:])).sum(-1);tv=(p[1:]-p[0:1]).abs().sum(-1)*.5
                    assert float(kl.min())>-1e-10
                    compiled[:,index]=bits(readout)
                    yes,no=answer_ids[row['language']]
                    answer=logits[:,[yes,no]].double().softmax(-1)[:,0]
                    reference_value=unbits(reference[index]).astype(float)
                    reference_record={k:row[k] for k in ['point_id','sample_id','source_group','family','kind','step','language']}
                    reference_record.update(reference_B21_argmax=ref_id,original_B8_native_emitted_token=native_id,
                        reference_argmax_matches_original_native=ref_id==native_id,
                        reference_conditional_yes_probability=float(answer[0]),
                        reference_original_native_token_logprob=float(lp[0,native_id]))
                    references.append(reference_record)
                    for j,name in enumerate(names):
                        rec={**reference_record,'route':name,'postnorm_MSE':float(np.mean((forecast[j].astype(float)-reference_value)**2)),
                            'KL_reference_prediction':max(float(kl[j]),0.),'TV':float(tv[j]),
                            'argmax_agrees_reference':int(argmax[j+1])==ref_id,
                            'argmax_agrees_original_native':int(argmax[j+1])==native_id,
                            'predicted_argmax':int(argmax[j+1]),'original_native_token_logprob':float(lp[j+1,native_id]),
                            'conditional_yes_probability':float(answer[j+1])}
                        if row['kind']=='controlled':rec.update({k:row[k] for k in ['pair_id','world','target']})
                        records.append(rec)
                    del logits,lp,p,readout,cache
            if (begin+3)%48==0:print('HISTORY_TEST',begin+3,len(testrows),round(time.monotonic()-start,1),flush=True)
    pairs=[];pairids=sorted({r['pair_id'] for r in testrows if r['kind']=='controlled' and r['step']==0})
    lookup={(r['point_id'],r['route']):r for r in records}
    for pairid in pairids:
        points=sorted([(i,r) for i,r in enumerate(testrows) if r.get('pair_id')==pairid and r['step']==0],key=lambda t:t[1]['world'])
        assert len(points)==2
        (ai,ar),(bi,br)=points;actual=unbits(reference[bi]).astype(float)-unbits(reference[ai]).astype(float)
        sign=1 if ar['target'] in ['Yes','是'] else -1
        for route_id,name in enumerate(names):
            delta=fields[route_id,bi].astype(float)-fields[route_id,ai].astype(float)
            a=lookup[(ar['point_id'],name)];b=lookup[(br['point_id'],name)]
            pairs.append({'pair_id':pairid,'source_group':ar['source_group'],'family':ar['family'],'language':ar['language'],
                'route':name,'pair_change_MSE':float(np.mean((delta-actual)**2)),
                'zero_change_MSE':float(np.mean(actual*actual)),
                'answer_aligned_separation':sign*(a['conditional_yes_probability']-b['conditional_yes_probability']),
                'reference_answer_aligned_separation':sign*(a['reference_conditional_yes_probability']-b['reference_conditional_yes_probability'])})
    path=FIELD_STORE/'history_test/heldout.npz';arrays={'forecast_postnorm_FP32':fields,
        'fixed_B21_readout_inputs_BF16':compiled,'test_row_indices':np.array(selected),
        'fitted_targets':np.stack([target_predictions[k] for k in target_keys])}
    assert all(np.isfinite(unbits(a) if a.dtype==np.uint16 else a).all() for a in arrays.values())
    verify_storage(sum(a.nbytes for a in arrays.values()));npz(path,**arrays)
    compressed(directory/'records.json.gz',records);compressed(directory/'pair_changes.json.gz',pairs)
    save(directory/'reference_shape_checks.json',{'timestamp':stamp(),'points':len(references),
        'disagreements':[r for r in references if not r['reference_argmax_matches_original_native']],
        'scope':'FixedB21readout of stored originalBF16postnorm versus prior nativeB8emission; no generated history is replaced or rescored as new native behavior.'})
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'frozen_sha256':sha(OUT/'frozen.json'),
        'field_path':path.relative_to(BASE).as_posix(),'field_sha256':sha(path),'field_bytes':path.stat().st_size,
        'points':len(testrows),'routes':names,'full_vocab_forecasts':len(records),'controlled_first_position_pairs':len(pairids),
        'row_ids':[r['point_id'] for r in testrows],'fitted_target_keys':[list(k) for k in target_keys],
        'route_target_key_mapping':[list(k) for k in route_keys],
        'records_sha256':sha(directory/'records.json.gz'),'pair_records_sha256':sha(directory/'pair_changes.json.gz'),
        'seconds':time.monotonic()-start,'scope':'Frozen forecasts on existing original-native histories; independent new-source confirmation and autonomous deployment are not implied by these scores.'}
    save(directory/'heldout.json',result);ledger('phase2746_history_test',result['seconds'])
    print('HISTORY_TEST_COMPLETE',len(records),len(pairids),result['seconds'],flush=True)


if __name__=='__main__':
    start=time.monotonic()
    try:main()
    except Exception as exc:
        failure(OUT/'test',start,exc);raise
