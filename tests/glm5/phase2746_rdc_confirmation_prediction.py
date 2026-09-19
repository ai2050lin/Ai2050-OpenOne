"""All frozen routes, unchanged old fit, independently new native histories."""
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_native_tail import loaded_block,checkpoint_tensor,final_norm,config,cuda_singleton,CUDA_TASKS
from rdc_history_prediction import past_cache,tensor,position_factors,complete_block,source_permutation,weights
from phase2746_rdc_history_prediction_contract import OUT as PRED,freeze as oldfreeze
from phase2746_rdc_confirmation_material import OUT,freeze
from phase2746_rdc_history_validation import INPUTS,compile_routes
from phase2746_rdc_history_test import predict_targets
from phase2746_rdc_history_analysis import summarize


def main():
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    cuda_singleton(CUDA_TASKS|{'phase2746_rdc_confirmation_capture.py',Path(__file__).name})
    verify_storage(2*1024**3);start=time.monotonic();protocol,material=freeze();oldprotocol,oldrows=oldfreeze()
    frozen=read(PRED/'frozen.json');assert sha(PRED/'frozen.json')==protocol['frozen_predictor_sha256']
    native=gzread(OUT/'native/records.json.gz');byid={r['sample_id']:r for r in native};rows=[]
    for row in material:
        reference=byid[row['sample_id']]
        for step in range(min(3,len(reference['generated_ids']))):
            rr={k:row[k] for k in ['sample_id','source_group','family','kind','language','split']}
            rr.update(point_id=row['sample_id']+'_t'+str(step),step=step,prompt_ids=row['prompt_ids']+reference['generated_ids'][:step],
                field_path=reference['field_path'],field_sha256=reference['field_sha256'],next_native_token_id=reference['generated_ids'][step])
            if row['kind']=='controlled':rr.update({k:row[k] for k in ['pair_id','world','target']})
            rows.append(rr)
    compressed(OUT/'prediction/material.json.gz',rows);n=len(rows)
    names=[r['name'] for r in frozen['routes']]+['no_fit_native_block35_on_H12','no_fit_source_value_permuted','training_global_H36_mean','training_current_input_token_H36_mean']
    answer_ids=read(PRED/'test/protocol.json')['answer_first_token_ids']
    immutable(OUT/'prediction/protocol.json',{'timestamp':stamp(),'source':snapshot(__file__),'frozen_predictor_sha256':sha(PRED/'frozen.json'),
        'routes':names,'points':n,'source_rows':len(material),'steps':'first3actuallyavailable, no earlyEOS imputation','fixed_readout_batch':21,
        'inputs':'Only originalH0,H12,strictlypastKV35,position. All future fields are evaluation targets or numeric oracles, never features.',
        'selection':'No new fit or regularization/route selection. Original validation-defined16routes and4fixed baselines.'})
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    c=config();rotary=Qwen3RotaryEmbedding(c,device='cuda');layer=loaded_block(35,torch.float32);bf=loaded_block(35,torch.bfloat16)
    norm=checkpoint_tensor('model.norm.weight',torch.float32)
    arrays={k:np.empty((n,2560),np.uint16) for k in ['H0','H12','target_postnorm']}
    arrays.update({k:np.empty((n,2560),np.float32) for k in ['candidate_native','candidate_source_value_shuffled','oracle_FP32_postnorm']})
    arrays['oracle_BF16_postnorm']=np.empty((n,2560),np.uint16);diagnostics=[]
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['sample_id'],[]).append(i)
    with torch.no_grad():
        for count,(sid,indices) in enumerate(groups.items()):
            source=BASE/rows[indices[0]]['field_path'];assert sha(source)==rows[indices[0]]['field_sha256']
            with np.load(source) as z:
                hidden=z['hidden'];post=z['postnorm']
                for i in indices:
                    row=rows[i];step=row['step'];position=len(row['prompt_ids'])-1
                    kk,vv=past_cache(z,step);assert kk.shape==vv.shape==(8,position,128)
                    key,value=tensor(kk)[None],tensor(vv)[None];cos,sin=position_factors(rotary,position)
                    early=tensor(hidden[step,12])[None,None]
                    candidate=complete_block(layer,early,key,value,cos,sin)
                    perm=torch.tensor(source_permutation(sid,step,position),device='cuda')
                    shuffled=complete_block(layer,early,key,value.index_select(-2,perm),cos,sin)
                    arrays['H0'][i]=hidden[step,0];arrays['H12'][i]=hidden[step,12]
                    arrays['candidate_native'][i]=candidate[0,0].cpu().numpy();arrays['candidate_source_value_shuffled'][i]=shuffled[0,0].cpu().numpy()
                    # Predictor features have already been computed; the following uses future state solely as an oracle check.
                    arrays['target_postnorm'][i]=post[step];actual=tensor(hidden[step,35])[None,None]
                    smooth=complete_block(layer,actual,key,value,cos,sin)
                    nativeblock=complete_block(bf,actual.bfloat16(),key.bfloat16(),value.bfloat16(),cos.bfloat16(),sin.bfloat16())
                    arrays['oracle_FP32_postnorm'][i]=final_norm(smooth,norm,c.rms_norm_eps)[0,0].cpu().numpy()
                    arrays['oracle_BF16_postnorm'][i]=bits(final_norm(nativeblock,norm.bfloat16(),c.rms_norm_eps)[0,0])
                    diagnostics.append({'point_id':row['point_id'],'BF16_B1_B8_MSE':float(np.mean((unbits(arrays['oracle_BF16_postnorm'][i]).astype(float)-unbits(post[step]))**2)),
                        'FP32_BF16_MSE':float(np.mean((arrays['oracle_FP32_postnorm'][i].astype(float)-unbits(post[step]))**2))})
            if (count+1)%32==0:print('CONFIRMATION_FEATURES',count+1,len(groups),round(time.monotonic()-start,1),flush=True)
    del bf;gc.collect();torch.cuda.empty_cache()
    fm=read(PRED/'features/result.json');assert sha(BASE/fm['field_path'])==fm['field_sha256']
    train=[i for i,r in enumerate(oldrows) if r['split']=='train'];tw=weights([oldrows[i] for i in train])
    with np.load(BASE/fm['field_path']) as z:trainH36=unbits(z['target_H36'][train]).astype(float)
    globalmean=(trainH36*tw[:,None]).sum(0);means={};denom={}
    for h,w,i in zip(trainH36,tw,train):
        token=oldrows[i]['prompt_ids'][-1];means[token]=means.get(token,np.zeros(2560))+w*h;denom[token]=denom.get(token,0)+w
    means={k:v/denom[k] for k,v in means.items()};predictions={}
    for inp,name in enumerate(INPUTS):
        meta=read(PRED/'fit'/(name+'.json'));assert sha(BASE/meta['field_path'])==meta['field_sha256']
        with np.load(BASE/meta['field_path']) as z:data={k:z[k] for k in ['train_Z','sqrt_train_weights','dual_eigenvectors','dual_eigenvalues','feature_mean','feature_scale','projected_targets','target_means']}
        x=np.concatenate([unbits(arrays['H0']),unbits(arrays['H12']),arrays['candidate_native' if inp==0 else 'candidate_source_value_shuffled']],-1)
        for r in frozen['routes']:
            if r['input_index']!=inp:continue
            key=(inp,r['control_index'],r['lambda_index'])
            if key not in predictions:predictions[key]=predict_targets(x,data,r['control_index'],r['lambda'])
        del x,data;gc.collect();torch.cuda.empty_cache()
    wu=checkpoint_tensor('model.embed_tokens.weight' if c.tie_word_embeddings else 'lm_head.weight',torch.bfloat16)
    fields=np.empty((20,n,2560),np.float32);readouts=np.empty((21,n,2560),np.uint16);records=[];refchecks=[]
    with torch.no_grad():
        for count,(sid,indices) in enumerate(groups.items()):
            with np.load(BASE/rows[indices[0]]['field_path']) as z:
                for i in indices:
                    row=rows[i];kk,vv=past_cache(z,row['step']);key,value=tensor(kk)[None],tensor(vv)[None]
                    cos,sin=position_factors(rotary,len(row['prompt_ids'])-1)
                    compiled={k:compile_routes(pred[i],layer,norm,key,value,cos,sin) for k,pred in predictions.items()}
                    forecast=[compiled[(r['input_index'],r['control_index'],r['lambda_index'])][r['route_index']] for r in frozen['routes']]
                    for h in [arrays['candidate_native'][i],arrays['candidate_source_value_shuffled'][i],globalmean,means.get(row['prompt_ids'][-1],globalmean)]:
                        forecast.append(final_norm(tensor(h)[None,None],norm,c.rms_norm_eps)[0,0].cpu().numpy())
                    forecast=np.stack(forecast);fields[:,i]=forecast
                    readout=torch.cat([tensor(arrays['target_postnorm'][i])[None],tensor(forecast)],0).bfloat16();assert readout.shape==(21,2560)
                    logits=torch.nn.functional.linear(readout,wu).float();lp=logits.double().log_softmax(-1);p=lp.exp();argmax=logits.argmax(-1)
                    kl=(p[:1]*(lp[:1]-lp[1:])).sum(-1);tv=(p[1:]-p[:1]).abs().sum(-1)*.5;assert float(kl.min())>-1e-10
                    readouts[:,i]=bits(readout);nativeid=row['next_native_token_id'];ref=int(argmax[0]);yes,no=answer_ids[row['language']]
                    answer=logits[:,[yes,no]].double().softmax(-1)[:,0];target=unbits(arrays['target_postnorm'][i]).astype(float)
                    rr={k:row[k] for k in ['point_id','sample_id','source_group','family','kind','step','language']}
                    rr.update(reference_B21_argmax=ref,original_B8_native_emitted_token=nativeid,reference_argmax_matches_original_native=ref==nativeid,
                        reference_conditional_yes_probability=float(answer[0]),reference_original_native_token_logprob=float(lp[0,nativeid]))
                    if ref!=nativeid:refchecks.append(rr)
                    for j,name in enumerate(names):
                        rec={**rr,'route':name,'postnorm_MSE':float(np.mean((forecast[j].astype(float)-target)**2)),
                            'KL_reference_prediction':max(float(kl[j]),0.),'TV':float(tv[j]),'argmax_agrees_reference':int(argmax[j+1])==ref,
                            'argmax_agrees_original_native':int(argmax[j+1])==nativeid,'predicted_argmax':int(argmax[j+1]),
                            'original_native_token_logprob':float(lp[j+1,nativeid]),'conditional_yes_probability':float(answer[j+1])}
                        if row['kind']=='controlled':rec.update({k:row[k] for k in ['pair_id','world','target']})
                        records.append(rec)
                    del logits,lp,p,compiled,readout
            if (count+1)%32==0:print('CONFIRMATION_PREDICT',count+1,len(groups),round(time.monotonic()-start,1),flush=True)
    pairs=[];lookup={(r['point_id'],r['route']):r for r in records}
    for pair in sorted({r['pair_id'] for r in rows if r['kind']=='controlled' and r['step']==0}):
        ab=sorted([(i,r) for i,r in enumerate(rows) if r.get('pair_id')==pair and r['step']==0],key=lambda t:t[1]['world']);assert len(ab)==2
        (ai,ar),(bi,br)=ab;actual=unbits(arrays['target_postnorm'][bi]).astype(float)-unbits(arrays['target_postnorm'][ai]).astype(float)
        sign=1 if ar['target'] in ['Yes','是'] else -1
        for j,name in enumerate(names):
            delta=fields[j,bi].astype(float)-fields[j,ai].astype(float);a,b=lookup[(ar['point_id'],name)],lookup[(br['point_id'],name)]
            pairs.append({'pair_id':pair,'source_group':ar['source_group'],'family':ar['family'],'language':ar['language'],'route':name,
                'pair_change_MSE':float(np.mean((delta-actual)**2)),'zero_change_MSE':float(np.mean(actual*actual)),
                'answer_aligned_separation':sign*(a['conditional_yes_probability']-b['conditional_yes_probability']),
                'reference_answer_aligned_separation':sign*(a['reference_conditional_yes_probability']-b['reference_conditional_yes_probability'])})
    arrays.update(forecast_postnorm_FP32=fields,fixed_B21_readout_inputs_BF16=readouts)
    assert all(np.isfinite(unbits(a) if a.dtype==np.uint16 else a).all() for a in arrays.values())
    file=FIELD_STORE/'history_confirmation_prediction/points.npz';verify_storage(sum(a.nbytes for a in arrays.values()));npz(file,**arrays)
    for name,value in [('records',records),('pair_changes',pairs),('numerical_diagnostics',diagnostics)]:compressed(OUT/'prediction'/(name+'.json.gz'),value)
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'frozen_predictor_sha256':sha(PRED/'frozen.json'),'routes':names,
        'points':n,'full_vocabulary_forecasts':len(records),'source_rows':512,'row_ids':[r['point_id'] for r in rows],
        'field_path':file.relative_to(BASE).as_posix(),'field_sha256':sha(file),'field_bytes':file.stat().st_size,
        'B21_B8_reference_argmax_disagreements':refchecks,'no_new_fitting_or_selection':True,'seconds':time.monotonic()-start}
    save(OUT/'prediction/result.json',result);ledger('phase2746_confirmation_prediction',result['seconds'])
    analyze(records,pairs,names,answer_ids);print('CONFIRMATION_PREDICTION_COMPLETE',len(records),result['seconds'],flush=True)


def analyze(records,pairs,names,answer_ids):
    reports=[];metrics=['postnorm_MSE','KL_reference_prediction','TV','argmax_agrees_original_native']
    for subset in ['all','natural','controlled',*sorted({r['family'] for r in records})]:
        rr=[r for r in records if subset=='all' or r['kind']==subset or r['family']==subset]
        for first in [True,False]:
            selected=[r for r in rr if not first or r['step']==0];report=summarize(selected,names,metrics,2750)
            report.update(subset=subset,horizon='first_position' if first else 'first3_available_positions',points_per_route=len(selected)//len(names),
                scope='Independent of the frozen fit/selection under the explicit inventory; source-group bootstrap, not pretraining-unseen guarantee.')
            reports.append(report)
    for r in pairs:r['pair_MSE_minus_zero']=r['pair_change_MSE']-r['zero_change_MSE']
    pair_reports=[]
    for subset in ['all',*sorted({r['family'] for r in pairs})]:
        rr=[r for r in pairs if subset=='all' or r['family']==subset];report=summarize(rr,names,['pair_change_MSE','pair_MSE_minus_zero','answer_aligned_separation'],2751)
        report.update(subset=subset,scope='Prospective new expression combinations, clustered by controlled case across languages/worlds.');pair_reports.append(report)
    answers=[r.copy() for r in records if r['kind']=='controlled' and r['step']==0]
    for r in answers:
        truth=r['target'] in ['Yes','是'];r['first_token_gold_match']=r['predicted_argmax']==answer_ids[r['language']][0 if truth else 1]
        r['candidate_conditional_gold_match']=(r['conditional_yes_probability']>=.5)==truth
    answer_report=summarize(answers,names,['first_token_gold_match','candidate_conditional_gold_match'],2752)
    save(OUT/'analysis/result.json',{'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'reports':reports,'pair_reports':pair_reports,
        'controlled_first_answers':answer_report,'no_new_fitting_or_selection':True,'independent_confirmation_executed':True,
        'autonomous_deployment_executed':False,'scope':'New-source current-state forecast and first-answer checks, not self-fed ability.'})


if __name__=='__main__':
    start=time.monotonic()
    try:main()
    except Exception as exc:failure(OUT/'prediction',start,exc);raise
