"""Self-fed early-native/predicted-late cache, never refreshed from teacher history."""
import argparse
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_native_tail import loaded_block,checkpoint_tensor,final_norm,cuda_singleton,CUDA_TASKS
from rdc_history_prediction import complete_block
from rdc_runtime_observer import Observer
from phase2746_rdc_confirmation_material import OUT,freeze
from phase2746_rdc_history_prediction_contract import OUT as PRED
from phase2744_rdc_query_identifiability import language_score


class OwnCache:
    def __init__(self,initial):
        self.values={i:(k.clone(),v.clone()) for i,(k,v) in initial.items()};self.position=None;self.updated=set()

    def begin(self,position):
        self.position=position;self.updated=set()

    def update(self,key,value,layer_idx,*args,**kwargs):
        import torch
        assert layer_idx not in self.updated and key.shape[-2]==1
        k,v=self.values[layer_idx];assert k.shape[-2]==v.shape[-2]==self.position,(layer_idx,self.position,k.shape)
        self.values[layer_idx]=(torch.cat([k,key],-2),torch.cat([v,value],-2));self.updated.add(layer_idx)
        return self.values[layer_idx]


def native(model,tok,observer,row,prefix,stops):
    import torch
    cache=clone_cache(prefix,model.config);current=row['prompt_ids'][-1];fields=[];posts=[];tokens=[];stats=[];first_cache=None
    for step in range(row['max_new_tokens']):
        position=len(row['prompt_ids'])-1+step;observer.reset(False)
        value=model.model(input_ids=torch.tensor([[current]],device='cuda'),position_ids=torch.tensor([[position]],device='cuda'),past_key_values=cache,use_cache=True)
        cache=value.past_key_values;h=value.last_hidden_state[0,-1];field=observer.collect()['hidden'][0]
        logits=model.lm_head(h[None]).float()[0];current=int(logits.argmax());lp=logits.double().log_softmax(-1);p=lp.exp()
        fields.append(field);posts.append(bits(h));tokens.append(current);stats.append([float(-(p*lp).sum()),float(p[current])])
        if step==0:first_cache={i:tuple(bits(getattr(cache.layers[i],k)[0,:,-1]) for k in ['keys','values']) for i in range(12)}
        del value,h,logits,lp,p
        if current in stops:break
    observer.active=False
    return {'hidden':np.stack(fields),'postnorm':np.stack(posts),'statistics':np.array(stats),'generated_ids':np.array(tokens)},first_cache


def learned(model,tok,observer,row,initial,layer,normalization,linear,route,stops,nativefields,native_first_cache):
    import torch
    coefficients,mean,scale,targetmean=linear;cache=OwnCache(initial);current=row['prompt_ids'][-1];observer.active=False
    values={k:[] for k in ['early_H0_H12','available_candidate','predicted_H35_H36_Q35','compiled_postnorm',
        'readout_BF16','appended_early_K','appended_early_V','appended_K35','appended_V35','statistics','generated_ids']}
    early_checked=False;maxnorm=0.
    for step in range(row['max_new_tokens']):
        position=len(row['prompt_ids'])-1+step;cache.begin(position)
        h=model.model.embed_tokens(torch.tensor([[current]],device='cuda'));early=[bits(h[0,0])]
        cos,sin=model.model.rotary_emb(h,torch.tensor([[position]],device='cuda'))
        for block in range(12):
            h=model.model.layers[block](hidden_states=h,attention_mask=None,position_embeddings=(cos,sin),past_key_values=cache,use_cache=True)
            early.append(bits(h[0,0]))
        early=np.stack(early)
        if step==0:
            assert np.array_equal(early,nativefields['hidden'][0,:13]),'Early native module replay does not match same-shape teacher at initial step'
            for block in range(12):
                for j,k in enumerate(['keys','values']):assert np.array_equal(bits(cache.values[block][j][0,:,-1]),native_first_cache[block][j])
            early_checked=True
        # Available candidate uses past-only35; do not append its provisional current K/V.
        k,v=cache.values[35];assert k.shape[-2]==position
        candidate=complete_block(layer,h.float(),k,v,cos.float(),sin.float())
        x=torch.cat([model.model.embed_tokens(torch.tensor([[current]],device='cuda')).float(),h.float(),candidate],-1)[0,0]
        standardized=((x.double()-mean)/scale).float();prediction=standardized@coefficients+targetmean
        assert bool(torch.isfinite(prediction).all()),('Nonfinite self-fed prediction',row['sample_id'],route['name'],step)
        h35=prediction[:2560][None,None];query=prediction[5120:].reshape(1,1,32,128)
        # True current K/V are constructed exclusively from predicted H35. The
        # provisional candidate, saved target H35 and native later states never
        # enter this update. Both routes use the same learned-H35 cache rule.
        handle=None
        if route['route_index']==2:handle=layer.self_attn.q_norm.register_forward_hook(lambda m,a,o:query)
        try:
            compiled=layer(hidden_states=h35,attention_mask=None,position_embeddings=(cos.float(),sin.float()),past_key_values=cache,use_cache=True)
        finally:
            if handle is not None:handle.remove()
        assert cache.updated==set(range(12))|{35}
        endpoint=prediction[2560:5120][None,None] if route['route_index']==0 else compiled
        post=final_norm(endpoint,normalization,model.config.rms_norm_eps)[0,0]
        readout=post.bfloat16();logits=model.lm_head(readout[None]).float()[0];current=int(logits.argmax())
        lp=logits.double().log_softmax(-1);p=lp.exp();maxnorm=max(maxnorm,float(prediction.norm()))
        values['early_H0_H12'].append(early);values['available_candidate'].append(candidate[0,0].cpu().numpy())
        values['predicted_H35_H36_Q35'].append(prediction.cpu().numpy());values['compiled_postnorm'].append(post.cpu().numpy())
        values['readout_BF16'].append(bits(readout));values['generated_ids'].append(current)
        for j,kind in enumerate(['K','V']):
            values['appended_early_'+kind].append(np.stack([bits(cache.values[b][j][0,:,-1]) for b in range(12)]))
            values['appended_'+kind+'35'].append(cache.values[35][j][0,:,-1].cpu().numpy())
        values['statistics'].append([float(-(p*lp).sum()),float(p[current])])
        del h,candidate,x,standardized,prediction,h35,compiled,endpoint,post,readout,logits,lp,p
        if current in stops:break
    arrays={k:np.stack(v) for k,v in values.items()};assert all(np.isfinite(unbits(a) if a.dtype==np.uint16 else a).all() for a in arrays.values())
    return arrays,{'first_step_all_early_H_and_KV_bit_equal_native':early_checked,'max_predicted_target_L2':maxnorm,
        'true_current_late_state_inputs':0,'post_initialization_teacher_cache_injections':0,
        'early_cache_precision':'NativeBF16','late_cache_precision':'Same-valued nativeFP32 block on predictedFP32H35; original-prefixBF16 promotedFP32, newly appendedFP32 retained.'}


def record(row,route,arrays,tok,stops,execution,mode,extra):
    tokens=arrays['generated_ids'].tolist();file=FIELD_STORE/('history_autonomous_'+mode)/(row['sample_id']+'__'+route+'.npz')
    verify_storage(sum(a.nbytes for a in arrays.values()));npz(file,**arrays)
    r={k:row[k] for k in ['sample_id','source_group','family','kind','language']}
    r.update(timestamp=stamp(),route=route,generated_ids=tokens,generated_text=tok.decode(tokens,skip_special_tokens=True),
        EOS=tokens[-1] in stops,censored=tokens[-1] not in stops and len(tokens)==row['max_new_tokens'],max_new_tokens=row['max_new_tokens'],
        field_path=file.relative_to(BASE).as_posix(),field_sha256=sha(file),field_bytes=file.stat().st_size,execution=execution,**extra)
    if row['kind']=='controlled':
        r.update({k:row[k] for k in ['pair_id','world','target']});r['answer_scoring']=language_score(dict(row,kind='controlled_language'),r['generated_text'],tokens,stops,row['max_new_tokens'])
    save(OUT/'autonomous'/mode/'commits'/(row['sample_id']+'__'+route+'.json'),r);return r


def run_row(model,tok,observer,row,layer,normalization,linear,routes,stops,execution,mode,repeat=False):
    import torch
    observer.active=False;start=time.monotonic();assert len(row['prompt_ids'])>1
    prefix_output=model.model(input_ids=torch.tensor([row['prompt_ids'][:-1]],device='cuda'),use_cache=True)
    prefix=prefix_output.past_key_values;del prefix_output
    blocks=list(range(12))+[35]
    initial={b:(prefix.layers[b].keys.to(dtype=torch.float32 if b==35 else torch.bfloat16),
        prefix.layers[b].values.to(dtype=torch.float32 if b==35 else torch.bfloat16)) for b in blocks}
    nativefields,first_cache=native(model,tok,observer,row,prefix,stops)
    packets=[('native_B1_cache',nativefields,{'role':'Original full36block model, same B1 prefix-initialization and current-step execution as candidate early blocks.'})]
    for route,label in zip(routes,['frozen_direct','frozen_generalQ_native']):
        a,extra=learned(model,tok,observer,row,initial,layer,normalization,linear,route,stops,nativefields,first_cache)
        if repeat:
            b,rextra=learned(model,tok,observer,row,initial,layer,normalization,linear,route,stops,nativefields,first_cache)
            assert set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a) and extra==rextra
            extra['same_input_repeat_all_arrays_bit_equal']=True;del b
        extra['frozen_route_name']=route['name'];packets.append((label,a,extra))
    # Retain exactly the shared allowed initialization once, not all native
    # intermediate prefix states and never a future generated teacher token.
    pfile=FIELD_STORE/('history_autonomous_'+mode)/(row['sample_id']+'__initial.npz')
    initarrays={'blocks':np.array(blocks),'prompt_ids':np.array(row['prompt_ids']),
        'prefix_keys':np.stack([bits(prefix.layers[b].keys[0]) for b in blocks]),
        'prefix_values':np.stack([bits(prefix.layers[b].values[0]) for b in blocks])}
    verify_storage(sum(a.nbytes for a in initarrays.values()));npz(pfile,**initarrays)
    init={'timestamp':stamp(),'sample_id':row['sample_id'],'field_path':pfile.relative_to(BASE).as_posix(),'field_sha256':sha(pfile),
        'field_bytes':pfile.stat().st_size,'prefix_positions':len(row['prompt_ids'])-1,'scope':'Shared original-prefill past-token initialization only; excludes current last prompt token and all generation.'}
    save(OUT/'autonomous'/mode/'commits'/(row['sample_id']+'__initial.json'),init)
    elapsed=time.monotonic()-start;rr=[]
    for label,arrays,extra in packets:
        extra.update(shared_initialization=init,allocated_three_branch_seconds=elapsed/3)
        rr.append(record(row,label,arrays,tok,stops,execution,mode,extra))
    del prefix,initial,packets,nativefields,initarrays;return rr,elapsed


def main(pilot_only=False):
    import torch
    cuda_singleton(CUDA_TASKS|{'phase2746_rdc_confirmation_capture.py','phase2746_rdc_confirmation_prediction.py',Path(__file__).name})
    verify_storage(1024**3);start=time.monotonic();protocol,rows=freeze();cm=read(PRED/'deployment/coefficients.json')
    assert sha(PRED/'frozen.json')==cm['frozen_sha256']==protocol['frozen_predictor_sha256']
    execution={'source':snapshot(__file__),'frozen_sha256':sha(PRED/'frozen.json'),'material_sha256':sha(OUT/'material.json.gz'),
        'coefficients_sha256':cm['field_sha256'],'native_helper':snapshot(Path(__file__).with_name('rdc_history_prediction.py'))}
    immutable(OUT/'autonomous/protocol.json',{'timestamp':stamp(),'source':execution['source'],'execution':execution,
        'rows':512,'routes':['native_B1_cache','frozen_direct','frozen_generalQ_native'],'selected_frozen_names':[r['name'] for r in cm['routes']],
        'state':'Original full-model prefill of all prompt tokens except the final current token is allowed once per row. Retain pastKV0..11and35. Native baseline also uses same prefix with a B1 current step.',
        'transition':'Each chosen candidate computes native blocks0..11 from its own current token and own early cache, constructs provisional B35(H12;ownpast35), predicts all H35/H36/Q35, updates current K35/V35 from predictedH35 only, and emits from its own compiled full-vocabulary distribution.',
        'current_late_blocks_used':'No actual current blocks12..34; same-valued FP32 block35 is applied to predicted state. Original native baseline has all36blocks.',
        'teacher_access_after_initialization':0,'early_cache':'NativeBF16','predicted_late_cache':'FP32 original scalar values applied to FP32prediction, no future teacher cache refresh',
        'numerical_admission':'All13initialearlyH and12blockcurrentK/V must match B1native. Eightrow pilot repeats all candidate arrays bitwise. FP32fullprimal conversion checked independently, not refitted.',
        'natural_cap':32,'controlled_cap':128,'selection':'All512 frozen rows, not filtered by native success. No route reselection on test/confirmation or own histories.',
        'scope':'Self-fed surrogate with native-prefix initialization, not cold-start prefix extraction, full-model equivalence, semantic accuracy guarantee or new-training formation evidence.'})
    model=None;observer=None
    try:
        model,tok=load('qwen4',OUT/'autonomous/loader');observer=Observer(model);observer.active=False
        layer=loaded_block(35,torch.float32);norm=checkpoint_tensor('model.norm.weight',torch.float32)
        assert sha(BASE/cm['field_path'])==cm['field_sha256']
        with np.load(BASE/cm['field_path']) as z:
            linear=tuple(torch.from_numpy(z[k]).to('cuda') for k in ['coefficients_FP32','feature_mean_FP64','feature_scale_FP64','target_mean_FP32'])
        stops=model.generation_config.eos_token_id or tok.eos_token_id;stops=set(stops if isinstance(stops,list) else [stops])
        with torch.inference_mode():
            pilotpath=OUT/'autonomous/pilot/result.json'
            if not pilotpath.exists():
                pilotrows=[rows[i] for i in [0,25,192,256,320,384,448,511]];pilots=[];times=[]
                for row in pilotrows:
                    rr,elapsed=run_row(model,tok,observer,row,layer,norm,linear,cm['routes'],stops,execution,'pilot',repeat=True)
                    pilots+=rr;times.append(elapsed);print('SELF_HISTORY_PILOT',row['sample_id'],elapsed,[len(r['generated_ids']) for r in rr],flush=True)
                projected=sum(r['field_bytes'] for r in pilots)/8*512*1.5
                # Include original-prefix storage and a1GiB writing allowance.
                projected+=sum(r['shared_initialization']['field_bytes'] for r in pilots[::3])/8*512*1.5+1024**3
                verify_storage(int(projected))
                save(pilotpath,{'timestamp':stamp(),'all_passed':True,'execution':execution,'rows':8,'branches':24,'initial_early_H_KV_bit_equal':True,
                    'candidate_repeat_all_arrays_bit_equal':True,'seconds_per_threebranch_row_including_repeats':times,'projected_main_bytes':projected,
                    'scope':'Timing includes repeat checks; main does not repeat each output. No behavior-based candidate or row selection.'})
            else:assert read(pilotpath)['execution']==execution
            if pilot_only:return
            records=[]
            for row in rows:
                receipts=[OUT/'autonomous/main/commits'/(row['sample_id']+'__'+name+'.json') for name in ['native_B1_cache','frozen_direct','frozen_generalQ_native']]
                if all(p.exists() for p in receipts):
                    rr=[read(p) for p in receipts];assert all(r['execution']==execution and sha(BASE/r['field_path'])==r['field_sha256'] for r in rr)
                else:rr,_=run_row(model,tok,observer,row,layer,norm,linear,cm['routes'],stops,execution,'main')
                records+=rr;save(OUT/'autonomous/progress.json',{'timestamp':stamp(),'rows':len(records)//3,'total_rows':512,'seconds':time.monotonic()-start})
                if len(records)%24==0:print('SELF_HISTORY_MAIN',len(records)//3,512,round(time.monotonic()-start,1),flush=True)
            compressed(OUT/'autonomous/records.json.gz',records)
            result={'timestamp':stamp(),'all_passed':True,'execution':execution,'rows':512,'branches':len(records),
                'generated_steps':sum(len(r['generated_ids']) for r in records),'field_bytes':sum(r['field_bytes'] for r in records),
                'initialization_bytes':sum(r['shared_initialization']['field_bytes'] for r in records[::3]),
                'initial_early_H_and_KV_checks':1024,'teacher_refresh_after_initialization':0,'seconds':time.monotonic()-start,
                'scope':'All selected self-fed branches executed; semantic success must be reported independently. Native prefix initialization is not removed.'}
            save(OUT/'autonomous/result.json',result);ledger('phase2746_autonomous',result['seconds']);print('SELF_HISTORY_COMPLETE',result,flush=True)
    except Exception as exc:failure(OUT/'autonomous',start,exc);raise
    finally:
        if observer is not None:observer.close()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');main(p.parse_args().pilot)
