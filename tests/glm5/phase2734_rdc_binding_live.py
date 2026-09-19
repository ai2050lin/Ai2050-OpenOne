"""Frozen source-binding predictions and continued-training deltas on own KV histories."""
import gc
from rdc_binding_common import *

class LiveBinding:
    def __init__(self,model,block):
        import torch
        torch.backends.cuda.matmul.allow_tf32=False
        from rdc_binding_kernels import source_arrays,apply_roles,feature_pack
        rows=gzread(BASE/'natural_discovery.json.gz');spec=read(BASE/'prediction/frozen.json')['selected'][str(block)]
        with np.load(BASE/'prediction/banks'/f'b{block}_{spec["kernel"]}_{spec["df"]}.npz') as z:
            ix=z['train_indices'];self.coeff=torch.tensor(z['coefficients'],device='cuda');self.center=torch.tensor(z['center'],device='cuda');self.scale=float(z['scale'])
        rows=[rows[i] for i in ix];s,q,e,_,_,_=source_arrays(rows)
        with np.load(BASE/'prediction/role_probe.npz') as z:coef=z['coefficients']
        self.rolecoef=torch.tensor(coef,device='cuda');self.right=feature_pack(s,q,e,apply_roles(s,coef))
        self.model=model;self.block=block;self.spec=spec;self.enabled=False;self.history=[];self.fields=[];self.embedding_id=0
        self.handles=[model.model.layers[11].register_forward_hook(self.capture),model.model.layers[block].mlp.register_forward_hook(self.replace)]
    def reset(self):self.history=[];self.fields=[]
    def capture(self,module,args,result):
        if self.enabled:
            h=result[0] if isinstance(result,tuple) else result
            self.history.append(h[0].detach().float().clone())
    def replace(self,module,args,result):
        if not self.enabled:return
        import torch
        h=torch.cat(self.history);q=h[-1];h=h/h.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
        q=q/q.square().mean().sqrt().clamp_min(1e-8)
        e=self.model.get_input_embeddings().weight[self.embedding_id].float();e=e/e.square().mean().sqrt().clamp_min(1e-8)
        right=self.right;d=h.shape[-1];base=(right['q']@q+right['e']@e)/(2*d)
        if self.spec['kernel']=='source_mean':source=right['mean']@h.mean(0)/d
        else:
            assert self.spec['kernel']=='role_position_pair'
            n,t=right['h'].shape[:2];z=torch.linspace(-1,0,len(h),device='cuda');pos=torch.stack([torch.ones_like(z),z,z*z],1)
            role=(h@self.rolecoef[:-1]+self.rolecoef[-1]).clamp_min(0)+1e-6;role=role/role.sum(-1,keepdim=True)
            dot=(h@right['h'].flatten(0,1).T/d).square()
            product=dot*(pos@right['pos'].flatten(0,1).T)*(1+role@right['role'].flatten(0,1).T)
            source=product.reshape(len(h),n,t).sum((0,2))/(len(h)*right['length'])
        kernel=1+base+source+base*source
        prediction=kernel/self.scale@self.coeff+self.center
        if self.spec['decoder']=='direct_mlp':m=prediction[:d]
        else:
            assert self.spec['decoder']=='predicted_joint_native'
            m=torch.nn.functional.linear(prediction[2*d:],module.down_proj.weight.float())
        native=result[0,-1].float();pred=m.to(result.dtype);err=(m-native).square().sum()/native.square().sum().clamp_min(1e-8)
        self.fields.append({'query':h[-1].cpu().numpy(),'native_mlp':native.cpu().numpy(),'predicted_mlp':m.cpu().numpy(),
          'relative_squared_error':float(err),'cosine':float(torch.nn.functional.cosine_similarity(m[None],native[None]))})
        out=result.clone();out[0,-1]=pred;return out
    def close(self):
        for h in self.handles:h.remove()

def rollout(model,tok,prompt,cap,predictor=None,evaluation_target=None,evaluation_candidates=None):
    import torch
    device=model.get_input_embeddings().weight.device;ids=torch.tensor([prompt],device=device);cache=None;seq=[];steps=[]
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos])
    evaluation=None
    if predictor:predictor.reset();predictor.enabled=True
    with torch.inference_mode():
      for step in range(cap):
        if predictor:predictor.embedding_id=int(ids[0,-1])
        out=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
        logits=out.logits[0,-1].float();lp=logits.log_softmax(-1);token=int(logits.argmax());cache=out.past_key_values
        if step==0 and evaluation_target is not None:
            evaluation={'target_id':evaluation_target,'native_first_target_nll':float(-lp[evaluation_target]),
              'first_target_argmax_correct':token==evaluation_target,
              'scope':'Evaluation-only labels read after logits; not supplied to model, predictor or token selection.'}
            if evaluation_candidates:
                logmass=torch.logsumexp(lp[evaluation_candidates],0)
                evaluation.update(digit_probability_mass=float(logmass.exp()),content_nll=float(-lp[evaluation_target]+logmass),
                  format_nll=float(-logmass),conditional_digit_argmax=evaluation_candidates[int(lp[evaluation_candidates].argmax())])
        steps.append({'step':step,'token_id':token,'entropy':float(-(lp.exp()*lp).sum()),'chosen_probability':float(lp[token].exp())})
        seq.append(token);del out,logits,lp
        if token in stop:break
        ids=torch.tensor([[token]],device=device)
    if predictor:predictor.enabled=False
    del cache,ids
    text=tok.decode(seq,skip_special_tokens=True)
    repeated=[tuple(seq[i:i+4]) for i in range(max(0,len(seq)-3))]
    return {'generated':text,'generated_ids':seq,'steps':steps,'initial_evaluation':evaluation,'eos':any(t in stop for t in seq),
      'token_limit_censored':not any(t in stop for t in seq) and len(seq)>=cap,
      'repeated_4gram_fraction':0 if not repeated else 1-len(set(repeated))/len(repeated)}

def apply_delta(model,block,path,step=None):
    import torch
    from rdc_law_native import parameter
    arrays=np.load(path);report={}
    with torch.no_grad():
      for k,name in [('g','gate'),('u','up'),('d','down')]:
        p=getattr(model.model.layers[block].mlp,name+'_proj').weight
        original=parameter(f'model.layers.{block}.mlp.{name}_proj.weight')
        delta=torch.tensor(arrays[k if step is not None else name+'_proj.weight'],device=p.device)
        updated=(original-step*delta if step is not None else original+delta).to(p.dtype)
        report[k]={'changed_scalar_fraction':float((updated!=p).float().mean()),'deployed_delta_norm':float((updated.float()-original).norm())}
        p.copy_(updated);del original,delta,updated
    arrays.close();return report

def reset_block(model,block):
    import torch
    from rdc_law_native import parameter
    with torch.no_grad():
      for name in ('gate','up','down'):
        p=getattr(model.model.layers[block].mlp,name+'_proj').weight;p.copy_(parameter(f'model.layers.{block}.mlp.{name}_proj.weight'))

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    out=BASE/'binding_live'
    if (out/'result.json').exists():return
    rows=[]
    for cohort in ('gum','ewt'):
      for split in ('connected_test','matched_test'):
        rows.extend([r for r in gzread(BASE/'natural_confirmation.json.gz') if r['cohort']==cohort and r['split']==split][:4])
    branches=['native','frozen_binding16','frozen_binding35','middle_coherent_2733','middle_order_control_2733']
    immutable(out/'protocol.json',{'timestamp':stamp(),'sample_ids':[r['sample_id'] for r in rows],'branches':branches,'cap':32,
      'selection':'First4per natural cohort/confirmation stratum; first predeclared training seed, not best seed.',
      'prediction_inputs':'Actual current causal allH12source history, current embedding, frozen role scores; no future/gold/target-block activation input.',
      'cache':'Prefill applies replacement only at current final query; subsequent steps append their own affected KV. Histories never reset to native tokens.',
      'first_target_scoring':'Evaluation only: final heldout material token after the last anchor. It is often punctuation and is not an adequate semantic-quality metric; never fed to model/predictor.',
      'precision':'Original BF16 model. Source-kernel FP32 matrix products disable TF32. Predicted vector and training deltas explicitly cast to native BF16 at deployment.'})
    start=time.monotonic();model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2);records=[];deployment={}
    try:
      for branch in branches:
        predictor=None
        if branch.startswith('frozen_binding'):predictor=LiveBinding(model,int(branch.split('binding')[-1]))
        elif branch.startswith('middle_'):
            condition=branch[len('middle_'):-len('_2733')]
            deployment[branch]=apply_delta(model,16,BASE/'middle_training'/f'{condition}_2733/parameter_deltas.npz')
        for row in rows:
            sid=row['sample_id'];commit=out/'commits'/branch/f'{sid}.json'
            if commit.exists():records.append(read(commit));continue
            p=row['anchors'][-1];target=row['prompt_ids'][p+1]
            record=rollout(model,tok,row['prompt_ids'][:p+1],32,predictor,evaluation_target=target)
            record.update({k:row[k] for k in ('sample_id','source_group','cohort','split')});record.update(branch=branch,prompt_ids=row['prompt_ids'][:p+1],anchor=p)
            record['first_evaluation_target_text']=tok.decode([target]);record['evaluation_target_is_last_material_token']=p+1==len(row['prompt_ids'])-1
            if predictor:
                fields=predictor.fields;npz(out/'fields'/branch/f'{sid}.npz',**{k:np.array([f[k] for f in fields]) for k in fields[0]})
                record['state_error_first']=fields[0]['relative_squared_error'];record['state_error_after_first']=float(np.mean([f['relative_squared_error'] for f in fields[1:]])) if len(fields)>1 else None
            save(commit,record);records.append(record)
            print('BINDING_OWN_HISTORY',branch,sid,len(record['generated_ids']),flush=True)
        if predictor:predictor.close();del predictor;gc.collect();torch.cuda.empty_cache()
        if branch.startswith('middle_'):reset_block(model,16)
        guard(80*1024**2);assert time.monotonic()-start<7200
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'trajectories':len(records),'rows':len(rows),'branches':branches,
      'native_parameter_deployment':deployment,'seconds':time.monotonic()-start,
      'scope':'Own-history natural continuations; no unique gold continuation or generic semantic-quality score. State prediction error and divergence are observations, not proof of language correctness.'})
    ledger('binding_own_history',time.monotonic()-start)

if __name__=='__main__':main()
