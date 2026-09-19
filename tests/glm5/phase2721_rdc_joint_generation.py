"""32-step real-prefix conditional forecasts and three unrefreshed autonomous branches."""
import gc
from rdc_joint_common import *
from rdc_joint_kernels import Rule
from rdc_joint_prior_rules import QueryProposal
from rdc_joint_native_attention import NativeAttention
from rdc_joint_features import available_temporal
from rdc_joint_capture import ledger
from phase2721_rdc_joint_confirmation import frozen_check


class HistoryTrace:
    def __init__(self,model):
        self.data={};self.handles=[]
        for layer in (12,36):
            def hook(m,a,o,layer=layer):self.data['h'+str(layer)]=bits(o[0])
            self.handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def metrics(lq,lp):
    q=lq.exp()
    return {'KL':float((q*(lq-lp)).sum()),'argmax_agreement':bool(lp.argmax()==lq.argmax()),
        'native_argmax':int(lq.argmax()),'predicted_argmax':int(lp.argmax()),
        'native_entropy':float(-(q*lq).sum()),'predicted_entropy':float(-(lp.exp()*lp).sum())}


def repetition(ids,n=3):
    grams=[tuple(ids[j:j+n]) for j in range(max(0,len(ids)-n+1))]
    return 0 if not grams else 1-len(set(grams))/len(grams)


def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    frozen=frozen_check();out=BASE/'generation'
    if (out/'result.json').exists():return
    start=time.monotonic()
    material=[r for r in rows(True) if int(r['sample_id'].rsplit('j',1)[1])<32]
    assert len(material)==64
    names={'MSE_embedding':('embedding_bilinear',False),'KL_embedding':('embedding_bilinear',True),'KL_history':('history_trilinear',True)}
    immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'frozen_sha':sha(BASE/'frozen.json'),
        'source_ids':[r['sample_id'] for r in material],'source_selection':'First32 pre-frozen fresh source IDs per language, not selected by outcomes.',
        'initial_prefix':'Through first predeclared quantile anchor only; no remaining natural-text suffix supplied.',
        'max_new_tokens':32,'native':'Full native BF16 body/head greedy, unpadded raw prefix, no chat, use_cache=False; EOS distinguished from32-token budget.',
        'conditional':'On each independently native-generated prefix: current trueH12 -> KL current rule; previous trueH36 plus known newE -> frozen temporal rules. History branch has true pastH12 at ONE selected layer and old-only query proposal.',
        'autonomous_branches':names,'autonomous_state':'Initialize trueH36 once. History branch also initializes all trueH12 for the initial prefix once. All subsequent H36 and appended H12 are predictions only; no true-state refresh.',
        'same_own_prefix_reference':'Separate full native forward on each surrogate prefix measures distribution/state error; never fed into the surrogate update.',
        'complete_output':'Every2560 coordinate and all151936 vocab entries used; original H36 fields suffice to reconstruct full distributions, no Top-K summaries define the computation.',
        'scope':'Natural continuation has no unique correct answer. Agreement/entropy/NLL/repetition/stopping are separate from factual, referential or reasoning correctness; no automatic correctness label from graph family.'})
    temporal={k:Rule('temporal',n,kl=kl) for k,(n,kl) in names.items()}
    current=Rule('current','current_linear',kl=True)
    proposal=QueryProposal()
    model,tok=load_native('qwen4');device=model.get_input_embeddings().weight.device
    native_attention=NativeAttention()
    trace=HistoryTrace(model)
    weight=model.lm_head.weight.float();gamma=model.model.norm.weight.float();eps=model.config.rms_norm_eps
    eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [tok.eos_token_id])
    def forward(ids):
        trace.data={};x=torch.tensor([ids],device=device)
        post=model.model(input_ids=x,use_cache=False).last_hidden_state[0,-1]
        lp=model.lm_head(post[None])[0].float().log_softmax(-1)
        return lp,{k:v.copy() for k,v in trace.data.items()}
    def readout(h):
        x=torch.as_tensor(np.asarray(h),dtype=torch.float32,device=device)
        return ((x*torch.rsqrt(x.square().mean()+eps)*gamma)@weight.T).log_softmax(-1)
    def embedding(tid):return model.get_input_embeddings().weight[tid].float().cpu().numpy()
    records=[]
    try:
      with torch.inference_mode():
        for ri,r in enumerate(material):
            cp=out/'commits'/f'{r["sample_id"]}.json'
            if cp.exists():records.append(read(cp));continue
            initial=r['prompt_ids'][:r['anchors'][0]+1]
            initial_lp,initial_field=forward(initial)
            initial_h=unbits(initial_field['h36'][-1]);initial_history=unbits(initial_field['h12'])
            native_ids=list(initial);native_rows=[];native_tokens=[];native_h36=[];native_h12=[];previous=None;previous_history=None
            for step in range(32):
                lq,f=forward(native_ids);h=unbits(f['h36'][-1]);hh=unbits(f['h12'])
                ph=current({'current':hh[-1]})[0]
                rr={'step':step,'current_conditional':metrics(lq,readout(ph)),
                    'actual_H36_FP32_floor':metrics(lq,readout(h)), 'temporal':{}}
                if previous is not None:
                    e=embedding(native_ids[-1])
                    features=available_temporal(previous,e,previous_history,native_ids,proposal,native_attention)
                    for name,rule in temporal.items():
                        p=rule(features)[0]
                        rr['temporal'][name]=metrics(lq,readout(p))|{'state_MSE':float(np.mean((p.astype(float)-h)**2))}
                chosen=int(lq.argmax());rr['selected_token_id']=chosen
                native_rows.append(rr);native_tokens.append(chosen);native_h36.append(f['h36'][-1]);native_h12.append(f['h12'][-1])
                native_ids.append(chosen);previous=h;previous_history=hh
                if chosen in eos:break
            packet={'initial_h12':initial_field['h12'],'initial_h36':initial_field['h36'][-1],
                    'native_h36':np.stack(native_h36),'native_h12_last':np.stack(native_h12)}
            branches={}
            for name,rule in temporal.items():
                ids=list(initial);state=initial_h.copy();history=initial_history.copy();tokens=[];rr=[];states=[];actuals=[];proposals=[]
                for step in range(32):
                    lq,f=forward(ids)
                    lp=readout(state);chosen=int(lp.argmax())
                    row={'step':step,'same_prefix':metrics(lq,lp),'state_MSE':float(np.mean((state.astype(float)-unbits(f['h36'][-1]))**2)),
                         'selected_token_id':chosen,'predicted_state_RMS':float(np.sqrt(np.mean(state.astype(float)**2)))}
                    rr.append(row);states.append(state.copy());actuals.append(f['h36'][-1]);tokens.append(chosen);ids.append(chosen)
                    if chosen in eos:break
                    e=embedding(chosen)
                    if name=='KL_history':
                        feature=available_temporal(state,e,history,ids,proposal,native_attention)
                        next_state=rule(feature)[0]
                        proposed=feature['query_proposal'];history=np.concatenate([history,proposed[None]],0);proposals.append(proposed)
                    else:
                        next_state=rule({'previous':state,'embedding':e})[0]
                    assert np.isfinite(next_state).all()
                    state=next_state
                common=min(len(native_tokens),len(tokens));div=next((i for i in range(common) if native_tokens[i]!=tokens[i]),None)
                if div is None and len(tokens)!=len(native_tokens):div=common
                branches[name]={'tokens':tokens,'text':tok.decode(tokens),'steps':rr,
                    'stop':'EOS' if tokens[-1] in eos else '32_token_budget','first_divergence_zero_based':div,
                    'full_native_tokens_exact':tokens==native_tokens,'repeated_trigram_fraction':repetition(tokens),
                    'longest_identical_token_run':max(len(list(g)) for _,g in __import__('itertools').groupby(tokens))}
                packet[name+'_predicted_h36']=np.stack(states).astype(np.float32)
                packet[name+'_native_h36_same_own_prefix']=np.stack(actuals)
                if proposals:packet[name+'_appended_predicted_h12']=np.stack(proposals).astype(np.float32)
            path=out/'fields'/f'{r["sample_id"]}.npz'
            guard(sum(a.nbytes for a in packet.values())+1024**2)
            npz(path,**packet)
            report=({k:r[k] for k in ('sample_id','source_group','language','genre','language_mode_families')}|
                {'timestamp':stamp(),'initial_prefix_ids':initial,'initial_prefix_text':tok.decode(initial),'native_tokens':native_tokens,'native_text':tok.decode(native_tokens),
                 'native_rows':native_rows,'native_stop':'EOS' if native_tokens[-1] in eos else '32_token_budget','native_repeated_trigram_fraction':repetition(native_tokens),
                 'branches':branches,'field_sha':sha(path),'state_refresh_after_initialization':False})
            save(cp,report);records.append(report)
            trace.data={}
            print('JOINT_GENERATION',ri+1,64,{k:(len(b['tokens']),b['first_divergence_zero_based']) for k,b in branches.items()},flush=True)
    finally:
        trace.close();native_attention.close();del model,weight,gamma,trace;gc.collect();torch.cuda.empty_cache()
    result={'timestamp':stamp(),'sources':len(records),'max_new_tokens':32,'native_EOS':sum(r['native_stop']=='EOS' for r in records),
        'native_repetition':float(np.mean([r['native_repeated_trigram_fraction'] for r in records])),'branches':{}}
    for name in names:
        parts=[(r,s) for r in records for s in r['branches'][name]['steps']]
        result['branches'][name]={'steps':len(parts),'EOS':sum(r['branches'][name]['stop']=='EOS' for r in records),
            'full_native_tokens_exact':sum(r['branches'][name]['full_native_tokens_exact'] for r in records),
            'same_prefix_KL':paired_summary([s['same_prefix']['KL'] for r,s in parts],[r['source_group'] for r,s in parts]),
            'argmax_agreement':float(np.mean([s['same_prefix']['argmax_agreement'] for r,s in parts])),
            'repeated_trigram_fraction':float(np.mean([r['branches'][name]['repeated_trigram_fraction'] for r in records])),
            'by_step':[{'step':j,'count':len(pp),'KL':float(np.mean([s['same_prefix']['KL'] for r,s in pp])),
                       'state_MSE':float(np.mean([s['state_MSE'] for r,s in pp]))} for j in range(32) if (pp:=[(r,s) for r,s in parts if s['step']==j])]}
    conditional={}
    for name in names:
        values=[(r,s['temporal'][name]) for r in records for s in r['native_rows'] if name in s['temporal']]
        conditional[name]={'steps':len(values),'KL':paired_summary([s['KL'] for r,s in values],[r['source_group'] for r,s in values]),
            'argmax_agreement':float(np.mean([s['argmax_agreement'] for r,s in values]))}
    result['observed_state_temporal_on_native_prefix']=conditional
    result['limits']='Every branch uses its own prefix; model comparison is not content or referent correctness. Initial history is actual once only; same-prefix native diagnostic is never an update input.32 steps is a finite horizon, not all-state/infinite-time stability proof.'
    save(out/'result.json',result);ledger('joint_32step_generation',time.monotonic()-start,sources=64);guard()
    print('JOINT_GENERATION_COMPLETE',result,usage(),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
