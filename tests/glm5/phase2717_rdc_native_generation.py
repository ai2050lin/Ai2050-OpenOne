"""All-source native block compilation and explicit conditional versus autonomous generation."""
import argparse,gc
from rdc_relation_common import *
from rdc_relation_inference import CurrentRule,TemporalRule,check_frozen


def metric(actual,pred):
    a=np.asarray(actual,float);p=np.asarray(pred,float);e=(a-p)**2
    return {'MSE':float(e.mean()),'relative_MSE':float(e.mean()/max(np.mean(a*a),1e-30))}


class BlockTrace:
    def __init__(self,model):
        self.block=model.model.layers[23];self.positions=[];self.data={};self.handles=[]
        for name,module in [('attention',self.block.self_attn),('gate',self.block.mlp.gate_proj),('up',self.block.mlp.up_proj),('mlp',self.block.mlp)]:
            def hook(m,a,o,name=name):
                if name=='attention':
                    self.data['probability']=o[1][0,:,self.positions].float().cpu().numpy();o=o[0]
                self.data[name]=o[0,self.positions].float().cpu().numpy()
            self.handles.append(module.register_forward_hook(hook))
        def pre(m,a):self.data['activation']=a[0][0,self.positions].float().cpu().numpy()
        self.handles.append(self.block.mlp.down_proj.register_forward_pre_hook(pre))
    def close(self):
        for h in self.handles:h.remove()


def compile_native(model,rule):
    import torch
    out=BASE/'native';material=rows(True);trace=BlockTrace(model);device=model.get_input_embeddings().weight.device
    if not (out/'protocol.json').exists():save(out/'protocol.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'frozen_sha':sha(BASE/'frozen.json'),'sources':128,'anchors_each':2,
      'primary':'Predict H23 at EVERY known source from its own H12 with the frozen full-quadratic rule; run actual whole block23 attention and MLP, both residuals and norms.',
      'hybrid':'Replace only current H23 at each query while actual past H23 supplies its K/V; explicit extra deeper-state information, not the primary equal-input regime.',
      'oracle':'Feed all actual saved H23 with original full sequence shape through the actual BF16 block; compare saved actual H24 bits.',
      'attention':'All32 heads and all known sources, full causal mask. Current/past Q/K RMSNorm and RoPE are native code.',
      'all_unit_policy':'9728 MLP units and2560 output coordinates included in errors/energies; every path reconstructible from original fields, frozen rule, native weights and this protocol. Full factor fixtures for first4 source units, chosen by material order.',
      'limits':'One actual block consumes learned H23 estimates, not a complete decoded language model; rule was trained at quantile anchors, not every prefix position.'})
    def run(h,positions):
        n=len(h);x=torch.as_tensor(h,dtype=torch.bfloat16,device=device)[None];pos=torch.arange(n,device=device)[None]
        mask=torch.full((n,n),torch.finfo(x.dtype).min,dtype=x.dtype,device=device).triu(1)[None,None];trace.positions=positions;trace.data={}
        y=trace.block(x,attention_mask=mask,position_ids=pos,position_embeddings=model.model.rotary_emb(x,pos),use_cache=False)
        return y[0,positions].float().cpu().numpy(),{k:v.copy() for k,v in trace.data.items()},bits(y[0,positions])
    try:
      with torch.inference_mode():
       for i,r in enumerate(material):
        cp=out/f'commits/{r["sample_id"]}.json'
        if cp.exists():continue
        z=load_field(r,True);actual=unbits(z['h23']);pred=rule(unbits(z['h12']))[:,:2560];positions=r['anchors'];h24=unbits(z['h24'][[0,3]])
        oracle,ot,ob=run(actual,positions);assert np.array_equal(ob,z['h24'][[0,3]]),('Native whole-block arithmetic mismatch',r['sample_id'])
        compiled,ct,cb=run(pred,positions);hybrid=[];hybridbits=[];parts=[]
        for p in positions:
            hh=actual.copy();hh[p]=pred[p];hy,ht,hb=run(hh,[p]);hybrid.append(hy[0]);hybridbits.append(hb[0]);parts.append(ht)
        hybrid=np.stack(hybrid);ht={k:np.concatenate([t[k] for t in parts],axis=1 if k=='probability' else 0) for k in ot}
        methods={'all_predicted_sources':(compiled,ct),'actual_past_hybrid':(hybrid,ht)};reports={};packet={'oracle_h24':ob,'all_predicted_h24':cb,'hybrid_h24':np.stack(hybridbits),'positions':np.array(positions)}
        for name,(hh,tt) in methods.items():
            q=np.maximum(ot['probability'],1e-30);p=np.maximum(tt['probability'],1e-30);pkl=np.sum(q*np.log(q/p),axis=-1)
            reports[name]={'H24':metric(h24,hh),'attention':metric(ot['attention'],tt['attention']),'MLP':metric(ot['mlp'],tt['mlp']),'mean_head_attention_KL':float(pkl.mean())}
            packet[name+'_all_unit_activation_MSE']=np.mean((tt['activation']-ot['activation'])**2,axis=0);packet[name+'_H24_coordinate_MSE']=np.mean((hh-h24)**2,axis=0);packet[name+'_all_head_KL']=pkl
        packet['actual_all_unit_activation_energy']=np.mean(ot['activation']**2,axis=0)
        if i<4:
            for name,tt in [('oracle',ot),('predicted',ct),('hybrid',ht)]:
                for k,v in tt.items():packet[f'fixture_{name}_{k}']=v.astype(np.float32)
        path=out/f'fields/{r["sample_id"]}.npz';npz(path,**packet);save(cp,{'timestamp':stamp(),'sample_id':r['sample_id'],'language':r['language'],'oracle_bitwise_equal':True,'reports':reports,'field_sha':sha(path),'code_sha':sha(Path(__file__))})
        print('NATIVE_FULL_BLOCK',i+1,128,reports['all_predicted_sources']['H24']['MSE'],flush=True);guard(12*1024**2)
    finally:trace.close()
    commits=[read(p) for p in sorted((out/'commits').glob('*.json'))];unit={};energy=[]
    for c in commits:
        with np.load(out/f'fields/{c["sample_id"]}.npz') as z:
            energy.append(z['actual_all_unit_activation_energy'])
            for name in ('all_predicted_sources','actual_past_hybrid'):unit.setdefault(name,[]).append(z[name+'_all_unit_activation_MSE'])
    npz(out/'all_unit_profiles.npz',actual_activation_energy=np.mean(energy,axis=0),**{k+'_MSE':np.mean(v,axis=0) for k,v in unit.items()})
    result={'timestamp':stamp(),'sources':len(commits),'queries':2*len(commits),'oracle_all_bitwise_equal':all(c['oracle_bitwise_equal'] for c in commits),'methods':{}}
    for name in ('all_predicted_sources','actual_past_hybrid'):
        result['methods'][name]={part+'_MSE':float(np.mean([c['reports'][name][part]['MSE'] for c in commits])) for part in ('H24','attention','MLP')}
        result['methods'][name]['mean_head_attention_KL']=float(np.mean([c['reports'][name]['mean_head_attention_KL'] for c in commits]))
    save(out/'result.json',result);print('NATIVE_FULL_BLOCK_COMPLETE',result,flush=True)


class LastState:
    def __init__(self,model):
        self.data={};self.handles=[]
        for layer in (12,36):
            def hook(m,a,o,layer=layer):self.data['h'+str(layer)]=bits(o[0,-1])
            self.handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def distribution_metrics(lq,lp):
    return {'KL':float((lq.exp()*(lq-lp)).sum()),'native_argmax':int(lq.argmax()),'predicted_argmax':int(lp.argmax()),'argmax_agreement':bool(lq.argmax()==lp.argmax())}


def generation(model,tok,rule):
    import torch
    out=BASE/'generation';material=[r for lang in ('en','zh') for r in rows(True) if r['language']==lang][:0]
    material=[r for r in rows(True) if int(r['sample_id'].rsplit('r',1)[1])<32];assert len(material)==64
    temporal=TemporalRule();trace=LastState(model);device=model.get_input_embeddings().weight.device;weight=model.lm_head.weight.float();gamma=model.model.norm.weight.float();eps=model.config.rms_norm_eps
    eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [tok.eos_token_id])
    if not (out/'protocol.json').exists():save(out/'protocol.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'frozen_sha':sha(BASE/'frozen.json'),'source_ids':[r['sample_id'] for r in material],
      'initial_prefix':'First quantile anchor of each preselected first32 fresh source units per language, no future observed tokens.',
      'native':'Greedy natural continuation, raw prefix, original BF16 full model body/head, no chat template, no KV cache, max16 new tokens or native EOS.',
      'conditional':'At each NATIVE-generated prefix, current rule receives true H12; temporal conditional receives prior true H36 and known newly generated token embedding. Neither is autonomous.',
      'self_fed':'Initialize one true H36 once at original prefix. Own FP32 norm/head selects each next token; update only previous PREDICTED H36 + that own selected token embedding + available prefix descriptor. No HiddenState refresh after initialization.',
      'own_prefix_reference':'For diagnosis only, independently run native model on the exact self-generated prefix; reference states/probabilities are never fed to the self-fed learner.',
      'readout':'Complete151936 vocabulary each step; actual gamma and all2560 coordinates. Actual BF16 native versus FP32 surrogate arithmetic tracked separately.',
      'records':'Full native current H12/H36, full self-predicted H36 and actual H36 on same self prefix; predicted conditional states exactly recomputable; tokenIDs/text/divergence/EOS/budget/repetition.',
      'limitations':'Natural continuation has no unique target answer. NLL/agreement are model comparisons, not semantic correctness. Branches with different prefixes are not compared as same-state forecasts.'})
    def forward(ids):
        x=torch.tensor([ids],device=device);trace.data={};post=model.model(input_ids=x,use_cache=False).last_hidden_state[0,-1];lq=model.lm_head(post[None])[0].float().log_softmax(-1)
        return lq,{k:v.copy() for k,v in trace.data.items()}
    def readout(h):
        h=torch.as_tensor(np.asarray(h),dtype=torch.float32,device=device);return ((h*torch.rsqrt(h.square().mean()+eps)*gamma)@weight.T).log_softmax(-1)
    try:
      with torch.inference_mode():
       for i,r in enumerate(material):
        cp=out/f'commits/{r["sample_id"]}.json'
        if cp.exists():continue
        initial=r['prompt_ids'][:r['anchors'][0]+1];native_ids=list(initial);native_tokens=[];native_rows=[];native_fields=[];previous=None;initial_state=None
        for step in range(16):
            lq,field=forward(native_ids);actual=unbits(field['h36']);early=unbits(field['h12']);ph=rule(early)[0,2560:];lp=readout(ph)
            row={'step':step,'conditional':distribution_metrics(lq,lp),'conditional_state':metric(actual,ph),'actual_H36_FP32_oracle':distribution_metrics(lq,readout(actual))}
            if previous is not None:
                embedding=model.get_input_embeddings().weight[native_ids[-1]].float().cpu().numpy();th=temporal(previous,embedding,native_ids,r['language']);row['observed_state_temporal']=distribution_metrics(lq,readout(th));row['observed_state_temporal_error']=metric(actual,th)
            if initial_state is None:initial_state=actual.copy()
            chosen=int(lq.argmax());row['selected_token_id']=chosen;native_rows.append(row);native_fields.append(field);native_tokens.append(chosen);native_ids.append(chosen);previous=actual
            if chosen in eos:break
        own_ids=list(initial);own_tokens=[];own_rows=[];own_states=[];own_actual=[];state=initial_state.copy()
        for step in range(16):
            lq,field=forward(own_ids);lp=readout(state);chosen=int(lp.argmax());row={'step':step,'same_own_prefix_reference':distribution_metrics(lq,lp),'same_own_prefix_state_error':metric(unbits(field['h36']),state),'selected_token_id':chosen}
            own_rows.append(row);own_states.append(state.copy());own_actual.append(field['h36']);own_tokens.append(chosen);own_ids.append(chosen)
            if chosen in eos:break
            embedding=model.get_input_embeddings().weight[chosen].float().cpu().numpy();state=temporal(state,embedding,own_ids,r['language']);assert np.isfinite(state).all()
        packet={'native_h12':np.stack([f['h12'] for f in native_fields]),'native_h36':np.stack([f['h36'] for f in native_fields]),'self_predicted_h36':np.stack(own_states).astype(np.float32),'native_h36_on_self_prefix':np.stack(own_actual)}
        path=out/f'fields/{r["sample_id"]}.npz';npz(path,**packet);common=min(len(native_tokens),len(own_tokens));first=next((s for s in range(common) if native_tokens[s]!=own_tokens[s]),None)
        if first is None and len(native_tokens)!=len(own_tokens):first=common
        def repeat(ids):
            triples=[tuple(ids[j:j+3]) for j in range(max(0,len(ids)-2))];return 0. if not triples else 1-len(set(triples))/len(triples)
        record={'timestamp':stamp(),'sample_id':r['sample_id'],'language':r['language'],'initial_prefix_ids':initial,'initial_prefix_text':tok.decode(initial),
          'native_tokens':native_tokens,'native_text':tok.decode(native_tokens),'native_rows':native_rows,'self_tokens':own_tokens,'self_text':tok.decode(own_tokens),'self_rows':own_rows,
          'first_token_divergence_zero_based':first,'full_budget_or_EOS_exact_tokens':native_tokens==own_tokens,
          'native_stop':'EOS' if native_tokens[-1] in eos else '16_token_budget','self_stop':'EOS' if own_tokens[-1] in eos else '16_token_budget',
          'native_repeated_trigram_fraction':repeat(native_tokens),'self_repeated_trigram_fraction':repeat(own_tokens),'field_sha':sha(path),'code_sha':sha(Path(__file__))}
        save(cp,record);print('NATURAL_GENERATION',i+1,64,'first_divergence',first,flush=True);guard(8*1024**2)
    finally:trace.close();del weight,gamma;gc.collect();torch.cuda.empty_cache()
    allrows=[read(p) for p in sorted((out/'commits').glob('*.json'))];native=[s for r in allrows for s in r['native_rows']];own=[s for r in allrows for s in r['self_rows']];report={'timestamp':stamp(),'source_units':len(allrows),'native_steps':len(native),'self_steps':len(own)}
    for name,rr in [('conditional',[s['conditional'] for s in native]),('observed_state_temporal',[s['observed_state_temporal'] for s in native if 'observed_state_temporal' in s]),('self_fed_same_prefix',[s['same_own_prefix_reference'] for s in own]),('actual_H36_FP32_oracle',[s['actual_H36_FP32_oracle'] for s in native])]:report[name]={'steps':len(rr),'mean_KL':float(np.mean([x['KL'] for x in rr])),'argmax_agreement':float(np.mean([x['argmax_agreement'] for x in rr]))}
    report.update(full_continuation_exact_count=sum(r['full_budget_or_EOS_exact_tokens'] for r in allrows),native_EOS_count=sum(r['native_stop']=='EOS' for r in allrows),self_EOS_count=sum(r['self_stop']=='EOS' for r in allrows),
      first_divergence_counts={str(v):sum(r['first_token_divergence_zero_based']==v for r in allrows) for v in sorted({r['first_token_divergence_zero_based'] for r in allrows},key=lambda x:999 if x is None else x)},
      native_repeated_trigram_fraction=float(np.mean([r['native_repeated_trigram_fraction'] for r in allrows])),self_repeated_trigram_fraction=float(np.mean([r['self_repeated_trigram_fraction'] for r in allrows])))
    save(out/'result.json',report);print('NATURAL_GENERATION_COMPLETE',report,flush=True)


def main(mode):
    import torch
    check_frozen();guard(50*1024**2);from phase2662_symmetric_mapping_contract import load_native
    rule=CurrentRule();model,tok=load_native('qwen4')
    try:
        if mode in ('all','native'):compile_native(model,rule)
        if mode in ('all','generation'):generation(model,tok,rule)
    finally:del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','native','generation'],default='all');a=p.parse_args();main(a.mode)
