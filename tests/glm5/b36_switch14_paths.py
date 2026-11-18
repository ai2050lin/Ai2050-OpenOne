"""Same-live-generation current-token B36 branches, with exact no-op gates."""
import time,gc,sys,io
import numpy as np
import b36_switch14_design as D
import b36_switch14_engine as E
M=D.M;S=D.S


class ReadOnlyPast:
    """Tail module cache: append current K/V to an immutable cloned past for one block."""
    def __init__(self,block,data):self.block=block;self.data=data
    def update(self,k,v,layer_idx,*args,**kw):
        import torch
        assert layer_idx==self.block
        return torch.cat([self.data[0].to(k.device),k],-2),torch.cat([self.data[1].to(v.device),v],-2)


def branch(engine,cap,p,selected,original_logits):
    import torch
    arms=p['native_conditions'];states=[cap['pre36'].clone() for arm in arms]
    before_hash={str(b):[E.tensor_hash(t) for t in cap['past'][b]] for b in range(36,40)}
    deltas={};products_ok=True
    for b in range(36,40):
        assert engine.model.config._attn_implementation=='sdpa'
        layer=engine.R.C.load_block(b,engine.model.config)
        if b==36:
            products_ok=torch.equal(torch.nn.functional.silu(cap['G'])*cap['V'],cap['A'])
        for ai,arm in enumerate(arms):
            if arm=='baseline':continue
            hooks=[]
            if b==36:
                units=p['matched_unit_ids'] if arm=='matched' else p['unit_ids']
                def edit(m,args,arm=arm,units=units):
                    x=args[0].clone()
                    if arm=='double':x[...,units]*=2
                    elif arm!='noop':x[...,units]=0
                    return (x,)
                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(edit))
            elif arm in ['hold_attention','hold_mlp','hold_both']:
                if arm in ['hold_attention','hold_both']:
                    hooks.append(layer.self_attn.o_proj.register_forward_hook(lambda m,a,y,b=b:cap['attention'][b].clone()))
                if arm in ['hold_mlp','hold_both']:
                    hooks.append(layer.mlp.down_proj.register_forward_hook(lambda m,a,y,b=b:cap['mlp'][b].clone()))
            kw=dict(cap['kwargs'][b]);kw['past_key_values']=ReadOnlyPast(b,cap['past'][b])
            states[ai]=layer(states[ai],**kw)
            for h in hooks:h.remove()
            if b==36:deltas[arm]=E.bits(states[ai])
        del layer;torch.cuda.empty_cache()
    ii=torch.tensor([s['row_index'] for s in selected],device='cuda')
    logits=[original_logits[ii]]
    exact=False
    for ai,arm in enumerate(arms[1:],1):
        value=engine.model.lm_head(engine.model.model.norm(states[ai]))[:,-1]
        if arm=='noop':exact=torch.equal(value,original_logits)
        logits.append(value[ii]);del value
    raw=np.stack([E.bits(x) for x in logits])
    encoded=raw.copy();encoded[1:]^=raw[0]
    after_hash={str(b):[E.tensor_hash(t) for t in cap['past'][b]] for b in range(36,40)}
    fields=np.stack([cap['fields'][b][[s['row_index'] for s in selected]] for b in range(41)],axis=1)
    arrays=dict(logits_baseline_and_xor=encoded,hidden=fields,**{name:E.bits(cap[name][ii,0]) for name in ['G','V','A']},
        after36=np.stack([E.bits(cap['raw37'][ii,0])]+[deltas[arm][[s['row_index'] for s in selected],0] for arm in arms[1:]]))
    return arrays,dict(noop_full_batch_logits_bitwise=exact,cache_immutable=before_hash==after_hash,cache_hashes=before_hash,product_bitwise=bool(products_ok))


def run():
    import torch
    S.configure(D.PEAK);p=S.read(S.OUT/'plan.json');M.admit(p)
    behavior=S.read(S.OUT/'behavior-analysis.json');assert behavior['passed'],'Behavior gate failed'
    for arm,h in behavior['source_hashes'].items():assert h==S.file_sha(S.OUT/(arm+'.json'))
    start=time.monotonic();records=[];replay_checks=[];all_events=[];weights_saved=False
    assert not (S.OUT/'native-summary.json').exists()
    with S.GpuLease(),torch.inference_mode():
        for i,f in enumerate(p['model_files']):assert S.file_sha(S.MODEL/f['path'])==f['sha256'];S.log('path_identity',i+1,len(p['model_files']),start)
        engine=E.Stream(start,p['budget']['hard_minutes_per_run'])
        for arm in D.ARMS:
            rows=[r for r in p['rows'] if r['arm']==arm];source=S.read(S.OUT/(arm+'.json'))['results'];inputs=engine.inputs(rows);width=inputs['input_ids'].shape[1]
            state=dict(index=0,tokens=[],potential=[],cap={},seen=set());hooks=[]
            def before(m,args,kw):
                potential=[]
                for i,row in enumerate(rows):
                    prefix=engine.tok.decode([int(ids[i]) for ids in state['tokens']],skip_special_tokens=False)
                    for site in D.candidate_sites(prefix):
                        if (i,site) not in state['seen']:potential.append(dict(row_index=i,id=row['id'],family=row['family'],arm=arm,site=site,generated_cut=state['index'],prefix=prefix,target_class=D.target_class(row,site)))
                state['potential']=potential
                state['cap']=dict(fields={},past={},kwargs={},attention={},mlp={}) if potential else {}
            hooks.append(engine.model.register_forward_pre_hook(before,with_kwargs=True))
            def emb_after(m,a,y):
                if state['potential']:state['cap']['fields'][0]=E.bits(y[:,-1])
            hooks.append(engine.model.model.embed_tokens.register_forward_hook(emb_after))
            for b,layer in enumerate(engine.model.model.layers):
                def layer_before(m,args,kw,b=b):
                    if not state['potential']:return
                    cap=state['cap'];x=args[0] if args else kw['hidden_states'];assert x.shape[1]==1
                    if b>=36:
                        cache=kw['past_key_values'];cap['past'][b]=(cache.layers[b].keys.cpu().clone(),cache.layers[b].values.cpu().clone())
                        cap['kwargs'][b]={k:v for k,v in kw.items() if k not in ['past_key_values','hidden_states']}
                    if b==36:cap['pre36']=x.clone()
                def layer_after(m,args,y,b=b):
                    if state['potential']:
                        state['cap']['fields'][b+1]=E.bits(y[:,-1])
                        if b==36:state['cap']['raw37']=y.clone()
                hooks.extend([layer.register_forward_pre_hook(layer_before,with_kwargs=True),layer.register_forward_hook(layer_after)])
                if b>=36:
                    def attention(m,a,y,b=b):
                        if state['potential']:state['cap']['attention'][b]=y.clone()
                    def mlp(m,a,y,b=b):
                        if state['potential']:state['cap']['mlp'][b]=y.clone()
                    hooks.extend([layer.self_attn.o_proj.register_forward_hook(attention),layer.mlp.down_proj.register_forward_hook(mlp)])
                if b==36:
                    for name,module in [('G',layer.mlp.gate_proj),('V',layer.mlp.up_proj)]:
                        def gv(m,a,y,name=name):
                            if state['potential']:state['cap'][name]=y.clone()
                        hooks.append(module.register_forward_hook(gv))
                    def activation(m,args):
                        nonlocal weights_saved
                        if state['potential']:
                            state['cap']['A']=args[0].clone()
                            if not weights_saved:
                                np.savez_compressed(S.OUT/'native-down-columns.npz',columns=E.bits(m.weight[:,p['unit_ids']+p['matched_unit_ids']].T));weights_saved=True
                    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(activation))
            def after(m,args,kw,out):
                nextids=out.logits[:,-1].argmax(-1)
                selected=[r for r in state['potential'] if D.eligible(r['site'],engine.tok.decode([int(nextids[r['row_index']])],skip_special_tokens=False))]
                if selected:
                    arrays,checks=branch(engine,state['cap'],p,selected,out.logits[:,-1])
                    name=f'native-{arm}-step{state["index"]:03d}.npz';path=S.OUT/name;assert not path.exists()
                    # Lossless XOR against baseline greatly reduces redundant full-vocabulary storage.
                    buf=io.BytesIO();np.savez_compressed(buf,**arrays);payload=buf.getvalue()
                    S.check_write(path,len(payload),operation='same-cache native exact compressed artifact')
                    path.write_bytes(payload);del payload,buf,arrays
                    for r in selected:r['natural_token']=int(nextids[r['row_index']]);state['seen'].add((r['row_index'],r['site']))
                    item=dict(path=name,sha256=S.file_sha(path),events=selected,checks=checks)
                    records.append(item);all_events.extend(selected)
                    S.save('native-progress.json',dict(records=records,n_events=len(all_events),plan_sha256=S.file_sha(S.OUT/'plan.json')))
                    S.log('path_events',len(all_events),576,start,arm=arm,step=state['index'])
                state['tokens'].append(nextids.detach().cpu());state['index']+=1;state['cap']={};gc.collect()
            hooks.append(engine.model.register_forward_hook(after,with_kwargs=True))
            out=engine.generate('path_generation_'+arm,inputs,p['generation']['max_new_tokens']);rr=engine.records(rows,out,width)
            exact=[a['tokens']==b['tokens'] for a,b in zip(source,rr)];replay_checks.append(dict(arm=arm,exact=sum(exact),n=96,mismatch_ids=[r['id'] for r,ok in zip(rows,exact) if not ok]))
            S.save('path-generation-'+arm+'.json',dict(results=rr,plan_sha256=S.file_sha(S.OUT/'plan.json')))
            for h in hooks:h.remove()
            del out,inputs;state.clear();gc.collect();torch.cuda.empty_cache()
        engine.close();M.admit(p)
    complete=len(all_events)==576 and len({(r['id'],r['site']) for r in all_events})==576
    sources={r['id']:r for arm in D.ARMS for r in S.read(S.OUT/(arm+'.json'))['results']}
    site_mismatches=[dict(id=r['id'],site=r['site']) for r in all_events if any(sources[r['id']]['sites'][r['site']][key]!=r[key] for key in ['generated_cut','prefix','natural_token'])]
    passed=complete and not site_mismatches and all(r['exact']==r['n'] for r in replay_checks) and all(all(r['checks'][k] for k in ['noop_full_batch_logits_bitwise','cache_immutable','product_bitwise']) for r in records)
    S.save('native-summary.json',dict(status='completed' if passed else 'numeric_admission_failed',passed=passed,records=records,replay_checks=replay_checks,site_mismatches=site_mismatches,n_events=len(all_events),complete=complete,seconds=time.monotonic()-start,plan_sha256=S.file_sha(S.OUT/'plan.json'),behavior_analysis_sha256=S.file_sha(S.OUT/'behavior-analysis.json'),down_columns_sha256=S.file_sha(S.OUT/'native-down-columns.npz')))
    print('Same-cache native admission',passed,'events',len(all_events),'replay',replay_checks,flush=True)


if __name__=='__main__':run()
