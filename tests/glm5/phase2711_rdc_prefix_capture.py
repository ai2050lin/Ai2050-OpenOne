"""Native causal full-sentence observation, full coordinates, explicit retained positions."""
import argparse
import gc
import sys
from rdc_prefix_common import *


class Capture:
    def __init__(self,model,native=True):
        self.enabled=False;self.data={};self.hooks=[];self.positions=[]
        self.depth=len(model.model.layers);self.native=native
        self.module=sys.modules[model.model.layers[0].self_attn.__class__.__module__]
        self.original=getattr(self.module,'eager_attention_forward',None)
        if native:
            assert self.original is not None
            def attention(module,q,k,v,*args,**kwargs):
                result=self.original(module,q,k,v,*args,**kwargs)
                if self.enabled and module.layer_idx==23:
                    self.data['L23_q']=bits(q[0,:,self.positions])
                    self.data['L23_k']=bits(k[0]);self.data['L23_v']=bits(v[0])
                    self.data['L23_p']=bits(result[1][0,:,self.positions])
                return result
            self.module.eager_attention_forward=attention
        def register(module,key,allpositions=False,pre=False):
            fn=(lambda m,a:self.put(key,a[0],allpositions)) if pre else (lambda m,a,o:self.put(key,o,allpositions))
            self.hooks.append(module.register_forward_pre_hook(fn) if pre else module.register_forward_hook(fn))
        register(model.model.embed_tokens,'H0',True);register(model.model.norm,'postnorm')
        for layer,block in enumerate(model.model.layers):
            register(block,f'H{layer+1}',True)
            if native and layer==23:
                for key,m in [('gate',block.mlp.gate_proj),('up',block.mlp.up_proj),('down',block.mlp.down_proj),
                    ('attention_out',block.self_attn.o_proj),('attention_x',block.input_layernorm),('mlp_x',block.post_attention_layernorm)]:
                    register(m,'L23_'+key)
                register(block.mlp.down_proj,'L23_a',pre=True)
                register(block.self_attn.o_proj,'L23_head_output',pre=True)

    def put(self,key,tensor,allpositions):
        if self.enabled:self.data[key]=bits(tensor[0] if allpositions else tensor[0,self.positions])

    def close(self):
        for h in self.hooks:h.remove()
        if self.native:self.module.eager_attention_forward=self.original


def selected_rows(model,confirmation):
    path=CAMPAIGN/('confirmation_material.json' if confirmation else 'material_stratified.json')
    rows=read(path)
    if model!='qwen4':
        # 32 training / 16 validation / 16 test, shared source text and char anchors.
        selected=[]
        for language in ('en','zh'):
            for split,n in (('train',16),('validation',8),('test',8)):
                selected.extend([r for r in rows if r['language']==language and r['split']==split][:n])
        rows=sorted(selected,key=lambda r:(r['split'],r['sample_id'].rsplit('-',1)[-1],r['language']))
        assert len(rows)==64
    return rows


def prepare(key,confirmation):
    run=key+('_confirmation' if confirmation else '')
    out=CAMPAIGN/run;rows=selected_rows(key,confirmation)
    if confirmation:
        assert (CAMPAIGN/'shared_rules/frozen_models.json').exists(), 'Freeze before confirmation capture'
    protocol_path=out/f'protocols/{sha(Path(__file__))[:16]}.json'
    immutable(protocol_path,{'phase':2713 if confirmation or key!='qwen4' else 2711,
      'source_sha':sha(Path(__file__)),'run':run,'maximum_units':len(rows),
      'material_sha':sha(CAMPAIGN/('confirmation_material.json' if confirmation else 'material_stratified.json')),
      'execution':'Original nonquantized BF16 eager attention, batch1, natural length, no padding/truncation, no KV cache, plain source text (not chat template). Full causal prefill yields all teacher-forced token positions.',
      'retained':'Six quantile/offset full-coordinate H0..depth states, postnorm and full vocabulary logits at two anchors. Qwen4: all-source native L23 K/V, six Q/P/gate/up/a/down/inputs/writes. Full H every position only 16 predeclared main panels.',
      'streamed':'Every observed token at every H layer enters language/split full-coordinate sums and square sums. Individual nonpanel nonanchor H is not retained.',
      'causal_checks':'First four units same-shape hooked/unhooked equivalence and rewritten-future suffix invariance at first anchor.',
      'token_alignment':'Qwen4 actual tokenizer offsets. Scale models align same character endpoints to their own enclosing token; offsets are not token-index correspondence. Visible graph text is decoded token prefix, never a character-offset slice that can complete an unfinished UTF-8 character.',
      'forecast_scope':'Teacher-forced observations are not natural-generation success. No future UD tags are input features.',
      'precision':'BF16 states/logits bit-preserved uint16; streaming sums float64; no checkpoint quantization.',
      'budget':{'disk_floor':FLOOR,'campaign_ceiling':CEILING,'maximum_model_seconds':5400}})
    return run,out,rows,protocol_path


def finish_capture(run,out,rows,limit,key,confirmation):
    commits=[read(out/f'commits/{r["sample_id"]}.json') for r in rows[:limit]]
    moment_path=out/'all_token_moments.npz'
    with np.load(moment_path) as z:
        counts=z['counts'];processed=set(z['processed_ids'].tolist())
    assert all(r['sample_id'] in processed for r in rows[:limit])
    if limit==4:
        data_bytes=sum(p.stat().st_size for sub in ('fields','rows','behavior','commits') for p in (out/sub).glob('*'))
        panel_bytes=sum(p.stat().st_size for p in (out/'full_panels').glob('*'))
        projected=data_bytes/4*len(rows)+(panel_bytes/4*16 if key=='qwen4' and not confirmation else 0)+moment_path.stat().st_size
        forecast_reserve=900*1024**2 if key=='qwen4' and not confirmation else 150*1024**2
        times=[c['elapsed_seconds'] for c in commits];checks=[c['checks'] for c in commits]
        save(out/'pilot_audit.json',{'timestamp':stamp(),'passed':bool(len(checks)==4 and all(x.get('same_shape_noop') and x.get('rewritten_future_suffix_bitwise_invariant') for x in checks)
          and projected+forecast_reserve < CEILING and np.mean(times)*len(rows)<5400),
          'pilot_cases':4,'projected_this_run_bytes':projected,'forecast_and_other_run_reserve':forecast_reserve,
          'mean_seconds':float(np.mean(times)),'projected_capture_seconds':float(np.mean(times)*len(rows)),
          'source_bytes_not_in_projection':usage()-data_bytes-panel_bytes-moment_path.stat().st_size,
          'note':'Main pilot includes 16 full panels in extrapolation. Recheck actual campaign usage before each expansion.'})
    save(out/f'capture_{limit}.json',{'completed':len(commits),'case_seconds':[c['elapsed_seconds'] for c in commits],
      'checks':[c['checks'] for c in commits],'all_token_counts':counts.tolist(),'physical_campaign_bytes':usage(),'timestamp':stamp()})
    status(run,state='captured' if limit==len(rows) else 'pilot_complete',completed=len(commits),total=len(rows))


def main(key,limit,confirmation):
    import torch
    run,out,rows,protocol_path=prepare(key,confirmation);wanted=rows[:limit]
    if all((out/f'commits/{r["sample_id"]}.json').exists() for r in wanted):
        finish_capture(run,out,rows,limit,key,confirmation)
        print('CAPTURE_ALREADY_COMPLETE',run,limit,flush=True);return
    if limit>4:assert read(out/'pilot_audit.json')['passed']
    guard(100*1024**2)
    status(run,state='loading',completed=len(list((out/'commits').glob('*.json'))),total=len(rows))
    if key=='qwen4':
        from phase2662_symmetric_mapping_contract import load_native
        model,tok=load_native(key)
    else:
        from phase2713_rdc_prefix_scale_loader import load
        model,tok=load(key,out)
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    cap=Capture(model,key=='qwen4');device=model.get_input_embeddings().weight.device
    width=model.config.hidden_size;groups=['en/train','zh/train','en/validation','zh/validation','en/test','zh/test','en/confirmation','zh/confirmation']
    moment_path=out/'all_token_moments.npz'
    if moment_path.exists():
        with np.load(moment_path) as z:
            sums=z['sums'];squares=z['squares'];counts=z['counts'];processed=set(z['processed_ids'].tolist())
    else:
        sums=np.zeros((len(groups),cap.depth+1,width),np.float64);squares=np.zeros_like(sums)
        counts=np.zeros(len(groups),np.int64);processed=set()
    runtime={'timestamp':stamp(),'torch':torch.__version__,'model':key,'dtype':str(model.dtype),'quantized':False,
      'device_map':getattr(model,'hf_device_map',{'actual':str(device)}),'native_model_code_sha':sha(Path(cap.module.__file__)),'depth':cap.depth,'width':width,
      'groups':groups,'cache':False,'shape':'natural length, batch1','model_config':model.config.to_dict()}
    save(out/'runtime.json',runtime);started=time.monotonic();times=[];checks=[]
    try:
      with torch.inference_mode():
       for i,r in enumerate(wanted):
        cp=out/f'commits/{r["sample_id"]}.json'
        if cp.exists():assert r['sample_id'] in processed;continue
        assert time.monotonic()-started<5400
        start=time.monotonic();enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
        ids=enc['input_ids'];offsets=enc['offset_mapping']
        if key=='qwen4':assert ids==r['prompt_ids'];positions=r['positions']
        else:
            anchors=[]
            for a in r['anchors']:
                endpoint=r['token_offsets'][a][1]
                candidates=[j for j,(b,e) in enumerate(offsets) if b<endpoint<=e]
                assert candidates,(key,r['sample_id'],endpoint)
                anchors.append(min(candidates[-1],len(ids)-4))
            positions=[a+d for a in anchors for d in (0,1,2)]
        cap.positions=positions;cap.enabled=True;cap.data={}
        inputs=torch.tensor([ids],device=device)
        hidden=model.model(input_ids=inputs,use_cache=False).last_hidden_state
        logits=model.lm_head(hidden[:,[positions[0],positions[3]]])
        cap.enabled=False
        h=np.stack([cap.data.pop(f'H{j}') for j in range(cap.depth+1)])
        assert h.shape==(cap.depth+1,len(ids),width)
        h_float=unbits(h);assert np.isfinite(h_float).all()
        for a in cap.data.values():assert np.isfinite(unbits(a)).all()
        check={'sample_id':r['sample_id']}
        if i<4:
            plain=model.model(input_ids=inputs,use_cache=False).last_hidden_state
            assert torch.equal(hidden,plain)
            changed=inputs.clone();changed[0,positions[0]+1:]=tok.eos_token_id
            future=model.model(input_ids=changed,use_cache=False).last_hidden_state
            assert torch.equal(hidden[0,:positions[0]+1],future[0,:positions[0]+1])
            check.update(same_shape_noop=True,rewritten_future_suffix_bitwise_invariant=True)
            del plain,future,changed
        checks.append(check)
        pooled={}
        for layer in sorted({0,cap.depth//3,2*cap.depth//3,cap.depth}):
            pooled[f'H{layer}_prefix_mean']=np.stack([h_float[layer,:p+1].mean(0,dtype=np.float64).astype(np.float32) for p in positions])
        packet=dict(h=h[:,positions],positions=np.array(positions),logits=bits(logits[0]),**cap.data,**pooled)
        fp=out/f'fields/{r["sample_id"]}.npz';npz(fp,**packet)
        files=[fp]
        if key=='qwen4' and not confirmation and r['full_panel']:
            panel=out/f'full_panels/{r["sample_id"]}.npz';npz(panel,h=h);files.append(panel)
        row=dict(r,model=key,prompt_ids=ids,tokens=tok.convert_ids_to_tokens(ids),token_offsets=offsets,positions=positions,
          actual_anchor_graphs=[prefix_graph(tok.decode(ids[:p+1],skip_special_tokens=False,clean_up_tokenization_spaces=False),p,r['language']) for p in positions],
          char_anchor_alignment='Qwen original' if key=='qwen4' else 'own token enclosing Qwen anchor endpoint; subsequent offsets are own token steps')
        rp=out/f'rows/{r["sample_id"]}.json';save(rp,row);files.append(rp)
        logp=logits.float().log_softmax(-1)[0]
        observed=[ids[positions[k]+1] for k in (0,3)]
        bp=out/f'behavior/{r["sample_id"]}.json';save(bp,{'teacher_forced':True,'anchor_positions':[positions[0],positions[3]],
          'next_observed_ids':observed,'native_argmax_ids':logits[0].argmax(-1).cpu().tolist(),
          'observed_token_nll':[-float(logp[j,x]) for j,x in enumerate(observed)],
          'entropy':(-(logp.exp()*logp).sum(-1)).cpu().tolist(),'checks':check});files.append(bp)
        # Stats snapshot precedes commit. A crash after snapshot can safely rerun the uncommitted case without double counting.
        if r['sample_id'] not in processed:
            g=groups.index(r['language']+'/'+r['split']);sums[g]+=h_float.sum(1,dtype=np.float64)
            squares[g]+=np.square(h_float,dtype=np.float64).sum(1);counts[g]+=len(ids);processed.add(r['sample_id'])
            npz(moment_path,sums=sums,squares=squares,counts=counts,processed_ids=np.array(sorted(processed)))
        save(cp,{'sample_id':r['sample_id'],'protocol_sha':sha(protocol_path),'protocol_file':str(protocol_path.relative_to(out)),
          'files':{str(p.relative_to(out)):sha(p) for p in files},
          'elapsed_seconds':time.monotonic()-start,'observed_allH_shape':list(h.shape),'retained_anchor_shape':list(packet['h'].shape),
          'observed_token_count':len(ids),'all_token_stats_snapshot_includes_case':True,'checks':check})
        times.append(time.monotonic()-start)
        print('PREFIX_CAPTURE',run,i+1,limit,len(ids),round(times[-1],2),flush=True)
        status(run,state='running',completed=len(list((out/'commits').glob('*.json'))),total=len(rows))
        del hidden,logits,h,h_float,packet,logp;cap.data={};gc.collect()
        if i%16==15:guard(100*1024**2)
    finally:cap.close()
    finish_capture(run,out,rows,limit,key,confirmation)
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('qwen4','qwen14','glm4'),default='qwen4')
    p.add_argument('--limit',type=int,default=4);p.add_argument('--confirmation',action='store_true');a=p.parse_args()
    main(a.model,a.limit,a.confirmation)
