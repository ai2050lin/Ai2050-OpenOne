"""Bounded original-BF16 batching recovery, with explicit native shape controls.

No changed labels/materials, quantization, cached reference states, or shared
histories between samples. Batch shape is recorded, not assumed bit-invariant.
"""
import argparse,gc
from collections import defaultdict
from rdc_update_common import *
from phase2738_rdc_update_scale import material,native_prompt


class BatchTrace:
    def __init__(self,model):
        self.depth=len(model.model.layers);self.early=self.depth//3;self.block=self.depth-1
        self.enabled=False;self.handles=[];self.reset([])
        self.handles.append(model.get_input_embeddings().register_forward_hook(lambda m,a,o:self.hidden(0,o) if self.enabled else None))
        for layer,module in enumerate(model.model.layers):
            self.handles.append(module.register_forward_hook(lambda m,a,o,layer=layer:self.hidden(layer+1,o[0] if isinstance(o,tuple) else o) if self.enabled else None))
        last=model.model.layers[-1]
        def before(m,a):
            if self.enabled:self.residual=a[0]
        def attn(m,a,o):
            if self.enabled:self.factor('residual',self.residual+o[0])
        self.handles.extend([last.register_forward_pre_hook(before),last.self_attn.register_forward_hook(attn)])
        for name,module in [('x',last.post_attention_layernorm),('mlp',last.mlp)]:
            self.handles.append(module.register_forward_hook(lambda m,a,o,name=name:self.factor(name,o) if self.enabled else None))
        if hasattr(last.mlp,'gate_proj'):
            for name,module in [('gate',last.mlp.gate_proj),('up',last.mlp.up_proj)]:
                self.handles.append(module.register_forward_hook(lambda m,a,o,name=name:self.factor(name,o) if self.enabled else None))
        else:
            def gu(m,a,o):
                if self.enabled:
                    g,u=o.chunk(2,-1);self.factor('gate',g);self.factor('up',u)
            self.handles.append(last.mlp.gate_up_proj.register_forward_hook(gu))
        self.handles.append(last.mlp.down_proj.register_forward_pre_hook(lambda m,a:self.factor('activation',a[0]) if self.enabled else None))

    def reset(self,specs):
        self.specs=specs;self.fields=[{} for _ in specs];self.hs=[{} for _ in specs];self.hashes=[{} for _ in specs];self.moments=[{} for _ in specs];self.residual=None

    def factor(self,name,tensor):
        for i,r in enumerate(self.specs):self.fields[i][name]=bits(tensor[i,[r['pad']+p for p in r['positions']]])

    def hidden(self,layer,tensor):
        import torch
        for i,r in enumerate(self.specs):
            value=tensor[i,r['pad']:r['pad']+r['length']];a=bits(value);self.hs[i][layer]=a[r['positions']];self.hashes[i][layer]=identity(a)
            v=value.float();u=v/v.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
            self.moments[i][layer]=torch.stack([v.sum(0),v.square().sum(0),u.sum(0),u.square().sum(0)]).cpu().numpy()
            if layer==self.early:
                self.fields[i]['H_early_sources']=a
                self.fields[i]['history']=np.stack([u[:p+1].double().mean(0).cpu().numpy() for p in r['positions']]).astype(np.float32)

    def close(self):
        for h in self.handles:h.remove()


def main(key):
    import torch
    import accelerate.utils.offload as offload
    import rdc_operator_model as loader
    from phase2730_rdc_law_scale import Trace
    from rdc_update_scoring import score
    out=BASE/('scale_batch' if key=='qwen4' else 'scale')/key
    if (out/'result.json').exists():return
    protocol,rows=material();start=time.monotonic();guard(500*1024**2)
    pre=offload.safe_open
    def pread(*a,**kw):kw['backend']='pread';return pre(*a,**kw)
    offload.safe_open=pread;loader.snapshot=snapshot;torch.set_num_threads(2)
    recovery={'timestamp':stamp(),'source':snapshot(__file__),'original_protocol_sha256':sha(BASE/'scale/protocol.json'),
      'prefill_batch_max':8,'generation_batches':'natural8; mixed8; direct-language10; explain-language10; fixed material grouping, no outcome-based batching',
      'precision':'Original BF16, eager, per-sample positions preserved with left padding and explicit masks. No quantization.',
      'reason':'Resource recovery after observed batch1 generation214.9seconds/16steps. Reuse each offloaded weight transfer across independent sample rows. Native B1 shape comparisons are measured, never assumed exact.',
      'unchanged':['all128material IDs','all36generation IDs (Q4 shadow capture has no new generations)','native tokenizer/chat formatting','128token generation cap','all native coordinate/unit coverage','single CUDA process'],
      'numeric_boundary':'Generation histories belong to their own batch row. No original reference token or KV is injected; batch-vs-single numerical changes remain explicit.'}
    immutable(out/'batch_protocol.json',recovery)
    model,tok=loader.load(key,out/'residency',cpu_gib=6);model.eval();device=model.get_input_embeddings().weight.device
    trace=BatchTrace(model);single=Trace(model);single.enabled=False
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);pad=tok.pad_token_id
    if pad is None:pad=next(iter(stop))
    runtime={'timestamp':stamp(),'source':snapshot(__file__),'width':model.config.hidden_size,'units':model.config.intermediate_size,
      'depth':trace.depth,'early':trace.early,'last_block':trace.block,'dtype':str(model.dtype),'quantized':bool(getattr(model,'is_quantized',False)),
      'device_map':getattr(model,'hf_device_map',{}),'model':key,'execution_shape_protocol':recovery}
    save(out/'runtime.json',runtime);prepared={}
    for row in rows:
        text=native_prompt(tok,key,row);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        if row['kind']=='natural':ends=[row['token_offsets'][a][1] for a in row['anchors']]
        elif row['kind']=='controlled_language':ends=[text.index(row['body'])+len(row['body']),len(text)]
        else:ends=[len(text)]
        positions=[max(j for j,(a,b) in enumerate(enc['offset_mapping']) if b>a and b<=end) for end in ends]
        if key=='qwen4':assert enc['input_ids']==row['prompt_ids'];assert positions==row['anchors']
        prepared[row['sample_id']]={'row':row,'text':text,'ids':enc['input_ids'],'offsets':enc['offset_mapping'],'positions':positions,'ends':ends}
    kinds=('natural','controlled_program','controlled_language');first=[r for kind in kinds for r in [x for x in rows if x['kind']==kind][:2]]
    firstids={r['sample_id'] for r in first};order=first+[r for r in rows if r['sample_id'] not in firstids]
    records={};moments={};tokens=defaultdict(int);shapes=[];batch_times=[];generation_times=[]
    def pack_inputs(items,generation=False):
        ids=[x['ids'][:x['positions'][0]+1] if generation and x['row']['kind']=='natural' else x['ids'] for x in items]
        length=max(map(len,ids));mask=torch.tensor([[0]*(length-len(x))+[1]*len(x) for x in ids],device=device)
        values=torch.tensor([[pad]*(length-len(x))+x for x in ids],device=device)
        position=(mask.cumsum(-1)-1).clamp_min(0)
        return values,mask,position,ids
    try:
      with torch.inference_mode():
        for bstart in range(0,len(order),8):
            tick=time.monotonic();batch=order[bstart:bstart+8];items=[prepared[r['sample_id']] for r in batch]
            ids,mask,pos,_=pack_inputs(items);width=ids.shape[1]
            specs=[{'positions':x['positions'],'length':len(x['ids']),'pad':width-len(x['ids'])} for x in items]
            trace.reset(specs);trace.enabled=True
            post=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
            trace.enabled=False
            for i,item in enumerate(items):
                row=item['row'];sid=row['sample_id'];positions=item['positions'];offset=specs[i]['pad']
                fields=trace.fields[i]|{'H':np.stack([trace.hs[i][j] for j in range(trace.depth+1)]),
                  'postnorm':bits(post[i,[offset+p for p in positions]]),'token_ids':np.array(item['ids'],np.int32),'positions':np.array(positions)}
                logits=model.lm_head(post[i,[offset+p for p in positions]]).float();lp=logits.double().log_softmax(-1)
                rec={k:row[k] for k in ('sample_id','source_group','cohort','split','kind','language')}
                rec.update(prompt_text=item['text'],prompt_ids=item['ids'],offsets=item['offsets'],positions=positions,
                  endpoint_exact=[item['offsets'][p][1]==end for p,end in zip(positions,item['ends'])],
                  alltoken_layer_identities=trace.hashes[i],tokens=len(item['ids']),target=row.get('target'),
                  execution_batch={'size':len(batch),'padded_tokens':width,'left_padding':offset,'position_origin':0,'mask_excludes_padding':True})
                if row['kind']=='natural':
                    target=[item['ids'][p+1] for p in positions];fields['full_loss']=(-lp[torch.arange(len(positions),device=device),torch.tensor(target,device=device)]).cpu().numpy()
                    rec['first_accuracy']=float(np.mean(logits.argmax(-1).cpu().numpy()==target))
                else:
                    target=tok(row['target'],add_special_tokens=False)['input_ids'];rec['native_target_ids']=target
                    rec['first_accuracy']=int(logits[-1].argmax())==target[0];fields['full_loss']=np.array([float(-lp[-1,target[0]])])
                    texts=row.get('candidate_texts',[str(j) for j in range(1,9)]);cc=[tok(s,add_special_tokens=False)['input_ids'] for s in texts];rec['native_candidates']=cc
                    if len(target)==1 and all(len(c)==1 for c in cc):
                        candidates=[c[0] for c in cc];clp=logits[-1,candidates].double().log_softmax(-1)
                        fields.update(content_loss=np.array([float(-clp[candidates.index(target[0])])]),format_loss=np.array([float(-torch.logsumexp(lp[-1,candidates],0))]))
                        rec['conditional_accuracy']=candidates[int(clp.argmax())]==target[0]
                    else:rec['candidate_scope']='Multi-token target: no single-token content/format claim'
                fields['argmax']=logits.argmax(-1).cpu().numpy();rec['full_loss']=float(fields['full_loss'].mean())
                for name in ('content_loss','format_loss'):
                    if name in fields:rec[name]=float(fields[name][0])
                path=out/'fields'/f'{sid}.npz';npz(path,**fields);rec['array_sha256']=sha(path)
                value=np.stack([trace.moments[i][j] for j in range(trace.depth+1)]).astype(float);cohort=row['cohort']
                moments[cohort]=moments.get(cohort,np.zeros_like(value))+value;tokens[cohort]+=len(item['ids'])
                if key=='qwen4':
                    with np.load(BASE/'scale/qwen4/fields'/f'{sid}.npz') as z:
                        dh=unbits(fields['H']).astype(float)-unbits(z['H'])
                        rec['batch_vs_original_B1_H_relative_RMS']=float(np.linalg.norm(dh)/max(np.linalg.norm(unbits(z['H'])),1e-30))
                        rec['batch_vs_original_B1_H_bit_exact']=bool(np.array_equal(fields['H'],z['H']))
                save(out/'commits'/f'{sid}.json',rec);records[sid]=rec
                del fields,logits,lp,value
            batch_times.append(time.monotonic()-tick)
            del post,ids,mask,pos;trace.reset([]);gc.collect();torch.cuda.empty_cache()
            print('UPDATE_BATCH_PREFILL',key,bstart+len(batch),128,round(batch_times[-1],2),flush=True)
            # Six complete native B1 reference prefills: before any rollout.
            if bstart==0:
                for row in first:
                    item=prepared[row['sample_id']];sid=row['sample_id'];single.reset(item['positions']);single.enabled=True
                    singlepost=model.model(input_ids=torch.tensor([item['ids']],device=device),use_cache=False).last_hidden_state;single.enabled=False
                    baseline=np.stack([single.H[j] for j in range(trace.depth+1)])
                    with np.load(out/'fields'/f'{sid}.npz') as z:batched=z['H'].copy();bp=unbits(z['postnorm'])
                    difference=unbits(batched).astype(float)-unbits(baseline)
                    ref=bits(singlepost[0,item['positions']]);lp0=model.lm_head(singlepost[0,item['positions']]).double().log_softmax(-1)
                    lp1=model.lm_head(torch.tensor(bp,device=device,dtype=torch.bfloat16)).double().log_softmax(-1)
                    measured={'sample_id':sid,'batch_H_bit_exact':bool(np.array_equal(batched,baseline)),
                      'all_H_relative_RMS':float(np.linalg.norm(difference)/max(np.linalg.norm(unbits(baseline)),1e-30)),
                      'all_H_max_abs_difference':float(abs(difference).max()),'first_logits_argmax_agreement':float((lp0.argmax(-1)==lp1.argmax(-1)).double().mean()),
                      'full_vocab_KL_B1_to_batch':((lp0.exp()*(lp0-lp1)).sum(-1)).cpu().numpy().tolist()}
                    npz(out/'shape_audit'/f'{sid}.npz',B1_H=baseline,batch_H=batched,B1_postnorm=ref,batch_postnorm=bp,
                      B1_logprob=lp0.cpu().numpy(),batch_logprob=lp1.cpu().numpy())
                    shapes.append(measured);save(out/'shape_audit/result.json',{'timestamp':stamp(),'rows':shapes,'scope':'Measured execution-shape differences, not assumed zeros; independent native B1 prefills, no history injection.'})
                    del singlepost,baseline,batched,difference,ref,lp0,lp1;single.reset([])
                gc.collect();torch.cuda.empty_cache()
            guard(10*1024**2);assert time.monotonic()-start<7200
        if key!='qwen4':
            genrows=[r for r in rows if r['sample_id'] in protocol['generation_ids']]
            bins=[[r for r in genrows if r['kind']=='natural'],[r for r in genrows if r['kind']=='controlled_program']]
            bins.extend([r for r in genrows if r['kind']=='controlled_language' and r['answer_style']==style] for style in ('direct','explain'))
            assert [len(b) for b in bins]==[8,8,10,10]
            for bi,batch in enumerate(bins):
                tick=time.monotonic();items=[prepared[r['sample_id']] for r in batch];gid,mask,pos,prompts=pack_inputs(items,True)
                cache=None;generated=[[] for _ in batch];stats=[[] for _ in batch];done=[False]*len(batch)
                for step in range(128):
                    out0=model.model(input_ids=gid,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True);cache=out0.past_key_values
                    logit=model.lm_head(out0.last_hidden_state[:,-1]).float();chosen=logit.argmax(-1);lp=logit.double().log_softmax(-1)
                    for i,token in enumerate(chosen.tolist()):
                        if not done[i]:
                            generated[i].append(token);stats[i].append({'step':step,'chosen':token,'logprob':float(lp[i,token])})
                            done[i]=token in stop
                    if all(done):break
                    gid=chosen[:,None];mask=torch.cat([mask,torch.ones((len(batch),1),device=device,dtype=mask.dtype)],-1)
                    pos=(mask.sum(-1)-1)[:,None]
                    if (step+1)%16==0:
                        elapsed=time.monotonic()-tick
                        save(out/'generation_progress.json',{'timestamp':stamp(),'batch':bi,'step':step+1,'seconds':elapsed,'active':sum(not d for d in done),'all_process_seconds':time.monotonic()-start})
                        print('UPDATE_BATCH_GENERATION',key,bi,step+1,round(elapsed,2),'active',sum(not d for d in done),flush=True)
                    assert time.monotonic()-start<7200
                elapsed=time.monotonic()-tick;generation_times.append(elapsed)
                for i,row in enumerate(batch):
                    sid=row['sample_id'];rec=records[sid];text=tok.decode(generated[i],skip_special_tokens=True);graded=score(row,text,generated[i],stop,128)
                    rec.update(generation_prompt_ids=prompts[i],generated_ids=generated[i],generated_text=text,steps=stats[i],answer_scoring=graded,
                      parsed_answer=graded['conservative_final_answer'],parsed_accuracy=graded['conservative_final_correct'],EOS=graded['EOS'],censored=graded['censored'],
                      generation_execution_batch={'batch_index':bi,'size':len(batch),'left_padding':int(max(map(len,prompts))-len(prompts[i])),
                        'native_own_row_history':True,'shared_batch_seconds':elapsed,'not_independent_timing':True})
                    save(out/'commits'/f'{sid}.json',rec)
                del out0,cache,gid,mask,pos,logit,lp;gc.collect();torch.cuda.empty_cache()
        for cohort,value in moments.items():npz(out/'alltoken_cohort_moments'/f'{cohort}.npz',sums=value,tokens=np.array(tokens[cohort]))
        summaries=[]
        for cohort in sorted({r['cohort'] for r in rows}):
            rr=[r for r in records.values() if r['cohort']==cohort];gg=[r for r in rr if 'generated_ids' in r]
            report={'cohort':cohort,'rows':len(rr),'generation_rows':len(gg)}
            for metric in ('full_loss','content_loss','format_loss','first_accuracy','conditional_accuracy'):
                vv=[float(r[metric]) for r in rr if metric in r]
                if vv:report[metric]=float(np.mean(vv))
            for metric in ('parsed_accuracy','EOS','censored'):
                vv=[float(r[metric]) for r in gg if r.get(metric) is not None]
                if vv:report[metric]=float(np.mean(vv))
            summaries.append(report)
        result={'timestamp':stamp(),'source':snapshot(__file__),'model':key,'rows':128,'generation_rows':0 if key=='qwen4' else 36,
          'runtime':runtime,'summaries':summaries,'execution_shape_audit':shapes,'prefill_batch_seconds':batch_times,'generation_batch_seconds':generation_times,
          'seconds':time.monotonic()-start,'scope':'Batch native forward with all coordinates and per-row causal history; B1 differences measured. Q4 is additional matched-shape capture, preserving its original B1 collection. Cross-model differences remain confounded by tokenizer/architecture/training.'}
        save(out/'result.json',result);ledger('native_batched_scale_'+key,result['seconds']);print('UPDATE_BATCH_SCALE_DONE',key,result['seconds'],flush=True)
    except Exception as exc:failure(out/'batch_failure',start,exc);raise
    finally:
        trace.close();single.close();offload.safe_open=pre;del model,trace,single;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True);main(p.parse_args().model)
