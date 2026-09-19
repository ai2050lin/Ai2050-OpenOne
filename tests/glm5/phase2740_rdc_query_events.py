"""Native cached replay at actual generated event times and ordered native paths."""
import argparse,re
from rdc_query_common import *

OUT=BASE/'events'

def event_rows(row,commit,tok):
    ids=commit['generated_ids'];decoded=[tok.decode(ids[:j],skip_special_tokens=True) for j in range(1,len(ids)+1)]
    text=decoded[-1];events=[]
    # All syntactically explicit assignments, not only correct ones. Annotation only.
    pattern=r'(?P<variable>v_\d+_\d+)\s*(?:=|is|的值为|的值是)\s*\*{0,2}(?P<value>[1-8])\b'
    for m in re.finditer(pattern,text):
        step=next((i for i,t in enumerate(decoded) if len(t)>=m.end()),len(ids)-1)
        events.append({'type':'explicit_variable_value','variable':m['variable'],'value':m['value'],
          'character_span':[m.start(),m.end()],'emitted_token_step':step,'quote':m.group(0)})
    for m in re.finditer(r'(?i)(?:final\s+answer|answer\s*[:：]|最终答案|答案\s*[:：]|\\boxed\{)',text):
        step=next((i for i,t in enumerate(decoded) if len(t)>=m.end()),len(ids)-1)
        events.append({'type':'terminal_marker','character_span':[m.start(),m.end()],'emitted_token_step':step,'quote':m.group(0)})
    anchors={0,len(ids)-1,*[j for j in [32,64,128,256] if j<len(ids)]};seen=set()
    for e in sorted(events,key=lambda e:e['emitted_token_step']):
        key=(e['type'],e.get('variable'))
        if key not in seen:anchors.add(e['emitted_token_step']);seen.add(key)
    return events,sorted(anchors)

def freeze():
    p=OUT/'protocol.json'
    if p.exists():return read(p),gzread(OUT/'material.json.gz')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    old=gzread(PRIOR/'long_answers/material.json.gz');rows=[]
    for row in old:
        file=PRIOR/'long_answers/commits/native'/f"{row['sample_id']}.json";commit=read(file)
        events,anchors=event_rows(row,commit,tok)
        rows.append({'row':row,'native_record':str(file.relative_to(ROOT)),'native_sha256':sha(file),'events':events,'anchors':anchors,
          'source_path_steps':[0,anchors[-1]] if len(rows)<4 else []})
    detail=set(read(BASE/'material/protocol.json')['detailed_prefix_ids'])
    natural=[r for r in gzread(BASE/'material/natural.json.gz') if r['sample_id'] in detail]
    pathrows=[r for cohort in ['gum','ewt','cmrc'] for split in ['train','test'] for r in [x for x in natural if x['cohort']==cohort and x['split']==split][:2]]
    value={'timestamp':stamp(),'source':snapshot(__file__),'native_trajectories':32,'original_groups':8,
      'event_anchors':sum(len(r['anchors']) for r in rows),'event_annotations':sum(len(r['events']) for r in rows),
      'full_coordinates':'All37 raw layer anchors and block16/35gate/up/activation at each saved actual generation step.',
      'ordered_paths':'First4 old native expressions at prompt-end and actual terminal step, plus12 natural sources with fixed probe indices0,50,99; native blocks16,35. All source positions, all9728units and all2560coordinates, factorized without dropping any source or unit.',
      'natural_path_ids':[r['sample_id'] for r in pathrows],'natural_query_indices':[0,50,99],
      'timing':'Step j state predicts generated token j, before appending it to history. Regex events annotate output text, not an internal symbolic execution proof.',
      'scope':'Old32native trajectories are same-material numerical replay, not independent behavior confirmation. First syntactic assignment per variable is an anchor regardless of correctness.',
      'prefixes':'Cached B1 prefill and single-token updates match old native execution; all IDs must match. No teacher-forced full-sequence equivalence claim.'}
    compressed(OUT/'material.json.gz',{'trajectories':rows,'natural':pathrows});immutable(p,value)
    return value,{'trajectories':rows,'natural':pathrows}

class Observer:
    def __init__(self,model):
        self.model=model;self.active=False;self.data={};self.handles=[]
        def put(key,value):
            if self.active:self.data[key]=value.detach()
        self.handles.append(model.model.layers[-1].register_forward_hook(lambda m,a,o:put('raw36',o)))
        for b in [16,35]:
            layer=model.model.layers[b]
            self.handles.append(layer.register_forward_pre_hook(lambda m,a,b=b:put(f'{b}_residual',a[0][0,-1])))
            self.handles.append(layer.self_attn.register_forward_hook(lambda m,a,o,b=b:(put(f'{b}_attn',o[0][0,-1]),put(f'{b}_A',o[1][0,:,-1])) and None))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a,b=b:put(f'{b}_rms_input',a[0][0,-1])))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(lambda m,a,o,b=b:put(f'{b}_x',o[0,-1])))
            for name in ['gate_proj','up_proj','down_proj']:
                self.handles.append(getattr(layer.mlp,name).register_forward_hook(lambda m,a,o,b=b,name=name:put(f'{b}_{name}',o[0,-1])))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m,a,b=b:put(f'{b}_activation',a[0][0,-1])))
    def reset(self,active):self.active=active;self.data={}
    def close(self):
        for h in self.handles:h.remove()

def path_arrays(model,obs,cache):
    import torch
    arrays={};reports=[]
    for b in [16,35]:
        layer=model.model.layers[b];d=obs.data;a=d[f'{b}_A'].float();v=cache.layers[b].values[0].float().repeat_interleave(4,dim=0)
        wo=layer.self_attn.o_proj.weight.float().reshape(2560,32,128)
        c=torch.einsum('hs,hsf,dhf->sd',a,v,wo)
        r=d[f'{b}_rms_input'].float();den=(r.square().mean()+layer.post_attention_layernorm.variance_epsilon).sqrt()
        xs=c*layer.post_attention_layernorm.weight.float()/den
        gs=xs@layer.mlp.gate_proj.weight.float().T;us=xs@layer.mlp.up_proj.weight.float().T
        g=d[f'{b}_gate_proj'].float();u=d[f'{b}_up_proj'].float();sig=g.sigmoid()
        gg=torch.cat([gs,(g-gs.sum(0))[None]]);uu=torch.cat([us,(u-us.sum(0))[None]])
        pair=(gg*sig)@uu.T;energy=(gg.square()*sig.square())@uu.square().T
        ordered_sum=(gg*sig).sum(0)*uu.sum(0);native_act=d[f'{b}_activation'].float()
        anti=.5*sig.square()*(gg.square().sum(0)*uu.square().sum(0)-(gg*uu).sum(0).square())
        sumerr=float((ordered_sum-sig*g*u).norm()/(sig*g*u).norm())
        arrays.update({f'L{b}_'+k:x.cpu().numpy() for k,x in {
          'source_attention_write':c,'source_gate_read':gg,'source_up_read':uu,'gate':g,'up':u,'sigmoid_gate':sig,
          'ordered_pair_all_unit_sum':pair,'ordered_pair_all_unit_energy':energy,'unit_antisymmetric_energy':anti,
          'activation_rounding_remainder':native_act-ordered_sum,'activation':native_act,'MLP':d[f'{b}_down_proj'].float(),
          'attention_rounding_remainder':d[f'{b}_attn'].float()-c.sum(0),'native_A':a,'native_V':cache.layers[b].values[0].float(),
          'rms_input':r,'rms_denominator':den,'residual':d[f'{b}_residual'].float()}.items()})
        reports.append({'block':b,'visible_sources':a.shape[-1],'other_source_index':gg.shape[0]-1,
          'all_unit_sum_relative_error':sumerr,'attention_FP32_vs_native_relative_error':float((d[f'{b}_attn'].float()-c.sum(0)).norm()/d[f'{b}_attn'].float().norm())})
        assert sumerr<1e-5
    return arrays,reports

def main():
    import torch
    from rdc_query_dynamic import QueryEngine
    protocol,material=freeze()
    if (OUT/'result.json').exists():return
    assert read(OUT/'alignment_audit.json')['all_passed']
    start=time.monotonic();guard(900*1024**2);model=None;obs=None
    try:
      immutable(OUT/'dynamic_protocol.json',{'anchor_protocol_sha256':sha(OUT/'protocol.json'),'probes':100,
        'anchor_count':protocol['event_anchors'],'scope':'At every predeclared actual-time anchor, append100fixeddiagnostic queries to copied own native cache; all coordinate endpoints and full vocabulary divergences retained. Native next emitted token still comes from unmodified original history.',
        'source':snapshot(Path(__file__).with_name('rdc_query_dynamic.py'))})
      model,tok=load('qwen4',OUT);obs=Observer(model);engine=QueryEngine(model);records=[];paths=[]
      digit_ids=[tok(str(j),add_special_tokens=False)['input_ids'][0] for j in range(1,9)]
      with torch.inference_mode():
        for i,item in enumerate(material['trajectories']):
            row=item['row'];sid=row['sample_id'];file=OUT/'commits'/f'{sid}.json'
            if file.exists():records.append(read(file));continue
            old=read(ROOT/item['native_record']);assert sha(ROOT/item['native_record'])==item['native_sha256']
            anchors=set(item['anchors']);source_steps=set(item['source_path_steps']);cache=None
            ids=torch.tensor([row['prompt_ids']],device='cuda');hs=[];units=[];trace=[];emitted=[];query_post=[];query_stats=[]
            for step,want in enumerate(old['generated_ids']):
                obs.reset(step in anchors);v=model.model(input_ids=ids,past_key_values=cache,use_cache=True,output_hidden_states=step in anchors);cache=v.past_key_values
                z=model.lm_head(v.last_hidden_state[0,-1]).float();lp=z.double().log_softmax(-1);chosen=int(z.argmax());emitted.append(chosen)
                assert chosen==want,('Native replay diverged',sid,step,chosen,want)
                trace.append({'step':step,'token_id':chosen,'entropy':float(-(lp.exp()*lp).sum()),'selected_logprob':float(lp[chosen]),
                  'digit_mass':float(lp[digit_ids].exp().sum())})
                if step in anchors:
                    hs.append(np.stack([bits(h[0,-1]) for h in list(v.hidden_states[:-1])+[obs.data['raw36']]]))
                    units.append(np.stack([bits(obs.data[f'{b}_{name}']) for b in [16,35] for name in ['gate_proj','up_proj','activation']]))
                if step in source_steps:
                    arr,audits=path_arrays(model,obs,cache);pf=OUT/'paths'/f'{sid}_step{step}.npz';npz(pf,**arr)
                    itempath={'sample_id':sid,'step':step,'path':str(pf.relative_to(BASE)),'sha256':sha(pf),'audits':audits,'mode':'old_native_generated'}
                    save(OUT/'ordered_commits'/f'{sid}_step{step}.json',itempath);paths.append(itempath)
                if step in anchors:
                    obs.reset(False);qp,qs=engine.run(cache,verify=step==0);query_post.append(qp);query_stats.append(qs)
                ids=torch.tensor([[chosen]],device='cuda');del v,z,lp
            fp=OUT/'fields'/f'{sid}.npz';npz(fp,H=np.stack(hs),all_units=np.stack(units),steps=np.array(item['anchors']),
              dynamic_query_postnorm=np.stack(query_post),dynamic_full_vocabulary_statistics=np.stack(query_stats))
            record={'sample_id':sid,'source_group':row['source_group'],'representation':row['representation'],'steps':trace,
              'events':item['events'],'anchors':item['anchors'],'archive_sha256':sha(fp),'all_generated_IDs_exact':True,'native_steps':len(emitted)}
            save(file,record);records.append(record);del cache,hs,units;guard()
            print('QUERY_EVENT_REPLAY',i+1,32,round(time.monotonic()-start,1),flush=True)
        probes=read(BASE/'probes/protocol.json')['probes']
        for row in material['natural']:
          for q in protocol['natural_query_indices']:
            sid=row['sample_id'];pf=OUT/'paths'/f'{sid}_probe{q}.npz';cp=OUT/'natural_commits'/f'{sid}_probe{q}.json'
            if cp.exists():paths.append(read(cp));continue
            obs.reset(False);pre=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=True)
            obs.reset(True);post=model.model(input_ids=torch.tensor([probes[q]['token_ids']],device='cuda'),past_key_values=pre.past_key_values,use_cache=True)
            arr,audits=path_arrays(model,obs,post.past_key_values);npz(pf,**arr)
            r={'sample_id':sid,'probe_index':q,'path':str(pf.relative_to(BASE)),'sha256':sha(pf),'audits':audits,'mode':'new_natural_fixed_query'}
            save(cp,r);paths.append(r);del pre,post,arr;guard()
          print('QUERY_NATURAL_PATH',sid,round(time.monotonic()-start,1),flush=True)
        paths=[read(p) for area in ['ordered_commits','natural_commits'] for p in sorted((OUT/area).glob('*.json'))]
        assert len(paths)==44 and all(sha(BASE/r['path'])==r['sha256'] for r in paths)
        save(OUT/'paths_index.json',paths)
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'native_trajectories':len(records),
          'native_generated_tokens':sum(r['native_steps'] for r in records),'event_anchors':sum(len(r['anchors']) for r in records),'source_path_records':len(paths),
          'dynamic_query_endpoints':100*sum(len(r['anchors']) for r in records),
          'all_cached_native_IDs_reproduced':True,'seconds':time.monotonic()-start,'precision':'Native BF16; provenance contraction FP32 with explicit remainders.',
          'scope':'Directed gate/up source allocation retains orientation at fixed factors, while double-source summation cancels antisymmetric parts. This is exact algebraic accounting, not proof of causal linguistic roles.'}
        save(OUT/'result.json',result);ledger('native_generation_event_and_ordered_paths',result['seconds']);print('QUERY_EVENTS_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(OUT,start,exc);raise
    finally:
        if obs is not None:obs.close()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print('EVENT_PROTOCOL',freeze()[0])
    else:main()
