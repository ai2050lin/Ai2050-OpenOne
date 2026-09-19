"""Original BF16, own-tokenizer query response replication, serial CUDA loading."""
import argparse,sys
from collections import defaultdict
from rdc_query_common import *

OUT=BASE/'scale'

def freeze():
    p=OUT/'protocol.json'
    if p.exists():return read(p),gzread(OUT/'material.json.gz')
    rows=gzread(BASE/'material/natural.json.gz');pool={};chosen=[]
    for c in ['gum','ewt','cmrc']:
        rr=sorted([r for r in rows if r['cohort']==c and r['split']=='test'],key=lambda r:rank('scale/'+r['sample_id']))
        unique={}
        for r in rr:unique.setdefault(r['source_group'],r)
        pool[c]=list(unique.values())[:22]
    for j in range(22):
        for c in ['gum','ewt','cmrc']:
            if len(chosen)<64:chosen.append(pool[c][j])
    value={'timestamp':stamp(),'source':snapshot(__file__),'models':['qwen4','qwen14','glm4'],'maximum_sources':64,'query_strings':100,
      'source_ids':[r['sample_id'] for r in chosen],'query_batch_max':16,
      'precision':'Original BF16 local checkpoints, device_map auto/CPU/disk-reference offload when necessary. One model process at a time; no quantization.',
      'adaptive_cost_rule':'First3predeclared balanced sources estimate full64 cost. Pick largest of64,32,16 whose projected source capture fits3200seconds. At least16 unless resource audit fails; compare models on shared available intersection only.',
      'retention':'All own coordinates for all-layer prefix last anchors,100query postnorm endpoints, early prefix native KV; first3sources additionally full native query Q beforeRoPE and actual all-source attention for every probe.',
      'numerical_controls':'Exact-length query batches, no suffix padding, copied own cache; Q4repeatB16 compared against main atlas. Cross-model common text is retokenized natively, not token-ID aligned.',
      'interpretation':'Query effect distributions and full-coordinate within-model Gram relations; no equal coordinate indices across models, no retraining larger models, no causal attribution to size.'}
    compressed(OUT/'material.json.gz',chosen);immutable(p,value);return value,chosen

def main(key):
    import torch
    import rdc_operator_model as loader
    protocol,rows=freeze();out=OUT/key
    if (out/'result.json').exists():return
    start=time.monotonic();model=None;handles=[];guard(400*1024**2)
    try:
      # Higher residency is permitted only after the same tested commit-headroom audit.
      loader.snapshot=snapshot
      model,tok=loader.load(key,out/'residency',cpu_gib=11) if key=='qwen14' else load(key,out/'residency')
      torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
      assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
      depth=len(model.model.layers);early=depth//3;att=model.model.layers[early].self_attn;D=model.config.hidden_size
      probes=read(BASE/'probes/protocol.json')['probes'];tokenized=[tok(p['text'],add_special_tokens=False)['input_ids'] for p in probes]
      groups=defaultdict(list)
      for i,ids in enumerate(tokenized):groups[len(ids)].append(i)
      data={};active=[False]
      def put(k,v):
        if active[0]:data[k]=v.detach()
      handles.append(model.model.layers[-1].register_forward_hook(lambda m,a,o:put('raw_last',o[0] if isinstance(o,tuple) else o)))
      module=att.q_norm if hasattr(att,'q_norm') else att.q_proj
      handles.append(module.register_forward_hook(lambda m,a,o:put('q',o)))
      handles.append(att.register_forward_hook(lambda m,a,o:put('A',o[1])))
      def rotary_inputs(m,a,kw):
          put('cos',kw['position_embeddings'][0]);put('sin',kw['position_embeddings'][1])
      handles.append(att.register_forward_pre_hook(rotary_inputs,with_kwargs=True))
      prototype=np.empty((100,D),np.uint16);proto_lp=torch.zeros((100,model.config.vocab_size),dtype=torch.float64,device='cuda')
      with torch.inference_mode():
        for length,indices in sorted(groups.items()):
          for j0 in range(0,len(indices),16):
            batch=indices[j0:j0+16];o=model.model(input_ids=torch.tensor([tokenized[q] for q in batch],device='cuda'),use_cache=False)
            h=o.last_hidden_state[:,-1];prototype[batch]=bits(h);proto_lp[batch]=model.lm_head(h).float().double().log_softmax(-1);del o,h
        npz(out/'query_only.npz',postnorm=prototype)
        save(out/'tokenization.json',{'queries':tokenized,'original_probe_ids':[p['probe_id'] for p in probes],'model':key})
        records=[];times=[];selected=64;shape_checks=[];native_attention_checks=[]
        for i,row in enumerate(rows):
            if i>=selected:break
            sid=row['sample_id'];file=out/'fields'/f'{sid}.npz';cp=out/'commits'/f'{sid}.json'
            if cp.exists():
                r=read(cp);assert sha(file)==r['sha256'];records.append(r);times.append(r['seconds'])
                native_attention_checks.extend(r.get('native_attention_checks',[]))
                if r.get('shape_control') is not None:shape_checks.append(r['shape_control'])
                if i==2:selected=read(out/'cost_decision.json')['selected_sources']
                continue
            tick=time.monotonic();prefix=tok(row['text'],add_special_tokens=False)['input_ids'];active[0]=True;data.clear();ck=None;check_begin=len(native_attention_checks)
            pre=model.model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=True,output_hidden_states=True)
            cache=pre.past_key_values;hs=list(pre.hidden_states[:-1])+[data['raw_last']]
            arrays={'prefix_layers':np.stack([bits(h[0,-1]) for h in hs]),'prefix_early_keys':bits(cache.layers[early].keys[0]),
              'prefix_early_values':bits(cache.layers[early].values[0])}
            assert len(hs)==depth+1
            saved_cache=cache_id(cache) if i<3 else None;post=np.empty((100,D),np.uint16);stats=np.zeros((100,4));qparts={}
            for length,indices in sorted(groups.items()):
              for j0 in range(0,len(indices),16):
                batch=indices[j0:j0+16];own=clone_cache(cache,model.config,len(batch));data.clear()
                o=model.model(input_ids=torch.tensor([tokenized[q] for q in batch],device='cuda'),past_key_values=own,use_cache=True)
                h=o.last_hidden_state[:,-1];z=model.lm_head(h).float();lp=z.double().log_softmax(-1);prob=lp.exp();ref=proto_lp[batch]
                stats[batch]=torch.stack([-(prob*lp).sum(-1),(prob*(lp-ref)).sum(-1),(ref.exp()*(ref-lp)).sum(-1),z.argmax(-1).double()],-1).cpu().numpy();post[batch]=bits(h)
                if i<3:
                    qbare=data['q'].reshape(len(batch),length,-1,att.head_dim).transpose(1,2)
                    apply_rope=sys.modules[att.__class__.__module__].apply_rotary_pos_emb
                    rotated,_=apply_rope(qbare,torch.zeros_like(qbare),data['cos'],data['sin'])
                    kk=own.layers[early].keys.repeat_interleave(att.num_key_value_groups,dim=1)
                    scores=(rotated@kk.transpose(-2,-1))*att.scaling
                    manual=scores[:,:,-1].float().softmax(-1).to(torch.bfloat16);actual=data['A'][:,:,-1]
                    err=float((manual.float()-actual.float()).abs().max())
                    native_attention_checks.append({'sample_id':sid,'batch':batch,'max_abs_error':err,'bit_equal':bool(torch.equal(manual,actual)),
                      'RoPE_scope':'Actual own-architecture Q/K rotation only; native Qnorm if present, GLM partial/interleaved rotation remains native.'})
                    assert err<.005,('Native Q/K/attention reconstruction beyond declared BF16 tolerance',key,sid,err)
                    for j,q in enumerate(batch):
                        qparts[f'probe{q}_q_before_rope']=bits(data['q'][j,-1])
                        qparts[f'probe{q}_attention']=bits(data['A'][j,:,-1])
                del own,o,h,z,lp,prob,ref
            if i<3:assert saved_cache==cache_id(cache)
            arrays.update(postnorm=post,full_vocabulary_statistics=stats,**qparts);npz(file,**arrays)
            if key=='qwen4':
                with np.load(BASE/'capture/fields'/f'{sid}.npz') as old:
                    ck={'sample_id':sid,'prefix_ids_equal':prefix==row['prompt_ids'],'prefix_all_layers_bit_equal':np.array_equal(arrays['prefix_layers'],old['prefix_layers']),
                      'query_postnorm_bit_equal':np.array_equal(post,old['postnorm']),'KL_max_error':float(abs(stats[:,1]-old['full_vocabulary_statistics'][:,1]).max())}
                assert all(ck[k] for k in ['prefix_ids_equal','prefix_all_layers_bit_equal','query_postnorm_bit_equal']);shape_checks.append(ck)
            r={'sample_id':sid,'source_group':row['source_group'],'cohort':row['cohort'],'prefix_ids':prefix,'prefix_tokens':len(prefix),
              'queries':100,'sha256':sha(file),'all_source_cache_lengths_unchanged':all(l.keys.shape[-2]==len(prefix) for l in cache.layers),
              'source_cache_bits_verified':i<3,'seconds':time.monotonic()-tick,
              'shape_control':ck,'native_attention_checks':native_attention_checks[check_begin:]}
            assert r['all_source_cache_lengths_unchanged'];save(cp,r);records.append(r);times.append(r['seconds'])
            del pre,cache,hs,arrays,post,stats,qparts;data.clear();torch.cuda.empty_cache()
            if i==2:
                rate=float(np.mean(times));eligible=[n for n in [64,32,16] if n*rate<3200]
                assert eligible,('No planned replication panel within per-model source budget',rate)
                selected=eligible[0];save(out/'cost_decision.json',{'timestamp':stamp(),'first3_source_seconds':times,'projected64seconds':64*rate,'selected_sources':selected,
                  'rule':'Frozen3200second source budget; decision uses time only, not outcomes.'})
            print('QUERY_SCALE',key,i+1,selected,'seconds',round(time.monotonic()-start,1),flush=True);guard();assert time.monotonic()-start<7200
        effect=np.stack([np.load(out/'fields'/f"{r['sample_id']}.npz")['full_vocabulary_statistics'][:,1] for r in records])
        # Descriptive own-coordinate kernel over every response coordinate; all100queries retained.
        gram=np.zeros((100,100));energy=np.zeros((100,D))
        for r in records:
            with np.load(out/'fields'/f"{r['sample_id']}.npz") as z:h=unbits(z['postnorm']).astype(float)
            center=h-h.mean(0);gram+=center@center.T/D;energy+=center.square() if hasattr(center,'square') else center**2
        npz(out/'all_coordinate_query_geometry.npz',centered_query_gram=gram/len(records),full_coordinate_query_energy=energy/len(records),query_KL=effect)
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,'sources':len(records),'queries':len(records)*100,'depth':depth,'width':D,
          'early':early,'head_dim':att.head_dim,'q_norm_used':hasattr(att,'q_norm'),'rope_parameters':getattr(model.config,'rope_parameters',None),
          'native_dtype':str(model.dtype),'quantized':False,'actual_parameter_devices':sorted({str(p.device) for p in model.parameters()}),
          'device_map':getattr(model,'hf_device_map',{}),'shape_controls':shape_checks,'native_QK_RoPE_attention_checks':native_attention_checks,
          'seconds':time.monotonic()-start,'mean_query_KL':float(effect.mean()),
          'scope':'Full-vocabulary changes are within each native tokenizer. Full-coordinate Gram is within-model only; neither cross-model axis matching nor manifold equivalence is claimed.'}
        save(out/'result.json',result);ledger('original_model_query_replication_'+key,result['seconds']);print('QUERY_SCALE_DONE',key,result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        for h in handles:h.remove()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',nargs='?',choices=['qwen4','qwen14','glm4']);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print('SCALE_PROTOCOL',freeze()[0])
    else:main(a.model)
