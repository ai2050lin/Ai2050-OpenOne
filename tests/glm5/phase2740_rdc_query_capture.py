"""One million finite diagnostic suffixes, original BF16 and full vocabulary.

Every endpoint's final-normalized native vector is retained losslessly. Full
coordinate H12/H24/rawH36 and all early sources are additionally retained for the
prospectively declared576prefix detail panel. This is not all-token retention for
every main row, and fixed suffixes are not free-generated natural continuations.
"""
import argparse
from collections import defaultdict
from rdc_query_common import *

def prototype(model,tok,probes):
    import torch
    out=BASE/'prototypes';file=out/'qwen4.npz'
    if file.exists():
        assert read(out/'result.json')['archive_sha256']==sha(file)
        with np.load(file) as z:return {k:z[k].copy() for k in z.files}
    groups=defaultdict(list)
    for i,p in enumerate(probes):groups[len(p['token_ids'])].append(i)
    arrays={};post=np.zeros((100,2560),np.uint16);lp=np.zeros((100,model.config.vocab_size),np.float64);reports=[];data={}
    handles=[model.model.layers[12].self_attn.q_norm.register_forward_hook(lambda m,a,o:data.__setitem__('q_norm',o.detach()))]
    try:
      with torch.inference_mode():
        for length,indices in sorted(groups.items()):
          for begin in range(0,len(indices),16):
            batch=indices[begin:begin+16];ids=torch.tensor([probes[q]['token_ids'] for q in batch],device='cuda')
            data.clear();value=model.model(input_ids=ids,use_cache=True,output_hidden_states=True)
            z=model.lm_head(value.last_hidden_state[:,-1]).float();ll=z.double().log_softmax(-1)
            for j,q in enumerate(batch):
                post[q]=bits(value.last_hidden_state[j,-1]);lp[q]=ll[j].cpu().numpy()
                arrays.update({f'p{q}_H12':bits(value.hidden_states[12][j]),f'p{q}_H13':bits(value.hidden_states[13][j]),
                  f'p{q}_q_before_rope':bits(data['q_norm'][j]),f'p{q}_keys_after_rope':bits(value.past_key_values.layers[12].keys[j]),
                  f'p{q}_values':bits(value.past_key_values.layers[12].values[j])})
                reports.append({'probe_index':q,'probe_id':probes[q]['probe_id'],'query_ids':probes[q]['token_ids'],'batch_indices':batch,
                  'argmax':int(z[j].argmax()),'scope':'Known query string alone, with native positions0..L-1. No tested history, target or future answer used.'})
            del value,z,ll,ids
    finally:
        for h in handles:h.remove()
    arrays.update(postnorm=post,logprobs=lp);npz(file,**arrays)
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(__file__),'model':'qwen4','probes':reports,'archive_sha256':sha(file),
      'native_module_boundary':'H12 entersblock12; Q is q_proj followed by native q_norm, BEFORE RoPE. KV are native block12 after RoPE. Original positions are recorded.',
      'prediction_input_scope':'Prototype responses depend only on known query strings, not on a heldout prefix future response. This costs a fixed100-query native pass, not a learned semantic manifold.'})
    return arrays

def main(begin=0,end=2000):
    import torch
    out=BASE/'capture';finish=out/'chunks'/f'{begin:05d}_{end:05d}.json'
    if finish.exists():print('QUERY_CAPTURE_CHUNK_ALREADY_COMPLETE',begin,end);return
    protocol=read(BASE/'material/protocol.json');rows=gzread(BASE/'material/natural.json.gz');probes=read(BASE/'probes/protocol.json')['probes']
    assert 0<=begin<end<=len(rows)
    detail=set(protocol['detailed_prefix_ids']);fixtures=set(protocol['full_layer_all_token_fixture_ids'])
    start=time.monotonic();guard((end-begin)*750000+160*1024**2)
    model=None;handle=None
    try:
      model,tok=load('qwen4',out);pr=prototype(model,tok,probes);ref=torch.tensor(pr['logprobs'],device='cuda');del pr
      data={};handle=model.model.layers[-1].register_forward_hook(lambda m,a,o:data.__setitem__('raw36',o.detach()))
      groups=defaultdict(list)
      for i,p in enumerate(probes):groups[len(p['token_ids'])].append(i)
      counters={'new_prefixes':0,'reused_commits':0,'native_prefill_tokens':0,'probe_endpoints':0,'alltoken_alllayer_values_scanned':0};committed=[]
      with torch.inference_mode():
        for index in range(begin,end):
            row=rows[index];sid=row['sample_id'];commit=out/'commits'/f'{sid}.json';field=out/'fields'/f'{sid}.npz'
            if commit.exists():
                saved=read(commit);assert saved['array_sha256']==sha(field);committed.append(saved['array_sha256']);counters['reused_commits']+=1;continue
            tick=time.monotonic();prefix=row['prompt_ids'];data.clear()
            base=model.model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=True,output_hidden_states=True)
            cache=base.past_key_values;layers=list(base.hidden_states[:-1])+[data['raw36']]
            prefix_layers=[];layer_identity=[]
            for b,h in enumerate(layers):
                assert torch.isfinite(h).all();raw=bits(h[0]);prefix_layers.append(raw[-1]);layer_identity.append(identity(raw))
            prefix_layers=np.stack(prefix_layers)
            arrays={'prefix_layers':prefix_layers,'prefix_postnorm':bits(base.last_hidden_state[0,-1])}
            if sid in detail:
                arrays.update(prefix_H12_sources=bits(base.hidden_states[12][0]),prefix_block12_keys=bits(cache.layers[12].keys[0]),
                  prefix_block12_values=bits(cache.layers[12].values[0]))
            if sid in fixtures:arrays['full_prefix_layers']=np.stack([bits(h[0]) for h in layers])
            baseline_z=model.lm_head(base.last_hidden_state[0,-1]).float();baseline_lp=baseline_z.double().log_softmax(-1)
            post=np.zeros((100,2560),np.uint16);later=np.zeros((100,3,2560),np.uint16) if sid in detail else None
            stats=np.zeros((100,6),np.float64);batches=[];verified=(index%100==0)
            original_cache_identity=cache_id(cache) if verified else None
            for length,indices in sorted(groups.items()):
              for j0 in range(0,len(indices),16):
                batch=indices[j0:j0+16];ids=torch.tensor([probes[q]['token_ids'] for q in batch],device='cuda')
                own=clone_cache(cache,model.config,len(batch))
                assert all(a.keys.data_ptr()!=b.keys.data_ptr() and a.values.data_ptr()!=b.values.data_ptr() for a,b in zip(own.layers,cache.layers))
                data.clear();value=model.model(input_ids=ids,past_key_values=own,use_cache=True,output_hidden_states=sid in detail)
                h=value.last_hidden_state[:,-1];z=model.lm_head(h).float();assert torch.isfinite(z).all()
                lp=z.double().log_softmax(-1);prob=lp.exp();llref=ref[batch]
                measured=torch.stack([-(prob*lp).sum(-1),(prob*(lp-llref)).sum(-1),(llref.exp()*(llref-lp)).sum(-1),
                  (prob*(lp-baseline_lp[None])).sum(-1),(baseline_lp[None].exp()*(baseline_lp[None]-lp)).sum(-1),z.argmax(-1).double()],dim=1)
                post[batch]=bits(h);stats[batch]=measured.cpu().numpy()
                if later is not None:
                    later[batch]=np.stack([bits(value.hidden_states[12][:,-1]),bits(value.hidden_states[24][:,-1]),bits(data['raw36'][:,-1])],axis=1)
                batches.append(batch);del own,value,h,z,lp,prob,llref,measured,ids
            assert all(l.keys.shape[-2]==len(prefix) for l in cache.layers)
            if verified:assert original_cache_identity==cache_id(cache)
            arrays.update(postnorm=post,full_vocabulary_statistics=stats)
            if later is not None:arrays['query_H12_H24_rawH36']=later
            npz(field,**arrays);digest=sha(field);committed.append(digest)
            rec={'timestamp':stamp(),'sample_id':sid,'source_group':row['source_group'],'cohort':row['cohort'],'split':row['split'],'index':index,
              'prefix_tokens':len(prefix),'queries':100,'probe_protocol_sha256':sha(BASE/'probes/protocol.json'),'array_sha256':digest,
              'full_layer_prefix_identities':layer_identity,'all_saved_and_scanned_values_finite':True,'exact_suffix_batch_indices':batches,
              'source_cache_tensors_not_aliased':True,'source_cache_lengths_unchanged_all_layers':True,'source_cache_bits_checked':verified,
              'detailed':sid in detail,'full_prefix_field_fixture':sid in fixtures,
              'statistics_columns':['entropy','KL_to_query_only','KL_query_only_to_native','KL_to_unqueried_prefix','KL_unqueried_prefix_to_native','argmax_token_id'],
              'boundary':'postnorm is actual final-normalized BF16. prefix_layers hasrawH0..36. Detailed query_H12_H24_rawH36 are targets, never prefix-only features. All vocabulary statistics cover151936logits, not a selected candidate set.',
              'seconds':time.monotonic()-tick}
            save(commit,rec);counters['new_prefixes']+=1;counters['probe_endpoints']+=100;counters['native_prefill_tokens']+=len(prefix)
            counters['alltoken_alllayer_values_scanned']+=len(prefix)*37*2560
            del base,cache,layers,raw,prefix_layers,arrays,post,later,stats,baseline_z,baseline_lp
            if (index+1)%25==0 or index==begin:
                elapsed=time.monotonic()-start;print('NATURAL_QUERY_ATLAS',index+1,'/10000','chunk_seconds',round(elapsed,1),'GiB',round(usage()/2**30,3),flush=True)
                guard(25*850000);assert elapsed<read(BASE/'resources.json')['per_model_process_seconds']
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'begin':begin,'end':end,'counters':counters,
          'ordered_archive_hash':hashlib.sha256(''.join(committed).encode()).hexdigest(),'seconds':time.monotonic()-start,
          'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'precision':'OriginalBF16, eager attention, TF32disabled, exact-length no-padding suffix batches up to16; fullvocab probabilities FP64.',
          'retention':'Every100-query postnorm and prefixall-layer anchor retained. Source+multilayerquery field only576predeclared rows; fullprefixalltokenalllayer only9predeclared fixtures.'}
        save(finish,result);ledger(f'natural_query_capture_{begin}_{end}',result['seconds']);print('QUERY_CAPTURE_CHUNK_DONE',begin,end,round(result['seconds'],1),flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        if handle is not None:handle.remove()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--begin',type=int,default=0);p.add_argument('--end',type=int,default=2000);a=p.parse_args();main(a.begin,a.end)
