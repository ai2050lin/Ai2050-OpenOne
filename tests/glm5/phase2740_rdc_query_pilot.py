"""Measure all100 fixed-query costs before the main scientific sample is frozen."""
from collections import defaultdict
from rdc_query_common import *

def main():
    import torch
    from rdc_query_probes import freeze
    out=BASE/'pilot';target=out/'result.json'
    if target.exists():print('QUERY_PILOT_ALREADY_COMPLETE');return
    start=time.monotonic();guard(100*1024**2)
    natural=gzread(PRIOR/'natural_material.json.gz');law=gzread(LAW/'material.json.gz')
    rows=[next(r for r in natural if r['cohort']==c) for c in ('gum','ewt')]
    rows.append(next(r for r in law if r['kind']=='natural' and r['language']=='zh'))
    if not (out/'protocol.json').exists():immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(__file__),'sample_ids':[r['sample_id'] for r in rows],
      'material':'Three pre-existing diagnostic prefixes, excluded from formal independent confirmation.',
      'execution':'B1prefix prefill; copied own prefix cache, exact-length suffix batches up to16, no padding. Compare one B1cached/fullprefill endpoint and deterministic repeated batch. Full original cache identity before/after.'})
    save(out/'execution_source.json',{'timestamp':stamp(),'source':snapshot(__file__),'common':snapshot(ROOT/'tests/glm5/rdc_query_common.py')})
    try:model,tok=load('qwen4',out)
    except Exception as exc:failure(out,start,exc);raise
    probes=freeze(tok);groups=defaultdict(list)
    for i,p in enumerate(probes):groups[len(p['token_ids'])].append(i)
    reports=[]
    try:
      with torch.inference_mode():
        for row in rows:
            prefix=row['prompt_ids'][:row['anchors'][0]+1] if row.get('anchors') else row['prompt_ids']
            tick=time.monotonic();base=model.model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=True,output_hidden_states=True)
            cache=base.past_key_values;before=cache_id(cache);prefill=time.monotonic()-tick
            fields=np.zeros((100,4,model.config.hidden_size),np.uint16);scores=[];lp_first={};suffix_start=time.monotonic();forward_calls=0
            first_batch_check=None
            for length,indices in sorted(groups.items()):
              for k in range(0,len(indices),16):
                batch=indices[k:k+16];ids=torch.tensor([probes[j]['token_ids'] for j in batch],device='cuda')
                own=clone_cache(cache,model.config,len(batch));val=model.model(input_ids=ids,past_key_values=own,use_cache=True,output_hidden_states=True)
                h=val.last_hidden_state[:,-1];z=model.lm_head(h).float();lp=z.double().log_softmax(-1);forward_calls+=1
                for j,qi in enumerate(batch):
                    # HF hidden_states[-1] is final-normalized, unlike raw block-output hooks.
                    fields[qi]=np.stack([bits(val.hidden_states[b][j,-1]) for b in (0,12,24,36)])
                    scores.append({'probe_id':probes[qi]['probe_id'],'argmax':int(z[j].argmax()),'entropy':float(-(lp[j].exp()*lp[j]).sum())})
                    if qi in (0,1):lp_first[qi]=lp[j].cpu().numpy()
                if first_batch_check is None:
                    duplicate=model.model(input_ids=ids,past_key_values=clone_cache(cache,model.config,len(batch)),use_cache=True)
                    first_batch_check=torch.equal(h,duplicate.last_hidden_state[:,-1]);assert first_batch_check
                    del duplicate
                del own,val,h,z,lp
            seconds=time.monotonic()-suffix_start;after=cache_id(cache);assert before==after
            # Shape controls compare the same known tokenIDs; numerical differences are measured, not called semantic effects.
            probe=probes[0];cached=model.model(input_ids=torch.tensor([probe['token_ids']],device='cuda'),past_key_values=clone_cache(cache,model.config),use_cache=True)
            full=model.model(input_ids=torch.tensor([prefix+probe['token_ids']],device='cuda'),use_cache=False)
            ca=cached.last_hidden_state[0,-1];fu=full.last_hidden_state[0,-1]
            cla=model.lm_head(ca).float().double().log_softmax(-1);flp=model.lm_head(fu).float().double().log_softmax(-1)
            npz(out/'fields'/f'{row["sample_id"]}.npz',probe_H0_H12_H24_postnorm=fields,
              full_vocab_probe0=lp_first[0],full_vocab_probe1=lp_first[1])
            r={'sample_id':row['sample_id'],'source_group':row['source_group'],'language':row['language'],'prefix_ids':prefix,'prefix_tokens':len(prefix),
              'prefill_seconds':prefill,'suffix100_seconds':seconds,'suffix_forward_batches':forward_calls,'prefix_cache_unchanged_all_layers':True,
              'repeated_batch_bitwise_equal':first_batch_check,'cached_B1_vs_full_B1_relative_RMS':float((ca.float()-fu.float()).norm()/fu.float().norm()),
              'cached_B1_vs_full_B1_KL':float((cla.exp()*(cla-flp)).sum()),'cached_B1_vs_batched_KL':float((cla.exp()*(cla-torch.tensor(lp_first[0],device='cuda'))).sum()),
              'per_probe':scores,'prefix_cache_identity':before}
            save(out/'commits'/f'{row["sample_id"]}.json',r);reports.append(r)
            print('QUERY_PILOT_PREFIX',len(reports),'/3','queries100','seconds',round(seconds,3),flush=True)
            del base,cache,cached,full,ca,fu,cla,flp,fields;torch.cuda.empty_cache()
        per=float(np.mean([r['prefill_seconds']+r['suffix100_seconds'] for r in reports]))
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'prefixes':3,'probe_endpoints':300,'reports':reports,
          'mean_seconds_per_prefix100':per,'projected_576_prefix_seconds_without_other_research':576*per,
          'projected_10000_prefix_seconds_without_other_research':10000*per,
          'raw_full_vocab_FP32_10000x100_bytes':10000*100*model.config.vocab_size*4,
          'raw_H_three_layers_BF16_10000x100_bytes':10000*100*3*model.config.hidden_size*2,
          'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-start,
          'scope':'Engineering pilot only; sample reused, no new generalization claim. Estimates exclude native-path, fitting, training, long generation, crossmodel and QA costs.'}
        save(target,result);ledger('query100_engineering_pilot',result['seconds']);print('QUERY_PILOT_DONE',per,flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
