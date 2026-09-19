"""New native own-history capture with the already-qualified passive observer."""
import argparse
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_native_tail import cuda_singleton,CUDA_TASKS
from rdc_runtime_observer import Observer,COORDINATE_FIELDS,UNIT_FIELDS
from phase2746_rdc_confirmation_material import OUT,freeze
from phase2744_rdc_query_identifiability import language_score


def generate(model,tok,observer,batch,full_steps=8):
    import torch
    n=len(batch);maximum=max(len(r['prompt_ids']) for r in batch)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop])
    ids=torch.full((n,maximum),pad,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
    for b,row in enumerate(batch):
        ids[b,-len(row['prompt_ids']):]=torch.tensor(row['prompt_ids'],device='cuda');mask[b,-len(row['prompt_ids']):]=1
    positions=(mask.cumsum(-1)-1).clamp_min(0);cache=None;done=[False]*n;blocks=[0,12,35];start=time.monotonic()
    items=[{k:[] for k in ['generated_ids','hidden','postnorm','statistics','coordinates','units','Q_before_RoPE','attention','appended_keys','appended_values']} for _ in batch]
    for step in range(max(r['max_new_tokens'] for r in batch)):
        observer.reset(step<full_steps)
        value=model.model(input_ids=ids,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True)
        cache=value.past_key_values;h=value.last_hidden_state[:,-1];logits=model.lm_head(h).float();choice=logits.argmax(-1)
        lp=logits.double().log_softmax(-1);prob=lp.exp();stats=torch.stack([-(prob*lp).sum(-1),prob.gather(1,choice[:,None]).squeeze(1)],-1).cpu().numpy()
        fields=observer.collect();post=bits(h)
        for b,row in enumerate(batch):
            if done[b]:continue
            r=items[b];token=int(choice[b]);r['generated_ids'].append(token)
            r['hidden'].append(fields['hidden'][b]);r['postnorm'].append(post[b]);r['statistics'].append(stats[b])
            if step<full_steps:
                for name in ['coordinates','units','Q_before_RoPE']:r[name].append(fields[name][b])
                columns=np.flatnonzero(mask[b].cpu().numpy());assert len(columns)==len(row['prompt_ids'])+step
                r['attention'].append(fields['attention'][b].take(columns,axis=-1))
                if step==0:
                    col=torch.tensor(columns,device='cuda')
                    for kind in ['keys','values']:r['prefix_'+kind]=np.stack([bits(getattr(cache.layers[l],kind)[b].index_select(-2,col)) for l in blocks])
                else:
                    for kind in ['keys','values']:r['appended_'+kind].append(np.stack([bits(getattr(cache.layers[l],kind)[b,:,-1]) for l in blocks]))
            done[b]=token in stop or len(r['generated_ids'])>=row['max_new_tokens']
        del value,h,logits,lp,prob,fields,post
        if all(done):break
        active=torch.tensor([not d for d in done],device='cuda',dtype=torch.long);mask=torch.cat([mask,active[:,None]],-1)
        positions=(mask.sum(-1)-1).clamp_min(0)[:,None];ids=choice[:,None];ids[active==0]=pad
    elapsed=time.monotonic()-start;packets=[]
    for row,r in zip(batch,items):
        arrays={k:np.stack(r[k]) for k in ['hidden','postnorm','statistics','coordinates','units','Q_before_RoPE']}
        tokens=r['generated_ids'];arrays.update(generated_ids=np.array(tokens),positions=np.arange(len(tokens))+len(row['prompt_ids'])-1,cache_blocks=np.array(blocks))
        for kind in ['keys','values']:
            arrays['prefix_cache_'+kind]=r['prefix_'+kind]
            arrays['appended_cache_'+kind]=np.stack(r['appended_'+kind]) if r['appended_'+kind] else np.empty((0,3,8,128),np.uint16)
        for step,a in enumerate(r['attention']):arrays['attention_step'+str(step)]=a
        assert all(np.isfinite(unbits(a) if a.dtype==np.uint16 else a).all() for a in arrays.values())
        record={k:row[k] for k in ['sample_id','source_group','kind','cohort','family','language','split','novelty']}
        record.update(timestamp=stamp(),model='qwen4',actual_prompt_ids=row['prompt_ids'],generated_ids=tokens,
            actual_input_text=tok.decode(row['prompt_ids'],skip_special_tokens=False),generated_text=tok.decode(tokens,skip_special_tokens=True),
            EOS=tokens[-1] in stop,censored=tokens[-1] not in stop and len(tokens)==row['max_new_tokens'],max_new_tokens=row['max_new_tokens'],
            full_field_steps=min(full_steps,len(tokens)),coordinate_field_names=COORDINATE_FIELDS,unit_field_names=UNIT_FIELDS,
            batch_ids=[s['sample_id'] for s in batch],allocated_batch_seconds=elapsed/n,
            boundaries='Field at t precedes emittedtoken[t]; H0..H36 pre-finalnorm and postnorm separate. All source positions retained; prefixcache includes current prefix-final token, so exclude it when forecasting step0.')
        if row['kind']=='controlled':
            record.update({k:row[k] for k in ['pair_id','world','case','target']})
            record['answer_scoring']=language_score(dict(row,kind='controlled_language'),record['generated_text'],tokens,stop,row['max_new_tokens'])
        packets.append((arrays,record))
    del items,cache,ids,mask;return packets,elapsed


def commit(packets,mode,execution):
    records=[]
    for arrays,rec in packets:
        path=FIELD_STORE/('history_confirmation_'+mode)/(rec['sample_id']+'.npz');receipt=OUT/'native'/mode/'commits'/(rec['sample_id']+'.json')
        if receipt.exists():
            old=read(receipt);assert old['execution']==execution and sha(path)==old['field_sha256']
            with np.load(path) as z:assert set(z.files)==set(arrays) and all(np.array_equal(z[k],a) for k,a in arrays.items())
            records.append(old);continue
        verify_storage(sum(a.nbytes for a in arrays.values()));npz(path,**arrays)
        rec.update(field_path=path.relative_to(BASE).as_posix(),field_sha256=sha(path),field_bytes=path.stat().st_size,execution=execution)
        save(receipt,rec);records.append(rec)
    return records


def main(pilot_only=False):
    import torch
    cuda_singleton(CUDA_TASKS|{Path(__file__).name});protocol,rows=freeze();start=time.monotonic();model=None;observer=None
    execution={'source':snapshot(__file__),'observer':snapshot(Path(__file__).with_name('rdc_runtime_observer.py')),'protocol_sha256':sha(OUT/'protocol.json')}
    try:
        model,tok=load('qwen4',OUT/'native/loader');observer=Observer(model)
        with torch.inference_mode():
            # Admission against existing observed B1 fixtures, without modifying the new confirmation inputs.
            from phase2746_rdc_runtime_contract import freeze as oldfreeze
            oldp,oldrows=oldfreeze();checks=[]
            for index in oldp['B1_fixture_indices']:
                row=oldrows[index];observer.reset(False)
                value=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=True)
                field=observer.collect()['hidden'][0];post=bits(value.last_hidden_state[0,-1])
                with np.load(ROOT/row['reference_field']) as z:assert np.array_equal(field,z['prefix_layers']) and np.array_equal(post,z['prefix_postnorm'])
                checks.append(row['sample_id']);del value
            pilot=[rows[i] for i in [0,24,80,191,192,256,384,511]]
            a,seconds=generate(model,tok,observer,pilot);b,repeat=generate(model,tok,observer,pilot)
            for (aa,ar),(bb,br) in zip(a,b):
                assert ar['generated_ids']==br['generated_ids'] and set(aa)==set(bb) and all(np.array_equal(aa[k],bb[k]) for k in aa)
            pr=commit(a,'pilot',execution);projected=sum(r['field_bytes'] for r in pr)/8*512*1.25;verify_storage(int(projected))
            save(OUT/'native/pilot/result.json',{'timestamp':stamp(),'all_passed':True,'execution':execution,'B1_old_fullfield_admissions':checks,
                'new_B8_repeat_all_arrays_bit_equal':True,'projected_field_bytes_with25percent_margin':projected,'first_seconds':seconds,'repeat_seconds':repeat})
            del a,b;print('CONFIRMATION_NATIVE_PILOT',seconds,projected,flush=True)
            if pilot_only:return
            records=[]
            for begin in range(0,len(rows),8):
                batch=rows[begin:begin+8];paths=[OUT/'native/main/commits'/(r['sample_id']+'.json') for r in batch]
                if all(p.exists() for p in paths):
                    rr=[read(p) for p in paths];assert all(r['execution']==execution and sha(BASE/r['field_path'])==r['field_sha256'] for r in rr);records+=rr
                else:
                    packets,_=generate(model,tok,observer,batch);records+=commit(packets,'main',execution);del packets
                save(OUT/'native/progress.json',{'timestamp':stamp(),'rows':len(records),'total':512,'seconds':time.monotonic()-start})
                print('CONFIRMATION_NATIVE',len(records),512,round(time.monotonic()-start,1),flush=True)
            compressed(OUT/'native/records.json.gz',records)
            result={'timestamp':stamp(),'all_passed':True,'execution':execution,'rows':len(records),
                'generated_steps':sum(len(r['generated_ids']) for r in records),'full_field_steps':sum(r['full_field_steps'] for r in records),
                'field_bytes':sum(r['field_bytes'] for r in records),'all_native_unit_products_checked':observer.product_checks,
                'product_max_error':observer.product_max_error,'seconds':time.monotonic()-start,
                'scope':'New original-model histories only. Scoring frozen extracted algorithms and self-fed deployment remain separate.'}
            save(OUT/'native/result.json',result);ledger('phase2746_confirmation_native',result['seconds']);print('CONFIRMATION_NATIVE_COMPLETE',result,flush=True)
    except Exception as exc:failure(OUT/'native',start,exc);raise
    finally:
        if observer is not None:observer.close()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');main(p.parse_args().pilot)
