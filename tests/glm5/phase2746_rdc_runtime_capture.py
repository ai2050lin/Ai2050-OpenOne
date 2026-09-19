"""All-layer/all-unit native own-history fields, with explicit finite retention."""
import argparse
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from phase2746_rdc_runtime_contract import freeze,OUT
from rdc_runtime_observer import Observer,COORDINATE_FIELDS,UNIT_FIELDS


def generate(model,tok,observer,batch,full_steps):
    import torch
    n=len(batch);maximum=max(len(r['prompt_ids']) for r in batch)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stops=model.generation_config.eos_token_id or tok.eos_token_id
    stops=set(stops if isinstance(stops,list) else [stops])
    ids=torch.full((n,maximum),pad,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
    for b,row in enumerate(batch):
        ids[b,-len(row['prompt_ids']):]=torch.tensor(row['prompt_ids'],device='cuda');mask[b,-len(row['prompt_ids']):]=1
    positions=(mask.cumsum(-1)-1).clamp_min(0)
    result=[{'tokens':[],'hidden':[],'postnorm':[],'statistics':[],'coordinates':[],'units':[],
        'Q_before_RoPE':[],'attention':[],'appended_keys':[],'appended_values':[]} for _ in batch]
    cache=None;done=[False]*n;cache_blocks=[0,12,35];start=time.monotonic()
    for step in range(max(r['max_new_tokens'] for r in batch)):
        observer.reset(step<full_steps)
        value=model.model(input_ids=ids,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True)
        cache=value.past_key_values;h=value.last_hidden_state[:,-1]
        logits=model.lm_head(h).float();choice=logits.argmax(-1)
        lp=logits.double().log_softmax(-1);prob=lp.exp()
        statistics=torch.stack([-(prob*lp).sum(-1),prob.gather(1,choice[:,None]).squeeze(1)],-1).cpu().numpy()
        collected=observer.collect();post=bits(h)
        for b,row in enumerate(batch):
            if done[b]:continue
            r=result[b];token=int(choice[b]);r['tokens'].append(token)
            r['hidden'].append(collected['hidden'][b]);r['postnorm'].append(post[b]);r['statistics'].append(statistics[b])
            if step<full_steps:
                for name in ['coordinates','units','Q_before_RoPE']:r[name].append(collected[name][b])
                columns=np.flatnonzero(mask[b].cpu().numpy())
                assert len(columns)==len(row['prompt_ids'])+step
                r['attention'].append(collected['attention'][b].take(columns,axis=-1))
                if step==0:
                    col=torch.tensor(columns,device='cuda')
                    r['prefix_keys']=np.stack([bits(cache.layers[l].keys[b].index_select(-2,col)) for l in cache_blocks])
                    r['prefix_values']=np.stack([bits(cache.layers[l].values[b].index_select(-2,col)) for l in cache_blocks])
                else:
                    r['appended_keys'].append(np.stack([bits(cache.layers[l].keys[b,:,-1]) for l in cache_blocks]))
                    r['appended_values'].append(np.stack([bits(cache.layers[l].values[b,:,-1]) for l in cache_blocks]))
            done[b]=token in stops or len(r['tokens'])>=row['max_new_tokens']
        del value,h,logits,lp,prob,collected,post
        if all(done):break
        active=torch.tensor([not d for d in done],device='cuda',dtype=torch.long)
        mask=torch.cat([mask,active[:,None]],-1);positions=(mask.sum(-1)-1).clamp_min(0)[:,None]
        ids=choice[:,None];ids[active==0]=pad
    elapsed=time.monotonic()-start
    packets=[]
    for b,(row,r) in enumerate(zip(batch,result)):
        token_ids=r['tokens'];full_count=min(full_steps,len(token_ids))
        arrays={name:np.stack(r[name]) for name in ['hidden','postnorm','statistics','coordinates','units','Q_before_RoPE']}
        arrays.update(generated_ids=np.array(token_ids),positions=np.arange(len(token_ids))+len(row['prompt_ids'])-1,
            cache_blocks=np.array(cache_blocks),prefix_cache_keys=r['prefix_keys'],prefix_cache_values=r['prefix_values'])
        for name in ['keys','values']:
            arrays['appended_cache_'+name]=np.stack(r['appended_'+name]) if r['appended_'+name] else np.empty((0,3,model.config.num_key_value_heads,model.config.head_dim),dtype=np.uint16)
        for step,a in enumerate(r['attention']):arrays['attention_step'+str(step)]=a
        assert all(np.isfinite(unbits(a) if a.dtype==np.uint16 else a).all() for a in arrays.values())
        reference=ROOT/row['reference_field'];assert sha(reference)==row['reference_field_sha256']
        with np.load(reference) as z:
            old=unbits(z['prefix_postnorm']).astype(float)
            control={'B8_vs_original_B1_postnorm_MSE':float(np.mean((unbits(arrays['postnorm'][0]).astype(float)-old)**2)),
                'B8_vs_original_B1_allH_bit_equal':bool(np.array_equal(arrays['hidden'][0],z['prefix_layers']))}
        record={k:row[k] for k in ['sample_id','source_group','kind','cohort','family','language','split','novelty']}
        record.update(timestamp=stamp(),model='qwen4',actual_prompt_ids=row['prompt_ids'],actual_input_text=tok.decode(row['prompt_ids'],skip_special_tokens=False),
            generated_ids=token_ids,generated_text=tok.decode(token_ids,skip_special_tokens=True),
            EOS=token_ids[-1] in stops,censored=token_ids[-1] not in stops and len(token_ids)==row['max_new_tokens'],
            max_new_tokens=row['max_new_tokens'],full_field_steps=full_count,
            source_position_order='All actual prefix tokens then already-generated tokens; padding columns removed, never attention-ranked.',
            coordinate_field_names=COORDINATE_FIELDS,unit_field_names=UNIT_FIELDS,
            statistics_columns=['full_vocabulary_entropy_FP64','chosen_token_probability_FP64'],
            input_reference_sha256=row['reference_field_sha256'],first_shape_control=control,
            batch_ids=[s['sample_id'] for s in batch],allocated_batch_seconds=elapsed/n,
            boundaries='hidden H0..H36 before finalnorm; postnorm separate. Field at step t precedes emitted generated_ids[t].')
        if row['kind']=='controlled':
            old=read(OLD/'identifiability/behavior/native/commits'/(row['sample_id']+'.json'))
            assert token_ids==old['generated_ids'],('Controlled native own-history replay changed',row['sample_id'])
            record.update(pair_id=row['pair_id'],world=row['world'],case=row['case'],target=row['target'],
                answer_scoring=old['answer_scoring'],all_old_native_B8_tokens_equal=True,reused_behavior_sha256=sha(OLD/'identifiability/behavior/native/commits'/(row['sample_id']+'.json')))
        packets.append((arrays,record))
    del result,cache,ids,mask
    return packets,elapsed


def commit(packets,folder,execution):
    records=[]
    for arrays,record in packets:
        name=record['sample_id'];path=FIELD_STORE/folder/(name+'.npz');receipt=OUT/folder/'commits'/(name+'.json')
        verify_storage(sum(a.nbytes for a in arrays.values()))
        if receipt.exists():
            previous=read(receipt);assert sha(path)==previous['field_sha256']
            with np.load(path) as old:
                assert set(old.files)==set(arrays) and all(np.array_equal(old[k],a) for k,a in arrays.items())
            assert previous['generated_ids']==record['generated_ids'];records.append(previous);continue
        npz(path,**arrays)
        record.update(field_path=path.relative_to(BASE).as_posix(),field_sha256=sha(path),field_bytes=path.stat().st_size,
            decoded_field_bytes=sum(a.nbytes for a in arrays.values()),execution=execution)
        save(receipt,record);records.append(record)
    return records


def main(pilot_only):
    import torch,psutil
    protocol,rows=freeze();ancestors={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    forbidden={'phase2746_rdc_runtime_capture.py','phase2745_rdc_construction_language.py',
        'phase2745_rdc_construction_capture.py','phase2745_rdc_construction_compile.py','phase2745_rdc_construction_fit.py'}
    for process in psutil.process_iter(['pid','cmdline']):
        if process.info['pid'] not in ancestors and any(Path(a).name in forbidden for a in process.info['cmdline'] or []):
            raise RuntimeError('Another registered CUDA task is running: '+str(process.info['pid']))
    start=time.monotonic();model=None;observer=None
    execution={'source':snapshot(__file__),'observer':snapshot(Path(__file__).with_name('rdc_runtime_observer.py')),
        'common':snapshot(Path(__file__).with_name('rdc_construction_common.py')),'protocol_sha256':sha(OUT/'protocol.json')}
    try:
        model,tok=load('qwen4',OUT/'native_loader');observer=Observer(model)
        with torch.inference_mode():
            checks=[]
            for index in protocol['B1_fixture_indices']:
                row=rows[index];observer.reset(False)
                value=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=True)
                field=observer.collect()['hidden'][0];post=bits(value.last_hidden_state[0,-1])
                with np.load(ROOT/row['reference_field']) as z:
                    same=np.array_equal(field,z['prefix_layers']) and np.array_equal(post,z['prefix_postnorm'])
                assert same,('Original native B1 allfield replay differs',index,row['sample_id'])
                checks.append({'index':index,'sample_id':row['sample_id'],'all_hidden_and_postnorm_bit_equal':same})
                del value
            print('RUNTIME_B1_ADMITTED',len(checks),flush=True)
            pilot_path=OUT/'pilot/result.json'
            valid=pilot_path.exists() and read(pilot_path)['execution']==execution
            if not valid:
                batch=[rows[i] for i in protocol['pilot_batch_indices']]
                a,seconds=generate(model,tok,observer,batch,protocol['full_field_steps'])
                b,repeat_seconds=generate(model,tok,observer,batch,protocol['full_field_steps'])
                for (aa,ar),(bb,br) in zip(a,b):
                    assert ar['generated_ids']==br['generated_ids'] and set(aa)==set(bb)
                    assert all(np.array_equal(aa[k],bb[k]) for k in aa),'B8 pilot full-array replay differs'
                records=commit(a,'pilot',execution)
                projected=sum(r['field_bytes'] for r in records)/len(records)*len(rows)*1.25
                verify_storage(int(projected))
                pilot={'timestamp':stamp(),'all_passed':True,'execution':execution,'B1_checks':checks,
                    'B8_repeat_all_arrays_bit_equal':True,'records':records,'first_seconds':seconds,'repeat_seconds':repeat_seconds,
                    'projected_upper_field_bytes':projected,'projection':'Pilot32step natural rows scaled to896with25percent margin; controlled rows often shorter, but not assumed in the safety check.',
                    'all_native_unit_products_checked':observer.product_checks,'unit_product_max_error':observer.product_max_error}
                save(pilot_path,pilot);del a,b
                print('RUNTIME_PILOT_DONE',seconds,projected,flush=True)
            if pilot_only:return
            records=[]
            for begin in range(0,len(rows),8):
                batch=rows[begin:begin+8];receipts=[OUT/'main/commits'/(r['sample_id']+'.json') for r in batch]
                if all(p.exists() for p in receipts):
                    old=[read(p) for p in receipts]
                    assert all(r['execution']==execution and sha(BASE/r['field_path'])==r['field_sha256'] for r in old)
                    records+=old;continue
                packets,elapsed=generate(model,tok,observer,batch,protocol['full_field_steps'])
                records+=commit(packets,'main',execution);del packets
                progress={'timestamp':stamp(),'rows':len(records),'total':len(rows),'seconds':time.monotonic()-start,
                    'committed_field_bytes':sum(r['field_bytes'] for r in records),'native_unit_product_checks':observer.product_checks}
                save(OUT/'progress.json',progress)
                print('RUNTIME_CAPTURE',len(records),len(rows),round(progress['seconds'],1),flush=True)
            compressed(OUT/'records.json.gz',records)
            result={'timestamp':stamp(),'all_passed':True,'execution':execution,'rows':len(records),'B1_checks':checks,
                'natural_new_own_histories':sum(r['kind']=='natural' for r in records),
                'controlled_old_own_histories_replayed':sum(r['kind']=='controlled' for r in records),
                'total_generated_steps':sum(len(r['generated_ids']) for r in records),
                'total_full_field_steps':sum(r['full_field_steps'] for r in records),
                'all_native_unit_product_checks':observer.product_checks,'unit_product_max_error':observer.product_max_error,
                'field_bytes':sum(r['field_bytes'] for r in records),'seconds':time.monotonic()-start,
                'scope':protocol['scope'],'retention':protocol['fields'],'noncoverage':protocol['noncoverage']}
            save(OUT/'result.json',result);ledger('phase2746_full_runtime',result['seconds'])
            print('RUNTIME_COMPLETE',result['total_generated_steps'],result['field_bytes'],flush=True)
    except Exception as exc:
        failure(OUT,start,exc);raise
    finally:
        if observer is not None:observer.close()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
