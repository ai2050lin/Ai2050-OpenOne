"""Paired relational decisions under each actual trained model's own history."""
from phase2744_rdc_query_identifiability import *


def main():
    import torch
    from rdc_query_scoring import checks
    protocol,material=freeze();out=OUT/'behavior';finish=out/'result.json'
    if finish.exists():return
    assert read(OUT/'relations/result.json')['all_passed']
    start=time.monotonic();version=snapshot(__file__);model=None;records=[]
    try:
      model,tok=load('qwen4',out);original={n:p.detach().float().cpu().clone() for n,p in model.model.layers[16].mlp.named_parameters()}
      stop=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop]);pad=tok.pad_token_id or tok.eos_token_id
      regression={'original':checks(),'prospective_language':language_checks()};rows=material['controlled'];cap=protocol['generation_cap']
      with torch.inference_mode():
       for variant in VARIANTS:
        deployed_parameters(model,variant,original)
        for begin in range(0,len(rows),8):
            batch=rows[begin:begin+8];paths=[out/variant/'commits'/f"{r['sample_id']}.json" for r in batch]
            if all(p.exists() for p in paths):records.extend(read(p) for p in paths);continue
            tick=time.monotonic();B=len(batch);length=max(len(r['prompt_ids']) for r in batch)
            ids=torch.full((B,length),pad,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
            for b,row in enumerate(batch):ids[b,-len(row['prompt_ids']):]=torch.tensor(row['prompt_ids'],device='cuda');mask[b,-len(row['prompt_ids']):]=1
            pos=(mask.cumsum(-1)-1).clamp_min(0);cache=None;emitted=[[] for r in batch];done=[False]*B;fields=[[] for r in batch];initial=[]
            for step in range(cap):
                o=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True);cache=o.past_key_values
                h=o.last_hidden_state[:,-1];z=model.lm_head(h).float();chosen=z.argmax(-1)
                for b,row in enumerate(batch):
                    if done[b]:continue
                    token=int(chosen[b]);emitted[b].append(token);last=token in stop or step+1==cap
                    if step==0:
                        with np.load(OUT/'relations'/variant/'fields'/f"{row['sample_id']}.npz") as q:single=unbits(q['postnorm_original_prompt'])
                        initial.append({'B8_vs_B1_original_prompt_postnorm_MSE':float(np.mean((h[b].float().cpu().numpy()-single)**2)),
                          'B8_initial_token_id':token,'B1_initial_token_id':read(OUT/'relations'/variant/'commits'/f"{row['sample_id']}.json")['actual_current']['argmax_id']})
                    if step==0 or last:fields[b].append(bits(h[b]))
                    done[b]=last
                del o,h,z
                if all(done):break
                active=torch.tensor([not v for v in done],device='cuda',dtype=torch.long)
                mask=torch.cat([mask,active[:,None]],-1);pos=(mask.sum(-1)-1).clamp_min(0)[:,None];ids=chosen[:,None];ids[active==0]=pad
            elapsed=time.monotonic()-tick
            for b,row in enumerate(batch):
                sid=row['sample_id'];text=tok.decode(emitted[b],skip_special_tokens=True);graded=language_score(row,text,emitted[b],stop,cap)
                native=read(out/'native/commits'/f'{sid}.json') if variant!='native' else None;div=None
                if native:
                    n=native['generated_ids'];div=next((j for j,(a,bid) in enumerate(zip(emitted[b],n)) if a!=bid),None)
                    if div is None and len(emitted[b])!=len(n):div=min(len(emitted[b]),len(n))
                r={k:row[k] for k in ['sample_id','source_group','pair_id','family','language','world','target','kind']}
                r.update(variant=variant,generated_ids=emitted[b],generated_text=text,answer_scoring=graded,
                  first_divergence_from_native_B8=div,initial_shape_control=initial[b],batch_ids=[a['sample_id'] for a in batch],
                  batch_row=b,left_padding=length-len(row['prompt_ids']),batch_seconds=elapsed,seconds=elapsed/B,
                  execution='OriginalBF16 unmodified native or actually trained parameters; B8 same grouping/masks/positions, own history, no readout bias or answer input.',
                  seconds_scope='Equal allocation of a shared observed batch, not independently measured expression time.')
                if paths[b].exists():
                    old=read(paths[b]);assert old['generated_ids']==emitted[b] and old['answer_scoring']==graded;records.append(old)
                else:
                    fp=out/variant/'fields'/f'{sid}.npz';npz(fp,first_and_final_postnorm=np.stack(fields[b]));r['field_sha256']=sha(fp);save(paths[b],r);records.append(r)
            del cache,ids,mask
            if (begin+8)%64==0:guard();print('IDENTITY_BEHAVIOR',variant,begin+8,len(rows),round(time.monotonic()-start,1),flush=True)
        assert time.monotonic()-start<1800
      deployed_parameters(model,'native',original)
      result={'timestamp':stamp(),'source':version,'all_passed':True,'trajectories':len(records),'variants':VARIANTS,'parser_checks':regression,
        'seconds':time.monotonic()-start,'scope':'320expressions times5actual parameter variants. Correct explicit terminal answer plusEOS, capped/unparsed/wrong separately. No reasoning-chain validity inference.'}
      save(finish,result);ledger('identity_own_history_behavior',result['seconds']);print('IDENTITY_BEHAVIOR_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
