"""Same frozen96x5 long behavior panel, explicit independent B8 own histories."""
import argparse,re
from rdc_query_common import *
from phase2742_rdc_query_late import OUT,BRANCHES,freeze,threshold,measure

def execution():
    protocol,rows=freeze();file=OUT/'batch_protocol.json'
    if file.exists():return read(file),rows
    old=read(PRIOR/'long_answers/result.json');value={'timestamp':stamp(),'source':snapshot(__file__),'original_protocol_sha256':sha(OUT/'protocol.json'),
      'batch_rows':8,'batches':[[r['sample_id'] for r in rows[j:j+8]] for j in range(0,len(rows),8)],
      'precision':'OriginalBF16, eager, leftpadding with explicit attentionmask and per-rowpositionIDs; no quantization. Finished rows receive masked dummy inputs and never generate additional recorded tokens.',
      'unchanged':['All96expressions and24semanticgroups','All5gold-freebranches','1024token cap','Single-shot +8category bias','Frozen entropy threshold and terminal grammar'],
      'reason':'Verified previous1024token B1 study measured'+str(old['observed_seconds_per_token'])+'seconds/token;480x1024worstcase would consume'+str(old['observed_seconds_per_token']*480*1024)+'seconds before other required tests. Amortize native forward overhead across independent rows; do not reduce materials.',
      'numeric_boundary':'Native B8 first2rows compared with separate current B1 pilot. All intervention branches use the same frozen B8 grouping; no B1 expected token is injected.',
      'cache_audit':'At trigger, compare every valid KV byte for that row before/after external logit change; selected next-step own-history KV compared against native B8. Padded slots excluded explicitly, actual valid source order retained.'}
    immutable(file,value);return value,rows

def row_cache(cache,b,valid):
    return [{'block':j,'keys':identity(bits(l.keys[b:b+1,:,valid,:])),'values':identity(bits(l.values[b:b+1,:,valid,:]))} for j,l in enumerate(cache.layers)]

def marker(text):return bool(re.search(r'(?:final\s+answer|answer|答案|最终结果)\s*[:：]\s*[\s*$`{\\]*$',text[-180:],re.I))

def main(branch):
    import torch
    from rdc_query_scoring import score,checks
    p,rows=execution();base=read(OUT/'protocol.json');out=OUT/branch;finish=out/'result.json'
    if finish.exists():return
    assert read(OUT/'pilot_native/pilot.json')['all_passed'];calibration=threshold();start=time.monotonic();model=None;records=[];shape=[]
    try:
      model,tok=load('qwen4',out);digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
      letter=[tok(c,add_special_tokens=False)['input_ids'] for c in 'ABCDEFGH'];assert all(len(x)==1 for x in letter);letters=[x[0] for x in letter]
      stop=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop]);pad=tok.pad_token_id or tok.eos_token_id;tests=checks()
      with torch.inference_mode():
        for bi,groupids in enumerate(p['batches']):
            group=[next(r for r in rows if r['sample_id']==sid) for sid in groupids];commits=[out/'commits'/f'{sid}.json' for sid in groupids]
            if all(cp.exists() for cp in commits):records.extend(read(cp) for cp in commits);continue
            # A failed batch may leave committed rows, which are verified after deterministic replay.
            tick=time.monotonic();B=len(group);length=max(len(r['prompt_ids']) for r in group);mask=torch.zeros((B,length),dtype=torch.long,device='cuda')
            ids=torch.full((B,length),pad,dtype=torch.long,device='cuda')
            for b,r in enumerate(group):ids[b,-len(r['prompt_ids']):]=torch.tensor(r['prompt_ids'],device='cuda');mask[b,-len(r['prompt_ids']):]=1
            pos=(mask.cumsum(-1)-1).clamp_min(0);cache=None;done=[False]*B;emitted=[[] for _ in group];traces=[[] for _ in group];fires=[None]*B
            fields=[[] for _ in group];first=[None]*B;snaps=[{} for _ in group];planned=[{0,128,129} for _ in group];entstep=[None]*B;markstep=[None]*B
            natives=[read(OUT/'native/commits'/f'{r["sample_id"]}.json') for r in group] if branch!='native' else [None]*B
            for step in range(base['max_new_tokens']):
                output=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True);cache=output.past_key_values
                h=output.last_hidden_state[:,-1];logits=model.lm_head(h).float();chosen=torch.full((B,),pad,dtype=torch.long,device='cuda')
                for b,row in enumerate(group):
                    if done[b]:continue
                    z=logits[b];m=measure(z,digits);apply=False
                    if branch=='native':
                        if entstep[b] is None and step>=128 and m['entropy']>=calibration:entstep[b]=step;planned[b].update([step,step+1])
                        if markstep[b] is None and marker(tok.decode(emitted[b],skip_special_tokens=True)):markstep[b]=step;planned[b].update([step,step+1])
                    if fires[b] is None:
                        if branch=='fixed128_digit':apply=step==128
                        elif branch.startswith('entropy_'):apply=step>=128 and m['entropy']>=calibration
                        elif branch=='terminal_marker_digit':apply=marker(tok.decode(emitted[b],skip_special_tokens=True))
                    modified=z.clone()
                    if apply:
                        before=row_cache(cache,b,mask[b].bool());which=letters if branch=='entropy_letter' else digits;modified[which]+=8
                        after=row_cache(cache,b,mask[b].bool());assert before==after;after_measure=measure(modified,digits)
                        if which==digits:assert after_measure['conditional_digit_argmax']==m['conditional_digit_argmax']
                        fires[b]={'step':step,'before':m,'after':after_measure,'all_same_history_KV_bit_equal':True,'cache_identities':before,
                          'category':'letters' if which==letters else 'digits','scope':'All valid KV positions for this batch row, no padded slots.'}
                        fields[b].append(bits(h[b]));planned[b].update([step,step+1])
                    if step in planned[b]:snaps[b][str(step)]=row_cache(cache,b,mask[b].bool())
                    token=int(modified.argmax());chosen[b]=token;emitted[b].append(token)
                    native=natives[b]
                    if native is not None and first[b] is None and (step>=len(native['generated_ids']) or token!=native['generated_ids'][step]):first[b]=step
                    traces[b].append({'step':step,'token_id':token,'bias_applied':apply,**m})
                    last=token in stop or step+1==base['max_new_tokens']
                    if step==0 or last:fields[b].append(bits(h[b]))
                    done[b]=last
                del output,h,logits
                if all(done):break
                active=torch.tensor([not d for d in done],device='cuda',dtype=torch.long)
                mask=torch.cat([mask,active[:,None]],-1);pos=(mask.sum(-1)-1).clamp_min(0)[:,None];ids=chosen[:,None]
                ids[active==0]=pad
            batch_seconds=time.monotonic()-tick
            for b,row in enumerate(group):
                sid=row['sample_id'];text=tok.decode(emitted[b],skip_special_tokens=True);graded=score(row,text,emitted[b],stop,base['max_new_tokens']);comparison=None
                fire=fires[b];native=natives[b]
                if native is not None:
                    if fire is None:
                        assert emitted[b]==native['generated_ids'],('No-intervention branch changed native output',branch,sid)
                    else:
                        assert first[b] is None or first[b]>=fire['step'],('Divergence before first intervention',branch,sid,first[b],fire['step'])
                        native_at_fire=native.get('selected_cache_identities',{}).get(str(fire['step']))
                        if native_at_fire is not None:
                            assert fire['cache_identities']==native_at_fire,('Same pre-intervention history differs from native',branch,sid)
                            fire['matched_native_before_trigger_KV_exact']=True
                if fire and native is not None:
                    nxt=str(fire['step']+1)
                    if nxt in snaps[b] and nxt in native.get('selected_cache_identities',{}):
                        a=snaps[b][nxt];bb=native['selected_cache_identities'][nxt];comparison={'step':int(nxt),'own_history_cache_equal':a==bb,
                          'different_blocks':[j for j,(u,v) in enumerate(zip(a,bb)) if u!=v],
                          'new_token_at_trigger_differs':emitted[b][fire['step']]!=native['generated_ids'][fire['step']]}
                record={k:row[k] for k in ['sample_id','source_group','representation','target','split','depth']}
                record.update(branch=branch,generated_ids=emitted[b],generated_text=text,steps=traces[b],answer_scoring=graded,intervention=fire,
                  first_token_divergence_from_native=first[b],selected_cache_identities=snaps[b],own_history_next_KV_comparison=comparison,
                  original_prompt_ids=row['prompt_ids'],batch_ids=groupids,batch_row=b,left_padding=length-len(row['prompt_ids']),
                  execution='OriginalBF16 B8, explicit masks and per-rowRoPEpositions; own-generated histories only.',batch_seconds=batch_seconds,
                  seconds=batch_seconds/B,seconds_scope='Equal per-row allocation of observed shared batch elapsed time, not separately measured row time.')
                if branch=='native' and (OUT/'pilot_native/commits'/f'{sid}.json').exists():
                    previous=read(OUT/'pilot_native/commits'/f'{sid}.json');a=emitted[b];bb=previous['generated_ids'];div=next((j for j,(u,v) in enumerate(zip(a,bb)) if u!=v),None)
                    if div is None and len(a)!=len(bb):div=min(len(a),len(bb))
                    with np.load(OUT/'pilot_native/fields'/f'{sid}.npz') as z:initial=unbits(z['first_trigger_final_postnorm'][0])
                    ck={'sample_id':sid,'B1_B8_generated_prefix_first_divergence':div,'B1_steps':len(bb),'B8_steps':len(a),
                      'first_postnorm_MSE':float(np.mean((unbits(fields[b][0])-initial)**2)),'numerical_shape_not_assumed_equal':True};shape.append(ck);record['B1_shape_control']=ck
                if commits[b].exists():
                    previous=read(commits[b]);assert previous['generated_ids']==record['generated_ids'] and previous['answer_scoring']==record['answer_scoring'];records.append(previous)
                else:npz(out/'fields'/f'{sid}.npz',first_trigger_final_postnorm=np.stack(fields[b]));save(commits[b],record);records.append(record)
            del cache,ids,mask;guard();assert time.monotonic()-start<7200
            print('LATE_QUERY_B8',branch,bi+1,len(p['batches']),'tokens',sum(map(len,emitted)),'seconds',round(time.monotonic()-start,1),flush=True)
        summary=[]
        for rep in ['all','en','zh','python','en_reordered']:
            rr=[r for r in records if rep=='all' or r['representation']==rep]
            summary.append({'representation':rep,'expressions':len(rr),'groups':len({r['source_group'] for r in rr}),
              'mean_tokens':float(np.mean([len(r['generated_ids']) for r in rr])),'triggered':sum(r['intervention'] is not None for r in rr),
              **{k:sum(bool(r['answer_scoring'][k]) for r in rr) for k in ['parsed_and_stopped_correct','EOS','censored']},
              'parsed':sum(r['answer_scoring']['conservative_final_answer'] is not None for r in rr),
              'correct_cluster':clustered([int(r['answer_scoring']['parsed_and_stopped_correct']) for r in rr],[r['source_group'] for r in rr])})
        shape=[r['B1_shape_control'] for r in records if 'B1_shape_control' in r]
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'branch':branch,'trajectories':len(records),'summary':summary,
          'batch_protocol_sha256':sha(OUT/'batch_protocol.json'),'B1_shape_controls':shape,'parser_checks':tests,'seconds':time.monotonic()-start,
          'all_own_history_divergences_not_before_intervention':True,'untriggered_trajectories_exact_native':True,
          'scope':'All96frozen expressions, paired nativeB8 and policyB8 own histories; batch shape can change native outputs compared withB1. Full vocabulary entropy and whole valid KV byte checks retained.'}
        save(finish,result);ledger('gold_free_late_B8_'+branch,result['seconds']);print('LATE_QUERY_B8_DONE',branch,result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('branch',nargs='?',choices=BRANCHES);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print('LATE_BATCH_PROTOCOL',execution()[0])
    else:main(a.branch)
