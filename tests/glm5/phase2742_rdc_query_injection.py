"""Full-vocabulary transfer and prospective one-shot code-to-EN readout injection."""
import argparse
from collections import defaultdict
from rdc_query_common import *
from phase2741_rdc_query_transfer import OUT,DIRECTIONS,NAMES,freeze

def vocabulary():
    import torch
    path=OUT/'vocabulary_result.json'
    if path.exists():return
    protocol,rows=freeze();assert read(OUT/'fit_result.json')['all_passed'];lookup=defaultdict(dict)
    for r in rows:lookup[r['source_group']][r['representation']]=r
    groups=sorted(g for g,rs in lookup.items() if next(iter(rs.values()))['split'] in ['test','mixed_holdout'])
    probes=read(BASE/'probes/protocol.json')['probes'];batches=defaultdict(list)
    for i,p in enumerate(probes):batches[len(p['token_ids'])].append(i)
    with np.load(BASE/'prototypes/qwen4.npz') as z:prototype=unbits(z['postnorm']).astype(float)
    with np.load(OUT/'mapping.npz') as z:mapping={k:z[k].copy() for k in z.files}
    start=time.monotonic();model=None
    try:
      model,tok=load('qwen4',OUT/'vocabulary');records=[]
      with torch.inference_mode():
        for source,target in DIRECTIONS:
          direction=source+'_to_'+target
          for gi,g in enumerate(groups):
            sr=lookup[g][source];tr=lookup[g][target]
            with np.load(OUT/'fields'/f"{sr['sample_id']}.npz") as z:s=unbits(z['postnorm']).astype(float)
            with np.load(OUT/'fields'/f"{tr['sample_id']}.npz") as z:t=unbits(z['postnorm']).astype(float)
            x=np.stack([np.ones_like(s),s,prototype],-1);pred=[s,prototype]
            for c in ['affine','query_conditioned','shuffled_pair']:
                beta=mapping[direction+'__'+c];pred.append(np.einsum('qdi,di->qd',x[...,:beta.shape[-1]],beta,optimize=True))
            metrics=np.zeros((5,100,2))
            for _,indices in sorted(batches.items()):
              for j0 in range(0,len(indices),16):
                batch=indices[j0:j0+16];tl=model.lm_head(torch.tensor(t[batch],device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
                for c,pp in enumerate(pred):
                    pl=model.lm_head(torch.tensor(pp[batch],device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
                    metrics[c,batch]=torch.stack([(tl.exp()*(tl-pl)).sum(-1),(tl.argmax(-1)==pl.argmax(-1)).double()],-1).cpu().numpy()
            npz(OUT/'vocabulary_fields'/direction/f'{g}.npz',metrics=metrics)
            for qs in ['train_query','validation_query','unseen_query']:
                ix=[i for i,p in enumerate(probes) if p['split']==qs]
                records.append({'direction':direction,'source_group':g,'split':tr['split'],'query_split':qs,'metrics':metrics[:,ix].mean(1).tolist()})
            if (gi+1)%16==0:print('TRANSFER_FULL_VOCAB',direction,gi+1,len(groups),round(time.monotonic()-start,1),flush=True)
        compressed(OUT/'vocabulary_metrics.json.gz',records);summary=[]
        for direction,split,qs in sorted({(r['direction'],r['split'],r['query_split']) for r in records}):
            rr=[r for r in records if (r['direction'],r['split'],r['query_split'])==(direction,split,qs)];gs=[r['source_group'] for r in rr]
            summary.append({'direction':direction,'split':split,'query_split':qs,'KL':{n:clustered([r['metrics'][c][0] for r in rr],gs) for c,n in enumerate(NAMES)},
              'control_minus_mapped':{NAMES[c]:clustered([r['metrics'][c][0]-r['metrics'][3][0] for r in rr],gs) for c in [0,1,2,4]}})
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'paired_groups_per_direction':len(groups),'directions':4,'summary':summary,
          'full_vocabulary':151936,'seconds':time.monotonic()-start,'scope':'True target fullvocabulary distributions used only as evaluation references. Observed source responses are disclosed paired-transfer inputs.'}
        save(path,result);ledger('paired_transfer_full_vocabulary',result['seconds']);print('TRANSFER_VOCAB_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(OUT/'vocabulary',start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

def injection():
    import torch
    from rdc_query_scoring import score,checks
    out=OUT/'injection';finish=out/'result.json'
    if finish.exists():return
    protocol,rows=freeze();assert read(OUT/'fit_result.json')['all_passed'];lookup={(r['source_group'],r['representation']):r for r in rows}
    targets=[r for r in rows if r['split']=='mixed_holdout' and r['representation']=='en'];assert len(targets)==32
    # Qualification is disclosed before behavior is observed. The frozen transfer
    # protocol explicitly includes failed-map stress tests, not only qualified repair.
    reference=read(OUT/'vocabulary_result.json')
    held=next(r for r in reference['summary'] if r['direction']=='python_to_en' and r['split']=='test' and r['query_split']=='unseen_query')
    qualified=all(held['control_minus_mapped'][name]['interval95'][0]>0 for name in ['identity','shuffled_pair'])
    qualification={'timestamp':stamp(),'mapping_sha256':sha(OUT/'mapping.npz'),
      'source_test_and_unseen_query_KL':held,'qualified_against_identity_and_shuffled':qualified,
      'execution_interpretation':'Qualified bounded deployment test' if qualified else 'Explicit failed-map stress test; not validated repair',
      'protocol_scope_note':'The original broad contract proposed validated injection; the prospectively frozen transfer protocol also retained failed-map stress tests. All branches are reported without relabeling this as a successful repair.',
      'behavior_gold_or_new_outcome_used_for_decision':False}
    if (out/'mapping_qualification.json').exists():
        old_qualification=read(out/'mapping_qualification.json');qualification['timestamp']=old_qualification['timestamp']
    immutable(out/'mapping_qualification.json',qualification)
    probes=read(BASE/'probes/protocol.json')['probes'];q=protocol['injection_query_index']
    with np.load(BASE/'prototypes/qwen4.npz') as z:prototype=unbits(z['postnorm'][q]).astype(float)
    with np.load(OUT/'mapping.npz') as z:beta=z['python_to_en__query_conditioned'].copy()
    start=time.monotonic();model=None
    try:
      model,tok=load('qwen4',out);records=[];stop=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop]);tests=checks()
      with torch.inference_mode():
        for branch in protocol['injection_branches']:
          for i,row in enumerate(targets):
            sid=row['sample_id'];cp=out/'commits'/branch/f'{sid}.json'
            if cp.exists():records.append(read(cp));continue
            source=lookup[(row['source_group'],'python')]
            with np.load(OUT/'fields'/f"{source['sample_id']}.npz") as z:src=unbits(z['postnorm'][q]).astype(float)
            x=np.stack([np.ones_like(src),src,prototype],-1);mapped=np.einsum('di,di->d',x,beta)
            replacement=torch.tensor(src if branch=='code_identity' else mapped,device='cuda',dtype=torch.bfloat16)
            tick=time.monotonic();pre=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=True);cache=pre.past_key_values;del pre
            ids=torch.tensor([probes[q]['token_ids']],device='cuda');emitted=[];trace=[];first=None;field=[];cache_audit=None
            native=read(out/'commits/native'/f'{sid}.json') if branch!='native' else None
            for step in range(protocol['max_new_tokens']):
                value=model.model(input_ids=ids,past_key_values=cache,use_cache=True);cache=value.past_key_values;h=value.last_hidden_state[0,-1]
                if step==0:
                    with np.load(OUT/'fields'/f'{sid}.npz') as z:stored=unbits(z['postnorm'][q])
                    numerical={'B1_vs_stored_B16_postnorm_mse':float(np.mean((h.float().cpu().numpy()-stored)**2))}
                    before=cache_id(cache);use=h if branch=='native' else replacement;z=model.lm_head(use).float();after=cache_id(cache)
                    assert before==after;cache_audit={'same_current_history_KV_exact':True,'source_expression_id':source['sample_id'],'source_query_index':q,
                      'available_code_prefix_plus_known_query':True,'used_correct_label':False,'parameter_changed':False,**numerical}
                    field.extend([bits(h),bits(replacement)])
                else:z=model.lm_head(h).float()
                chosen=int(z.argmax());emitted.append(chosen);lp=z.double().log_softmax(-1)
                trace.append({'step':step,'token_id':chosen,'entropy':float(-(lp.exp()*lp).sum()),'injected':step==0 and branch!='native'})
                if native is not None and first is None and (step>=len(native['generated_ids']) or chosen!=native['generated_ids'][step]):first=step
                last=chosen in stop or step+1==protocol['max_new_tokens']
                if last:field.append(bits(h))
                del value,h,z,lp
                if last:break
                ids=torch.tensor([[chosen]],device='cuda')
            text=tok.decode(emitted,skip_special_tokens=True);r={k:row[k] for k in ['sample_id','source_group','target','depth','representation']}
            r.update(branch=branch,generated_ids=emitted,generated_text=text,steps=trace,answer_scoring=score(row,text,emitted,stop,protocol['max_new_tokens']),
              first_divergence=first,cache_audit=cache_audit,seconds=time.monotonic()-tick)
            npz(out/'fields'/branch/f'{sid}.npz',native_replacement_final_postnorm=np.stack(field));save(cp,r);records.append(r);del cache,ids;guard()
            if (i+1)%8==0:print('CODE_RESPONSE_INJECTION',branch,i+1,32,round(time.monotonic()-start,1),flush=True)
        summary=[]
        for branch in protocol['injection_branches']:
            rr=[r for r in records if r['branch']==branch];summary.append({'branch':branch,'expressions':len(rr),
              **{k:sum(bool(r['answer_scoring'][k]) for r in rr) for k in ['parsed_and_stopped_correct','EOS','censored']},
              'parsed':sum(r['answer_scoring']['conservative_final_answer'] is not None for r in rr),'mean_tokens':float(np.mean([len(r['generated_ids']) for r in rr])),
              'correct_cluster':clustered([int(r['answer_scoring']['parsed_and_stopped_correct']) for r in rr],[r['source_group'] for r in rr])})
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'trajectories':len(records),'summary':summary,'parser_checks':tests,'seconds':time.monotonic()-start,
          'mapping_qualified_against_identity_and_shuffled':qualified,'qualification':read(out/'mapping_qualification.json'),
          'scope':'One-shot full-coordinate mapped or identity readout after a declared Answer suffix, with native subsequent own-history generation. Not a persistent reasoning-state transplant, not cross-modal transfer, not elimination of hallucinations.'}
        save(finish,result);ledger('prospective_code_response_injection',result['seconds']);print('CODE_INJECTION_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['vocabulary','injection']);a=p.parse_args()
    if a.stage=='vocabulary':vocabulary()
    else:injection()
