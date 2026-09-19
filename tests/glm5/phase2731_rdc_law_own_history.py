"""Native references on each approximation's actual self-selected history.

No cache, token, hidden state or probability from the reference is fed back to
the autonomous branch.  This separates same-history output error from a direct
cache change without interpreting divergent histories as memory corruption.
"""
import gc
from rdc_law_common import *
from rdc_law_live import Live, compare_caches
from phase2730_rdc_law_deployment import inputs, classifier
from phase2730_rdc_law_protocol import protocol
from rdc_operator_qa import evaluate, repeated_ngrams


def freeze():
    out=BASE/'own_history';path=out/'protocol.json'
    if path.exists():return read(path)
    p=protocol();material={r['sample_id']:r for r in gzread(BASE/'material.json.gz')+gzread(BASE/'confirmation_material.json.gz')}
    pool=[material[s] for s in p['natural_ids']+p['QA_ids']]
    rows=[r for c in ('gum','ewt','cmrc','squad_qa','cmrc_qa','hotpot_qa') for r in [x for x in pool if x['cohort']==c][:4]]
    spec={'timestamp':stamp(),'source':snapshot(Path(__file__)),'sources':[r['sample_id'] for r in rows],
        'branches':['early_prediction_L16','early_prediction_L35','coherent_seed2728','prefix_order_control_seed2728'],
        'selection':'First four in frozen deployment order per cohort, 24 sources; same scope-audit panel, not outcome-selected.',
        'operation':'Two separate caches. Approximate branch greedily selects every new token. Native companion receives exactly those already selected tokens, never native chosen tokens. No reference quantities are fed back.',
        'comparison':'All-vocabulary KL and half logit variance every step. All36layers/allKVelements compared every8steps, first and last. Native-reference cache is incrementally exact for that same prefix; it is never substituted into the branch.',
        'architecture_prediction':'LastMLP-only replacement or parameter updates leave everyKV exactly equal on same history; middleMLP may change upperKV. Logit errors can remain with identicalKV.',
        'formation':'Two fixed seed2728 coherent/order-control actual64step BF16 parameter states, no outcome selection; both seed2729 states already tested in main deployment.',
        'independence':'Diagnostic followup on existing deployment materials, not new independent confirmation. Generated IDs compared to main deployment for exact replay; failures recorded before raising.',
        'max_new_tokens':48,'reference_prediction_use':'None. Companion is retrospective evidence, not a language predictor or a correction method.',
        'limits':['No claim that identical KV implies identical language behavior.','Logit variance is local second-order geometry, not a global KL identity.','Natural text generation is not scored as having a unique gold continuation.']}
    immutable(path,spec);return spec


def main():
    import torch
    from rdc_operator_model import load
    out=BASE/'own_history';p=freeze()
    if (out/'result.json').exists():return
    assert (BASE/'deployment/result.json').exists(), 'Main autonomous deployment must finish first'
    start=time.monotonic();guard(240*1024**2)
    material={r['sample_id']:r for r in gzread(BASE/'material.json.gz')+gzread(BASE/'confirmation_material.json.gz')}
    model,tok=load('qwen4',out/'native_load');model.eval();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    live=Live(model);eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);records=[]
    try:
      with torch.inference_mode():
        for branch in p['branches']:
          for j,sid in enumerate(p['sources']):
            cp=out/'commits'/branch/f'{sid}.json'
            if cp.exists():records.append(read(cp));continue
            t0=time.monotonic();row=material[sid];promptids=inputs(row);ids=torch.tensor([promptids],device=live.device)
            live.enabled=True;live.set_branch(branch,False);live.reset(classifier(row));branch_cache=None;native_cache=None
            generated=[];steps=[];fields=[];kvchecks=[]
            for step in range(p['max_new_tokens']):
                live.query_fields={}
                approx=model.model(input_ids=ids,past_key_values=branch_cache,use_cache=True);branch_cache=approx.past_key_values
                ah=approx.last_hidden_state[0,-1];alog=model.lm_head(ah).float();alp=alog.log_softmax(-1);chosen=int(alog.argmax())
                # Hooks are disabled ONLY for the companion so its pass cannot
                # append to or otherwise change the branch's H12 source memory.
                source_count=len(live.source);live.enabled=False;live.set_branch('native')
                reference=model.model(input_ids=ids,past_key_values=native_cache,use_cache=True);native_cache=reference.past_key_values
                nh=reference.last_hidden_state[0,-1];nlog=model.lm_head(nh).float();nlp=nlog.log_softmax(-1);np_=nlp.exp()
                assert len(live.source)==source_count
                live.set_branch(branch,False);live.enabled=True
                d=alog-nlog;center=d-(np_*d).sum();kl=float((np_*(nlp-alp)).sum());variance=float((np_*center.square()).sum())
                last=chosen in stop or step==p['max_new_tokens']-1
                if step%8==0 or last:
                    comp=compare_caches(branch_cache,native_cache)
                    kvchecks.append({'step':step,'known_prefix_length':len(promptids)+step,**comp})
                    if branch!='early_prediction_L16':assert comp['all_bitwise_equal'], ('Same-history lastMLP cache mismatch',sid,branch,step)
                steps.append({'step':step,'known_prefix_length':len(promptids)+step,'chosen_token_id':chosen,
                    'native_same_history_argmax':int(nlog.argmax()),'argmax_agrees':int(nlog.argmax())==chosen,
                    'native_to_branch_KL':kl,'half_native_logit_variance':variance/2,
                    'branch_chosen_logprob':float(alp[chosen]),'native_chosen_logprob':float(nlp[chosen]),
                    'postnorm_relative_MSE':float((ah.float()-nh.float()).square().mean()/nh.float().square().mean().clamp_min(1e-20))})
                fields.append(np.stack([bits(ah),bits(nh)]));generated.append(chosen)
                assert torch.isfinite(alog).all() and torch.isfinite(nlog).all()
                del approx,reference,ah,nh,alog,nlog,alp,nlp,np_,d,center
                if last:break
                ids=torch.tensor([[chosen]],device=live.device)
            previous=read(BASE/'deployment/rollouts'/branch/f'{sid}.json')
            packet={k:row[k] for k in ('sample_id','source_group','cohort','language','kind')}
            packet.update(branch=branch,actual_prompt=tok.decode(promptids,skip_special_tokens=False),prompt_ids=promptids,generated_ids=generated,generated_text=tok.decode(generated,skip_special_tokens=True),
                main_rollout_IDs_exact=generated==previous['generated_ids'],steps=steps,KV_comparisons=kvchecks,
                stopped_by_native_EOS=generated[-1] in stop,repeated_4gram_fraction=repeated_ngrams(generated),
                held_relation_combinations=row.get('held_relation_combinations',[]),seconds=time.monotonic()-t0)
            if row['kind']=='QA':packet.update(question=row['question'],answers=row['answers'],question_type=row['question_type'],
                **evaluate(packet['generated_text'],row['answers'],row['language']))
            npz(out/'fields'/branch/f'{sid}.npz',branch_native_postnorm=np.stack(fields))
            save(cp,packet);records.append(packet)
            assert packet['main_rollout_IDs_exact'],('Native companion altered or failed to replay branch trajectory',sid,branch)
            del branch_cache,native_cache,ids,fields;live.reset(0);gc.collect();torch.cuda.empty_cache();guard()
            print('LAW_OWN_HISTORY',branch,j+1,len(p['sources']),'elapsed',round(time.monotonic()-start,1),flush=True)
            assert time.monotonic()-start<read(BASE/'resources.json')['per_process_ceiling_seconds']
    finally:
        live.enabled=True;live.close();del live,model;gc.collect();torch.cuda.empty_cache()
    summaries=[]
    for branch in p['branches']:
      for kind in ('all','natural','QA'):
        rr=[r for r in records if r['branch']==branch and (kind=='all' or r['kind']==kind)]
        source_groups=[r['source_group'] for r in rr];kk=[k for r in rr for k in r['KV_comparisons']]
        summaries.append({'branch':branch,'kind':kind,'trajectories':len(rr),'steps':sum(len(r['steps']) for r in rr),
            'source_mean_KL':clustered([np.mean([s['native_to_branch_KL'] for s in r['steps']]) for r in rr],source_groups),
            'source_argmax_agreement':clustered([np.mean([s['argmax_agrees'] for s in r['steps']]) for r in rr],source_groups),
            'source_half_variance':clustered([np.mean([s['half_native_logit_variance'] for s in r['steps']]) for r in rr],source_groups),
            'all_KV_equal_checks':sum(k['all_bitwise_equal'] for k in kk),'total_KV_checks':len(kk),
            'changed_KV_blocks':sorted(set(x['block'] for k in kk for x in k['layers'] if not x['bitwise_equal']))})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'trajectories':len(records),'summaries':summaries,
        'all_main_rollouts_exact':all(r['main_rollout_IDs_exact'] for r in records),'seconds':time.monotonic()-start,
        'scope':'Native retrospective references on autonomous self-selected histories; no oracle inputs, resets or native tokens enter approximate branch.'}
    save(out/'result.json',result);ledger('own_selected_history_native_reference',result['seconds'],trajectories=len(records))
    print('LAW_OWN_HISTORY_COMPLETE',summaries,flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');a=ap.parse_args()
    if a.freeze_only:freeze()
    else:
        started=time.monotonic()
        try:main()
        except BaseException as exc:
            import traceback
            failure={'timestamp':stamp(),'source':snapshot(Path(__file__)),'seconds':time.monotonic()-started,
                'exception':repr(exc),'traceback':traceback.format_exc(),'scope':'Interrupted attempt, retained commits, not a completed scientific result.'}
            save(BASE/'own_history/failures'/f'{time.time_ns()}.json',failure)
            ledger('own_history_interrupted_attempt',failure['seconds'])
            raise
