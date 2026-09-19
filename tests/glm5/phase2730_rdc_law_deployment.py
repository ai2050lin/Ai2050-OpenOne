"""Frozen native-model rollout, actual trained BF16 deployment and scope controls."""
import argparse,gc
from collections import defaultdict
from rdc_law_common import *
from rdc_law_live import Live,compare_caches
from phase2730_rdc_law_protocol import protocol
from rdc_operator_qa import evaluate,repeated_ngrams


def inputs(row):
    return row['prompt_ids'] if row['kind']=='QA' else row['prompt_ids'][:row['anchors'][-1]+1]


def classifier(row):return (0 if row['language']=='en' else 2)+int(row['kind']=='QA')


def scoring_protocol():
    path=BASE/'deployment/first_gold_scoring_protocol.json'
    if path.exists():return read(path)
    assert not list((BASE/'deployment/rollouts').glob('*/*.json')), 'Declare additional scoring before any live rollout'
    packet={'timestamp':stamp(),'source':snapshot(Path(__file__)),
        'operation':'For48 naturalprefixes only, score the single originalnexttoken atstep0 from actualfullvocabularylogits. Do NOT append it; generation still chooses its ownargmax.',
        'comparison':'Allseven fixedbranches use identicalactualpromptshape. Four trainedBF16states have paired originalnative NLL changes; available originaltokens are scoringlabels, not onlinepredictor inputs.',
        'scope':'One retrospective nexttokenprobability per source is distinct from fullautonomoussemanticquality. QA has multiplepossibleanswers and receives fullanswerEM/F1 instead; no invented unique firstQA goldtoken.'}
    immutable(path,packet);return packet


def replay(model,live,material):
    import torch
    out=BASE/'deployment/replay';meta=gzread(BASE/'prediction/query_catalog.json.gz');at={r['sample_id']:[] for r in meta}
    for i,r in enumerate(meta):at[r['sample_id']].append(i)
    examples=[next(r for r in material.values() if r['cohort']==cohort and r['split']=='test') for cohort in ('gum','ewt','cmrc','squad_qa','cmrc_qa','hotpot_qa')]
    frozen=read(BASE/'prediction/frozen.json');observations=[]
    for row in examples:
        live.set_branch('native');live.reset(classifier(row));ids=torch.tensor([row['prompt_ids']],device=live.device)
        model.model(input_ids=ids,use_cache=False)
        for b in (16,35):
            predicted=live.predict(b,row['anchors']).cpu().numpy()
            decoder=frozen['winners'][str(b)]['decoder']
            with np.load(BASE/'prediction/predictions'/f'L{b}_{decoder}.npz') as z:ref=z['mlp'][at[row['sample_id']]]
            relative=float(np.mean((predicted-ref)**2)/max(np.mean(ref**2),1e-20))
            assert relative<1e-8,('Live/offline feature predictor disagreement',row['sample_id'],b,relative)
            observations.append({'sample_id':row['sample_id'],'block':b,'full_coordinate_relative_MSE':relative,'same_full_prompt_shape':True})
        live.reset(0)
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'checks':observations,'passed':True,
        'scope':'Live earlier-state inputs and full-coordinate forecasts agree with saved offline forecasts on exact full inputs. Reduction ordering may differ; threshold1e-8relativeMSE, not bitwise assertion.'})
    print('LAW_LIVE_REPLAY_PASS',max(r['full_coordinate_relative_MSE'] for r in observations),flush=True)


def scope_audit(model,live,rows):
    import torch
    out=BASE/'deployment/scope';reports=[]
    for i,row in enumerate(rows):
        sid=row['sample_id'];cp=out/'matched_shape_commits'/f'{sid}.json'
        if cp.exists():reports.extend(read(cp)['comparisons']);continue
        ids=torch.tensor([inputs(row)],device=live.device);live.set_branch('native');live.reset(classifier(row))
        ref=model.model(input_ids=ids,use_cache=True);nativecache=ref.past_key_values
        reflog=model.lm_head(ref.last_hidden_state[0,-1]).float();lp=reflog.log_softmax(-1);chosen=int(reflog.argmax())
        localrows=[]
        for branch in ('early_prediction_L16','early_prediction_L35','coherent_seed2728','prefix_order_control_seed2728'):
            mode_outputs=[]
            for local in (True,False):
                live.set_branch(branch,local);live.reset(classifier(row))
                result=model.model(input_ids=ids,use_cache=True);cache=result.past_key_values
                logits=model.lm_head(result.last_hidden_state[0,-1]).float();pp=logits.log_softmax(-1)
                comp=compare_caches(cache,nativecache)
                mode_outputs.append((logits.clone(),cache))
                localrows.append({'sample_id':sid,'source_group':row['source_group'],'branch':branch,'scope':'query_only' if local else 'all_prefill',
                    'native_query_KL':float((lp.exp()*(lp-pp)).sum()),'native_query_argmax_agrees':int(logits.argmax())==chosen,
                    'same_known_history_cache':comp})
                if branch!='early_prediction_L16':assert comp['all_bitwise_equal'],('LastMLP unexpectedly changes same-history KV',sid,branch,local)
            same=torch.equal(mode_outputs[0][0],mode_outputs[1][0]);sc=compare_caches(mode_outputs[0][1],mode_outputs[1][1])
            if branch!='early_prediction_L16':assert same and sc['all_bitwise_equal']
            # Common next token is fixed from native prefill, not each branch's answer.
            decode=[]
            for (logits,cache),local in zip(mode_outputs,(True,False)):
                live.set_branch(branch,local);live.reset(classifier(row))
                # Rebuild matching H12 source memory with identical known prefill; never refresh approximate KV.
                live.enabled=True
                setup=model.model(input_ids=ids,use_cache=False)
                nxt=model.model(input_ids=torch.tensor([[chosen]],device=live.device),past_key_values=cache,use_cache=True)
                decode.append((model.lm_head(nxt.last_hidden_state[0,-1]).float(),nxt.past_key_values));del setup,nxt
            deq=torch.equal(decode[0][0],decode[1][0]);dc=compare_caches(decode[0][1],decode[1][1])
            if branch!='early_prediction_L16':assert deq and dc['all_bitwise_equal']
            localrows.append({'sample_id':sid,'branch':branch,'scope':'query_only_vs_all_prefill','same_query_logits_bitwise':same,'same_prefill_KV_bitwise':sc['all_bitwise_equal'],
                'common_native_next_token_id':chosen,'same_after_known_next_logits_bitwise':deq,'same_after_known_next_KV':dc})
            del mode_outputs,decode,result,cache
        save(cp,{'timestamp':stamp(),'comparisons':localrows,'prompt_ids':ids[0].tolist()});reports.extend(localrows)
        del nativecache,ref,ids;live.reset(0);gc.collect();torch.cuda.empty_cache()
        print('LAW_SCOPE_AUDIT',i+1,len(rows),flush=True)
    summary={'timestamp':stamp(),'source':snapshot(Path(__file__)),'source_rows':len(rows),'comparisons':len(reports),
        'commits':'deployment/scope/matched_shape_commits','predictor_execution_shape':'Identical all-position predictor batches for both action scopes; only write locations differ. Original all-prefill rollouts unchanged.',
        'initial_shape_failure_audit':'deployment/scope_precision/result.json','initial_passed_commit_retained':'deployment/scope/commits',
        'last_MLP_prefill_KV_equal':sum(r['same_known_history_cache']['all_bitwise_equal'] for r in reports if 'same_known_history_cache' in r and r['branch']!='early_prediction_L16'),
        'middle_MLP_prefill_KV_equal':sum(r['same_known_history_cache']['all_bitwise_equal'] for r in reports if 'same_known_history_cache' in r and r['branch']=='early_prediction_L16'),
        'architecture_scope':'LastMLP after every attention: same history means identical actual all-layerKV, independently of changed logits. MiddleMLP can directly affect upperKV. Known-nextID comparison is teacher-controlled scope audit, not free generation.'}
    save(out/'result.json',summary);return summary


def main():
    import torch
    from rdc_operator_model import load
    out=BASE/'deployment';start=time.monotonic();p=protocol();scoring_protocol()
    if (out/'result.json').exists():return
    existing=list((out/'rollouts').glob('*/*.json'))
    # Already committed trajectories consume existing usage, not future reserve.
    # Never expand the original result ceiling merely to resume scope checks.
    remaining=max(0,672-len(existing))
    guard((20+780*remaining/672)*1024**2)
    if existing and not (out/'resume_manifest.json').exists():
        saved=existing+list((out/'rollout_fields').glob('*/*.npz'))
        immutable(out/'resume_manifest.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
            'operation':'Resume retained rollout commits and preserve every output/field byte while correcting only scope-control execution shape.',
            'completed_rollout_commits':len(existing),'files_sha256':{str(p.relative_to(BASE)):sha(p) for p in saved}})
    material={r['sample_id']:r for r in gzread(BASE/'material.json.gz')+gzread(BASE/'confirmation_material.json.gz')}
    rows=[material[s] for s in p['natural_ids']+p['QA_ids']]
    model,tok=load('qwen4',out/'native_load');model.eval();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    live=Live(model);eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);results=[]
    try:
      with torch.inference_mode():
        replay(model,live,material)
        for branch in p['rollout_branches']:
            live.set_branch(branch,False)
            for i,row in enumerate(rows):
                sid=row['sample_id'];cp=out/'rollouts'/branch/f'{sid}.json'
                if cp.exists():results.append(read(cp));continue
                t0=time.monotonic();promptids=inputs(row);ids=torch.tensor([promptids],device=live.device)
                live.reset(classifier(row));cache=None;generated=[];states=[];steps=[];allHmoments=np.zeros((37,2560),np.float64)
                for step in range(48):
                    live.query_fields={};result=model.model(input_ids=ids,past_key_values=cache,use_cache=True);cache=result.past_key_values
                    post=result.last_hidden_state[0,-1];logits=model.lm_head(post).float();lp=logits.log_softmax(-1);chosen=int(logits.argmax())
                    h=np.stack([bits(live.embedding[-1])]+[live.query_fields[b] for b in range(1,37)])
                    if step==0:first=h
                    allHmoments+=unbits(h).astype(float)
                    states.append(np.stack([h[b] for b in (0,12,17,36)]+[bits(post)]))
                    steps.append({'step':step,'known_prefix_length':len(promptids)+step,'chosen_token_id':chosen,
                        'entropy':float(-(lp.exp()*lp).sum()),'chosen_logprob':float(lp[chosen]),'logits_finite':bool(torch.isfinite(logits).all())})
                    if step==0 and row['kind']=='natural':
                        gold=row['prompt_ids'][row['anchors'][-1]+1]
                        steps[-1].update(given_original_next_token_id=gold,given_original_next_token_logprob=float(lp[gold]),
                            given_original_next_token_used_for_generation=False)
                    assert steps[-1]['logits_finite'];generated.append(chosen)
                    if chosen in stop:break
                    ids=torch.tensor([[chosen]],device=live.device)
                text=tok.decode(generated,skip_special_tokens=True)
                packet={k:row[k] for k in ('sample_id','source_group','cohort','language','kind')}
                packet.update(branch=branch,actual_prompt=tok.decode(promptids,skip_special_tokens=False),prompt_ids=promptids,generated_ids=generated,generated_text=text,steps=steps,
                    stopped_by_native_EOS=any(t in stop for t in generated),repeated_4gram_fraction=repeated_ngrams(generated),
                    hit_48_limit=len(generated)==48 and not any(t in stop for t in generated),held_relation_combinations=row.get('held_relation_combinations',[]),seconds=time.monotonic()-t0)
                if row['kind']=='QA':packet.update(question=row['question'],answers=row['answers'],question_type=row['question_type'],**evaluate(text,row['answers'],row['language']))
                npz(out/'rollout_fields'/branch/f'{sid}.npz',prompt_query_all_H=first,query_H0_H12_H17_H36_postnorm=np.stack(states),mean_query_all_H=allHmoments/len(states))
                save(cp,packet);results.append(packet)
                if i<2 or (i+1)%16==0:print('LAW_ROLLOUT',branch,i+1,96,'elapsed',round(time.monotonic()-start,1),flush=True)
                if i==2:
                    done=[r for r in results if r['branch']==branch];forecast=np.mean([r['seconds'] for r in done])*96
                    save(out/'pilots'/f'{branch}.json',{'timestamp':stamp(),'first_three_actual_seconds':[r['seconds'] for r in done],
                        'estimated_96row_branch_seconds':forecast,'warning':'First3natural examples do not cover longQA; perprocess/diskguards continue.'})
                del result,cache,states,ids;live.reset(0);gc.collect();guard()
                assert time.monotonic()-start<read(BASE/'resources.json')['per_process_ceiling_seconds']
        audited=[r for cohort in ('gum','ewt','cmrc','squad_qa','cmrc_qa','hotpot_qa') for r in [x for x in rows if x['cohort']==cohort][:4]]
        scope=scope_audit(model,live,audited)
    finally:
        live.close();del live,model;gc.collect();torch.cuda.empty_cache()
    summaries=[];native={r['sample_id']:r for r in results if r['branch']=='native'}
    for branch in p['rollout_branches']:
        for kind in ('natural','QA'):
            rr=[r for r in results if r['branch']==branch and r['kind']==kind];divergence=[]
            for r in rr:
                a=r['generated_ids'];b=native[r['sample_id']]['generated_ids'];same=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
                divergence.append(same)
            summaries.append({'branch':branch,'kind':kind,'sources':len(rr),'generated_tokens':sum(len(r['generated_ids']) for r in rr),
                'EOS_fraction':float(np.mean([r['stopped_by_native_EOS'] for r in rr])),'repeated4gram_mean':float(np.mean([r['repeated_4gram_fraction'] for r in rr])),
                'mean_matching_prefix_length_vs_native':float(np.mean(divergence)),
                **({'whole_normalized_EM':float(np.mean([r['normalized_full_EM'] for r in rr])),'F1':float(np.mean([r['answer_F1'] for r in rr]))} if kind=='QA' else {})})
    if (out/'resume_manifest.json').exists():
        for name,digest in read(out/'resume_manifest.json')['files_sha256'].items():assert sha(BASE/name)==digest,('Resumed archive changed',name)
    previous_seconds=sum(read(f)['seconds'] for f in (out/'failures').glob('*.json'))
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'branches':len(p['rollout_branches']),'source_rows':len(rows),'trajectories':len(results),
        'summaries':summaries,'scope_audit':scope,'seconds':time.monotonic()-start,
        'previous_interrupted_attempt_seconds':previous_seconds,'total_attempt_seconds':previous_seconds+time.monotonic()-start,
        'original_rollout_bytes_preserved_on_resume':bool((out/'resume_manifest.json').exists()),
        'limits':['PartialMLP prediction, all other native layers remain; not an extracted wholeLLM.','Natural continuation has no unique correct answer; repetition/tokenagreement are diagnostics, not fullsemantic quality.','QA exactmatch can reject valid phrasing, bounded32/48caps differ acrossscale/main.','FourBF16trainedstates are restrictedcontinuation, not reconstructedpretraining.']}
    save(out/'result.json',result);ledger('native_model_live_predictions_training_scope',result['seconds'],trajectories=len(results))
    print('LAW_DEPLOYMENT_COMPLETE',summaries,flush=True)


if __name__=='__main__':
    started=time.monotonic()
    try:main()
    except BaseException as exc:
        import traceback
        failure={'timestamp':stamp(),'source':snapshot(Path(__file__)),'seconds':time.monotonic()-started,
            'exception':repr(exc),'traceback':traceback.format_exc(),'scope':'Interrupted attempt; completed commits retained. No scientific completion is inferred.'}
        save(BASE/'deployment/failures'/f'{time.time_ns()}.json',failure)
        ledger('native_deployment_interrupted_attempt',failure['seconds'])
        raise
