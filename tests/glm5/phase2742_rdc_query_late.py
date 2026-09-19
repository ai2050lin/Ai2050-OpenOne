"""Gold-free late readout tests with each branch's own continuing history."""
import argparse,re
from collections import defaultdict
from rdc_query_common import *

OUT=BASE/'late'
BRANCHES=['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']

def freeze():
    p=OUT/'protocol.json'
    if p.exists():return read(p),gzread(OUT/'material.json.gz')
    old_groups={r['source_group'] for r in gzread(PRIOR/'long_answers/material.json.gz')}
    rows=[r for r in gzread(PRIOR/'program_material.json.gz') if r['split']=='mixed_holdout' and r['source_group'] not in old_groups]
    assert len(rows)==96 and len({r['source_group'] for r in rows})==24
    value={'timestamp':stamp(),'source':snapshot(__file__),'samples':[r['sample_id'] for r in rows],'new_behavior_expressions':96,'semantic_groups':24,
      'previous_prefix_exposure':'Same frozen program task corpus; these24groups were outside the prior32case1024token own-history study. Other earlier prefix/128token scale diagnostics may have exposed some groups.',
      'branches':BRANCHES,'max_new_tokens':1024,'bias_strength':8.,'earliest_entropy_step':128,
      'entropy_threshold':'75thpercentile native fullvocabulary entropy for steps>=128 from all32old native trajectories; no new-test result or gold used.',
      'policy':'Single logit-bias application per trajectory; identical+8for everydigit1..8, or matched8lettertokens in negative control. No target digit singled out; no forced EOS; all later tokens native under branch-own prefix.',
      'fixed':'Atstep128, if not already EOS; entropy policy atfirststep>=128 above threshold; terminal-marker policy atfirst own-prefix terminalmarker followed only by formatting/space.',
      'entropy_measures':['Fullvocabulary entropy','Binary digit-vs-other category entropy','Conditional8digit entropy'],
      'interpretation':'Entropy alone is not nonsense, and digit bias alone cannot alter current conditional ranking among digits. Same-current-history cache bytes do not change from external logit addition; emitted-token divergence can change subsequent KV.',
      'scoring_source':snapshot(Path(__file__).with_name('rdc_query_scoring.py')),
      'pilot':'First2native trajectories measure actual token cost; each branch is a separate process, all96fixedtestexpressions retained. Global6h budget still applies.'}
    compressed(OUT/'material.json.gz',rows);immutable(p,value);return value,rows

def threshold():
    file=OUT/'entropy_calibration.json'
    if file.exists():return read(file)['threshold']
    assert read(BASE/'events/result.json')['all_passed'];records=[read(p) for p in sorted((BASE/'events/commits').glob('*.json'))]
    a=[s['entropy'] for r in records for s in r['steps'] if s['step']>=128];value=float(np.quantile(a,.75))
    immutable(file,{'timestamp':stamp(),'threshold':value,'quantile':.75,'native_trajectories':len(records),'eligible_steps':len(a),
      'source_result_sha256':sha(BASE/'events/result.json'),'used_test_outcome':False,'used_gold':False});return value

def measure(z,digits):
    import torch
    lp=z.double().log_softmax(-1);prob=lp.exp();mass=prob[digits].sum();q=mass.clamp(1e-15,1-1e-15);conditional=z[digits].double().log_softmax(-1)
    return {'entropy':float(-(prob*lp).sum()),'digit_mass':float(mass),'binary_format_entropy':float(-q*q.log()-(1-q)*(1-q).log()),
      'conditional_digit_entropy':float(-(conditional.exp()*conditional).sum()),'conditional_digit_argmax':int(z[digits].argmax()),'native_argmax':int(z.argmax())}

def main(branch,pilot=False):
    import torch
    from rdc_query_scoring import score,checks
    protocol,rows=freeze();out=OUT/('pilot_'+branch if pilot else branch);finish=out/('pilot.json' if pilot else 'result.json')
    if finish.exists():return
    calibration=threshold();selected=rows[:2] if pilot else rows;start=time.monotonic();model=None
    try:
      model,tok=load('qwen4',out);digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
      letters=[tok(c,add_special_tokens=False)['input_ids'] for c in 'ABCDEFGH'];assert all(len(x)==1 for x in letters);letters=[x[0] for x in letters]
      assert len(set(digits))==8 and not set(digits)&set(letters)
      stop=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop]);records=[];tests=checks()
      with torch.inference_mode():
        for i,row in enumerate(selected):
            sid=row['sample_id'];file=out/'commits'/f'{sid}.json'
            if file.exists():records.append(read(file));continue
            tick=time.monotonic();cache=None;ids=torch.tensor([row['prompt_ids']],device='cuda');emitted=[];trace=[];fired=False;fire=None;field=[];first_divergence=None
            cache_snapshots={};planned={0,128,129};native_entropy_step=None;native_marker_step=None
            native=read(OUT/'native/commits'/f'{sid}.json') if branch!='native' else None
            for step in range(protocol['max_new_tokens']):
                output=model.model(input_ids=ids,past_key_values=cache,use_cache=True);cache=output.past_key_values;h=output.last_hidden_state[0,-1]
                z=model.lm_head(h).float();m=measure(z,digits);apply=False
                if branch=='native':
                    if native_entropy_step is None and step>=128 and m['entropy']>=calibration:
                        native_entropy_step=step;planned.update([step,step+1])
                    if native_marker_step is None:
                        text=tok.decode(emitted,skip_special_tokens=True)
                        if re.search(r'(?:final\s+answer|answer|答案|最终结果)\s*[:：]\s*[\s*$`{\\]*$',text[-180:],re.I):
                            native_marker_step=step;planned.update([step,step+1])
                if not fired:
                    if branch=='fixed128_digit':apply=step==128
                    elif branch.startswith('entropy_'):apply=step>=128 and m['entropy']>=calibration
                    elif branch=='terminal_marker_digit':
                        text=tok.decode(emitted,skip_special_tokens=True)
                        apply=bool(re.search(r'(?:final\s+answer|answer|答案|最终结果)\s*[:：]\s*[\s*$`{\\]*$',text[-180:],re.I))
                modified=z.clone()
                if apply:
                    before=cache_id(cache);which=letters if branch=='entropy_letter' else digits;modified[which]+=8
                    after=cache_id(cache);assert before==after
                    after_measure=measure(modified,digits)
                    if which==digits:assert after_measure['conditional_digit_argmax']==m['conditional_digit_argmax']
                    fire={'step':step,'before':m,'after':after_measure,'all_same_history_KV_bit_equal':True,'cache_identities':before,'category':'letters' if which==letters else 'digits'}
                    field.append(bits(h));fired=True;planned.update([step,step+1])
                if step in planned:cache_snapshots[str(step)]=cache_id(cache)
                chosen=int(modified.argmax());emitted.append(chosen)
                if native is not None and first_divergence is None and (step>=len(native['generated_ids']) or chosen!=native['generated_ids'][step]):first_divergence=step
                trace.append({'step':step,'token_id':chosen,'bias_applied':apply,**m})
                last=chosen in stop or step+1==protocol['max_new_tokens']
                if step==0 or last:field.append(bits(h))
                del output,h,z,modified
                if last:break
                ids=torch.tensor([[chosen]],device='cuda')
            text=tok.decode(emitted,skip_special_tokens=True);graded=score(row,text,emitted,stop,protocol['max_new_tokens'])
            record={k:row[k] for k in ['sample_id','source_group','representation','target','split','depth']}
            comparison=None
            if fire and native is not None:
                next_step=str(fire['step']+1)
                if next_step in cache_snapshots and next_step in native.get('selected_cache_identities',{}):
                    a=cache_snapshots[next_step];b=native['selected_cache_identities'][next_step]
                    comparison={'step':int(next_step),'own_history_cache_equal':a==b,
                      'different_blocks':[j for j,(u,v) in enumerate(zip(a,b)) if u!=v],
                      'new_token_at_trigger_differs':emitted[fire['step']]!=native['generated_ids'][fire['step']]}
            record.update(branch=branch,generated_ids=emitted,generated_text=text,steps=trace,answer_scoring=graded,intervention=fire,
              selected_cache_identities=cache_snapshots,own_history_next_KV_comparison=comparison,
              first_token_divergence_from_native=first_divergence,seconds=time.monotonic()-tick,original_prompt_ids=row['prompt_ids'])
            npz(out/'fields'/f'{sid}.npz',first_trigger_final_postnorm=np.stack(field));save(file,record);records.append(record)
            del cache,ids;assert time.monotonic()-start<7200;guard()
            if i<2 or (i+1)%8==0:print('LATE_QUERY_BRANCH',branch,i+1,len(selected),'tokens',len(emitted),'trigger',fire['step'] if fire else None,round(time.monotonic()-start,1),flush=True)
        rate=sum(r['seconds'] for r in records)/sum(len(r['generated_ids']) for r in records)
        summary=[]
        for rep in ['all','en','zh','python','en_reordered']:
            rr=[r for r in records if rep=='all' or r['representation']==rep]
            if not rr:continue
            summary.append({'representation':rep,'expressions':len(rr),'groups':len({r['source_group'] for r in rr}),
              'mean_tokens':float(np.mean([len(r['generated_ids']) for r in rr])),'triggered':sum(r['intervention'] is not None for r in rr),
              **{k:sum(bool(r['answer_scoring'][k]) for r in rr) for k in ['parsed_and_stopped_correct','EOS','censored']},
              'parsed':sum(r['answer_scoring']['conservative_final_answer'] is not None for r in rr),
              'correct_cluster':clustered([int(r['answer_scoring']['parsed_and_stopped_correct']) for r in rr],[r['source_group'] for r in rr])})
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'pilot':pilot,'branch':branch,'trajectories':len(records),'summary':summary,
          'observed_seconds_per_token':rate,'projected_max_all480x1024seconds':480*1024*rate,'parser_checks':tests,'seconds':time.monotonic()-start,
          'scope':'One-shot category bias without correct-answer input. Output entropy is only an observable; this experiment does not label high entropy as nonsense. Native and intervention branches are paired, not independent samples.'}
        save(finish,result);ledger('gold_free_late_'+branch+('_pilot' if pilot else ''),result['seconds']);print('LATE_QUERY_DONE',branch,result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('branch',nargs='?',choices=BRANCHES);p.add_argument('--freeze',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.freeze:print('LATE_PROTOCOL',freeze()[0])
    else:main(a.branch,a.pilot)
