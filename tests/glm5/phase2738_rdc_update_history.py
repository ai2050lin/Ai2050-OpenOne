"""Full native BF16 autonomous deployment, with explicit censoring and no gold history."""
import argparse, gc, re
from collections import defaultdict
from rdc_update_common import *

def freeze():
    out=BASE/'own_history';p=out/'protocol.json'
    if p.exists():return read(p),gzread(out/'material.json.gz')
    programs=gzread(BASE/'program_material.json.gz');language=gzread(BASE/'language_material.json.gz');natural=gzread(BASE/'natural_material.json.gz')
    groups=sorted({r['source_group'] for r in programs if r['split']=='mixed_holdout'})[:8]
    rows=[r for r in programs if r['source_group'] in groups]
    for family in sorted({r['family'] for r in language}):
        gg=sorted({r['source_group'] for r in language if r['family']==family and r['split']=='language_test'})[:2]
        rows.extend(r for r in language if r['source_group'] in gg)
    for cohort in ('gum','ewt'):
        for split in ('new_connected','new_matched'):
            rows.extend([r for r in natural if r['cohort']==cohort and r['split']==split][:4])
    assert len(rows)==88
    directions=[r['direction'] for r in read(BASE/'learning/finite_result.json')['matched_native_deployments']]
    p={'timestamp':stamp(),'sources':[r['sample_id'] for r in rows],
       'branches':['native']+directions+[f'middle_{c}_{s}' for s in (2737,2738) for c in ('content','batch_format_constrained')],
       'max_new_tokens':128,'rows':88,'groups':len({r['source_group'] for r in rows}),
       'selection':'First8 mixed-held semantic groups, first2 held groups per language family (all4 language/style expressions), first4 per natural cohort/condition; selected by material order, no outcome selection.',
       'generation':'Native greedy BF16, own separate KV/history per trajectory, no reference token/cache/state injection; natural prompt ends at first content anchor.',
       'parameter_control':'Last MLP uses calibrated actual BF16 norms ~0.02. Middle training deltas are cast/deployed BF16; their accumulated norms are measured, NOT matched to last MLP.',
       'scoring':'First full/C/F losses only on declared candidate tasks; exact digit or leading Yes/No answer parsing, EOS and cap censoring reported separately. Natural text has no single free-generation gold answer.',
       'retention':'Full-coordinate postnorm at first and last generated steps; all coordinates scanned every step for RMS. Intermediate vectors are not claimed retained.'}
    compressed(out/'material.json.gz',rows);immutable(out/'protocol.json',p);return p,rows

def parse_answer(text,row):
    cleaned=text.strip()
    if row['kind']=='controlled_program':
        return cleaned if re.fullmatch(r'[1-8]',cleaned) else None
    if row['kind']=='controlled_language':
        match=re.match(r'(?i)^(yes|no)(?=\W|$)|^(是|否)',cleaned)
        return (match.group(1) or match.group(2)).lower() if match else None
    return None

def main(pilot=False):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    p,rows=freeze();out=BASE/'own_history';result=out/('pilot.json' if pilot else 'result.json')
    if result.exists():return
    if not pilot:assert read(out/'pilot.json')['passed']
    selected=[rows[0],rows[32],rows[34],rows[-1]] if pilot else rows
    branches=['native'] if pilot else p['branches'];start=time.monotonic();guard(100*1024**2)
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    device=model.get_input_embeddings().weight.device
    weights={b:dict(model.model.layers[b].mlp.named_parameters()) for b in (16,35)}
    original={b:{k:v.detach().clone() for k,v in w.items()} for b,w in weights.items()}
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);records=[];norms=[]
    def reset():
        for b,w in weights.items():
            for k,v in w.items():v.copy_(original[b][k])
    try:
      with torch.inference_mode():
        for branch in branches:
            reset();norm=0.
            if branch!='native':
                middle=branch.startswith('middle_');b=16 if middle else 35
                path=BASE/'middle_training'/branch[len('middle_'):]/'parameter_deltas.npz' if middle else BASE/'learning/native_deltas'/f'{branch}.npz'
                with np.load(path) as z:
                    for k,v in weights[b].items():
                        alias=k if middle else {'gate_proj.weight':'g','up_proj.weight':'u','down_proj.weight':'d'}[k]
                        v.copy_((original[b][k].float()+torch.tensor(z[alias],device=device)).to(torch.bfloat16))
                        norm+=float((v.float()-original[b][k].float()).double().square().sum())
                norms.append({'branch':branch,'block':b,'actual_BF16_delta_norm':norm**.5,'delta_sha256':sha(path)})
            for j,row in enumerate(selected):
                commit=out/'commits'/branch/f'{row["sample_id"]}.json'
                if commit.exists():records.append(read(commit));continue
                tick=time.monotonic();prompt=row['prompt_ids'] if row['kind']!='natural' else row['prompt_ids'][:row['anchors'][0]+1]
                ids=torch.tensor([prompt],device=device);cache=None;generated=[];trace=[];ends=[];first={}
                for step in range(p['max_new_tokens']):
                    value=model.model(input_ids=ids,past_key_values=cache,use_cache=True);cache=value.past_key_values
                    h=value.last_hidden_state[0,-1];logits=model.lm_head(h).float();lp=logits.double().log_softmax(-1);chosen=int(logits.argmax())
                    assert torch.isfinite(logits).all()
                    if step==0:
                        ends.append(bits(h));first={'argmax':chosen,'entropy':float(-(lp.exp()*lp).sum())}
                        target=row['target_ids'][0] if row['kind']!='natural' else row['prompt_ids'][len(prompt)]
                        first.update(full_loss=float(-lp[target]),first_token_correct=chosen==target)
                        if row['kind']!='natural':
                            candidates=row.get('candidate_ids',[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)])
                            clp=logits[candidates].double().log_softmax(-1);first.update(content_loss=float(-clp[candidates.index(target)]),
                              format_loss=float(-torch.logsumexp(lp[candidates],0)),conditional_correct=candidates[int(clp.argmax())]==target)
                    trace.append({'step':step,'token_id':chosen,'logprob':float(lp[chosen]),'postnorm_RMS':float(h.float().square().mean().sqrt())})
                    generated.append(chosen);last=chosen in stop or step+1==p['max_new_tokens']
                    if last:ends.append(bits(h))
                    del value,h,logits,lp
                    if last:break
                    ids=torch.tensor([[chosen]],device=device)
                text=tok.decode(generated,skip_special_tokens=True);parsed=parse_answer(text,row)
                target=str(row.get('target','')).lower();target={'yes':'yes','no':'no'}.get(target,target)
                rec={k:row[k] for k in ('sample_id','source_group','cohort','kind','split')}
                rec.update(branch=branch,prompt_ids=prompt,generated_ids=generated,generated_text=text,first=first,steps=trace,
                  parsed_answer=parsed,parsed_answer_correct=parsed==target if row['kind']!='natural' else None,
                  target=target,stopped_by_EOS=generated[-1] in stop,censored=generated[-1] not in stop,
                  generation_steps=len(generated),seconds=time.monotonic()-tick)
                if branch!='native':
                    native=read(out/'commits/native'/f'{row["sample_id"]}.json');a=generated;b0=native['generated_ids']
                    rec['first_divergence_step']=next((i for i,(x,y) in enumerate(zip(a,b0)) if x!=y),min(len(a),len(b0)) if len(a)!=len(b0) else None)
                npz(out/'fields'/branch/f'{row["sample_id"]}.npz',first_last_postnorm=np.stack(ends));save(commit,rec);records.append(rec)
                del cache,ids;assert time.monotonic()-start<7200;guard(3*1024**2)
                if j<2 or (j+1)%8==0:print('UPDATE_HISTORY',branch,j+1,len(selected),round(time.monotonic()-start,1),flush=True)
        reset();summary=[]
        for branch in branches:
          for cohort in sorted({r['cohort'] for r in records}):
            rr=[r for r in records if r['branch']==branch and r['cohort']==cohort]
            if not rr:continue
            stats={'branch':branch,'cohort':cohort,'trajectories':len(rr),'generated_tokens':sum(r['generation_steps'] for r in rr),
              'EOS_rate':float(np.mean([r['stopped_by_EOS'] for r in rr])),'censored':sum(r['censored'] for r in rr),
              'first_accuracy':float(np.mean([r['first']['first_token_correct'] for r in rr]))}
            if rr[0]['kind']!='natural':stats.update(parsed_accuracy=float(np.mean([r['parsed_answer_correct'] for r in rr])),conditional_accuracy=float(np.mean([r['first']['conditional_correct'] for r in rr])))
            for part in ('full_loss','content_loss','format_loss'):
                if part in rr[0]['first']:stats[part]=float(np.mean([r['first'][part] for r in rr]))
            summary.append(stats)
        seconds=time.monotonic()-start
        result_data={'timestamp':stamp(),'source':snapshot(__file__),'pilot':pilot,'passed':True,'trajectories':len(records),'summary':summary,
          'deployment_norms':norms,'seconds':seconds,'estimated_main_seconds':seconds/4*88*12 if pilot else None,
          'original_parameters_restored_bitwise':all(torch.equal(v,original[b][k]) for b,w in weights.items() for k,v in w.items())}
        if pilot:result_data['passed']=result_data['estimated_main_seconds']<7000
        save(result,result_data);ledger('autonomous_history_pilot' if pilot else 'autonomous_history_remaining',seconds);print('UPDATE_HISTORY_DONE',seconds,flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        del model,weights,original;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');main(p.parse_args().pilot)
