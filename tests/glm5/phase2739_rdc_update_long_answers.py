"""Censoring-driven bounded long-answer continuation; short records stay unchanged."""
import argparse,gc
from rdc_update_common import *

def freeze():
    out=BASE/'long_answers'
    if (out/'protocol.json').exists():return read(out/'protocol.json'),gzread(out/'material.json.gz')
    rows=[r for r in gzread(BASE/'own_history/material.json.gz') if r['kind']=='controlled_program'];assert len(rows)==32
    p={'timestamp':stamp(),'source':snapshot(__file__),'sample_ids':[r['sample_id'] for r in rows],'semantic_groups':8,'expressions':32,
      'branches':['native','content_format_constrained_1e-6','middle_content_2737','middle_batch_format_constrained_2737','middle_content_2738','middle_batch_format_constrained_2738'],
      'max_new_tokens':1024,'first_pilot_ids':[r['sample_id'] for r in rows[:2]],
      'reason':'Every available32-case mixed-program128-token branch was censored before a terminal answer. Preserving those records, extend the SAME first8held semantic groups, no success-selected examples.',
      'scoring':'Strict digit-only, explicit terminal answer parsing, EOS and censoring separate. Correct parsed-and-stopped fraction uses all32expressions as denominator; report8-group uncertainty.',
      'precision':'Same original native BF16 model and stored rounded updates as128-token run; the1024token cap is the only generation change.',
      'pilot_gate':'First2native cases include observed seconds/token and projected full192trajectory cost; per-process7200s, global21600s ledger. No extension beyond1024.'}
    compressed(out/'material.json.gz',rows);immutable(out/'protocol.json',p);return p,rows

def main(pilot=False):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from rdc_update_deployment import Deployer
    from rdc_update_scoring import score,checks
    p,rows=freeze();out=BASE/'long_answers';result_path=out/('pilot.json' if pilot else 'result.json');start=time.monotonic()
    if result_path.exists():return
    if not pilot:assert read(out/'pilot.json')['passed']
    selected=rows[:2] if pilot else rows;branches=['native'] if pilot else p['branches']
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2);deploy=Deployer(model);records=[];norms=[]
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);parser_checks=checks()
    try:
      with torch.inference_mode():
        for branch in branches:
            norms.append(deploy.select(branch))
            for j,row in enumerate(selected):
                cp=out/'commits'/branch/f'{row["sample_id"]}.json'
                if cp.exists():records.append(read(cp));continue
                tick=time.monotonic();ids=torch.tensor([row['prompt_ids']],device='cuda');cache=None;generated=[];trace=[];fields=[]
                for step in range(p['max_new_tokens']):
                    output=model.model(input_ids=ids,past_key_values=cache,use_cache=True);cache=output.past_key_values
                    h=output.last_hidden_state[0,-1];z=model.lm_head(h).float();chosen=int(z.argmax());generated.append(chosen)
                    trace.append({'step':step,'token_id':chosen,'logprob':float(z.double().log_softmax(-1)[chosen])})
                    last=chosen in stop or step+1==p['max_new_tokens']
                    if step==0 or last:fields.append(bits(h))
                    del output,h,z
                    if last:break
                    ids=torch.tensor([[chosen]],device='cuda')
                text=tok.decode(generated,skip_special_tokens=True);graded=score(row,text,generated,stop,p['max_new_tokens'])
                short=read(BASE/'own_history/commits'/branch/f'{row["sample_id"]}.json');same=generated[:len(short['generated_ids'])]==short['generated_ids']
                rec={k:row[k] for k in ('sample_id','source_group','kind','cohort','split','representation','target','depth')}
                rec.update(branch=branch,prompt_ids=row['prompt_ids'],generated_ids=generated,generated_text=text,steps=trace,
                  answer_scoring=graded,short128_prefix_exact=same,generation_steps=len(generated),seconds=time.monotonic()-tick)
                npz(out/'fields'/branch/f'{row["sample_id"]}.npz',first_last_postnorm=np.stack(fields));save(cp,rec);records.append(rec)
                assert same,('Generation cap changed an already selected prefix',branch,row['sample_id'])
                del cache,ids;guard(3*1024**2);assert time.monotonic()-start<7200
                if j<2 or (j+1)%4==0:print('LONG_UPDATE_ANSWER',branch,j+1,len(selected),'tokens',len(generated),'EOS',graded['EOS'],round(time.monotonic()-start,1),flush=True)
        deploy.reset();assert deploy.restored();summary=[]
        for branch in branches:
          for rep in ('en','zh','python','en_reordered','all'):
            rr=[r for r in records if r['branch']==branch and (rep=='all' or r['representation']==rep)]
            if not rr:continue
            summary.append({'branch':branch,'representation':rep,'expressions':len(rr),'semantic_groups':len({r['source_group'] for r in rr}),
              'mean_generated_tokens':float(np.mean([r['generation_steps'] for r in rr])),
              **{metric:float(np.mean([bool(r['answer_scoring'][metric]) for r in rr])) for metric in ('strict_answer_only','parsed_and_stopped_correct','EOS','censored')},
              'parse_coverage':float(np.mean([r['answer_scoring']['conservative_final_answer'] is not None for r in rr])),
              'stopped_correct_cluster':clustered([int(r['answer_scoring']['parsed_and_stopped_correct']) for r in rr],[r['source_group'] for r in rr])})
        seconds=time.monotonic()-start;rate=sum(r['seconds'] for r in records)/sum(r['generation_steps'] for r in records)
        worst=rate*32*6*1024+300
        result={'timestamp':stamp(),'source':snapshot(__file__),'pilot':pilot,'passed':worst<7200 if pilot else True,
          'observed_seconds_per_token':rate,'projected_max192x1024_seconds':worst,'trajectories':len(records),'summary':summary,'deployment_norms':norms,
          'parser_checks':parser_checks,'all_short128_prefixes_exact':True,'original_parameters_restored':True,'seconds':seconds,
          'scope':'Same8held semantic groups, longer bounded diagnostics; not independent confirmation or unlimited language capability. No answer inference from censored unfinished reasoning.'}
        save(result_path,result);ledger('long_answer_pilot' if pilot else 'long_answer_remaining',seconds);print('LONG_ANSWER_DONE',seconds,'pilot',pilot,'passed',result['passed'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:del model,deploy;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');main(p.parse_args().pilot)
