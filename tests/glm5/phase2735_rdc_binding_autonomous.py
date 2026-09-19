"""Deploy oracle/proxy and train-only gradient directions with each branch's own KV."""
import gc
from rdc_binding_common import *
from rdc_binding_gradients import dense_gradient,normalized
from phase2734_rdc_binding_live import rollout,apply_delta,reset_block
from phase2735_rdc_binding_long_capture import score

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    out=BASE/'format_content/autonomous'
    if (out/'result.json').exists():return
    protocol=read(BASE/'format_content/protocol.json')
    programs=gzread(BASE/'program_material.json.gz');future=gzread(BASE/'format_content/prospective_material.json.gz')
    beta_ids=read(BASE/'beta_updates/protocol.json')['sample_ids'];candidates=[r for r in programs if r['sample_id'] in beta_ids]
    selected=[]
    for family in ('alias','conditional','mapping','addition'):
      for split in ('test','depth_test'):
       for rep in ('en','zh','python','en_reordered'):
        selected.extend([r for r in candidates if r['family']==family and r['split']==split and r['representation']==rep][:1])
    new=[]
    for family in ('alias','conditional','mapping','addition'):
      for rep in ('en','zh','python','en_reordered'):
        new.extend([r for r in future if r['family']==family and r['representation']==rep][:1])
    rows=selected+new
    branches=['native','full_EN_projected_code','content_EN_projected_code','mean_EN_format','random_global',
      'oracle_gold_next_digit_CE','prefix_repeat_probability_mass','random_beta']
    immutable(out/'protocol.json',{'timestamp':stamp(),'old_ids':[r['sample_id'] for r in selected],'future_ids':[r['sample_id'] for r in new],
      'branches':branches,'cap':protocol['autonomous_max_new_tokens'],'global_parameter_norm':.02,'beta_parameter_norm':.10,
      'beta_scope':'Only32old heldout examples. Gold oracle consumes that example target; proxy and random do not.',
      'norm_precision':'Declared norms target the FP32 update before native BF16 rounding. Actual per-matrix norms and changed-scalar fractions are recorded; equality of realized BF16 norms is not assumed.',
      'prospective_scope':'48total rows for native/train-only global directions;16new depth6 rows never used to fit directions.',
      'reset':'Original checkpoint weights restored before each independent case-specific update and after every global branch.'})
    start=time.monotonic();model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2);records=[];deployments={}
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);cap=protocol['autonomous_max_new_tokens']
    digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
    paths={'full_EN_projected_code':BASE/'gradient_span/directions/en_projected_to_code_span.npz',
      'content_EN_projected_code':BASE/'format_content/directions/content_EN_projected_to_code_content_span.npz',
      'mean_EN_format':BASE/'format_content/directions/mean_EN_format.npz'}
    original={k:getattr(model.model.layers[35].mlp,name+'_proj').weight.detach().clone() for k,name in [('g','gate'),('u','up'),('d','down')]}
    def set_unit(unit,norm):
        report={}
        with torch.no_grad():
          for k,name in [('g','gate'),('u','up'),('d','down')]:
            p=getattr(model.model.layers[35].mlp,name+'_proj').weight
            updated=(original[k].float()-norm*unit[k]).to(p.dtype)
            report[k]={'changed_scalar_fraction':float((updated!=original[k]).float().mean()),'actual_delta_norm':float((updated.float()-original[k].float()).norm())}
            p.copy_(updated)
        return report
    def reset():
        with torch.no_grad():
          for k,name in [('g','gate'),('u','up'),('d','down')]:getattr(model.model.layers[35].mlp,name+'_proj').weight.copy_(original[k])
    try:
      for branch in branches:
        reset();use=selected if branch in ('oracle_gold_next_digit_CE','prefix_repeat_probability_mass','random_beta') else rows
        if branch in paths:deployments[branch]=apply_delta(model,35,paths[branch],.02)
        if branch=='random_global':
            gen=torch.Generator(device='cuda').manual_seed(2735)
            dense={k:torch.randint(0,2,v.shape,device='cuda',generator=gen,dtype=torch.int8).float().mul_(2).sub_(1) for k,v in original.items()}
            unit,norm=normalized(dense);deployments[branch]=set_unit(unit,.02);del dense,unit
        for row in use:
            sid=row['sample_id'];commit=out/'commits'/branch/f'{sid}.json'
            if commit.exists():records.append(read(commit));continue
            update=None
            if branch in ('oracle_gold_next_digit_CE','prefix_repeat_probability_mass','random_beta'):
                reset()
                if branch=='random_beta':
                    i=beta_ids.index(sid);gen=torch.Generator(device='cuda').manual_seed(273300+i)
                    dense={k:torch.randint(0,2,v.shape,device='cuda',generator=gen,dtype=torch.int8).float().mul_(2).sub_(1) for k,v in original.items()}
                else:
                    prefix='oracle' if branch.startswith('oracle') else 'proxy'
                    with np.load(BASE/'beta_updates/factors'/f'{sid}.npz') as z:
                        factor={k:torch.tensor(z[prefix+'_'+k],device='cuda') for k in ('x','a','s','bg','bu')}
                    dense=dense_gradient(factor);del factor
                unit,norm=normalized(dense);del dense
                update={'original_gradient_norm':norm,'skipped_negligible_gradient':norm<1e-8}
                if norm>=1e-8:update['deployment']=set_unit(unit,.10)
                del unit
            result=rollout(model,tok,row['prompt_ids'],cap,evaluation_target=row['target_ids'][0],evaluation_candidates=digits)
            result.update(score(result['generated'],row['target'],result['generated_ids'],stop,cap))
            result.update({k:row[k] for k in ('sample_id','source_group','family','representation','split','depth','target')})
            result.update(branch=branch,cap=cap,prompt_ids=row['prompt_ids'],case_update=update)
            if branch=='native':
                earlier=read(BASE/'format_content/native_commits'/f'{sid}.json')
                result['long_native_prefix_exact']=result['generated_ids']==earlier['generated_ids'][:cap]
                assert result['long_native_prefix_exact'],sid
            save(commit,result);records.append(result)
            print('AUTONOMOUS_PARAMETER_UPDATE',branch,sid,len(result['generated_ids']),flush=True)
            assert time.monotonic()-start<7200
        guard(25*1024**2)
      reset()
    finally:
        del model,original;gc.collect();torch.cuda.empty_cache()
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'trajectories':len(records),'old_rows':32,'new_rows':16,
      'global_deployments':deployments,'seconds':time.monotonic()-start,
      'deployment_precision_scope':'Nominal FP32 update norms .02/.10 can change after BF16 rounding, differently by direction. Use actual per-matrix deployment norms; these branches are not asserted exactly norm-matched in BF16.',
      'scope':'All branches generated on their own tokens/KV; gold-oracle is supervised per-case adaptation, not zero-shot. Every model checkpoint remains unchanged on disk.'})
    ledger('format_content_autonomous',time.monotonic()-start)

if __name__=='__main__':main()
