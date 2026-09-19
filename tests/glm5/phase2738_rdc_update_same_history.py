"""Native retrospective companion on the update branch's independently chosen history."""
import gc
from rdc_update_common import *

def freeze():
    out=BASE/'same_history'
    if (out/'protocol.json').exists():return read(out/'protocol.json'),gzread(out/'material.json.gz')
    pool=gzread(BASE/'own_history/material.json.gz');program=[r for r in pool if r['kind']=='controlled_program'];language=[r for r in pool if r['kind']=='controlled_language'];natural=[r for r in pool if r['kind']=='natural']
    group=min(r['source_group'] for r in program);rows=[r for r in program if r['source_group']==group]
    for family in sorted({r['family'] for r in language}):
        group=min(r['source_group'] for r in language if r['family']==family);rows.extend(r for r in language if r['source_group']==group and r['answer_style']=='direct')
    rows.extend(next(r for r in natural if r['cohort']==cohort and r['split']==split) for cohort in ('gum','ewt') for split in ('new_connected','new_matched'))
    assert len(rows)==18
    p={'timestamp':stamp(),'source':snapshot(__file__),'samples':[r['sample_id'] for r in rows],
      'branches':['content_format_constrained_1e-6','mean_EN_format','gaussian_native_control','middle_content_2737','middle_batch_format_constrained_2737','middle_batch_format_constrained_2738'],
      'max_new_tokens':128,'KV_comparison_steps':'First/every8steps/last, all36layers and everyKVelement',
      'operation':'Separate caches; original native companion receives branch-chosen past tokens, never supplies its choices/cache/states to branch. Switch only in-memory declared weights; no disk checkpoint writes.',
      'selection':'First mixed-held semantic group, first held group per language family EN/ZH direct, first per natural cohort/condition;18 existing sources, not independent new material.',
      'architecture_prediction':'LastMLP update cannot change anyKV on SAME history in this architecture. MiddleMLP can affect blocks17..35 KV; behavioral divergence alone is not proof of direct memory change.'}
    compressed(out/'material.json.gz',rows);immutable(out/'protocol.json',p);return p,rows

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from rdc_law_live import compare_caches
    from rdc_update_deployment import Deployer
    from rdc_update_scoring import score
    p,rows=freeze();out=BASE/'same_history';start=time.monotonic()
    if (out/'result.json').exists():return
    assert (BASE/'own_history/result.json').exists();guard(140*1024**2)
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2);deploy=Deployer(model)
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);records=[];norms=[]
    try:
      with torch.inference_mode():
        for branch in p['branches']:
            norms.append(deploy.select(branch))
            for j,row in enumerate(rows):
                cp=out/'commits'/branch/f'{row["sample_id"]}.json'
                if cp.exists():records.append(read(cp));continue
                prompt=row['prompt_ids'] if row['kind']!='natural' else row['prompt_ids'][:row['anchors'][0]+1]
                ids=torch.tensor([prompt],device='cuda');acache=None;ncache=None;generated=[];trace=[];kv=[];fields=[];tick=time.monotonic()
                for step in range(p['max_new_tokens']):
                    deploy.activate(True);a=model.model(input_ids=ids,past_key_values=acache,use_cache=True);acache=a.past_key_values
                    ah=a.last_hidden_state[0,-1];az=model.lm_head(ah).float();alp=az.double().log_softmax(-1);chosen=int(az.argmax())
                    deploy.activate(False);n=model.model(input_ids=ids,past_key_values=ncache,use_cache=True);ncache=n.past_key_values
                    nh=n.last_hidden_state[0,-1];nz=model.lm_head(nh).float();nlp=nz.double().log_softmax(-1);prob=nlp.exp()
                    d=az.double()-nz.double();center=d-(prob*d).sum()
                    trace.append({'step':step,'chosen_token_id':chosen,'native_same_history_argmax':int(nz.argmax()),'argmax_agrees':chosen==int(nz.argmax()),
                      'native_to_branch_KL':float((prob*(nlp-alp)).sum()),'half_logit_variance':float((prob*center.square()).sum()/2),
                      'postnorm_relative_MSE':float((ah.float()-nh.float()).square().sum()/nh.float().square().sum().clamp_min(1e-20))})
                    fields.append(np.stack([bits(ah),bits(nh)]));generated.append(chosen);last=chosen in stop or step+1==p['max_new_tokens']
                    if step%8==0 or last:
                        c=compare_caches(acache,ncache);kv.append({'step':step,'prefix_length':len(prompt)+step,**c})
                        if not branch.startswith('middle_'):assert c['all_bitwise_equal'],(branch,row['sample_id'],step)
                    del a,n,ah,nh,az,nz,alp,nlp,prob,d,center
                    if last:break
                    ids=torch.tensor([[chosen]],device='cuda')
                previous=read(BASE/'own_history/commits'/branch/f'{row["sample_id"]}.json');assert generated==previous['generated_ids'],(branch,row['sample_id'],'companion replay changed')
                text=tok.decode(generated,skip_special_tokens=True);rec={k:row[k] for k in ('sample_id','source_group','kind','cohort','split')}
                rec.update(branch=branch,prompt_ids=prompt,generated_ids=generated,generated_text=text,steps=trace,KV_comparisons=kv,
                  same_main_rollout_IDs_exact=True,answer_scoring=score(row,text,generated,stop,p['max_new_tokens']),seconds=time.monotonic()-tick)
                npz(out/'fields'/branch/f'{row["sample_id"]}.npz',branch_native_postnorm=np.stack(fields));save(cp,rec);records.append(rec)
                del acache,ncache,ids;guard(3*1024**2);assert time.monotonic()-start<7200
                if (j+1)%6==0:print('UPDATE_SAME_HISTORY',branch,j+1,len(rows),round(time.monotonic()-start,1),flush=True)
        deploy.reset();assert deploy.restored();summary=[]
        for branch in p['branches']:
            rr=[r for r in records if r['branch']==branch];cc=[c for r in rr for c in r['KV_comparisons']]
            summary.append({'branch':branch,'rows':len(rr),'mean_source_KL':clustered([np.mean([s['native_to_branch_KL'] for s in r['steps']]) for r in rr],[r['source_group'] for r in rr]),
              'argmax_agreement':float(np.mean([np.mean([s['argmax_agrees'] for s in r['steps']]) for r in rr])),
              'all_KV_equal_checks':sum(c['all_bitwise_equal'] for c in cc),'total_KV_checks':len(cc),
              'changed_KV_blocks':sorted({x['block'] for c in cc for x in c['layers'] if not x['bitwise_equal']})})
        result={'timestamp':stamp(),'source':snapshot(__file__),'trajectories':len(records),'deployment_norms':norms,'summaries':summary,
          'all_main_IDs_replayed_exact':True,'original_parameters_restored':True,'seconds':time.monotonic()-start,
          'scope':'Retrospective native reference on actual branch history; repeated diagnostic panel, no new independent confirmation. No reference quantities repair branch.'}
        save(out/'result.json',result);ledger('updated_native_same_history_reference',result['seconds']);print('UPDATE_SAME_HISTORY_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:del model,deploy;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
