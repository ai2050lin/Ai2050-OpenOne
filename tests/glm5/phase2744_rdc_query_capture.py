"""Native relation fields and full-vocabulary calibration, no new model training."""
import argparse
from phase2744_rdc_query_identifiability import *


def query_fields(model,cache,probes,indices,reference):
    import torch
    groups=defaultdict(list);post=np.zeros((len(indices),2560),np.uint16);statistics=np.zeros((len(indices),4));where={q:i for i,q in enumerate(indices)}
    for q in indices:groups[len(probes[q]['token_ids'])].append(q)
    with torch.inference_mode():
      for _,items in sorted(groups.items()):
       for j in range(0,len(items),16):
        batch=items[j:j+16];own=clone_cache(cache,model.config,len(batch))
        o=model.model(input_ids=torch.tensor([probes[q]['token_ids'] for q in batch],device='cuda'),past_key_values=own,use_cache=True)
        h=o.last_hidden_state[:,-1];lp=model.lm_head(h).float().double().log_softmax(-1);prob=lp.exp();ref=reference[batch]
        ix=[where[q] for q in batch];post[ix]=bits(h)
        statistics[ix]=torch.stack([-(prob*lp).sum(-1),(prob*(lp-ref)).sum(-1),(ref.exp()*(ref-lp)).sum(-1),lp.argmax(-1).double()],-1).cpu().numpy()
        del own,o,h,lp,prob,ref
    return post,statistics


def main(stage):
    import torch
    from phase2741_rdc_query_rules import prototype_native,rule_features
    from phase2742_rdc_query_formation import evaluate
    protocol,material=freeze();out=OUT/stage;finish=out/'result.json'
    if finish.exists():return
    start=time.monotonic();source_version=snapshot(__file__);model=None;handles=[];guard(250*1024**2)
    try:
      model,tok=load('qwen4',out);original={n:p.detach().float().cpu().clone() for n,p in model.model.layers[16].mlp.named_parameters()}
      panel=gzread(BASE/'formation/material.json.gz')['panel'];old_controls=panel[:4];deployment_checks=[]
      if stage=='relations':
        probes=read(BASE/'probes/protocol.json')['probes']
        with np.load(BASE/'prototypes/qwen4.npz') as z:
            pr={k:z[k].copy() for k in z.files if k!='logprobs'};reference=torch.tensor(z['logprobs'],device='cuda')
        with np.load(BASE/'rules/decoder.npz') as z:beta=z['beta'][:,2].copy()
        proto13=np.stack([unbits(pr[f'p{i}_H13'][-1]) for i in range(100)]).astype(float)
        with torch.inference_mode():protos,checks=prototype_native(model,pr,probes)
        data={};observing=[True]
        def take(name,tensor):
            if observing[0]:data[name]=bits(tensor[0,-1])
        handles.append(model.model.layers[35].register_forward_hook(lambda m,a,o:take('raw36',o)))
        for block in [16,35]:
            layer=model.model.layers[block].mlp
            for name in ['gate_proj','up_proj']:
                handles.append(getattr(layer,name).register_forward_hook(lambda m,a,o,key=f'L{block}_{name}':take(key,o)))
            handles.append(layer.down_proj.register_forward_pre_hook(lambda m,a,key=f'L{block}_activation':take(key,a[0])))
        records=[];previous_pair=None;pair_metrics=[]
      else:
        natural=material['natural'];temps=torch.tensor(TEMPERATURES,device='cuda',dtype=torch.float64)
        counts=material['train_token_counts'];N=sum(counts.values());prior={int(k):(v+1)/(N+151936) for k,v in counts.items()};unseen=1/(N+151936)
        records=[]
      for variant in VARIANTS:
        deployed_parameters(model,variant,original)
        # Same B1 uncached implementation and panel as the original formation study.
        if stage=='relations':observing[0]=False
        actual=evaluate(model,old_controls)
        file=BASE/'formation/native_baseline.npz' if variant=='native' else BASE/'formation'/variant/'deployed_BF16.npz'
        with np.load(file) as z:expected=z['loss'][:4]
        error=float(np.max(abs(actual['loss']-expected)));assert error<1e-8,(variant,error)
        deployment_checks.append({'variant':variant,'first4_original_formation_losses_exact':bool(np.array_equal(actual['loss'],expected)),'max_NLL_error':error})
        if stage=='relations':
          for i,row in enumerate(material['controlled']):
            cp=out/variant/'commits'/f"{row['sample_id']}.json";fp=out/variant/'fields'/f"{row['sample_id']}.npz"
            # Native predictions for pair metrics are streamed, so replay the whole
            # native block on partial restart, then verify already committed arrays.
            if cp.exists() and variant!='native':records.append(read(cp));continue
            with torch.inference_mode():
              observing[0]=True;data.clear();pre=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=True,output_hidden_states=True)
              cache=pre.past_key_values;states=list(pre.hidden_states[:-1])+[None];raw36=data['raw36'].copy()
              layers=np.stack([bits(h[0,-1]) for h in states[:-1]]+[raw36])
              arrays={'postnorm_original_prompt':bits(pre.last_hidden_state[0,-1]),**{k:v.copy() for k,v in data.items() if k!='raw36'}}
              arrays['prefix_layers' if variant=='native' else 'prefix_selected_layers_H16_H17_H36']=layers if variant=='native' else layers[[16,17,36]]
              z=model.lm_head(pre.last_hidden_state[0,-1]).float();lp=z.double().log_softmax(-1);choice=row['candidate_ids'];binlp=z[choice].double().log_softmax(-1)
              current={'full_vocab_NLL':float(-lp[row['target_ids'][0]]),'binary_conditional_NLL':float(-binlp[0 if row['truth'] else 1]),
                'binary_conditional_yes_probability':float(binlp[0].exp()),'candidate_total_probability':float(lp[choice].exp().sum()),
                'argmax_id':int(lp.argmax()),'first_argmax_correct':int(lp.argmax())==row['target_ids'][0],'entropy':float(-(lp.exp()*lp).sum())}
              original_cache=cache_id(cache) if i%32==0 else None;observing[0]=False
              subset,statistics=query_fields(model,cache,probes,QUERIES,reference)
              arrays.update(matched_subset_postnorm=subset,matched_subset_full_vocabulary_statistics=statistics,matched_query_indices=np.array(QUERIES))
              pred_metrics=None
              if variant=='native':
                full,stats=query_fields(model,cache,probes,list(range(100)),reference);arrays.update(postnorm=full,full_vocabulary_statistics=stats)
                arrays['matched_subset_vs_full_B16_coordinate_difference']=unbits(subset)-unbits(full[QUERIES])
                cand,merge=rule_features(model,protos,pr,cache.layers[12].keys[0].repeat_interleave(4,0),cache.layers[12].values[0].repeat_interleave(4,0),verifymerge=i<2)
                pred=[]
                for c in range(5):
                    x=np.stack([np.ones_like(proto13),np.broadcast_to(unbits(layers[12]),proto13.shape),proto13,unbits(cand[c])],-1)
                    pred.append(np.einsum('qdi,di->qd',x,beta[c],optimize=True))
                prediction=np.stack(pred);target=unbits(full).astype(float);mse=((prediction-target[None])**2).mean(-1)
                arrays['frozen_rule_all_query_MSE']=mse;pred_metrics=mse.mean(1).tolist();kl=np.zeros((5,100));groups=defaultdict(list)
                for q,p in enumerate(probes):groups[len(p['token_ids'])].append(q)
                for _,qq in sorted(groups.items()):
                  for j in range(0,len(qq),16):
                    ix=qq[j:j+16];lp_target=model.lm_head(torch.tensor(target[ix],device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
                    if i<2:assert np.max(abs((-(lp_target.exp()*lp_target).sum(-1)).cpu().numpy()-stats[ix,0]))<1e-7
                    for c in range(5):
                        lp_prediction=model.lm_head(torch.tensor(prediction[c,ix],device='cuda',dtype=torch.bfloat16)).float().double().log_softmax(-1)
                        kl[c,ix]=(lp_target.exp()*(lp_target-lp_prediction)).sum(-1).cpu().numpy()
                arrays['frozen_rule_all_query_KL']=kl
                del lp_target,lp_prediction
                if previous_pair is not None:
                    prev_row,prev_pred,prev_target=previous_pair;assert row['pair_id']==prev_row['pair_id'] and row['world']==1
                    diff=target-prev_target;forecast=prediction-prev_pred;err=(forecast-diff[None])**2
                    pair_metrics.append({'pair_id':row['pair_id'],'source_group':row['source_group'],'family':row['family'],'language':row['language'],
                      'actual_query_relation_displacement_MSE':float(np.mean(diff**2)),
                      'frozen_rule_relation_displacement_MSE':err.mean((1,2)).tolist(),
                      'query_split_displacement_MSE':{split:err[:,[q for q,p in enumerate(probes) if p['split']==split]].mean((1,2)).tolist() for split in ['train_query','validation_query','unseen_query']}})
                    # Full-coordinate signed changes are already recomputable from
                    # the two permanently retained complete response arrays.
                    previous_pair=None
                else:previous_pair=(row,prediction.copy(),target.copy())
                del cand,prediction,pred,target,mse
              if original_cache is not None:assert original_cache==cache_id(cache)
              record={'timestamp':stamp(),'sample_id':row['sample_id'],'source_group':row['source_group'],'pair_id':row['pair_id'],'world':row['world'],
                'family':row['family'],'language':row['language'],'variant':variant,'actual_current':current,'frozen_rule_MSE':pred_metrics,
                'statistics_columns':['entropy','KL_to_original_untrained_standalone','reverse_KL','argmax'],
                'whole_prefix_cache_bitchecked':original_cache is not None,'prefix_cache_not_modified':True,
                'query_scope':'Native full100 plus separate matched6 execution; trainedvariants only matched6. Same trained-vs-native query batch shapes.',
                'layer_indices':list(range(37)) if variant=='native' else [16,17,36]}
              if cp.exists():
                old=read(cp);assert old['actual_current']==current
                with np.load(fp) as old_arrays:
                    assert set(old_arrays.files)==set(arrays) and all(np.array_equal(old_arrays[k],v) for k,v in arrays.items())
                records.append(old)
              else:npz(fp,**arrays);record['sha256']=sha(fp);save(cp,record);records.append(record)
              del pre,cache,states,arrays,layers,z,lp,binlp
            if (i+1)%32==0:guard();print('IDENTITY_FIELDS',variant,i+1,320,round(time.monotonic()-start,1),flush=True)
          if variant=='native':assert previous_pair is None;compressed(out/'frozen_relation_change_metrics.json.gz',pair_metrics)
        else:
          fp=out/variant/'all_fields.npz';cp=out/variant/'result.json'
          if cp.exists():records.append(read(cp));continue
          fields=[];values=[];grid=[]
          with torch.inference_mode():
            for i,row in enumerate(natural):
                o=model.model(input_ids=torch.tensor([row['ids']],device='cuda'),use_cache=False);h=o.last_hidden_state[0,-1]
                logits=model.lm_head(h).float();ll=logits.double()[None]/temps[:,None];lp=ll.log_softmax(-1);prob=lp.exp();target=row['target']
                ptarget=prob[:,target];qtarget=prior.get(target,unseen)
                ng=torch.stack([-lp[:,target] if alpha==0 else -torch.logaddexp(lp[:,target]+np.log1p(-alpha),torch.full_like(ptarget,np.log(alpha*qtarget))) for alpha in MIXTURES],-1)
                fields.append(bits(h));grid.append(ng.cpu().numpy());values.append({'sample_id':row['sample_id'],'source_group':row['source_group'],
                  'split':row['split'],'cohort':row['cohort'],'native_NLL':float(-lp[2,target]),'entropy':float(-(prob[2]*lp[2]).sum()),
                  'argmax_id':int(logits.argmax()),'argmax_correct':int(logits.argmax())==target,
                  'temperature_entropies':(-(prob*lp).sum(-1)).cpu().tolist()})
                del o,h,logits,ll,lp,prob,ptarget,ng
                if (i+1)%96==0:print('CALIBRATION_FIELDS',variant,i+1,len(natural),round(time.monotonic()-start,1),flush=True)
          npz(fp,postnorm=np.stack(fields),temperature_prior_mixture_NLL=np.stack(grid));compressed(out/variant/'observations.json.gz',values)
          result={'timestamp':stamp(),'variant':variant,'examples':len(values),'field_sha256':sha(fp),'all_passed':True};save(cp,result);records.append(result)
        guard();assert time.monotonic()-start<1900,('Whole stage reserve breached before behavior',stage)
      deployed_parameters(model,'native',original)
      result={'timestamp':stamp(),'source':source_version,'all_passed':True,'variants':VARIANTS,'deployment_checks':deployment_checks,
        'records':len(records),'seconds':time.monotonic()-start,'prototype_checks':checks if stage=='relations' else [],
        'scope':'Every sampled native coordinate retained; no gold enters predictor features or free generation. Actual training deltas are deployed, not refitted on these materials.'}
      save(finish,result);ledger('identity_'+stage,result['seconds']);print('IDENTITY_CAPTURE_DONE',stage,result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        for h in handles:h.remove()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['relations','calibration']);a=p.parse_args();main(a.stage)
