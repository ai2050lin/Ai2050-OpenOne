"""Original-model unit/group interventions with frozen material and controls."""
import argparse
import gc
import time

import numpy as np

import recursive_branch14 as R
import recursive_branch14_analysis as A

ARMS=['baseline','restore','single_0','single_2','group_0','group_2',
      'single_norm_control','group_norm_control','matched_group_0','matched_group_2']
CHUNK=32


def perturb(module,args,output,rows,units,dose,direction=None):
    """Recompute native down projection after scaling actual MLP units."""
    import torch
    a=args[0];mod=a.clone();pos=torch.arange(R.LENGTH,device='cuda')[None,:]
    region=(pos>=torch.tensor([r['content_start'] for r in rows],device='cuda')[:,None]) & (pos<=torch.tensor([r['content_end'] for r in rows],device='cuda')[:,None])
    selected=mod[:,:,units]
    mod[:,:,units]=torch.where(region[:,:,None],selected*dose,selected)
    changed=torch.nn.functional.linear(mod,module.weight)
    delta=changed.float()-output.float();target=delta.norm(dim=-1)
    if direction is None:return changed,dict(target_norm=target,region=region)
    scale=target.clone();best=None;best_error=torch.full_like(target,float('inf'))
    # Match the realized BF16 perturbation, keeping all mismatch information.
    for iteration in range(5):
        trial=(output.float()+scale[:,:,None]*direction).to(output.dtype)
        actual=(trial.float()-output.float()).norm(dim=-1)
        err=(actual-target).abs()/target.clamp_min(1e-12)
        better=err<best_error
        best=trial if best is None else torch.where(better[:,:,None],trial,best)
        best_error=torch.minimum(best_error,err)
        scale=scale*target/actual.clamp_min(1e-12)
    actual=(best.float()-output.float()).norm(dim=-1)
    return best,dict(target_norm=target,actual_norm=actual,relative_error=best_error,region=region)


def run(hypothesis):
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    p=R.admission();selection=R.read(R.OUT/'selection.json');spec=selection['groups'][hypothesis]
    block=spec['block'];unit=spec['original_unit'];allrows=R.materials();cases=[];cache={};refs={}
    for split in R.SPLITS[1:]:
        meta=R.read(R.OUT/split/'collection.json');cache[split]=np.load(R.OUT/split/f'{hypothesis}-replay-input.npy',mmap_mode='r')
        assert meta['behavior_passed'],'Behavior gate failed: do not conduct affected mechanism interventions'
        refs[split]=np.load(R.OUT/split/'final-hidden.npy',mmap_mode='r')
        for cached,local in enumerate(meta['entrance_local_indices'][hypothesis]):
            global_i=meta['row_indices'][local]
            cases.append(dict(split=split,cached=cached,local=local,global_i=global_i,row=allrows[global_i]))
    assert len(cases)==1440
    destination=R.OUT/f'intervention-{hypothesis}.json'
    assert not destination.exists(),'Completed intervention is immutable'
    partial=R.OUT/f'intervention-{hypothesis}.jsonl'
    assert not partial.exists(),'Partial results exist; preserve before retry with new identity'
    start=time.monotonic();results=[];checks=[]
    R.log('intervention_start_'+hypothesis,0,len(cases),start)
    with R.GpuLease(),torch.inference_mode():
      cfg,pos,pe,causal=R.context()
      rng=np.random.default_rng(R.SEED+block*100)
      directions={name:torch.from_numpy(rng.normal(size=R.DIM).astype(np.float32)).cuda() for name in ['single_norm_control','group_norm_control']}
      for name in directions:directions[name]/=directions[name].norm()
      for lo in range(0,len(cases),CHUNK):
        chunk=cases[lo:lo+CHUNK];rows=[c['row'] for c in chunk];size=len(chunk)
        raw=np.stack([cache[c['split']][c['cached']] for c in chunk])
        original=torch.from_numpy(raw).view(torch.bfloat16).cuda()
        states=original[None].repeat(len(ARMS),1,1,1);del original,raw
        mask=R.mask_for(rows,causal);local_controls={}
        for b in range(block,R.DEPTH):
          layer=R.C.load_block(b,cfg)
          for arm_index,arm in enumerate(ARMS):
            for i in range(0,size,R.BATCH):
              rr=rows[i:i+R.BATCH];handle=None;record={}
              if b==block and arm!='baseline':
                units=([unit] if arm.startswith('single') else spec['matched_units'] if arm.startswith('matched') else spec['units'])
                dose=1. if arm=='restore' else 2. if arm.endswith('_2') else 0.
                direction=directions.get(arm)
                def hook(m,a,y,units=units,dose=dose,direction=direction,rr=rr):
                    value,stats=perturb(m,a,y,rr,units,dose,direction);record.update(stats);return value
                handle=layer.mlp.down_proj.register_forward_hook(hook)
              states[arm_index,i:i+len(rr)]=layer(states[arm_index,i:i+len(rr)],attention_mask=mask[i:i+len(rr)],position_ids=pos,position_embeddings=pe,use_cache=False)
              if handle:handle.remove()
              if b==block and 'relative_error' in record:
                for j,r in enumerate(rr):
                    region=record['region'][j];e=record['relative_error'][j][region]
                    target=record['target_norm'][j][region];actual=record['actual_norm'][j][region]
                    pooled=float((actual.norm()-target.norm()).abs()/target.norm().clamp_min(1e-12))
                    local_controls[arm,i+j]=dict(all_token_norm_matched=bool((e<=.02).all()),
                        matched_token_fraction=float((e<=.02).float().mean()),max_token_norm_error=float(e.max()),
                        pooled_norm_error=pooled,pooled_norm_matched=pooled<=.02)
              record.clear()
          del layer;gc.collect();torch.cuda.empty_cache()
          if (b-block)%5==0 or b==39:
            R.log('intervene_'+hypothesis,lo+(b-block+1)/(R.DEPTH-block)*size,len(cases),start,block=b,chunk=lo)
        norm=Qwen3RMSNorm(R.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
        picked=torch.stack([R.pick(states[a],rows)[:,1] for a in range(len(ARMS))])
        final=norm(picked);refraw=np.stack([refs[c['split']][c['local'],1] for c in chunk])
        reference=torch.from_numpy(refraw).view(torch.bfloat16).cuda()
        restored=torch.equal(final[0],final[1]);assert restored,'Native restore is not bitwise'
        relative=float((final[0].float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12))
        wu=R.C.weight('lm_head.weight').cuda();batch_rows=[];replay_kl=[]
        for i in range(0,size,R.BATCH):
          count=min(R.BATCH,size-i)
          base=torch.nn.functional.linear(final[0,i:i+count],wu).float();blp=base.log_softmax(-1);bp=blp.exp()
          ref=torch.nn.functional.linear(reference[i:i+count],wu).float();rlp=ref.log_softmax(-1)
          replay_kl.extend((bp*(blp-rlp)).sum(-1).cpu().tolist())
          for arm_index,arm in enumerate(ARMS):
            logits=torch.nn.functional.linear(final[arm_index,i:i+count],wu).float();lp=logits.log_softmax(-1)
            kl=(bp*(blp-lp)).sum(-1)
            for j,c in enumerate(chunk[i:i+count]):
                r=c['row'];s=logits[j,r['choice_ids']].cpu().tolist();bs=base[j,r['choice_ids']].cpu().tolist()
                margin=s[r['gold']]-max(v for k,v in enumerate(s) if k!=r['gold']);bm=bs[r['gold']]-max(v for k,v in enumerate(bs) if k!=r['gold'])
                row=dict(row_index=c['global_i'],id=r['id'],family=r['family'],hypothesis=r['hypothesis'],split=r['split'],domain=r['domain'],role=r['role'],lexical=r['lexical'],
                    intervention_hypothesis=hypothesis,block=block,arm=arm,gold=r['gold'],scores=s,baseline_scores=bs,
                    correct=int(np.argmax(s))==r['gold'],baseline_correct=int(np.argmax(bs))==r['gold'],margin=margin,baseline_margin=bm,delta_margin=margin-bm,
                    KL=float(kl[j]),vocabulary_top1=int(logits[j].argmax()),baseline_top1=int(base[j].argmax()),
                    norm_control=local_controls.get((arm,i+j)))
                batch_rows.append(row)
        maxkl=max(replay_kl);check=dict(chunk=lo,n=size,restore_bitwise=restored,baseline_final_relative_L2=relative,replay_KL_max=maxkl,
            baseline_bitwise=bool(torch.equal(final[0],reference)))
        checks.append(check)
        assert relative<=.02 and maxkl<=.001,'Frozen baseline replay gate failed'
        results.extend(batch_rows)
        with partial.open('a',encoding='utf8') as f:
            for r in batch_rows:f.write(R.json.dumps(r,ensure_ascii=False)+'\n')
        R.save(f'intervention-{hypothesis}-progress.json',dict(completed=lo+size,total=len(cases),checks=checks))
        del states,mask,norm,picked,final,reference,refraw,wu,base,blp,bp,ref,rlp,logits,lp,kl;gc.collect();torch.cuda.empty_cache()
        if lo==0:
            estimate=(time.monotonic()-start)*len(cases)/size
            assert estimate<=p['resource']['intervention_minutes_max']*60,'Intervention exceeds frozen time budget'
    R.save(destination.name,dict(kind='real',rows=results,checks=checks,cases=len(cases),arms=ARMS,
        source_sha256=R.file_sha(__file__),plan_sha256=R.file_sha(R.OUT/'plan.json'),selection_sha256=R.file_sha(R.OUT/'selection.json'),seconds=time.monotonic()-start))
    R.log('intervention_complete_'+hypothesis,len(cases),len(cases),start)


def summarize():
    result=[]
    for h in R.CANDIDATES:
        data=R.read(R.OUT/f'intervention-{h}.json');rows=data['rows']
        for arm in ARMS:
          if arm in ['baseline','restore']:continue
          for scope in ['target','other_task']:
            chosen=[r for r in rows if r['arm']==arm and (r['hypothesis']==h)==(scope=='target')]
            for role in ['all','operative','quoted','inventory']:
                use=[r for r in chosen if role=='all' or r['role']==role]
                family={f:[r for r in use if r['family']==f] for f in {r['family'] for r in use}}
                delta=np.array([np.mean([r['delta_margin'] for r in rr]) for rr in family.values()])
                primary=scope=='target' and role=='all' and arm in ['single_0','group_0']
                summary=dict(hypothesis=h,arm=arm,scope=scope,role=role,rows=len(use),
                    margin_change=A.inference(delta,alpha=.05/4 if primary else .05),primary_test=primary,
                    accuracy=float(np.mean([r['correct'] for r in use])),baseline_accuracy=float(np.mean([r['baseline_correct'] for r in use])),
                    mean_KL=float(np.mean([r['KL'] for r in use])),top1_change=sum(r['vocabulary_top1']!=r['baseline_top1'] for r in use),
                    originally_correct_now_wrong=sum(r['baseline_correct'] and not r['correct'] for r in use),
                    originally_wrong_now_correct=sum(not r['baseline_correct'] and r['correct'] for r in use))
                if arm.endswith('norm_control'):
                    summary['strict_norm_matched_rows']=sum(r['norm_control']['all_token_norm_matched'] for r in use)
                    summary['pooled_norm_matched_rows']=sum(r['norm_control']['pooled_norm_matched'] for r in use)
                    native_arm='single_0' if arm.startswith('single') else 'group_0'
                    native={r['id']:r for r in rows if r['arm']==native_arm}
                    comparison=[dict(r,native_minus_control=native[r['id']]['delta_margin']-r['delta_margin']) for r in use if r['norm_control']['all_token_norm_matched']]
                    fam={f:[r for r in comparison if r['family']==f] for f in {r['family'] for r in comparison}}
                    summary['strict_matched_native_minus_control']=A.inference([np.mean([r['native_minus_control'] for r in rr]) for rr in fam.values()]) if fam else None
                result.append(summary)
    R.save('intervention-report.json',dict(kind='real',results=result,
        inference='Family-aggregated paired t intervals; 4 primary tests alpha .05/4. Secondary comparisons exploratory, no semantic promotion from significant damage.',
        limitations=['No free natural generation or broad general capability battery','Norm controls use only strictly matched rows for the reported paired direction comparison; qualification may select a subset.']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','summarize']);ap.add_argument('--hypothesis',choices=R.CANDIDATES,default='structure');a=ap.parse_args();R.configure()
    run(a.hypothesis) if a.stage=='run' else summarize()


if __name__=='__main__':main()
