"""Native A scaling in the original model, and predicted downstream effects."""
import argparse
import gc
import time
import numpy as np
from threadpoolctl import threadpool_limits
import branch_reuse14 as B
import branch_reuse14_collect as C
import branch_reuse14_analysis as A
import branch_reuse14_execution as E
import recursive_branch14 as R
from recursive_branch14_intervene import perturb

CHUNK=16


def cases(rows):
    wanted=set()
    for branch in A.BRANCHES:wanted.update(sorted({r['family'] for r in rows if r['branch']==branch})[:40])
    return [i for i,r in enumerate(rows) if r['family'] in wanted]


def specifications(kind,selection):
    if kind=='B19':
        group=selection['group'];units=group['units'];match=group['matched']
        return dict(baseline=None,restore=(units,1.),single_0=([64],0.),single_2=([64],2.),
            group_0=(units,0.),group_2=(units,2.),matched_0=(match,0.),matched_2=(match,2.),
            single_norm=([64],0.),group_norm=(units,0.))
    result=dict(baseline=None,restore=(selection['graphs']['correlation']['edges'][-1]['parents'],1.))
    for name,graph in selection['graphs'].items():
        edge=graph['edges'][-1]
        for key,units in [(name,edge['parents']),(name+'_matched',edge['random'][0])]:
            for dose in [0,2]:result[f'{key}_{dose}']=(units,float(dose))
    return result


def final_position(module,args,output,rows,units,dose):
    import torch
    value=args[0].clone();positions=torch.tensor([r['answer_pos'] for r in rows],device='cuda')
    for i,pos in enumerate(positions):value[i,pos,units]*=dose
    return torch.nn.functional.linear(value,module.weight)


def run(kind):
    B.configure(200*2**20);E.admit();selection=A.admit();C.bind()
    rows,meta=A.load('confirm');assert meta['behavior']['passed']
    chosen=cases(rows);selected=[rows[i] for i in chosen];assert len(selected)==40*4*len(A.BRANCHES)
    arms=specifications(kind,selection);names=list(arms);block=19 if kind=='B19' else 36
    destination=B.OUT/f'intervention-{kind}';destination.mkdir(exist_ok=True)
    assert not (destination/'result.json').exists(),'Completed intervention immutable'
    refs=A.state('confirm',40,1)[chosen];start=time.monotonic();all_results=[];checks=[];all_final=[];all_targets={}
    targets={}
    if kind=='B36':
        for b in [37,38,39]:targets[b]=sorted({u for g in selection['graphs'].values() for e in g['edges'] if e['child']==b for u in e['children']})
        all_targets={b:[] for b in targets}
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context();rng=np.random.default_rng(B.SEED+block)
        directions={name:torch.from_numpy(rng.normal(size=C.DIM).astype(np.float32)).cuda() for name in names if name.endswith('_norm')}
        for direction in directions.values():direction/=direction.norm()
        for lo in range(0,len(selected),CHUNK):
          rr=selected[lo:lo+CHUNK];size=len(rr);receipt=destination/f'chunk-{lo:05d}.json';archive=destination/f'chunk-{lo:05d}.npz'
          if receipt.exists():
            commit=B.read(receipt);assert commit['row_ids']==[r['id'] for r in rr]
            assert commit['execution_sha256']==B.file_sha(B.OUT/'execution-freeze.json') and commit['archive_sha256']==B.file_sha(archive)
            all_results.extend(commit['rows']);checks.append(commit['check'])
            with np.load(archive) as data:
                all_final.append(data['final_raw'])
                for b in targets:all_targets[b].append(data[f'B{b}'])
            continue
          assert not archive.exists(),'Uncommitted intervention chunk requires explicit inspection'
          emb=R.C.weight('model.embed_tokens.weight').cuda()
          stage=torch.nn.functional.embedding(torch.from_numpy(R.padded(rr)).cuda(),emb);del emb;torch.cuda.empty_cache()
          mask=R.mask_for(rr,causal);states=None;control={};captured={b:np.empty((len(names),size,len(units),3),np.float32) for b,units in targets.items()}
          for b in range(40):
            layer=R.C.load_block(b,cfg)
            if b==block:states=stage[None].repeat(len(names),1,1,1);del stage
            active=['prefix'] if b<block else names
            for ai,name in enumerate(active):
              for i in range(0,size,C.BATCH):
                local=rr[i:i+C.BATCH];hooks=[];record={};capture={}
                if b in targets:
                    hooks.extend([layer.mlp.gate_proj.register_forward_hook(lambda m,a,y:capture.update(G=y)),
                        layer.mlp.up_proj.register_forward_hook(lambda m,a,y:capture.update(V=y)),
                        layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:capture.update(A=a[0]))])
                if b==block and name!='baseline':
                    units,dose=arms[name]
                    def hook(m,a,y,units=units,dose=dose,name=name,local=local):
                        if kind=='B36':return final_position(m,a,y,local,units,dose)
                        value,stats=perturb(m,a,y,local,units,dose,directions.get(name));record.update(stats);return value
                    hooks.append(layer.mlp.down_proj.register_forward_hook(hook))
                current=stage if b<block else states[ai]
                current[i:i+len(local)]=layer(current[i:i+len(local)],attention_mask=mask[i:i+len(local)],position_ids=pos,position_embeddings=pe,use_cache=False)
                for handle in hooks:handle.remove()
                if b in targets:
                    captured[b][ai,i:i+len(local)]=torch.stack([R.pick(capture[f],local)[:,1,targets[b]].float() for f in 'GVA'],-1).cpu().numpy()
                if 'relative_error' in record:
                  for j,r in enumerate(local):
                    region=record['region'][j];error=record['relative_error'][j][region]
                    control[name,i+j]=dict(all_token_norm_matched=bool((error<=.02).all()),matched_fraction=float((error<=.02).float().mean()),max_relative_error=float(error.max()))
                capture.clear();record.clear()
            del layer;gc.collect();torch.cuda.empty_cache()
            if b%5==4 or b==block:B.log('native_intervention_'+kind,lo+size*(b+1)/40,len(selected),start,block=b,chunk=lo)
          picked=torch.stack([R.pick(states[a],rr)[:,1] for a in range(len(names))])
          rawbits=R.bits(picked);norm=Qwen3RMSNorm(C.DIM,eps=cfg.rms_norm_eps).cuda()
          norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
          final=norm(picked);reference=norm(torch.from_numpy(refs[lo:lo+size]).to('cuda',torch.bfloat16))
          assert torch.equal(final[0],final[1]),'Dose1 numerical no-op failed'
          relative=float((final[0].float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12))
          head=R.C.weight('lm_head.weight').cuda();items=[];klref=[]
          for i in range(0,size,C.BATCH):
            count=min(C.BATCH,size-i);baseline=torch.nn.functional.linear(final[0,i:i+count],head).float();lp=baseline.log_softmax(-1);prob=lp.exp()
            ref=torch.nn.functional.linear(reference[i:i+count],head).float().log_softmax(-1);klref.extend((prob*(lp-ref)).sum(-1).cpu().tolist())
            for ai,name in enumerate(names):
              logits=torch.nn.functional.linear(final[ai,i:i+count],head).float();kl=(prob*(lp-logits.log_softmax(-1))).sum(-1)
              for j,r in enumerate(rr[i:i+count]):
                scores=logits[j,r['choice_ids']].cpu().tolist();bs=baseline[j,r['choice_ids']].cpu().tolist();gold=r['gold']
                margin=scores[gold]-max(x for k,x in enumerate(scores) if k!=gold);bm=bs[gold]-max(x for k,x in enumerate(bs) if k!=gold)
                items.append(dict(id=r['id'],family=r['family'],branch=r['branch'],role=r['role'],title=r['title'],arm=name,
                    scores=scores,baseline_scores=bs,gold=gold,correct=int(np.argmax(scores))==gold,baseline_correct=int(np.argmax(bs))==gold,
                    delta_margin=margin-bm,KL=float(kl[j]),top1=int(logits[j].argmax()),baseline_top1=int(baseline[j].argmax()),
                    norm_control=control.get((name,i+j))))
          check=dict(chunk=lo,restore_bitwise=True,replay_relative_L2=relative,replay_KL_max=max(klref),baseline_bitwise=bool(torch.equal(final[0],reference)))
          assert relative<=.02 and max(klref)<=.001,'Baseline replay gate failed'
          C.write_arrays(archive,dict(final_raw=rawbits,**{f'B{b}':value for b,value in captured.items()}))
          B.save(str(receipt.relative_to(B.OUT)),dict(row_ids=[r['id'] for r in rr],rows=items,check=check,
              archive_sha256=B.file_sha(archive),execution_sha256=B.file_sha(B.OUT/'execution-freeze.json')))
          all_results.extend(items);checks.append(check);all_final.append(rawbits)
          for b,value in captured.items():all_targets[b].append(value)
          del states,mask,picked,final,reference,norm,head,baseline,lp,prob,ref,logits,kl,captured;gc.collect();torch.cuda.empty_cache()
        B.save(str((destination/'result.json').relative_to(B.OUT)),dict(kind='real',rows=all_results,checks=checks,
            arms=names,row_indices=chosen,row_ids=[r['id'] for r in selected],targets={str(k):v for k,v in targets.items()},
            chunk_archives={p.name:B.file_sha(p) for p in sorted(destination.glob('*.npz'))},seconds=time.monotonic()-start,
            source_sha256=B.file_sha(__file__),execution_sha256=B.file_sha(B.OUT/'execution-freeze.json')))
    B.log('intervention_complete_'+kind,len(selected),len(selected),start)


def summarize():
    B.configure();E.admit();selection=A.admit();result=[];causal=[]
    for kind in ['B19','B36']:
      data=B.read(B.OUT/f'intervention-{kind}/result.json');rows=data['rows']
      for arm in data['arms']:
       if arm in ['baseline','restore']:continue
       for branch in A.BRANCHES:
        use=[r for r in rows if r['branch']==branch and r['arm']==arm]
        family=sorted({r['family'] for r in use});delta=np.array([np.mean([r['delta_margin'] for r in use if r['family']==f]) for f in family])
        primary=kind=='B19' and arm in ['single_0','group_0']
        interaction=[]
        for f in family:
            fr=[r for r in use if r['family']==f]
            interaction.append(np.mean([r['delta_margin'] for r in fr if r['role']=='rule'])-np.mean([r['delta_margin'] for r in fr if r['role']=='record']))
        item=dict(kind=kind,arm=arm,branch=branch,primary=primary,margin=A.inference(delta,.05/6 if primary else .05),
            relation_specificity=A.inference(interaction),accuracy=float(np.mean([r['correct'] for r in use])),
            baseline_accuracy=float(np.mean([r['baseline_correct'] for r in use])),mean_KL=float(np.mean([r['KL'] for r in use])))
        if arm.endswith('_norm'):
            qualified=[r for r in use if r['norm_control']['all_token_norm_matched']]
            item['strict_norm_matched_rows']=len(qualified);item['total_rows']=len(use)
        result.append(item)
    data=B.read(B.OUT/'intervention-B36/result.json');allrows,_=A.load('confirm');ix=data['row_indices'];rows=[allrows[i] for i in ix]
    arrays={b:[] for b in [37,38,39]}
    for path in sorted((B.OUT/'intervention-B36').glob('chunk-*.npz')):
        with np.load(path) as file:
            for b in arrays:arrays[b].append(file[f'B{b}'])
    arrays={b:np.concatenate(values,axis=1) for b,values in arrays.items()}
    with threadpool_limits(limits=4):
      for method,graph in selection['graphs'].items():
        seed=A.field('confirm',36,'A',1,graph['edges'][-1]['parents'])[ix]
        observed_indices=[data['targets']['39'].index(u) for u in graph['roots']]
        rootmodel=A.get(f'{method}-B38-r1-parent');scale=rootmodel['ys']
        for mode in ['direct','composed']:
          base=A.rollout(graph,method,1,initial=seed,compose=mode=='composed')
          for dose in [0,2]:
            arm=f'{method}_{dose}';ai=data['arms'].index(arm)
            pred=A.rollout(graph,method,1,initial=seed*dose,compose=mode=='composed')-base
            actual=(arrays[39][ai][:,observed_indices]-arrays[39][0][:,observed_indices]).reshape(len(rows),-1)
            loss=np.mean(((pred-actual)/scale)**2,axis=1);zero=np.mean((actual/scale)**2,axis=1)
            for branch in A.BRANCHES:
              take=np.array([i for i,r in enumerate(rows) if r['branch']==branch]);rr=[rows[i] for i in take]
              ratio=A.ratio_summary(rr,loss[take],zero[take]);magnitude=float(np.sqrt(zero[take].mean()))
              causal.append(dict(method=method,mode=mode,dose=dose,branch=branch,ratio_to_zero_effect=ratio,
                  standardized_actual_RMS=magnitude,informative=magnitude>=.05,passed=magnitude>=.05 and ratio['ratio']<=.9))
    B.save('intervention-report.json',dict(kind='real',effects=result,causal_predictions=causal,
        source_sha256=B.file_sha(__file__),scope='B19 whole body, B36 answer position only. Dose1 is numerical no-op, not independent rescue. Causal prediction is root G/V/A effect, separate from behavior and natural prediction fidelity.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','summarize']);p.add_argument('--kind',choices=['B19','B36'],default='B19');args=p.parse_args()
    run(args.kind) if args.stage=='run' else summarize()
