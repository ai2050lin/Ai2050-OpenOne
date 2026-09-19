"""Gold-oracle and prefix-only repetition proxy are separate test-time updates."""
import gc
from collections import Counter
from rdc_binding_common import *
from rdc_binding_gradients import *

def main():
    import torch
    from rdc_law_native import Tail
    out=BASE/'beta_updates'
    if (out/'result.json').exists():return
    if (BASE/'suite_progress.json').exists() and not (out/'serialization_recovery.json').exists():
        previous=[r for r in read(BASE/'suite_progress.json') if r['job']=='phase2733_rdc_binding_beta.py' and r['returncode']!=0]
        if previous:
            save(out/'serialization_recovery.json',{'error':'numpy boolean first_token_correct not JSON serializable',
              'failed_process':previous[-1],'correction':'Explicit Python bool conversion; mathematical operation and material unchanged.',
              'completed_records_persisted':0,'action':'Rerun finite64case experiment from original checkpoints; saved first-case factors are reproducible.'})
            ledger('failed_beta_serialization',previous[-1]['wall_seconds'])
    rows=gzread(BASE/'program_material.json.gz');selected=[]
    for family in ('alias','conditional','mapping','addition'):
      for split in ('test','depth_test'):
       for rep in ('en','zh','python','en_reordered'):
        selected.extend([r for r in rows if r['family']==family and r['split']==split and r['representation']==rep][:2])
    assert len(selected)==64
    immutable(out/'protocol.json',{'sample_ids':[r['sample_id'] for r in selected],'parameter_step_norm':.10,
      'objectives':['oracle_gold_next_digit_CE','prefix_repeat_probability_mass','random_norm_control'],
      'proxy':'Sum next-token probability over nonspecial token IDs occurring at least3times in visible prefix. Uses no target or future token.',
      'scope':'Per-example transient lastMLP test-time parameter updates; gold is supervised oracle, not zero-shot/no-finetuning.'})
    start=time.monotonic();tail=Tail();original={k:v.detach().clone() for k,v in tail.w.items()}
    xa,ra,ta=program_arrays(selected);x=torch.tensor(xa,device='cuda');r=torch.tensor(ra,device='cuda');targets=torch.tensor(ta,device='cuda')
    natural=sorted([r for r in gzread(BASE/'natural_discovery.json.gz') if r['split']=='test'],key=lambda r:ranked('beta/'+r['sample_id']))[:16]
    cx=[];cr=[];ct=[]
    for row in natural:
        pos=row['anchors'][-1]
        with np.load(source_path(row)) as z:cx.append(unbits(z['x'])[pos]);cr.append(unbits(z['residual'])[pos])
        ct.append(row['prompt_ids'][pos+1])
    cx=torch.tensor(np.array(cx),device='cuda');cr=torch.tensor(np.array(cr),device='cuda');ct=torch.tensor(ct,device='cuda')
    with torch.no_grad():natural_loss=tail.forward(cx,cr,ct)['loss'].cpu().numpy()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True);special=set(tok.all_special_ids)
    records=[];checked=False
    for i,row in enumerate(selected):
        with torch.no_grad():
            for k,v in tail.w.items():v.copy_(original[k])
            z=tail.forward(x[i:i+1],r[i:i+1],targets[i:i+1],True)
            repeats=[token for token,n in Counter(row['prompt_ids']).items() if n>=3 and token not in special]
            mask=torch.zeros_like(z['probabilities']);mask[:,repeats]=1
            mass=(z['probabilities']*mask).sum(-1,keepdim=True)
            error=z['probabilities']*(mask-mass)
            proxy=error_factors(tail,z,x[i:i+1],error)
            gradients={'oracle_gold_next_digit_CE':dense_gradient(z['factors']),'prefix_repeat_probability_mass':dense_gradient(proxy)}
            npz(out/'factors'/f'{row["sample_id"]}.npz',**{'oracle_'+k:v.cpu().numpy() for k,v in z['factors'].items()},
              **{'proxy_'+k:v.cpu().numpy() for k,v in proxy.items()},repeated_ids=np.array(repeats,dtype=np.int32))
            initial={'loss':float(z['loss'][0]),'argmax':int(z['argmax'][0]),'entropy':float(z['entropy'][0]),'repeat_mass':float(mass[0,0])}
        if not checked:
            for v in tail.w.values():v.requires_grad_(True)
            autodiff=tail.forward(x[i:i+1],r[i:i+1]);objective=(autodiff['probabilities']*mask).sum();objective.backward()
            errors={k:float((v.grad-gradients['prefix_repeat_probability_mass'][k]).abs().max()/v.grad.abs().max().clamp_min(1e-12)) for k,v in tail.w.items()}
            assert max(errors.values())<3e-5,errors
            for v in tail.w.values():v.grad=None;v.requires_grad_(False)
            save(out/'proxy_autograd_check.json',{'source':snapshot(Path(__file__)),'all_parameter_relative_max_errors':errors,'passed':True})
            checked=True;del autodiff,objective
        gen=torch.Generator(device='cuda').manual_seed(273300+i)
        gradients['random_norm_control']={k:torch.randint(0,2,v.shape,device='cuda',generator=gen,dtype=torch.int8).float().mul_(2).sub_(1) for k,v in tail.w.items()}
        for name,gradient in gradients.items():
            unit,norm=normalized(gradient)
            if norm<1e-8:
                records.append({'sample_id':row['sample_id'],'objective':name,'status':'skipped_negligible_gradient','gradient_norm':norm});continue
            with torch.no_grad():
                for k,v in tail.w.items():v.copy_(original[k]);v.add_(unit[k],alpha=-.10)
                new=tail.forward(x[i:i+1],r[i:i+1],targets[i:i+1])
                coll=tail.forward(cx,cr,ct)['loss'].cpu().numpy()-natural_loss
            rec={k:row[k] for k in ('sample_id','source_group','family','representation','split','target')}
            rec.update(objective=name,status='evaluated',gradient_norm=norm,initial=initial,loss=float(new['loss'][0]),loss_delta=float(new['loss'][0])-initial['loss'],
              argmax=int(new['argmax'][0]),first_token_correct=bool(int(new['argmax'][0])==ta[i]),repeat_mass=float(new['probabilities'][:,repeats].sum()),
              repeat_set_includes_target=ta[i] in repeats,natural_collateral_loss_delta=float(coll.mean()),
              natural_collateral_cluster=clustered(coll,[r['source_group'] for r in natural]))
            records.append(rec)
            del unit,new
        save(out/'progress.json',records)
        del gradients,z,proxy
        if (i+1)%8==0:print('BETA_CURRENT_PREFIX_UPDATES',i+1,len(selected),flush=True)
    with torch.no_grad():
        for k,v in tail.w.items():v.copy_(original[k])
    summary=[]
    for name in ('oracle_gold_next_digit_CE','prefix_repeat_probability_mass','random_norm_control'):
        rr=[r for r in records if r['objective']==name and r['status']=='evaluated']
        summary.append({'objective':name,'rows':len(rr),'mean_loss_delta':float(np.mean([r['loss_delta'] for r in rr])),
          'first_token_accuracy':float(np.mean([r['first_token_correct'] for r in rr])),
          'mean_repeat_mass_delta':float(np.mean([r['repeat_mass']-r['initial']['repeat_mass'] for r in rr])),
          'natural_collateral_delta':float(np.mean([r['natural_collateral_loss_delta'] for r in rr]))})
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'records':records,'summary':summary,'seconds':time.monotonic()-start,
      'limitations':['Gold oracle cannot be deployed without answers; prefix proxy may penalize legitimately repeated digits/content.','All updated outputs are cached-prefix first-token FP32 scores. Separate nativeBF16 own-history deployment is required.','Local loss decrease does not prove hallucination/knowledge repair.','Every case starts from original weights, no accumulation or checkpoint overwrite.']})
    ledger('beta_oracle_prefix_proxy',time.monotonic()-start)
    del tail,original;gc.collect();torch.cuda.empty_cache()
    print('BETA_COMPLETE',summary,flush=True)

if __name__=='__main__':main()
