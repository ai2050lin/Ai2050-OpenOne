"""Actual block16 parameter training through the entire unmodified native suffix.

No frozen final-tail surrogate is used here. All original checkpoint files stay
read-only. FP32 MLP arithmetic is explicitly bridged to the BF16 residual stream.
"""
import argparse
import gc
from collections import defaultdict
from rdc_binding_common import *

def dataset():
    rows=gzread(BASE/'natural_discovery.json.gz');result=[]
    for cohort in ('gum','ewt'):
      for split,n in [('train',48),('validation',12),('test',24)]:
        pool=sorted([r for r in rows if r['cohort']==cohort and r['split']==split],key=lambda r:ranked('middle/'+r['sample_id']))[:n]
        for r in pool:
            ids=r['prompt_ids'][:128];begin=len(ids)//2;positions=np.unique(np.linspace(begin,len(ids)-2,8,dtype=int)).tolist()
            perm=np.random.default_rng(int(ranked(r['sample_id'])[:8],16)).permutation(begin)
            control=[ids[j] for j in perm]+ids[begin:]
            assert [ids[p+1] for p in positions]==[control[p+1] for p in positions]
            result.append({'sample_id':r['sample_id'],'source_group':r['source_group'],'split':split,'cohort':cohort,
              'ids':ids,'control_ids':control,'positions':positions,'targets':[ids[p+1] for p in positions]})
    return result

def setup(baseline_rows):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from torch.utils.checkpoint import checkpoint
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    for p in model.parameters():p.requires_grad_(False)
    native_baseline=evaluate(model,baseline_rows)
    target=model.model.layers[16].mlp
    original={k:p.detach().float().cpu().clone() for k,p in target.named_parameters()}
    target.float()
    for p in target.parameters():p.requires_grad_(True)
    def before(m,args):return (args[0].float(),)+args[1:]
    def after(m,args,out):return out.to(torch.bfloat16)
    handles=[target.register_forward_pre_hook(before),target.register_forward_hook(after)]
    # Explicit checkpointing works in eval mode and keeps native attention computation.
    for layer in model.model.layers[17:]:
        forward=layer.forward
        def wrapped(*args,_forward=forward,**kwargs):
            if torch.is_grad_enabled():return checkpoint(_forward,*args,use_reentrant=False,**kwargs)
            return _forward(*args,**kwargs)
        layer.forward=wrapped
    return model,tok,target,original,handles,native_baseline

def forward(model,row,control=False):
    import torch
    device=model.get_input_embeddings().weight.device
    ids=torch.tensor([row['control_ids'] if control else row['ids']],device=device)
    hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,row['positions']]
    logits=model.lm_head(hidden).float();lp=logits.log_softmax(-1)
    targets=torch.tensor(row['targets'],device=device)
    loss=-lp[torch.arange(len(targets),device=device),targets]
    return loss,hidden,lp

def evaluate(model,rows):
    import torch
    packet=defaultdict(list)
    with torch.no_grad():
      for row in rows:
        loss,h,lp=forward(model,row)
        packet['loss'].append(loss.cpu().numpy());packet['H36_postnorm'].append(bits(h));packet['argmax'].append(lp.argmax(-1).cpu().numpy())
    return {k:np.concatenate(v) for k,v in packet.items()}

def main(pilot=False):
    import torch
    out=BASE/'middle_training'
    if (out/('pilot.json' if pilot else 'result.json')).exists():return
    rows=dataset();compressed(out/'material.json.gz',rows)
    baseline_rows=rows[:4] if pilot else [r for r in rows if r['split']!='train']
    start=time.monotonic();model,tok,target,original,handles,native_baseline=setup(baseline_rows)
    parameters=dict(target.named_parameters());trace=[]
    try:
        # Pilot computes real gradients through 19 downstream attention/MLP blocks.
        tick=time.monotonic();loss,_,_=forward(model,rows[0]);loss.mean().backward();torch.cuda.synchronize()
        gradients={k:p.grad.detach() for k,p in parameters.items()}
        assert all(torch.isfinite(g).all() for g in gradients.values())
        report={'timestamp':stamp(),'source':snapshot(Path(__file__)),'forward_backward_seconds':time.monotonic()-tick,
          'loss':float(loss.detach().mean()),'gradient_norm':float(torch.stack([g.square().sum() for g in gradients.values()]).sum().sqrt()),
          'parameters':sum(p.numel() for p in parameters.values()),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
          'precision':'Only block16 gate/up/down FP32; native input/output bridge BF16. All other weights frozen BF16; no quantization.',
          'suffix':'All native17..35blocks and finalnorm/fullvocabulary readout participate in autograd; checkpointed exact recomputation.',
          'passed':True}
        bridge=evaluate(model,baseline_rows)
        report['bridge_minus_original_native_mean_nll']=float((bridge['loss']-native_baseline['loss']).mean())
        report['bridge_original_argmax_agreement']=float(np.mean(bridge['argmax']==native_baseline['argmax']))
        npz(out/('pilot_original_native.npz' if pilot else 'original_native.npz'),**native_baseline)
        for p in parameters.values():p.grad=None
        del gradients,loss
        if pilot:
            save(out/'pilot.json',report);ledger('middle_backward_pilot',time.monotonic()-start);print('MIDDLE_NATIVE_PILOT',report,flush=True);return
        assert read(out/'pilot.json')['passed']
        protocol={'seeds':[2733,2734],'steps':32,'accumulation_rows':4,'learning_rate':.02,'gradient_clip':1.,'checkpoints':[1,8,32],
          'train_rows':96,'validation_rows':24,'test_rows':48,'target_positions_per_row':8,'same_draws_and_target_ids':True,
          'scope':'Restricted continued training, not original pretraining. Order control changes difficulty; token identity/targets/length held fixed.'}
        immutable(out/'protocol.json',protocol)
        train=[r for r in rows if r['split']=='train'];panel=[r for r in rows if r['split']!='train']
        baseline=evaluate(model,panel);npz(out/'bridge_baseline.npz',**baseline)
        initial_norm=float(torch.stack([p.detach().square().sum() for p in parameters.values()]).sum().sqrt())
        runs=[]
        for seed in protocol['seeds']:
          draws=np.random.default_rng(seed).integers(len(train),size=(protocol['steps'],protocol['accumulation_rows']))
          npz(out/f'draws_{seed}.npz',indices=draws)
          for condition in ('coherent','order_control'):
            folder=out/f'{condition}_{seed}'
            if (folder/'result.json').exists():runs.append(read(folder/'result.json'));continue
            with torch.no_grad():
                for k,p in parameters.items():p.copy_(original[k].to(p.device))
            runstart=time.monotonic();trace=[];checkpoints=[]
            for step,indices in enumerate(draws,1):
                batchloss=[]
                for ix in indices:
                    loss,_,_=forward(model,train[ix],condition=='order_control')
                    (loss.mean()/len(indices)).backward();batchloss.append(float(loss.detach().mean()));del loss
                norm=float(torch.nn.utils.clip_grad_norm_(list(parameters.values()),protocol['gradient_clip']))
                with torch.no_grad():
                    for p in parameters.values():p.add_(p.grad,alpha=-protocol['learning_rate']);p.grad=None
                trace.append({'step':step,'loss':float(np.mean(batchloss)),'gradient_norm':norm})
                if step in protocol['checkpoints']:
                    packet=evaluate(model,panel);npz(folder/f'checkpoint{step}.npz',**packet)
                    delta=packet['loss']-baseline['loss'];meta=[r for r in panel for p in r['positions']]
                    stats=[]
                    for split in ('validation','test'):
                      for cohort in ('gum','ewt'):
                        ix=np.array([i for i,r in enumerate(meta) if r['split']==split and r['cohort']==cohort])
                        stats.append({'split':split,'cohort':cohort,'n':len(ix),'loss_delta':float(delta[ix].mean()),
                          'cluster':clustered(delta[ix],[meta[i]['source_group'] for i in ix])})
                    checkpoints.append({'step':step,'stats':stats})
                    save(folder/'progress.json',{'trace':trace,'checkpoints':checkpoints})
                    print('MIDDLE_NATIVE_CHECKPOINT',condition,seed,step,stats,flush=True)
                assert time.monotonic()-start<read(BASE/'resources.json')['per_process_ceiling_seconds']
            delta={k:p.detach().cpu().numpy()-original[k].numpy() for k,p in parameters.items()}
            npz(folder/'parameter_deltas.npz',**delta)
            norm=float(np.sqrt(sum(np.sum(a.astype(float)**2) for a in delta.values())))
            run={'condition':condition,'seed':seed,'trace':trace,'checkpoints':checkpoints,'seconds':time.monotonic()-runstart,
              'relative_parameter_displacement':norm/initial_norm,'delta_sha':sha(folder/'parameter_deltas.npz')}
            save(folder/'result.json',run);runs.append(run);guard(400*1024**2)
        save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'pilot':report,'runs':runs,
          'seconds':time.monotonic()-start,'scope':protocol['scope'],'original_checkpoint_written':False})
        ledger('middle_native_training',time.monotonic()-start)
    except Exception as exc:
        import traceback
        save(out/('failure_'+str(int(time.time()))+'.json'),{'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start})
        ledger('failed_middle_training',time.monotonic()-start);raise
    finally:
        for h in handles:h.remove()
        del model,target,parameters,original;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');main(p.parse_args().pilot)
