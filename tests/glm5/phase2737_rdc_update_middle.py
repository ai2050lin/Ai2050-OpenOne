"""Actual content vs batch-format-constrained block16 training through native suffix."""
import argparse
import gc
from collections import defaultdict
from rdc_update_common import *

def datasets():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
    program=gzread(BASE/'program_material.json.gz');language=gzread(BASE/'language_material.json.gz');natural=gzread(BASE/'natural_material.json.gz')
    def convert(r):
        rr=dict(r,ids=r['prompt_ids'],positions=[len(r['prompt_ids'])-1],targets=r['target_ids'],candidates=r.get('candidate_ids',digits))
        if r['kind']=='controlled_language':rr['cohort']=r['family']+'/'+r['language']+'/'+r['answer_style']
        return rr
    train=[convert(r) for r in program if r['split']=='train' and r['representation'] in ('en','python')]
    panel=[convert(r) for r in program if r['split']!='train']+[convert(r) for r in language if r['split']=='language_test']
    for r in natural:
        panel.append(dict(r,ids=r['prompt_ids'],positions=r['anchors'],targets=[r['prompt_ids'][p+1] for p in r['anchors']],candidates=None))
    assert len(train)==192 and len(panel)==672
    return train,panel

def objectives(model,row):
    import torch
    device=model.get_input_embeddings().weight.device;ids=torch.tensor([row['ids']],device=device)
    post=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,row['positions']]
    logits=model.lm_head(post).float();lp=logits.double().log_softmax(-1);target=torch.tensor(row['targets'],device=device)
    full=-lp[torch.arange(len(target),device=device),target]
    if row['candidates'] is None:return full,None,None,lp.argmax(-1),None
    candidates=row['candidates'];clp=logits[:,candidates].double().log_softmax(-1)
    local=torch.tensor([candidates.index(t) for t in row['targets']],device=device)
    content=-clp[torch.arange(len(target),device=device),local];format_=-torch.logsumexp(lp[:,candidates],-1)
    return full,content,format_,lp.argmax(-1),torch.tensor(candidates,device=device)[clp.argmax(-1)]

def evaluate(model,rows):
    import torch
    values=defaultdict(list)
    with torch.no_grad():
      for row in rows:
        f,c,fmt,top,ctop=objectives(model,row)
        values['full_loss'].append(f.cpu().numpy());values['content_loss'].append(np.zeros(len(f)) if c is None else c.cpu().numpy())
        values['format_loss'].append(np.zeros(len(f)) if fmt is None else fmt.cpu().numpy());values['has_candidate'].append(np.full(len(f),c is not None))
        values['argmax'].append(top.cpu().numpy());values['conditional_argmax'].append(np.full(len(f),-1) if ctop is None else ctop.cpu().numpy())
    return {k:np.concatenate(v) for k,v in values.items()}

def summarize(rows,packet,baseline):
    meta=[r for r in rows for _ in r['positions']];out=[]
    for split,cohort in sorted({(r['split'],r['cohort']) for r in meta}):
        ix=[i for i,r in enumerate(meta) if (r['split'],r['cohort'])==(split,cohort)];r={'split':split,'cohort':cohort,'positions':len(ix)}
        for part in ('full_loss','content_loss','format_loss'):
            if part!='full_loss' and not packet['has_candidate'][ix].all():continue
            delta=packet[part][ix]-baseline[part][ix];r[part+'_delta']=float(delta.mean());r[part+'_cluster']=clustered(delta,[meta[i]['source_group'] for i in ix])
        targets=np.array([t for row in rows for t in row['targets']]);r['first_argmax_accuracy']=float(np.mean(packet['argmax'][ix]==targets[ix]))
        if packet['has_candidate'][ix].all():r['conditional_accuracy']=float(np.mean(packet['conditional_argmax'][ix]==targets[ix]))
        out.append(r)
    return out

def setup(panel):
    import torch
    from torch.utils.checkpoint import checkpoint
    from phase2662_symmetric_mapping_contract import load_native
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    for p in model.parameters():p.requires_grad_(False)
    native=evaluate(model,panel);target=model.model.layers[16].mlp
    original={name:p.detach().float().cpu().clone() for name,p in target.named_parameters()};target.float()
    for p in target.parameters():p.requires_grad_(True)
    handles=[target.register_forward_pre_hook(lambda m,a:(a[0].float(),)+a[1:]),target.register_forward_hook(lambda m,a,o:o.to(torch.bfloat16))]
    for layer in model.model.layers[17:]:
        forward=layer.forward
        def wrapped(*args,_forward=forward,**kwargs):
            if torch.is_grad_enabled():return checkpoint(_forward,*args,use_reentrant=False,**kwargs)
            return _forward(*args,**kwargs)
        layer.forward=wrapped
    return model,tok,target,original,handles,native

def gradient_pair(model,parameters,row):
    import torch
    full,content,format_,_,_=objectives(model,row)
    cg=torch.autograd.grad(content.mean(),parameters,retain_graph=True)
    fg=torch.autograd.grad(format_.mean(),parameters)
    return cg,fg,{'full':float(full.detach().mean()),'content':float(content.detach().mean()),'format':float(format_.detach().mean())}

def main(pilot=False):
    import torch
    out=BASE/'middle_training';start=time.monotonic();path=out/('pilot.json' if pilot else 'result.json')
    if path.exists():return
    train,panel=datasets();small=panel[:4] if pilot else panel
    if not pilot:assert read(out/'pilot.json')['passed']
    model,tok,target,original,handles,native=setup(small);parameters=list(target.parameters());named=dict(target.named_parameters())
    try:
        tick=time.monotonic();cg,fg,loss=gradient_pair(model,parameters,train[0]);torch.cuda.synchronize();gradient_seconds=time.monotonic()-tick
        assert all(torch.isfinite(g).all() for g in cg+fg)
        report={'timestamp':stamp(),'source':snapshot(__file__),'two_objective_backward_seconds':gradient_seconds,'loss':loss,
          'parameters':sum(p.numel() for p in parameters),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'passed':gradient_seconds*512+600<7200,
          'precision':'Block16gate/up/down FP32, native BF16 input/output bridge; complete17..35suffix and full vocabulary, FP64 probability/conditional scoring.',
          'scope':'Library autograd through dtype casts is the actual training convention, not a classical derivative of a discrete BF16 map.'}
        del cg,fg
        bridge=evaluate(model,small);npz(out/('pilot_native.npz' if pilot else 'native_baseline.npz'),**native);npz(out/('pilot_bridge.npz' if pilot else 'bridge_baseline.npz'),**bridge)
        report['bridge_minus_native_full_loss']=float((bridge['full_loss']-native['full_loss']).mean())
        if pilot:
            report['seconds']=time.monotonic()-start;save(path,report);ledger('middle_content_backward_pilot',report['seconds']);print('MIDDLE_CONTENT_PILOT',report,flush=True);return
        protocol={'seeds':[2737,2738],'steps':32,'batch_rows':4,'step_parameter_norm':.02,'checkpoints':[1,8,32],
          'train_rows':192,'evaluation_rows':672,'conditions':['content','batch_format_constrained'],
          'constraint':'At each current batch, subtract from mean content gradient its component along that SAME batch mean format gradient. Exact rank1 constraint in all scalar coordinates, not the separate last-MLP192-gradient ridge constraint.',
          'matching':'Identical draws, labels, full native suffix, normalized FP32 step length. No matching claim for final accumulated or BF16 deployed norms.',
          'all_scalar_scope':'Only all74711040 block16 parameters trained. Other parameters are fixed but downstream computation participates in both backpropagations.'}
        immutable(out/'protocol.json',protocol);compressed(out/'train_panel_material.json.gz',{'train':train,'panel':panel})
        runs=[]
        for seed in protocol['seeds']:
            draws=np.random.default_rng(seed).integers(len(train),size=(32,4));npz(out/f'draws_{seed}.npz',indices=draws)
            for condition in protocol['conditions']:
                folder=out/f'{condition}_{seed}'
                if (folder/'result.json').exists():runs.append(read(folder/'result.json'));continue
                with torch.no_grad():
                    for name,p in named.items():p.copy_(original[name].to(p.device))
                runstart=time.monotonic();trace=[];checkpoints=[]
                for step,indices in enumerate(draws,1):
                    csum=[torch.zeros_like(p) for p in parameters];fsum=[torch.zeros_like(p) for p in parameters];losses=[]
                    for index in indices:
                        cg,fg,scores=gradient_pair(model,parameters,train[index]);losses.append(scores)
                        for c,f,a,b in zip(csum,fsum,cg,fg):c.add_(a,alpha=.25);f.add_(b,alpha=.25)
                        del cg,fg
                    dot=sum((c.double()*f.double()).sum() for c,f in zip(csum,fsum));fn2=sum(f.double().square().sum() for f in fsum)
                    cn2=sum(c.double().square().sum() for c in csum)
                    multiplier=dot/fn2.clamp_min(1e-30) if condition=='batch_format_constrained' else dot.new_tensor(0.)
                    direction=[c-multiplier.float()*f for c,f in zip(csum,fsum)];norm=torch.stack([v.square().sum() for v in direction]).sum().sqrt()
                    response=sum((v.double()*f.double()).sum() for v,f in zip(direction,fsum))
                    assert torch.isfinite(norm) and norm>0
                    with torch.no_grad():
                        for p,v in zip(parameters,direction):p.add_(v,alpha=-.02/float(norm))
                        after_losses=[]
                        for index in indices:
                            af,ac,am,_,_=objectives(model,train[index])
                            after_losses.append({'full':float(af.mean()),'content':float(ac.mean()),'format':float(am.mean())})
                    trace.append({'step':step,'training_loss':{k:float(np.mean([a[k] for a in losses])) for k in losses[0]},
                      'content_norm':float(cn2.sqrt()),'format_norm':float(fn2.sqrt()),'original_cosine':float(dot/(cn2*fn2).sqrt().clamp_min(1e-30)),
                      'direction_norm':float(norm),'format_inner_product_after_constraint':float(response),
                      'predicted_batch_content_delta':float(-.02*sum((v.double()*c.double()).sum() for v,c in zip(direction,csum))/norm),
                      'predicted_batch_format_delta':float(-.02*response/norm),
                      'actual_batch_delta':{k:float(np.mean([a[k] for a in after_losses])-np.mean([a[k] for a in losses])) for k in losses[0]}})
                    del csum,fsum,direction
                    if step in protocol['checkpoints']:
                        packet=evaluate(model,panel);npz(folder/f'checkpoint{step}.npz',**packet)
                        stats=summarize(panel,packet,bridge);checkpoints.append({'step':step,'reports':stats})
                        save(folder/'progress.json',{'trace':trace,'checkpoints':checkpoints});print('MIDDLE_CONTENT_CHECKPOINT',condition,seed,step,round(time.monotonic()-runstart,1),flush=True)
                    assert time.monotonic()-start<7200
                delta={name:p.detach().cpu().numpy()-original[name].numpy() for name,p in named.items()};npz(folder/'parameter_deltas.npz',**delta)
                run={'condition':condition,'seed':seed,'trace':trace,'checkpoints':checkpoints,'seconds':time.monotonic()-runstart,
                  'delta_norm':float(np.sqrt(sum(np.sum(d.astype(float)**2) for d in delta.values()))),'delta_sha256':sha(folder/'parameter_deltas.npz'),
                  'distinct_training_rows_drawn':len(set(draws.ravel().tolist()))}
                save(folder/'result.json',run);runs.append(run);guard(400*1024**2)
        with torch.no_grad():
            for name,p in named.items():p.copy_(original[name].to(p.device))
        reset=evaluate(model,panel[:4]);assert np.array_equal(reset['full_loss'],bridge['full_loss'][:4])
        result={'timestamp':stamp(),'source':snapshot(__file__),'runs':runs,'runtime_pilot':report,'seconds':time.monotonic()-start,
          'restored_bridge_first4_exact':True,'scope':'Restricted continued training on192 mixed-program training expressions. Includes untouched language-test and authentic natural content boundaries; no historical pretraining reconstruction or pure-semantic-subspace claim.'}
        save(path,result);ledger('native_middle_content_training',result['seconds']);print('MIDDLE_CONTENT_COMPLETE',result['seconds'],flush=True)
    except Exception as exc:failure(out,start,exc);raise
    finally:
        for handle in handles:handle.remove()
        del model,target,parameters,named,original;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');main(p.parse_args().pilot)
