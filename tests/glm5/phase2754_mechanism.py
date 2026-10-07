"""Freeze and test a joint-slot probe ablation, with native MLP unit accounting."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from phase2754_relation_stability import ROOT,OUT,FAMILIES,decode,write,sha,now,snapshot,parameter_vectors

def freeze():
    if (OUT/'mechanism_selection.json').exists():return
    mat=json.loads((OUT/'material.json').read_text(encoding='utf-8'));src={r['id']:r for r in mat['rows']}
    meta=[m for p in sorted((OUT/'4B/discovery').glob('chunk_*.json')) for m in json.loads(p.read_text(encoding='utf-8')) if src[m['id']]['split']=='train']
    signed=np.array([src[m['id']]['truth_sign']*np.array(m['source_numerator']['mlp'])/m['rms'] for m in meta]).mean(0)
    layers=np.argsort(signed)[-2:][::-1].tolist()
    sel=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));method=sel['selected']['semantic']
    assert method.startswith('slots'),'Mechanism protocol requires a jointly decoded slot method.'
    cut=int(method[5:]);d=2560
    with np.load(OUT/'probe_fits'/f'{method}.npz') as z:
        c=z['coef'][:,1];a=c[:d]/float(z['sd0']);b=c[d:2*d]/float(z['sd1']);e=c[2*d:]/float(z['sd2'])
        vectors=np.stack([a,b+e,-b+e])
        intercept=float(z['ym'][1])-sum(float(z[f'mu{j}']@c[j*d:(j+1)*d])/float(z[f'sd{j}']) for j in range(3))
    rng=np.random.default_rng(2754007);random=rng.normal(size=vectors.shape);random-=np.sum(random*vectors)/np.sum(vectors*vectors)*vectors
    random*=np.linalg.norm(vectors)/np.linalg.norm(random)
    np.savez(OUT/'mechanism_directions.npz',semantic=vectors,random=random,intercept=intercept)
    write(OUT/'mechanism_selection.json',dict(created_utc=now(),source=snapshot(Path(__file__)),method=method,cut=cut,unit_layers_zero_based=layers,
        layer_selection='Two largest positive train-mean truth-aligned MLP contributions after division by each run finalRMS. All-layer evidence retained; not defining backbone byTopK.',
        train_mlp_truth_contribution=signed.tolist(),conditions=['baseline','erase_probe','equal_norm_random'],
        intervention='At the chosen early boundary, jointly edit end/query-source/query-target token states. Remove frozen centered truth-probe projection; random shift has identical FP32 norm and is orthogonal to that joint probe direction.',
        formula='p=V dot X+intercept; delta_sem=-p V/||V||^2; delta_random=-p R/||V||^2, ||R||=||V|| and R dot V=0. No current truth label enters the intervention.',
        cases='Exactly the192 predefined4B prompts selected for14B replication,24worlds. Paired world bootstrap; no filtering by model correctness.',
        unit_accounting='All units of selected actual SwiGLU down-projection inputs; fixed parameter coefficient c=W_down.T*(gamma*(W_yes-W_no)); native matmul rounding retained as discrepancy.',
        limitations=['Probe ablation can be off-distribution and is not a minimal or unique semantic circuit.','Fixed one random direction and small24world subset limit causal generalization.','Native-coordinate predictions do not establish a unique code.'],
        budget_seconds=600,selection_sha256=sha(OUT/'selection.json'),directions_sha256=sha(OUT/'mechanism_directions.npz')))
    print(dict(method=method,cut=cut,unit_layers=layers),flush=True)

def run():
    freeze();assert not (OUT/'mechanism_done.json').exists(),'Completed intervention preserved.'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    start=time.time();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((OUT/'mechanism_selection.json').read_text(encoding='utf-8'))
    if cfg['source']['sha256']!=sha(Path(__file__)):
        revision=json.loads((OUT/'mechanism_execution_revision.json').read_text(encoding='utf-8'))
        assert revision['source']['sha256']==sha(Path(__file__)) and revision['protocol_sha256']==sha(OUT/'mechanism_selection.json')
    source={r['id']:r for r in json.loads((OUT/'material.json').read_text(encoding='utf-8'))['rows']}
    rows=[r for r in source.values() if r['replicate14B']]
    native={r['id']:r for p in (OUT/'4B/confirmation').glob('chunk_*.json') for r in json.loads(p.read_text(encoding='utf-8'))}
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    yes=tok.encode(' yes',add_special_tokens=False)[0];no=tok.encode(' no',add_special_tokens=False)[0]
    direction,gamma=parameter_vectors(ROOT/'models/hf/qwen3-4b',yes,no);readvector=direction.astype(float)*gamma
    with np.load(OUT/'mechanism_directions.npz') as z:vec=torch.tensor(z['semantic'],dtype=torch.float32,device='cuda');rand=torch.tensor(z['random'],dtype=torch.float32,device='cuda');intercept=float(z['intercept'])
    den=float(vec.square().sum())
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').eval().to('cuda')
    coefficients={str(i):(model.model.layers[i].mlp.down_proj.weight.float().T@torch.tensor(readvector,dtype=torch.float32,device='cuda')).detach().cpu().numpy() for i in cfg['unit_layers_zero_based']}
    np.savez(OUT/'unit_parameter_coefficients.npz',**coefficients)
    context={};units={};projected={};records=[];unit_bank={str(i):[] for i in cfg['unit_layers_zero_based']}
    def edit(module,args,out):
        hh=out[0] if isinstance(out,tuple) else out;positions=context['positions'];old=hh[0,positions].float();p=(old*vec).sum()+intercept
        mode=context['mode'];change=torch.zeros_like(old) if mode=='baseline' else -p/den*(vec if mode=='erase_probe' else rand)
        new=hh.clone();new[0,positions]=(old+change).to(hh.dtype)
        realized=new[0,positions].float()-old
        context['edit']=dict(probe_before=float(p),probe_after=float((new[0,positions].float()*vec).sum()+intercept),
            intended_norm=float(change.norm()),realized_norm=float(realized.norm()),relative_norm=float(realized.norm()/old.norm()))
        return (new,)+out[1:] if isinstance(out,tuple) else new
    handles=[model.model.layers[cfg['cut']-1].register_forward_hook(edit)]
    def unit_hook(index):
        def rec(module,args):units[str(index)]=args[0][0,-1].detach().float().cpu().numpy()
        return rec
    def projected_hook(index):
        def rec(module,args,out):projected[str(index)]=out[0,-1].detach().float().cpu().numpy()
        return rec
    raw={}
    def prenorm(module,args):raw['last']=args[0][0,-1].detach().float().cpu().numpy()
    handles.append(model.model.norm.register_forward_pre_hook(prenorm))
    for i in cfg['unit_layers_zero_based']:
        handles.append(model.model.layers[i].mlp.down_proj.register_forward_pre_hook(unit_hook(i)))
        handles.append(model.model.layers[i].mlp.register_forward_hook(projected_hook(i)))
    max_baseline=0
    try:
        for j,row in enumerate(rows):
            tm=row['tokenization']['4B'];context['positions']=[tm['length']-1]+tm['query_source_target_last_tokens'];assert len(set(context['positions']))==3
            ids=torch.tensor([tm['token_ids']],device='cuda')
            for mode in cfg['conditions']:
                context['mode']=mode;units.clear();projected.clear()
                with torch.inference_mode():
                    out=model(input_ids=ids,use_cache=False,logits_to_keep=1)
                    lg=out.logits[0,-1].float();pred=int(lg.argmax());margin=float(lg[yes]-lg[no])
                    rec=dict(id=row['id'],world=row['world'],group=row['group'],family=row['family'],split=row['split'],cell=row['cell'],mode=mode,
                        truth_sign=row['truth_sign'],margin=margin,prediction_id=pred,prediction_text=tok.decode([pred]),correct=tok.decode([pred]).strip().lower()==row['expected'],edit=context['edit'])
                    if mode=='baseline':
                        diff=abs(margin-native[row['id']]['margin']);max_baseline=max(max_baseline,diff);assert diff==0 and pred==native[row['id']]['prediction_id']
                        rms=float(np.sqrt(np.mean(raw['last'].astype(float)**2)+model.config.rms_norm_eps));rec['rms']=rms;rec['unit_projection_check']={}
                        for i in cfg['unit_layers_zero_based']:
                            k=str(i);unit_bank[k].append(units[k]);n1=float(units[k].astype(float)@coefficients[k]);n2=float(projected[k].astype(float)@readvector)
                            rec['unit_projection_check'][k]=dict(fp32_units_numerator=n1,native_mlp_numerator=n2,difference=n2-n1)
                    records.append(rec);del out,lg
            if (j+1)%32==0:print('mechanism',j+1,len(rows),time.time()-start,flush=True)
            assert time.time()-start<600,'Bounded intervention budget exceeded.'
        write(OUT/'mechanism_rows.json',records);np.savez(OUT/'unit_activations.npz',**{k:np.stack(v) for k,v in unit_bank.items()})
        write(OUT/'mechanism_done.json',dict(created_utc=now(),elapsed_seconds=time.time()-start,prompts=len(rows),forwards=len(records),worlds=24,baseline_repeat_max=max_baseline,
            source=snapshot(Path(__file__)),protocol_sha256=sha(OUT/'mechanism_selection.json'),unit_count={k:len(v) for k,v in coefficients.items()}))
    finally:
        for handle in handles:handle.remove()
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);a=p.parse_args();freeze() if a.mode=='freeze' else run()
