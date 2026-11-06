"""Fit evidence directions from exactly paired native writes, with draft and key nuisance removal."""
from collections import defaultdict

import numpy as np
import torch

import mlp_factor_steer_common as C
from mlp_factor_steer_materials import WRITE_ROLES, WORDS
from mlp_kv_behavior import score, word_indices
from mlp_fidelity_v2_tail import cluster_interval
from mlp_read_route_causal import branch
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease

LAYERS=[26,28,30,32,33,34]


def inputs():
    path=C.OUT/'inputs.json.gz';plan=C.read(C.OUT/'prereg.json');assert C.file_sha(path)==plan['inputs_sha256']
    return C.read_gz(path),plan


def array_key(b,kind):return f'B{b}_{kind}'


def residualize(e,d,s):
    """Per native writer, remove the orthonormal nuisance span across role coordinates."""
    original=e.shape;e=e.reshape(len(e),-1).astype(np.float64);d=d.reshape(len(d),-1).astype(np.float64);s=s.reshape(len(s),-1).astype(np.float64)
    for j in range(len(e)):
        basis=[]
        for x in [d[j].copy(),s[j].copy()]:
            for v in basis:x-=np.dot(v,x)*v
            norm=np.linalg.norm(x)
            if norm>1e-9:basis.append(x/norm)
        for v in basis:e[j]-=np.dot(v,e[j])*v
    return e.reshape(original).astype(np.float32)


def statistics(rows):
    factors=defaultdict(list);group=defaultdict(dict)
    for r in rows:group[r['b']][r['evidence'],r['selector'],r['draft']]=r
    for _,v in group.items():
        other=v[0,0,0]['other'];g=lambda e,s,d:v[e,s,d]['canonical_gap']
        factors['evidence_when_A_selected'].append(np.mean([g(1,0,d)-g(0,0,d) for d in range(2)]))
        factors['evidence_when_B_selected'].append(np.mean([g(1,1,d)-g(0,1,d) for d in range(2)]))
        factors['key_switch_different_attributes'].append(np.mean([(g(1-other,0,d)-g(1-other,1,d))*(1-2*other) for d in range(2)]))
        factors['key_switch_same_attribute'].append(np.mean([g(other,0,d)-g(other,1,d) for d in range(2)]))
        factors['draft_at_fixed_truth_and_key'].append(np.mean([g(e,s,1)-g(e,s,0) for e in range(2) for s in range(2)]))
    return {key:cluster_interval(v,C.SEED) for key,v in factors.items()}


def main():
    dest=C.OUT/'factorial-result.json.gz'
    if dest.exists():return
    corpus,plan=inputs();fit=corpus['fit'];assert len(fit)==C.FIT*8 and all(v['all_one_token'] for v in corpus['audits'])
    execution={**C.identity(),'schema':'factor_steer_fit_execution.v1','source_sha256':C.file_sha(__file__),
               'prereg_sha256':C.file_sha(C.OUT/'prereg.json'),'candidate_layers':LAYERS,'write_roles':WRITE_ROLES}
    C.save(C.OUT/'fit-execution.json',execution);torch.set_num_threads(4)
    rows=[];totals={};observed={};context={'capture':False}
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();words=word_indices(tok);handles=[]
        def store(b,kind,v):
            if not context['capture']:return
            i=torch.arange(len(v),device=v.device)[:,None];j=context['positions']
            cut=v[i,j].float()
            if kind=='H':cut=cut.reshape(len(v),len(WRITE_ROLES),32,128).transpose(1,2)
            else:cut=cut[:,None]
            observed[b,kind]=cut.cpu().numpy()
        for b in LAYERS:
            layer=model.model.layers[b]
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda _,args,b=b:store(b,'H',args[0])))
            handles.append(layer.mlp.register_forward_hook(lambda _,args,value,b=b:store(b,'M',value)))
        try:
            for lo in range(0,len(fit),8):
                case=fit[lo:lo+8];buffers=defaultdict(list)
                for start in [0,4]:
                    batch=case[start:start+4];x=torch.tensor([r['tokens'] for r in batch],device='cuda')
                    context.update(capture=True,positions=torch.tensor([[r['anchors'][k] for k in WRITE_ROLES] for r in batch],device='cuda'))
                    cache=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
                    context['capture']=False
                    for k,v in observed.items():buffers[k].append(v)
                    logits,generated=branch(model,x,cache,eos)
                    for j,r in enumerate(batch):
                        v=score(r,logits[j],generated[j],tok,words)
                        rows.append({k:z for k,z in r.items() if k!='tokens'}|{'clean':v,'canonical_gap':v['gap']*(1 if r['expected']=='fishing' else -1)})
                for k,v in buffers.items():
                    a=np.concatenate(v).reshape(2,2,2,*v[0].shape[1:]);e=(a[1,0]-a[0,0]).mean(0)
                    d=(a[:,:,1]-a[:,:,0]).mean((0,1));other=case[0]['other'];s=(a[other,1]-a[other,0]).mean(0)
                    accum=totals.setdefault(k,np.zeros((3,*e.shape),np.float64));accum+=np.stack([e,d,s])
                if ((lo+8)//8)%8==0:print('fit factorial backgrounds',(lo+8)//8,'/',C.FIT,flush=True)
        finally:
            for h in handles:h.remove()
    arrays={};summary={}
    for (b,kind),v in totals.items():
        e,d,s=v/C.FIT;pure=residualize(e,d,s);key=array_key(b,kind)
        arrays[key]=pure;arrays[key+'_raw']=e.astype(np.float32);arrays[key+'_draft']=d.astype(np.float32);arrays[key+'_key_same']=s.astype(np.float32)
        summary[key]={'raw_E_norm':float(np.linalg.norm(e)),'adjusted_E_norm':float(np.linalg.norm(pure)),'draft_norm':float(np.linalg.norm(d)),
                      'key_same_norm':float(np.linalg.norm(s))}
    path=C.OUT/'directions.npz';np.savez_compressed(path,**arrays)
    result={**execution,'status':'complete','directions_sha256':C.file_sha(path),'one_token_pairs':len(fit)//8*12,
            'backgrounds':C.FIT,'factorial_effects':statistics(rows),'writer_norms':summary,'rows':rows}
    C.write_gz(dest,result);print('factorial effects',result['factorial_effects'],flush=True)


if __name__=='__main__':main()
