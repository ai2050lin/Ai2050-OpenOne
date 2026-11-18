"""Posthoc natural-state B39 arithmetic counterfactual; not replay of edited branches."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_behavior import Runtime
from source_formation_causal import damped_output
from source_binding_native_v2 import unbits

def run(model,split):
    import torch
    from source_formation_native import verify
    verify();root=D.OUT/model;meta=D.read(root/'formation-fit.json');c=next(x for x in meta['mlps'] if x['block']==39)
    data=D.read(root/f'{split}-causal.json');rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];n=len(rows)
    rt=Runtime(model);timer=time.monotonic()
    output=np.load(root/f'{split}-native/mlp_output.npy',mmap_mode='r');units=np.load(root/f'{split}-native/units.npy',mmap_mode='r')
    w=rt.weight('model.layers.39.mlp.down_proj.weight','cpu').float()
    reference={a:np.zeros(n,np.float32) for a in ['mlp_selected','mlp_random']}
    for entry in data['local']:
        if entry['kind']=='mlp' and entry['block']==39:
            lo=entry['chunk']+entry['offset'];reference[entry['arm']][lo:lo+len(entry['target_norm'])]=entry['target_norm']
    assert np.array_equal(reference['mlp_selected'],reference['mlp_random'])
    stored={};parts={a:{'bf16_18':[],'bf16_28':[],'continuous_fp32':[],'largest_coordinate_energy_fraction':[],'nearest_output_spacing':[]} for a in reference};targets=[]
    for lo in range(0,n,64):
        hi=min(lo+64,n);base=torch.from_numpy(np.array(output[lo:hi,1])).view(torch.bfloat16)
        a=torch.from_numpy(unbits(units[lo:hi,1]));delta={};fullnorm={}
        for arm,key in [('mlp_selected','selected'),('mlp_random','random')]:
            idx=c[key];delta[arm]=-(a[:,idx]@w[:,idx].T)
            fullnorm[arm]=((base.float()+delta[arm]).bfloat16().float()-base.float()).norm(dim=1)
        target=.5*torch.minimum(*fullnorm.values());targets.append(target.numpy())
        for arm,d in delta.items():
            for steps in [18,28]:
                changed,norm,scale=damped_output(base,d,target,iterations=steps)
                parts[arm]['bf16_'+str(steps)].append((torch.abs(norm-target)/target.clamp_min(1e-8)).numpy())
            scale=target/d.norm(dim=1).clamp_min(1e-8);changed=base.float()+scale[:,None]*d
            error=((changed-base.float()).norm(dim=1)-target).abs()/target.clamp_min(1e-8)
            parts[arm]['continuous_fp32'].append(error.numpy())
            energy=d.square();ix=energy.argmax(1);fraction=energy.max(1).values/energy.sum(1).clamp_min(1e-12)
            parts[arm]['largest_coordinate_energy_fraction'].append(fraction.numpy())
            b=base[torch.arange(len(base)),ix]
            up=torch.nextafter(b,torch.full_like(b,float('inf'))).float()-b.float()
            down=b.float()-torch.nextafter(b,torch.full_like(b,float('-inf'))).float()
            parts[arm]['nearest_output_spacing'].append(torch.minimum(up.abs(),down.abs()).numpy())
        D.log(model+'/'+split+'/posthoc-native39-matching-numerics',hi,n,timer)
    target=np.concatenate(targets);results={}
    for arm,groups in parts.items():
        arrays={k:np.concatenate(v) for k,v in groups.items()}
        results[arm]={k:{'mean':float(v.mean()),'p95':float(np.quantile(v,.95)),'maximum':float(v.max()),
            **({'fraction_within_5pct':float(np.mean(v<=.05))} if k in ['bf16_18','bf16_28','continuous_fp32'] else {})} for k,v in arrays.items()}
        results[arm]['bf16_18_versus28_max_error_change']=float(np.max(np.abs(arrays['bf16_18']-arrays['bf16_28'])))
        stored.update({arm+'_'+k:v for k,v in arrays.items()})
    stored['cpu_target_norm']=target;stored['original_gpu_target_norm']=reference['mlp_selected']
    path=root/f'{split}-native39-matching-numerics.npz';assert not path.exists();np.savez(path,**stored)
    relative=np.abs(target-reference['mlp_selected'])/np.maximum(reference['mlp_selected'],1e-8)
    D.save(f'{model}/{split}-native39-matching-numerics.json',{'stage':'posthoc numerical diagnosis only','block':39,'n':n,
        'native_counterfactual':'Uses unedited original B39. Actual causal B39 inputs differ after B38 edits; CPU-vs-GPU target difference is descriptive, not an exact-replay criterion.',
        'cpu_vs_gpu_target_relative_error_max':float(relative.max()),'results':results,'arrays_sha256':D.sha(path),'source_sha256':D.sha(Path(__file__)),
        'scope':'Natural-state arithmetic counterfactual only, not replay of actual edited B39. FP32 is not a new original-model intervention. This cannot rescue either failed frozen BF16 matching gate.'},model)
    output._mmap.close();units._mmap.close();del w;gc.collect()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--split',required=True);a=ap.parse_args();run(a.model,a.split)
