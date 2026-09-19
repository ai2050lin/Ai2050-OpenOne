"""Explicit tensor audit and 16-prefix measured cost gate, before any new inference."""
from rdc_update_common import *
from rdc_update_graph import kernels

def main():
    import torch
    from rdc_update_graph import fit_head,pack,save_pack
    out=BASE/'graph';start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if (out/'pilot.json').exists():return
    rng=np.random.default_rng(2736);n,t,d=3,5,4
    h=torch.tensor(rng.normal(size=(n,t,d)),device='cuda',dtype=torch.float32);a=torch.tensor(rng.uniform(size=(n,t,t)),device='cuda',dtype=torch.float32);a/=a.sum(-1,keepdim=True)
    q=torch.tensor(rng.normal(size=(n,d)),device='cuda',dtype=torch.float32);p=torch.tensor(rng.normal(size=(n,t,3)),device='cuda',dtype=torch.float32)
    toy={'h':h,'unit':h,'a':a,'shuffled':a.flip(-1),'pos':p,'q':q,'e':q,'length':torch.full((n,),float(t),device='cuda'),
      'mean':h.mean(1),'signed':(h.transpose(1,2)@p/t).flatten(1),
      'message':(h.transpose(1,2)@a@(h@q[:,:,None])).squeeze(-1)/t/d,
      'reverse_message':(h.transpose(1,2)@a.transpose(1,2)@(h@q[:,:,None])).squeeze(-1)/t/d}
    kk=kernels(toy,block=2,verbose=False);base=(q@q.T)/d;errors={}
    for name,aa in [('directed_raw',a),('directed_rms',a),('shuffled_heads_raw',a.flip(-1))]:
        T=(h.transpose(1,2)@aa@h/t/d).flatten(1);s=T@T.T;expected=1+base+s+base*s
        errors[name]=float((kk[name]-expected).abs().max());assert errors[name]<2e-5
    # A global transpose is an invertible representation change: its Frobenius
    # kernel is identical, so it is NOT a meaningful negative control by itself.
    T=h.transpose(1,2)@a@h;rev=h.transpose(1,2)@a.transpose(1,2)@h
    transpose_error=float((T.flatten(1)@T.flatten(1).T-rev.flatten(1)@rev.flatten(1).T).abs().max())
    assert transpose_error<1e-4
    rows=gzread(PRIOR/'natural_discovery.json.gz');coef=fit_head(rows,out)
    pilotrows=[r for cohort in ('gum','ewt') for r in rows if r['cohort']==cohort and r['split']=='train']
    pilotrows=pilotrows[:8]+pilotrows[160:168];pk,_,audit=pack(pilotrows,coef);tick=time.monotonic();torch.cuda.synchronize()
    km=kernels(pk,verbose=False);torch.cuda.synchronize();seconds=time.monotonic()-tick
    estimate=seconds*(512/16)**2
    save_pack(out/'pilot',pilotrows,pk,audit)
    result={'timestamp':stamp(),'source':snapshot(__file__),'explicit_tensor_errors':errors,'global_edge_transpose_kernel_error':transpose_error,
      'kernel_16_seconds':seconds,'estimated_512_kernel_seconds':estimate,'pilot_rows':len(pilotrows),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
      'passed':estimate<3600,'scope':'Full coordinates and all soft head edges. Exact tensor contraction is not arbitrary-rank compression. Finite directed moments are not asserted sufficient states.',
      'seconds':time.monotonic()-start}
    save(out/'pilot.json',result);ledger('head_fit_and_directed_math_pilot',result['seconds']);assert result['passed'];print('DIRECTED_PILOT',result,flush=True)

if __name__=='__main__':main()
