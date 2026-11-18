"""Fit equal-input QKV forecasts, then optimize conditional routing on discovery only."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay_design as D
import source_relay_math as M
from source_binding_native_v2 import unbits
from source_relay_behavior import Runtime
from source_formation_analysis import rotary

KINDS=['native','random','pca','embedding']

def load(split,name):return np.load(D.OUT/D.MODEL/f'{split}-native/{name}.npy',mmap_mode='r')
def close(*aa):
    for a in aa:
        if hasattr(a,'_mmap') and a._mmap is not None:a._mmap.close()

def fit():
    import torch
    D.verify_internal();D.admission('discovery');root=D.OUT/D.MODEL
    assert not (root/'fit.npz').exists()
    rows=D.read(D.OUT/'qwen14-materials.json')['splits']['discovery'];manifest=D.read(root/'discovery-native-manifest.json');assert manifest['replay']['passed']
    n=len(rows);L=manifest['length'];timer=time.monotonic();D.log('fit/prepare',0,8,timer)
    h=load('discovery','early');q=load('discovery','qpre');k=load('discovery','kpre');v=load('discovery','vproj')
    xq=unbits(h[:,-1]);yq=unbits(q).reshape(n,-1);xk=[];yk=[];yv=[];ids=[]
    for i,r in enumerate(rows):
        ps=np.unique(np.linspace(0,len(r['tokens'])-1,min(24,len(r['tokens'])),dtype=int));ix=ps+L-len(r['tokens'])
        xk.append(unbits(h[i,ix]));yk.append(unbits(k[i,:,ix]).reshape(len(ix),-1));yv.append(unbits(v[i,:,ix]).reshape(len(ix),-1));ids.extend(r['tokens'][p] for p in ps)
    xk=np.concatenate(xk);yk=np.concatenate(yk);yv=np.concatenate(yv)
    # NumPy advanced indices place the sampled token dimension first; check explicit slices.
    first=np.unique(np.linspace(0,len(rows[0]['tokens'])-1,min(24,len(rows[0]['tokens'])),dtype=int))[0]+L-len(rows[0]['tokens'])
    assert np.array_equal(yk[0],unbits(k[0,:,first]).reshape(-1))
    close(h,q,k,v);rt=Runtime(D.MODEL)
    emb=rt.weight('model.embed_tokens.weight','cpu');eq=emb[torch.tensor([r['tokens'][-1] for r in rows])].float().numpy();ek=emb[torch.tensor(ids)].float().numpy();del emb;gc.collect()
    store={};feature_shapes={}
    for j,kind in enumerate(KINDS):
        qx,kx=(eq,ek) if kind=='embedding' else (xq,xk)
        qp,qz=M.transform_fit(qx,yq,'native' if kind=='embedding' else kind,'q/'+kind)
        kp,kz=M.transform_fit(kx,np.concatenate([yk,yv],axis=1),'native' if kind=='embedding' else kind,'k/'+kind)
        M.pack(store,'q_transform_'+kind,qp);M.pack(store,'k_transform_'+kind,kp)
        store['q_'+kind]=M.linear_fit(qz,yq);store['k_'+kind]=M.linear_fit(kz,yk);store['v_'+kind]=M.linear_fit(kz,yv)
        feature_shapes[kind]={'input_width':128,'qkv_coefficients':sum(store[a+'_'+kind].size for a in ['q','k','v']),
            'transform_floats':sum(a.size for p in [qp,kp] for a in p.values())}
        D.log('fit/ridge',j+1,8,timer,method=kind)
    del xq,xk,yq,yk,yv,eq,ek;gc.collect()
    ctx=load('discovery','head_context');actual=unbits(ctx);close(ctx)
    for bi,c in enumerate(D.HEADS):
        w=rt.weight(f'model.layers.{c["block"]}.self_attn.o_proj.weight','cpu');wo=w[:,c['head']*rt.hd:(c['head']+1)*rt.hd].float().numpy().copy();del w
        store[f'wo_{bi}']=wo;store[f'G_{bi}']=wo.T@wo;store[f'mean_context_{bi}']=actual[:,bi].mean(0)
    cos,sin=rotary(rt,rows,L);cos=cos.float();sin=sin.float();valid=np.arange(L)[None]>=L-np.array([len(r['tokens']) for r in rows])[:,None]
    p=load('discovery','attention');target_p=np.array(p);close(p);v=load('discovery','vproj');values=unbits(v);close(v)
    losses={}
    with D.gpu_lease('query-aware-fit'):
        torch.set_num_threads(4);torch.manual_seed(int(D.SEED%(2**31-1)))
        for ki,kind in enumerate(['native','pca']):
            h=load('discovery','early');qp={k[len('q_transform_'+kind)+1:]:v for k,v in store.items() if k.startswith('q_transform_'+kind+'_')};kp={k[len('k_transform_'+kind)+1:]:v for k,v in store.items() if k.startswith('k_transform_'+kind+'_')}
            qz=[];kz=[]
            for lo in range(0,n,32):
                x=unbits(h[lo:lo+32]);qz.append(M.transform(x[:,-1],qp));kz.append(M.transform(x,kp));del x
            close(h);qz=torch.tensor(np.concatenate(qz),device='cuda');kz=torch.tensor(np.concatenate(kz),device='cuda')
            cq=torch.nn.Parameter(torch.tensor(store['q_'+kind],device='cuda'));ck=torch.nn.Parameter(torch.tensor(store['k_'+kind],device='cuda'))
            optimizer=torch.optim.Adam([cq,ck],lr=.003)
            pt=torch.tensor(target_p,device='cuda');vt=torch.tensor(values,device='cuda');at=torch.tensor(actual,device='cuda')
            ct=cos.to('cuda');st=sin.to('cuda');validt=torch.tensor(valid,device='cuda');gt=torch.tensor(np.stack([store[f'G_{bi}'] for bi in range(2)]),device='cuda')
            denom=torch.tensor([M.error_rows(actual[:,bi],store[f'mean_context_{bi}'],store[f'G_{bi}']).mean() for bi in range(2)],device='cuda').clamp_min(1e-8)
            history=[];random=D.rng('optimizer/'+kind)
            for step in range(400):
                ii=torch.tensor(random.choice(n,32,replace=False),device='cuda');pq=cq[0]+qz[ii]@cq[1:];pk=ck[0]+kz[ii]@ck[1:]
                objective=0;klvalue=0;writevalue=0
                for bi in range(2):
                    prob,_=M.distribution_torch(pq[:,bi*128:(bi+1)*128],pk[:,:,bi*128:(bi+1)*128],ct[ii],st[ii],validt[ii])
                    true=pt[ii,bi];kl=(true*(true.clamp_min(1e-12).log()-prob.clamp_min(1e-12).log())).sum(-1).mean()
                    pred=torch.einsum('nt,ntd->nd',prob,vt[ii,bi]);delta=pred-at[ii,bi]
                    err=torch.einsum('ni,ij,nj->n',delta,gt[bi],delta).mean()/denom[bi]
                    objective=objective+kl+.1*err;klvalue+=float(kl.detach());writevalue+=float(err.detach())
                optimizer.zero_grad();objective.backward();optimizer.step()
                assert torch.isfinite(objective)
                if step%25==0 or step==399:
                    history.append({'step':step+1,'loss':float(objective.detach()),'KL_two_heads_sum':klvalue,'normalized_write_two_heads_sum':writevalue})
                    D.log('fit/query-aware/'+kind,step+1,400,timer,loss=float(objective.detach()))
            store['q_aware_'+kind]=cq.detach().cpu().numpy();store['k_aware_'+kind]=ck.detach().cpu().numpy();losses[kind]=history
            del qz,kz,cq,ck,optimizer,pt,vt,at,ct,st,validt,gt,denom,objective,pred,delta,pq,pk,prob,ii,err,kl,true
            gc.collect();torch.cuda.empty_cache()
    path=root/'fit.npz';np.savez(path,**store)
    D.save(f'{D.MODEL}/fit.json',{'fit_sha256':D.sha(path),'internal_freeze_sha256':D.sha(D.OUT/'internal-freeze.json'),
        'discovery_manifest_sha256':D.sha(root/'discovery-native-manifest.json'),'heads':D.HEADS,'feature_budgets':feature_shapes,
        'optimization':losses,'steps':400,'lr':.003,'batch':32,'inference':'hs20 prefix only for learned Q/K/V; true future V labelled separately',
        'rounding':'BF16 Q/K and RoPE forward boundaries with straight-through gradients; actual original model is never trained.',
        'elapsed_seconds':time.monotonic()-timer})

if __name__=='__main__':argparse.ArgumentParser().parse_args();fit()
