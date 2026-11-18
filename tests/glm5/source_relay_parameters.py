"""Native K-score source ledger: effective residual increments and fixed-scale algebra."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay_design as D
from source_relay_fit import load,close
from source_relay_behavior import Runtime
from source_binding_native_v2 import unbits
from source_formation_analysis import rotary

def analyze(split):
    import torch
    D.verify_internal();D.admission(split);root=D.OUT/D.MODEL;rows=D.read(D.OUT/'qwen14-materials.json')['splits'][split]
    manifest=D.read(root/f'{split}-native-manifest.json');assert manifest['replay']['passed'];n=len(rows);L=manifest['length'];timer=time.monotonic();rt=Runtime(D.MODEL)
    st=load(split,'trace_state');post=load(split,'trace_postattn');kp=load(split,'kpre');qp=load(split,'qpre');saved={};results=[]
    for bi,c in enumerate(D.HEADS):
        b=c['block'];wk=rt.weight(f'model.layers.{b}.self_attn.k_proj.weight','cpu')[c['kv_head']*128:(c['kv_head']+1)*128].float().numpy().copy()
        gamma=rt.weight(f'model.layers.{b}.input_layernorm.weight','cpu').float().numpy()
        kg=rt.weight(f'model.layers.{b}.self_attn.k_norm.weight','cpu').float().numpy()
        saved[f'B{b}_K_weight']=wk;saved[f'B{b}_input_norm_weight']=gamma;saved[f'B{b}_key_norm_weight']=kg
        labels=['hs20']+[f'B{bb}_{part}' for bb in range(20,b) for part in ['attention_effective','MLP_effective']]+['rounding_residual']
        scores=np.zeros((n,5,len(labels)),np.float64);actualscores=np.zeros((n,5),np.float64);keyerrors=[];coord_mean=np.zeros((5,rt.dim),np.float64);coord_abs=np.zeros_like(coord_mean)
        for lo in range(0,n,32):
            hi=min(lo+32,n);rs=rows[lo:hi];h=unbits(st[lo:hi]);p=unbits(post[lo:hi]);hb=h[:,b-20]
            rh=np.sqrt(np.mean(hb.astype(np.float64)**2,axis=-1)+rt.cfg.rms_norm_eps)
            norm=rt.Norm(rt.dim,eps=rt.cfg.rms_norm_eps).to(dtype=torch.bfloat16);norm.weight=torch.nn.Parameter(torch.tensor(gamma).bfloat16(),requires_grad=False)
            with torch.inference_mode():u=torch.nn.functional.linear(norm(torch.tensor(hb).bfloat16()),torch.tensor(wk).bfloat16()).float().numpy()
            ru=np.sqrt(np.mean(u.astype(np.float64)**2,axis=-1)+rt.cfg.rms_norm_eps)
            denom=rh*ru;effective=kg[:,None]*wk*gamma[None]
            klinear=np.einsum('nrd,hd->nrh',hb,effective,optimize=True)/denom[:,:,None]
            actual=np.array([unbits(kp[lo+i,bi,np.array(r['trace_positions'])+L-len(r['tokens'])]) for i,r in enumerate(rs)])
            keyerrors.extend((np.linalg.norm(klinear-actual,axis=-1)/np.maximum(np.linalg.norm(actual,axis=-1),1e-9)).tolist())
            cos,sin=rotary(rt,rs,L);q=unbits(qp[lo:hi,bi])
            # Rotate query at the real final position. Compute the transpose key rotation explicitly.
            qt=torch.tensor(q).bfloat16()[:,None,None]
            qrot,_=rt.mod.apply_rotary_pos_emb(qt,qt,cos[:,-1:],sin[:,-1:]);qrot=qrot[:,0,0].float().numpy()
            cq=np.stack([cos[i,np.array(r['trace_positions'])+L-len(r['tokens'])].float().numpy() for i,r in enumerate(rs)])
            sq=np.stack([sin[i,np.array(r['trace_positions'])+L-len(r['tokens'])].float().numpy() for i,r in enumerate(rs)])
            qh=np.concatenate([-qrot[:,64:],qrot[:,:64]],axis=-1)
            read=qrot[:,None]*cq-qh[:,None]*sq
            gradient=np.einsum('nrh,hd->nrd',read,effective,optimize=True)/(denom[:,:,None]*np.sqrt(128))
            sources=[h[:,0]]
            for bj in range(b-20):sources.extend([p[:,bj]-h[:,bj],h[:,bj+1]-p[:,bj]])
            for j,value in enumerate(sources):scores[lo:hi,:,j]=np.sum(gradient*value,axis=-1)
            # Original BF16 RoPE of the actual stored K gives the actual per-position score.
            kt=torch.tensor(actual).bfloat16();ct=torch.tensor(cq).bfloat16();snt=torch.tensor(sq).bfloat16()
            half=torch.cat([-kt[...,64:],kt[...,:64]],dim=-1);kr=(kt*ct+half*snt).float().numpy()
            actualscores[lo:hi]=np.sum(qrot[:,None]*kr,axis=-1)/np.sqrt(128)
            scores[lo:hi,:,-1]=actualscores[lo:hi]-scores[lo:hi,:,:-1].sum(-1)
            coord=gradient*hb;coord_mean+=coord.sum(0);coord_abs+=np.abs(coord).sum(0)
            del h,p,hb,norm,u,actual,klinear,gradient,sources;gc.collect();D.log(split+f'/parameter-B{b}',hi,n,timer)
        saved[f'B{b}_score_components']=scores;saved[f'B{b}_actual_scores']=actualscores
        saved[f'B{b}_coordinate_signed_mean']=coord_mean/n;saved[f'B{b}_coordinate_absolute_mean']=coord_abs/n
        reconstruction=float(np.max(np.abs(scores.sum(-1)-actualscores)))
        # Read-before-query keys must not depend on which later query was asked.
        groups={}
        for i,r in enumerate(rows):groups.setdefault((r['family'],r['order']),[]).append(i)
        comparisons=0;unequal=0
        for ii in groups.values():
            anchor=ii[0];ar=rows[anchor]
            for j in range(6):
                ix=ar['records'][str(j)]['value_span'][-1]+L-len(ar['tokens']);reference=np.array(kp[anchor,bi,ix])
                for i in ii[1:]:
                    r=rows[i];ix=r['records'][str(j)]['value_span'][-1]+L-len(r['tokens']);unequal+=int(not np.array_equal(reference,kp[i,bi,ix]));comparisons+=1
        results.append({**c,'labels':labels,'roles':rows[0]['trace_roles'],'score_components_mean':scores.mean(0).tolist(),
            'score_components_mean_abs':np.abs(scores).mean(0).tolist(),'key_linearization_max_relative_error':float(np.max(keyerrors)),
            'key_linearization_mean_relative_error':float(np.mean(keyerrors)),
            'explicit_rounding_residual_score_mean_abs':float(np.abs(scores[:,:,-1]).mean()),'with_residual_reconstruction_max_error':reconstruction,
            'prefix_invariance':{'comparisons':comparisons,'nonidentical':unequal,'passed':unequal==0},
            'scalar_parameter_rows':f'model.layers.{b}.self_attn.k_proj.weight[{c["kv_head"]*128}:{(c["kv_head"]+1)*128},:]',
            'coordinate_count':rt.dim,'no_coordinate_sort_or_pruning':True})
    close(st,post,kp,qp);path=root/f'{split}-parameter-ledger.npz';assert not path.exists();np.savez(path,**saved)
    D.save(f'{D.MODEL}/{split}-parameter-ledger.json',{'heads':results,'arrays_sha256':D.sha(path),'elapsed_seconds':time.monotonic()-timer,
        'scope':'Algebra at true late-layer state, Q, and normalization scales. Native effective attention/MLP increments include BF16 rounding. Explicit rounding residual closes the ledger by definition; this is neither independent prediction nor causal necessity. Query-dependent read gradients do not imply pre-query K can see a future question.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',required=True);a=ap.parse_args();analyze(a.split)
