"""Post-confirmation descriptive residual accounting; no new blind claim.

Replay every original item with frozen targets/deltas, not selected successes.
The identity includes actual BF16 rounding in every recorded residual state.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import common_bench_b1_repair as R
from common_bench_b1_repair_run import Runner

def main():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    assert not (R.OUT/'path-account.json').exists()
    R.checked();R.check_write(R.OUT,1<<30,operation='B1原生路径复算')
    R.save('path-account-plan.json',{'scope':'post-confirmation descriptive decomposition on behavior-eligible lockbox only, no independent new confirmation',
        'excluded_panel':'lockbox_frame failed its complete behavior gate; no additional internal replay on that panel',
        'code_sha256':R.sha(__file__),'arms':['pooled','cross_language','cross_source','unembed'],
        'identity':'delta_final_raw = actual_BF16_delta_at_hs32 + sum over blocks32..35 of (delta_attention_write + delta_MLP_write)',
        'normalization':'locally exact secant readout: integrate final RMSNorm change along base-to-edited segment numerically using 64 midpoint steps, then compare direct vs downstream continuous margin contributions',
        'readout':'fixed original/target token indices chosen by baseline scores within frozen accepted variants; avoids arm-dependent max switching',
        'warning':'Secant attribution is a fixed-path attribution, not a causal percentage or unique route'},True)
    summaries={};records=[];start=time.monotonic()
    with GpuLease(),torch.inference_mode():
        rr=Runner('path-account');bits=lambda x:x.contiguous().view(torch.uint16).cpu().numpy()
        for split in ['lockbox']:
            assert R.read(R.OUT/f'{split}-gate.json')['passed']
            rows=R.read(R.OLD/f'{split}-materials.json');targets=R.read(R.OUT/f'{split}-started.json')['targets']
            folder=R.OUT/f'{split}-path-account';folder.mkdir()
            # per item x base/four arms x five boundaries (hs32..36raw) x 2560.
            hfile=np.lib.format.open_memmap(folder/'raw.npy',mode='w+',dtype=np.uint16,shape=(len(rows),5,5,2560))
            pfile=np.lib.format.open_memmap(folder/'postattn.npy',mode='w+',dtype=np.uint16,shape=(len(rows),5,4,2560))
            for lo in range(0,len(rows),8):
                rs=rows[lo:lo+8];batch=rr.batch(rs);base,_,_=rr.run(batch,capture=True)
                baseh=torch.stack([rr.grab['raw',b] for b in range(31,36)],1).clone()
                basep=torch.stack([rr.grab['post',b] for b in range(32,36)],1).clone()
                hfile[lo:lo+8,0]=bits(baseh);pfile[lo:lo+8,0]=bits(basep)
                ds,_=rr.vectors(rs,targets,baseh[:,0]);goldids=[];targetids=[]
                for i,r in enumerate(rs):
                    lang=0 if r['language']==1 else 5;k=R.B.COLORS.index(r['gold_color']);kt=targets[r['id']]
                    goldids.append(rr.ids[lang+k][int(base[i,rr.ids[lang+k]].argmax())]);targetids.append(rr.ids[lang+kt][int(base[i,rr.ids[lang+kt]].argmax())])
                u=rr.m.lm_head.weight[targetids].float()-rr.m.lm_head.weight[goldids].float()
                for ai,arm in enumerate(['pooled','cross_language','cross_source','unembed'],1):
                    lg,_,_=rr.run(batch,mode='add',block=31,delta=ds[arm],capture=True)
                    edited=torch.stack([rr.grab['raw',b] for b in range(31,36)],1).clone()
                    edited[:,0]=(baseh[:,0].float()+ds[arm]).to(baseh.dtype)
                    post=torch.stack([rr.grab['post',b] for b in range(32,36)],1)
                    hfile[lo:lo+8,ai]=bits(edited);pfile[lo:lo+8,ai]=bits(post)
                    delta=edited[:,0].float()-baseh[:,0].float()
                    da=(post.float()-edited[:,:-1].float())-(basep.float()-baseh[:,:-1].float())
                    dm=(edited[:,1:].float()-post.float())-(baseh[:,1:].float()-basep.float())
                    finaldiff=edited[:,-1].float()-baseh[:,-1].float()
                    err=(finaldiff-delta-da.sum(1)-dm.sum(1)).abs().max(1).values
                    # Continuous RMSNorm secant gradient; BF16 output rounding left as explicit residual.
                    gamma=rr.m.model.norm.weight.float();eps=rr.m.model.norm.variance_epsilon
                    grad=torch.zeros_like(delta)
                    for j in range(64):
                        h=baseh[:,-1].float()+(j+.5)/64*finaldiff;rms=(h.square().mean(1,keepdim=True)+eps).sqrt()
                        v=u*gamma
                        grad+=(v/rms-h*(v*h).mean(1,keepdim=True)/rms.pow(3))/64
                    direct=(grad*delta).sum(1);att=(grad[:,None]*da).sum(-1);mlp=(grad[:,None]*dm).sum(-1)
                    index=torch.arange(len(rs),device='cuda')
                    actual=(lg[index,targetids]-lg[index,goldids])-(base[index,targetids]-base[index,goldids])
                    for i,r in enumerate(rs):
                        records.append({'id':r['id'],'split':split,'arm':arm,'source':r['source'],'language':r['language'],
                            'identity_max_abs_error':float(err[i]),'actual_norm_delta':float(delta[i].norm()),
                            'final_norm_delta':float(finaldiff[i].norm()),'direct_margin_attribution':float(direct[i]),
                            'attention_margin_attribution':att[i].cpu().tolist(),'MLP_margin_attribution':mlp[i].cpu().tolist(),
                            'actual_fixed_token_margin_change':float(actual[i]),
                            'secant_and_BF16_residual':float(actual[i]-direct[i]-att[i].sum()-mlp[i].sum()),
                            'gold_token':goldids[i],'target_token':targetids[i]})
                if lo%128==0 or lo+8==len(rows):
                    hfile.flush();pfile.flush();rr.deadline();R.log('path-account-'+split,lo+8,len(rows),rr.start)
            hfile.flush();pfile.flush();del hfile,pfile
            summaries[split]={'files':{p.name:{'sha256':R.sha(p),'bytes':p.stat().st_size} for p in folder.glob('*.npy')}}
    R.save('path-account.json',{'rows':records,'native':summaries,'elapsed_seconds':time.monotonic()-start,'scope':'post-confirmation descriptive replay, fixed targets and arms'},True)

if __name__=='__main__':main()
