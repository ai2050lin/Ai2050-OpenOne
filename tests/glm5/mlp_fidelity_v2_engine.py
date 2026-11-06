"""Fixed-variance sparse dictionaries. New method; archived transcoder.v1 unchanged."""
import math
import numpy as np
import torch
from shared.dist_protocol.transcoder import TopKTranscoder

def indices(docs,seed):
    from shared.dist_protocol.mechanism_trace import digest
    sources=sorted({d['source'] for d in docs});train=[];val=[]
    for source in sources:
        rows=sorted([i for i,d in enumerate(docs) if d['source']==source],key=lambda i:digest({'seed':seed,'id':docs[i]['doc_id']}));assert len(rows)==512
        train+=rows[:384];val+=rows[384:]
    return np.asarray(sorted(train)),np.asarray(sorted(val))

@torch.inference_mode()
def evaluate(t,x,y):
    sse=0.;mean=y.mean(0);den=float(((y.double()-mean.double())**2).sum())
    for lo in range(0,len(x),256):
        pred=t(x[lo:lo+256]);sse+=float(((pred.double()-y[lo:lo+256].double())**2).sum())
    return sse/max(den,1e-30)

def train(x,y,vx,vy,*,m,k,steps,batch,lr,lr_end,seed,checkpoints,log=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);t=TopKTranscoder(x.shape[1],y.shape[1],m,k,device=x.device,seed=seed)
    with torch.no_grad():
        t.x_scale=float(x.norm(dim=-1).mean()/math.sqrt(x.shape[1]));t.y_scale=float(y.norm(dim=-1).mean()/math.sqrt(y.shape[1]));t.b_pre.copy_((x/t.x_scale).mean(0));t.b_dec.copy_((y/t.y_scale).mean(0));var=float(((y/t.y_scale-(y/t.y_scale).mean(0))**2).sum(-1).mean())
    assert var>0
    for p in t.params():p.requires_grad_(True)
    optimizer=torch.optim.Adam(t.params(),lr=lr,betas=(.9,.999),fused=x.is_cuda);since=torch.zeros(m,device=x.device);curve=[]
    for step in range(1,steps+1):
        idx=torch.from_numpy(rng.integers(len(x),size=batch)).to(x.device);xx=x[idx];yy=y[idx]/t.y_scale
        rate=(lr_end+.5*(lr-lr_end)*(1+math.cos(math.pi*(step-1)/max(1,steps-1))))*min(1.,step/100)
        for pg in optimizer.param_groups:pg['lr']=rate
        with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
            z=torch.relu(t.pre_acts(xx));values,ix=z.topk(k,dim=-1);a=torch.zeros_like(z).scatter(-1,ix,values);pred=a@t.W_dec+t.b_dec
        err=yy-pred.float();loss=err.square().sum(-1).mean()/var;dead=since>=16384
        if dead.any():
            with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
                dz=z.masked_fill(~dead,0);vv,ii=dz.topk(min(128,int(dead.sum())),dim=-1);resid=torch.zeros_like(dz).scatter(-1,ii,vv)@t.W_dec
            loss+=((err.detach()-resid.float())**2).sum(-1).mean()/var/32
        optimizer.zero_grad(set_to_none=True);loss.backward()
        with torch.no_grad():t.W_dec.grad-=(t.W_dec.grad*t.W_dec).sum(-1,keepdim=True)*t.W_dec
        optimizer.step()
        with torch.no_grad():
            t.W_dec/=t.W_dec.norm(dim=1,keepdim=True).clamp_min(1e-8);fired=torch.zeros(m,dtype=torch.bool,device=x.device);fired[ix.reshape(-1)]=True;since=torch.where(fired,0,since+batch)
        if step in checkpoints:
            assert all(torch.isfinite(p).all() for p in t.params());r={'step':step,'training_batch_FVU':float(err.square().sum(-1).mean()/var),'development_validation_FVU':evaluate(t,vx,vy),'zero_recent_features':int((since>=16384).sum()),'lr':rate};curve.append(r)
            if log:log(r)
    for p in t.params():p.requires_grad_(False)
    return t,curve
