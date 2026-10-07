"""Post-primary CPU diagnostic: quantify truncation vs local numerical reconstruction.
This uses observed upstream g/u for both conditions. It is NOT text-only prediction.
"""
import argparse,json
from datetime import datetime,timezone
import numpy as np
import torch
from phase2751_trusted_rebuild import OUT,write,load_bank,sha


def run(side):
    folder=OUT/side
    # Independent directional finite difference of the full two-input product.
    gen=torch.Generator().manual_seed(2751019)
    gs,us,dgs,dus=[torch.randn(32,dtype=torch.float64,generator=gen) for _ in range(4)]
    ss=torch.sigmoid(gs);fp=ss+gs*ss*(1-ss);fpp=ss*(1-ss)*(2+gs*(1-2*ss))
    expected=us*fpp*dgs*dgs+2*fp*dgs*dus
    f=lambda a,b:torch.nn.functional.silu(a)*b
    ee=1e-4
    fd=(f(gs+ee*dgs,us+ee*dus)+f(gs-ee*dgs,us-ee*dus)-2*f(gs,us))/(ee*ee)
    check_error=float((fd-expected).abs().max())
    assert check_error<1e-6,check_error
    design=folder/'second_order_design.json'
    if not design.exists():write(design,dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='post-primary diagnostic, no fitting; not independent discovery confirmation',
        formula='da2=u*phi_prime(g)*dg+phi(g)*du+phi_prime(g)*dg*du+0.5*u*phi_second(g)*dg^2',
        inputs='Observed gate/up preactivations for both conditions and true local W_down.',
        endpoint='Relative error vs native delta m; compare first, second, exact nonlinear local reconstruction.'))
    data,rows=load_bank(folder);groups={}
    for i,r in enumerate(rows):groups.setdefault(r['group'],{})[r['cond']]=i
    bs=[];cs=[]
    for group,idx in groups.items():
        if 0 not in idx:continue
        for cond,i in idx.items():
            if cond:bs.append(idx[0]);cs.append(i)
    torch.set_num_threads(4)
    t=lambda x:torch.from_numpy(np.asarray(x,dtype=np.float64))
    g,u=t(data['g'][bs]),t(data['u'][bs]);dg=t(data['g'][cs])-g;du=t(data['u'][cs])-u
    s=torch.sigmoid(g);phi=g*s;first=s+g*s*(1-s);second=s*(1-s)*(2+g*(1-2*s))
    da1=u*first*dg+phi*du
    da2=da1+first*dg*du+.5*u*second*dg*dg
    exact=torch.nn.functional.silu(g+dg)*(u+du)-phi*u
    with np.load(folder/'lastblock_weights.npz') as z:wd=t(z['Wd'])
    truth=t(data['m'][cs])-t(data['m'][bs]);den=torch.linalg.vector_norm(truth,dim=1)
    errors={}
    for key,da in [('first',da1),('second',da2),('exact_nonlinear',exact)]:
        pred=da@wd.T
        errors[key]=(torch.linalg.vector_norm(pred-truth,dim=1)/den).numpy()
    output=[]
    for n,i in enumerate(cs):output.append(dict(id=rows[i]['id'],group=rows[i]['group'],split=rows[i]['split'],
        **{k:float(v[n]) for k,v in errors.items()}))
    summary={split:{k:float(np.median([r[k] for r in output if r['split']==split])) for k in errors} for split in sorted(set(r['split'] for r in output))}
    write(folder/'second_order.json',dict(summary=summary,rows=output,
        second_derivative_finite_difference_max_abs=check_error,
        interpretation='Taylor terms are known calculus, not new language theory. Exact nonlinear endpoint is a reconstruction/numerical reference, not mechanism extraction.',
        input_weights_sha256=sha(folder/'lastblock_weights.npz')))
    print(json.dumps(dict(side=side,summary=summary)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--side',default='4B',choices=['4B','14B']);run(p.parse_args().side)
