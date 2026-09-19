"""Exact current-loss full-parameter gradient factors and explicit finite spans."""
from rdc_binding_common import *
from rdc_law_native import factor_gram,dense_gradient,parameter_norm

def program_arrays(rows):
    xx=[];rr=[];targets=[]
    for row in rows:
        with np.load(BASE/'capture/program'/f'{row["sample_id"]}.npz') as z:
            xx.append(unbits(z['last_x'])[-1]);rr.append(unbits(z['last_residual'])[-1])
        targets.append(row['target_ids'][0])
    return np.array(xx),np.array(rr),np.array(targets)

def collect(tail,x,r,targets,batch=16):
    import torch
    result={k:[] for k in ('x','a','s','bg','bu')};stats={k:[] for k in ('loss','argmax','entropy')}
    with torch.no_grad():
      for i in range(0,len(x),batch):
        z=tail.forward(x[i:i+batch],r[i:i+batch],targets[i:i+batch],True)
        for k in result:result[k].append(z['factors'][k].double())
        for k in stats:stats[k].append(z[k].cpu().numpy())
    return {k:torch.cat(v) for k,v in result.items()},{k:np.concatenate(v) for k,v in stats.items()}

def combination(f,coeff):
    # Sum c_i grad_i, without forming N dense parameter copies.
    return {'g':(f['bg']*coeff[:,None]).T@f['x'],
      'u':(f['bu']*coeff[:,None]).T@f['x'],
      'd':(f['s']*coeff[:,None]).T@f['a']}

def normalized(direction):
    norm=parameter_norm(direction)
    return {k:v/norm.clamp_min(1e-30) for k,v in direction.items()},float(norm)

def projection_coefficients(gram,cross,relative_tolerance=1e-9):
    import torch
    eig,vec=torch.linalg.eigh((gram+gram.T)*.5)
    keep=eig>eig.max()*relative_tolerance
    coefficients=vec[:,keep]@((vec[:,keep].T@cross)/eig[keep])
    return coefficients,{'span_samples':len(gram),'numerical_rank':int(keep.sum()),'relative_numerical_tolerance':relative_tolerance,
      'smallest_eigenvalue':float(eig.min()),'largest_eigenvalue':float(eig.max()),
      'scope':'Complete declared sample-gradient span; only numerical zero eigenvalues discarded, no energy/Top-K rank selection.'}

def error_factors(tail,z,x,error):
    """Backpropagate arbitrary d(loss)/d(logits) through all native last-MLP weights."""
    r=z['r'];den=(r.square().mean(-1,keepdim=True)+tail.eps).sqrt()
    v=(error@tail.head)*tail.norm
    s=v/den-r*(v*r).mean(-1,keepdim=True)/den.pow(3)
    unit=s@tail.w['d'];g=z['g'];sig=g.sigmoid();prime=sig+g*sig*(1-sig)
    return {'x':x,'a':z['activation'],'s':s,'bg':unit*z['u']*prime,'bu':unit*z['phi']}

