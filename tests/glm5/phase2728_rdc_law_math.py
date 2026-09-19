"""CPU synthetic tests of the production full-factor training and output derivatives."""
from rdc_law_common import *
from rdc_law_native import tail_forward, factor_gram, dense_gradient, readout_actions


def main():
    import torch
    rng=torch.Generator().manual_seed(2728)
    def normal(*shape):return torch.randn(shape,generator=rng,dtype=torch.float64)
    x=normal(9,7);residual=normal(9,7)
    w={'g':normal(11,7)*.2,'u':normal(11,7)*.2,'d':normal(7,11)*.2}
    norm=normal(7);head=normal(13,7);eps=1e-6;targets=torch.arange(9)%13
    output=tail_forward(x,residual,w,norm,head,eps,targets,True)
    explicit=[];maxima=[]
    for i in range(9):
        ww={k:v.clone().requires_grad_(True) for k,v in w.items()}
        z=tail_forward(x[i:i+1],residual[i:i+1],ww,norm,head,eps,targets[i:i+1])
        z['loss'].sum().backward()
        f={k:v[i:i+1] for k,v in output['factors'].items()}
        dg=dense_gradient(f)
        maxima.extend(float((dg[k]-ww[k].grad).abs().max()) for k in ww)
        explicit.append(torch.cat([ww[k].grad.flatten() for k in ('g','u','d')]))
    explicit=torch.stack(explicit)
    gram=factor_gram(output['factors'])
    gram_error=float((gram['total']-explicit@explicit.T).abs().max())
    assert max(maxima)<1e-11 and gram_error<1e-10
    # Production direction action is the derivative of the actual norm+head map.
    direction=normal(9,7)
    dn,var=readout_actions(output['r'],direction,norm,head,eps,output['probabilities'])
    r=output['r'].clone().requires_grad_(True)
    def normalize(rr):return norm*rr/(rr.square().mean(-1,keepdim=True)+eps).sqrt()
    actual=torch.autograd.functional.jvp(normalize,r,direction)[1]
    norm_error=float((dn-actual).abs().max());assert norm_error<1e-11
    dense_errors=[]
    for i in range(9):
        J=torch.autograd.functional.jacobian(lambda rr:normalize(rr)[i],r)[...,i,:]
        p=output['probabilities'][i]
        G=head.T@(torch.diag(p)-p[:,None]*p[None,:])@head
        expected=direction[i]@J.T@G@J@direction[i]
        dense_errors.append(float(abs(expected-var[i])))
    assert max(dense_errors)<1e-9
    # Small SGD update, full dense parameters: derivative predicts actual held-out losses.
    dg=dense_gradient({k:v[:1] for k,v in output['factors'].items()})
    step_checks=[]
    for eta in (1e-6,1e-5,1e-4):
        modified={k:v-eta*dg[k] for k,v in w.items()}
        after=tail_forward(x,residual,modified,norm,head,eps,targets)
        actual_delta=after['loss']-output['loss']
        predicted=-eta*gram['total'][:,0]
        error=float((actual_delta-predicted).abs().max())
        step_checks.append({'eta':eta,'max_abs_nonlinear_remainder':error})
        assert error<eta**2*1e5
    report={'timestamp':stamp(),'source':snapshot(Path(__file__)),'production_source':snapshot(ROOT/'tests/glm5/rdc_law_native.py'),
        'passed':True,'device':'CPU','dtype':'float64','shape':{'samples':9,'coordinates':7,'units':11,'vocabulary':13},
        'full_parameter_gradient_autograd_max':max(maxima),'full_parameter_gram_max':gram_error,
        'RMS_direction_derivative_max':norm_error,'full_dense_Fisher_direction_max':max(dense_errors),
        'SGD_checks':step_checks,'scope':'Synthetic implementation validation only, not native language evidence or historical pretraining.'}
    save(BASE/'verification/math.json',report)
    print('LAW_PRODUCTION_MATH_CHECK_PASS',report,flush=True)


if __name__=='__main__':main()
