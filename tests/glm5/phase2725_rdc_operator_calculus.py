"""Independent calculus checks and native full-matrix operator identities, with RMSNorm included."""
import gc
from rdc_operator_common import *
from rdc_native_conditional_operator import weights


def synthetic():
    import torch
    gen=torch.Generator().manual_seed(2725)
    g=torch.randn(11,7,generator=gen,dtype=torch.float64)/3
    u=torch.randn(11,7,generator=gen,dtype=torch.float64)/3
    d=torch.randn(7,11,generator=gen,dtype=torch.float64)/3
    x=torch.randn(7,generator=gen,dtype=torch.float64)
    gate,value=g@x,u@x
    sig=gate.sigmoid();phi=gate*sig;der=sig*(1+gate*(1-sig))
    K=(d*phi)@u
    J=(d*(der*value))@g+K
    fn=lambda xx:d@(torch.nn.functional.silu(g@xx)*(u@xx))
    true=torch.autograd.functional.jacobian(fn,x)
    assert torch.allclose(K@x,fn(x),atol=1e-12,rtol=1e-12)
    assert torch.allclose(J,true,atol=1e-12,rtol=1e-12)
    gamma=torch.rand(7,generator=gen,dtype=torch.float64)+.3
    eps=1e-6;s=torch.sqrt(x.square().mean()+eps)
    JR=torch.diag(gamma/s)-torch.outer(gamma*x,x)/(len(x)*s**3)
    norm=lambda xx:gamma*xx/torch.sqrt(xx.square().mean()+eps)
    assert torch.allclose(JR,torch.autograd.functional.jacobian(norm,x),atol=1e-12,rtol=1e-12)
    x0=x*.4;g0=g@x0;u0=u@x0;dg=g@(x-x0);du=u@(x-x0)
    ss=g0.sigmoid();ph0=g0*ss;dp=ss*(1+g0*(1-ss));d2=ss*(1-ss)*(2+g0*(1-2*ss))
    hh=torch.autograd.functional.hessian(lambda t:fn(x0+t*(x-x0)).sum(),torch.tensor(0.,dtype=torch.float64))
    second=(d@(2*dp*dg*du+d2*dg.square()*u0)).sum()
    assert torch.allclose(hh,second,atol=1e-12,rtol=1e-12)
    return {'K_action_max_error':float((K@x-fn(x)).abs().max()),'full_J_max_error':float((J-true).abs().max()),
            'RMS_J_max_error':float((JR-torch.autograd.functional.jacobian(norm,x)).abs().max()),
            'quadratic_second_derivative_error':float(abs(hh-second)),
            'frozen_gate_is_not_J_relative_error':float((K-true).square().sum()/true.square().sum()),
            'scope':'Independent deterministic float64 seven-coordinate/eleven-unit calculation, not a language experiment.'}


def main():
    import torch
    from safetensors import safe_open
    out=BASE/'calculus'
    if (out/'result.json').exists():
        return
    torch.set_num_threads(2)
    start=time.monotonic();checks=synthetic()
    checkpoint=ROOT/'models/hf/qwen3-4b'
    index=read(checkpoint/'model.safetensors.index.json')['weight_map']
    source=next(r for r in rows() if r['sample_id']=='train-en-o0000')
    with np.load(BASE/'capture/main/factors'/f'{source["sample_id"]}.npz') as z:
        arrays={k:z[k] for k in z.files}
    results=[]
    for block in (6,16,34):
        w=weights(block)
        x=torch.tensor(unbits(arrays[f'L{block}_x'][0]),device='cuda:0')
        gate,value=w['g']@x,w['u']@x
        sig=gate.sigmoid();phi=gate*sig;der=sig*(1+gate*(1-sig))
        K=(w['d']*phi)@w['u']
        Jgate=(w['d']*(der*value))@w['g'];J=Jgate+K
        fn=lambda xx:w['d']@(torch.nn.functional.silu(w['g']@xx)*(w['u']@xx))
        normname=f'model.layers.{block}.post_attention_layernorm.weight'
        with safe_open(str(checkpoint/index[normname]),framework='pt',device='cpu',backend='pread') as f:
            gamma=f.get_tensor(normname).float().to('cuda:0')
        r=(torch.tensor(unbits(arrays[f'L{block}_input'][0]),device='cuda:0').bfloat16()+
           torch.tensor(unbits(arrays[f'L{block}_attention'][0]),device='cuda:0').bfloat16()).float()
        eps=read(checkpoint/'config.json')['rms_norm_eps'];s=torch.sqrt(r.square().mean()+eps)
        norm=lambda rr:gamma*rr/torch.sqrt(rr.square().mean()+eps)
        checks_direction=[]
        gen=torch.Generator(device='cuda:0').manual_seed(2725+block)
        for i in range(8):
            v=torch.randn(2560,device='cuda:0',generator=gen);v=v/torch.linalg.vector_norm(v)
            _,target=torch.autograd.functional.jvp(fn,x,v)
            approx=J@v
            jrv=gamma/s*v-gamma*r*(r@v)/(2560*s**3)
            _,true_r=torch.autograd.functional.jvp(norm,r,v)
            _,combined=torch.autograd.functional.jvp(lambda rr:fn(norm(rr)),r,v)
            # The Jacobian at normalized FP32 r must be used in the composition, not at rounded native x.
            xn=norm(r);gg=w['g']@xn;uu=w['u']@xn;ss=gg.sigmoid();dp=ss*(1+gg*(1-ss))
            jn_v=w['d']@(dp*uu*(w['g']@jrv)+(gg*ss)*(w['u']@jrv))
            checks_direction.append({'J_action_relative_MSE':float((target-approx).square().sum()/target.square().sum()),
                'RMS_action_relative_MSE':float((true_r-jrv).square().sum()/true_r.square().sum()),
                'composed_J_action_relative_MSE':float((combined-jn_v).square().sum()/combined.square().sum())})
        reconstruction=float((K@x-fn(x)).square().sum()/fn(x).square().sum())
        checks_max=max(v for record in checks_direction for v in record.values())
        assert reconstruction<1e-8 and checks_max<1e-8
        npz(out/f'L{block}_complete_native_operators.npz',K=K.detach().cpu().numpy(),J=J.detach().cpu().numpy(),
            x=x.cpu().numpy(),pre_MLP_residual=r.cpu().numpy(),norm_weight=gamma.cpu().numpy())
        results.append({'block':block,'sample_id':source['sample_id'],'native_position':source['anchors'][0],
            'native_matrix_shape':[2560,2560],'K_action_relative_MSE':reconstruction,'directional_checks':checks_direction,
            'gate_branch_Frobenius_squared':float(Jgate.square().sum()),'up_branch_Frobenius_squared':float(K.square().sum()),
            'two_branch_cross_term':float(2*(Jgate*K).sum()),'full_J_Frobenius_squared':float(J.square().sum()),
            'RMS_radial_action_norm':float(torch.linalg.vector_norm(gamma/s*r-gamma*r*(r@r)/(2560*s**3))),
            'all_native_coordinates_preserved':True})
        del w,x,gate,value,sig,phi,der,K,Jgate,J,gamma,r
        gc.collect();torch.cuda.empty_cache()
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'synthetic':checks,'native':results,
        'formulas':{'K':'Wd diag(phi(g)) Wu','J':'Wd [diag(phi_prime(g)*u) Wg + diag(phi(g)) Wu]',
            'RMS_J':'diag(gamma)/s - (gamma*r) r^T/(D*s^3); s=sqrt(mean(r^2)+epsilon)',
            'quadratic':'phi0*u0 + phi0*du + phi0_prime*dg*u0 + phi0_prime*dg*du + 0.5*phi0_second*dg^2*u0'},
        'limits':'These are checked known derivatives and architecture identities. No new semantic theorem, native hard gate, exact global quadratic law or omission of RMS is justified. Local calculus at FP32 same-valued weights is separated from native BF16 rounding.'}
    save(out/'result.json',result);ledger('full_native_K_J_calculus',time.monotonic()-start,native_matrices=6)
    guard();print('NATIVE_CALCULUS_COMPLETE',checks,[(r['block'],r['K_action_relative_MSE']) for r in results],flush=True)


if __name__=='__main__':
    main()
