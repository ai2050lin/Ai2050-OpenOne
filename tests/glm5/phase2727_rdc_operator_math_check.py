"""Independent finite-vocabulary math tests for the output-path audit (CPU only)."""
from rdc_operator_common import *
from phase2727_rdc_operator_metric import logit_metric


def main():
    import torch
    out=BASE/'metric_followup';start=time.monotonic();torch.set_num_threads(2)
    generator=torch.Generator(device='cpu').manual_seed(2727)
    z=torch.randn(37,generator=generator,dtype=torch.float64)
    direction=torch.randn(37,generator=generator,dtype=torch.float64)
    rr=[]
    for scale in (1e-3,.1,1.,10.):
        result=logit_metric(z,z+scale*direction)
        assert result['integral64_absolute_error']<1e-10
        shifted=logit_metric(z+17,z+scale*direction+29)
        assert abs(result['exact_KL']-shifted['exact_KL'])<1e-11
        rr.append({'scale':scale,**result,'independent_constant_shift_KL_difference':abs(result['exact_KL']-shifted['exact_KL'])})
    assert abs(rr[0]['exact_KL']/rr[0]['endpoint_Fisher_half_variance']-1)<.005
    stress=[]
    for scale in (50.,200.,1000.):
        result=logit_metric(z,z+scale*direction)
        assert result['adaptive_absolute_error']<1e-7
        stress.append({'scale':scale,**result})
    q=(z+torch.randn(37,generator=generator,dtype=torch.float64)).softmax(-1);p0=z.softmax(-1)
    d0=float((q*(q.log()-z.log_softmax(-1))).sum());d1=float((q*(q.log()-(z+direction).log_softmax(-1))).sum())
    r=logit_metric(z,z+direction);linear=float(((p0-q)*direction).sum())
    assert abs(d1-d0-linear-r['integral64'])<1e-10
    w=torch.randn(37,7,generator=generator,dtype=torch.float64);dh=torch.randn(7,generator=generator,dtype=torch.float64)
    mean=p0@w;g=w.T@(p0[:,None]*w)-mean[:,None]*mean[None,:];delta=w@dh
    direct=float((p0*(delta-(p0*delta).sum()).square()).sum());matrix=float(dh@g@dh)
    assert abs(direct-matrix)<1e-10 and float(torch.linalg.eigvalsh(g).min())>=-1e-12
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'CPU_only':True,'synthetic_vocab_size':37,'cases':rr,'adaptive_stress':stress,
        'full_readout_pullback_check':{'width':7,'direct_variance':direct,'complete_matrix_form':matrix,'absolute_error':abs(direct-matrix),'minimum_eigenvalue':float(torch.linalg.eigvalsh(g).min())},
        'general_reference_check':{'Dq_p0':d0,'Dq_p1':d1,'linear_correction':linear,'integral':r['integral64'],'absolute_error':abs(d1-d0-linear-r['integral64'])},
        'scope':'Deterministic synthetic algebra/shift invariance/quadratic small-displacement tests, not language evidence. The pure variance integral requires reference q=p0; otherwise initial KL and linear term are mandatory.'}
    save(out/'math_check_readout.json',result);ledger('output_path_and_readout_math_CPU',time.monotonic()-start)
    print('OUTPUT_PATH_MATH_PASS',max(r['integral64_absolute_error'] for r in rr),flush=True)


if __name__=='__main__':main()
