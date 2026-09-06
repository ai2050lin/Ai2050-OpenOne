"""Synthetic CPU algebra checks for next native gate/up/down measurements, NOT LLM evidence."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
OUT=RESULT/'phase2669_symmetric_multitoken_delivery'


def main():
    torch.set_num_threads(2);reports=[]
    for seed in range(16):
        rng=torch.Generator(device='cpu').manual_seed(2670+seed);T,D,K=17,31,43
        x=torch.randn(T,D,generator=rng,dtype=torch.float64);G=(torch.randn(K,D,generator=rng,dtype=torch.float64)/D**.5).requires_grad_();U=(torch.randn(K,D,generator=rng,dtype=torch.float64)/D**.5).requires_grad_();W=(torch.randn(D,K,generator=rng,dtype=torch.float64)/K**.5).requires_grad_()
        g=x@G.T;u=x@U.T;s=torch.nn.functional.silu(g);a=s*u;y=a@W.T
        objective=(torch.sin(y)+.13*y*y).sum();exact=torch.autograd.grad(objective,(G,U,W))
        with torch.no_grad():
            gy=torch.cos(y)+.26*y;ga=gy@W;sig=torch.sigmoid(g);sp=sig+g*sig*(1-sig)
            predicted=((ga*u*sp).T@x,(ga*s).T@x,gy.T@a)
            error=[float(torch.max(torch.abs(p-e))) for p,e in zip(predicted,exact)]
            x1=x+.07*torch.randn(T,D,generator=rng,dtype=torch.float64);g1=x1@G.T;u1=x1@U.T;s1=torch.nn.functional.silu(g1)
            delta=(s1*u1)-(s*u);expanded=s*(u1-u)+u*(s1-s)+(s1-s)*(u1-u)
            product_error=float(torch.max(torch.abs(delta-expanded)))
            # Explicit row sums retain every input coordinate; both positive and negative small values occur.
            layer_error=float(torch.max(torch.abs(y-sum(a[:,j,None]*W[:,j][None,:] for j in range(K)))))
        reports.append({'seed':seed,'tokens':T,'hidden':D,'mlp_units':K,'full_matrix_gradient_abs_error_gate_up_down':error,'finite_product_expansion_error':product_error,'all_unit_output_reconstruction_error':layer_error})
    checks={'16_cpu_float64_cases':len(reports)==16,'all_three_full_matrix_adjoints':max(max(r['full_matrix_gradient_abs_error_gate_up_down']) for r in reports)<1e-12,
        'finite_product_identity':max(r['finite_product_expansion_error'] for r in reports)<1e-12,'all_unit_down_sum':max(r['all_unit_output_reconstruction_error'] for r in reports)<1e-12}
    report={'checks':checks,'all_checks_passed':all(checks.values()),'runs':reports,'formula':{'gate':'sum_t (gy[t]@Wdown)[j]*u[t,j]*silu_prime(g[t,j])*x[t,k]','up':'sum_t (gy[t]@Wdown)[j]*silu(g[t,j])*x[t,k]','down':'sum_t gy[t,i]*a[t,j]','finite':'delta_a=s0*delta_u+u0*delta_s+delta_s*delta_u'},
        'boundary':'Synthetic CPU float64 function tests only. NOT local-model experiments, semantic findings, evidence that candidate units are used, or validation of next campaign hooks. Finite expansion is basepoint-dependent accounting, not uniquely assigned causal credit. Real model bias/norm/architecture and numerical rounding still require explicit native measurement.'}
    save(OUT/'analysis/mlp_formula_preflight.json',report);print(json.dumps({'checks':checks,'max_gradient_error':max(max(r['full_matrix_gradient_abs_error_gate_up_down']) for r in reports)}));assert report['all_checks_passed']


if __name__=='__main__':main()
