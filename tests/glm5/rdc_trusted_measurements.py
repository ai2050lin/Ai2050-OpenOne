"""Correct, independently checked measurement primitives for RDC repair runs."""
import numpy as np
import torch


def silu_jvp(g, u, dg, du):
    s = torch.sigmoid(g)
    return u * (s + g * s * (1 - s)) * dg + (g * s) * du


def rms(x, gamma, eps):
    return gamma * x / torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True) + eps)


def rms_jvp(x, gamma, dx, eps):
    r = torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True) + eps)
    return gamma * (dx/r - x*torch.mean(x*dx, dim=-1, keepdim=True)/(r**3))


def measure(pred, truth):
    pn, tn = torch.linalg.vector_norm(pred), torch.linalg.vector_norm(truth)
    return dict(cos=float(torch.dot(pred, truth)/(pn*tn)) if float(pn*tn)>0 else None,
                relative_error=float(torch.linalg.vector_norm(pred-truth)/tn) if float(tn)>0 else None,
                norm_ratio=float(pn/tn) if float(tn)>0 else None)


def energy(a, h):
    total = a+h
    den = torch.dot(total, total)
    if float(den)==0:
        return {'degenerate':True}
    return dict(norm_a=float(a@a/den),norm_h=float(h@h/den),cross=float(2*(a@h)/den),
                projected_a=float(a@total/den),projected_h=float(h@total/den),
                note='Projected contributions sum to one, can be negative; not causal probabilities.')


def rankcorr(a,b):
    def ranks(x):
        _,inv,n=np.unique(x,return_inverse=True,return_counts=True)
        end=np.cumsum(n)
        return ((end-n+end-1)/2)[inv]
    a,b=ranks(a),ranks(b)
    if np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])


def self_test():
    torch.manual_seed(2751001)
    x=torch.randn(19,dtype=torch.float64)
    g,u,dg,du=[torch.randn(19,dtype=torch.float64) for _ in range(4)]
    fun=lambda gg,uu: torch.nn.functional.silu(gg)*uu
    expected=torch.autograd.functional.jvp(fun,(g,u),(dg,du))[1]
    actual=silu_jvp(g,u,dg,du)
    e=1e-5
    finite=(fun(g+e*dg,u+e*du)-fun(g-e*dg,u-e*du))/(2*e)
    jerr=float((expected-actual).abs().max())
    ferr=float((finite-actual).abs().max())
    gamma=torch.randn_like(x); dx=torch.randn_like(x)
    ref=torch.autograd.functional.jvp(lambda xx:rms(xx,gamma,1e-6),(x,),(dx,))[1]
    rerr=float((ref-rms_jvp(x,gamma,dx,1e-6)).abs().max())
    en=energy(g,u)
    assert jerr<1e-12 and ferr<1e-8 and rerr<1e-12
    assert abs(en['norm_a']+en['norm_h']+en['cross']-1)<1e-12
    assert abs(en['projected_a']+en['projected_h']-1)<1e-12
    assert rankcorr([1,1,1],[2,3,4]) is None
    assert abs(rankcorr([1,1,2],[2,1,3])-rankcorr([2,1,1],[3,2,1]))<1e-12
    return dict(swiglu_autograd_error=jerr,swiglu_finite_difference_error=ferr,rms_autograd_error=rerr,
                energy_identity_passed=True,tie_invariance_passed=True,status='passed')
