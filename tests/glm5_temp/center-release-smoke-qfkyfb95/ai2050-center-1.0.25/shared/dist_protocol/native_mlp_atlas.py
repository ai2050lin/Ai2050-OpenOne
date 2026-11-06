"""Whole-native-unit conditional response tests; no automatic mechanism promotion."""
from __future__ import annotations
import numpy as np
from scipy.stats import binom

SCHEMA='native_mlp_atlas_analysis.v1'

def decision_contrast(values):
    """Input B×C×M×T×Q×readout×unit; cancel main fact and question effects."""
    x=np.asarray(values)
    if x.ndim!=7 or x.shape[2:5]!=(2,2,2):raise ValueError('expected background,code,mapping,fact,question,readout,unit')
    return .5*((x[:,:,:,0,0]-x[:,:,:,1,0])+(x[:,:,:,1,1]-x[:,:,:,0,1]))

def fit_sign(delta,reference_code=0):
    d=np.asarray(delta)
    if d.ndim!=5:raise ValueError('expected background,code,mapping,readout,unit')
    return np.sign(d[:,reference_code].mean((0,1),dtype=np.float64)).astype(np.int8)

def evaluate(delta,sign,*,alpha,min_accuracy=.8,min_nonzero=.8):
    d=np.asarray(delta);sign=np.asarray(sign)
    if d.ndim!=5 or sign.shape!=d.shape[-2:]:raise ValueError('sign andnative response shapes differ')
    n=d.shape[0];oriented=d*sign[None,None,None];success=(oriented>0).sum(0);nonzero=(d!=0).sum(0)
    accuracy=success/n;p=binom.sf(success-1,n,.5)
    passed=(accuracy>=min_accuracy)&(nonzero/n>=min_nonzero)&(p<alpha)&(sign[None,None]!=0)
    return {'n':n,'successes':success.astype(np.int16),'nonzero':nonzero.astype(np.int16),'accuracy':accuracy.astype(np.float32),'p':p,'passed':passed,'all_components_passed':passed.all((0,1))}

def factorial_coefficients(values):
    """Ordinary eight-cell coefficients; split mean and background RMS, no new algebra."""
    x=np.asarray(values);bits=np.array([-1.,1.],np.float32)
    if x.ndim!=7 or x.shape[2:5]!=(2,2,2):raise ValueError('invalid factorial grid')
    T=bits[None,:,None];Q=bits[None,None,:];M=bits[:,None,None]
    terms=[np.ones((2,2,2)),T,Q,M,T*Q,T*M,Q*M,T*Q*M]
    names=['constant','fact','question','mapping','decision','fact_mapping','question_mapping','lexical_decision']
    means=[];rms=[]
    for term in terms:
        b=(x*term[None,None,:,:, :,None,None]).mean((2,3,4),dtype=np.float64)
        means.append(b.mean(0));rms.append(np.sqrt((b*b).mean(0)))
    return names,np.stack(means,1).astype(np.float32),np.stack(rms,1).astype(np.float32)

def prediction_fvu(discovery,heldout):
    """Frozen discovery cell means, evaluated on new backgrounds. Native unit arrays retained."""
    a=np.asarray(discovery);b=np.asarray(heldout)
    if a.shape[1:]!=b.shape[1:]:raise ValueError('factor/readout/native axes differ')
    cell=a.mean(0,dtype=np.float64);global_mean=a.mean((0,1,2,3,4),dtype=np.float64)
    num=((b-cell[None])**2).sum((0,1,2,3,4),dtype=np.float64)
    den=((b-global_mean[None,None,None,None,None])**2).sum((0,1,2,3,4),dtype=np.float64)
    out=np.divide(num,den,out=np.ones_like(num),where=den>1e-20)
    return {'fvu':out.astype(np.float32),'prediction_sse':num,'global_mean_baseline_sse':den,'zero_variance':den<=1e-20}

def power_for_intersection(n,p=.85,*,alpha,min_accuracy=.8,components=12):
    critical=max(int(binom.isf(alpha,n,.5))+1,int(np.ceil(n*min_accuracy)))
    marginal=float(binom.sf(critical-1,n,p))
    return {'n':n,'assumed_order_probability':p,'critical_successes':critical,'marginal_power':marginal,'components':components,'joint_power_union_lower_bound':max(0.,1-components*(1-marginal)),'shared_background_dependence':'not assumed independent'}
