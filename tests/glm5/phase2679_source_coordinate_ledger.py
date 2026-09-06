"""Exact-coordinate conditional attention/MLP arithmetic, without donor states.

This is preparation for Phase2679, not a model result or phase completion.
All computations are NumPy float64 on already observed native activations.
Native BF16/FP32 rounding is retained as an explicit residual, not silently
assigned to a source word. V at a source position is contextualized already.
"""
import numpy as np


def finite64(value):
    value=np.asarray(value,dtype=np.float64)
    assert np.isfinite(value).all()
    return value


def head_source_terms(probabilities,values,output_weight):
    """Yield every head's [query,source,output-coordinate] array.

    probabilities: [query,query_head,source], actual post-softmax values.
    values: [source,KV_head,head_coordinate], actual V projection outputs.
    output_weight: [output_coordinate,query_head*head_coordinate], actual Wo.
    GQA repeats *consecutive* heads, following the inspected local implementation.
    Iteration is exact memory blocking, never head or coordinate selection.
    """
    p,v,w=map(finite64,(probabilities,values,output_weight))
    assert p.ndim==v.ndim==3 and w.ndim==2
    nq,nh,ns=p.shape;sv,nkv,dh=v.shape
    assert sv==ns and nh%nkv==0 and w.shape[1]==nh*dh
    assert (p>=0).all() and (p<=1).all()
    for head in range(nh):
        kv=head//(nh//nkv)
        projected=v[:,kv,:]@w[:,head*dh:(head+1)*dh].T
        yield head,p[:,head,:,None]*projected[None,:,:]


def attention_ledger(probabilities,values,output_weight,actual_output,
                     actual_head_concat,output_bias=None,observer=None):
    """Separate AV rounding from Wo rounding in the observed forward pass.

    actual_head_concat is the actual input of o_proj at the queried tokens.
    observer(head, all_terms) can stream every head/coordinate to an exact
    accumulator; its return value is ignored and it may not alter terms.
    """
    p,v,w,out,z=map(finite64,(probabilities,values,output_weight,actual_output,actual_head_concat))
    nq,nh,ns=p.shape;_,nkv,dh=v.shape;d=w.shape[0]
    assert out.shape==(nq,d) and z.shape==(nq,nh*dh)
    bias=np.zeros(d) if output_bias is None else finite64(output_bias)
    assert bias.shape==(d,)
    source=np.zeros((nq,ns,d));heads=np.zeros((nq,nh,d))
    av_exact=np.empty((nq,nh,dh))
    for head,terms in head_source_terms(p,v,w):
        source+=terms;heads[:,head]=terms.sum(axis=1)
        av_exact[:,head]=p[:,head,:]@v[:,head//(nh//nkv),:]
        if observer is not None:
            terms.flags.writeable=False
            observer(head,terms)
    av_flat=av_exact.reshape(nq,nh*dh)
    native_wo_exact=z@w.T+bias
    ideal_wo_exact=av_flat@w.T+bias
    av_rounding=(z-av_flat)@w.T
    wo_rounding=out-native_wo_exact
    # This is accumulation-order disagreement between two float64 identities,
    # not an extra native rounding stage. Do not conflate the two quantities.
    accounting_order=ideal_wo_exact-(source.sum(axis=1)+bias)
    recovered=source.sum(axis=1)+bias+av_rounding+wo_rounding+accounting_order
    return {'source_terms':source,'head_terms':heads,'output_bias':bias,
            'native_attention_output':out,'native_head_concat':z,
            'exact64_head_concat':av_flat,'av_rounding_output':av_rounding,
            'wo_rounding_output':wo_rounding,'accounting_order_residual':accounting_order,
            'reconstruction_error':out-recovered}


def conditional_norm_ledger(residual,attention,gamma,epsilon,actual_pre_norm,actual_x):
    """Allocate observed RMSNorm using its actual pre-norm state.

    The denominator is endogenous. Keeping it fixed provides a conditional
    additive ledger, NOT the result of deleting or replacing any source.
    All residual stages are stored as full physical-coordinate vectors.
    """
    h,gamma,pre,x=map(finite64,(residual,gamma,actual_pre_norm,actual_x))
    source=attention['source_terms'];out=attention['native_attention_output']
    nq,ns,d=source.shape
    assert h.shape==pre.shape==x.shape==(nq,d) and gamma.shape==(d,)
    assert np.isfinite(epsilon) and epsilon>0
    denominator=np.sqrt(np.mean(pre*pre,axis=-1,keepdims=True)+epsilon)
    scale=gamma[None,:]/denominator
    branches={'residual':h*scale,'attention_bias':attention['output_bias'][None,:]*scale,
              'av_rounding':attention['av_rounding_output']*scale,
              'wo_rounding':attention['wo_rounding_output']*scale,
              'float64_accounting_order':attention['accounting_order_residual']*scale,
              'residual_add_rounding':(pre-h-out)*scale,
              'rmsnorm_rounding':x-pre*scale}
    normalized_source=source*scale[:,None,:]
    reconstructed=normalized_source.sum(axis=1)+sum(branches.values())
    return {'source_x':normalized_source,'branches_x':branches,
            'observed_denominator64':denominator[:,0],
            'reconstruction_error':x-reconstructed,
            'interpretation':'Conditional observed-normalizer allocation, not source ablation.'}


def input_weight_ledger(norm,weight_row,actual_projection,bias=0.0,observer=None):
    """Every incoming scalar W[j,k] times every normalized source coordinate.

    All signed terms are kept through the source-coordinate matrix. Positive
    and negative sums are descriptive arithmetic, not statistical attribution.
    """
    w,actual=map(finite64,(weight_row,actual_projection))
    source=norm['source_x'];nq,ns,d=source.shape
    assert w.shape==(d,) and actual.shape==(nq,) and np.isfinite(bias)
    terms=source*w[None,None,:]
    branches={k:a*w[None,:] for k,a in norm['branches_x'].items()}
    x_recovered=source.sum(axis=1)+sum(norm['branches_x'].values())+norm['reconstruction_error']
    exact64=x_recovered@w+float(bias)
    projection_rounding=actual-exact64
    norm_accounting=norm['reconstruction_error']*w[None,:]
    reconstructed=terms.sum(axis=(1,2))+sum(a.sum(axis=-1) for a in branches.values())+norm_accounting.sum(axis=-1)+float(bias)+projection_rounding
    if observer is not None:
        terms.flags.writeable=False
        observer(terms)
    return {'source_coordinate_terms':terms,'branch_coordinate_terms':branches,
            'norm_accounting_coordinate_terms':norm_accounting,
            'projection_rounding':projection_rounding,'bias':float(bias),
            'positive_source_sum':np.maximum(terms,0).sum(axis=(1,2)),
            'negative_source_sum':np.minimum(terms,0).sum(axis=(1,2)),
            'source_signed_sum':terms.sum(axis=(1,2)),
            'reconstruction_error':actual-reconstructed}
