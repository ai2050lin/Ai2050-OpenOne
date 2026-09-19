"""Ambient-state collision certificate for the frozen source-moment candidates.

This is a synthetic mathematical counterexample, NOT two observed LLM histories.
The actual fitted role probe is used; every native coordinate remains present.
"""
from rdc_binding_common import *

def main():
    from threadpoolctl import threadpool_limits
    from rdc_binding_kernels import source_arrays
    threadpool_limits(limits=2)
    start=time.monotonic();out=BASE/'verification/source_moment_collision'
    with np.load(BASE/'prediction/role_probe.npz') as z:coef=z['coefficients'].astype(np.float64)
    w=coef[:-1];d=len(w);rng=np.random.default_rng(2735001);v=rng.standard_normal(d)
    weights,_,rank,singular=np.linalg.lstsq(w,v,rcond=None)
    v-=w@weights;v/=np.sqrt(np.mean(v*v))
    null_error=float(np.max(abs(v@w)));assert null_error<1e-10
    row=next(r for r in gzread(BASE/'natural_confirmation.json.gz') if r['split']=='connected_test')
    p=row['anchors'][-1]
    with np.load(BASE/'capture/natural'/f'{row["sample_id"]}.npz') as z:
        h=unbits(z['H12_sources'])[:p+1].astype(np.float64)
        e=unbits(z['embedding'])[-1].astype(np.float64)
    h/=np.sqrt(np.mean(h*h,-1,keepdims=True)).clip(1e-8)
    e/=np.sqrt(np.mean(e*e));a=h.copy();b=h.copy()
    a[0]=v;a[1]=-v;b[0]=-v;b[1]=v
    assert len(a)>2 and np.array_equal(a[-1],b[-1])
    outer=np.outer(v,v);outer_other=np.outer(-v,-v)
    assert np.array_equal(outer,outer_other)
    outer_identity=identity(outer);del outer,outer_other
    def roles(x):
        r=np.maximum(x@w+coef[-1],0)+1e-6
        return r/r.sum(-1,keepdims=True)
    ra=roles(a);rb=roles(b);role_difference=float(np.max(abs(ra-rb)))
    assert role_difference<1e-10
    means_difference=float(np.max(abs(a.mean(0)-b.mean(0))));assert means_difference<1e-12
    references=[r for r in gzread(BASE/'natural_discovery.json.gz') if r['split']=='train'][:16]
    ss,qq,ee,_,_,_=source_arrays(references)
    scores={'source_mean':[],'role_position_pair':[]}
    zleft=np.linspace(-1,0,len(a));pl=np.stack([np.ones_like(zleft),zleft,zleft*zleft],1)
    for s,q,embedding in zip(ss,qq,ee):
        s=s.astype(np.float64);q=q.astype(np.float64);embedding=embedding.astype(np.float64)
        q/=np.sqrt(np.mean(q*q));embedding/=np.sqrt(np.mean(embedding*embedding))
        zr=np.linspace(-1,0,len(s));pr=np.stack([np.ones_like(zr),zr,zr*zr],1);rr=roles(s)
        base=(a[-1]@q+e@embedding)/(2*d);po=pl@pr.T
        for name in scores:
            values=[]
            for x,r in ((a,ra),(b,rb)):
                if name=='source_mean':term=x.mean(0)@s.mean(0)/d
                else:term=np.mean((x@s.T/d)**2*po*(1+r@rr.T))
                values.append(1+base+term+base*term)
            scores[name].append(values)
    differences={k:float(np.max(abs(np.array(value)[:,0]-np.array(value)[:,1]))) for k,value in scores.items()}
    assert max(differences.values())<1e-10,differences
    npz(out/'synthetic_normalized_state_pair.npz',history_a=a,history_b=b,probe_null_vector=v,
      role_a=ra,role_b=rb,query=a[-1],embedding=e,probe_projection_coefficients=weights,
      probe_all_singular_values=singular,**{k+'_reference_kernels':np.array(value) for k,value in scores.items()})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'all_passed':True,
      'classification':'Known sign-invariance / finite-moment non-injectivity certificate on ambient states, not native reachable language evidence.',
      'actual_role_probe_sha256':sha(BASE/'prediction/role_probe.npz'),'actual_context_scaffold':row['sample_id'],
      'native_coordinates':d,'source_positions':len(a),'probe_linear_rank':int(rank),'probe_nullity_at_least':d-int(rank),
      'probe_null_max_error':null_error,'roles_max_difference':role_difference,'source_mean_max_difference':means_difference,
      'full_D_squared_sign_outer_product_identical':outer_identity,
      'distinct_history_frobenius_distance':float(np.linalg.norm(a-b)),
      'actual_training_reference_rows':len(references),'reference_ids':[r['sample_id'] for r in references],
      'operational_kernel_max_differences':differences,
      'proof':'Choose v in ker(C_role^T), normalized to RMS1. The fitted affine role score is identical at v and -v. Swap (+v,-v) to (-v,+v) at two earlier positions. Current q/e and mean stay fixed; each h outer h, position, and predicted role feature stays fixed, hence both selected source kernels agree against every reference in real arithmetic.',
      'limits':['Synthetic H12 arrays need not be reachable under actual model/token histories.',
        'This rules out ambient injectivity of these summaries; it does not prove an observed failure for every natural-language domain.',
        'Retaining all scalar coordinates before aggregation does not itself retain all source identity/order information.',
        'No new mathematical theory or physical neuron/sign-flip intervention is claimed.'],
      'seconds':time.monotonic()-start}
    save(out/'result.json',result);ledger('binding_source_moment_collision_certificate',result['seconds'])
    print('BINDING_SOURCE_MOMENT_COLLISION',json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
