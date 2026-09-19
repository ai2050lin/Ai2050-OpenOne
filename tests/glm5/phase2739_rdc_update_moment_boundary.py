"""Exact finite-degree counterexample, not a naturally reachable Transformer state."""
from rdc_update_common import *

def main():
    out=BASE/'moment_boundary'
    if (out/'result.json').exists():return
    start=time.monotonic();d=2560;n=32;indices=np.arange(n,dtype=np.int64)
    signs=np.array([(-1)**int(i).bit_count() for i in indices],dtype=np.int64)
    exact=[int(sum(int(signs[i])*int(i)**power for i in range(n))) for power in range(5)]
    assert exact==[0]*5
    with np.load(PRIOR/'prediction/role_probe.npz') as z:c=z['coefficients'][:-1].astype(float);bias=z['coefficients'][-1].astype(float)
    rng=np.random.default_rng(2739);v=rng.normal(size=d);v-=c@np.linalg.solve(c.T@c,c.T@v);v/=np.sqrt(np.mean(v*v))
    null_error=float(np.max(abs(v@c)));assert null_error<1e-10
    # At each of the same32 ordered positions, signs differ. All signed moment
    # orders1..4 combined with position polynomials0..4 agree nevertheless.
    ha=signs[:,None]*v;hb=-ha;pos=indices/31
    moment_coefficients=[]
    for order in range(1,5):
        for power in range(5):
            difference=float(np.mean((signs.astype(float)**order-(-signs).astype(float)**order)*pos**power))
            assert abs(difference)<1e-14
            moment_coefficients.append({'state_order':order,'position_order':power,'coefficient_difference':difference,
              'exact_reason':'even state order equal per position; odd state order uses exact integer Prouhet sums'})
    ra=np.maximum(ha@c+bias,0)+1e-6;ra/=ra.sum(-1,keepdims=True)
    rb=np.maximum(hb@c+bias,0)+1e-6;rb/=rb.sum(-1,keepdims=True)
    role_error=float(np.max(abs(ra-rb)));assert role_error<1e-10
    attention=[];responses=[]
    # Same scalar keys (source positions); different signed value vectors.
    # This is an ordinary softmax attention counterexample, not a native hook.
    for query in (0.,.25,.5,1.):
        score=query*indices;weights=np.exp(score-score.max());weights/=weights.sum()
        va=weights@ha;vb=weights@hb
        responses.append(np.stack([va,vb]));attention.append({'query':query,'attention_output_difference_rms':float(np.sqrt(np.mean((va-vb)**2))),
          'same_degree4_moments':True})
    assert attention[0]['attention_output_difference_rms']<1e-12 and attention[-1]['attention_output_difference_rms']>.1
    npz(out/'counterexample_full_coordinates.npz',positions=indices,signs=signs,v=v,history_a=ha,history_b=hb,
      coarse_roles_a=ra,coarse_roles_b=rb,attention_responses=np.array(responses),role_probe=c,role_intercept=bias)
    result={'timestamp':stamp(),'source':snapshot(__file__),'positions':n,'coordinates':d,'exact_integer_sums_powers0through4':exact,
      'all_state_position_moment_checks':moment_coefficients,'role_nullspace_error':null_error,'role_scores_max_difference':role_error,'attention':attention,
      'new_statement':'Signed state moments through degree4 with bounded polynomial position features through degree4 need not identify an ordered history, even with all native-coordinate dimensions retained.',
      'proof':'For epsilon_s=(-1)^popcount(s), sum epsilon_s*s^j=0 for j<5. H and -H have identical even moments, while odd moments cancel against each declared position polynomial. Softmax exp(q*s) is not such a bounded-degree polynomial.',
      'boundary':'Not a counterexample to injective one-hot position storage, all-order moments with suitable assumptions, full KV, or every conceivable compression. Synthesized FP64 ambient vectors, not demonstrated natural reachable H/K/V; no causal claim about forgetting.',
      'scope_decision':'The proposed universal collision-free3rd/4th moment prerequisite fails. Do not start100K lossless-KV/forgetting-elimination benchmark on this premise. Continue finite natural-prefix/query prediction and actual native KV bookkeeping without claiming exact compression.',
      'seconds':time.monotonic()-start,'all_passed':True}
    save(out/'result.json',result);ledger('finite_degree_moment_boundary',result['seconds']);print('FINITE_MOMENT_BOUNDARY',attention,flush=True)

if __name__=='__main__':main()
