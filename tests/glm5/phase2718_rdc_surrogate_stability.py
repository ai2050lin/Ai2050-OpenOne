"""Exact all-coordinate piecewise-affine structure of the learned temporal surrogate."""
from scipy.linalg import eig,solve
from rdc_relation_common import *
from rdc_relation_inference import TemporalRule
from phase2716_rdc_relation_dynamics import embeddings
from phase2716_rdc_relation_probability import Readout


def main():
    import torch
    out=BASE/'surrogate_stability';guard(3*1024**2)
    if (out/'result.json').exists():return
    save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'phase':2718,
      'question':'Can repeated continuations of the frozen learned state updater be explained by its own complete affine operators?',
      'scope':'This analyzes the learned finite-data kernel surrogate, NOT an inferred spectral law or low-rank collapse of the original Transformer.',
      'identity':'Before final FP32 rounding, for fixed incoming embedding e: G_e(h)=h A_e+b_e; A_e=U V_e, U=Htrain.T/sH, V_e=diag((1+mix*kE)/(1+mix)) B, B=alpha*target_scale. b_e=(1+mix*kE/(1+mix)) B+mean. Actual prediction returns FP32; its numerical difference from the FP64 affine expression is measured.',
      'full_spectrum':'Nonzero eigenvalues of U V equal those of V U, plus D-N exact algebraic zero eigenvalues. This factor identity does not truncate any native coordinate or empirical singular value.',
      'cycles':'Every distinct minimal period<=3 exactly repeated in the last6 selected tokens, plus every distinct final incoming token among64 self-fed runs. No selection by favorable spectrum.',
      'greedy_guard':'Evaluate ALL151936 readout logits at each candidate fixed-cycle state. RMS scalar is positive, so own-normalized argmax regions are the same polyhedral inequalities of W*gamma*h. An affine fixed cycle only describes greedy dynamics if each chosen token is self-consistent.',
      'growth_diagnostic':'Also evaluate every real positive eigenvalue tied for maximal modulus, both signs of its complete native-width left eigenvector. Check homogeneous-cycle greedy guards and observed full-state norm growth/alignment. Full spectrum is still retained; this diagnostic is not a truncated hidden-state model. A growing direction can repeat tokens without a finite stable fixed state.',
      'retention':'Every spectrum including algebraic zeros, every full-coordinate fixed state, all token-dependent640-term coefficient weights; full native-coordinate operator entries are exactly reconstructible from retained frozen training states and alpha.'})
    records=[read(p) for p in sorted((BASE/'generation/commits').glob('*.json'))];cycles={};periods=[]
    for r in records:
        tokens=r['self_tokens'];p=next((p for p in (1,2,3) if len(tokens)>=6 and all(tokens[-6+i]==tokens[-6+i%p] for i in range(6))),None)
        if p is not None:
            cycle=tuple(tokens[-p:]);cycles.setdefault(cycle,[]).append(r['sample_id'])
        cycles.setdefault((tokens[-1],),[])
        periods.append({'sample_id':r['sample_id'],'minimal_observed_suffix_period':p,'last6_tokens':tokens[-6:]})
    ids=sorted({t for c in cycles for t in c});embed=embeddings(ids);rule=TemporalRule();H=rule.train['h36'].astype(float);E=rule.train['embedding'].astype(float);mix=rule.info['mix'];B=rule.model['alpha'].astype(float)*rule.model['target_scale'];U=H.T/rule.scales['h36'];factors={};identity=[]
    for tid in ids:
        k=embed[tid].astype(float)@E.T/rule.scales['embedding'];w=(1+mix*k)/(1+mix);c=1+mix*k/(1+mix);V=w[:,None]*B;b=c@B+rule.model['mean'];C=V@U;factors[tid]=(V,b,C)
        npz(out/f'token_coefficients/{tid}.npz',incoming_embedding_kernel=k,linear_kernel_weights=w,constant_kernel_weights=c)
        test=H[:3]
        explicit=test@U@V+b
        direct=np.stack([rule(h,embed[tid],[tid],'en') for h in test]);err=float(np.max(np.abs(explicit-direct)));assert err<2e-4;identity.append({'token_id':tid,'max_abs_native_width_formula_recompute_error':err})
    rd=Readout();reports=[]
    try:
      with torch.inference_mode():
       for number,(cycle,sources) in enumerate(sorted(cycles.items())):
        V,b,C=factors[cycle[0]];Vtotal=V.copy();btotal=b.copy();Ctotal=C.copy()
        for tid in cycle[1:]:
            VV,bb,CC=factors[tid];btotal=btotal@U@VV+bb;Vtotal=Ctotal@VV;Ctotal=Ctotal@CC
        assert np.max(np.abs(Vtotal@U-Ctotal))<2e-7
        eigen,vectors=eig(Ctotal.T,check_finite=False);full=np.concatenate([eigen,np.zeros(H.shape[1]-H.shape[0],complex)]);rho=float(np.max(np.abs(full)))
        condition=float(np.linalg.cond(np.eye(len(H))-Ctotal));fixed=None;residual=None;guards=[]
        if condition<1e10:
            coeff=solve((np.eye(len(H))-Ctotal).T,(btotal@U),check_finite=False);fixed=btotal+coeff@Vtotal;residual=float(np.linalg.norm(fixed@U@Vtotal+btotal-fixed)/max(np.linalg.norm(fixed),1e-30));assert residual<1e-6
            h=fixed.copy();phases=[]
            for tid in cycle:
                lp=rd.logprob(h[None])[0];selected=int(lp.argmax());target=float(lp[tid]);lp[tid]=-torch.inf;margin=target-float(lp.max());guards.append({'expected_token':tid,'actual_greedy_token':selected,'target_minus_best_other_logit':margin});phases.append(h.copy());VV,bb,_=factors[tid];h=h@U@VV+bb
            packet={'spectrum_real':full.real,'spectrum_imaginary':full.imag,'fixed_cycle_states':np.stack(phases),'constant_vector':btotal}
        else:packet={'spectrum_real':full.real,'spectrum_imaginary':full.imag,'constant_vector':btotal}
        rays=[];ray_states=[]
        for j,lam in enumerate(eigen):
          if lam.real>1 and abs(lam.imag)<1e-8*max(abs(lam),1) and abs(lam)>=rho*(1-1e-7):
            direction=vectors[:,j].real@Vtotal;direction/=np.linalg.norm(direction);ray_residual=float(np.linalg.norm(direction@U@Vtotal-lam.real*direction)/max(abs(lam),1))
            for sign in (1,-1):
                v=direction*sign;ray_states.append(v.copy());h=v.copy();gg=[];phase_directions=[]
                for tid in cycle:
                    phase_directions.append(h.copy())
                    lp=rd.logprob(h[None])[0];choice=int(lp.argmax());score=float(lp[tid]);lp[tid]=-torch.inf;gg.append({'expected_token':tid,'actual_greedy_token':choice,'margin':score-float(lp.max())});VV,_,_=factors[tid];h=h@U@VV;h/=np.linalg.norm(h)
                observed=[]
                for sid in sources:
                    with np.load(BASE/f'generation/fields/{sid}.npz') as z:hs=z['self_predicted_h36'].astype(float);native=unbits(z['native_h36_on_self_prefix']).astype(float)
                    norms=np.linalg.norm(hs,axis=1);period=len(cycle);observed.append({'sample_id':sid,'last_state_cosine_to_phase_aligned_direction':float(hs[-1]@phase_directions[-1]/max(norms[-1],1e-30)),
                      'last5_state_norm_ratio_median':float(np.median(norms[-5:]/np.maximum(norms[-6:-1],1e-30))),'last3_full_cycle_norm_ratio_median':float(np.median(norms[-3:]/np.maximum(norms[-3-period:-period],1e-30))),
                      'last_self_to_same_prefix_native_norm_ratio':float(norms[-1]/np.linalg.norm(native[-1]))})
                rays.append({'positive_eigenvalue':float(lam.real),'sign':sign,'full_coordinate_eigenvector_residual':ray_residual,'all_vocabulary_homogeneous_greedy_guards':gg,
                  'greedy_self_consistent':all(g['expected_token']==g['actual_greedy_token'] and g['margin']>0 for g in gg),'observed_full_state_checks':observed,
                  'limitation':'Asymptotic homogeneous-cycle candidate.16 steps may still be transient; norm growth and direction do not by themselves prove a basin or native Transformer dynamics.'})
        if ray_states:packet['complete_native_width_growth_directions']=np.stack(ray_states)
        npz(out/f'cycles/{number:03d}.npz',**packet)
        report={'cycle_id':number,'incoming_token_cycle':cycle,'observed_suffix_sources':sources,'period':len(cycle),'spectral_radius':rho,'spectral_modulus_above_one':int(np.sum(np.abs(full)>1+1e-8)),
          'fixed_solve_condition':condition,'relative_fixed_point_residual':residual,'all_vocabulary_greedy_guards':guards,'locally_attracting_affine_cycle':rho<1,
          'greedy_self_consistent':bool(guards) and all(g['expected_token']==g['actual_greedy_token'] and g['target_minus_best_other_logit']>0 for g in guards),'growing_projective_candidates':rays}
        reports.append(report);print('SURROGATE_CYCLE',number,cycle,rho,report['greedy_self_consistent'],flush=True)
    finally:rd.close()
    save(out/'observed_periods.json',periods);save(out/'result.json',{'timestamp':stamp(),'cycles':reports,'token_coefficient_identity_checks':identity,'token_ids':ids,'native_width':H.shape[1],'training_anchors':len(H),'mix':mix,'state_kernel_scale':rule.scales['h36'],
      'sources_with_observed_period_le3':sum(r['minimal_observed_suffix_period'] is not None for r in periods),'candidate_cycles':len(reports),
      'attracting_and_greedy_self_consistent_cycles':sum(r['locally_attracting_affine_cycle'] and r['greedy_self_consistent'] for r in reports),
      'cycles_with_greedy_consistent_growth_direction':sum(any(g['greedy_self_consistent'] for g in r['growing_projective_candidates']) for r in reports),
      'limits':'Only16-step observed runs. Spectra and fixed states describe the real-arithmetic affine map from stored fitted coefficients; implementation also rounds to FP32. No claim that all observed runs reached a basin, that the original LLM shares its spectrum, or that hidden-state rank decay caused language failure.'});guard(1024**2)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
