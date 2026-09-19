"""Full-coordinate direction and ordered branch-pair audits of retained factors."""
from rdc_query_common import *

def direction():
    from rdc_update_common import sources
    rows=gzread(PRIOR/'natural_material.json.gz')
    chosen=[]
    for cohort in ('gum','ewt'):
        chosen.extend(sorted([r for r in rows if r['cohort']==cohort],key=lambda r:rank(r['sample_id']))[:8])
    with np.load(PRIOR/'graph/head_mapping.npz') as z:coef=z['coefficients']
    packs=[]
    for r in chosen:
        h=sources(r)[:r['anchors'][-1]+1].astype(np.float64);h=h/rms(h)
        pred=np.c_[h,np.ones(len(h))]@coef;pred/=rms(pred)
        score=pred@h.T/(.15*h.shape[1]);np.fill_diagonal(score,-np.inf)
        aa=np.exp(score-score.max(1,keepdims=True));aa/=aa.sum(1,keepdims=True)
        packs.append((h,aa))
    n=len(packs);gram=np.zeros((n,n));one=np.zeros_like(gram);both=np.zeros_like(gram)
    for i,(hi,ai) in enumerate(packs):
      for j,(hj,aj) in enumerate(packs):
        dot=hi@hj.T;den=len(hi)*len(hj)*hi.shape[1]**2
        gram[i,j]=np.sum((ai.T@dot@aj)*dot)/den
        one[i,j]=np.sum((ai@dot@aj)*dot)/den
        both[i,j]=np.sum((ai@dot@aj.T)*dot)/den
    sym=(gram+one)/2;anti=(gram-one)/2
    assert np.max(abs(both-gram))<1e-10
    assert np.linalg.eigvalsh(sym).min()>-1e-9 and np.linalg.eigvalsh(anti).min()>-1e-9
    # Independent full2560x2560 materialization, not a reduced-coordinate toy.
    hi,ai=packs[0];hj,aj=packs[1];ti=hi.T@ai@hi/(len(hi)*hi.shape[1]);tj=hj.T@aj@hj/(len(hj)*hj.shape[1])
    dense={'gram_error':abs(float(np.sum(ti*tj))-gram[0,1]),'single_transpose_error':abs(float(np.sum(ti.T*tj))-one[0,1]),
      'orthogonal_sym_anti_inner':float(np.sum(((ti+ti.T)/2)*((tj-tj.T)/2)))}
    assert max(abs(v) for v in dense.values())<1e-9
    npz(BASE/'algebra/direction_full_coordinate.npz',gram=gram,single_transpose=one,common_transpose=both,symmetric_gram=sym,antisymmetric_gram=anti,
        native_T_first=ti.astype(np.float32),native_T_second=tj.astype(np.float32))
    return {'samples':[r['sample_id'] for r in chosen],'full_coordinate_width':2560,'common_transpose_max_abs_error':float(abs(both-gram).max()),
      'single_transpose_max_abs_change':float(abs(one-gram).max()),'antisymmetric_energy_fraction':(np.diag(anti)/np.diag(gram).clip(1e-30)).tolist(),
      'dense_independent_check':dense,
      'correction':'Common orientation reversal is unidentifiable from this Gram alone, but its antisymmetric Gram can retain relative orientation information. Single-sample reversal need not be invariant. Neither fact identifies a causal language arrow.'}

def ordered():
    prior=read(PRIOR/'native_paths/result.json');reports=[]
    for row in prior['reports']:
        path=PRIOR/'native_paths/fields'/f'{row["sample_id"]}.npz';assert sha(path)==row['archive_sha256']
        arrays={}
        with np.load(path) as z:
          for b in (16,35):
            for a,pos in enumerate(row['positions']):
                g=np.concatenate([z[f'L{b}_source_gate_read'][a,:pos+1],z[f'L{b}_other_gate_read'][a,None]],axis=0).astype(np.float64)
                u=np.concatenate([z[f'L{b}_source_up_read'][a,:pos+1],z[f'L{b}_other_up_read'][a,None]],axis=0).astype(np.float64)
                nativeg=z[f'L{b}_gate'][a].astype(np.float64);nativeu=z[f'L{b}_up'][a].astype(np.float64)
                sig=1/(1+np.exp(-nativeg));gsq=(g*g).sum(0);usq=(u*u).sum(0);gu=(g*u).sum(0)
                total=sig**2*gsq*usq;symmetric=.5*sig**2*(gsq*usq+gu**2);antisymmetric=.5*sig**2*(gsq*usq-gu**2)
                assert antisymmetric.min()>-1e-5*total.max()
                pair_unit_sum=(g*sig)@u.T
                pair_all_unit_energy=(g*g*sig**2)@(u*u).T
                contracted=sig*g.sum(0)*u.sum(0);expected=sig*nativeg*nativeu
                err=float(np.linalg.norm(contracted-expected)/max(np.linalg.norm(expected),1e-12));assert err<1e-5
                original=.5*sig[None,:]*(g*nativeu[None,:]+u*nativeg[None,:])
                from_pairs=.5*sig[None,:]*(g*u.sum(0)[None,:]+u*g.sum(0)[None,:])
                marginal_err=float(np.linalg.norm(from_pairs-original)/max(np.linalg.norm(original),1e-12));assert marginal_err<1e-5
                prefix=f'L{b}_a{a}_'
                arrays.update({prefix+'ordered_pair_sum_over_units':pair_unit_sum,prefix+'ordered_pair_energy_all_units':pair_all_unit_energy,
                  prefix+'unit_total_pair_energy':total,prefix+'unit_symmetric_pair_energy':symmetric,prefix+'unit_antisymmetric_pair_energy':antisymmetric,
                  prefix+'unit0_directed_pairs':sig[0]*g[:,0,None]*u[None,:,0]})
                reports.append({'sample_id':row['sample_id'],'source_group':row['source_group'],'cohort':row['cohort'],'block':b,'anchor':a,'position':pos,
                  'all_visible_sources_plus_other':len(g),'all_MLP_units':len(sig),'native_factor_file':str(path),'native_factor_sha256':row['archive_sha256'],
                  'pair_sum_relative_error':err,'old_symmetric_marginal_relative_error':marginal_err,
                  'total_antisymmetric_pair_energy_fraction':float(antisymmetric.sum()/total.sum()),
                  'fractional_energy_scope':'A Frobenius identity in branch-pair allocation space, not a semantic/causal contribution proportion. Antisymmetric terms cancel in the all-source double sum.'})
        npz(BASE/'algebra/ordered_pairs'/f'{row["sample_id"]}.npz',**arrays)
    return reports

def main():
    path=BASE/'algebra/result.json'
    if path.exists():return
    start=time.monotonic();guard(140*1024**2);d=direction();pairs=ordered()
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'direction':d,'ordered_pair_records':pairs,
      'native_samples':24,'anchor_block_records':len(pairs),'seconds':time.monotonic()-start,
      'method':'All source/coordinate/unit factors, exact contractions; no new native-model outcomes and no selection by correctness. Old source factors and parameter references stay byte-identical.',
      'formulas':['T=S+A; <T_i,T_j>=<S_i,S_j>+<A_i,A_j>; <T_i^T,T_j>=<S_i,S_j>-<A_i,A_j>',
        'a_sr,k=sigmoid(g_k)*g_s,k*u_r,k; sum_sr a_sr,k=SiLU(g_k)*u_k',
        '||A_antisym,k||_F^2=.5*sigmoid(g_k)^2*(||g_.k||^2*||u_.k||^2-<g_.k,u_.k>^2)'],
      'scope':'Recovery of directional information hidden by symmetric marginalization is a more detailed bookkeeping object, not proof of unique native semantic causation.'}
    save(path,result);ledger('full_coordinate_direction_and_native_ordered_pair_audit',result['seconds']);print('QUERY_ALGEBRA_PASS',len(pairs),d['single_transpose_max_abs_change'],flush=True)

if __name__=='__main__':main()
