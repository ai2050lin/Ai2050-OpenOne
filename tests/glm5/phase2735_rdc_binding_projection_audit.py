"""Full train-span spectrum audit; do not select a new predictor from test results."""
from rdc_binding_common import *


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2);start=time.monotonic();out=BASE/'format_content'
    result=read(out/'decomposition_result.json');rows=gzread(BASE/'program_material.json.gz')
    ix=[i for i,r in enumerate(rows) if r['split']=='train' and r['representation']=='python']
    with np.load(out/'all_parameter_gram_decomposition.npz') as z:g=z['content'][np.ix_(ix,ix)]
    g=(g+g.T)*.5;eig=np.linalg.eigvalsh(g);norm=np.sqrt(np.maximum(np.diag(g),0))
    threshold=result['projection']['relative_numerical_tolerance']*eig[-1];keep=eig>threshold
    normalized=np.divide(g,norm[:,None]*norm[None,:],out=np.zeros_like(g),where=(norm[:,None]*norm[None,:])>0)
    normalized_eig=np.linalg.eigvalsh((normalized+normalized.T)*.5)
    npz(out/'full_training_projection_spectrum.npz',training_indices=np.array(ix),raw_gram=g,
      raw_eigenvalues=eig,gradient_norms=norm,row_normalized_gram=normalized,row_normalized_eigenvalues=normalized_eig,
      retained_by_declared_cutoff=keep)
    assert int(keep.sum())==result['projection']['numerical_rank']
    report={'timestamp':stamp(),'source':snapshot(Path(__file__)),'training_rows':len(ix),
      'input_gram_sha256':sha(out/'all_parameter_gram_decomposition.npz'),'all_training_gradient_norms_nonzero':bool((norm>0).all()),
      'raw_gradient_norm_range':[float(norm.min()),float(norm.max())],
      'declared_relative_cutoff':result['projection']['relative_numerical_tolerance'],'absolute_eigenvalue_cutoff':float(threshold),
      'retained_numerical_rank':int(keep.sum()),'below_cutoff_count':int((~keep).sum()),
      'below_cutoff_positive_eigenvalue_count':int(((eig>0)&~keep).sum()),
      'largest_below_cutoff_eigenvalue':float(eig[~keep].max()),'smallest_retained_eigenvalue':float(eig[keep].min()),
      'row_normalized_diagnostic_rank_at_same_relative_cutoff':int((normalized_eig>normalized_eig[-1]*1e-9).sum()),
      'all_declared_features_and_gram_entries_preserved':True,'new_direction_fitted_or_deployed':False,
      'scope':'The inverse uses a declared numerical eigenvalue cutoff, not a proof of exact rank or semantic irrelevance of weak directions. Full native parameters and all96 gradients/Gram entries are retained. Row normalization is only a conditioning diagnostic; it is not used to refit or select a direction after seeing heldout results.',
      'seconds':time.monotonic()-start}
    save(out/'projection_condition_audit.json',report);ledger('content_gradient_projection_condition_audit',report['seconds'])
    print('CONTENT_PROJECTION_CONDITION',report,flush=True)


if __name__=='__main__':main()
