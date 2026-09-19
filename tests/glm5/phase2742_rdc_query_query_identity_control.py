"""Exploratory query-identity control for the observed high native-model Gram correlations."""
from rdc_query_common import *


def compare(grams):
    names=list(grams);upper=np.triu_indices(100,1);rows=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            A=grams[a];B=grams[b]
            rows.append({'models':[a,b],'entry_correlation':float(np.corrcoef(A[upper],B[upper])[0,1]),
              'unit_Frobenius_difference':float(np.linalg.norm(A/np.linalg.norm(A)-B/np.linalg.norm(B)))})
    return rows


def main():
    out=BASE/'analysis/query_identity_control.json'
    if out.exists():return
    start=time.monotonic();source=snapshot(__file__);a=read(BASE/'analysis/phase2742.json');assert a['all_passed']
    ids=a['scale']['common_source_ids'];assert len(ids)==16;names=['qwen4','qwen14','glm4'];kernels={k:{} for k in ['native','query_alone','history_minus_alone','between_history']}
    checks=[];within=[]
    with np.load(BASE/'analysis/matched_model_query_geometry.npz') as z:old={m:z[m+'__centered_Gram'].copy() for m in names}
    for model in names:
        with np.load(BASE/'scale'/model/'query_only.npz') as z:alone=unbits(z['postnorm']).astype(float)
        responses=[]
        for sid in ids:
            with np.load(BASE/'scale'/model/'fields'/f'{sid}.npz') as z:responses.append(unbits(z['postnorm']).astype(float))
        H=np.stack(responses);D=H.shape[-1]
        def kernel(values):
            values=values-values.mean(-2,keepdims=True)
            return sum(row@row.T/D for row in values)/len(values)
        kernels['native'][model]=kernel(H)
        kernels['query_alone'][model]=kernel(alone[None])
        kernels['history_minus_alone'][model]=kernel(H-alone[None])
        kernels['between_history'][model]=kernel(H-H.mean(0,keepdims=True))
        error=float(abs(kernels['native'][model]-old[model]).max());assert error<1e-10
        checks.append({'model':model,'original_primary_Gram_reconstruction_max_error':error,'documents':len(ids),'native_width':D})
        within.append({'model':model,'native_versus_query_alone':compare({'native':kernels['native'][model],'query_alone':kernels['query_alone'][model]})[0],
          'trace_energy':{k:float(np.trace(v[model])) for k,v in kernels.items()}})
    npz(BASE/'analysis/full_query_identity_control_grams.npz',**{kind+'__'+model:g for kind,models in kernels.items() for model,g in models.items()})
    result={'timestamp':stamp(),'source':source,'all_passed':True,'design_status':'Exploratory after observing high cross-model native Gram correlations; same16documents, not a new independent confirmation.',
      'definitions':{'native':'Average over histories of query-centered H(S,q) Gram /D.',
        'query_alone':'Query-centered independent H(q alone) Gram /D.',
        'history_minus_alone':'Average query-centered Gram of H(S,q)-H(q alone); removes the observed independent-query vector, not a unique semantic or linear additive component.',
        'between_history':'Average query-centered Gram of H(S,q)-mean_S H(S,q); isolates variation among the16observed histories after removing their common query profile. Not a population variance estimate.'},
      'checks':checks,'between_model_comparisons':{kind:compare(grams) for kind,grams in kernels.items()},'within_model_controls':within,
      'native_precision_scope':'All native BF16 values exactly decoded to FP64 for these complete-coordinate statistics; no native inference, fitted transfer, coordinate alignment, PCA or TopK.',
      'interpretation':'High original Gram correlation alone can reflect shared query identity or generic query effects. Residual/among-history correlation measures a different descriptive object; subtraction is not a causal semantic decomposition and no resulting kernel proves isomorphism.',
      'artifact':'analysis/full_query_identity_control_grams.npz','seconds':time.monotonic()-start}
    save(out,result);ledger('exploratory_cross_model_query_identity_control',result['seconds']);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
