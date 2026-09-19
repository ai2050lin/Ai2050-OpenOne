"""Full-coordinate within-language controls for the visible two-language Gram blocks."""
from rdc_query_common import *
from phase2742_rdc_query_query_identity_control import compare


def main():
    out=BASE/'analysis/query_language_control.json'
    if out.exists():return
    start=time.monotonic();a=read(BASE/'analysis/phase2742.json');ids=a['scale']['common_source_ids'];probes=read(BASE/'probes/protocol.json')['probes']
    groups={lang:[i for i,q in enumerate(probes) if q['language']==lang] for lang in ['en','zh']};assert all(len(v)==50 for v in groups.values())
    kernels={};energy=[];archives={}
    for model in ['qwen4','qwen14','glm4']:
        with np.load(BASE/'scale'/model/'query_only.npz') as z:alone=unbits(z['postnorm']).astype(float)
        H=[]
        for sid in ids:
            with np.load(BASE/'scale'/model/'fields'/f'{sid}.npz') as z:H.append(unbits(z['postnorm']).astype(float))
        H=np.stack(H);D=H.shape[-1];mean=H.mean(1,keepdims=True);B=np.empty_like(H);W=np.empty_like(H)
        for lang,ix in groups.items():
            groupmean=H[:,ix].mean(1,keepdims=True);B[:,ix]=groupmean-mean;W[:,ix]=H[:,ix]-groupmean
            for name,value in [('native',H),('history_minus_alone',H-alone[None]),('between_history',H-H.mean(0,keepdims=True))]:
                x=value[:,ix];x=x-x.mean(1,keepdims=True);g=sum(v@v.T/D for v in x)/len(x)
                key=lang+'__'+name;kernels.setdefault(key,{})[model]=g;archives[key+'__'+model]=g
        total=float(np.sum((H-mean)**2));between=float(np.sum(B**2));within=float(np.sum(W**2))
        error=abs(total-between-within)/total;assert error<1e-12
        energy.append({'model':model,'between_language_fraction_of_query_centered_coordinate_energy':between/total,
          'within_language_fraction':within/total,'scalar_energy_decomposition_relative_error':error})
    reports={}
    # Same comparison as the100query utility, now with50x50matrices.
    for kind,models in kernels.items():
        rows=[];names=list(models);ix=np.triu_indices(50,1)
        for i,m in enumerate(names):
            for n in names[i+1:]:
                A=models[m];B=models[n];rows.append({'models':[m,n],'entry_correlation':float(np.corrcoef(A[ix],B[ix])[0,1]),
                  'unit_Frobenius_difference':float(np.linalg.norm(A/np.linalg.norm(A)-B/np.linalg.norm(B)))})
        reports[kind]=rows
    npz(BASE/'analysis/full_within_language_grams.npz',**archives)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
      'design_status':'Exploratory prompted by the visible EN/ZH block structure after main results; same16sources, no new independent confirmation.',
      'scope':'Every native coordinate and all50queries in each language retained. Center within each language; no cross-model coordinate alignment, PCA or TopK. Query-language energy fraction is a scalar sum-of-squares identity, not semantic causal attribution and not an additive full-Gram decomposition.',
      'query_indices':groups,'energy':energy,'within_language_comparisons':reports,
      'definitions':'B_s,q=mean_{r in language(q)}H_s,r - mean_r H_s,r; W_s,q=H_s,q-mean_{r in language(q)}H_s,r. Total scalar squared energy equals ||B||^2+||W||^2. Gram entries include cross terms, so only scalar energy is partitioned.',
      'seconds':time.monotonic()-start}
    save(out,result);ledger('exploratory_within_language_query_control',result['seconds']);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
