"""Stronger source-aware nuisance controls for full-parameter gradient relationships."""
from collections import defaultdict
from rdc_law_common import *


def main():
    out=BASE/'formation/gradient_controls';start=time.monotonic()
    if (out/'result.json').exists():return
    panel=read(BASE/'formation/protocol.json')['panel'];n=len(panel)
    with np.load(BASE/'formation/initial_stable/complete_gradient_factors.npz') as z:gram=z['gram_total']
    scale=np.sqrt(np.maximum(np.diag(gram),1e-30));cos=gram/(scale[:,None]*scale[None,:])
    types=sorted({t for r in panel for t in r['relation_types']}-{'punct','root','det','case'})
    labels=np.array([[int(t in r['relation_types']) for t in types] for r in panel],dtype=float)
    known=labels.sum(1)>0
    base=np.array([[i!=j and a['source_group']!=b['source_group'] and a['target_id']==b['target_id'] and
        a['language']==b['language'] and a['prefix_tokens']//32==b['prefix_tokens']//32 for j,b in enumerate(panel)] for i,a in enumerate(panel)])
    masks={'target_language_position':base,
        'also_cohort':base & np.array([[a['cohort']==b['cohort'] for b in panel] for a in panel]),
        'also_cohort_current_token':base & np.array([[a['cohort']==b['cohort'] and a['current_token_id']==b['current_token_id'] for b in panel] for a in panel])}
    def evaluate(lab,mask):
        inter=lab@lab.T;union=lab.sum(1)[:,None]+lab.sum(1)[None,:]-inter
        high=inter/np.maximum(union,1)>=.5
        valid=mask & known[:,None]&known[None,:]
        hm=valid&high;lm=valid&~high;hn=hm.sum(1);ln=lm.sum(1);ok=(hn>0)&(ln>0)
        gain=(cos*hm).sum(1)/np.maximum(hn,1)-(cos*lm).sum(1)/np.maximum(ln,1)
        groups=defaultdict(list)
        for i in np.flatnonzero(ok):groups[panel[i]['source_group']].append(gain[i])
        values=np.array([np.mean(v) for v in groups.values()])
        return (float(values.mean()) if len(values) else None),gain,ok,int(hm.sum()),int(lm.sum())
    reports=[];permutations={}
    for name,mask in masks.items():
        observed,gain,ok,hp,lp=evaluate(labels,mask)
        keys=defaultdict(list)
        for i,r in enumerate(panel):
            if not known[i]:continue
            key=(r['target_id'],r['language'],r['prefix_tokens']//32)
            if name!='target_language_position':key+=(r['cohort'],)
            if name=='also_cohort_current_token':key+=(r['current_token_id'],)
            keys[key].append(i)
        null=[];rng=np.random.default_rng(272800)
        for repeat in range(256):
            perm=np.arange(n)
            for ids in keys.values():perm[ids]=rng.permutation(ids)
            mean,*_=evaluate(labels[perm],mask);null.append(np.nan if mean is None else mean)
        finite=np.asarray(null)[np.isfinite(null)]
        packet={'control':name,'queries_with_both_comparisons':int(ok.sum()),'high_ordered_pairs':hp,'low_ordered_pairs':lp,
            'source_cluster':clustered(gain[ok],[panel[i]['source_group'] for i in np.flatnonzero(ok)]),
            'null_finite_permutations':len(finite),'permutable_queries':sum(len(v) for v in keys.values() if len(v)>1),
            'conditional_label_shuffle_tail_fraction':float((1+np.sum(finite>=observed))/(len(finite)+1)) if observed is not None and len(finite) else None,
            'null_mean':float(finite.mean()) if len(finite) else None,
            'scope':'Exploratory sensitivity analysis after initial association. Conditional label permutations are not a causal experiment or a confirmatory population p-value; graph endpoint labels may depend on full sentences, and exchangeability across source clusters is not established.'}
        reports.append(packet);permutations[name]=np.asarray(null)
    npz(out/'conditional_permutations.npz',**permutations)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'panel_queries':n,'reports':reports,
        'limits':['All pair relations are dependent; no independent-pair significance claim.','Same-target strata can be dominated by punctuation and common tokens.','Stronger exact controls may leave too few comparisons; missing evidence is reported rather than generalized.'],
        'seconds':time.monotonic()-start}
    save(out/'result.json',result);ledger('gradient_nuisance_sensitivity',result['seconds'])
    print('LAW_GRADIENT_CONTROLS_COMPLETE',reports,flush=True)


if __name__=='__main__':main()
