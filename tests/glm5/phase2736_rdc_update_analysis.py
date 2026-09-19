"""Source-group evidence, nonpunctuation response fields, and candidate limits."""
from collections import Counter
from rdc_update_common import *

def main():
    out=BASE/'analysis';start=time.monotonic();rows=gzread(BASE/'natural_material.json.gz')
    if (out/'phase2736.json').exists():return
    confirmed=read(BASE/'graph/confirmation.json');frozen=read(BASE/'graph/frozen.json')
    sample_index=[];covariances=[];h=[];targets={16:[],35:[]};identities=[]
    for row in rows:
        with np.load(native_path(row)) as z:
            h.append(unbits(z['H'])[12]);
            for b in targets:targets[b].append(unbits(z[f'L{b}_mlp']))
            identities.append({'sample_id':row['sample_id'],'shape':list(z['H'].shape),'source_tokens':len(z['H12_sources']),
              'arrays':len(z.files),'archive_sha256':sha(native_path(row))})
        for a,p in enumerate(row['anchors']):sample_index.append({'sample_id':row['sample_id'],'anchor':a,'position':p,'source_group':row['source_group'],
          'cohort':row['cohort'],'split':row['split'],'next_token_id':row['prompt_ids'][p+1],**row['content_boundary_annotations'][a]})
    assert len(sample_index)==384 and len({(r['sample_id'],r['position']) for r in sample_index})==384
    h=np.concatenate(h).astype(float);hc=h-h.mean(0)
    for b,v in targets.items():
        m=np.concatenate(v).astype(float);mc=m-m.mean(0)
        cov=hc.T@mc/len(h);corr=cov/np.sqrt(np.mean(hc*hc,0)[:,None]*np.mean(mc*mc,0)[None,:]).clip(1e-20)
        npz(out/f'new_natural_H12_to_block{b}_full_coordinates.npz',cross_covariance=cov,cross_correlation=corr,
          H12_mean=h.mean(0),MLP_mean=m.mean(0),H12_values=h.astype(np.float32),MLP_values=m.astype(np.float32))
        covariances.append({'block':b,'rows':len(h),'input_coordinates':2560,'output_coordinates':2560,'entries':2560**2,
          'center':'Allnew384 anchors descriptive center only, not used to fit/select any predictor','raw_max_abs_covariance':float(np.max(abs(cov)))})
    compressed(out/'natural_content_boundaries.json.gz',sample_index)
    selected=[r for r in confirmed['records'] if r['validation_selected']]
    controls=[]
    for b in (16,35):
        # Compare frozen winner to each query-only decoder at same128 df; this is
        # not a new outcome-based choice of predictor or metric.
        with np.load(BASE/'graph/selected_predictions'/f'b{b}.npz') as z:sel=z['squared_error'];den=z['baseline_squared_error']
        decoder=frozen['selected'][str(b)]['decoder']
        with np.load(BASE/'graph/confirmation_errors'/f'b{b}_query_128_{decoder}.npz') as z:base=z['squared_error']
        for split,cohort in sorted({(r['split'],r['cohort']) for r in sample_index}):
            ix=[i for i,r in enumerate(sample_index) if (r['split'],r['cohort'])==(split,cohort)];delta=(sel-base)/den.clip(1e-12)
            controls.append({'block':b,'decoder':decoder,'split':split,'cohort':cohort,'winner_minus_query_cluster':clustered(delta[ix],[sample_index[i]['source_group'] for i in ix]),
              'pooled_relative_error_difference':float((sel[ix]-base[ix]).sum()/den[ix].sum())})
    head=read(BASE/'graph/confirmation/head_evaluation_summary.json');counts=Counter(r['upos'] for r in sample_index)
    report={'timestamp':stamp(),'source':snapshot(__file__),'phase':2736,'natural_windows':128,'nonpunctuation_boundaries':384,
      'boundary_UPOS':dict(counts),'source_documents':len({r['source_group'] for r in rows}),'identities':identities,
      'selected_predictors':selected,'query_controls':controls,'full_coordinate_relations':covariances,'head_readout':head,
      'training_evidence':'Recomputed original/FP32-BF16 bridge/continued-training comparisons from2733, see contract.bridge_reanalysis; no new training claimed in this phase.',
      'limits':['Head mapping accuracy is limited and labels can depend on full-sentence context; visible endpoints do not ensure an online-unique parse.',
        'Natural training uses earlier end anchors; new content boundaries shift target-position distribution. This is measured generalization, not a controlled estimate of only new semantics.',
        'RMS-selected directed candidate has no matched-RMS head-shuffle candidate in the original freeze. Raw shuffled advantage does not establish binding for theRMSwinner. Add matched controls as explicitly new diagnostics and prospective follow-up; do not rewrite this selection.',
        'All-coordinate directed tensor retains more typed interaction than a mean, but not necessarily sufficient autoregressive state. Query/position/lexical baselines constrain semantic interpretation.',
        'These are fitted source operators, not actual attention probabilities. Native source-to-parameter accounting is separately pursued.'],
      'seconds':time.monotonic()-start}
    save(out/'phase2736.json',report);ledger('full_coordinate_source_analysis',report['seconds']);print('PHASE2736_ANALYSIS',controls,flush=True)

if __name__=='__main__':main()
