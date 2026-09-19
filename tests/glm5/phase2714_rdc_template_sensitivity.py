"""Post-hoc audit of numerical boilerplate across source splits; frozen results preserved."""
from collections import defaultdict
from phase2714_rdc_source_kernels import dataset
from rdc_prefix_estimators import *
OUT=CAMPAIGN/'full_source_history'


def canonical(s):return re.sub(r'\W','',re.sub(r'\d+(?:[.,]\d+)*','NUM',s.casefold()))


def main():
    material=read(CAMPAIGN/'material_stratified.json');prior=material+read(CAMPAIGN/'confirmation_material.json');fresh=read(OUT/'fresh_material.json')
    train_sigs={canonical(r['text']) for r in material if r['split']=='train'};groups=defaultdict(list)
    for r in prior:groups[canonical(r['text'])].append(r)
    overlaps=[{'fresh':r,'prior':groups[canonical(r['text'])]} for r in fresh if canonical(r['text']) in groups]
    main_groups=defaultdict(list)
    for r in material:main_groups[canonical(r['text'])].append(r)
    crossing=[rs for rs in main_groups.values() if len({r['split'] for r in rs})>1]
    main_excluded={r['sample_id'] for r in material if r['split'] in ('validation','test') and canonical(r['text']) in train_sigs}
    fresh_excluded={r['fresh']['sample_id'] for r in overlaps};metrics=[]
    # Never refit or change a frozen selection. Compare the same stored predictions after excluding flagged source units.
    for scope,rows,excluded in [('test',read(OUT/'main_rows.json'),main_excluded),('fresh',read(OUT/'fresh_rows.json'),fresh_excluded)]:
        ii=splits(rows)[2] if scope=='test' else np.arange(len(rows));selected=[rows[i] for i in ii];keep=np.array([r['sample_id'] not in excluded for r in selected])
        for rule in ('current','mean_history','absolute_history','relative_history'):
            with np.load(OUT/f'predictions/{scope}_{rule}.npz') as z:mse=z['row_mse']
            pr=read(OUT/f'probability/{scope}_{rule}.json')['rows'];kp=[r for r in pr if r['sample_id'] not in excluded]
            metrics.append({'scope':scope,'rule':rule,'kept_source_units':len({r['sample_id'] for r,k in zip(selected,keep) if k}),
              'kept_anchors':int(keep.sum()),'MSE_frozen_predictions':float(mse[keep].mean()),'KL':float(np.mean([r['KL'] for r in kp])),
              'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in kp]))})
    # Main2712 early/graph/quadratic sensitivity uses exactly the same frozen predictions and source exclusions.
    oldrows=read(CAMPAIGN/'shared_rules/qwen4/rows.json');ii=splits(oldrows)[2];keep=np.array([oldrows[i]['sample_id'] not in main_excluded for i in ii]);oldmetrics=[]
    for name in ('early_linear','full_linear','full_quadratic','graph_interaction','graph_interaction_df128'):
        with np.load(CAMPAIGN/f'shared_rules/predictions/{name}.npz') as z:mse=z['H36_row_mse']
        oldmetrics.append({'model':name,'kept_anchors':int(keep.sum()),'H36_MSE':float(mse[keep].mean())})
    # Score already-frozen2714 candidates on validation excluding the matched boilerplate; no new lambda selection.
    _,rows,_,_,target=dataset(False);tr,va,te=splits(rows);valid=np.array([i for i in va if rows[i]['sample_id'] not in main_excluded]);validation=[]
    with np.load(OUT/'input_grams.npz') as grams:
        for name in ('current','mean_history','absolute_history','relative_history'):
            with np.load(OUT/f'models/{name}.npz') as z:
                pred=(grams[name][np.ix_(valid,tr)].astype(float)@z['alpha'])*z['target_scales']+z['means']
                validation.append({'rule':name,'anchors':len(valid),'frozen_lambda_validation_normalized_MSE':float(np.mean(((pred-target[valid])/z['target_scales'])**2))})
    result={'timestamp':stamp(),'phase':2714,'source_sha':sha(Path(__file__)),
      'definition':'Casefold, replace numerical runs including decimal/comma runs by NUM, remove non-word characters; exact canonical-string equality. This identifies numeric boilerplate only, not all paraphrase or semantic family overlap.',
      'fresh_overlaps':overlaps,'main_cross_split_groups':crossing,'main_validation_test_excluded':sorted(main_excluded),'fresh_excluded':sorted(fresh_excluded),
      'source_kernel_sensitivity':metrics,'prior_main_prediction_sensitivity':oldmetrics,'frozen_validation_candidate_sensitivity':validation,
      'old_confirmation128_numeric_template_matches_to_main':sum(canonical(r['text']) in {canonical(x['text']) for x in material} for r in read(CAMPAIGN/'confirmation_material.json')),
      'correction':'Source-group and normalized-text disjointness does not guarantee unseen construction families. One Chinese numerical town-statistics template spans original train/validation/test and fresh. Preserve all official reports, add this post-hoc sensitivity, and do not rebrand the filtered subsets as prospectively independent family-heldout experiments.',
      'selection_changed':False,'refitting':False,'primary_frozen_results_preserved':True,'next_requirement':'Prospectively group normalized numeric boilerplate and known rewrite/entity families before selecting new material; control lexical identity/POS beyond exact signatures.'}
    save(OUT/'template_sensitivity.json',result)
    print('TEMPLATE_SENSITIVITY',sorted(main_excluded),sorted(fresh_excluded),metrics,validation,flush=True)


if __name__=='__main__':main()
