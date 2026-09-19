"""Paired whole-context uncertainty; all registered targets/routes retained."""
import argparse
from rdc_question_common import *


def bootstrap(rows, left, right, seed):
    cohort_results, distributions = {}, {}
    for cohort in ['drop','quoref']:
        ids = sorted({r['group_id'] for r in rows if r['cohort'] == cohort})
        positions = [[i for i,r in enumerate(rows) if r['group_id'] == gid] for gid in ids]
        assert all(len(v) == 4 for v in positions)
        group_delta = np.array([(left[take]-right[take]).mean() for take in positions])
        rng = np.random.default_rng(seed+(0 if cohort == 'drop' else 1))
        take = rng.integers(0,len(ids),size=(2000,len(ids)))
        draws = group_delta[take].mean(1)
        distributions[cohort] = draws
        cohort_results[cohort] = {'contexts':len(ids),'questions':4*len(ids),
            'mean_left_minus_right':float(group_delta.mean()),
            'paired_context_bootstrap_95_percent_interval':np.quantile(draws,[.025,.975]).tolist(),
            'context_delta_minmax': [float(group_delta.min()),float(group_delta.max())]}
    draws = (distributions['drop']+distributions['quoref'])/2
    cohort_results['equal_cohort'] = {'mean_left_minus_right':float(np.mean([r['mean_left_minus_right'] for r in cohort_results.values()])),
        'paired_context_bootstrap_95_percent_interval':np.quantile(draws,[.025,.975]).tolist()}
    return cohort_results


def main(key):
    start = time.monotonic()
    folder = Path('fit')/key
    resultpath = OUT/folder/'paired_analysis.json'
    if resultpath.exists():
        assert read(resultpath)['source']['sha256'] == sha(__file__)
        print('NATURAL_FIT_PAIRED_ALREADY_COMPLETE',key,flush=True)
        return
    completed = read(OUT/folder/'result.json')
    assert completed['all_passed']
    selection = read(OUT/folder/'validation_selection.json')
    primary = selection['primary_rule']
    records, summaries, matrices = {}, [], {}
    for variant in completed['variants']:
        result = read(OUT/folder/variant/'evaluation.json')
        records[variant] = result
        for row in result['evaluations']:
            ident = (variant,row['kind'],row['target'],row['split'])
            ref = row['field']
            assert sha(ROOT/ref['path']) == ref['sha256']
            with np.load(ROOT/ref['path']) as z:
                matrices[ident] = {k:z[k].copy() for k in z.files if k.endswith('_by_question')}
            summaries.append({'variant':variant,**{k:row[k] for k in ['kind','target','split','summary']}})
    comparisons = []
    targets = completed['full_targets']
    # Every target/control is reported; no effect-size-filtered comparison list.
    for split in ['validation','diagnostic']:
        rows = records[primary][split]
        assert all(records[v][split] == rows for v in completed['variants'])
        for target in targets:
            for variant in completed['variants']:
                left = matrices[(variant,'selected',target,split)]
                right = matrices[(variant,'alpha0',target,split)]
                for metric in ['absolute_MSE_by_question','within_MSE_by_question']:
                    comparisons.append({'left':variant+':selected','right':variant+':alpha0',
                        'target':target,'split':split,'metric':metric,
                        'paired':bootstrap(rows,left[metric],right[metric],2748005)})
            left = matrices[(primary,'selected',target,split)]
            for control in [v for v in completed['variants'] if v != primary]+['zero_response_change']:
                for metric in ['absolute_MSE_by_question','within_MSE_by_question']:
                    if control == 'zero_response_change' and metric.startswith('absolute'):
                        continue
                    right = left['zero_change_MSE_by_question'] if control == 'zero_response_change' else matrices[(control,'selected',target,split)][metric]
                    comparisons.append({'left':primary+':selected','right':control,'target':target,
                        'split':split,'metric':metric,'paired':bootstrap(rows,left[metric],right,2748005)})
    result = {'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,
        'fit_result_sha256':sha(OUT/folder/'result.json'),'selection_sha256':sha(OUT/folder/'validation_selection.json'),
        'primary_rule':primary,'summaries':summaries,'paired_comparisons':comparisons,
        'bootstrap':{'resamples':2000,'unit':'All4questions of one unchanged context; within-cohort independent resampling, arithmetic mean of cohort estimates.',
            'seeds':[2748005,2748006],'paired':True,'interval':'Percentile95percent, descriptive and not familywise/multiplicity adjusted.',
            'validation_scope':'Selection-set intervals are post-selection descriptive; diagnostic intervals hold the validation choices fixed.',
            'not_estimated':'Article/pretraining overlap not detected by retained grouping, model-training randomness, or alternate language-task coverage.'},
        'remaining':'Native BF16full-vocabulary readout, own histories, other models and training effects separately required.',
        'seconds':time.monotonic()-start}
    immutable(resultpath,result)
    print('NATURAL_FIT_PAIRED_COMPLETE',key,len(comparisons),round(result['seconds'],1),flush=True)
    for item in comparisons:
        if item['target']=='postnorm' and item['split']=='diagnostic' and item['left']==primary+':selected' and item['metric']=='within_MSE_by_question':
            print('NATURAL_FIT_DIAGNOSTIC_WITHIN',item['right'],item['paired']['equal_cohort'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True)
    main(parser.parse_args().model)
