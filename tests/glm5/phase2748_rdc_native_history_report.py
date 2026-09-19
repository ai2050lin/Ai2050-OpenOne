"""Describe fixed native-history predictor diagnostics without further fitting."""
import argparse
import json
from collections import defaultdict
from rdc_question_common import *
from phase2748_rdc_native_history_prediction import COLUMNS, BIN_NAMES


def main(key):
    folder=OUT/'native_history_prediction'/key
    final=folder/'descriptive_report_v2.json'
    if final.exists():
        assert read(final)['source']['sha256']==sha(__file__)
        print('NATURAL_HISTORY_DESCRIPTIVE_ALREADY_COMPLETE',key,flush=True);return
    result=read(folder/'result.json');assert result['all_passed']
    selection=read(OUT/'fit'/key/'validation_selection.json')
    summaries=[]
    for record in result['records']:
        ref=record['question_bins'];assert sha(ROOT/ref['path'])==ref['sha256']
        rows=gzread(ROOT/ref['path'])
        for name in BIN_NAMES:
            by_cohort={}
            for cohort in ['drop','quoref']:
                rr=[r for r in rows if r['bin']==name and r['cohort']==cohort]
                grouped=defaultdict(list)
                for row in rr:grouped[row['group_id']].append(row)
                if not rr:continue
                context_means={c:np.array([np.mean([r['means'][c]for r in gg])for gg in grouped.values()])for c in COLUMNS}
                by_cohort[cohort]={c:float(v.mean())for c,v in context_means.items()}
                by_cohort[cohort].update({'contexts':len(grouped),'questions':len(rr),'tokens':sum(r['tokens']for r in rr)})
            if len(by_cohort)==2:
                by_cohort['equal_cohort']={c:float(np.mean([by_cohort[v][c]for v in ['drop','quoref']]))for c in COLUMNS}
            summaries.append({'variant':record['variant'],'split':record['split'],'bin':name,'summary':by_cohort})
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,
        'prediction_result_sha256':sha(folder/'result.json'),'primary_rule':selection['primary_rule'],
        'control_rule':selection['control_rule'],'summaries':summaries,
        'scope':'Every metric is first token-averaged within question/bin, then question-averaged within context, then context-averaged within cohort; two cohorts equal. No row or coordinate picked by amplitude/effect.',
        'interpretation':'Query standardizedRMS is distance from frozen first-prefix training mean in that rule\'s own coordinatewise scale. It is not a semantic distance or full-support proof. Context vector unchanged across a question history. Later-tail bins select longer outputs.'}
    previous=folder/'descriptive_report.json'
    if previous.exists():
        assert read(previous)['summaries']==summaries
        value['repair']={'previous_numerical_report_sha256':sha(previous),
            'reason':'Initial report successfully saved all numerical summaries, then console printing failed due to missing json import. Import fixed; recomputed numerical summaries exactly equal; original result and source preserved.'}
    immutable(final,value)
    for row in summaries:
        if row['variant']==selection['primary_rule']and row['split']=='diagnostic':
            print('NATURAL_HISTORY_FEATURE_SHIFT',row['bin'],json.dumps(row['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True);main(parser.parse_args().model)
