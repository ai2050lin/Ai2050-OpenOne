"""Descriptive aggregation of already frozen output/path measurements; no fitting."""
import numpy as np
import b36_composition14 as B
import b36_composition14_analysis as A
import b36_composition14_tail as T
from b36_composition14_tail_analysis import family_values
from recursive_branch14_analysis import inference


def run():
    B.configure();T.admit();reports={}
    for split in ['confirm','format']:
        folder=B.OUT/f'tail-{split}'
        if not (folder/'summary.json').exists():continue
        raw=B.read(folder/'summary.json');stats=B.read(B.OUT/f'{split}-tail-report.json');predictions=raw['predictions'];summary=[]
        for key in sorted({(r['branch'],r['depth'],r['mode'],r['dose'],r['operator']) for r in predictions}):
            rr=[r for r in predictions if (r['branch'],r['depth'],r['mode'],r['dose'],r['operator'])==key]
            error=np.array([r['KL'] for r in rr]);base=np.array([r['zero_effect_KL'] for r in rr])
            summary.append(dict(zip(['branch','depth','mode','dose','operator'],key),n=len(rr),families=len({r['family'] for r in rr}),
                mean_KL=float(error.mean()),mean_zero_effect_KL=float(base.mean()),
                ratio=A.grouped_ratio(rr,error,base) if base.mean()>1e-12 else None,
                negative_float_KL_count=int((error<0).sum()),gate='No new pass/fail threshold; separately reported conditional edited-output fidelity.'))
        nonlinear=[]
        terms=stats['gate_value_decomposition'];fields=['actual_A_rms','gate_rms','value_rms','cross_rms','rounding_rms']
        for key in sorted({(r['branch'],r['depth'],r['mode'],r['dose']) for r in terms}):
            rr=[r for r in terms if (r['branch'],r['depth'],r['mode'],r['dose'])==key]
            nonlinear.append(dict(zip(['branch','depth','mode','dose'],key),mean_per_row_RMS={f:float(np.mean([r[f] for r in rr])) for f in fields}))
        rounding=[]
        for path in sorted(folder.glob('chunk-*.npz')):
            with np.load(path) as data:rounding.extend(data['unit_seed_rounding_relative'].astype(float).tolist())
        effects={(r['id'],r['arm']):r for r in raw['effects']};base=[r for r in raw['effects'] if r['arm']=='baseline'];controls=[]
        valid={r['id']:r['passed'] for r in raw['norm_controls'] if r['arm']=='random_norm'}
        for key in sorted({(r['branch'],r['depth'],r['mode']) for r in base}):
            rr=[r for r in base if (r['branch'],r['depth'],r['mode'])==key]
            controls.append(dict(zip(['branch','depth','mode'],key),comparison='target_zero minus random_norm',all_controls_qualified=all(valid[r['id']] for r in rr),
                statistics=inference(family_values(rr,[effects[r['id'],'zero']['delta_margin']-effects[r['id'],'random_norm']['delta_margin'] for r in rr])),
                inference_scope='Descriptive nominal95% interval; no added confirmatory gate or multiplicity claim.'))
        reports[split]=dict(edited_output=summary,gate_value_terms=nonlinear,random_norm_paired=controls,
            unit_seed_rounding_relative=dict(mean=float(np.mean(rounding)),maximum=float(max(rounding)),p95=float(np.quantile(rounding,.95))),
            actual_root_effects='See original frozen criteria and results in '+split+'-tail-report.json',
            raw_summary_sha256=B.file_sha(folder/'summary.json'),frozen_statistics_sha256=B.file_sha(B.OUT/f'{split}-tail-report.json'))
    materials=B.read(B.OUT/'materials.json')['rows'];top1={};scores={}
    material_by_id={r['id']:r for r in materials};paired_format=[]
    main=B.read(B.OUT/'tail-confirm/summary.json');fmt=B.read(B.OUT/'tail-format/summary.json') if (B.OUT/'tail-format/summary.json').exists() else None
    def pair_key(e):
        r=material_by_id[e['id']];return r['family'],r['sequence'],r['mode'],r['title'],e['arm']
    lookup={pair_key(e):e for e in main['effects'] if e['title']==0}
    if fmt:
        for key in sorted({(e['branch'],e['depth'],e['mode'],e['arm']) for e in fmt['effects'] if e['arm'] in ['zero','double']}):
            rr=[e for e in fmt['effects'] if (e['branch'],e['depth'],e['mode'],e['arm'])==key];aa=[lookup[pair_key(e)] for e in rr]
            paired_format.append(dict(zip(['branch','depth','mode','arm'],key),n=len(rr),
                ABC=inference([e['delta_margin'] for e in aa]),XYZ=inference([e['delta_margin'] for e in rr]),
                XYZ_minus_ABC=inference([e['delta_margin']-a['delta_margin'] for a,e in zip(aa,rr)]),
                scope='Same20 families and same semantic options, title0; descriptive95% paired intervals, not a new sample or confirmatory gate.'))
    for split in ['confirm','format']:
        for path in sorted((B.OUT/split).glob('chunk-*/chunk.json')):
            chunk=B.read(path);top1.update(zip(chunk['row_ids'],chunk['top1']));scores.update(zip(chunk['row_ids'],chunk['scores']))
    for row in B.read(B.OUT/'behavior-only/calibration-results.json')['rows']:top1[row['id']]=row['top1'];scores[row['id']]=row['scores']
    next_token=[]
    for key in sorted({(r['split'],r['branch'],r['depth'],r['mode']) for r in materials if r['split'] in ['confirm','format','order']}):
        rr=[r for r in materials if (r['split'],r['branch'],r['depth'],r['mode'])==key]
        next_token.append(dict(zip(['split','branch','depth','mode'],key),n=len(rr),
            full_vocab_top1_correct=sum(top1[r['id']]==r['choice_ids'][r['gold']] for r in rr),
            full_vocab_top1_in_choices=sum(top1[r['id']] in r['choice_ids'] for r in rr),
            forced_correct=sum(scores[r['id']][r['gold']]>max(v for k,v in enumerate(scores[r['id']]) if k!=r['gold']) for r in rr)))
    B.save('path-output-report.json',dict(kind='descriptive_aggregation_of_preplanned_measurements',reports=reports,baseline_next_token=next_token,paired_format=paired_format,source_sha256=B.file_sha(__file__),
        scope='No new model run, selection, fit or gate. Whole-vocabulary edited-target KL and finite gate/value decomposition were specified before confirmation. Family bootstrap ratios are descriptive. RMS terms may cancel; do not interpret as additive percentages.'))
    print('Path/output summaries',list(reports),flush=True)


if __name__=='__main__':run()
