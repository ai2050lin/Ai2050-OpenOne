"""All differential conditions, source-cluster uncertainty and scope boundaries."""
from collections import defaultdict
from rdc_construction_common import *

OUT=BASE/'phase2746/differential/analysis'


def main():
    start=time.monotonic();path=BASE/'phase2746/differential/main/records.json.gz'
    rows=gzread(path);execution=read(BASE/'phase2746/differential/main/result.json')
    assert execution['all_passed'] and len(rows)==576
    families=sorted({r['family'] for r in rows});reports=[];gains=[]
    for family in ['all',*families]:
      for begin in [12,24,35]:
        selected=[r for r in rows if r['start']==begin and (family=='all' or r['family']==family)]
        for direction in range(3):
            groups=[r['source_group'] for r in selected]
            gain=[r['cross_layer_direction_gain'][direction] for r in selected]
            gains.append({'family':family,'start':begin,'direction':direction,'endpoints':len(selected),
                'source_groups':len(set(groups)),'gain':clustered(gain,groups),
                'endpoint_quantiles_0_25_50_75_100':np.quantile(gain,[0,.25,.5,.75,1]).tolist(),
                'fixed_native_previous_unit':selected[0]['fixed_previous_unit'] if direction==2 else None})
            for epsilon in [.001,.01,.1]:
                comparison=[next(c for c in r['comparisons'] if c['direction']==direction and c['epsilon']==epsilon) for r in selected]
                measures={}
                for name in ['postnorm_central','postnorm_one_sided','full_vocab_logits_central',
                    'full_vocab_probability_central','BF16_postnorm_central','bare_readout_control']:
                    values=[c[name]['relative_L2'] for c in comparison]
                    measures[name]={'source_group_mean':clustered(values,groups),
                        'unweighted_endpoint_quantiles_25_50_75':np.quantile(values,[.25,.5,.75]).tolist()}
                measures['bare_minus_full_logit_error']=clustered([
                    c['bare_readout_control']['relative_L2']-c['full_vocab_logits_central']['relative_L2'] for c in comparison],groups)
                reports.append({'family':family,'start':begin,'direction':direction,'epsilon':epsilon,
                    'endpoints':len(selected),'source_groups':len(set(groups)),'measures':measures})
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'input_sha256':sha(path),
        'native_tails':576,'direction_epsilon_comparisons':5184,'source_rows':96,'same_history_endpoints':192,
        'independent_source_groups':len({r['source_group'] for r in rows}),
        'source_group_limits':'36natural source documents and15controlled semanticgroups. Each controlled family has only3semanticgroups; its bootstrap interval is descriptive and cannot establish broad generality.',
        'reports':reports,'direction_gains':gains,'numerical_execution':execution,
        'interpretation':'Complete chain AD predicts small smooth-reference changes and differs from identity transport; nativeBF16 finite changes can be dominated by rounding at tiny perturbations. Same fixed unit write direction has condition-dependent tail gain. These support a local conditioned propagation object, not a semantic gear label, universal law or all-history closed state.',
        'limitations':['FP32 same-valued-weight smooth extension is not the derivative of discrete BF16 rounding.',
            'Past-only KV is fixed in these local tests; this is not a derivative through all earlier token computations or future chosen tokens.',
            'Directions are three prescribed complete-coordinate probes, not a full materialized Jacobian nor proof about every direction.',
            'One fixed native down column per starting boundary is a structural probe; every residual coordinate and intermediate module remains in the propagation.',
            'Formal adjoint correctness verifies calculus implementation, not predictive identification of language semantics.',
            'Examples and source families were exposed discovery material, not independent confirmation.'],
        'seconds':time.monotonic()-start}
    save(OUT/'result.json',result);ledger('phase2746_tail_analysis',result['seconds'])
    print('TAIL_ANALYSIS_COMPLETE',len(reports),len(gains),result['independent_source_groups'],flush=True)


if __name__=='__main__':main()
