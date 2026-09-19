"""Saved deployment coefficients reproduce every Q4 validation score."""
from rdc_question_common import *
from rdc_question_predictor import FrozenPredictor
from rdc_question_fit import evaluation_arrays
import rdc_question_data as data


def main(key='qwen4'):
    selection = read(OUT/'fit'/key/'validation_selection.json')
    rows, groups, questions = data.index(key,{'validation'})
    features = data.native_features(key,rows,groups,questions)
    actual = data.targets(rows,questions,'postnorm')
    group_ids = np.array([r['group_id'] for r in rows])
    checks = []
    for variant in [selection['primary_rule'],selection['control_rule']]:
        rule = FrozenPredictor(key,variant)
        predicted = rule.predict(features['H12'],features['read'],features['context_H12'])
        scores = evaluation_arrays(predicted,actual,group_ids)
        record = read(OUT/'fit'/key/variant/'evaluation.json')
        ref = next(r['field'] for r in record['evaluations'] if r['kind']=='selected' and r['target']=='postnorm' and r['split']=='validation')
        assert sha(ROOT/ref['path']) == ref['sha256']
        with np.load(ROOT/ref['path']) as z:
            deltas = {k:float(np.abs(scores[k]-z[k]).max()) for k in scores}
            assert all(v < 1e-8 for v in deltas.values()),deltas
        # B1CPU execution is the future deployment shape. Record its roundoff
        # relative to the earlier192-row CPU matmul, not assume bit equality.
        individual = np.concatenate([rule.predict(features['H12'][i],features['read'][i],features['context_H12'][i]) for i in range(len(rows))])
        error = float(np.max(np.abs(individual-predicted)))
        assert error < 1e-7
        checks.append({'variant':variant,'all_validation_questions':len(rows),
            'all_coordinate_and_question_score_max_errors':deltas,
            'individual_vs_matrix_CPU_prediction_maximum_error':error,'predictor':rule.identity})
    result = {'timestamp':stamp(),'source':snapshot(__file__),
        'predictor':snapshot(Path(__file__).with_name('rdc_question_predictor.py')),
        'model':key,'all_passed':True,'checks':checks,
        'scope':'Saved CPUcoefficients/normalization deployment check; actual native early-onlyCUDA path and free histories need separate qualification.'}
    immutable(OUT/'unit'/('predictor_'+key+'_'+str(time.time_ns())+'.json'),result)
    save(OUT/'unit'/('predictor_'+key+'_current.json'),result)
    print('NATURAL_PREDICTOR_COEFFICIENT_PASS',key,len(checks),flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],default='qwen4')
    main(parser.parse_args().model)
