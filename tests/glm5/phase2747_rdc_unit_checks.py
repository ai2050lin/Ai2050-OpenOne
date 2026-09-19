"""Implementation unit checks only; synthetic arrays are not model evidence."""
from rdc_formation_common import *


def main():
    import torch
    from phase2747_rdc_transfer_readout import bias_metrics, direct_check
    from phase2747_rdc_calibration import validation_weights
    from rdc_formation_propagation import mlp_function
    start = time.monotonic(); torch.manual_seed(2747)
    lp = torch.randn(1000, dtype=torch.float64).log_softmax(-1)
    lq = torch.randn(1000, dtype=torch.float64).log_softmax(-1)
    digit, letter = list(range(8)), list(range(8, 16))
    errors = [direct_check(lp, lq, bias, digit, letter, gold,
        bias_metrics(lp, lq, digit, letter, gold, bias)) for bias in [digit, letter] for gold in digit]
    x = torch.randn(1, 9, 5, dtype=torch.float64); residual = torch.randn_like(x)
    weights = (torch.randn(7, 5, dtype=torch.float64), torch.randn(7, 5, dtype=torch.float64), torch.randn(5, 7, dtype=torch.float64))
    direction = tuple(torch.randn_like(v)*.01 for v in weights)
    fn = mlp_function(x, residual); base, tangent = torch.func.jvp(fn, weights, direction)
    scales = [.01, .001, .0001]; curves = []
    for scale in scales:
        finite = fn(*(w+scale*d for w, d in zip(weights, direction)))
        curves.append([float(((v-b)/scale-t).square().mean()) for v, b, t in zip(finite, base, tangent)])
    assert curves[-1][0] < curves[0][0]/100
    hidden, pullback = torch.func.vjp(lambda *w: fn(*w)[0], *weights)
    covector = torch.randn_like(hidden); gradient = pullback(covector)
    dual_error = float(abs((covector*tangent[0]).sum()-sum((g*d).sum() for g, d in zip(gradient, direction))))
    assert dual_error < 1e-12
    data = gzread(OUT/'material/rows.json.gz'); rows = data['validation']+data['diagnostic']+data['fresh']
    weight = validation_weights(rows)
    assert abs(weight.sum()-1) < 1e-12 and all(weight[i] == 0 for i, r in enumerate(rows) if r['split'] != 'validation')
    scripts = ['phase2747_rdc_transfer_readout.py', 'phase2747_rdc_calibration.py', 'rdc_formation_readout.py',
        'rdc_formation_history.py', 'phase2747_rdc_own_history.py', 'phase2747_rdc_program_own_history.py',
        'rdc_formation_propagation.py', 'phase2747_rdc_parameter_capture.py', 'phase2747_rdc_parameter_propagation.py',
        'phase2747_rdc_radius_analysis.py']
    import py_compile
    for name in scripts: py_compile.compile(str(Path(__file__).with_name(name)), doraise=True)
    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'uniform_bias_exact_distribution_checks': len(errors), 'uniform_bias_max_error': max(errors),
        'local_parameter_finite_scales': scales, 'local_parameter_JVP_quotient_MSE': curves,
        'full_parameter_adjoint_absolute_error': dual_error,
        'validation_source_weights_sum': float(weight.sum()), 'test_weights_all_zero': True,
        'sources': [snapshot(Path(__file__).with_name(name)) for name in scripts],
        'seconds': time.monotonic()-start, 'scope': 'CPU synthetic-array implementation checks and declared validation weighting, not a local LLM test, new language evidence or artificial corpus result.'}
    path = OUT/'engineering'/('unit_checks_'+str(time.time_ns())+'.json')
    save(path, result); save(OUT/'engineering/latest_unit_checks.json', {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'all_passed': True})
    print('FORMATION_UNIT_CHECKS', result['uniform_bias_max_error'], dual_error, result['seconds'], flush=True)


if __name__ == '__main__': main()
