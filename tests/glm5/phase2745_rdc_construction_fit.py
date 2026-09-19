"""Full-native-coordinate pair-transition operators with prospective controls.

The eigensystem is only a full-rank ridge solver.  Every input coordinate,
output coordinate and eigencomponent is retained; no activation is patched.
Run only after all native CUDA model jobs have exited.
"""
import argparse
from collections import defaultdict
from phase2745_rdc_construction_contract import freeze, freeze_extraction
from rdc_construction_common import *

INPUTS = ['actual_query_H1', 'available_prefix_ordered_candidate_early_output']
TARGETS = ['Hearly', 'postnorm']
CONTROLS = ['true_correspondence', 'training_pair_correspondence_shuffled_within_family_language']
SPLITS = ['train', 'validation', 'test']
QUERY_SPLITS = ['train_query', 'validation_query', 'unseen_query']


def eigensystem(train):
    import torch
    scale = train.square().mean(0).sqrt().clamp_min(1e-8)
    normalized = train/scale
    covariance = normalized.T@normalized/len(normalized)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    assert float(eigenvalues.min()) > -1e-8
    reconstructed = (vectors*eigenvalues[None])@vectors.T
    residual = float(torch.linalg.vector_norm(reconstructed-covariance)/torch.linalg.vector_norm(covariance))
    assert residual < 1e-9
    return normalized, scale, covariance, eigenvalues, vectors, residual


def solve_projected(vectors, eigenvalues, projected_target, lam):
    return vectors@(projected_target/(eigenvalues+lam)[:, None])


def dataset(key, rows, early):
    width = read(BASE / 'capture' / key / 'result.json')['width']
    n = len(rows)//2
    inputs = {name: np.empty((n, 100, width)) for name in INPUTS}
    targets = {name: np.empty((n, 100, width)) for name in TARGETS}
    pairs = []
    for i in range(n):
        a, b = rows[2*i:2*i+2]
        assert a['pair_id'] == b['pair_id'] and a['world'] == 0 and b['world'] == 1
        x, y = [], []
        for row in [a, b]:
            file = BASE / 'capture' / key / 'fields' / (row['sample_id']+'.npz')
            cp = read(BASE / 'capture' / key / 'commits' / (row['sample_id']+'.json'))
            assert sha(file) == cp['sha256']
            with np.load(file) as z:
                ix = z['query_layer_indices'].tolist()
                h = unbits(z['query_selected_states']).astype(float)
                x.append([h[ix.index(1)], unbits(z['candidate_early_output'][4]).astype(float)])
                y.append([h[ix.index(early)], unbits(z['postnorm']).astype(float)])
        for j, name in enumerate(INPUTS):
            inputs[name][i] = x[1][j]-x[0][j]
        for j, name in enumerate(TARGETS):
            targets[name][i] = y[1][j]-y[0][j]
        pairs.append({k: a[k] for k in ['pair_id', 'source_group', 'family', 'language', 'case', 'split']} |
                     {'sample_ids': [a['sample_id'], b['sample_id']]})
    return inputs, targets, pairs


def correspondence(pairs, training):
    permutation = np.arange(len(training))
    for family, language in sorted({(pairs[i]['family'], pairs[i]['language']) for i in training}):
        slots = [j for j, i in enumerate(training) if (pairs[i]['family'], pairs[i]['language']) == (family, language)]
        seed = int(rank(f'pair_permutation/{family}/{language}')[:16], 16)
        permutation[slots] = np.random.default_rng(seed).permutation(slots)
    return permutation


def reports(pair_errors, zero, pairs, queries):
    results = []
    subsets = [('all', 'all')] + [('family', f) for f in sorted({p['family'] for p in pairs})] + [('language', x) for x in ['en', 'zh']]
    for source_split in SPLITS:
      for query_split in QUERY_SPLITS:
        qq = [q for q, p in enumerate(queries) if p['split'] == query_split]
        for stype, svalue in subsets:
            ii = [i for i, p in enumerate(pairs) if p['split'] == source_split and (stype == 'all' or p[stype] == svalue)]
            groups = [pairs[i]['source_group'] for i in ii]
            for ti, target in enumerate(TARGETS):
              zz = zero[ti, ii][:, qq].mean(-1)
              for xi, input_name in enumerate(INPUTS):
                real = pair_errors[xi, 0, ti, ii][:, qq].mean(-1)
                shuffled = pair_errors[xi, 1, ti, ii][:, qq].mean(-1)
                results.append({'source_split': source_split, 'query_split': query_split, 'subset_type': stype,
                    'subset': svalue, 'target': target, 'input': input_name, 'pairs': len(ii), 'groups': len(set(groups)),
                    'zero_change_MSE': clustered(zz, groups), 'true_correspondence_MSE': clustered(real, groups),
                    'shuffled_correspondence_MSE': clustered(shuffled, groups),
                    'zero_minus_true_MSE': clustered(zz-real, groups),
                    'shuffled_minus_true_MSE': clustered(shuffled-real, groups)})
    return results


def main(key):
    import torch
    protocol, material = freeze()
    extraction = freeze_extraction()
    out = BASE / 'fit' / key
    if (out / 'result.json').exists():
        assert read(out / 'result.json')['all_passed']
        return
    # Avoid running a GPU solver alongside any native model job.
    import psutil
    own_process_chain = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if process.info['pid'] not in own_process_chain and any(Path(arg).name in
             {'phase2745_rdc_construction_capture.py', 'phase2745_rdc_construction_pilot.py',
              'phase2745_rdc_construction_batch_pilot.py', 'phase2745_rdc_construction_norm.py',
              'phase2745_rdc_construction_fit.py', 'phase2745_rdc_construction_compile.py'} for arg in command):
            raise RuntimeError('Wait for native CUDA job to exit before ridge solve: '+str(process.info['pid']))
    assert read(BASE / 'capture' / key / 'result.json')['all_passed']
    preflight = read(BASE / 'fit_preflight/result.json')
    assert preflight['all_passed'] and preflight['fit_source']['sha256'] == sha(__file__)
    start = time.monotonic()
    source = snapshot(__file__)
    guard(2*1024**3)
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        native = read(BASE / 'capture' / key / 'result.json')
        rows, queries = material['models'][key]['rows'], material['models'][key]['probes']
        xdata, ydata, pairs = dataset(key, rows, native['early'])
        width = native['width']
        pi = {s: np.array([i for i, r in enumerate(pairs) if r['split'] == s]) for s in SPLITS}
        qi = {s: np.array([i for i, q in enumerate(queries) if q['split'] == s]) for s in QUERY_SPLITS}
        permutation = correspondence(pairs, pi['train'])
        save(out / 'pair_index.json', pairs)
        save(out / 'training_correspondence.json', {'training_pair_indices': pi['train'].tolist(),
             'shuffled_training_slots': permutation.tolist(), 'scope': extraction['shuffle']})
        pair_errors = np.zeros((2, 2, 2, len(pairs), 100))
        zero = np.stack([np.mean(ydata[t]**2, axis=-1) for t in TARGETS])
        coordinate_errors = np.zeros((2, 2, 2, 3, 3, width))
        counts = np.zeros((3, 3), dtype=int)
        for si, ss in enumerate(SPLITS):
            for qj, qs in enumerate(QUERY_SPLITS):
                counts[si, qj] = len(pi[ss])*len(qi[qs])
        def stacked_targets(indices, query_indices):
            return torch.cat([torch.from_numpy(ydata[t][indices][:, query_indices].reshape(-1, width)).to('cuda') for t in TARGETS], -1)
        ytrain = stacked_targets(pi['train'], qi['train_query'])
        yvalidation = stacked_targets(pi['validation'], qi['validation_query'])
        nquery = len(qi['train_query'])
        perm_flat = (permutation[:, None]*nquery+np.arange(nquery)[None]).reshape(-1)
        selections, diagnostics = [], []
        with torch.inference_mode():
          for xi, input_name in enumerate(INPUTS):
            train = torch.from_numpy(xdata[input_name][pi['train']][:, qi['train_query']].reshape(-1, width)).to('cuda')
            train, scale, covariance, eigenvalues, vectors, residual = eigensystem(train)
            validation = torch.from_numpy(xdata[input_name][pi['validation']][:, qi['validation_query']].reshape(-1, width)).to('cuda')
            validation_in_eigenbasis = (validation/scale)@vectors
            npz(out / ('solver_'+input_name+'.npz'), training_coordinate_RMS=scale.cpu().numpy(),
                all_eigenvalues=eigenvalues.cpu().numpy())
            diagnostics.append({'input': input_name, 'native_input_coordinates': width, 'training_pairs': len(pi['train']),
                'training_pair_query_rows': len(train), 'minimum_eigenvalue': float(eigenvalues.min()),
                'negative_roundoff_eigenvalues': int((eigenvalues < 0).sum()), 'full_eigen_reconstruction_relative_error': residual,
                'components_discarded': 0, 'eigenvectors_used_only_for_full_rank_solver': True})
            for ci, control in enumerate(CONTROLS):
                target = ytrain if ci == 0 else ytrain[torch.tensor(perm_flat, device='cuda')]
                cross = train.T@target/len(train)
                projected_target = vectors.T@cross
                val_loss = np.zeros((len(extraction['ridge_lambdas']), 2))
                for li, lam in enumerate(extraction['ridge_lambdas']):
                    predicted = (validation_in_eigenbasis/(eigenvalues+lam)[None])@projected_target
                    for ti in range(2):
                        val_loss[li, ti] = float((predicted[:, ti*width:(ti+1)*width]-yvalidation[:, ti*width:(ti+1)*width]).square().mean())
                chosen = val_loss.argmin(0)
                selection = {'timestamp': stamp(), 'input': input_name, 'control': control,
                    'validation_MSE': val_loss.tolist(), 'selected_lambda_indices': chosen.tolist(),
                    'selected_lambdas': [extraction['ridge_lambdas'][i] for i in chosen],
                    'selection_inputs': 'Validation cases and validation queries only; test evaluation has not started for this operator.'}
                save(out / ('selection_'+input_name+'__'+control+'.json'), selection)
                selections.append(selection)
                for ti, target_name in enumerate(TARGETS):
                    lam = extraction['ridge_lambdas'][int(chosen[ti])]
                    standardized = solve_projected(vectors, eigenvalues, projected_target[:, ti*width:(ti+1)*width], lam)
                    normal_residual = float(torch.linalg.vector_norm(covariance@standardized+lam*standardized-cross[:, ti*width:(ti+1)*width]) /
                                            torch.linalg.vector_norm(cross[:, ti*width:(ti+1)*width]).clamp_min(1e-20))
                    assert normal_residual < 1e-7
                    operator = standardized/scale[:, None]
                    file = out / 'operators' / (input_name+'__'+control+'__'+target_name+'.npz')
                    npz(file, operator=operator.cpu().numpy(), training_input_scale=scale.cpu().numpy())
                    save(file.with_suffix('.json'), {'timestamp': stamp(), 'input': input_name, 'control': control,
                        'target': target_name, 'lambda': lam, 'operator_sha256': sha(file),
                        'axes': 'Rows=every native input coordinate; columns=every native output coordinate. Row-vector input multiplied on the right.',
                        'normal_equation_relative_residual': normal_residual,
                        'effective_ridge_degrees_of_freedom': float((eigenvalues/(eigenvalues+lam)).sum()),
                        'matrix_rank_not_used_for_pruning': True, 'coefficient_count': width*width})
                    for p, pair in enumerate(pairs):
                        predicted = torch.from_numpy(xdata[input_name][p]).to('cuda')@operator
                        truth = torch.from_numpy(ydata[target_name][p]).to('cuda')
                        squared = (predicted-truth).square()
                        pair_errors[xi, ci, ti, p] = squared.mean(-1).cpu().numpy()
                        si = SPLITS.index(pair['split'])
                        for qj, qs in enumerate(QUERY_SPLITS):
                            coordinate_errors[xi, ci, ti, si, qj] += squared[qi[qs]].sum(0).cpu().numpy()
                    print('CONSTRUCTION_OPERATOR', key, input_name, control, target_name,
                          'lambda', lam, 'residual', normal_residual, flush=True)
                    del standardized, operator, predicted, truth, squared
                del cross, projected_target, target
            del train, scale, covariance, eigenvalues, vectors, validation, validation_in_eigenbasis
            torch.cuda.empty_cache()
        coordinate_errors /= counts[None, None, None, :, :, None]
        npz(out / 'all_coordinate_pair_errors.npz', pair_query_MSE=pair_errors, zero_change_pair_query_MSE=zero,
            all_coordinate_stratum_MSE=coordinate_errors, pair_query_counts=counts)
        summaries = reports(pair_errors, zero, pairs, queries)
        primary = [r for r in summaries if r['source_split'] == 'test' and r['query_split'] == 'unseen_query'
                   and r['subset_type'] == 'all' and r['input'] == 'actual_query_H1']
        qualified = {r['target']: r['zero_minus_true_MSE']['interval95'][0] > 0
                     and r['shuffled_minus_true_MSE']['interval95'][0] > 0 for r in primary}
        result = {'timestamp': stamp(), 'source': source, 'all_passed': True, 'model': key, 'width': width,
            'input_names': INPUTS, 'target_names': TARGETS, 'controls': CONTROLS, 'selections': selections,
            'solver_diagnostics': diagnostics, 'summaries': summaries, 'primary': primary,
            'primary_relation_change_qualification': qualified, 'seconds': time.monotonic()-start,
            'global_encoding_mechanism_closed': False, 'new_universal_mathematics_claimed': False,
            'scope': extraction['limitations']}
        save(out / 'result.json', result)
        ledger('construction_full_coordinate_operator_' + key, result['seconds'])
        print('CONSTRUCTION_OPERATOR_FIT_DONE', key, qualified, result['seconds'], flush=True)
    except Exception as exc:
        failure(out, start, exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen4', 'qwen14', 'glm4'])
    main(parser.parse_args().model)
