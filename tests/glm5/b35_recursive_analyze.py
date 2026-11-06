"""Compute all B35 native feature profiles and held-out recursive-depth results."""
import numpy as np
from scipy.stats import binom

import mlp_factor_steer_common as C
from b35_recursive_collect import BLOCK, OUT, PARTS, SEED, WIDTH, identity
from server import neuron_census_service as NS
from shared.dist_protocol.mechanism_trace import unbits


def checked_npz(name, expected):
    path = OUT / name
    assert C.file_sha(path) == expected
    return np.load(path)


def behavior(rows, source):
    result = []
    for frame in [0, 1]:
        selected = [r for r, i in zip(rows, source) if i['frame'] == frame]
        depth = len(selected[0]['free'])
        for step in range(depth):
            result.append({'frame': frame, 'step': step + 1, 'n': len(selected),
                           'teacher_token_correct': sum(r['teacher'][step]['correct'] for r in selected),
                           'free_token_correct': sum(r['free'][step]['correct'] for r in selected),
                           'free_prefix_correct': sum(all(t['correct'] for t in r['free'][:step+1]) for r in selected)})
    return result


def natural():
    target = identity()['model_snapshot']
    info = max((r for r in NS.census_runs()
                if NS.census_manifest(r['model_key'], r['run'])['model_snapshot']['sha256'] == target),
               key=lambda row: row['n_tokens'])
    key, run = info['model_key'], info['run']
    manifest = NS.census_manifest(key, run)
    assert manifest['model_snapshot']['sha256'] == identity()['model_snapshot']
    root = NS.native_root() / '_census' / key / run
    meta = next(f for f in manifest['files'] if f['path'] == f'L{BLOCK}.npz')
    assert C.file_sha(root / meta['path']) == meta['sha256']
    census = NS.census_layer(key, run, BLOCK)
    n = int(census['n'])
    hist = census['hist'].sum(1)
    assert hist.shape == (WIDTH,4) and np.all(hist.sum(1) == n)
    mean = census['s1']/n
    fields = {'mean':mean, 'std':np.sqrt(np.maximum(census['s2']/n-mean**2,0)),
              'min':census['amin'], 'max':census['amax'], 'regime_fraction':hist/n}
    cman, streams = NS.corpus(manifest['corpus_fp'])
    tok = NS.tokenizer(str(C.MODEL))

    def pole(u, k):
        gidx = int(census[f'{k}_idx'][u,0])
        src, off = gidx >> NS.SRC_SHIFT, gidx & ((1<<NS.SRC_SHIFT)-1)
        return {'a':float(census[f'{k}_val'][u,0]),'g':float(census[f'{k}_g'][u,0]),
                'v':float(census[f'{k}_v'][u,0]),'source':cman['sources'][src]['source'],
                'offset':off, 'gidx':gidx, **NS.context(tok, streams[src], off)}

    def tokens(indices):
        return [{'id':int(i), 'text':NS._decode(tok,[i])} for i in indices]

    items = [{'unit':u, **{k:v[u].tolist() for k,v in fields.items()},
              'poles':{'max':pole(u,'top'),'min':pole(u,'bot')},
              'readout':{'promotes':tokens(census['readout_top'][u]), 'suppresses':tokens(census['readout_bot'][u])}}
             for u in range(WIDTH)]
    scope = {'tokens':n,'corpus_fp':manifest['corpus_fp'],'sources':[s['source'] for s in cman['sources']],
             'regime_order':list(NS.REGIMES),'census_layer_sha256':meta['sha256'],
             'census_manifest_sha256':C.file_sha(root/'manifest.json'),
             'readout':'Unembedding of RMS-weighted B35 output column, before contextual RMS normalization. Token labels are readout directions.'}
    return items, scope


def signed_test(local, sign, shuffle_seed):
    n = len(local)
    oriented = local*sign
    same = (oriented > 0).sum(0)
    mean = oriented.mean(0)
    rms = np.sqrt(np.square(oriented).mean(0))
    scaled = mean/np.maximum(rms,1e-12)
    p = binom.sf(same-1,n,.5)
    passed = (same/n >= .65) & (p < .05/WIDTH) & (scaled >= .10)
    random = np.random.default_rng(shuffle_seed).permutation(np.where(np.arange(n)<n//2,-1,1))
    null = oriented*random[:,None]
    null_same = (null>0).sum(0)
    control = (null_same/n >= .65) & (binom.sf(null_same-1,n,.5) < .05/WIDTH) & (
        null.mean(0)/np.maximum(rms,1e-12)>=.10)
    return {'same':same,'mean':mean,'rms':rms,'scaled':scaled,'p':p,'passed':passed,
            'n':n,'null_passed_count':int(control.sum())}


def prediction(teacher, prior, candidate):
    x = unbits(teacher['teacher_B34'][:,1:])[...,candidate].reshape(-1,WIDTH)
    y = unbits(teacher['teacher_A'][:,1:]).reshape(-1,WIDTH)
    xm, ym = x.mean(0), y.mean(0)
    dx = x-xm
    slope = (dx*(y-ym)).sum(0)/np.maximum((dx*dx).sum(0),1e-12)
    fit = {'mean':ym,'slope':slope}
    for key, data, start in prior:
        x = unbits(data['teacher_B34'][:,start:])[...,candidate]
        y = unbits(data['teacher_A'][:,start:])
        last = unbits(data['teacher_A'][:,start-1:-1])
        mse = ((y-ym-(x-xm)*slope)**2).mean((0,1))
        mean = ((y-ym)**2).mean((0,1))
        prior_mse = ((y-last)**2).mean((0,1))
        fit[key] = {'mse':mse,'mean_baseline_mse':mean,'previous_baseline_mse':prior_mse,
                    'gain_vs_mean':1-mse/np.maximum(mean,1e-12),'gain_vs_previous':1-mse/np.maximum(prior_mse,1e-12)}
    return fit


def features():
    C.save(OUT/'analysis-execution.json', {**identity(),'schema':'b35_recursive_analysis_execution.v1',
          'source_sha256':C.file_sha(__file__),'design_sha256':C.file_sha(OUT/'prereg.json')})
    paths = ('discovery','confirm','lockbox')
    runs = {k:C.read_gz(OUT/f'{k}-result.json.gz') for k in paths}
    z = {k:checked_npz(f'{k}-native.npz',v['native_sha256']) for k,v in runs.items()}
    source = C.read_gz(OUT/'inputs.json.gz')['sets']
    for key in paths:
        assert [r['id'] for r in runs[key]['rows']] == [r['id'] for r in source[key]]
    depths = {k:behavior(runs[k]['rows'],source[k]) for k in paths}

    def rates(key):
        final = [v['free_prefix_correct']/v['n'] for v in depths[key] if v['step']==PARTS[key][1]]
        return {'pooled':sum(final)/len(final),'by_frame':final}

    c, l = rates('confirm'),rates('lockbox')
    passed = c['pooled']>=.5 and l['pooled']>=.35 and min(c['by_frame']+l['by_frame'])>=.30
    discovery = z['discovery']['local'].mean(1)
    discovery_mean = discovery.mean(0)
    sign = np.sign(discovery_mean).astype(np.int8)
    tests = {key:signed_test(z[key]['local'][:,start:].mean(1),sign,SEED+start+idx)
             for idx,(key,start) in enumerate([('confirm',0),('lockbox',3)])}
    up = dict(np.load(OUT/'upstream-candidates.npz'))
    candidate = np.where(np.abs(up['gate_cosine'][:,0])>=np.abs(up['value_cosine'][:,0]),
                         up['gate_unit'][:,0],up['value_unit'][:,0])
    pred = prediction(z['discovery'],[('confirm',z['confirm'],1),('lockbox',z['lockbox'],3)],candidate)
    native, scope = natural()
    stats = {}
    for key,start in [('confirm',0),('lockbox',3)]:
        gold = unbits(z[key]['teacher_A'][:,start:])
        free = unbits(z[key]['free_A'][:,start:])
        match = np.array([[all(t['correct'] for t in row['free'][:s]) for s in range(gold.shape[1]+start)]
                          for row in runs[key]['rows']],bool)[:,start:]
        stats[key] = {'teacher_rms':np.sqrt((gold**2).mean((0,1))),
                      'free_rms':np.sqrt((free**2).mean((0,1))),
                      'history_error_rms_delta':np.sqrt(((gold[~match]-free[~match])**2).mean(0))
                          if (~match).any() else np.zeros(WIDTH),
                      'history_error_positions':int((~match).sum())}
    replicated = tests['confirm']['passed'] & tests['lockbox']['passed']
    for u, row in enumerate(native):
        row.update({'discovery_sign':int(sign[u]),
                    'discovery_mean_contribution':float(discovery_mean[u]),
                    'replicated':bool(replicated[u]),
                    'behavior_eligible':bool(replicated[u] and passed),
                    'upstream':{k:[{'unit':int(i),'cosine':float(v)} for i,v in
                                   zip(up[k+'_unit'][u],up[k+'_cosine'][u])] for k in ['gate','value']},
                    'upstream_prediction_unit':int(candidate[u])})
        for key in tests:
            t = tests[key]
            row[key] = {'n':t['n'],'same_sign':int(t['same'][u]),'local_mean':float(t['mean'][u]),
                        'local_rms':float(t['rms'][u]),'standardized_mean':float(t['scaled'][u]),
                        'p':float(t['p'][u]),'passed':bool(t['passed'][u]),
                        'prediction':{k:float(v[u]) for k,v in pred[key].items()},
                        **{k:float(v[u]) if isinstance(v,np.ndarray) else v for k,v in stats[key].items()}}
    C.write_gz(OUT/'units.json.gz', {'rows':native, **identity()})
    result = {**identity(),'schema':'b35_recursive_analysis.v1','block':BLOCK,'d_ff':WIDTH,'natural':scope,
              'steps':depths,'behavior_gate':{'passed':passed,'confirm':c,'lockbox':l},
              'replicated_units':int(replicated.sum()),'eligible_units':int(replicated.sum()) if passed else 0,
              'confirm_passed_units':int(tests['confirm']['passed'].sum()),
              'lockbox_passed_units':int(tests['lockbox']['passed'].sum()),
              'shuffle_passed_units':{k:v['null_passed_count'] for k,v in tests.items()},
              'upstream_improves_mean':int(((pred['confirm']['gain_vs_mean']>0)&(pred['lockbox']['gain_vs_mean']>0)).sum()),
              'upstream_improves_both':int(((pred['confirm']['gain_vs_mean']>0)&(pred['lockbox']['gain_vs_mean']>0)&
                                           (pred['confirm']['gain_vs_previous']>0)&(pred['lockbox']['gain_vs_previous']>0)).sum()),
              'error_cases':{key:[i for i,r in enumerate(runs[key]['rows']) if not all(t['correct'] for t in r['free'])]
                             for key in paths},
              'file_hashes':{key:runs[key]['native_sha256'] for key in paths}}
    C.save(OUT/'analysis.json',result)
    print({k:v for k,v in result.items() if k not in ('steps','error_cases','provenance')},flush=True)


if __name__ == '__main__':
    features()
