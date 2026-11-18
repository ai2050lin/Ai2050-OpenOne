"""Decompose the already-exposed v1 atlas blow-up on its original 8192 rows."""
import time

import numpy as np

from mlp_b34_confirm_v3 import configure
import mlp_b34_confirm_design as D
from shared.dist_protocol.mechanism_trace import unbits


def main():
    configure()
    from mlp_b34_confirm_analyze import silu, predict

    plan = D.READ(D.OUT/'prereg.json')
    start = time.monotonic()
    folder = D.atlas.OUT
    atlas_manifest = D.READ(folder/'manifest.json')
    expected = {row['path']: row['sha256'] for row in atlas_manifest['files']}
    names = [f'B{i}-A.npy' for i in range(32, 36)] + ['B35-G.npy', 'B35-V.npy']
    arrays = {}
    digests = {}
    for name in names:
        digests[name] = D.HASH(folder/name)
        if digests[name] != expected[name]:
            raise ValueError(f'Historical bank file changed: {name}')
        arrays[name] = np.load(folder/name, mmap_mode='r')
    params = []
    for b in (32,33,34):
        name = f'B{b}-B{b+1}-predictor.npz'
        if D.HASH(folder/name) != plan['historical_predictors'][name]:
            raise ValueError(f'Predictor changed: {name}')
        with np.load(folder/name) as f:
            params.append({k:f[k] for k in ('parents','xmean','xstd','A_coef','A_mean','G_coef','G_mean','V_coef','V_mean')})
    print('historical CPU diagnostic: data and predictors verified, 0/8192', flush=True)
    errors = {k:0. for k in ('baseline','teacher_composed','rec_direct','rec_composed','rec_clipped',
                            'true_gate','true_value','gate_term','value_term','interaction_term')}
    diag = {'true_z_gt10': 0, 'recursive_z_gt10': 0, 'coordinates': 0, 'worst_row': None}
    doc_sse = []
    maximum = -1.
    for lo in range(24576,32768,64):
        rows = slice(lo,lo+64)
        source = unbits(arrays['B32-A.npy'][rows]);target = unbits(arrays['B35-A.npy'][rows])
        g, v = (unbits(arrays[f'B35-{key}.npy'][rows]) for key in ('G','V'))
        direct = composed = clipped = source
        for i, par in enumerate(params):
            if i == 2:
                true34 = unbits(arrays['B34-A.npy'][rows])
                diag['true_z_gt10'] += int((abs((true34-par['xmean'])/par['xstd'])>10).sum())
                diag['recursive_z_gt10'] += int((abs((composed-par['xmean'])/par['xstd'])>10).sum())
                diag['coordinates'] += composed.size
            direct = predict(direct,par,'A')
            g_hat, v_hat = predict(composed,par,'G'), predict(composed,par,'V')
            composed = silu(g_hat)*v_hat
            bounded = np.clip((clipped-par['xmean'])/par['xstd'],-10,10)*par['xstd']+par['xmean']
            clipped = silu(predict(bounded,par,'G'))*predict(bounded,par,'V')
        teach_g, teach_v = predict(true34,params[-1],'G'), predict(true34,params[-1],'V')
        base_gate = silu(g);gate_error = silu(g_hat)-base_gate;value_error = v_hat-v
        for name, delta in [('baseline',params[-1]['A_mean']-target),('teacher_composed',silu(teach_g)*teach_v-target),
            ('rec_direct',direct-target),('rec_composed',composed-target),('rec_clipped',clipped-target),
            ('true_gate',base_gate*v_hat-target),('true_value',silu(g_hat)*v-target),('gate_term',gate_error*v),
            ('value_term',base_gate*value_error),('interaction_term',gate_error*value_error)]:
            errors[name] += float(np.square(delta,dtype=np.float64).sum())
        err = np.square(composed-target,dtype=np.float64)
        doc_sse.extend(err.reshape(4,16,-1).sum(axis=(1,2)).tolist())
        r,u = np.unravel_index(err.argmax(),err.shape)
        if err[r,u] > maximum:
            maximum = err[r,u]
            diag['worst_row'] = {'flat_row':lo+int(r),'unit':int(u),'native_A':float(target[r,u]),
                'recursive_A':float(composed[r,u]),'native_G':float(g[r,u]),'native_V':float(v[r,u]),
                'recursive_G':float(g_hat[r,u]),'recursive_V':float(v_hat[r,u])}
        done = lo+64-24576
        if done%1024 == 0:
            elapsed = time.monotonic()-start
            print(f'historical CPU diagnostic {done}/8192 · {elapsed:.1f}s · ETA {elapsed*(8192-done)/done:.1f}s',flush=True)
    den = errors.pop('baseline')
    result = {'provenance':plan['provenance'],'schema':'mlp_b34_historical_diagnosis.v1','scope':'Descriptive reanalysis of previously exposed atlas holdout. Fixed ±10 input z clipping is diagnostic only.',
        'plan_sha256':D.HASH(D.OUT/'prereg.json'),'bank_files':digests,'n_rows':8192,
        'fvu':{k:v/den for k,v in errors.items()},'state':diag,
        'top_1pct_doc_error_share':sum(sorted(doc_sse)[-5:])/sum(doc_sse), 'seconds':time.monotonic()-start}
    D.SAVE(D.OUT/'historical-diagnosis.json',result)
    print({'fvu':result['fvu'],'state':diag},flush=True)


if __name__ == '__main__':
    main()
