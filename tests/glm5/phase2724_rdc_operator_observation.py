"""All-layer ordinary-response atlas, complete conditional moments, and cue-composition baselines."""
from collections import Counter, defaultdict
from rdc_operator_common import *


def main():
    mode = 'main'
    capture = BASE / 'capture' / mode
    assert (capture / 'result.json').exists()
    out = BASE / 'observation'
    if (out / 'result.json').exists():
        return
    start = time.monotonic()
    selected = [r for r in rows() if r['split'] != 'confirmation']
    data, groups, events, source_stats = {}, defaultdict(list), [], []
    for row in selected:
        with np.load(capture / 'energies' / (row['sample_id']+'.npz')) as z:
            h = z['H_energy'].astype(np.float64)
            b = z['block_terms'].astype(np.float64)
            n = h.shape[1]
            growth = h[23] / np.maximum(h[12], 1e-20)
            event = (h[23] > 10) & (growth >= 10)
            event[0] = False
            for p in np.flatnonzero(event):
                events.append({'sample_id': row['sample_id'], 'source_group': row['source_group'], 'language': row['language'],
                    'split': row['split'], 'position': int(p), 'token_id': row['prompt_ids'][p], 'token': row['tokens'][p],
                    'class': int(z['token_class'][p]), 'H12_energy': float(h[12,p]), 'H23_energy': float(h[23,p]), 'growth': float(growth[p]),
                    'largest_increment_block': int(np.argmax(b[:,7,p])), 'strongest_log_decline_block': int(np.argmin(np.diff(np.log(np.maximum(h[:,p],1e-20))))),
                    'native_next_NLL': float(z['next_NLL'][p]) if p < n-1 else None})
            residual_expected = b[:,0]+b[:,1]+b[:,2]+b[:,3]+b[:,4]+b[:,5]
            mismatch = np.abs(residual_expected-b[:,6]) / np.maximum(b[:,6],1e-20)
            rr = {'sample_id': row['sample_id'], 'source_group': row['source_group'], 'language': row['language'], 'split': row['split'],
                'tokens': n, 'events': int(event.sum()), 'native_next_NLL': float(np.mean(z['next_NLL'][:-1])),
                'native_next_argmax_match': float(np.mean(z['native_argmax'][:-1] == row['prompt_ids'][1:])),
                'mean_H_energy': h.mean(1).tolist(), 'ordinary_mean_H_energy': h[:,~event & (np.arange(n)>0)].mean(1).tolist(),
                'ordinary_mean_block_terms': b[:,:,~event & (np.arange(n)>0)].mean(-1).tolist(),
                'max_residual_energy_relative_discrepancy': float(mismatch.max())}
            source_stats.append(rr)
            groups[row['split']+'/'+row['language']].append(rr)
    compressed(out / 'source_stats.json.gz', source_stats)
    compressed(out / 'events.json.gz', events)
    for path in sorted((capture / 'moments').glob('*.npz')):
        with np.load(path) as z:
            data[path.stem] = {k: z[k] for k in z.files}
    training = data['train_en']['H_sums'] + data['train_zh']['H_sums']
    counts = data['train_en']['counts'] + data['train_zh']['counts']
    total = counts[1:7].sum()
    mean = training[:,0,1:7].sum(1) / total
    second = training[:,1,1:7].sum(1) / total
    sd = np.sqrt(np.maximum(second-mean*mean, 1e-12))
    npz(out / 'training_scales.npz', mean=mean.astype(np.float32), standard_deviation=sd.astype(np.float32))
    profiles, composition, condition_moments = {}, [], []
    design = np.array([[1, *[(mask>>j)&1 for j in range(4)]] for mask in range(16)], dtype=np.float64)
    for name, d in data.items():
        count = d['counts'].astype(np.float64)
        h = d['H_sums']
        profiles[name+'_raw'] = (h[:,0] / np.maximum(count[None,:,None],1)).astype(np.float32)
        profiles[name+'_RMS'] = (h[:,2] / np.maximum(count[None,:,None],1)).astype(np.float32)
        profiles[name+'_train_z'] = ((profiles[name+'_raw']-mean[:,None,:])/sd[:,None,:]).astype(np.float32)
        profiles[name+'_count'] = count.astype(np.int64)
        u = d['unit_sums']
        # Conditional covariance is per native unit, not cross-coordinate correlation or causal mediation.
        upmean, phimean = u[:,1]/np.maximum(count[None,:,None],1), u[:,3]/np.maximum(count[None,:,None],1)
        cov = u[:,8]/np.maximum(count[None,:,None],1)-upmean*phimean
        profiles[name+'_gate_up_covariance'] = cov.astype(np.float32)
        condition_moments.append({'scope': name, 'tokens': int(count[:7].sum()), 'observed_piece_classes': int((count[:7]>0).sum()),
            'observed_cue_combinations': int((count[7:]>0).sum()),
            'all_unit_covariance_RMS_by_block': np.sqrt(np.mean(cov*cov,axis=(1,2))).tolist()})
    npz(out / 'full_coordinate_condition_profiles.npz', **profiles)
    # Exact least-squares sufficient-statistic comparison, every token and every native coordinate.
    # Outcomes are NEVER supplied as inputs. Cues are known prefix strings, and fits use only train moments.
    for lang in ('en','zh'):
        tr = data['train_'+lang]
        n = tr['counts'][7:].astype(float)
        xtx = design.T @ (n[:,None]*design)
        for li in range(37):
            sy = tr['H_sums'][li,0,7:]
            mu = sy.sum(0)/n.sum()
            beta = np.linalg.pinv(xtx, rcond=1e-12) @ design.T @ sy
            full = (sy+32*mu[None])/(n[:,None]+32)
            predictions = {'global_mean': np.repeat(mu[None],16,0), 'additive_four_cues': design@beta, 'full16_shrunk_cue_means': full}
            for split in ('validation','test'):
                d = data[split+'_'+lang]
                ne = d['counts'][7:].astype(float)
                sums, squares = d['H_sums'][li,0,7:], d['H_sums'][li,1,7:]
                for name, p in predictions.items():
                    squared = np.sum(squares-2*p*sums+ne[:,None]*p*p)
                    composition.append({'language': lang, 'split': split, 'layer': li, 'name': name,
                        'all_coordinate_MSE': float(squared/(ne.sum()*2560)), 'tokens': int(ne.sum()),
                        'observed_training_combinations': int((n>0).sum()), 'design_rank': int(np.linalg.matrix_rank(xtx))})
    save(out / 'cue_composition.json', {'timestamp': stamp(), 'reports': composition,
        'scope': 'Simple all-token prediction from four prefix lexical cues only, no current H. Full vector means preserve all2560 coordinates. Different df(1,rank<=5,up to16) explicitly retained; no semantic/causal attribution or unseen logical-combination claim.',
        'formula': 'XTX = sum_c n_c f_c f_c^T; XTY=sum_c f_c S_c; beta=pinv(XTX)XTY; SSE=sum_c(Q_c-2p_c S_c+n_c p_c^2). Counts include initial positions; primary ordinary plots are separate.'})
    report = {'timestamp': stamp(), 'source': snapshot(Path(__file__)), 'sources': len(selected), 'tokens': sum(r['tokens'] for r in source_stats),
        'noninitial_tokens': sum(r['tokens']-1 for r in source_stats), 'events': len(events), 'event_sources': len({r['sample_id'] for r in events}),
        'event_class_counts': dict(Counter(str(r['class']) for r in events)), 'event_onset_counts': dict(Counter(str(r['largest_increment_block']) for r in events)),
        'event_decline_counts': dict(Counter(str(r['strongest_log_decline_block']) for r in events)), 'condition_moments': condition_moments,
        'strata': {k: {'sources': len(rr), 'tokens': sum(r['tokens'] for r in rr), 'events': sum(r['events'] for r in rr),
            'next_NLL_document': clustered([r['native_next_NLL'] for r in rr],[r['source_group'] for r in rr]),
            'ordinary_layer_mean_energy': np.mean([r['ordinary_mean_H_energy'] for r in rr],0).tolist()}
            for k,rr in groups.items()},
        'coordinate_coverage': [37,2560], 'unit_coverage': [3,9728],
        'max_FP32_summary_vs_native_residual_energy_relative_discrepancy': max(r['max_residual_energy_relative_discrepancy'] for r in source_stats),
        'limits': ['Teacher-forced natural continuation probabilities are not question-answer accuracy or free-generation correctness.',
            'Cue means do not control all lexical/position/domain mixing and have different fit capacities.',
            'Event threshold inherited from2722 is a numerical stratification, not a discrete native gate.',
            'No formal syntactic annotation on new Wikipedia windows; historical GUM/UD graph results remain separately indexed.',
            'Full-field norm is squared amplitude, not physical energy or an intrinsic semantic metric.',
            'Confirmation outputs have not been captured or used in these fits.']}
    save(out / 'result.json', report)
    ledger('full_coordinate_ordinary_observation', time.monotonic()-start, sources=len(selected))
    print('ORDINARY_OBSERVATION_COMPLETE', {k:report[k] for k in ('sources','tokens','noninitial_tokens','events','event_class_counts','max_FP32_summary_vs_native_residual_energy_relative_discrepancy')}, flush=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
