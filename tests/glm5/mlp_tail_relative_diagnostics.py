"""Nuisance-factor sensitivity, spot checks and figure for the archived census."""

import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

import mlp_tail_relative_encoding as study
from shared.dist_protocol.mechanism_trace import file_sha, unbits


def main():
    start = time.monotonic()
    output = study.OUTPUT
    report_path = output / 'report.json'
    report = study.atlas.read(report_path)
    tokens = np.load(study.atlas.TOKENS, mmap_mode='r')[:, report['positions']].reshape(-1)
    eligible_tokens = []
    for split in (study.DISC, study.HOLD):
        ids = tokens[split]
        token_ids, counts = np.unique(ids, return_counts=True)
        pairs = np.unique(np.column_stack((np.arange(len(ids)) // 16, ids)), axis=0)
        doc_counts = np.bincount(pairs[:, 1], minlength=int(token_ids.max()) + 1)
        eligible_tokens.append(token_ids[(counts >= 8) & (doc_counts[token_ids] >= 4)])
    rows = np.flatnonzero(np.isin(tokens[study.HOLD], np.intersect1d(*eligible_tokens)))
    assert len(rows) == report['eligible_holdout_positions']
    memberships = [np.unique(ids, return_inverse=True)[1]
                   for ids in (tokens[study.HOLD][rows], rows // 16, rows % 16)]
    sizes = np.array([len(set(ids)) for ids in memberships])
    columns = np.concatenate([ids + offset for ids, offset in zip(memberships, np.r_[0, sizes.cumsum()[:-1]])])
    design = csr_matrix((np.ones(len(columns)), (np.tile(np.arange(len(rows)),3), columns)),
                        shape=(len(rows), int(sizes.sum())))
    gram = (design.T @ design).toarray()
    eig, vectors = eigh(gram)
    kept = eig > eig.max()*1e-10
    basis = vectors[:, kept] / np.sqrt(eig[kept])
    inverse = basis @ basis.T
    log_path = output / 'diagnostics_progress.jsonl'

    def residual(x):
        return x - design @ (inverse @ (design.T @ x))

    def log(stage, done):
        elapsed = time.monotonic()-start
        line = json.dumps({'stage': stage, 'done': done, 'total': 7, 'seconds': elapsed,
                           'eta_seconds': elapsed/done*(7-done) if done else None})
        print(line, flush=True)
        with log_path.open('a',encoding='utf-8') as handle:
            handle.write(line+'\n')

    results = {}
    log('fixed_effects_start', 0)
    for block in range(28,35):
        name = f'B{block}-B{block+1}'
        with np.load(output/f'{name}.npz') as selections:
            methods = ['discovery_peak','same_index','random','gate_alignment_peak','up_alignment_peak']
            source = selections['discovery_peak_within_token'][:,0].astype(int)
            x_map, y_map = (np.load(study.directory(b)/f'B{b}-A.npy',mmap_mode='r')[study.HOLD]
                            for b in (block,block+1))
            x = unbits(x_map[np.ix_(rows,source)])
            rx = residual(x)
            maximum_group_mean = float(np.max(np.abs(design.T @ rx)))
            assert maximum_group_mean < 1e-5
            results[name] = {}
            for method in methods:
                target = selections[method+'_within_token'][:,1].astype(int)
                full = selections[method+'_full']
                for i in (0, len(source)//2, len(source)-1):
                    observed = np.corrcoef(unbits(x_map[:, source[i]]), unbits(y_map[:,target[i]]))[0,1]
                    assert np.isclose(observed,full[source[i],4],atol=2e-5), (name,method,i,observed)
                y = unbits(y_map[np.ix_(rows,target)])
                after = study.correlation(rx,residual(y))
                results[name][method] = {'median_abs_r': study.quantiles(abs(after))[1],
                                         'aligned_r': study.quantiles(after * np.sign(full[source,2]))}
            log(name, block-27)
    study.atlas.save_json(output/'fixed_effects.json', {'provenance': report['provenance'],
                         'source_report_sha256': file_sha(report_path), 'source_sha256': file_sha(__file__),
                         'scope':'descriptive within archived holdout; fitted additive token/document/position effects on this subset',
                         'columns_by_nuisance': sizes.tolist(),'design_rank': int(kept.sum()),
                         'sampled_edges_per_method':1024,'correlation_spot_checks':105,'edges':results})
    make_figure(report, results, output)


def make_figure(report, adjusted, output):
    layers = list(report['layers'].values())
    boundaries = list(report['edges'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    axes[0].plot(range(28,36),[x['within_token_variance_fraction'][1] for x in layers],'o-',color='#237fa0')
    axes[0].set(title='Same-token variation',ylabel='Median within-token / total variance',xlabel='Block',ylim=(0,1.02))
    axes[1].plot(range(28,36),[x['A_std'][1] for x in layers],'o-',color='#b45830')
    axes[1].set(title='Native MLP product',ylabel='Median per-unit standard deviation',xlabel='Block')
    for name, values in [('Discovery peaks', [report['edges'][b]['discovery_peak']['median_abs_r'][0] for b in boundaries]),
                         ('Same pairs, holdout', [report['edges'][b]['discovery_peak']['median_abs_r'][2] for b in boundaries]),
                         ('Adjusted holdout subset', [adjusted[b]['discovery_peak']['median_abs_r'] for b in boundaries]),
                         ('Adjusted random pairs', [adjusted[b]['random']['median_abs_r'] for b in boundaries])]:
        axes[2].plot(range(7),values,'o-',label=name)
    axes[2].set(title='Adjacent MLP association',ylabel='Median |Pearson r|',xticks=range(7),
                xticklabels=[f'{i}→{i+1}' for i in range(28,35)])
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output/'relative_encoding.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    with threadpool_limits(4):
        main()
