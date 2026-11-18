"""Recompute paired suffix-region comparisons from the complete frozen run."""
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

import source_binding_qkv_suffix_sites as suffix
import source_binding_transfer_factors as io
from source_binding_design import sha


REGIONS = ('between', 'after', 'tail')
ARMS = ('H', *REGIONS, 'question', 'all_suffix')
RECORDS_PER_FAMILY = 16


def load():
    root = suffix.OUT
    parent = suffix.exp.OUT
    design = io.read('design.json')
    manifest = io.read('manifest.json')
    source = json.loads((parent / 'holdout-manifest.json').read_bytes())
    assert manifest['passed'] and source['passed']
    assert design['main_design_sha256'] == sha(parent / 'design.json')
    assert design['code'] == sha(suffix.__file__)
    mats = {}
    for task, digest in design['files'].items():
        name = f'{task}-materials.json'
        assert sha(root / name) == digest
        rows = io.read(name)
        assert len(rows) == design['families'] * RECORDS_PER_FAMILY
        assert len({r['family'] for r in rows}) == design['families']
        for row in rows:
            assert row['cue'] == 0 and row['task'] == task and row['id'] not in mats
            mats[row['id']] = row
    targets = {}
    for shard in source['shards']:
        path = parent / shard['file']
        assert sha(path) == shard['sha256']
        for group in json.loads(path.read_bytes())['groups']:
            for i, id_ in enumerate(group['ids']):
                if id_ in mats:
                    assert id_ not in targets
                    targets[id_] = {k: v[i][0] for k, v in group['items'].items() if k == 'base' or k.startswith('W|')}
    assert targets.keys() == mats.keys()
    seen = set()
    events = defaultdict(list)
    controls = defaultdict(lambda: dict(n=0, score_difference=0.0, forced_correct=0))
    for shard in manifest['shards']:
        assert sha(root / shard['name']) == shard['sha256']
        for group in io.read(shard['name'])['items']:
            for i, base in enumerate(group['base']):
                id_ = base['id']
                assert id_ not in seen
                seen.add(id_)
                row, ref = mats[id_], targets[id_]
                assert base['pred_record'] == ref['base']
                audit = controls[row['task']]
                audit['n'] += 1
                audit['forced_correct'] += base['forced_correct']
                for arm in ('noop', 'self'):
                    result = group[arm][i]
                    assert result['id'] == id_ and result['pred_record'] == base['pred_record']
                    diff = max(abs(a - b) for a, b in zip(result['scores'], base['scores'], strict=True))
                    assert diff <= 1.0
                    audit['score_difference'] = max(audit['score_difference'], diff)
                for site, layer in suffix.CELLS:
                    cell = f'{site}@{layer}'
                    wide = ref[f'W|{cell}']
                    outcomes = []
                    for arm in ARMS:
                        candidate = group[f'{arm}|{cell}'][i]
                        assert candidate['id'] == id_
                        outcomes.append(int(candidate['pred_record'] == wide))
                    events[row['task']].append((row['family'], site, wide != ref['base'], outcomes))
    assert seen == set(mats)
    return events, dict(controls)


def summarize(events, seed):
    rng = np.random.default_rng(seed)
    families = sorted({f for f, *_ in events})
    index = {f: i for i, f in enumerate(families)}
    draws = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=20000)
    scopes = {}
    for scope in ('all', *suffix.exp.SITES):
        counts = np.zeros((len(families), len(ARMS) + 1), dtype=np.int64)
        total = 0
        for family, site, changed, hits in events:
            if scope != 'all' and site != scope:
                continue
            total += 1
            if changed:
                counts[index[family], 0] += 1
                counts[index[family], 1:] += hits
        actual = counts.sum(0)
        bs = draws @ counts
        bs = bs[bs[:, 0] > 0]
        rates = bs[:, 1:] / bs[:, :1]
        means = actual[1:] / actual[0]
        arms = {k: dict(hits=int(actual[i + 1]), rate=float(means[i]), gain=float(means[i] - means[0]))
                for i, k in enumerate(ARMS)}
        for i, k in enumerate(ARMS):
            arms[k]['gain_ci95'] = np.quantile(rates[:, i] - rates[:, 0], [.025, .975]).tolist()
        pairs = {}
        for a, b in itertools.combinations(REGIONS, 2):
            i, j = ARMS.index(a), ARMS.index(b)
            interval = np.quantile(rates[:, i] - rates[:, j], [.05 / 6, 1 - .05 / 6])
            pairs[f'{a}_minus_{b}'] = dict(delta=float(means[i] - means[j]), ci98_33=interval.tolist())
        scopes[scope] = dict(total_edits=total, changed=int(actual[0]), arms=arms, pairwise=pairs,
                             family_counts=dict(zip(families, counts.tolist(), strict=True)))
    return scopes


def main():
    io.OUT = suffix.OUT
    events, controls = load()
    tasks = {task: summarize(values, suffix.exp.SEED + i) for i, (task, values) in enumerate(sorted(events.items()))}
    gap = {k: tasks['colours']['all']['arms'][k]['rate'] - tasks['zh']['all']['arms'][k]['rate'] for k in ARMS}
    rng = np.random.default_rng(suffix.exp.SEED + 2)
    boot_gains = []
    for task in ('colours', 'zh'):
        counts = np.array(list(tasks[task]['all']['family_counts'].values()))
        n = len(counts)
        draws = rng.multinomial(n, np.full(n, 1 / n), size=20000)
        bs = draws @ counts
        assert np.all(bs[:, 0] > 0)
        boot_gains.append((bs[:, 1:] - bs[:, 1:2]) / bs[:, :1])
    gap_closed_ci = np.quantile(boot_gains[1] - boot_gains[0], [.025, .975], axis=0).T.tolist()
    io.save('region-comparison.json', dict(tasks=tasks, controls=controls, gap=gap,
        gap_closed={k: gap['H'] - d for k, d in gap.items()}, gap_closed_ci95=dict(zip(ARMS, gap_closed_ci, strict=True)),
        manifest_sha256=sha(suffix.OUT / 'manifest.json'),
        analysis_sha256=sha(Path(__file__)), method='Paired by prompt and edit, family bootstrap, 98.33% marginal intervals for three overall region comparisons per task. Site strata descriptive.'))
    for task, s in tasks.items():
        print(task, s['all']['changed'], s['all']['arms'], s['all']['pairwise'], flush=True)
    print('gap', gap, flush=True)


if __name__ == '__main__':
    main()
