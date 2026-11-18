"""Audit paired factor results without running the model."""
import argparse
from collections import defaultdict
import itertools

import numpy as np

import source_binding_transfer_factors as design
from source_binding_design import sha

AXES = design.FACTORS
TAGS = [design.tag(bits) for bits in design.CONDITIONS]
CONTROLS = ('H16', 'all', 'R0')


def count(base, wide, pred, keep):
    changed = (base != wide) & keep
    hit = (pred == wide) & keep
    return np.array([keep.sum(), changed.sum(), hit.sum(), (changed & hit).sum(),
                     ((base == wide) & keep & ~hit).sum()], dtype=np.int64)


def result_counts(c):
    n, changed, matched, changed_matched, unchanged_missed = map(int, c)
    return dict(n=n, wide_changed=changed, matched=matched, changed_matched=changed_matched,
                unchanged_missed=unchanged_missed, changed_rate=changed/n, agreement=matched/n,
                changed_agreement=changed_matched/changed if changed else None, net_gain=(changed_matched-unchanged_missed)/n)


def load(split):
    manifest = design.read(f'{split}-edits.json')
    assert manifest['passed']
    return load_shards(split, manifest['shards'], manifest.get('families'))


def load_shards(split, shards, family_limit=None):
    rows = design.read(f'{split}-materials.json')
    assert sha(design.OUT / f'{split}-materials.json') == design.read('design.json')['materials'][split]
    if family_limit is not None:
        rows = rows[:family_limit*design.CHUNK]
    by_group = defaultdict(dict)
    offset = 0
    for record in shards:
        name = record['file']
        assert sha(design.OUT / name) == record['sha256']
        shard = design.read(name)
        length = len(shard['ids'])
        chunk = rows[offset:offset+length]
        assert shard['ids'] == [r['id'] for r in chunk]
        assert shard['donors'] == {k.split('|', 1)[1]: v['donor'] for k, v in design.plan_arms(chunk).items() if k.startswith('wide|')}
        for i, row in enumerate(chunk):
            base = shard['items']['base'][i][0]
            for label, outcomes in shard['items'].items():
                if not label.startswith('wide|'):
                    continue
                suffix = label.split('|', 1)[1]
                site, edit = suffix.split('@')
                layer, rep = map(int, edit.split('#'))
                pred = [shard['items'][f'{arm}|{suffix}'][i][0] if f'{arm}|{suffix}' in shard['items'] else -1 for arm in CONTROLS]
                donor = chunk[shard['donors'][suffix][i]]
                direct_target = 2*(donor['source'] if site != 'qent' else row['source']) + (donor['entity'] if site != 'qsrc' else row['entity'])
                key = row['source'], row['entity'], row['order'], site, layer, rep
                group = by_group[row['pair_family'], row['condition']]
                assert key not in group
                group[key] = (base, outcomes[i][0], *pred, direct_target)
        offset += length
    assert offset == len(rows)
    first = next(iter(by_group.values()))
    keys = sorted(first)
    families = sorted({family for family, _ in by_group})
    matrix = np.array([[[by_group[f, condition][key] for key in keys] for condition in TAGS] for f in families], dtype=np.int8)
    return keys, matrix


def comparisons(keys, data):
    primary = np.array([key[-1] < 3 for key in keys])
    all_edits = np.ones(len(keys), bool)
    nf = len(data)
    rng = np.random.default_rng(design.SEED)
    draws = rng.multinomial(nf, np.full(nf, 1/nf), size=8000)

    def pack(delta, n, level):
        boot_n = draws @ n
        boot = (draws @ delta)[boot_n > 0] / boot_n[boot_n > 0]
        tail = (1-level)/2
        return dict(delta=float(delta.sum()/n.sum()) if n.sum() else None, n=int(n.sum()),
                    families=int((n > 0).sum()), interval_level=level,
                    interval=np.quantile(boot, [tail, 1-tail]).tolist() if len(boot) else None)

    def contrasts(edges, mask, level):
        left = data[:, [i for i, _ in edges]][:, :, mask]
        right = data[:, [j for _, j in edges]][:, :, mask]
        bl, wl, hl = [left[..., i] for i in range(3)]
        br, wr, hr = [right[..., i] for i in range(3)]
        cl, cr = wl != bl, wr != br
        hl, hr = hl == wl, hr == wr
        both = cl & cr
        same = both & (wl == wr)
        gain_diff = (hr.astype(int) - ~cr) - (hl.astype(int) - ~cl)
        flatten = lambda a: a.sum(axis=(1, 2))
        return dict(gain=pack(flatten(gain_diff), np.full(nf, len(edges)*mask.sum()), level),
                    changed_agreement=pack(flatten((hr.astype(int)-hl)*both), flatten(both), level),
                    same_target_agreement=pack(flatten((hr.astype(int)-hl)*same), flatten(same), level),
                    wide_change_rate=pack(flatten(cr.astype(int)-cl), np.full(nf, len(edges)*mask.sum()), level))

    def interaction(first, second):
        corners = []
        for i,code in enumerate(TAGS):
            if code[first]=='0' and code[second]=='0':
                a = list(code)
                a[first] = '1'
                j = TAGS.index(''.join(a))
                a[second] = '1'
                k = TAGS.index(''.join(a))
                a[first] = '0'
                m = TAGS.index(''.join(a))
                corners.append((i,j,m,k))
        x = data[:,corners]
        base,wide,head = [x[...,i] for i in range(3)]
        changed = base!=wide
        gain = (head==wide).astype(int) - ~changed
        d = gain[:,:,3] - gain[:,:,2] - gain[:,:,1] + gain[:,:,0]
        return pack(d.sum(axis=(1,2)),np.full(nf,len(corners)*len(keys)),.95)

    values = {}
    for k, code in enumerate(TAGS):
        base, wide, *preds, direct = np.moveaxis(data[:, k], -1, 0)
        values[code] = {}
        strata = {'all': all_edits, 'primary': primary, **{f'{s}@{l}': np.array([key[-3:-1] == (s, l) for key in keys]) for s, l in itertools.product(design.SITES, design.LAYERS)}}
        for name, keep in strata.items():
            values[code][name] = {arm: result_counts(count(base, wide, pred, keep & (pred >= 0))) for arm, pred in zip(CONTROLS, preds)}
            values[code][name]['donor_record'] = result_counts(count(base, wide, direct, keep))

    factors = {}
    for axis, name in enumerate(AXES):
        edges = [(i, TAGS.index(code[:axis]+'1'+code[axis+1:])) for i, code in enumerate(TAGS) if code[axis] == '0']
        factors[name] = dict(average=contrasts(edges, all_edits, .9875), primary=contrasts(edges, primary, .9875),
                             edges={f'{TAGS[i]}-{TAGS[j]}': contrasts([(i,j)], all_edits, .95) for i,j in edges})
    interactions = {f'{AXES[a]}:{AXES[b]}':interaction(a,b) for a,b in itertools.combinations(range(4),2)}
    return dict(conditions=values, factors=factors, diagonal=contrasts([(0,15)], all_edits, .95), interactions_descriptive=interactions,
                families=nf, records_per_condition=nf*len(keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=tuple(design.FAMILIES), default='holdout')
    args = parser.parse_args()
    keys, data = load(args.split)
    report = comparisons(keys, data)
    report['run'] = design.RUN
    report['split'] = args.split
    report['design_sha256'] = sha(design.OUT/'design.json')
    report['items_manifest_sha256'] = sha(design.OUT/f'{args.split}-edits.json')
    report['analysis_sha256'] = sha(__file__)
    report['formal_evidence_promotion'] = False
    name = 'report.json' if args.split == 'holdout' else 'pilot-analysis.json'
    design.save(name, report)
    print('condition       H16          all       random    H16 all agreement     change')
    for code, scopes in report['conditions'].items():
        a = scopes['all']
        print(code, *(f'{100*a[k]["changed_agreement"]:8.2f}' for k in CONTROLS),
              f'{100*a["H16"]["agreement"]:12.2f}', f'{100*a["H16"]["changed_rate"]:15.2f}')
    for name, stats in report['factors'].items():
        print(name, stats['average'])


if __name__ == '__main__':
    main()
