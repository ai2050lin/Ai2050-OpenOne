"""CPU-only secondary analysis of the archived B28–B35 BF16 census."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.sparse import csr_matrix

import mlp_tail_goodfire as atlas
from shared.dist_protocol.mechanism_trace import file_sha, unbits


OUTPUT = atlas.OUT.with_name('mlp_tail_relative_encoding_v1')
DIRECTORIES = [atlas.OUT.with_name('mlp_tail_b28_b31_v1'), atlas.OUT]
WIDTH = 9728
DISC = slice(0, 16384)
HOLD = slice(24576, 32768)
SEED = 2863


def directory(block):
    return DIRECTORIES[int(block >= 32)]


def quantiles(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [.1, .5, .9]).tolist() if finite.size else None


def fraction(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 0)


def group_sum(ids, count):
    return csr_matrix((np.ones(len(ids), np.float32), (ids, np.arange(len(ids)))),
                      shape=(count, len(ids)))


def correlation(x, y):
    x = x - x.mean(0)
    y = y - y.mean(0)
    return fraction((x * y).sum(0, dtype=float),
                    np.sqrt((x * x).sum(0, dtype=float) * (y * y).sum(0, dtype=float)))


def analyze_layer(block, raw, tokens, disc_rows, hold_rows, disc_groups, hold_groups,
                  common_tokens, token_strings):
    profile = atlas.read(directory(block) / f'B{block}-profile.json')
    gate = raw(block, 'G')
    activation = raw(block, 'A')
    disc_counts = np.asarray(disc_groups.sum(1))
    hold_counts = np.asarray(hold_groups.sum(1))
    within, token_error, base_error, sign_mix, negative_gate, double_negative, energy = ([] for _ in range(7))
    most_frequent = int(np.argmax(hold_counts))
    illustration_std = []
    for start in range(0, WIDTH, 256):
        columns = slice(start, start + 256)
        discovery = unbits(activation[DISC, columns])
        all_hold = unbits(activation[HOLD, columns])
        x = all_hold[hold_rows]
        means = (hold_groups @ x) / hold_counts
        group_ids = np.searchsorted(common_tokens, tokens[HOLD][hold_rows])
        residual = x - means[group_ids]
        within.append(fraction(np.square(residual).sum(0, dtype=float),
                               np.square(x - x.mean(0)).sum(0, dtype=float)))
        trained_mean = (disc_groups @ discovery[disc_rows]) / disc_counts
        token_error.append(np.square(x - trained_mean[group_ids]).sum(0, dtype=float))
        base_error.append(np.square(x - discovery.mean(0)).sum(0, dtype=float))
        positive = hold_groups @ (x > 0).astype(np.float32)
        negative = hold_groups @ (x < 0).astype(np.float32)
        mixed = (positive >= .1 * hold_counts) & (negative >= .1 * hold_counts)
        sign_mix.append(mixed.mean(0))
        example_std = x[group_ids == most_frequent].std(0)
        illustration_std.append(np.where(mixed[most_frequent], example_std, -1))
        g = unbits(gate[HOLD, columns])
        a_squared = np.square(all_hold)
        energy.append(a_squared.sum(0, dtype=float))
        negative_gate.append((a_squared * (g < 0)).sum(0, dtype=float))
        double_negative.append((a_squared * (g < 0) * (all_hold > 0)).sum(0, dtype=float))
    within, token_error, base_error, sign_mix, negative_gate, double_negative, energy = (
        np.concatenate(parts) for parts in (within, token_error, base_error, sign_mix,
                                            negative_gate, double_negative, energy))
    illustration_std = np.concatenate(illustration_std)
    example = None
    if illustration_std.max() >= 0:
        unit = int(illustration_std.argmax())
        token = int(common_tokens[most_frequent])
        rows = np.flatnonzero(tokens[HOLD] == token) + HOLD.start
        values = unbits(activation[rows, unit])
        value = raw(block, 'V')
        negative_row = int(rows[values.argmin()])
        distinct = np.flatnonzero(rows // 16 != negative_row // 16)
        positive_row = int(rows[distinct[values[distinct].argmax()]])
        example = {'unit': unit, 'token_id': token, 'token': token_strings.get(str(token)),
                   'points': [{'document_index': row // 16, 'position_index': row % 16,
                               'G': float(unbits(gate[row, unit])),
                               'V': float(unbits(value[row, unit])),
                               'A': float(unbits(activation[row, unit]))}
                              for row in (negative_row, positive_row)]}
    metrics = {'A_std': quantiles(profile['fields']['A']['std']),
               'within_token_variance_fraction': quantiles(within),
               'token_predictor_fvu': quantiles(fraction(token_error, base_error)),
               'pooled_token_predictor_fvu': float(token_error.sum() / base_error.sum()),
               'fraction_token_groups_with_both_signs': quantiles(sign_mix),
               'negative_gate_A_squared_fraction': quantiles(fraction(negative_gate, energy)),
               'double_negative_A_squared_fraction': quantiles(fraction(double_negative, energy)),
               'pooled_double_negative_A_squared_fraction': float(double_negative.sum()/energy.sum()),
               'gate_value_sign_regime_means': np.asarray(profile['sign_regime_fractions']).mean(0).tolist(),
               'illustration': example}
    unit_values = np.column_stack((within, fraction(token_error, base_error), sign_mix,
                                   fraction(negative_gate, energy), fraction(double_negative, energy)))
    return metrics, unit_values


def summarize_edges(source, matrix, raw, hold_rows, group_ids, groups, sampled):
    index = np.arange(WIDTH)
    rng = np.random.default_rng(SEED + source)
    targets = {'same_index': index, 'random': rng.permutation(WIDTH)}
    discovery = matrix(source, 'rho-discovery')
    for label, values in [('discovery_peak', discovery), ('gate_alignment_peak', matrix(source, 'weight-gate')),
                          ('up_alignment_peak', matrix(source, 'weight-up'))]:
        targets[label] = np.concatenate([np.abs(values[lo:lo+128]).argmax(1)
                                         for lo in range(0, WIDTH, 128)])
    confirm = matrix(source, 'rho-confirm')
    holdout = matrix(source, 'rho-holdout')
    counts = np.asarray(groups.sum(1))
    x = unbits(raw(source, 'A')[HOLD][np.ix_(hold_rows, sampled)])
    x_centered = x - ((groups @ x) / counts)[group_ids]
    target_field = raw(source + 1, 'A')[HOLD]
    summary = {}
    arrays = {}
    for label, target in targets.items():
        d, c, h = (m[index, target] for m in (discovery, confirm, holdout))
        y = unbits(target_field[np.ix_(hold_rows, target[sampled])])
        y_centered = y - ((groups @ y) / counts)[group_ids]
        raw_r = correlation(x, y)
        partial_r = correlation(x_centered, y_centered)
        summary[label] = {'median_abs_r': [float(np.median(np.abs(r))) for r in (d, c, h)],
                          'aligned_holdout_r': quantiles(np.sign(d) * h),
                          'discovery_holdout_same_sign_fraction': float(np.mean(d * h > 0)),
                          'same_sign_and_abs_r_at_least_0p1_in_all_splits': int(np.sum(
                              (d * h > 0) & (d * c > 0) & (abs(d) >= .1) & (abs(c) >= .1) & (abs(h) >= .1))),
                          'eligible_subset_median_abs_r': quantiles(abs(raw_r))[1],
                          'same_token_residual_median_abs_r': quantiles(abs(partial_r))[1]}
        arrays[label + '_full'] = np.column_stack((index, target, d, c, h))
        arrays[label + '_within_token'] = np.column_stack((sampled, target[sampled], raw_r, partial_r))
    return summary, arrays


def main():
    OUTPUT.mkdir(exist_ok=True)
    started = time.monotonic()
    git = ['git', '-c', f'safe.directory={atlas.ROOT.as_posix()}']
    provenance = {'git_sha': subprocess.check_output(git+['rev-parse','HEAD'], text=True).strip(),
                  'git_dirty': bool(subprocess.check_output(git+['status','--porcelain'])),
                  'model_path': str(atlas.MODEL), 'model_revision': atlas.SNAPSHOT, 'seed': SEED}
    manifests = [atlas.read(path / 'manifest.json') for path in DIRECTORIES]
    plans = [atlas.read(path / 'plan.json') for path in DIRECTORIES]
    assert all(m['state'] == 'complete' and m['model_snapshot'] == atlas.SNAPSHOT for m in manifests)
    positions = manifests[0]['positions']
    assert len(positions) == 16 and all(p['positions'] == positions for p in plans + manifests)
    assert all(p['source_tokens_sha256'] == file_sha(atlas.TOKENS) for p in plans)
    assert manifests[0]['base_manifest_sha256'] == file_sha(DIRECTORIES[1]/'manifest.json')
    contract = {'provenance': provenance, 'scope': 'secondary descriptive analysis of exposed material',
                'positions': positions, 'eligibility': 'token present at >=8 positions in >=4 documents in both discovery and holdout',
                'sign_mixing': '>=10% positive and >=10% negative within each eligible holdout token group',
                'edge_diagnostic_sample': '1024 source IDs drawn with fixed seed, before inspecting results',
                'source_manifests': {str(p): file_sha(p / 'manifest.json') for p in DIRECTORIES},
                'source_code_sha256': file_sha(__file__),
                'time': datetime.now(ZoneInfo('America/Chicago')).isoformat()}
    atlas.save_json(OUTPUT/'analysis_plan.json', contract)
    checked = {}
    records = {path: {x['path']: x for x in m['files']} for path,m in zip(DIRECTORIES,manifests)}

    def verify(path):
        if str(path) not in checked:
            record = records[path.parent][path.name]
            assert path.stat().st_size == record['bytes'] and file_sha(path) == record['sha256'], path
            checked[str(path)] = record['sha256']
        return path

    def raw(block, field):
        a = np.load(verify(directory(block)/f'B{block}-{field}.npy'), mmap_mode='r')
        assert a.shape == (32768, WIDTH) and a.dtype == np.uint16
        return a

    def matrix(source, field):
        a = np.load(verify(directory(source)/f'B{source}-B{source+1}-{field}.npy'), mmap_mode='r')
        assert a.shape == (WIDTH, WIDTH)
        return a

    def progress(stage, done):
        elapsed = time.monotonic() - started
        item = {'stage': stage, 'done': done, 'total': 15, 'seconds': round(elapsed, 2),
                'eta_seconds': round(elapsed/done*(15-done), 1) if done else None}
        line = json.dumps(item)
        print(line, flush=True)
        with (OUTPUT/'progress.jsonl').open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    progress('verified_input_identity', 0)
    tokens = np.load(atlas.TOKENS, mmap_mode='r')[:, positions].reshape(-1)
    eligible = []
    for split in (DISC, HOLD):
        ids = tokens[split]
        values, counts = np.unique(ids, return_counts=True)
        document_tokens = np.unique(np.column_stack((np.arange(len(ids))//16, ids)), axis=0)[:, 1]
        documents = np.bincount(document_tokens, minlength=int(values[-1])+1)
        eligible.append(values[(counts >= 8) & (documents[values] >= 4)])
    common = np.intersect1d(*eligible)
    disc_rows, hold_rows = (np.flatnonzero(np.isin(tokens[s], common)) for s in (DISC, HOLD))
    disc_ids, hold_ids = (np.searchsorted(common,tokens[s][rows])
                          for s,rows in ((DISC,disc_rows),(HOLD,hold_rows)))
    disc_groups, hold_groups = group_sum(disc_ids,len(common)), group_sum(hold_ids,len(common))
    documents = atlas.read(verify(DIRECTORIES[1]/'documents.json'))
    layers = {}
    unit_data = {}
    for block in range(28,36):
        verify(directory(block)/f'B{block}-profile.json')
        layers[str(block)], unit_data[f'B{block}'] = analyze_layer(
            block, raw, tokens, disc_rows, hold_rows, disc_groups, hold_groups, common, documents['token_strings'])
        for point in (layers[str(block)]['illustration'] or {}).get('points', []):
            point['token_position'] = positions[point['position_index']]
        progress(f'layer_{block}',block-27)
    np.savez_compressed(OUTPUT/'per_unit.npz', **unit_data, columns=np.array([
        'within_token_variance_fraction','token_predictor_fvu','fraction_token_groups_with_both_signs',
        'negative_gate_A_squared_fraction','double_negative_A_squared_fraction']))
    edges = {}
    sampled = np.sort(np.random.default_rng(SEED).choice(WIDTH,1024,replace=False))
    for source in range(28,35):
        name = f'B{source}-B{source+1}'
        edges[name], arrays = summarize_edges(source, matrix, raw, hold_rows, hold_ids, hold_groups, sampled)
        np.savez_compressed(OUTPUT/f'{name}.npz', **arrays)
        progress(name, source-19)
    atlas.save_json(OUTPUT/'report.json', {**contract,'verified_inputs': checked,
                    'eligible_token_types': int(len(common)), 'eligible_discovery_positions': int(len(disc_rows)),
                    'eligible_holdout_positions': int(len(hold_rows)),
                    'eligible_holdout_documents': int(len(np.unique(hold_rows//16))),
                    'layers': layers,'edges': edges,'seconds': time.monotonic()-started})


if __name__ == '__main__':
    main()
