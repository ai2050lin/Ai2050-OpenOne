"""Independent CPU admission and native-coordinate maps; never chooses a candidate.

Reads discovery by default. Confirmation requires an already sealed candidate.
Engineering integrity, observed contrasts, prediction and causality stay separate.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
import numpy as np
from deploy.storage_policy import checked_dir, check_write

SOURCE = ROOT / 'tests/glm5/result/conditional_lookup_native_v1'
OUTPUT = ROOT / 'tests/glm5/result/conditional_lookup_stage_review_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bits_to_float(a):
    assert a.dtype == np.uint16, 'native BF16 bits required'
    return (a.astype(np.uint32) << 16).view(np.float32)


def verify_rows(rows, scores, expected_families):
    """Reject missing cells, duplicate families/IDs, score and future-cue leaks."""
    assert len(rows) == len(scores) == 16 * expected_families
    assert len({r['id'] for r in rows}) == len(rows), 'duplicate sample ID'
    index = {}
    for r, s in zip(rows, scores):
        assert r['id'] == s['id'], 'missing/reordered score'
        key = (r['family'], r['r0'], r['r1'], r['b'], r['late'])
        assert key not in index, 'duplicate family condition'
        index[key] = r
        gold = 2 * r['b'] + (r['r0'], r['r1'])[r['b']]
        assert gold == r['target_record']
        assert r['gold_id'] == r['choice_ids'][gold]
        vals = s['scores']
        assert len(vals) == 4 and np.isfinite(vals).all()
        assert s['gold_margin'] == vals[gold] - max(v for j, v in enumerate(vals) if j != gold)
        assert s['top1_correct'] == (s['top1'] == r['gold_id'])
        assert s['forced_correct'] == (int(np.argmax(vals)) == gold)
    families = sorted({r['family'] for r in rows})
    assert len(families) == expected_families
    cells = []
    for cell in itertools.product((0, 1), repeat=4):
        ids = [i for i, r in enumerate(rows) if (r['r0'], r['r1'], r['b'], r['late']) == cell]
        assert len(ids) == expected_families, 'incomplete condition cell'
        cells.append(dict(cell=list(cell), n=len(ids),
                          strict=sum(scores[i]['gold_margin'] > 0 for i in ids) / len(ids),
                          top1=sum(scores[i]['top1_correct'] for i in ids) / len(ids)))
    for f, a, b in itertools.product(families, (0, 1), (0, 1)):
        r, s = index[f, a, b, 0, 1], index[f, a, b, 1, 1]
        assert r['entity_pos'] == s['entity_pos']
        pos = r['entity_pos'] + 1
        assert r['tokens'][:pos] == s['tokens'][:pos], 'future cue visible before readout'
    return cells


def main(split):
    import torch
    torch.set_num_threads(4)
    start = time.monotonic()
    checked_dir(OUTPUT, 64 << 20, operation='CLT independent audit')
    destination = OUTPUT / f'{split}-audit.json'
    assert not destination.exists(), 'immutable audit exists'
    pre = read(SOURCE / 'prereg.json')
    plan = read(SOURCE / 'natural-plan.json')
    assert sha(SOURCE / 'prereg.json') == plan['parent_sha']
    for name, digest in plan['code'].items():
        assert sha(ROOT / 'tests/glm5' / name) == digest
    if split == 'confirm':
        seal = read(SOURCE / 'candidate.json')
        assert seal['analysis_sha'] == sha(ROOT / 'tests/glm5/conditional_lookup_analyze.py')
        for name, digest in seal['files'].items():
            assert sha(SOURCE / name) == digest
    collection = read(SOURCE / f'{split}-collection.json')
    assert sha(SOURCE / f'{split}-materials.json') == pre['materials'][split]['sha256']
    rows = read(SOURCE / f'{split}-materials.json')
    all_scores, counts, maps, stderr, scale_maps = [], [], {}, {}, {}
    for shard_no, (name, digest) in enumerate(sorted(collection['shards'].items())):
        file = SOURCE / name
        assert sha(file) == digest
        summary = read(file.with_suffix('.json'))
        assert summary['native_sha'] == digest and summary['plan_sha'] == sha(SOURCE / 'natural-plan.json')
        assert summary['provenance'] == pre['provenance']
        offset = len(all_scores)
        batch = rows[offset:offset + len(summary['ids'])]
        assert [r['id'] for r in batch] == summary['ids']
        all_scores.extend(summary['scores'])
        idx = {(r['family'], r['r0'], r['r1'], r['b'], r['late']): i for i, r in enumerate(batch)}
        fam = sorted({r['family'] for r in batch})
        pick = np.array([[idx[f, a, b, 0, 1] for a, b in itertools.product((0, 1), repeat=2)] for f in fam])
        pairs = [(idx[f, a, b, 0, 1], idx[f, a, b, 1, 1]) for f, a, b in itertools.product(fam, (0, 1), (0, 1))]
        record = dict(shard=name, families=len(fam), prefix_pairs=len(pairs), gua={})
        with np.load(file) as data:
            for layer in range(41):
                key = f'h{layer:02}'
                a = data[key]
                assert all(np.array_equal(a[i], a[j]) for i, j in pairs)
            for layer in (20, 21, 22):
                g, u, a = [torch.from_numpy(data[f'{k}{layer}'].copy()).view(torch.bfloat16) for k in ('g', 'u', 'a')]
                rebuilt = torch.nn.functional.silu(g) * u
                record['gua'][str(layer)] = dict(bitwise=bool(torch.equal(rebuilt, a)), max_abs=float((rebuilt.float() - a.float()).abs().max()))
                assert record['gua'][str(layer)]['bitwise'], 'native GUA arithmetic failed'
            for layer in range(20, 31):
                mass = data[f'mass{layer}']
                assert np.isfinite(mass).all() and mass.min() >= -1e-7
                assert np.allclose(mass.sum(-1), 1, atol=2e-5, rtol=0), 'source partition is not normalized'
            # All coordinates retained in original order; contrasts are observations.
            for key in [f'h{i:02}' for i in range(41)] + [f'{k}{b}' for k in ('g', 'u', 'a') for b in (20, 21, 22)]:
                x = bits_to_float(data[key])[pick].astype(np.float64)
                contrasts = np.einsum('fc, ncd->nfd', np.array([[-1, -1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]) / 4, x)
                maps.setdefault(key, []).append(contrasts)
                scale_maps.setdefault(key, []).append(x)
        assert summary['source_audit_max'] <= collection['tolerance']
        counts.append(record)
        print(json.dumps(dict(stage='independent_shard_audit', split=split, completed=shard_no + 1,
                              total=len(collection['shards']), elapsed_seconds=time.monotonic() - start,
                              eta_seconds=(time.monotonic() - start) / (shard_no + 1) * (len(collection['shards']) - shard_no - 1))), flush=True)
    cells = verify_rows(rows, all_scores, pre['materials'][split]['families'])
    behavior_pass = all(min(c['strict'], c['top1']) >= .95 for c in cells)
    assert collection['behavior_pass'] == behavior_pass
    assert collection['technical_pass']
    derived = {}
    for key, chunks in maps.items():
        x = np.concatenate(chunks)
        background = np.concatenate(scale_maps[key]).reshape(-1, x.shape[-1])
        sd = background.std(0)
        derived[key + '_contrast_mean'] = x.mean(0).astype(np.float32)
        derived[key + '_contrast_se'] = (x.std(0, ddof=1) / np.sqrt(len(x))).astype(np.float32)
        derived[key + '_state_sd'] = sd.astype(np.float32)
        derived[key + '_standardized_contrast'] = np.divide(x.mean(0), sd[None], out=np.zeros_like(x.mean(0)), where=sd[None] > 1e-8).astype(np.float32)
    derived['contrast_labels'] = np.array(['r0', 'r1', 'r0_times_r1'])
    native = OUTPUT / f'{split}-full-coordinate-maps.npz'
    check_write(native, 16 << 20, operation='CLT native maps')
    assert not native.exists()
    np.savez_compressed(native, **derived)
    result = dict(schema='clt_independent_stage_audit.v1', kind='real', split=split,
                  provenance=pre['provenance'], auditor_sha256=sha(Path(__file__)),
                  collection_sha256=sha(SOURCE / f'{split}-collection.json'),
                  families=pre['materials'][split]['families'], inputs=len(rows), cells=cells,
                  strict_correct=sum(s['gold_margin'] > 0 for s in all_scores),
                  top1_correct=sum(s['top1_correct'] for s in all_scores), behavior_pass=behavior_pass,
                  artifact_and_arithmetic_pass=True, shards=counts, maps_sha256=sha(native),
                  elapsed_seconds=time.monotonic() - start,
                  evidence='technical calibration and descriptive full-coordinate contrasts only',
                  causal_tested=False, mechanism_accepted=False,
                  contrast_definition='mean s0*X, mean s1*X, mean s0*s1*X over each family balanced 2x2; s=2*r-1; all native indices retained',
                  limitations=['No new independent model forward', 'No confirm tuning or coordinate selection', 'No per-coordinate significance inference', 'Original calibration tolerance is absolute and not a relative-error guarantee'])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cells', 'shards', 'provenance')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['discovery', 'confirm'])
    main(parser.parse_args().split)
