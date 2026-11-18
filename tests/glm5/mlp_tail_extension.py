"""Extend the Qwen3-4B long-text native MLP atlas by four earlier layers."""
from collections import Counter
from pathlib import Path
import shutil
import time

import numpy as np

import mlp_tail_goodfire as C
import mlp_tail_profiles as profiles
from mlp_tail_finalize import words

BASE = C.OUT
OUT = BASE.with_name('mlp_tail_b28_b31_v1')
BLOCKS = list(range(28, 32))
RUN = 'MLP-TAIL-B28-B35-ATLAS-v2'
SPLITS = ('discovery', 'confirm', 'holdout')


def freeze():
    from shared.dist_protocol.natural_register_runner import snapshot

    if (OUT / 'plan.json').exists():
        raise RuntimeError('Frozen atlas plan already exists. Preserve the existing run and its evidence.')
    bank = C.read(C.BANK / 'bank/manifest.json')
    base = C.read(BASE / 'manifest.json')
    assert bank['state'] == base['state'] == 'complete'
    assert bank['model_snapshot'] == base['model_snapshot'] == C.SNAPSHOT
    assert base['positions'] == bank['positions'] and len(bank['positions']) == 16
    assert base['blocks'] == [32, 33, 34, 35] and base['width'] == C.WIDTH
    started = time.monotonic()
    C.log('verifying_frozen_inputs', started=started)
    assert snapshot(C.MODEL) == C.SNAPSHOT
    token_sha = C.file_sha(C.TOKENS)
    assert token_sha == C.read(BASE / 'plan.json')['source_tokens_sha256']
    records = {item['path']: item for item in bank['files']}
    native = [records[f'B{block:02d}-{field}.npy'] for block in BLOCKS for field in ('input', 'output')]
    original = {item['path']: item for item in base['files']}
    shared = ['B32-A.npy', 'B32-W-gate.npy', 'B32-W-up.npy', 'unembedding.npy', 'documents.json', 'tokenizer.json']
    for directory, items in ((C.BANK / 'bank', native), (BASE, [original[n] for n in shared])):
        for item in items:
            assert C.file_sha(directory / item['path']) == item['sha256'], item['path']
    assert shutil.disk_usage(OUT).free > 21 * 2**30
    plan = {
        'schema': 'mlp_tail_extension_plan.v1', 'kind': 'real', 'run_id': RUN,
        'model_snapshot': C.SNAPSHOT, 'precision': 'BF16', 'blocks': BLOCKS,
        'scope': '观察性普查。模型原始 MLP 边界复算；无新语义确认、干预或递推验收。',
        'exposure': '同一已使用过的中英文长文档库。分片复核不构成新的盲测。',
        'positions': bank['positions'], 'context_tokens': 1024, 'documents': 2048,
        'split': {'discovery': [0, 1024], 'confirm': [1024, 1536], 'holdout': [1536, 2048]},
        'source_tokens_sha256': token_sha, 'source_bank_manifest_sha256': C.file_sha(C.BANK / 'bank/manifest.json'),
        'base_manifest_sha256': C.file_sha(BASE / 'manifest.json'), 'source_files': native,
        'thresholds': {'replay_relative_output_error_max': .02, 'pole_positions_per_split': 8},
        'methods': ['原生 gate/up/乘积统计与分片极值', '128-token前缀关键词候选',
                    'down 列与全词表余弦读出', '相邻激活 Pearson 与静态权重余弦'],
        'source_code_sha256': C.file_sha(__file__),
        'implementation_sha256': {path.name: C.file_sha(path) for path in
                                  (Path(C.__file__), Path(profiles.__file__),
                                   Path(__file__).with_name('mlp_tail_extension_finish.py'))},
        'power': '描述性全量单元统计。未定义待检验效应或显著性阈值。',
        'resources': {'disk_GiB_max': 16, 'retained_disk_GiB_min': 5, 'gpu': 'CUDA'},
    }
    C.save_json(OUT / 'plan.json', plan)
    C.log('freeze_complete', 1, 1, started)
    return plan


def contexts():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(BASE / 'tokenizer.json'))
    positions = C.read(OUT / 'plan.json')['positions']
    tokens = np.load(C.TOKENS, mmap_mode='r')
    terms = [words(tokenizer.decode(tokens[doc, pos-128:pos+1].tolist()))
             for doc in range(2048) for pos in positions]
    frequencies = Counter(term for window in terms[:16384] for term in window)
    started = time.monotonic()
    for index, block in enumerate(BLOCKS, 1):
        profile = C.read(OUT / f'B{block}-profile.json')
        maps = {f: np.load(OUT / f'B{block}-{f}.npy', mmap_mode='r') for f in 'GVA'}
        shape = (3, 3, 2, C.WIDTH, 8, 3)
        poles = np.lib.format.open_memmap(OUT / f'B{block}-pole-values.npy', mode='w+', dtype=np.uint16, shape=shape)
        units = np.arange(C.WIDTH)[:, None]
        for fi, field in enumerate('GVA'):
            for si, split in enumerate(SPLITS):
                for pi, pole in enumerate(('positive', 'negative')):
                    ids = np.array(profile['fields'][field][split][f'{pole}_indices'])
                    for ci, component in enumerate('GVA'):
                        poles[fi, si, pi, :, :, ci] = maps[component][ids, units]
        poles.flush()
        del poles, maps
        g = profile['fields']['G']
        rows = []
        for unit, (high, low) in enumerate(zip(g['discovery']['positive_indices'], g['discovery']['negative_indices'])):
            def tally(indices):
                samples = {idx // 16: idx for idx in reversed(indices)}
                return Counter(term for idx in samples.values() for term in terms[idx]), len(samples)

            pos, npos = tally(high)
            neg, nneg = tally(low)
            def score(term):
                return (pos[term] - .5 * neg[term]) * np.log1p(16384 / frequencies[term])

            best = sorted((w for w in pos if frequencies[w] >= 3 and score(w) > 0), key=lambda w: (-score(w), w))[:6]
            negative = sorted((w for w in neg if w not in best and frequencies[w] >= 3),
                              key=lambda w: (-neg[w], w))[:6]
            repeated = {}
            for split in SPLITS[1:]:
                ids = g[split]['positive_indices'][unit]
                repeated[split] = {'cue_hit_count': sum(bool(terms[i].intersection(best)) for i in ids),
                                   'n_pole_positions': len(ids), 'distinct_documents': len({i//16 for i in ids})}
            rows.append({'unit': unit, 'positive_keywords': best, 'negative_keywords': negative,
                         'discovery_positive_documents': npos, 'discovery_negative_documents': nneg,
                         'repeat': repeated, 'status': 'candidate_context_keywords_not_explanation',
                         'description': '候选语境线索：' + ' / '.join(best[:3]) if best else '无合格关键词，保留原始响应。'})
        C.save_json(OUT / f'B{block}-cues.json', {'schema': 'mlp_tail_cues.v1', 'kind': 'real',
                    'run_id': RUN, 'block': block, 'units': rows, 'scope': '极值前缀中的候选线索。'})
        C.log('context_witnesses', index, len(BLOCKS), started, block=block)


def readouts():
    import torch
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(BASE / 'tokenizer.json'))
    decoded = {}
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    with C.GpuLease(), torch.inference_mode():
        embedding = torch.from_numpy(np.load(BASE / 'unembedding.npy')).view(torch.bfloat16).cuda().float()
        embedding = torch.nn.functional.normalize(embedding)
        for index, block in enumerate(BLOCKS, 1):
            weights = torch.from_numpy(np.load(OUT / f'B{block}-W-down.npy')).view(torch.bfloat16).cuda().float()
            weights = torch.nn.functional.normalize(weights, dim=0)
            rows = []
            for start in range(0, C.WIDTH, 256):
                scores = (embedding @ weights[:, start:start+256]).T
                assert torch.isfinite(scores).all()
                poles = []
                for largest in (True, False):
                    values, indices = scores.topk(8, largest=largest)
                    groups = []
                    for ids, vals in zip(indices.tolist(), values.tolist()):
                        group = []
                        for token_id, cosine in zip(ids, vals):
                            if token_id not in decoded:
                                decoded[token_id] = tokenizer.decode([token_id], skip_special_tokens=False)
                            group.append({'token_id': token_id, 'text': decoded[token_id], 'cosine': cosine})
                        groups.append(group)
                    poles.append(groups)
                # Emit in native order. The chunk is only a memory bound for the vocab scan.
                rows.extend({'unit': start+i, 'positive': positive, 'negative': negative}
                            for i, (positive, negative) in enumerate(zip(*poles)))
                del scores
            C.save_json(OUT / f'B{block}-readout.json', {'schema': 'mlp_tail_parameter_readout.v1',
                        'block': block, 'units': rows, 'scope': '静态词表余弦，未模拟最终归一化或实际输出。'})
            C.log('vocab_readout', index, len(BLOCKS), started, block=block)


def finalize(plan):
    replay = C.read(OUT / 'replay.json')
    assert all(row['exact_output_fraction'] == 1 for row in replay['checks']), replay
    assert [row['block'] for row in replay['checks']] == BLOCKS
    report = {'schema': 'mlp_tail_extension_report.v1', 'kind': 'real', 'run_id': RUN,
              'model_snapshot': C.SNAPSHOT, 'blocks': BLOCKS, 'replay': replay,
              'n_new_units': len(BLOCKS) * C.WIDTH, 'new_matrices': 20,
              'scope': plan['scope'], 'exposure': plan['exposure']}
    C.save_json(OUT / 'report.json', report)
    started = time.monotonic()
    files = []
    paths = sorted(path for path in OUT.iterdir() if path.is_file() and path.suffix in ('.npy', '.json'))
    for index, path in enumerate(paths, 1):
        if path.name.endswith('-profile.json'):
            data = C.read(path)
            assert len(data['sign_regime_fractions']) == C.WIDTH
            assert np.allclose(np.sum(data['sign_regime_fractions'], axis=1), 1)
            assert all(len(data['fields'][field][stat]) == C.WIDTH for field in 'GVA' for stat in ('mean', 'std', 'min', 'max'))
        if '-B' in path.name and path.suffix == '.npy':
            matrix = np.load(path, mmap_mode='r')
            assert matrix.shape == (C.WIDTH, C.WIDTH) and matrix.dtype == np.float32
            assert all(np.isfinite(matrix[i:i+256]).all() for i in range(0, C.WIDTH, 256))
            del matrix
        files.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': C.file_sha(path)})
        if index % 5 == 0:
            C.log('manifest_hashes', index, len(paths), started)
    assert sum(row['bytes'] for row in files) < 16 * 2**30
    C.save_json(OUT / 'manifest.json', {'schema': 'mlp_tail_extension.v1', 'kind': 'real', 'state': 'complete',
                'run_id': RUN, 'model_snapshot': C.SNAPSHOT, 'blocks': BLOCKS, 'width': C.WIDTH,
                'positions': plan['positions'], 'base_manifest_sha256': plan['base_manifest_sha256'], 'files': files})
    C.log('atlas_complete', len(BLOCKS)*C.WIDTH, len(BLOCKS)*C.WIDTH, started,
          bytes=sum(row['bytes'] for row in files))


def main():
    from mlp_tail_extension_finish import associations

    OUT.mkdir(exist_ok=True)
    C.OUT, C.RUN, C.BLOCKS = OUT, RUN, BLOCKS
    plan = freeze()
    C.derive()
    assert all(r['exact_output_fraction'] == 1 for r in C.read(OUT / 'replay.json')['checks'])
    profiles.main()
    for name in ('documents.json', 'token_ids.npy', 'token_classes.npy'):
        assert C.file_sha(OUT / name) == C.file_sha(BASE / name), name
        (OUT / name).unlink()
    contexts()
    readouts()
    associations()
    finalize(plan)


if __name__ == '__main__':
    main()
