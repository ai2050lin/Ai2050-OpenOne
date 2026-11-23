"""Replay the fullscreen field API against a retained native archive, on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write
from server import mechanism_trace_service
from shared.dist_protocol.mechanism_trace import TraceStore, file_sha

RESULTS = ROOT / 'tests/glm5/result'
ARCHIVE = RESULTS / 'live_traces/fcf90469ce80cc33091526080da6c8b11b03da3d9c36a9b9b3031d31f4b1876e'
RECEIPT = RESULTS / 'workspace_full_native/source-check.json'
TRACE_ID = '7e80eaac2b68c0f4dc97de55fe9ee23e41c12e7d4601f5e870bb861031d2f0d3'
BLOCK, HEAD, POSITION = 26, 14, 201


def verify():
    manifest_path = ARCHIVE / 'manifest.json'
    scratch = check_write(ROOT / 'tests/glm5_temp', 2 * 1024**2, operation='Archived native field API source check')
    with TemporaryDirectory(dir=scratch) as index:
        store = TraceStore(index, RESULTS)
        manifest = store.register(manifest_path)
        if manifest['trace_id'] != TRACE_ID or manifest['kind'] != 'real' or not manifest['deep']:
            raise ValueError('expected the frozen deep native observation')
        group = HEAD // (manifest['n_heads'] // manifest['n_kv_heads'])
        row = manifest['readout_positions'].index(POSITION)
        filename = f'block_{BLOCK:02d}.npz'
        path = store.native_path(TRACE_ID, filename)
        block_sha = next(f['sha256'] for f in manifest['files'] if f['path'] == filename)
        checks = {}
        with np.load(path, allow_pickle=False) as archive, patch.object(mechanism_trace_service, 'store', return_value=store):
            for boundary, key, index in [
                ('head_q', 'q', (HEAD, POSITION)), ('head_k', 'k', (group, POSITION)),
                ('head_v', 'v', (group, POSITION)), ('head_z', 'head_z', (row, HEAD)),
                ('mlp_activation', 'activation', (row,)),
            ]:
                is_head = boundary.startswith('head_')
                response = mechanism_trace_service.field(TRACE_ID, block=BLOCK, position=POSITION,
                                                         boundary=boundary, head=HEAD if is_head else None)
                if (response['trace_id'], response['model_snapshot'], response['block'], response['position'],
                    response['boundary'], response['source']) != (
                        TRACE_ID, manifest['model_snapshot'], BLOCK, POSITION, boundary, 'native_BF16_bits'):
                    raise ValueError(f'{boundary} metadata mismatch')
                if is_head and (response['head'], response['kv_group']) != (HEAD, group):
                    raise ValueError(f'{boundary} head or KV group mismatch')
                expected = archive[key][index]
                width = manifest['head_dim'] if is_head else manifest['d_ff']
                if expected.dtype != np.uint16 or expected.shape != (width,) or response['shape'] != [width]:
                    raise ValueError(f'{boundary} source shape or dtype mismatch')
                returned = np.asarray(response['values'], dtype='<f4')
                expected_bits = expected.astype('<u4') << 16
                matches = bool(np.array_equal(returned.view('<u4'), expected_bits))
                if not matches:
                    raise ValueError(f'{boundary} API values differ from the native archive')
                checks[boundary] = {
                    'source_array': key, 'source_indices': list(index), 'width': width,
                    'source_bf16_le_sha256': hashlib.sha256(expected.astype('<u2').tobytes()).hexdigest(),
                    'returned_f32_le_sha256': hashlib.sha256(returned.tobytes()).hexdigest(),
                    'all_values_bitwise_equal_after_bf16_expansion': matches,
                }
        return {
            'schema': 'full_native_field_source_check.v1', 'purpose': 'engineering source fidelity',
            'method': 'in-process FastAPI field handler, full vectors checked against indexed native BF16 archive',
            'model': 'Qwen3-4B', 'model_snapshot': manifest['model_snapshot'], 'run_id': manifest['run_id'],
            'trace_id': TRACE_ID, 'manifest_file': manifest_path.relative_to(ROOT).as_posix(),
            'manifest_file_sha256': file_sha(manifest_path), 'manifest_fingerprint_verified': True,
            'block_file': path.relative_to(RESULTS.resolve()).as_posix(), 'block_file_relative_to': 'tests/glm5/result',
            'block_file_sha256': block_sha, 'block_hash_verified': True, 'block': BLOCK, 'head': HEAD,
            'query_heads': manifest['n_heads'], 'kv_heads': manifest['n_kv_heads'], 'shared_kv_group': group,
            'absolute_position': POSITION, 'token_id': manifest['token_ids'][POSITION],
            'readout_positions': manifest['readout_positions'], 'readout_index': row,
            'vectors': checks, 'all_vectors_equal': True,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='save the verified source-check receipt')
    args = parser.parse_args()
    report = verify()
    if args.write:
        check_write(RECEIPT, 64 * 1024, operation='Fullscreen native field source receipt')
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    elif json.loads(RECEIPT.read_bytes()) != report:
        raise ValueError('saved receipt differs from this verification')
    print('Verified archived B26/H14, KV3, token 201: four 128-channel vectors and all 9728 FFN units.')
