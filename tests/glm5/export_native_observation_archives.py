"""Check and publish compact BF16 excerpts from existing verified research archives."""

import argparse
import json
import sys

import numpy as np

from build_free_model_data import CHECK, REPO, SNAPSHOTS, TRACE, encoded, sha

sys.path.insert(0, str(REPO))
from deploy.storage_policy import check_write, staging_file

B23 = REPO / 'research/glm5/docs/previews/client-review/b23-corpus-data.json'
B23_CHECK = REPO / 'tests/glm5/result/model_analysis_b23_review/source-check.json'
DESTINATION = REPO / 'frontend/src/components/v8/data'


def qwen4_archive():
    manifest_bytes = (TRACE / 'manifest.json').read_bytes()
    receipt = json.loads(CHECK.read_bytes())
    manifest = json.loads(manifest_bytes)
    assert sha(manifest_bytes) == receipt['manifest_file_sha256']
    assert manifest['kind'] == 'real' and manifest['model_snapshot'] == SNAPSHOTS['Qwen3-4B']
    position = receipt['absolute_position']
    site = manifest['readout_positions'].index(position)
    assert manifest['token_ids'][position] == receipt['token_id']
    hashes = {entry['path']: entry['sha256'] for entry in manifest['files']}
    heads, kv, width, ffn = (manifest[k] for k in ('n_heads', 'n_kv_heads', 'head_dim', 'd_ff'))
    blocks = []
    for block in range(manifest['n_layers']):
        file = TRACE / f'block_{block:02d}.npz'
        assert sha(file.read_bytes()) == hashes[file.name]
        with np.load(file, allow_pickle=False) as archive:
            fields = {}
            for boundary, key, shape in (
                ('head_z', 'head_z', (heads, width)), ('head_q', 'q', (heads, width)),
                ('head_k', 'k', (kv, width)), ('head_v', 'v', (kv, width)),
                ('mlp_activation', 'activation', (ffn,)), ('mlp_gate', 'gate', (ffn,)), ('mlp_value', 'value', (ffn,)),
            ):
                array = archive[key]
                row = array[:, position, :] if key in ('q', 'k', 'v') else array[site]
                assert row.dtype == np.dtype('<u2') and row.shape == shape
                fields[boundary] = encoded(row.tobytes())
        blocks.append({'sha256': hashes[file.name], 'fields': fields})
    return {
        'schema': 'native_observation_archive.v1', 'kind': 'real', 'snapshot': manifest['model_snapshot'],
        'run': manifest['run_id'], 'trace': manifest['trace_id'], 'caseId': manifest['sample_id'],
        'manifestSha256': sha(manifest_bytes), 'text': manifest['text'],
        'position': position, 'tokenId': receipt['token_id'], 'token': manifest['tokens'][position],
        'blocks': blocks,
    }


def b23_archive():
    source = B23.read_bytes()
    receipt = json.loads(B23_CHECK.read_bytes())
    assert sha(source) == receipt['data_sha256']
    data = json.loads(source)
    assert data['snapshot'] == receipt['snapshot'] and data['run'] == receipt['run']
    assert data['hashes'] == receipt['archive_sha256'] and data['source'] == receipt['manifest']
    return {
        'schema': 'native_observation_archive.v1', 'kind': 'real',
        **{key: data[key] for key in ('snapshot', 'run', 'block', 'width')},
        'manifestSha256': data['source']['sha256'], 'arraySha256': data['hashes']['B23-A.npy'],
        'rows': [{key: row[key] for key in ('key', 'doc', 'text', 'position', 'tokenId', 'bits')} for row in data['rows']],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    for name, payload in [('nativeObservationQwen4.json', qwen4_archive()), ('nativeObservationB23.json', b23_archive())]:
        path = DESTINATION / name
        content = (json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
        if args.write:
            target = check_write(path, len(content), operation='写入观察归档摘录')
            with staging_file(target) as temporary:
                temporary.write_bytes(content)
        assert path.read_bytes() == content, f'{path.name}: content differs'
        print(f'{name}: {len(content)} bytes SHA256 {sha(content)}', flush=True)


if __name__ == '__main__':
    main()
