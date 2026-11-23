"""Verify or export three archived B26/H36 output vectors for offline inspection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write

RUN = 'clt_b26_three_cues_v3'
SRC = ROOT / 'tests/glm5/result' / RUN
TARGET = ROOT / 'frontend/src/components/v8/data/nativeHeadChannels.js'


def read(name):
    return json.loads((SRC / name).read_bytes())


def build():
    receipt = read('delivery.json')
    for name in ['confirm-00.npz', 'confirm-00.json', 'confirm-materials.json', 'confirm-collection.json']:
        if hashlib.sha256((SRC / name).read_bytes()).hexdigest() != receipt['files'][name]:
            raise ValueError(f'{name}: source hash mismatch')
    digest = receipt['files']['confirm-00.npz']
    meta, material = read('confirm-00.json'), read('confirm-materials.json')
    if digest != meta['sha'] or digest != read('confirm-collection.json')['shards']['confirm-00.npz']:
        raise ValueError('collection, shard and receipt disagree')
    snapshot = receipt['provenance']['model_revision']
    evidence = json.loads((ROOT / 'frontend/src/components/v8/threeCueEvidence.json').read_bytes())['runs'][0]
    if (evidence['id'], evidence['snapshot'], evidence['readout']['layer'], evidence['readout']['head']) != (RUN, snapshot, 26, 36):
        raise ValueError('client source and original archive disagree')
    with np.load(SRC / 'confirm-00.npz', allow_pickle=False) as z:
        values = z['head_actual']
        if values.dtype != np.float32 or values.shape != (len(meta['scores']), 128):
            raise ValueError('unexpected native output layout')
        cases = {}
        for i in range(3):
            record = material[i]
            identity = f'confirm-000-{i:02d}'
            if record['id'] != identity or meta['scores'][i]['id'] != identity or not np.isfinite(values[i]).all():
                raise ValueError('case identity or native values mismatch')
            cases[identity] = {'position': len(record['tokens']) - 1, 'tokenId': record['tokens'][-1],
                               'values': values[i].tolist()}
    data = {'run': RUN, 'snapshot': snapshot, 'block': 26, 'head': 36, 'headWidth': 128,
            'boundary': 'head_z', 'source': (SRC / 'confirm-00.npz').relative_to(ROOT).as_posix(),
            'field': 'head_actual', 'sha256': digest, 'cases': cases}
    return '// Verified native pre-o_proj slices. Rebuild with tests/glm5/export_native_head_channels.py.\n' + \
           'globalThis.AI2050HeadArchive = ' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    text = build()
    if args.write:
        check_write(TARGET, len(text.encode('utf-8')) * 2, operation='Native Head archive excerpt')
        TARGET.write_text(text, encoding='utf-8', newline='\n')
    elif TARGET.read_text(encoding='utf-8') != text:
        raise ValueError('offline Head vectors differ from their original sources')
    print('Verified three B26/H36 pre-o_proj vectors, 128 channels each, source hashes, case IDs and final token positions.')
