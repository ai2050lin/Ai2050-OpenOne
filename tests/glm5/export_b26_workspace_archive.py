"""Export the verified Phase 3075 B26 replay without loading a model."""

import base64
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deploy.storage_policy import check_write, staging_file
from shared.dist_protocol.test_archive import digest, file_sha

RESULT = ROOT / 'tests/glm5/result/live_original_14b_v4'
RELEASE = ROOT / 'frontend/src/components/v8/data/nativeObservationB26.json'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    receipt = read_json(RESULT / 'live-original-receipt.json')['receipt']
    native = RESULT / 'live-original-package/native-0000.npz'
    source_sha = file_sha(native)
    receipt_identity = {key: value for key, value in receipt.items()
                        if key not in ('receipt_sha256', 'readback_verified')}
    if (receipt['archive_state'] != 'server_verified' or not receipt['readback_verified']
            or receipt['missing_core_roles'] or digest(receipt_identity) != receipt['receipt_sha256']
            or not any(entry['role'] == 'native_field' and entry['name'] == native.name
                       and entry['sha256'] == source_sha and entry['size'] == native.stat().st_size
                       for entry in receipt['artifacts'])):
        raise ValueError('B26 tensor is missing from the verified center receipt')

    result = read_json(RESULT / 'live-original-package/conclusion.json')
    case = result['case']
    with np.load(native, allow_pickle=False) as arrays:
        manifest = json.loads(arrays['manifest'].item())
        head_z, activation = arrays['head_z'], arrays['mlp_activation']
        positions, token_ids = arrays['readout_positions'].tolist(), arrays['token_ids'].tolist()
        if (not result['passed'] or case['run_id'] != receipt['run_id'] or case['trace_id'] != manifest['trace_id']
                or case['model_snapshot'] != manifest['model_snapshot'] or case['readout_positions'] != positions
                or manifest['kind'] != 'real' or manifest['precision'] != 'bf16'
                or manifest['token_ids'] != token_ids or manifest['readout_positions'] != positions
                or head_z.shape != (len(positions), 40, 128) or activation.shape != (len(positions), 17408)
                or head_z.dtype != np.uint16 or activation.dtype != np.uint16):
            raise ValueError('Case, snapshot, boundaries or BF16 field dimensions differ')

        def encoded(row):
            return base64.b64encode(row.astype('<u2', copy=False).tobytes()).decode('ascii')

        data = dict(schema='native_observation_archive.v1', kind='real', snapshot=manifest['model_snapshot'],
                    run=receipt['run_id'], caseId=case['case_id'], trace=manifest['trace_id'],
                    receiptSha256=receipt['receipt_sha256'], arraySha256=source_sha,
                    text=manifest['prompt'], defaultRow=str(positions[-1]),
                    rows=[dict(key=str(position), position=position, tokenId=token_ids[position],
                               blocks={'26': {'sha256': source_sha, 'fields': {
                                   'head_z': encoded(head_z[i]), 'mlp_activation': encoded(activation[i])}}})
                          for i, position in enumerate(positions)])
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n'
    target = check_write(RELEASE, len(payload), operation='导出 B26 客户端原场')
    with staging_file(target) as temporary:
        temporary.write_bytes(payload)
    print(f'{target.name}: {len(payload)} bytes, sha256={file_sha(target)}')


if __name__ == '__main__':
    main()
