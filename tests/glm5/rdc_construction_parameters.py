"""All-native-parameter catalog and exact, unpruned factor lookup (CPU only)."""
import json, struct
from functools import lru_cache
from rdc_construction_common import ROOT, BASE, MODELS, stamp, sha, immutable, snapshot
from rdc_relation_native_parameters import parameter, decode


@lru_cache(maxsize=3)
def catalog(key):
    assert key in MODELS
    folder = ROOT / 'models/hf' / MODELS[key]
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    mapping = json.loads((folder / 'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    headers = {}
    for name in sorted(set(mapping.values())):
        with (folder / name).open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            headers[name] = (length, json.loads(stream.read(length)))
    records = []
    for name, shard in sorted(mapping.items()):
        length, header = headers[shard]
        item = header[name]
        count = 1
        for size in item['shape']:
            count *= size
        records.append({'name': name, 'shape': item['shape'], 'dtype': item['dtype'],
            'scalar_count': count, 'shard': shard,
            'file_data_offset': 8+length+item['data_offsets'][0],
            'data_bytes': item['data_offsets'][1]-item['data_offsets'][0],
            'layer': int(name.split('.')[2]) if name.startswith('model.layers.') else None})
    return {'model': key, 'model_directory': str(folder), 'config': config,
        'config_sha256': sha(folder/'config.json'), 'weight_index_sha256': sha(folder/'model.safetensors.index.json'),
        'parameters': records, 'registered_scalar_count': sum(r['scalar_count'] for r in records),
        'coverage': 'Every indexed original tensor and every scalar address, including embeddings, all decoder blocks, normalization and readout. No copied or trained checkpoint is created.'}


def mlp_factors(key, block, unit):
    meta = catalog(key)
    assert 0 <= block < meta['config']['num_hidden_layers']
    assert 0 <= unit < meta['config']['intermediate_size']
    prefix = f'model.layers.{block}.mlp.'
    names = {r['name'] for r in meta['parameters']}
    if prefix+'gate_up_proj.weight' in names:
        # Native GLM stores [gate; up] as two contiguous halves of one matrix.
        # These are two factor addresses, not two separately stored tensors.
        joined = parameter(ROOT, prefix+'gate_up_proj.weight', MODELS[key])
        middle = meta['config']['intermediate_size']
        assert joined.shape == (2*middle,meta['config']['hidden_size'])
        gate = decode(joined[unit]).astype(float)
        up = decode(joined[middle+unit]).astype(float)
    else:
        gate = decode(parameter(ROOT, prefix+'gate_proj.weight', MODELS[key])[unit]).astype(float)
        up = decode(parameter(ROOT, prefix+'up_proj.weight', MODELS[key])[unit]).astype(float)
    down = decode(parameter(ROOT, prefix+'down_proj.weight', MODELS[key])[:, unit]).astype(float)
    return gate, up, down


def register():
    for key in MODELS:
        path = BASE / 'parameters' / (key+'.json')
        data = catalog(key)
        if not path.exists():
            immutable(path, {'timestamp': stamp(), 'source': snapshot(__file__), **data,
                'factorization': 'Gamma[k,j,i,r]=Wd[j,k]*Wg[k,i]*Wu[k,r]; retain three complete native vectors, never materialize or truncate the enormous product tensor.',
                'attention': 'Q/K/OV matrices and all head/GQA indices registered. Native QK norm, RoPE and bias must remain in the runtime calculation; a bare weight product is not the full conditional score.',
                'scope': 'Architecture-level exact index/factorization, not discovered semantic labels or original training-history recovery.'})
        print('CONSTRUCTION_PARAMETER_CATALOG', key, len(data['parameters']), data['registered_scalar_count'], flush=True)


if __name__ == '__main__':
    register()
