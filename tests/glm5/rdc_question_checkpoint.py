"""Persist every actual learned word; inverse-verified against native originals."""
from rdc_question_common import *
from rdc_question_checkpoint_codec import encode_FP32, decode_FP32, encode_BF16, decode_BF16
from rdc_formation_microbatch import TensorSliceReader


def original_manifest():
    path = OUT/'training/original_checkpoint_manifest.json'
    folder = ROOT/'models/hf/qwen3-4b'
    indexpath = folder/'model.safetensors.index.json'
    mapping = read(indexpath)['weight_map']
    if path.exists():
        result = read(path)
        assert result['index_sha256'] == sha(indexpath)
        for name,ref in result['shards'].items():
            stat = (folder/name).stat()
            assert stat.st_size == ref['bytes'] and stat.st_mtime_ns == ref['mtime_ns']
        return result
    result = {'timestamp':stamp(),'folder':str(folder),'index_sha256':sha(indexpath),
        'shards':{name:{'sha256':sha(folder/name),'bytes':(folder/name).stat().st_size,
            'mtime_ns':(folder/name).stat().st_mtime_ns} for name in sorted(set(mapping.values()))},
        'trainable_names':['model.layers.16.mlp.'+name for name in ['gate_proj.weight','up_proj.weight','down_proj.weight']]}
    immutable(path,result)
    return result


def save_final(target, run, protocol_sha):
    import torch
    manifest = original_manifest()
    folder = Path(manifest['folder'])
    mapping = read(folder/'model.safetensors.index.json')['weight_map']
    reader = TensorSliceReader()
    refs, total = [], 0
    for name,parameter in target.named_parameters():
        full = 'model.layers.16.mlp.'+name
        assert full in manifest['trainable_names'] and parameter.dtype == torch.float32
        original = reader.tensor(folder/mapping[full],full).view(torch.uint16).numpy().copy()
        fp32 = parameter.detach().cpu().numpy().copy()
        # Use the actual CUDA cast that will supply the deployed native layer.
        deployed = bits(parameter.detach().to(torch.bfloat16))
        encoded = encode_FP32(fp32,original)
        bfencoded = encode_BF16(deployed,original)
        assert np.array_equal(decode_FP32(encoded,original).view(np.uint32),fp32.view(np.uint32))
        assert np.array_equal(decode_BF16(bfencoded,original),deployed)
        reference = commit_arrays(Path('training')/run/'checkpoint96',name.replace('.','_'),
            {'FP32_XOR_words':encoded,'BF16_XOR_words':bfencoded})
        # Verify the actual persisted archive, not only in-memory codec input.
        with np.load(ROOT/reference['path']) as z:
            assert np.array_equal(decode_FP32(z['FP32_XOR_words'],original).view(np.uint32),fp32.view(np.uint32))
            assert np.array_equal(decode_BF16(z['BF16_XOR_words'],original),deployed)
        refs.append({'parameter':full,'shape':list(fp32.shape),'scalars':fp32.size,'field':reference,
            'original_shard':mapping[full],'original_shard_sha256':manifest['shards'][mapping[full]]['sha256'],
            'FP32_actual_word_sha256':hashlib.sha256(memoryview(fp32).cast('B')).hexdigest(),
            'BF16_actual_CUDA_cast_word_sha256':hashlib.sha256(memoryview(deployed).cast('B')).hexdigest(),
            'persisted_inverse_FP32_and_BF16_all_words_equal':True})
        total += fp32.size
        del original,fp32,deployed,encoded,bfencoded
    assert total == 74711040
    result = {'run':run,'step':96,'all_passed':True,'parameters':refs,'total_parameters':total,
        'training_protocol_sha256':protocol_sha,'original_manifest_sha256':sha(OUT/'training/original_checkpoint_manifest.json'),
        'codec':snapshot(Path(__file__).with_name('rdc_question_checkpoint_codec.py')),
        'checkpoint_writer':snapshot(__file__),'bytes':sum(r['field']['bytes'] for r in refs),
        'precision':'Every actual FP32training word and actual CUDA-castBF16deployment word, lossless relative to immutable original BF16shard.'}
    immutable(OUT/'training'/run/'checkpoint96.json',result)
    return result


def deploy_native(model, run):
    """Use on a fresh native BF16 model, not on an active FP32training bridge."""
    import torch
    receipt = read(OUT/'training'/run/'checkpoint96.json')
    assert receipt['all_passed'] and receipt['original_manifest_sha256'] == sha(OUT/'training/original_checkpoint_manifest.json')
    manifest = original_manifest()
    folder = Path(manifest['folder'])
    reader = TensorSliceReader()
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for record in receipt['parameters']:
            parameter = parameters[record['parameter']]
            assert parameter.dtype == torch.bfloat16
            original = reader.tensor(folder/record['original_shard'],record['parameter']).view(torch.uint16).numpy().copy()
            reference = record['field']
            assert sha(ROOT/reference['path']) == reference['sha256']
            with np.load(ROOT/reference['path']) as z:
                value = decode_BF16(z['BF16_XOR_words'],original)
            assert hashlib.sha256(memoryview(value).cast('B')).hexdigest() == record['BF16_actual_CUDA_cast_word_sha256']
            parameter.copy_(torch.from_numpy(value.copy()).view(torch.bfloat16).to(parameter.device))
            assert np.array_equal(bits(parameter),value)
    return receipt
