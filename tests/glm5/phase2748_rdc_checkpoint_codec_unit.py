"""Exact full-parameter codec trial on retained real Phase2747 checkpoints."""
import zlib
from rdc_question_common import *
from rdc_formation_microbatch import TensorSliceReader
from rdc_question_checkpoint_codec import encode_FP32, decode_FP32, encode_BF16, decode_BF16


def deflate_check(array):
    stream = zlib.compressobj(6, wbits=-15)
    inverse = zlib.decompressobj(wbits=-15)
    raw = memoryview(np.ascontiguousarray(array)).cast('B')
    before, after = hashlib.sha256(), hashlib.sha256()
    count = 0
    for begin in range(0, len(raw), 1024**2):
        chunk = raw[begin:begin+1024**2]
        before.update(chunk)
        compressed_bytes = stream.compress(chunk)
        count += len(compressed_bytes)
        after.update(inverse.decompress(compressed_bytes))
    rest = stream.flush()
    count += len(rest)
    after.update(inverse.decompress(rest))
    after.update(inverse.flush())
    assert inverse.eof and before.hexdigest() == after.hexdigest()
    return {'raw_bytes': len(raw), 'deflated_bytes': count, 'raw_word_sha256': before.hexdigest(), 'deflate_roundtrip_equal': True}


def main():
    import torch
    start = time.monotonic()
    path = OUT/'unit/checkpoint_codec_current.json'
    source = snapshot(__file__)
    codec = snapshot(Path(__file__).with_name('rdc_question_checkpoint_codec.py'))
    if path.exists() and read(path)['codec'] == codec and read(path)['all_passed']:
        return read(path)
    folder = ROOT/'models/hf/qwen3-4b'
    index = read(folder/'model.safetensors.index.json')['weight_map']
    reader = TensorSliceReader()
    runs = ['true_token_2747', 'within_surface_class_permuted_token_2747', 'surface_class_mass_2747']
    results = []
    for run in runs:
        reference_path = BASE/'phase2747/training'/run/'commits/delta_128.json'
        reference = read(reference_path)
        assert sha(ROOT/reference['field_path']) == reference['field_sha256']
        records = []
        with np.load(ROOT/reference['field_path']) as z:
            for name in ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']:
                full = 'model.layers.16.mlp.'+name
                original = reader.tensor(folder/index[full], full).view(torch.uint16).numpy().copy()
                actual = (unbits(original)+z[name])+z['reconstruction_residual__'+name]
                assert actual.dtype == np.float32 and np.isfinite(actual).all()
                encoded = encode_FP32(actual, original)
                assert np.array_equal(decode_FP32(encoded, original).view(np.uint32), actual.view(np.uint32))
                fp32 = deflate_check(encoded)
                native = torch.from_numpy(actual).to(torch.bfloat16).view(torch.uint16).numpy().copy()
                native_code = encode_BF16(native, original)
                assert np.array_equal(decode_BF16(native_code, original), native)
                bf16 = deflate_check(native_code)
                records.append({'parameter': full, 'shape': list(actual.shape), 'scalars': actual.size,
                    'FP32': fp32, 'BF16': bf16, 'all_parameter_words_reconstructed_exactly': True})
                del actual, encoded, native, native_code, original
        total = sum(r['FP32']['deflated_bytes']+r['BF16']['deflated_bytes'] for r in records)
        assert sum(r['scalars'] for r in records) == 74711040
        results.append({'historical_run': run, 'historical_receipt_sha256': sha(reference_path),
            'historical_data_sha256': reference['field_sha256'], 'parameters': records,
            'FP32_and_BF16_deflated_bytes': total})
        print('NATURAL_CHECKPOINT_CODEC', run, total, flush=True)
    value = {'timestamp': stamp(), 'source': source, 'codec': codec, 'all_passed': True,
        'historical_full_parameter_trials': results, 'actual_new2748_training_executed': False,
        'suggested_six_run_storage_projection_bytes': int(6*max(r['FP32_and_BF16_deflated_bytes'] for r in results)*1.25),
        'projection_scope': 'Largest actually tested historical128stepcheckpoint among3conditions, six future runs,25percent margin. Exact new96stepcompression is unknown and must be checked during actual writes; not a worst-case guarantee.',
        'retention': 'Codec trial reconstructed and deflated complete vectors in memory without creating duplicate historical weight files; original checkpoints and all reference fields unchanged.',
        'seconds': time.monotonic()-start}
    save(OUT/'unit'/('checkpoint_codec_'+str(time.time_ns())+'.json'), value)
    save(path, value)
    print('NATURAL_CHECKPOINT_CODEC_PASSED', value['suggested_six_run_storage_projection_bytes'], flush=True)
    return value


if __name__ == '__main__':
    main()
