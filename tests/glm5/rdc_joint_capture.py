"""All-native-coordinate fields in bounded RAM, with immutable per-array identities and replay checks."""
import gc
import hashlib
import sys
import os
from rdc_joint_common import *


def array_identity(a):
    a = np.asarray(a)
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def pilot_material():
    material = rows()
    chosen = []
    for i in range(2):
        for lang in ('en', 'zh'):
            chosen.append([r for r in material if r['split'] == 'train' and r['language'] == lang][i])
    return chosen


def retained_ids(fresh=False):
    material = rows(fresh)
    return {next(r['sample_id'] for r in material if r['split'] == split and r['language'] == lang)
            for split in (('confirmation',) if fresh else ('train', 'test')) for lang in ('en', 'zh')}


class Observer:
    def __init__(self, model):
        self.enabled = False
        self.positions = []
        self.data = {}
        self.layers = {}
        self.handles = []
        def embed(module, args, output):
            if self.enabled:
                self.layers[0] = bits(output[0, self.positions])
        self.handles.append(model.model.embed_tokens.register_forward_hook(embed))
        for index, layer in enumerate(model.model.layers, 1):
            def hook(module, args, output, index=index):
                if self.enabled:
                    value = output[0] if isinstance(output, tuple) else output
                    self.layers[index] = bits(value[0, self.positions])
                    if index in (12, 23, 36):
                        self.data['h' + str(index)] = bits(value[0])
            self.handles.append(layer.register_forward_hook(hook))

    def reset(self, positions):
        self.positions = positions
        self.data, self.layers = {}, {}

    def packet(self, final):
        assert len(self.layers) == 37
        return {**self.data, 'layers': np.stack([self.layers[i] for i in range(37)]),
                'postnorm': bits(final[0, self.positions]), 'positions': np.array(self.positions, dtype=np.int64)}

    def close(self):
        for handle in self.handles:
            handle.remove()


class FieldStore:
    def __init__(self, material, fresh):
        self.material = material
        self.fresh = fresh
        self.data = {}
        self.bytes = 0

    def add(self, row, packet):
        self.data[row['sample_id']] = packet
        self.bytes += sum(a.nbytes for a in packet.values())

    def __getitem__(self, row):
        return self.data[row['sample_id'] if isinstance(row, dict) else row]

    def clear(self):
        self.data.clear()
        self.bytes = 0
        gc.collect()


def ledger(kind, seconds, **extra):
    path = BASE / 'compute_ledger.json'
    records = read(path) if path.exists() else []
    records.append({'timestamp': stamp(), 'kind': kind, 'seconds': seconds, **extra})
    save(path, records)
    assert sum(r['seconds'] for r in records) < read(BASE / 'resource_allocation.json')['maximum_model_and_analysis_compute_seconds']


def capture(material=None, *, fresh=False, pilot=False):
    import torch
    import psutil
    from phase2662_symmetric_mapping_contract import load_native
    if material is None:
        material = pilot_material() if pilot else rows(fresh)
    if fresh:
        assert (BASE / 'frozen.json').exists(), 'Fresh responses must follow rule/selection freeze'
    elif not pilot:
        assert read(BASE / 'capture_pilot.json')['passed'], 'Measured pilot required'
    guard(4 * 1024**2)
    config = read(BASE / 'resource_allocation.json')
    store = FieldStore(material, fresh)
    start = time.monotonic()
    source = snapshot(Path(__file__))
    snapshot(ROOT / 'tests/glm5/rdc_joint_common.py')
    model, tokenizer = load_native('qwen4')
    assert model.dtype == torch.bfloat16 and not getattr(model, 'is_quantized', False)
    assert len(model.model.layers) == 36
    observer = Observer(model)
    device = model.get_input_embeddings().weight.device
    out = BASE / ('fresh' if fresh else 'main')
    keep = ({r['sample_id'] for r in material} if config.get('archive_all_full_fields', False) else retained_ids(fresh))
    pilot_ids = {r['sample_id'] for r in pilot_material()}
    material_digest = sha(material_path(fresh))
    summaries = []
    runtime = {'timestamp': stamp(), 'source': source, 'torch': torch.__version__, 'dtype': str(model.dtype), 'quantized': False,
               'model_source_sha': sha(Path(sys.modules[model.model.__class__.__module__].__file__)),
               'tokenizer_sha': sha(ROOT / 'models/hf/qwen3-4b/tokenizer.json'), 'config': model.config.to_dict(),
               'execution': 'CUDA native BF16 eager, batch1, unpadded/untruncated natural text, no chat template, use_cache=False.',
               'coverage': 'H12/H23/H36 every token; embedding and all36 block boundaries at positions0,1 and both anchors/+1; final postnorm separately.',
               'fresh': fresh, 'pilot': pilot, 'material_sha': material_digest,
               'retention': 'Complete fields in temporary RAM. Retained original fields for predeclared representatives; every other array has immutable shape/dtype/SHA and an exact native replay recipe.'}
    runtime['retention'] = ('All requested original BF16 arrays retained losslessly on D; per-array hashes checked on replay.'
                            if config.get('archive_all_full_fields', False) else runtime['retention'])
    if (out / 'runtime.json').exists():
        prior_runtime = read(out / 'runtime.json')
        immutable(out / 'runtime_history' / (sha(out / 'runtime.json')[:16]+'.json'), prior_runtime)
    save(out / 'runtime.json', runtime)
    try:
        with torch.inference_mode():
            for i, row in enumerate(material):
                tick = time.monotonic()
                assert tokenizer(row['text'], add_special_tokens=False)['input_ids'] == row['prompt_ids']
                x = torch.tensor([row['prompt_ids']], device=device)
                observer.reset(row['positions'])
                observer.enabled = True
                y = model.model(input_ids=x, use_cache=False).last_hidden_state
                observer.enabled = False
                packet = observer.packet(y)
                checks = {'finite_all_coordinates': all(np.isfinite(unbits(a)).all() for k, a in packet.items() if k != 'positions')}
                assert checks['finite_all_coordinates']
                if row['sample_id'] in pilot_ids:
                    plain = model.model(input_ids=x, use_cache=False).last_hidden_state
                    assert torch.equal(y, plain)
                    checks['hook_noop_bitwise'] = True
                    q = row['anchors'][0]
                    altered = x.clone()
                    altered[0, q+1:] = tokenizer.eos_token_id
                    observer.reset(row['positions'])
                    observer.enabled = True
                    changed = model.model(input_ids=altered, use_cache=False).last_hidden_state
                    observer.enabled = False
                    change_packet = observer.packet(changed)
                    for key in ('h12', 'h23', 'h36'):
                        assert np.array_equal(packet[key][:q+1], change_packet[key][:q+1])
                    mask = np.array(row['positions']) <= q
                    assert np.array_equal(packet['layers'][:, mask], change_packet['layers'][:, mask])
                    assert torch.equal(y[:, :q+1], changed[:, :q+1])
                    checks['rewritten_future_suffix_invariant'] = True
                    actual_embedding = bits(model.get_input_embeddings().weight[x[0, row['positions']]])
                    assert np.array_equal(packet['layers'][0], actual_embedding)
                    checks['embedding_original_table_bitwise'] = True
                    # Different execution shapes can differ in BF16: measure, do not require equality.
                    pref = model.model(input_ids=x[:, :q+1], use_cache=True)
                    step = model.model(input_ids=x[:, q+1:q+2], past_key_values=pref.past_key_values, use_cache=True).last_hidden_state
                    ref = y[:, q+1:q+2].float()
                    relative = float(((step.float()-ref)**2).mean()/ref.square().mean())
                    checks['cached_step_relative_MSE'] = relative
                    checks['cache_shape_coarse_check'] = relative < .01
                    assert checks['cache_shape_coarse_check'], relative
                    del plain, altered, changed, change_packet, pref, step, ref, actual_embedding
                identity = {k: array_identity(a) for k, a in packet.items()}
                cp = out / 'commits' / (row['sample_id'] + '.json')
                if cp.exists():
                    prior = read(cp)
                    assert prior['arrays'] == identity, ('Native replay differs', row['sample_id'])
                    replayed = True
                else:
                    pos = row['anchors']
                    logits = model.lm_head(y[:, pos])[0]
                    lp = logits.float().log_softmax(-1)
                    next_ids = [row['prompt_ids'][q+1] for q in pos]
                    commit = {'timestamp': stamp(), 'sample_id': row['sample_id'], 'material_sha': material_digest,
                        'source_sha': source['sha256'], 'arrays': identity, 'checks': {k: bool(v) if isinstance(v, np.bool_) else v for k, v in checks.items()},
                        'observed_next_token_ids': next_ids, 'native_argmax': logits.argmax(-1).cpu().tolist(),
                        'observed_next_token_NLL': [-float(lp[j, tid]) for j, tid in enumerate(next_ids)],
                        'raw_archive': row['sample_id'] in keep, 'no_coordinate_truncation': True}
                    save(cp, commit)
                    replayed = False
                    del logits, lp
                if row['sample_id'] in keep:
                    path = out / 'fields' / (row['sample_id'] + '.npz')
                    if path.exists():
                        with np.load(path) as z:
                            assert all(np.array_equal(z[k], a) for k, a in packet.items())
                    else:
                        guard(sum(a.nbytes for a in packet.values()) + 1024**2)
                        npz(path, **packet)
                store.add(row, packet)
                observer.reset([])
                summaries.append({'sample_id': row['sample_id'], 'seconds': time.monotonic()-tick,
                                  'bytes': sum(a.nbytes for a in packet.values()), 'replayed': replayed})
                assert store.bytes < config['full_stream_memory_estimate_ceiling_bytes']
                assert psutil.virtual_memory().available > config['host_runtime_floor_bytes']
                if i < 4 or (i+1) % 32 == 0:
                    print('JOINT_CAPTURE', 'fresh' if fresh else 'main', i+1, len(material), 'RAM_bytes', store.bytes, 'replay', replayed, flush=True)
                del x, y, packet
                assert time.monotonic()-start < config['per_process_max_seconds']
    finally:
        observer.close()
        del observer, model
        gc.collect()
        torch.cuda.empty_cache()
    seconds = time.monotonic()-start
    ledger('qwen4_native_capture', seconds, sources=len(material), fresh=fresh, pilot=pilot, bytes_in_RAM=store.bytes)
    record = {'timestamp': stamp(), 'sources': len(material), 'tokens': sum(len(r['prompt_ids']) for r in material),
              'seconds': seconds, 'field_RAM_bytes': store.bytes, 'all_replay_hashes_equal': all(r['replayed'] for r in summaries),
              'retained_source_ids': sorted(keep & store.data.keys()), 'no_lossy_coordinate_compression': True}
    stamp_id = str(time.time_ns())
    save(out / 'replays' / (stamp_id + '.json'), record)
    if pilot:
        total = rows() + rows(True)
        raw_bytes = sum((3*len(r['prompt_ids']) + 37*len(r['positions']) + len(r['positions'])) * 2560*2 + 8*len(r['positions']) for r in total)
        estimated_main = raw_bytes * len(rows()) / len(total)
        estimated_seconds = sum(r['seconds'] for r in summaries)/len(summaries)*len(total)
        passed = estimated_main < config['full_stream_memory_estimate_ceiling_bytes'] and estimated_seconds < config['per_process_max_seconds']
        report = {'timestamp': stamp(), 'passed': passed, 'pilot_sources': len(material), 'pilot_arrays_bytes': store.bytes,
            'measured_pilot_including_checks_seconds': seconds, 'estimated_all768_capture_seconds_including_pilot_control_rate': estimated_seconds,
            'exact_main_plus_fresh_array_size_bytes_from_token_counts': raw_bytes,
            'main_RAM_estimate_bytes': estimated_main, 'available_host_after_model_release': psutil.virtual_memory().available,
            'output_strategy': 'Primary fields streamed in bounded RAM and replayable, not all archived; disk estimate must include material/summary/representative budgets.',
            'result_bytes_now': usage(), 'execution_checks': 'Same-shape hooks/suffix tests bitwise; cache/uncached different-shape floor measured separately.'}
        save(BASE / 'capture_pilot.json', report)
        assert passed, report
        print('JOINT_CAPTURE_PILOT', report, flush=True)
    else:
        save(out / 'capture_result.json', record)
    if config.get('archive_all_full_fields', False):
        save(out / 'archive_manifest.json', {'timestamp': stamp(), 'complete_for_requested_material': True,
            'source_ids': [r['sample_id'] for r in material], 'runtime_source_sha': source['sha256'],
            'fields': {r['sample_id']: {'file_sha256': sha(out/'fields'/(r['sample_id']+'.npz')),
                'array_commit_sha256': sha(out/'commits'/(r['sample_id']+'.json'))} for r in material},
            'scope': 'Supersedes initial representative-only retention. Historical commits raw_archive flag records initial capture-time policy, not current file availability.'})
    return store


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--fresh', action='store_true')
    args = p.parse_args()
    cache = capture(pilot=args.pilot, fresh=args.fresh)
    cache.clear()
