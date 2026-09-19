"""Goal-prioritized, append-only continuation of the audited query campaign.

Old scientific artifacts and their resource contracts remain immutable.  New
work has no arbitrary elapsed-time/result-size ceiling; physical safety checks
and measured costs remain explicit.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
import gc, hashlib, shutil, time, traceback
from pathlib import Path
import numpy as np
from rdc_feature_common import ROOT, RESULT, stamp, sha, read, save, immutable, bits, unbits, npz
from rdc_law_common import gzread, compressed, clustered, identity
from rdc_query_common import clone_cache, cache_id, MODELS

OLD = RESULT / 'rdc_query_campaign_20260913'
BASE = RESULT / 'rdc_query_construction_20260913'
MEMO = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'


def rank(value):
    return hashlib.sha256(('construction2745/' + str(value)).encode()).hexdigest()


def snapshot(path):
    path = Path(path)
    digest = sha(path)
    dest = BASE / 'sources' / (path.stem + '_' + digest[:16] + path.suffix)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    assert sha(dest) == digest
    return {'path': str(path), 'sha256': digest, 'snapshot': str(dest.relative_to(BASE))}


def guard(expected_bytes=0):
    """Safety reserve is not an experiment-size quota or a completion criterion."""
    free = shutil.disk_usage(ROOT).free
    assert free - expected_bytes > 4 * 1024**3, ('Physical free-space safety reserve', free, expected_bytes)
    assert not (BASE / 'STOP_REQUESTED').exists(), 'Explicit local stop marker is present'


def ledger(kind, seconds, **extra):
    path = BASE / 'compute_ledger.json'
    lock = BASE / 'compute_ledger.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd = None
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            assert time.monotonic() - start < 15, 'Ledger writer lock did not clear'
            time.sleep(.025)
    try:
        records = read(path) if path.exists() else []
        records.append({'timestamp': stamp(), 'kind': kind, 'seconds': seconds, **extra})
        save(path, records)
    finally:
        os.close(fd)
        lock.unlink()


def failure(out, start, exc):
    seconds = time.monotonic() - start
    save(Path(out) / ('failure_' + str(time.time_ns()) + '.json'),
         {'timestamp': stamp(), 'seconds': seconds, 'error': str(exc), 'traceback': traceback.format_exc()})
    ledger('failed_' + Path(out).name, seconds)


def load(key, out, gpu_limit=None):
    import torch
    import rdc_operator_model as dispatch
    if key in ('qwen14', 'glm4'):
        import faulthandler
        import accelerate.utils.offload as offload_reader
        faulthandler.enable()
        if not getattr(offload_reader.safe_open, '_rdc_pread', False):
            original_open = offload_reader.safe_open
            def checkpoint_pread(*args, **kwargs):
                kwargs['backend'] = 'pread'
                return original_open(*args, **kwargs)
            checkpoint_pread._rdc_pread = True
            offload_reader.safe_open = checkpoint_pread
        # Transformer checkpoint loading already uses pread. The lazy runtime
        # offload reader has a separate imported safe_open; cover that path too.
        save(Path(out) / 'offload_reader.json', {'timestamp': stamp(), 'backend': 'pread',
            'scope': 'Process-local runtime reader override for original checkpoint tensors; no package/checkpoint file edits, quantization or altered math.'})
    dispatch.snapshot = snapshot
    requested = 6
    if key == 'qwen14' and gpu_limit is None:
        memory = dispatch.memory()
        eligible = [v for v in (11, 9, 6)
                    if memory['host_available_bytes'] > (v+4)*1024**3
                    and memory.get('system_commit_headroom', 100*1024**3) > (v+22)*1024**3]
        requested = 11 if eligible else 6
        save(Path(out) / 'residency_choice.json', {
            'timestamp': stamp(), 'memory': memory, 'requested_CPU_GiB': requested,
            'reason': 'Prefer previously tested13GPU adaptiveCPU residency when its measured headroom checks pass; otherwise use the separately tested12GPU/6CPU checkpoint-reference offload. No user process or system setting is changed.'})
    model, tok = dispatch.load(key, Path(out), cpu_gib=requested, gpu_limit=gpu_limit)
    if key == 'glm4' and any(p.device.type != 'cuda' for p in model.model.norm.parameters()):
        # The16source pilot reached all40blocks, then twice crashed in the
        # disk-offload storage read of this8KiB final vector. Keep the exact
        # checkpoint vector resident; do not change the native RMSNorm formula.
        from safetensors import safe_open
        from accelerate.hooks import remove_hook_from_module
        from accelerate.utils import set_module_tensor_to_device
        checkpoint = ROOT / 'models/hf' / MODELS[key]
        mapping = read(checkpoint / 'model.safetensors.index.json')['weight_map']
        with safe_open(str(checkpoint / mapping['model.norm.weight']), framework='pt', device='cpu', backend='pread') as f:
            norm_weight = f.get_tensor('model.norm.weight').clone()
        remove_hook_from_module(model.model.norm)
        set_module_tensor_to_device(model.model.norm, 'weight', 'cuda:0', value=norm_weight)
        assert torch.equal(model.model.norm.weight.cpu(), norm_weight)
        save(Path(out) / 'final_norm_residency.json', {'timestamp': stamp(),
            'original_tensor': 'model.norm.weight', 'bytes': norm_weight.numel()*norm_weight.element_size(),
            'dtype': str(norm_weight.dtype), 'checkpoint_values_bit_equal': True,
            'reason': 'Avoid reproduced Windows disk-offload storage access violation at final normalization. Original parameter values and native module unchanged; requires renewed GLM numerical pilot.'})
        del norm_weight
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert model.dtype == torch.bfloat16 and not getattr(model, 'is_quantized', False)
    save(Path(out) / 'original_precision.json', {
        'timestamp': stamp(), 'model': key, 'dtype': str(model.dtype), 'quantized': False,
        'device_map': getattr(model, 'hf_device_map', None),
        'actual_parameter_devices': sorted({str(p.device) for p in model.parameters()}),
        'configuration_sha256': sha(ROOT / 'models/hf' / MODELS[key] / 'config.json'),
        'loader': snapshot(dispatch.__file__)})
    return model, tok
