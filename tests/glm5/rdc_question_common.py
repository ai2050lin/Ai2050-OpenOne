"""Phase2748 inputs, identity receipts and bounded physical resource guards."""
from rdc_construction_common import *
from phase2748_rdc_experiment_contract import effective_contract

OUT = BASE/'phase2748'
FIELDS = OUT/'field_store'
PHYSICAL = BASE/'phase2748'


def storage_guard(expected=0):
    guard()
    assert OUT.resolve() == PHYSICAL.resolve()
    assert shutil.disk_usage(FIELDS).free-expected > 4*1024**3


def material(key, confirmation=False):
    contract = effective_contract()
    manifest = read(OUT/'material/manifest.json')
    assert sha(OUT/'material/manifest.json') == contract['material_manifest_sha256']
    for ref in [manifest['rows'], manifest['context_groups'], manifest['model_token_files'][key]]:
        assert sha(ROOT/ref['path']) == ref['sha256']
    rows = gzread(ROOT/manifest['rows']['path'])
    tokenrows = gzread(ROOT/manifest['model_token_files'][key]['path'])
    tokens = {r['question_id']: r for r in tokenrows}
    rows = [{**r, 'tokens': tokens[r['question_id']]} for r in rows]
    groups = gzread(ROOT/manifest['context_groups']['path'])
    if confirmation:
        certificate = read(OUT/'confirmation/freeze_certificate.json')
        assert certificate['all_passed'] and certificate['material_manifest_sha256'] == sha(OUT/'material/manifest.json')
        rows = [r for r in rows if r['split'] == 'confirmation']
        groups = [g for g in groups if g['split'] == 'confirmation']
    else:
        rows = [r for r in rows if r['split'] != 'confirmation']
        groups = [g for g in groups if g['split'] != 'confirmation']
    return contract, manifest, rows, groups


def commit_arrays(folder, name, arrays):
    path = FIELDS/folder/(name+'.npz')
    receipt = OUT/folder/'commits'/(name+'.json')
    if receipt.exists():
        old = read(receipt)
        assert sha(path) == old['sha256']
        with np.load(path) as z:
            assert set(z.files) == set(arrays) and all(np.array_equal(z[k], v) for k, v in arrays.items())
        return old
    assert not path.exists(), ('Uncommitted field already exists; audit it explicitly', path)
    storage_guard(sum(a.nbytes for a in arrays.values()))
    npz(path, **arrays)
    record = {'timestamp': stamp(), 'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
        'bytes': path.stat().st_size, 'uncompressed_bytes': sum(a.nbytes for a in arrays.values()),
        'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}}
    immutable(receipt, record)
    return record


def tensor_bits_to_cuda(array, device='cuda'):
    import torch
    return torch.from_numpy(np.asarray(array, dtype=np.uint16).copy()).view(torch.bfloat16).to(device)


def complete_vocabulary(model, postnorm_BF16, target_id=None):
    import torch
    h = tensor_bits_to_cuda(postnorm_BF16, next(model.lm_head.parameters()).device)[None]
    with torch.inference_mode():
        logits = model.lm_head(h).float()[0]
        lp = logits.double().log_softmax(-1)
        p = lp.exp()
        choice = int(logits.argmax())
        values = {'argmax': choice, 'entropy_FP64': float(-(p*lp).sum()),
                  'chosen_probability_FP64': float(p[choice])}
        if target_id is not None:
            values['first_teacher_token_NLL'] = float(-lp[target_id])
    return values
