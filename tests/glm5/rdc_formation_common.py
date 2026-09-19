"""Phase2747 uses a separately registered result-tree store; no old quota edits."""
from rdc_construction_common import *

OUT = BASE / 'phase2747'
FIELDS = OUT / 'field_store'
PHYSICAL = BASE / 'phase2747_fields'


def storage_guard(expected=0):
    guard()
    assert FIELDS.exists() and FIELDS.resolve() == PHYSICAL.resolve()
    assert shutil.disk_usage(PHYSICAL).free - expected > 4 * 1024**3
    if not (OUT / 'storage.json').exists():
        immutable(OUT / 'storage.json', {'timestamp': stamp(), 'source': snapshot(__file__),
            'logical_result_entry': str(FIELDS), 'physical_directory': str(PHYSICAL),
            'entry_type': 'New task-only Windows NTFS junction',
            'initial_free_bytes_C': shutil.disk_usage(PHYSICAL).free,
            'initial_free_bytes_D': shutil.disk_usage(ROOT).free,
            'old_model_or_result_files_moved_or_deleted': 0,
            'retention': 'Full training parameters and displayed scientific fields; retain while indexed.'})
    return FIELDS


def commit_array(category, name, **arrays):
    storage_guard(sum(a.nbytes for a in arrays.values()))
    path = FIELDS / category / (name + '.npz')
    assert not path.exists(), ('Preserve existing field', path)
    npz(path, **arrays)
    receipt = {'timestamp': stamp(), 'field_path': path.relative_to(ROOT).as_posix(),
               'physical_path': str(path.resolve()), 'field_sha256': sha(path),
               'bytes': path.stat().st_size, 'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}}
    save(OUT / category / 'commits' / (name + '.json'), receipt)
    return receipt
