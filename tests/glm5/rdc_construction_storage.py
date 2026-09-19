"""Registered task-only overflow storage; never moves or removes old evidence."""
from rdc_construction_common import *

FIELD_STORE=BASE/'phase2746/field_store'
PHYSICAL_STORE=BASE/'phase2746_fields'


def verify_storage(expected_bytes=0):
    assert FIELD_STORE.is_junction() and FIELD_STORE.resolve()==PHYSICAL_STORE.resolve()
    assert PHYSICAL_STORE.is_dir()
    free=shutil.disk_usage(PHYSICAL_STORE).free
    assert free-expected_bytes>4*1024**3,('Physical archive reserve',free,expected_bytes)
    guard(0)
    manifest=BASE/'phase2746/storage.json'
    if not manifest.exists():
        immutable(manifest,{'timestamp':stamp(),'source':snapshot(__file__),
            'logical_result_entry':str(FIELD_STORE),'physical_directory':str(PHYSICAL_STORE),
            'entry_type':'Windows NTFS directory junction, created only for new Phase2746 fields',
            'reason':'D free capacity insufficient for planned complete-unit trajectories; C has available physical space.',
            'initial_free_bytes_C':free,'initial_free_bytes_D':shutil.disk_usage(ROOT).free,
            'old_model_or_result_files_moved_or_deleted':0,
            'retention':'Research/client evidence; do not remove junction or physical directory while any index relies on it.'})
    return FIELD_STORE


if __name__=='__main__':
    print(verify_storage(),flush=True)
