"""Use newly user-cleared D space; leave old run budgets and all old fields unchanged."""
import shutil
from rdc_joint_common import *
from rdc_joint_capture import capture, FieldStore, array_identity


def load_archived(fresh=False):
    material = rows(fresh)
    folder = BASE/('fresh' if fresh else 'main')
    manifest = read(folder/'archive_manifest.json')
    assert manifest['complete_for_requested_material']
    store = FieldStore(material, fresh)
    for r in material:
        p = folder/'fields'/(r['sample_id']+'.npz')
        assert sha(p) == manifest['fields'][r['sample_id']]['file_sha256']
        z = field(r,fresh)
        cp = read(folder/'commits'/(r['sample_id']+'.json'))
        assert cp['arrays'] == {k:array_identity(v) for k,v in z.items()}
        store.add(r,z)
    print('JOINT_ARCHIVE_READ',len(material),store.bytes,flush=True)
    return store


def main():
    allocation = read(BASE/'resource_allocation.json')
    if not allocation.get('archive_all_full_fields',False):
        free = shutil.disk_usage(ROOT).free
        assert free > 20*1024**3, 'Need measured new D space, do not infer C permission'
        immutable(BASE/'resource_allocation_initial_streamed.json',allocation)
        revised = {**allocation,'timestamp':stamp(),'status':'Full D-drive archive allocation after user reports D cleanup; actual free space verified.',
            'result_ceiling_bytes':3*1024**3,'archive_all_full_fields':True,'measured_D_free_bytes_at_revision':free,
            'pending_larger_storage_choice':None,'allocation_revision_authority':'User explicitly says D now has substantial space and continue; stay in project result path, no C directory or junction.',
            'storage_mode':'Complete requested native BF16 fields losslessly archived plus full-coordinate features, models, probability and parameter summaries; engineer-selected3GiB bounded result allowance,8GiB volume floor unchanged.',
            'retention':'All archived raw fields will be connected to local query client. Do not delete old or unrelated user data.'}
        save(BASE/'resource_allocation.json',revised)
        save(BASE/'execution_update.json',{'timestamp':stamp(),'initial_plan_status_superseded':True,'scientific_work_packages_unchanged':True,
            'pilot_and_checks_passed':True,'phase2719_observation_complete':True,'new_D_space_bytes':free,
            'next':'Exact bitwise recapture archives all512 main arrays; fresh256 remains frozen and uncaptured until Phase2720 model selection.',
            'recorded_user_messages':'D盘已经有较大空间了，请继续完成以上任务', 'no_user_data_deleted_by_agent':True})
    if not (BASE/'main/archive_manifest.json').exists():
        store = capture()
        store.clear()
    store = load_archived()
    store.clear()
    guard()
    print('JOINT_ALL_MAIN_ARCHIVED',usage(),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
