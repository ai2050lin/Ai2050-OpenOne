"""Explicit bounded allocation for the authorized post-plan discriminating experiments."""
from rdc_joint_common import *


def main():
    path=BASE/'resource_allocation.json';out=BASE/'extension/resource_revision.json'
    if out.exists():return
    previous=read(path);extra=128*1024**2;free=shutil.disk_usage(BASE).free
    assert previous['result_ceiling_bytes']==3*1024**3
    assert free-extra>previous['result_volume_floor_bytes']
    revision={'timestamp':stamp(),'source':snapshot(Path(__file__)),'previous':previous,'previous_sha256':sha(path),'extra_result_bytes':extra,
        'new_ceiling_bytes':previous['result_ceiling_bytes']+extra,'current_usage_bytes':usage(),'measured_volume_free_bytes':free,
        'reason':'The user authorized automatic same-goal continuation and confirmed D-drive cleanup. New actual frozen-tail confirmation, event all-layer factorization, temperature geometry and client query evidence were added after the initial plan.128MiB explicit engineering extension, NOT a user-specified numeric budget or unbounded loop.',
        'unchanged':'8GiB free-volume floor,4h measured compute,5400s per process,one CUDA model at a time,no quantization,no old field deletion.'}
    save(out,revision)
    save(path,{**previous,'timestamp':stamp(),'result_ceiling_bytes':revision['new_ceiling_bytes'],'automatic_continuation_revision':'extension/resource_revision.json',
        'retention':'Original complete field archives and additional finite event/counterpart/fixture fields queryable in client; tail nonfixture raw buffers released after all-coordinate moments/event extraction/array SHA, with exact replay inputs.'})
    # Failed Q14 preflight did not load weights. Its duration was not timed; explicitly charge a conservative bound.
    from rdc_joint_capture import ledger
    ledger('qwen14_host_preflight_failed_attempt_conservative_charge',10,actual_exact_duration=False,observed_available_bytes=11730288640,old_required_bytes=11*1024**3,no_model_loaded=True)
    print('BOUNDED_EXTENSION_ALLOCATION',revision['new_ceiling_bytes'],free,flush=True)


if __name__=='__main__':main()
