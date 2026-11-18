"""Record observed end of the other model tests before local 4B execution."""
from datetime import datetime
import source_output_core as S
from source_output_core import C
from source_output_execute import guard
from shared.dist_protocol.mechanism_trace import GpuLease

assert not list(C.OUT.glob('behavior-calibration-[0-9]*.json'))
assert not (C.OUT/'execution-resume.json').exists()
other=C.ROOT/'tests/glm5/result/qwen14_mlp_atlas_v1'
progress=C.read(other/'capability_exact-progress.json')
assert progress['completed']==progress['total']==512
with GpuLease():
    free=guard()
    C.save('execution-resume.json',{
        'schema':'source_output_execution_resume.v1','execution_only':True,
        'time':datetime.now().astimezone().isoformat(),
        'other_thread':'01a1a045-c160-7150-acad-c557907e4e8a',
        'observed_cursor':'98b8e4ee-bbc7-4cb5-86ae-9f3d1d1ca1fd:72',
        'other_model_tests_completed':True,'other_thread_still_active_for_client_and_packaging':True,
        'other_confirmation':'512 道能力保持题也已跑完。接下来把真实结果接入客户端。',
        'other_capability_completed':progress['completed'],'other_capability_total':progress['total'],
        'other_progress_sha256':C.sha(other/'capability_exact-progress.json'),
        'other_log_sha256':C.sha(other/'capability-exact.log'),
        'C_free_GiB':free,'gpu_lease_available':True,
        'process_check':'Windows CIM verified no qwen14_mlp_intervene.py or qwen14_mlp_atlas.py process before resume',
        'cleanup_summary_sha256':C.sha(C.OUT/'storage-cleanup-summary.json'),
        'no_4B_behavior_exposure_before_resume':True,
    })
print('14B tests complete; 4B execution resume recorded',flush=True)
