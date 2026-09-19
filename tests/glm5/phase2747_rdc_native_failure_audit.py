"""Preserve the actual Windows native-runtime failure before an identical retry."""
from rdc_formation_common import *
import argparse


def main(retry=False):
    folder=OUT/'engineering/qwen14_own_load_failure'
    label='faulthandler_retry' if retry else 'initial_failure'
    target=folder/(label+'.json')
    if target.exists():return
    original=OUT/'own_history/qwen14/native/pilot_load'
    files=[]
    for p in sorted(original.rglob('*.json')):
        destination=folder/(label+'_load')/p.relative_to(original)
        destination.parent.mkdir(parents=True,exist_ok=True);assert not destination.exists()
        shutil.copyfile(p,destination);assert sha(p)==sha(destination)
        files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'preserved':destination.relative_to(ROOT).as_posix()})
    assert len(files)==4
    save(target,{'timestamp':stamp(),'source':snapshot(__file__),'actual_failure_time_local':'2026-09-14 07:42:38' if retry else '2026-09-14 07:38:47',
        'tool_exit_code':1,'Windows_Application_Error_ID':1000,'exception_code':'0xc0000005',
        'faulting_module':'torch_cpu.dll','faulting_process_id_hex':'0x1C34' if retry else '0x3394',
        'Windows_report_ID':'f6b3fa1d-05e3-4904-8221-193d39a9ebe1' if retry else 'dd840d1c-3115-459f-9e91-e6bf3405ef28',
        'checkpoint_load_completed':True,'own_generation_completed':False,'Python_failure_receipt_present':False,
        'cause':'Native runtime checkpoint tensor storage access during an offloaded Qwen3MLP linear pre-forward; particular tensor not identified.' if retry else 'Not yet localized. Native memory access violation is not an experimental model failure.',
        'actual_stack_locations':['torch/storage.py:471 __getitem__','accelerate/utils/offload.py:171 __getitem__',
            'accelerate/hooks.py:371 pre_forward','transformers/models/qwen3/modeling_qwen3.py:82 Qwen3MLP forward',
            'phase2747_rdc_own_history.py:18 native_admission'] if retry else [],
        'next':'Apply the existing process-local pread runtime loader to Qwen14 and repeat full numerical admission.' if retry else 'Exact original-precision pilot retry with Python faulthandler enabled; no model, sample or scoring modification.',
        'files':files,'engine':snapshot(Path(__file__).with_name('phase2747_rdc_own_history.py')),
        'common_loader':snapshot(Path(__file__).with_name('rdc_construction_common.py'))})
    print('FORMATION_NATIVE_FAILURE_PRESERVED',len(files),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--retry',action='store_true');main(parser.parse_args().retry)
