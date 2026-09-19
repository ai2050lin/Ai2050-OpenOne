"""Bounded subprocess queue: at most one CUDA experiment/model at any time."""
import argparse,subprocess,sys
from rdc_update_common import *

STAGES={
 'core':[
  ['phase2737_rdc_update_audit.py'],['phase2738_rdc_update_native_paths.py'],['phase2738_rdc_update_language_prediction.py'],
  ['phase2739_rdc_update_fresh.py','--stage','freeze'],['phase2739_rdc_update_fresh.py','--stage','capture'],
  ['phase2739_rdc_update_fresh.py','--stage','confirm'],['phase2739_rdc_update_predictive_state.py'],
  ['phase2739_rdc_update_long_answers.py','--pilot']],
 'replication':[['phase2738_rdc_update_same_history.py']]+[['phase2738_rdc_update_scale.py','--model',m] for m in ('qwen4','qwen14','glm4')],
 'finish_native':[['phase2739_rdc_update_scale_batch.py','--model',m] for m in ('glm4','qwen4')]+[['phase2739_rdc_update_causal_replay.py']],
 'long':[['phase2739_rdc_update_long_answers.py']]}

def main(stage):
    out=BASE/'serial_suite';records=[];attempt=out/f'{stage}_{time.time_ns()}_progress.json'
    for args in STAGES[stage]:
        guard();start=time.monotonic();print('SERIAL_NATIVE_START',args,flush=True)
        try:
            process=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tests/glm5'/args[0])]+args[1:],cwd=ROOT,timeout=7250)
            record={'job':args,'exit_code':process.returncode,'wall_seconds':time.monotonic()-start}
        except subprocess.TimeoutExpired:
            record={'job':args,'exit_code':None,'timeout':7250,'wall_seconds':time.monotonic()-start}
        records.append(record);save(out/f'{stage}_progress.json',records);save(attempt,records)
        assert record['exit_code']==0,record
    save(out/f'{stage}_result.json',{'timestamp':stamp(),'source':snapshot(__file__),'stage':stage,'jobs':records,'passed':True,'parallel_CUDA_processes':1,
      'timing':'Supervisor durations overlap child-ledger entries and are not added again to compute ledger.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=STAGES);main(p.parse_args().stage)
