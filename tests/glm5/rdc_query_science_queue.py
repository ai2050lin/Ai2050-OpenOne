"""Finite, fail-closed serial queue after million-endpoint capture and rule fit."""
import subprocess,sys
from rdc_query_common import *

TASKS=[
 (2740,'atlas',['phase2740_rdc_query_atlas.py']),
 (2740,'events',['phase2740_rdc_query_events.py']),
 (2740,'event_analysis',['phase2740_rdc_query_analysis.py']),
 (2740,'atlas_figures',['phase2743_rdc_query_figures.py','2740']),
 (2741,'rule_vocabulary',['phase2741_rdc_query_vocabulary.py']),
 (2741,'pairs',['phase2741_rdc_query_pairs.py']),
 (2741,'transfer_capture',['phase2741_rdc_query_transfer.py','capture']),
 (2741,'transfer_fit',['phase2741_rdc_query_transfer.py','fit']),
 (2741,'transfer_vocabulary',['phase2742_rdc_query_injection.py','vocabulary']),
 (2741,'prediction_analysis',['phase2741_rdc_query_analysis.py']),
 (2741,'prediction_figures',['phase2743_rdc_query_figures.py','2741']),
 (2742,'formation_pilot',['phase2742_rdc_query_formation.py','--pilot']),
 (2742,'formation',['phase2742_rdc_query_formation.py']),
 (2742,'injection',['phase2742_rdc_query_injection.py','injection']),
 (2742,'late_pilot',['phase2742_rdc_query_late.py','native','--pilot']),
 (2742,'late_native',['phase2742_rdc_query_late_batch.py','native']),
 (2742,'late_fixed',['phase2742_rdc_query_late_batch.py','fixed128_digit']),
 (2742,'late_entropy',['phase2742_rdc_query_late_batch.py','entropy_digit']),
 (2742,'late_letter',['phase2742_rdc_query_late_batch.py','entropy_letter']),
 (2742,'late_marker',['phase2742_rdc_query_late_batch.py','terminal_marker_digit']),
 (2742,'scale_qwen4',['phase2742_rdc_query_scale.py','qwen4']),
 (2742,'scale_qwen14',['phase2742_rdc_query_scale.py','qwen14']),
 (2742,'scale_glm4',['phase2742_rdc_query_scale.py','glm4']),
 (2742,'formation_history_scale_analysis',['phase2742_rdc_query_analysis.py']),
 (2742,'formation_history_scale_figures',['phase2743_rdc_query_figures.py','2742']),
 (2743,'independent_followup',['phase2743_rdc_query_followup.py','run'])]

def main():
    assert read(BASE/'queue/status.json')['all_passed'],'Finish the first CUDA queue; never overlap model jobs'
    out=BASE/'science_queue';out.mkdir(parents=True,exist_ok=True);records=[]
    if (out/'status.json').exists():save(out/'status_history'/(sha(out/'status.json')+'.json'),read(out/'status.json'))
    for j,(phase,name,args) in enumerate(TASKS):
        guard();print('SERIAL_QUERY_SCIENCE_START',phase,name,stamp(),flush=True);start=time.monotonic()
        save(out/'status.json',{'tasks':records,'active':name,'phase':phase,'all_passed':False})
        command=[sys.executable,'-X','utf8',str(ROOT/'tests/glm5'/args[0]),*args[1:]]
        with (out/(name+'.log')).open('a',encoding='utf-8') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=7300)
        record={'task':name,'phase':phase,'returncode':result.returncode,'seconds':time.monotonic()-start,'timestamp':stamp()};records.append(record)
        save(out/'status.json',{'tasks':records,'active':None,'all_passed':result.returncode==0 and j+1==len(TASKS)})
        print('SERIAL_QUERY_SCIENCE_DONE',record,flush=True);assert result.returncode==0,record
        if j+1==len(TASKS) or TASKS[j+1][0]!=phase:
            save(out/f'phase{phase}_experiments.json',{'timestamp':stamp(),'phase':phase,'experiments_complete':True,
              'delivery_scope':'This marks the listed experiment bundle only; memo, scientific synthesis and final integrity are separate required deliverables.',
              'tasks':[r for r in records if r['phase']==phase]})

if __name__=='__main__':main()
