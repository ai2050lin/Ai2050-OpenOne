"""Resource-admitted whole same-goal continuation, serial native models only."""
import subprocess,sys
from rdc_query_common import *


def main():
    assert read(BASE/'science_queue/status.json')['all_passed']
    assert read(BASE/'followup/result.json')['all_passed']
    resources=read(BASE/'resources.json');used=sum(r['seconds'] for r in read(BASE/'compute_ledger.json'));size=usage();free=shutil.disk_usage(ROOT).free
    expected=650*1024**2;remain=resources['compute_ceiling_seconds']-used
    admitted=remain>2000 and size+expected<resources['result_ceiling_bytes'] and free-expected>resources['disk_floor_bytes']
    admission={'timestamp':stamp(),'same_goal':True,'proposed_phase':2744,'already_executed_same_goal_followup':2743,
      'recorded_seconds_used':used,'remaining_seconds':remain,'current_result_bytes':size,'remaining_result_bytes':resources['result_ceiling_bytes']-size,
      'disk_free_bytes':free,'expected_complete_stage_extra_bytes':expected,'whole_stage_reference_seconds':1800,'final_audit_reserve_seconds':200,
      'same_goal_complete_stage_admitted':admitted,
      'scientific_question':'Identify strict token-matched relational sensitivity and distinguish actual continued-training gains from held-validation temperature and training-prior calibration.',
      'finite_bundle':'320controlledexpressions/160matchedpairs/80groups/5families; fiveactual native parameter variants; native100queries and fourvariants6queries; unchanged decoderfullvocabtests;1600B8ownhistorytrajectories capped128;384calibration/confirmationcontentpositions.',
      'estimate_scope':'Bounded protocol estimate, not an exact future runtime or physical lower bound. No top-coordinate reduction; full unit/layer/query arrays at the explicitly declared sampling boundaries. No evidence deletion to make room.'}
    file=BASE/'next_stage_admission.json'
    if file.exists():
        prior=read(file);admission=prior;admitted=prior['same_goal_complete_stage_admitted']
    else:immutable(file,admission)
    if not admitted:print('IDENTITY_WHOLE_STAGE_NOT_ADMITTED',admission,flush=True);return
    out=BASE/'identifiability/queue';status=out/'status.json'
    if status.exists() and read(status).get('all_passed'):return
    if status.exists():save(out/'status_history'/f'{sha(status)}.json',read(status))
    tasks=[('freeze',['phase2744_rdc_query_identifiability.py','freeze']),('relations',['phase2744_rdc_query_capture.py','relations']),
      ('calibration',['phase2744_rdc_query_capture.py','calibration']),('behavior',['phase2744_rdc_query_behavior.py']),('analysis',['phase2744_rdc_query_analysis.py']),
      ('figures',['phase2744_rdc_query_figures.py'])]
    done=[]
    for name,args in tasks:
        guard();save(status,{'tasks':done,'active':name,'all_passed':False});out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
        with (out/f'{name}.log').open('ab') as log:
            p=subprocess.run([sys.executable,'-X','utf8']+[str(ROOT/'tests/glm5'/args[0])]+args[1:],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=2100)
        done.append({'task':name,'timestamp':stamp(),'returncode':p.returncode,'seconds':time.monotonic()-start});save(status,{'tasks':done,'active':None,'all_passed':False})
        assert p.returncode==0,(name,p.returncode,str(out/f'{name}.log'))
    save(status,{'tasks':done,'active':None,'all_passed':True});print('IDENTITY_WHOLE_STAGE_COMPLETE',flush=True)


if __name__=='__main__':main()
