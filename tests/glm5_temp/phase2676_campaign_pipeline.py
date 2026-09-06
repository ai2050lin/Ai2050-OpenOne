"""Sequential campaign continuation; completion guards and durable logs, no concurrent models."""
import json,subprocess,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT,save
OUT=RESULT/'phase2676_native_mlp_delivery'

def main():
    # Only follow the current observed run, never relaunch it in parallel.
    while not (RESULT/'phase2671_native_mlp_field/analysis/final.json').exists():time.sleep(10)
    commands=[('2672', ['phase2672_native_mlp_paths.py']),('2673',['phase2673_native_mlp_confirmation.py']),
        ('2674_natural',['phase2674_native_mlp_scalar.py','natural']),('2674_fp',['phase2674_native_mlp_scalar.py','fp']),('2674_finalize',['phase2674_native_mlp_scalar.py','finalize']),
        ('2675_qwen14',['phase2675_native_mlp_crossmodel.py','qwen14']),('2675_glm4',['phase2675_native_mlp_crossmodel.py','glm4']),('2675_ds7',['phase2675_native_mlp_crossmodel.py','ds7']),('2675_finalize',['phase2675_native_mlp_crossmodel.py','finalize'])]
    logs=OUT/'runtime';logs.mkdir(parents=True,exist_ok=True)
    for label,args in commands:
        done=logs/(label+'_completed.json')
        if done.exists():continue
        save(OUT/'analysis/pipeline.json',{'runner_pid':os.getpid(),'stage':label,'status':'running','command':args})
        with (logs/(label+'.log')).open('a',encoding='utf-8') as stream:
            completed=subprocess.run([sys.executable,str(ROOT/'tests/glm5'/args[0]),*args[1:]],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:
            save(OUT/'analysis/pipeline.json',{'runner_pid':os.getpid(),'stage':label,'status':'failed','exit_code':completed.returncode,'log':str(logs/(label+'.log'))});raise SystemExit(completed.returncode)
        save(done,{'returncode':completed.returncode,'time':time.strftime('%Y-%m-%d %H:%M:%S')});print('SEQUENTIAL STAGE COMPLETE',label,flush=True)
    save(OUT/'analysis/pipeline.json',{'runner_pid':os.getpid(),'status':'model_and_analysis_stages_complete','remaining':'Human-readable scientific review, independent QA, publish, browser, allowlisted cleanup and2676MEMO still required.'})

if __name__=='__main__':main()
