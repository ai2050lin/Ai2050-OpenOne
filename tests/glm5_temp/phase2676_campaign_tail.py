"""Sequential posthoc-scoped exceptions and numerical resolution; never clean before UI QA."""
import subprocess,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT,save
OUT=RESULT/'phase2676_native_mlp_delivery'

def main():
    while not (RESULT/'phase2675_native_mlp_crossmodel/analysis/final.json').exists():time.sleep(10)
    commands=[('2676_expansion','tests/glm5/phase2676_chronology_expansion.py',['execute']),
        ('2676_resolution','tests/glm5/phase2676_numeric_resolution.py',['execute']),
        ('2676_science','tests/glm5_temp/phase2676_scientific_checks.py',[]),
        ('2676_publish','tests/glm5/phase2676_native_mlp_delivery.py',['publish']),
        ('2676_delivery','tests/glm5_temp/phase2676_delivery_checks.py',['--build'])]
    logs=OUT/'runtime';logs.mkdir(parents=True,exist_ok=True)
    for label,script,args in commands:
        done=logs/(label+'_completed.json')
        if done.exists():continue
        save(OUT/'analysis/tail_pipeline.json',{'runner_pid':os.getpid(),'stage':label,'status':'running','script':script})
        with (logs/(label+'.log')).open('a',encoding='utf-8') as stream:result=subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            save(OUT/'analysis/tail_pipeline.json',{'runner_pid':os.getpid(),'stage':label,'status':'failed','returncode':result.returncode});raise SystemExit(result.returncode)
        save(done,{'returncode':0,'time':time.strftime('%Y-%m-%d %H:%M:%S')});print('TAIL STAGE COMPLETE',label,flush=True)
    save(OUT/'analysis/tail_pipeline.json',{'runner_pid':os.getpid(),'status':'publication_and_direct_checks_complete','remaining':'ActualHTTP/backendrefresh, real browsernumeric+visualQA, explicitallowlistedcleanup, postcleanupQA, nextwholecampaigncontract andMEMO2676 stillrequired.'})

if __name__=='__main__':main()
