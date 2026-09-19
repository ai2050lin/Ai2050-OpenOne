"""Load and release three original checkpoints strictly one at a time."""
import subprocess
import sys
from rdc_binding_common import *

def main():
    rows=[]
    for key in ('qwen4','qwen14','glm4'):
        print('BINDING_SCALE_JOB_START',key,flush=True);start=time.monotonic()
        p=subprocess.run([sys.executable,str(ROOT/'tests/glm5/phase2734_rdc_binding_scale.py'),'--model',key],cwd=ROOT,timeout=7300)
        rows.append({'model':key,'returncode':p.returncode,'wall_seconds':time.monotonic()-start})
        save(BASE/'scale/suite_progress.json',rows)
        assert p.returncode==0,key
    save(BASE/'scale/suite_result.json',{'timestamp':stamp(),'rows':rows,'all_passed':True,'parallel_models':1})

if __name__=='__main__':main()
