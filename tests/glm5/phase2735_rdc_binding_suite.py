"""Run finite own-history and automatic follow-up stages strictly serially."""
import subprocess
import sys
from rdc_binding_common import *

def main():
    assert read(BASE/'scale/suite_result.json')['all_passed']
    jobs=['phase2734_rdc_binding_live.py','phase2735_rdc_binding_long_capture.py',
      'phase2735_rdc_binding_decomposition.py','phase2735_rdc_binding_autonomous.py',
      'phase2735_rdc_binding_signed_capture.py']
    records=[]
    for job in jobs:
        start=time.monotonic();print('BINDING_FOLLOWUP_START',job,flush=True)
        p=subprocess.run([sys.executable,str(ROOT/'tests/glm5'/job)],cwd=ROOT,timeout=7300)
        records.append({'job':job,'returncode':p.returncode,'wall_seconds':time.monotonic()-start})
        save(BASE/'format_content/suite_progress.json',records);assert p.returncode==0,job
    save(BASE/'format_content/suite_result.json',{'timestamp':stamp(),'records':records,'all_passed':True})

if __name__=='__main__':main()
