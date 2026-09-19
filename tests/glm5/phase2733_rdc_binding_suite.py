"""Serial CUDA suite. Child processes must exit before the next model is loaded."""
import subprocess
import sys
from rdc_binding_common import *

def main():
    assert read(BASE/'middle_training/pilot.json')['passed']
    jobs=['phase2733_rdc_binding_middle.py','phase2733_rdc_binding_native.py','phase2733_rdc_binding_alpha.py',
          'phase2733_rdc_binding_gradient_span.py','phase2733_rdc_binding_beta.py']
    immutable(BASE/'suite_protocol.json',{'jobs':jobs,'parallel_GPU_processes':1,
      'pilot_middle_backward_seconds':read(BASE/'middle_training/pilot.json')['forward_backward_seconds'],
      'estimated_middle_training_forward_backward_seconds':512*read(BASE/'middle_training/pilot.json')['forward_backward_seconds'],
      'middle_steps':32,'middle_seeds':2,'middle_conditions':2,'remaining_overhead':'Native panel evaluation/checkpoint writing additional, process capped7200s.',
      'Alpha_Gamma':'Complete declared natural96-percondition and program96-perrepresentation training spans; independent heldoutqueries, actual norm-matched updates.',
      'Beta':'64heldout program representations, oracle/proxy/random normmatched single-step tests with16natural collateral contexts.'})
    progress=[]
    for job in jobs:
        guard(400*1024**2)
        print('BINDING_SERIAL_JOB_START',job,flush=True);tick=time.monotonic()
        p=subprocess.run([sys.executable,str(ROOT/'tests/glm5'/job)],cwd=ROOT,timeout=7200)
        progress.append({'job':job,'returncode':p.returncode,'wall_seconds':time.monotonic()-tick})
        save(BASE/'suite_progress.json',progress)
        assert p.returncode==0,job
        print('BINDING_SERIAL_JOB_COMPLETE',job,flush=True)
    save(BASE/'suite_result.json',{'timestamp':stamp(),'jobs':progress,'all_passed':True,
      'timing':'Suite timers overlap individual script ledgers; do not add them twice.'})

if __name__=='__main__':main()
