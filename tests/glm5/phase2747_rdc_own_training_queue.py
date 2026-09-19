"""Exactly six frozen trained Q4 deployments, sequential exclusive CUDA jobs."""
import subprocess, sys
from rdc_formation_common import *


def main():
    assert read(OUT/'own_history/qwen4/native/result.json')['all_passed']
    finished=[]; start=time.monotonic()
    for run in read(OUT/'training/result.json')['runs']:
        name=run['condition']+'_'+str(run['seed'])
        command=[sys.executable,'-X','utf8',str(ROOT/'tests/glm5/phase2747_rdc_own_history.py'),'qwen4','--variant',name]
        result=subprocess.run(command,cwd=ROOT)
        assert result.returncode==0, ('Own-history training queue stopped at failed run; prior files retained', name, result.returncode)
        assert read(OUT/'own_history/qwen4'/name/'result.json')['all_passed']
        finished.append(name)
        save(OUT/'own_history/trained_queue/progress.json',{'timestamp':stamp(),'complete':finished,'total':6})
    save(OUT/'own_history/trained_queue/result.json',{'timestamp':stamp(),'source':snapshot(__file__),
        'all_passed':True,'completed':finished,'seconds':time.monotonic()-start,
        'scope':'Six declared variants only, no unbounded queue or concurrent GPU models.'})


if __name__=='__main__': main()
