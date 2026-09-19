"""Finite Qwen14 four-case pilot, strictly after Qwen4 process termination."""
import os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from rdc_conditional_common import CAMPAIGN,read,save,stamp


def main():
    import psutil
    begin=time.monotonic()
    while True:
        if (CAMPAIGN/'pause_requested').exists():return
        path=CAMPAIGN/'l_aligned_qwen4/status.json'
        running=any('phase2706_rdc_content_aligned_capture.py' in (cmd:=' '.join(p.info.get('cmdline') or [])) and 'qwen4' in cmd for p in psutil.process_iter(['cmdline']) if p.pid!=os.getpid())
        if path.exists() and read(path).get('state')=='captured' and not running:break
        if time.monotonic()-begin>600:raise RuntimeError('Qwen4 wait bound exceeded')
        time.sleep(2)
    with (ROOT/'tests/glm5_temp/phase2706_q14_pilot.stdout.log').open('w',encoding='utf-8') as out,(ROOT/'tests/glm5_temp/phase2706_q14_pilot.stderr.log').open('w',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'tests/glm5/phase2706_rdc_content_aligned_capture.py','qwen14','--limit','4'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        save(CAMPAIGN/'jobs/q14_pilot.json',{'timestamp':stamp(),'pid':p.pid})
        assert p.wait()==0,'Qwen14 pilot failed'
    save(CAMPAIGN/'jobs/q14_pilot_ready.json',{'timestamp':stamp(),'next':'Inspect forward timing, noops and physical device map before formal256.'})


if __name__=='__main__':main()
