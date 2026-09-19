"""Finite serial CUDA queue: each process exits before the next model is loaded."""
import subprocess,sys
from rdc_query_common import *

def main():
    out=BASE/'queue';out.mkdir(parents=True,exist_ok=True)
    tasks=[('capture_'+str(b),['phase2740_rdc_query_capture.py','--begin',str(b),'--end',str(b+2000)]) for b in [2000,4000,6000,8000]]
    tasks += [('rules',['phase2741_rdc_query_rules.py']),('fit',['phase2741_rdc_query_fit.py'])]
    records=[]
    assert (BASE/'capture/chunks/00000_02000.json').exists()
    for name,args in tasks:
        guard();print('SERIAL_QUERY_START',name,stamp(),flush=True);start=time.monotonic()
        command=[sys.executable,'-X','utf8',str(ROOT/'tests/glm5'/args[0]),*args[1:]]
        with (out/(name+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=7300)
        record={'task':name,'returncode':result.returncode,'seconds':time.monotonic()-start,'timestamp':stamp()};records.append(record)
        save(out/'status.json',{'tasks':records,'active':None,'all_passed':result.returncode==0 and len(records)==len(tasks)})
        print('SERIAL_QUERY_DONE',record,flush=True)
        assert result.returncode==0,record

if __name__=='__main__':main()
