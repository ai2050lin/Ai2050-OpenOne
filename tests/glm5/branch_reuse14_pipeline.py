"""Bounded serial confirmation, prediction and intervention pipeline."""
import subprocess
import sys
import time
import branch_reuse14 as B
import branch_reuse14_execution as E
from branch_reuse14_behavior_guard import check


def run():
    B.configure();E.admit();start=time.monotonic()
    steps=[('confirm',['branch_reuse14_collect.py','collect','--split','confirm'],'confirm/collection.json'),
           ('analyze',['branch_reuse14_analysis.py','analyze'],'response-and-graph-report.json'),
           ('output',['branch_reuse14_output.py'],'output-fidelity.json'),
           ('intervention-B19',['branch_reuse14_intervene.py','run','--kind','B19'],'intervention-B19/result.json'),
           ('intervention-B36',['branch_reuse14_intervene.py','run','--kind','B36'],'intervention-B36/result.json'),
           ('intervention-summary',['branch_reuse14_intervene.py','summarize'],'intervention-report.json')]
    for index,(name,args,artifact) in enumerate(steps):
        E.admit()
        if (B.OUT/artifact).exists():
            print('Verified completed step exists:',name,flush=True)
        else:
            B.save('pipeline-state.json',dict(stage=name,status='running',elapsed_seconds=time.monotonic()-start,
                command=[sys.executable,'-B','-X','utf8','-u',str(B.ROOT/'tests/glm5'/args[0]),*args[1:]],source_sha256=B.file_sha(__file__)))
            with (B.OUT/(name+'.log')).open('a',encoding='utf8') as log:
                code=subprocess.call([sys.executable,'-B','-X','utf8','-u',str(B.ROOT/'tests/glm5'/args[0]),*args[1:]],cwd=B.ROOT,stdout=log,stderr=subprocess.STDOUT)
            if code:
                B.save('pipeline-state.json',dict(stage=name,status='failed',exit_code=code,source_sha256=B.file_sha(__file__)))
                raise RuntimeError(f'{name} failed: inspect its log before recovery')
            assert (B.OUT/artifact).exists()
        if name=='confirm' and not check('confirm')['passed']:
            B.save('pipeline-state.json',dict(stage=name,status='behavior_gate_failed',source_sha256=B.file_sha(__file__),
                reason='Required behavior cells did not all pass. No affected internal analysis or intervention starts.'))
            raise RuntimeError('Formal behavior gate failed; review eligibility')
        B.log('confirmation_pipeline',index+1,len(steps),start,completed=name)
    B.save('pipeline-state.json',dict(status='completed',seconds=time.monotonic()-start,source_sha256=B.file_sha(__file__)))


if __name__=='__main__':run()
