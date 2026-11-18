"""Serial driver; never preempts another project's CUDA lease."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import recursive_branch14 as R


def wait_idle():
    start=time.monotonic();stable=None
    while True:
        free=False
        try:
            with R.GpuLease():free=True
        except RuntimeError:pass
        if free:
            if stable is None:stable=time.monotonic()
            if time.monotonic()-stable>=5:return
        else:stable=None
        print(json.dumps(dict(stage='waiting_for_serial_GPU',free=free,elapsed_seconds=round(time.monotonic()-start),
            policy='Wait for 5 seconds continuously free; child must acquire the exclusive GPU lease; do not interrupt another study.')),flush=True)
        time.sleep(2)


def call(name,script,args,gpu=False):
    if gpu:wait_idle()
    log=R.OUT/(name+'.log');cmd=[sys.executable,'-B','-X','utf8','-u',str(R.ROOT/'tests/glm5'/script)]+args
    print(json.dumps(dict(stage='launch',name=name,command=cmd)),flush=True)
    with log.open('w',encoding='utf8') as f:
        process=subprocess.Popen(cmd,cwd=R.ROOT,env=os.environ,stdout=f,stderr=subprocess.STDOUT)
        rc=process.wait()
    print(json.dumps(dict(stage='returned',name=name,code=rc)),flush=True)
    if rc:raise RuntimeError(f'{name} failed ({rc}); inspect {log}')


def run():
    R.configure();R.admission()
    archive=R.OUT/'frozen-source.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name in R.read(R.OUT/'plan.json')['source_hashes']:z.write(R.ROOT/name,name)
            z.write(__file__,'tests/glm5/'+Path(__file__).name)
            z.write(R.ROOT/'tests/glm5/test_recursive_branch14.py','tests/glm5/test_recursive_branch14.py')
    if not (R.OUT/'pilot.json').exists():call('pilot','recursive_branch14.py',['collect','--split','pilot'],gpu=True)
    pilot=R.read(R.OUT/'pilot.json')
    assert pilot['numerical_passed'] and pilot['behavior_passed'],'Pilot admission failed'
    if not (R.OUT/'discovery/collection.json').exists():call('discovery','recursive_branch14.py',['collect','--split','discovery'],gpu=True)
    if not (R.OUT/'selection.json').exists():call('selection','recursive_branch14_analysis.py',['select'])
    for split in R.SPLITS[1:]:
        if not (R.OUT/split/'collection.json').exists():call(split,'recursive_branch14.py',['collect','--split',split],gpu=True)
    if not (R.OUT/'response-report.json').exists():call('response-analysis','recursive_branch14_analysis.py',['analyze'])
    for h in R.CANDIDATES:
        if not (R.OUT/f'intervention-{h}.json').exists():call('intervention-'+h,'recursive_branch14_intervene.py',['run','--hypothesis',h],gpu=True)
    if not (R.OUT/'intervention-report.json').exists():call('intervention-summary','recursive_branch14_intervene.py',['summarize'])
    R.save('pipeline-complete.json',dict(completed=True,source_sha256=R.file_sha(__file__),frozen_source_sha256=R.file_sha(archive)))


if __name__=='__main__':run()
