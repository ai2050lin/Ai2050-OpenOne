"""Serial driver; children write flushed, per-stage progress logs on D:."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from deploy.storage_policy import check_write
import goodfire_middle as M


def run(model,script,stage):
    name=f'{stage}-{model}.log';path=check_write(M.BASE/name)
    command=[sys.executable,'-B','-u','-X','utf8',str(ROOT/'tests/glm5'/script),model,stage]
    start=time.monotonic();print(f'START {model} {script} {stage} log={path}',flush=True)
    with path.open('wb') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        last=start
        while child.poll() is None:
            time.sleep(2)
            if time.monotonic()-last>=30:
                print(f'RUNNING {model} {stage} elapsed={time.monotonic()-start:.0f}s pid={child.pid}',flush=True)
                last=time.monotonic()
        code=child.returncode
    print(f'END {model} {stage} exit={code} elapsed={time.monotonic()-start:.1f}s',flush=True)
    if code:
        print(path.read_text(encoding='utf-8',errors='replace')[-7000:],flush=True)
        raise SystemExit(code)


def main():
    p=argparse.ArgumentParser();p.add_argument('--start',choices=['qwen4-analysis','qwen14'],default='qwen4-analysis')
    args=p.parse_args()
    models=['qwen4','qwen14'] if args.start=='qwen4-analysis' else ['qwen14']
    for model in models:
        if model=='qwen14':
            run(model,'goodfire_middle.py','pilot');run(model,'goodfire_middle.py','collect')
        run(model,'goodfire_middle_audit.py','native')
        for stage in ['profiles','regimes','predict','neighbors','readout']:
            run(model,'goodfire_middle_analysis.py',stage)
        run(model,'goodfire_middle_audit.py','analysis')
        for stage in ['freeze','calibrate','natural','capability']:
            run(model,'goodfire_middle_intervene.py',stage)
    print('SERIAL MODEL EXPERIMENTS COMPLETE; REPORT, AUDIT AND CLIENT WORK REMAIN',flush=True)


if __name__=='__main__':main()
