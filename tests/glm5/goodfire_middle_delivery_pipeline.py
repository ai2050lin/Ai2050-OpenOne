"""CPU-only audited delivery, following each serial model experiment as it finishes."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import goodfire_middle as M
from deploy.storage_policy import check_write


def child(label,args):
    logfile=check_write(M.BASE/(label+'.log'));start=time.monotonic()
    print('START',label,flush=True)
    with logfile.open('wb') as stream:
        process=subprocess.Popen([sys.executable,'-B','-u','-X','utf8',*map(str,args)],cwd=M.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        last=start
        while process.poll() is None:
            time.sleep(2)
            if time.monotonic()-last>=30:
                print(f'RUNNING {label} elapsed={time.monotonic()-start:.0f}s pid={process.pid}',flush=True);last=time.monotonic()
    print(f'END {label} exit={process.returncode} elapsed={time.monotonic()-start:.1f}s',flush=True)
    if process.returncode:
        print(logfile.read_text(encoding='utf-8',errors='replace')[-7000:],flush=True)
        raise SystemExit(process.returncode)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['both',*M.SPECS],default='both');a=p.parse_args()
    for key in (M.SPECS if a.model=='both' else [a.model]):
        M.configure(key);start=time.monotonic()
        while not (M.OUT/'capability_exact-effects.json').exists():
            if time.monotonic()-start>4*3600:raise TimeoutError('Experiment did not finish in delivery wait budget')
            print(f'WAIT {key} experiment complete artifacts elapsed={time.monotonic()-start:.0f}s',flush=True);time.sleep(30)
        M.require_admission()
        if not (M.OUT/'manifest.json').exists():
            raw=sum(p.stat().st_size for p in M.OUT.iterdir() if p.is_file())
            # Conservative: all remaining experiment allowance plus a full-size archive.
            need=34*2**30+int(raw*1.02)+2*2**30
            check_write(M.BASE,need,operation='Middle atlas transport plus remaining experiment reserve')
            M.write(M.OUT/'delivery-storage-admission.json',dict(passed=True,additional_peak_bytes=need,
                current_native_bytes=raw,free_bytes=shutil.disk_usage(M.BASE).free,
                scope='Prospective delivery allowance; includes existing 34 GiB experiment upper bound again conservatively. No science threshold change.'))
            for stage in ['build','plots']:
                child(f'{stage}-{key}',[M.ROOT/'tests/glm5/goodfire_middle_report.py',key,stage])
            child(f'digest-{key}',[M.ROOT/'tests/glm5/goodfire_middle_digest.py',key])
            child(f'manifest-{key}',[M.ROOT/'tests/glm5/goodfire_middle_report.py',key,'manifest'])
        if not (M.OUT/'client-registration.json').exists():
            child(f'catalog-{key}',[M.ROOT/'tests/glm5/goodfire_middle_client.py',key])
        archive=M.BASE/f'{key}-b18-b28-native.tar.gz'
        if not archive.exists():child(f'package-{key}',[M.ROOT/'deploy/native_mlp_pack.py','build','--source',M.OUT,'--out',archive])
        child(f'package-verify-{key}',[M.ROOT/'deploy/native_mlp_pack.py','verify','--archive',archive])
        child(f'api-{key}',[M.ROOT/'tests/glm5/goodfire_middle_api.py',key])
    print('BOTH AUDITED NATIVE PACKS AND LOCAL REGISTRATIONS READY; BROWSER AND PACKAGED APP CHECKS REMAIN',flush=True)


if __name__=='__main__':main()
