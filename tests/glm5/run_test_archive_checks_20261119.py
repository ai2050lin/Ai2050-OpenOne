"""No-model engineering checks with durable logs and minute progress output."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import checked_dir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'tests/glm5/result/test_archive_20261119')
    parser.add_argument('--temp',type=Path,default=ROOT/'tests/glm5_temp/test_archive_20261119')
    args=parser.parse_args()
    out=checked_dir(args.out,20*2**20)
    temp=checked_dir(args.temp,400*2**20)
    research=checked_dir(temp/'isolated-research')
    env={**os.environ,'TEMP':str(temp),'TMP':str(temp),'TMPDIR':str(temp),
         'AI2050_DIST_ADMIN_TOKEN':'isolated-engineering-token','PYTHONDONTWRITEBYTECODE':'1',
         'AI2050_DIST_DIR':str(temp/'unittest-center'),'AI2050_NATIVE_DIR':str(temp/'native'),
         'AI2050_RESEARCH_DIR':str(research),
         'AI2050_DIST_NEWS_OFFLINE':'1','AI2050_SNAPSHOT_DISABLE':'1'}
    groups=[('archive',['-m','unittest','discover','-s','tests/glm5','-p','test_test_archive.py','-v']),
            ('storage',['-m','unittest','discover','-s','tests/glm5','-p','test_storage_policy.py','-v']),
            ('dist-protocol',['-m','unittest','discover','-s','tests/dist_protocol','-v'])]
    results=[]
    for index,(name,arguments) in enumerate(groups,1):
        start=time.monotonic();path=out/(name+'.log')
        print(f'[{index}/{len(groups)}] start {name}; log={path}',flush=True)
        with path.open('wb') as log:
            process=subprocess.Popen([sys.executable,'-B','-X','utf8',*arguments],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while True:
                try:
                    code=process.wait(timeout=45);break
                except subprocess.TimeoutExpired:
                    print(f'[{index}/{len(groups)}] {name} running {time.monotonic()-start:.0f}s; log={path.stat().st_size} bytes',flush=True)
        result={'group':name,'exit_code':code,'elapsed_seconds':round(time.monotonic()-start,2),'log':str(path)}
        results.append(result);print(json.dumps(result),flush=True)
        (out/'checks.json').write_text(json.dumps({'groups':results,'complete':index==len(groups),'models_run':0},indent=2),encoding='utf-8')
        if code:
            print(path.read_text(encoding='utf-8',errors='replace')[-9000:],flush=True)
            return code
    return 0


if __name__=='__main__':raise SystemExit(main())
