"""Exercise B35 data ingestion, native order, and browser endpoints on the extracted release."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from mlp_factor_steer_release_check import verify_sha


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/b35_recursive_v2'
RELEASE = ROOT / 'release'
VERSION = '1.0.27'
ART = '/api/research-library/artifacts/'


def check(port, hold):
    data = RELEASE / 'ai2050-research-qwen3-4b-b35.tar.gz'
    package = RELEASE / f'ai2050-center-{VERSION}.tar.gz'
    proof = {'package_sha256':verify_sha(package),'data_sha256':verify_sha(data)}
    with tempfile.TemporaryDirectory(dir=ROOT/'tests/glm5_temp',prefix='b35-release-') as directory:
        temp = Path(directory)
        with tarfile.open(package) as tar:
            tar.extractall(temp,filter='data')
        root = temp / f'ai2050-center-{VERSION}'
        center = temp/'center'
        snapshots = center/'snapshots'
        snapshots.mkdir(parents=True)
        shutil.copyfile(ROOT/'distributed_data/snapshots/lang_atlas.json',snapshots/'lang_atlas.json')
        token = secrets.token_hex(24)
        env = {**os.environ,'AI2050_DIST_DIR':str(center),'AI2050_NATIVE_DIR':str(temp/'native'),
               'AI2050_RESEARCH_DIR':str(temp/'research'),'AI2050_SNAPSHOT_DISABLE':'1',
               'AI2050_DIST_ADMIN_TOKEN':token,'AI2050_RESEARCH_ADMIN_TOKEN':token,
               'AI2050_LOCAL_TOOLS':'0','PYTHONUTF8':'1'}
        for name in ('AI2050_DIST_ALLOW_SYNTHETIC','AI2050_FRONTEND_DIST','PYTHONPATH'):
            env.pop(name,None)
        seed = ('from pathlib import Path; import os; from deploy.center_app import create_app; '
                'create_app(); import datapack; datapack.build(Path(os.environ["AI2050_DIST_DIR"]), "b35-release-smoke")')
        subprocess.run([sys.executable,'-c',seed],env=env,cwd=root,check=True,timeout=70)
        base = f'http://127.0.0.1:{port}'

        def get(path):
            with urlopen(base+path,timeout=7) as response:
                return json.load(response)

        with (OUT/'release-server.log').open('w',encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable,'deploy/center_app.py','--host','127.0.0.1','--port',str(port)],
                 env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                for _ in range(110):
                    if proc.poll() is not None:
                        raise RuntimeError('B35 release service stopped')
                    try:
                        health = get('/api/health')
                        break
                    except (URLError,TimeoutError):
                        time.sleep(.25)
                else:
                    raise RuntimeError('B35 release service timeout')
                assert health['version']==VERSION,health
                uploader = [sys.executable,'deploy/research_ingest_bundle.py',str(data),'--center',base]
                subprocess.run(uploader,env=env,cwd=root,check=True,timeout=150)
                reg = json.loads((OUT/'registration-final.json').read_bytes())
                index = get('/api/traces/research-space')
                study = next(s for s in index['studies'] if s['run_id']==reg['run_id'])
                assert study['report_sha256']==reg['report_sha256']
                assert study['stages']['b35_features']['index_sha256']==reg['feature_index_sha256']
                overview = get(ART+reg['feature_index_sha256'])
                assert overview['unit_count']==9728 and overview['model_snapshot']==reg['model_snapshot']
                assert all(len(arr)==9728 for arr in overview['plots'].values())
                served = []
                for sha in overview['chunks']:
                    shard = get(ART+sha)
                    assert shard['start']==len(served) and shard['model_snapshot']==reg['model_snapshot']
                    served.extend(r['unit'] for r in shard['units'])
                assert served==list(range(9728))
                report = get(ART+reg['report_sha256'])
                example = report['error_cases']['lockbox'][0]
                cases = report['case_indexes']['lockbox']
                case = get(ART+cases['chunks'][example//cases['chunk_size']])['cases'][example%cases['chunk_size']]
                assert len(case['input_ids'])==1024 and not all(x['correct'] for x in case['outcome']['free'])
                proof.update(endpoint=base,health=health,all_units=len(served),objects=reg['objects'],
                             run_id=reg['run_id'],example=case['id'],report_sha256=study['report_sha256'],
                             timestamp=datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds'))
                (OUT/'release-api-proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
                print(proof,flush=True)
                if hold:
                    time.sleep(hold)
            finally:
                proc.terminate()
                proc.wait(timeout=15)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=55135)
    parser.add_argument('--hold',type=int,default=0)
    args=parser.parse_args()
    check(args.port,args.hold)
