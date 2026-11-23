"""Verify release hashes and run the extracted center against isolated engineering data."""
import hashlib
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import checked_dir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'tests/glm5/result/test_archive_20261119')
    parser.add_argument('--work',type=Path,default=ROOT/'tests/glm5_temp/test_archive_20261119/package-smoke')
    parser.add_argument('--package',type=Path,help='待验收的中心发布包路径')
    parser.add_argument('--extract-only',action='store_true',help='核对文件并解包，随后用单独的中心进程验收')
    args=parser.parse_args()
    out=checked_dir(args.out,10*2**20)
    work=checked_dir(args.work,200*2**20)
    version=(ROOT/'VERSION').read_text().strip();name='ai2050-center-'+version
    package=(args.package or ROOT/'release'/f'{name}.tar.gz').resolve()
    expected=package.with_name(package.name+'.sha256').read_text().split()[0]
    assert hashlib.sha256(package.read_bytes()).hexdigest()==expected
    with tarfile.open(package) as archive:
        members=archive.getmembers()
        for member in members:
            p=PurePosixPath(member.name)
            assert p.parts[0]==name and '..' not in p.parts and not p.is_absolute()
            assert member.isfile() or member.isdir()
        archive.extractall(work,filter='data')
    app=work/name
    manifest=json.loads((app/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    for relative,sha in manifest['files'].items():
        assert hashlib.sha256((app/relative).read_bytes()).hexdigest()==sha,relative
    required=['deploy/test_archive_service.py','deploy/test_archive_client.py','shared/dist_protocol/test_archive.py',
              'deploy/README_TEST_ARCHIVE.md','AGENTS.md','tests/glm5/run_routine_test.py',
              'shared/dist_protocol/lang_tree.json','shared/dist_protocol/lang_map/manifest.json']
    assert all(p in manifest['files'] for p in required)
    assert not any(p.startswith('research/glm5/docs/') or 'test_archive_20261119' in p for p in manifest['files'])
    verification={'version':version,'package':str(package),'sha256':expected,'bytes':package.stat().st_size,
                  'manifest_files_verified':len(manifest['files']),'required_files':required,
                  'models_run':0,'online_server_modified':False}
    if args.extract_only:
        (out/'package-verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(verification,ensure_ascii=False))
        return 0
    code=r'''
import hashlib,json,os,sys
from pathlib import Path
from fastapi.testclient import TestClient
import center_app
from deploy import test_archive_service
root=Path.cwd().resolve()
assert Path(test_archive_service.__file__).resolve().is_relative_to(root)
client=TestClient(center_app.create_app())
headers={'X-Admin-Token':os.environ['AI2050_DIST_ADMIN_TOKEN']}
assert client.get('/api/health').json()['version']==sys.argv[1]
assert client.get('/api/test-archive/capabilities').json()['schema']=='research_test_archive_api.v1'
assert client.get('/api/test-archive/runs').status_code in (401,403)
assert client.get('/api/test-archive/runs',headers=headers).json()['total']==0
assert client.get('/api/models/local').status_code==403
assert client.get('/').status_code==200
mapping=client.get('/api/lang/map').json()
assert mapping['tree_compat']['KN.SOURCE']=='SEM.SOURCE'
assert mapping['tree_compat']['KN.ENTITY_REF']=='KN.ENTITY_REF'
assert mapping['tree_compat']['EXT.SOURCE_ORDER']=='VAR.SOURCE_ORDER'
assert mapping['tree_compat']['EXT.SOURCE_WORDING']=='VAR.PARAPHRASE'
raw=b'private package engineering fixture'
sha=hashlib.sha256(raw).hexdigest()
upload=client.post('/api/test-archive/uploads',headers=headers,json={'sha256':sha,'size':len(raw)}).json()
uid=upload['upload_id']
assert client.put('/api/test-archive/uploads/'+uid+'?offset=0',headers=headers,content=raw).status_code==200
assert client.post('/api/test-archive/uploads/'+uid+'/finish',headers=headers).json()['verified']
assert client.get('/api/test-archive/objects/'+sha).status_code in (401,403)
assert client.get('/api/test-archive/objects/'+sha,headers=headers).content==raw
print(json.dumps({'version':sys.argv[1],'package_imports_only':True,'health':True,'capabilities':True,'private_auth':True,'private_upload_readback':True,'public_gpu_guard':True,'frontend_served':True,'source_binding_compat':True}))
'''
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(app/'deploy'),str(app)]),'PYTHONDONTWRITEBYTECODE':'1',
         'AI2050_DIST_ADMIN_TOKEN':'isolated-package-fixture','AI2050_DIST_DIR':str(work/'center-data'),
         'AI2050_NATIVE_DIR':str(work/'native'),'AI2050_SNAPSHOT_DISABLE':'1','AI2050_DIST_NEWS_OFFLINE':'1',
         'AI2050_FRONTEND_DIST':str(app/'frontend/dist'),'TEMP':str(work),'TMP':str(work)}
    env.pop('AI2050_LOCAL_TOOLS',None)
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,version],cwd=app,env=env,capture_output=True,text=True,encoding='utf-8',timeout=90)
    (out/'package-smoke.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:
        print(result.stdout+result.stderr);return result.returncode
    verification['extracted_app_smoke']=json.loads(result.stdout.splitlines()[-1])
    (out/'package-verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
