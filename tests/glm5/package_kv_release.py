"""Assemble the November 17 handoff from the current build and immutable research."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.research_ingest_bundle import verified_objects
from deploy.storage_policy import check_write,staging_file
from package_receipt_research import build as receipt_pack


OUT=ROOT/'release/deployment-20261117'
WORK=ROOT/'tests/glm5_temp/deployment_20261117'
RESULT=ROOT/'tests/glm5/result'
SOURCE=RESULT/'kv_receipt_20261117_v1'
VERSION='1.0.40'
WEB=f'ai2050-web-{VERSION}.zip'
CENTER=f'ai2050-center-{VERSION}.tar.gz'
UPLOADS=('phase-2952-research-upload.tar.gz','phase-2955-research-upload.tar.gz')
EVIDENCE='ai2050-kv-review-data-20261117.zip'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export_zip(name,members):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
        for path,source in sorted(members.items()):
            archive.writestr(path,source.read_bytes())
    with buffer.getbuffer() as payload:
        target=check_write(OUT/name,len(payload),operation='写入交付压缩包')
        with staging_file(target) as partial:
            with partial.open('wb') as stream:stream.write(payload)


def input_files():
    names=('plan.json','frozen.json','audit.json','report.md','run.log',
           'discovery-behavior.json','words-behavior.json','frames-behavior.json',
           'discovery-material.json','words-material.json','frames-material.json',
           'words-predictions.npz','frames-predictions.npz','words-edits.npz','frames-edits.npz',
           'words-prediction-binding.json','frames-prediction-binding.json','smoke.json')
    original=json.loads((SOURCE/'manifest.json').read_bytes())
    files={'phase-2955/manifest.json':SOURCE/'manifest.json'}
    for k in names:
        source=SOURCE/k
        assert sha(source.read_bytes())==original['files'][k],k
        files['phase-2955/'+k]=source
    for pattern in ('kv_receipt_*.py','package_receipt_research.py','package_kv_release.py'):
        for p in (ROOT/'tests/glm5').glob(pattern):files[p.relative_to(ROOT).as_posix()]=p
    for folder,keys in [
        ('kv_length_client_20261117_v1',['client-readback.json','client-browser-verification.json']),
        ('kv_chain_20261117_v1',['independent-audit.json','report.md','plan.json','frame-repair-plan.json','sources.zip']),
    ]:
        for name in keys:files[folder+'/'+name]=RESULT/folder/name
    files['scripts/build_center_release.py']=ROOT/'scripts/build_center_release.py'
    return files


def assemble():
    target=OUT/UPLOADS[0]
    source=RESULT/'kv_length_client_20261117_v1'/target.name
    shutil.copyfile(source,check_write(target,source.stat().st_size))
    receipt_pack(OUT/UPLOADS[1])
    script=ROOT/'deploy/research_ingest_bundle.py'
    shutil.copyfile(script,check_write(OUT/script.name,script.stat().st_size))
    dist=WORK/'dist'
    export_zip(WEB,{f'frontend/dist/{p.relative_to(dist).as_posix()}':p for p in dist.rglob('*') if p.is_file()})
    export_zip(EVIDENCE,input_files())


def verify():
    web={}
    with zipfile.ZipFile(OUT/WEB) as z:
        assert z.testzip() is None
        web={n:sha(z.read(n)) for n in z.namelist()}
        assert len(web)==len(z.namelist()) and 'frontend/dist/index.html' in web
    with tarfile.open(OUT/CENTER,'r:gz') as tar:
        prefix=f'ai2050-center-{VERSION}/'
        manifest=json.load(tar.extractfile(prefix+'RELEASE_MANIFEST.json'))
        assert manifest['version']==VERSION
        assert set(tar.getnames())=={prefix+k for k in (*manifest['files'],'RELEASE_MANIFEST.json')}
        for path,checksum in manifest['files'].items():
            member=tar.getmember(prefix+path)
            assert member.isfile() and '..' not in Path(path).parts and sha(tar.extractfile(member).read())==checksum,path
        assert manifest['files']['VERSION']==sha((VERSION+'\n').encode())
        assert web=={n:h for n,h in manifest['files'].items() if n.startswith('frontend/dist/')}
        assert sha((OUT/'upgrade_center.sh').read_bytes())==manifest['files']['deploy/upgrade_center.sh']
        assert sha((OUT/'research_ingest_bundle.py').read_bytes())==manifest['files']['deploy/research_ingest_bundle.py']
    with zipfile.ZipFile(OUT/EVIDENCE) as z:
        files=input_files()
        assert set(z.namelist())==set(files)
        assert all(sha(z.read(k))==sha(path.read_bytes()) for k,path in files.items())
    studies=[]
    for p in UPLOADS:
        with tarfile.open(OUT/p,'r:gz') as tar:
            manifest_=json.load(tar.extractfile('MANIFEST.json'))
            assert len(list(verified_objects(tar,manifest_)))==len(manifest_['objects'])
            studies.extend(manifest_['studies'])
    declared=(OUT/(CENTER+'.sha256')).read_text().split()[0]
    assert declared==sha((OUT/CENTER).read_bytes())
    print(json.dumps(dict(center_version=VERSION,center_files=len(manifest['files']),web_files=len(web),
                          source_review_files=len(files),studies=studies),ensure_ascii=False))


if __name__=='__main__':
    if sys.argv[1:]==['assemble']:assemble()
    verify()
