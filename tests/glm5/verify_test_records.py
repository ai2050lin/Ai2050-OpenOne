"""Validate catalog identities and assemble the manually deployed, local-only release."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import tarfile
import zipfile

from build_test_records import ROOT, OUT, AUDIT, historical, read, sha, source, write, goodfire
from deploy.storage_policy import check_write

release=ROOT/'release/test-records-20261119'
temp=ROOT/'tests/glm5_temp/test_records_20261119'
server=next(release.glob('ai2050-center-*.tar.gz'))
history=ROOT/'deploy/test_record_history.json'


def verify():
    pilot=read(OUT/'pilot.json')
    hist=read(history)
    assert json.dumps(goodfire(),sort_keys=True)==json.dumps(pilot,sort_keys=True)
    rebuilt={r['id']:r for r in historical()['records']}
    assert all(rebuilt.get(r['id'])==r for r in hist['records'])
    records=pilot['records']+hist['records']
    ids={r['id'] for r in records}
    assert len(ids)==len(records)
    old=[link for row in pilot['records'] for link in row.get('earlier',[])]
    assert all(x['id'] in ids for x in old)
    for r in records:
        refs=[r.get('evidence'),r.get('protocol'),*(v.get('source') for v in r['runs'])]
        for s in filter(None,refs):
            assert source(ROOT/s['path'])==s, s['path']
    reg=read(ROOT/'ai2050_research_os/registry/runs.json')
    bound=[run['id'] for row in hist['records'] if row['type']=='冻结实验合同' for run in row['runs']]
    assert sorted(bound)==sorted(r['id'] for r in reg)
    with tarfile.open(server,'r:gz') as tar:
        members=tar.getmembers()
        prefix=server.name.removesuffix('.tar.gz')+'/'
        for m in members:
            assert m.isfile() and m.name.startswith(prefix) and '..' not in Path(m.name).parts
        for rel in ('VERSION','deploy/test_record_catalog.py','deploy/test_record_history.json','deploy/test_archive_service.py',
                    'shared/dist_protocol/test_archive.py','shared/dist_protocol/test_records/pilot.json','frontend/dist/index.html'):
            assert tar.extractfile(prefix+rel).read()==(ROOT/rel).read_bytes(),rel
        for p in (ROOT/'frontend/dist/assets').glob('TestRecords-*'):
            assert tar.extractfile(prefix+p.relative_to(ROOT).as_posix()).read()==p.read_bytes()
        assert not any('test_record_history' in m.name and '/deploy/' not in m.name for m in members)
        check_write(temp,800*2**20,operation='离线包验收')
        tar.extractall(temp/'package',filter='data')
    data={'verified':True,'pilot_records':len(pilot['records']),'mechanism_questions':len(pilot['groups']),
          'early_phase_links':len(old),'public_artifact_links':sum(bool(r.get('public_report')) for r in pilot['records']),
          'historical_records':len(hist['records']),'historical_types':dict(Counter(r['type'] for r in hist['records'])),
          'registered_research_os_runs':len(bound),'material_and_report_hashes_verified':True,'deterministic_rebuild':True,
          'catalogs':{p.name:source(p)['sha256'] for p in [history,OUT/'pilot.json']}}
    write(AUDIT/'coverage.json',data)
    print(json.dumps(data,ensure_ascii=False))


def deliver():
    check_write(release,700*2**20,operation='手动升级交付')
    coverage=read(AUDIT/'coverage.json')
    desktop=read(AUDIT/'desktop.json')
    assert read(AUDIT/'desktop-acceptance.json')['passed'] and desktop['passed']
    assert read(AUDIT/'history-acceptance.json')['passed'] and coverage['verified']
    zip_client=release/'AI2050-client-1.0.16-windows-x64.zip'
    folder=temp/'desktop/win-unpacked'
    candidate=zip_client.with_suffix('.partial')
    with zipfile.ZipFile(candidate,'w',zipfile.ZIP_DEFLATED,compresslevel=3,strict_timestamps=False) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,'AI2050-client/'+p.relative_to(folder).as_posix())
    with zipfile.ZipFile(candidate) as z:
        assert z.testzip() is None
        assert sha(z.read('AI2050-client/resources/app.asar'))==desktop['asar_sha256']
    candidate.replace(zip_client)
    with zipfile.ZipFile(release/'test-records-data.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [history,OUT/'pilot.json',AUDIT/'coverage.json']:
            z.write(p,p.relative_to(ROOT).as_posix())
    instructions=ROOT/'deploy/README_TEST_RECORDS.md'
    shutil.copyfile(instructions,release/instructions.name)
    bundle=release/'AI2050-test-records-server-1.0.43.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_STORED) as z:
        for p in (server,Path(str(server)+'.sha256'),release/'upgrade_center.sh',release/instructions.name):
            z.write(p,p.name)
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and sha(z.read(server.name))==source(server)['sha256']
    sums=[]
    for p in sorted(release.iterdir()):
        if p.is_file() and p.name!='SHA256SUMS.txt':
            with p.open('rb') as f: value=hashlib.file_digest(f,'sha256').hexdigest()
            sums.append(value+'  '+p.name)
    (release/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n',encoding='utf8')
    print('verified release files',len(sums))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('step',choices=['verify','deliver'])
    if parser.parse_args().step=='verify':verify()
    else:deliver()

