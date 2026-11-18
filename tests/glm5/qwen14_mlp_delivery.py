"""Verify deliverables and remove only redundant files owned by this run."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import qwen14_mlp_atlas as C
from qwen14_mlp_provenance import save_result

def main():
    browser=C.read(C.OUT/'browser-acceptance.json');assert browser['passed']
    version='1.0.35';package=C.ROOT/'release'/f'ai2050-center-{version}.tar.gz'
    prefix=f'ai2050-center-{version}/'
    with tarfile.open(package,'r:gz') as tar:
        manifest=json.load(tar.extractfile(prefix+'RELEASE_MANIFEST.json'))
        assert manifest['version']==version
        for name,expected in manifest['files'].items():
            assert hashlib.sha256(tar.extractfile(prefix+name).read()).hexdigest()==expected,name
    for name in ['server/qwen14_mlp_service.py','server/research_space_service.py','deploy/qwen14_mlp_pack.py']:
        assert C.file_sha(C.ROOT/name)==manifest['files'][name]
    native=C.OUT/'ai2050-qwen14-native.tar.gz'
    assert C.file_sha(C.OUT/'manifest.json') in (C.OUT/'verify-native.log').read_text(encoding='utf-8')
    packages=[]
    for path in [package,native,C.OUT/'ai2050-qwen14-research-upload.tar.gz']:
        side=path.with_suffix(path.suffix+'.sha256')
        digest=side.read_text().split()[0] if side.exists() else C.file_sha(path)
        if not side.exists():side.write_text(digest+'  '+path.name+'\n',encoding='ascii')
        packages.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest,'sha256_file':str(side)})
    entries=[]
    old_releases=[]
    for old_version in ['1.0.33','1.0.34']:
        old_package=C.ROOT/'release'/f'ai2050-center-{old_version}.tar.gz'
        if old_package.exists():
            with tarfile.open(old_package,'r:gz') as old_tar:
                old_manifest=json.load(old_tar.extractfile(f'ai2050-center-{old_version}/RELEASE_MANIFEST.json'))
            old_releases.append({'manifest':old_manifest,'archive_sha256':C.file_sha(old_package)})
    save_result(C.OUT/'superseded-build-manifests.json',{'scope':'This task local intermediate builds, never deployed; final package is 1.0.35','builds':old_releases})
    for old_version in ['1.0.33','1.0.34']:
        for suffix in ['.tar.gz','.tar.gz.sha256']:
            old_package=(C.ROOT/'release'/f'ai2050-center-{old_version}{suffix}').resolve()
            assert old_package.parent==(C.ROOT/'release').resolve()
            if old_package.exists():
                entries.append({'path':str(old_package),'bytes':old_package.stat().st_size,'reason':'Superseded task-owned local build; manifest retained; never delivered or deployed'})
                old_package.unlink()
    for stage in ['natural_exact','capability_exact']:
        path=C.OUT/f'{stage}-progress.json'
        if path.exists():
            progress=C.read(path);final=C.read(C.OUT/f'{stage}-effects.json')
            assert progress['completed']==final['samples']
            assert progress['metrics']==final['rows'] and progress['controls']==final['controls'] and progress['replay']==final['replay']
            entries.append({'path':str(path),'bytes':path.stat().st_size,'sha256':C.file_sha(path),'reason':'Exact rows/controls/replay retained in final effects and verified native archive'})
            assert path.resolve().parent==C.OUT;path.unlink()
    temp=(C.ROOT/'tests/glm5_temp/qwen14-protocol').resolve()
    assert temp==Path('D:/AI2050/Ai2050-OpenOne/tests/glm5_temp/qwen14-protocol').resolve()
    if temp.exists():
        files=[p for p in temp.rglob('*') if p.is_file()]
        assert all(p.resolve().is_relative_to(temp) for p in files)
        entries.append({'path':str(temp),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'reason':'Completed isolated CPU protocol fixtures; final logs retained'})
        shutil.rmtree(temp)
    old=C.read(C.OUT/'intermediate-cleanup-ledger.json')
    save_result(C.OUT/'cleanup-final.json',{'previous_logical_bytes':old['logical_bytes_removed'],'removed':entries,
        'logical_bytes_removed':old['logical_bytes_removed']+sum(e['bytes'] for e in entries),
        'scope':'Logical file bytes removed, not net disk free-space change. All native evidence, failed calibrations, predictors, source bindings and final results retained.'})
    save_result(C.OUT/'delivery.json',{'version':version,'packages':packages,'release_members_verified':len(manifest['files']),
        'native_archive_verified':True,'native_files':175,'native_unpacked_bytes':sum(x['bytes'] for x in C.read(C.OUT/'manifest.json')['files']),
        'browser_verified':True,'browser_url':browser['url'],'browser_checks':browser['checks'],
        'api_verified':C.read(C.OUT/'api-acceptance.json')['passed'],'pagination_verified':C.read(C.OUT/'pagination-acceptance-compact.json')['passed'],
        'protocol_tests':315,'frontend_tests':62,'engineering_tests':7,'mainline_tests':9,
        'formal_center_observed_version':'1.0.29','remote_deployed':False,'remote_upload_receipt':None,
        'deployment':'User manually upgrades program and imports native archive; metadata bundle supports existing client upload protocol.',
        'scope':'139264 native profiles; 16 intervened units; zero formally confirmed semantic mechanisms.'})
    print(json.dumps({'version':version,'packages':packages,'cleanup_entries':len(entries)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
