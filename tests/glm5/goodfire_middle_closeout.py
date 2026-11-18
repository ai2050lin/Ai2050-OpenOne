"""Verify packages, collect resource evidence and retain the final implementation."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.request
import zipfile
import goodfire_middle as M
from deploy.storage_policy import check_write
C=M.C


def get(url):
    with urllib.request.urlopen(url,timeout=120) as stream:return json.load(stream)


def main():
    reports={key:C.read(M.BASE/key/'report.json') for key in M.SPECS}
    for key in M.SPECS:
        M.configure(key);M.require_admission()
        assert C.read(M.BASE/key/'api-acceptance.json')['passed']
        assert C.file_sha(M.BASE/key/'manifest.json') in (M.BASE/f'package-verify-{key}.log').read_text(encoding='utf-8')
    for name in ['browser-acceptance.json','desktop-acceptance.json']:assert C.read(M.BASE/name)['passed'],name
    protocol=(M.BASE/'protocol-tests.log').read_text(encoding='utf-8')
    assert 'Ran 315 tests' in protocol and 'FAILED (failures=1)' in protocol and 'SOURCE-BINDING-20261110.source' in protocol
    assert 'Ran 9 tests' in (M.BASE/'native-service-tests.log').read_text(encoding='utf-8')
    assert 'Ran 7 tests' in (M.BASE/'engineering-tests.log').read_text(encoding='utf-8')
    assert re.search(r'Tests\s+63 passed', (M.BASE/'frontend-tests-final.log').read_text(encoding='utf-8'))
    version=(M.ROOT/'VERSION').read_text().strip();health=get('http://127.0.0.1:5035/api/health')
    assert health['version']==version
    center=M.ROOT/'release'/f'ai2050-center-{version}.tar.gz';members=[]
    with tarfile.open(center,'r:gz') as archive:
        release_manifest=None
        for member in archive:
            if not member.isfile():continue
            parts=Path(member.name).parts;rel=Path(*parts[1:])
            if rel.as_posix()=='RELEASE_MANIFEST.json':
                release_manifest=json.load(archive.extractfile(member));continue
            p=M.ROOT/rel
            assert p.is_file(),str(rel)
            raw=archive.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==C.file_sha(p),str(rel)
            members.append(str(rel))
        assert release_manifest['version']==version
        assert set(release_manifest['files'])=={n.replace('\\','/') for n in members}
        for name,digest in release_manifest['files'].items():assert C.file_sha(M.ROOT/name)==digest,name
    bundles=[]
    for key in M.SPECS:
        p=M.BASE/f'{key}-b18-b28-research-upload.tar.gz'
        with tarfile.open(p,'r:gz') as z:
            manifest=json.load(z.extractfile('MANIFEST.json'));ids={r['sha256'] for r in manifest['objects']};seen=set()
            for member in z:
                if member.name.startswith('objects/') and member.isfile():
                    digest=Path(member.name).stem;assert digest in ids
                    assert hashlib.sha256(z.extractfile(member).read()).hexdigest()==digest;seen.add(digest)
            assert seen==ids
        bundles.append(dict(model=key,objects=len(ids),run_id=reports[key]['run_id']))
    packages=[];start=time.monotonic()
    targets=[center,M.BASE/'desktop/AI2050-Setup-1.0.15-x64.exe']
    targets += [M.BASE/f'{key}-b18-b28-{kind}.tar.gz' for key in M.SPECS for kind in ['native','research-upload']]
    for i,p in enumerate(targets):
        digest=C.file_sha(p);side=p.with_suffix(p.suffix+'.sha256')
        if side.exists():assert side.read_text().split()[0]==digest
        else:check_write(side).write_text(digest+'  '+p.name+'\n',encoding='ascii')
        packages.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest))
        print(f'delivery package {i+1}/{len(targets)} elapsed={time.monotonic()-start:.1f}s {p.name}',flush=True)
    sources=['server/native_mlp_service.py','server/qwen14_mlp_service.py','server/server.py','deploy/center_app.py',
        'deploy/native_mlp_pack.py','deploy/qwen14_mlp_pack.py','deploy/research_library.py','deploy/README_NATIVE_MIDDLE.md',
        'scripts/build_center_release.py','scripts/build_desktop.cjs','frontend/package.json','frontend/package-lock.json',
        'frontend/src/components/app/rdc_fusion/api.js','frontend/src/components/v8/NativeMlpAtlas.jsx',
        'frontend/src/components/v8/ModelStructure.jsx','frontend/src/components/v8/ResearchInterventionReport.jsx','VERSION',
        'tests/glm5/test_native_mlp_service.py','tests/glm5/test_goodfire_middle.py']
    sources += [str(p.relative_to(M.ROOT)).replace('\\','/') for p in (M.ROOT/'tests/glm5').glob('goodfire_middle*') if p.is_file()]
    sources=sorted(set(sources));hashes={n:C.file_sha(M.ROOT/n) for n in sources}
    with zipfile.ZipFile(check_write(M.BASE/'integration-sources.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sources:archive.write(M.ROOT/name,name)
    resources={};plots={}
    for key in M.SPECS:
        import numpy as np
        from PIL import Image
        with np.load(M.BASE/key/'heatmap-native-arrays.npz') as z:
            assert z['A_std'].shape==z['double_negative_abs_mass_fraction'].shape==(11,M.SPECS[key]['width'])
            values={'log1p_A_std':np.log1p(z['A_std']), 'double_negative':z['double_negative_abs_mass_fraction']}
            plots[key]={'grouping':16,'ordering':'native, no sorting',
                'native_color_ranges':{n:[float(x.min()),float(x.max())] for n,x in values.items()},
                'overview_color_ranges':{n:[float(x.reshape(11,-1,16).mean(2).min()),float(x.reshape(11,-1,16).mean(2).max())] for n,x in values.items()},
                'scope':'Overview auto-scales after consecutive-16 averaging. Full-width images use native ranges. Color scales are not shared across models.'}
        for filename in ['A-std-native-pixels.png','double-negative-native-pixels.png']:
            with Image.open(M.BASE/key/filename) as im:assert im.size==(M.SPECS[key]['width'],264)
        rows=[]
        for p in M.BASE.glob(f'*-{key}.log'):
            for line in p.read_text(encoding='utf-8',errors='replace').splitlines():
                if not line.startswith('{'):continue
                try:row=json.loads(line)
                except ValueError:continue
                if row.get('model')==key and 'ram_available_bytes' in row:rows.append(row)
        maxima=['cuda_peak','dedicated_gpu_bytes','shared_gpu_bytes','commit_used_bytes']
        resources[key]={**{k:max((r[k] for r in rows if r.get(k) is not None),default=None) for k in maxima},
            'minimum_machine_available_RAM':min((r['ram_available_bytes'] for r in rows),default=None),'logged_rows':len(rows),
            'scope':'CUDA and DXGI metrics are per process; available RAM and commit are machine totals, including other processes.'}
    old=C.read(M.ROOT/'tests/glm5/result/qwen14_client_v2/lint-baseline.json')[0]['messages']
    new=C.read(M.BASE/'lint-final.json')[0]['messages']
    def signature(m):
        return m['ruleId'],m['severity'],re.sub(r'line \d+','line N',m['message'].split('\n\n')[0])
    assert sorted(map(signature,old))==sorted(map(signature,new))
    M.configure('qwen4');M.OUT=M.BASE;M.RUN='NATIVE-MIDDLE-DELIVERY-20261113-v2'
    body=dict(timestamp=datetime.now().astimezone().isoformat(),center_version=version,desktop_version='1.0.15',
        units=298496,semantic_confirmed=0,remote_deployed_by_this_task=False,installation_wizard_run=False,
        models={k:dict(run_id=r['run_id'],snapshot=r['model_snapshot'],units=r['n_units'],provenance=r['provenance']) for k,r in reports.items()},
        packages=packages,center_members=len(members),research_bundles=bundles,health=health,resources=resources,plot_audit=plots,
        source_hashes=hashes,source_archive_sha256=C.file_sha(M.BASE/'integration-sources.zip'),
        checks=dict(api_models=2,api_layers=22,endpoint_units=44,intervened_units=12,browser=C.read(M.BASE/'browser-acceptance.json'),
            desktop=C.read(M.BASE/'desktop-acceptance.json'),native_service_tests=9,frontend_tests=63,
            protocol_passed=314,protocol_total=315,preexisting_lint_findings_unchanged=True),
        known_failure='Existing SOURCE-BINDING-20261110.source language mapping absent; frozen grammar not changed.',
        provenance_scope='Engineering delivery envelope uses 4B helper; per-model scientific provenance and full snapshots are listed separately.')
    M.write(M.BASE/'delivery.json',body);print('Delivery identities verified',flush=True)


if __name__=='__main__':main()
