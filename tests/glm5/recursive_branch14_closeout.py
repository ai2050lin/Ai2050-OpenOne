"""Scoped cleanup and reviewable delivery manifests for this campaign."""
import argparse
import os
from pathlib import Path
import shutil
import time
import zipfile
import recursive_branch14 as R


def clean_tests():
    R.configure()
    p=R.OUT/'protocol-result.json';assert p.exists()
    before=shutil.disk_usage(R.ROOT).free;removed=[]
    for name in ['protocol','protocol-retry','protocol-isolated']:
        path=R.TEMP/name
        if not path.exists():continue
        target=path.resolve(strict=True);base=R.TEMP.resolve(strict=True)
        assert target.parent==base and base.is_relative_to(R.ROOT.resolve())
        assert target.drive.lower()=='d:'
        children=[target]+list(target.rglob('*'))
        assert all(not x.is_symlink() and not os.path.isjunction(x) for x in children),'Unexpected link/junction in exclusive test tree'
        size=sum(x.stat().st_size for x in children if x.is_file())
        removed.append(dict(path=str(target),bytes=size,ownership='Exclusive directory created by this campaign\'s completed/stopped software regression runs',
                            retained='protocol logs, protocol-result[-before-compat].json, protocol-isolation.json and diagnostic wrappers outside these directories'))
        shutil.rmtree(target)
        assert not target.exists()
    R.save('test-cleanup.json',dict(removed=removed,total_bytes=sum(x['bytes'] for x in removed),
        free_before=before,free_after=shutil.disk_usage(R.ROOT).free,
        limitation='Physical free-space difference can include concurrent tasks. Summed removed file bytes are the attributable cleanup size.',
        source_sha256=R.file_sha(__file__)))


def archive():
    R.configure();R.admission();assert (R.OUT/'audit.json').exists()
    files=sorted((R.ROOT/'tests/glm5').glob('recursive_branch14*'))
    files+=[R.ROOT/'tests/glm5/test_recursive_branch14.py']
    files += [R.ROOT/n for n in R.read(R.OUT/'plan.json')['source_hashes']]
    files += [R.ROOT/n for n in ['shared/dist_protocol/mechanism_trace.py','shared/dist_protocol/grammar.py',
              'deploy/storage_policy.py','deploy/storage_policy.json','tools/provenance.py']]
    files += list((R.ROOT/'shared/dist_protocol/grammars').glob('RG14-*.json'))
    files += list(R.TEMP.glob('protocol_*.py'))
    files += [R.ROOT/n for n in ['shared/dist_protocol/lang_tree.json','shared/dist_protocol/lang_catalog.json',
              'shared/dist_protocol/lang_map/structure.json','shared/dist_protocol/lang_map/lang_map.json',
              'shared/dist_protocol/lang_map/words.json','shared/dist_protocol/lang_map/manifest.json']]
    files=sorted(set(p for p in files if p.is_file()))
    archive=R.OUT/'delivery-source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(R.ROOT)))
    R.save('delivery-source.json',dict(files=[dict(path=str(p.relative_to(R.ROOT)),sha256=R.file_sha(p)) for p in files],
        archive=archive.name,archive_sha256=R.file_sha(archive),
        note='Frozen scientific sources remain separately preserved in frozen-source.zip. Current archive adds report, audit, descriptive analyses and external-label compatibility. No public deployment.'))


def clean_final():
    R.configure();assert R.read(R.OUT/'audit.json')['passed']
    assert R.read(R.OUT/'browser-check.json')['passed']
    archive=R.read(R.OUT/'delivery-source.json')
    assert R.file_sha(R.OUT/archive['archive'])==archive['archive_sha256']
    for item in archive['files']:
        assert R.file_sha(R.ROOT/item['path'])==item['sha256'],'Source changed after archive'
    removed=[];before=shutil.disk_usage(R.ROOT).free
    # A streamed recovery copy may be removed only after exact row equality.
    duplicates=[]
    for h in R.CANDIDATES:
        final=R.read(R.OUT/f'intervention-{h}.json')
        partial=R.OUT/f'intervention-{h}.jsonl'
        if partial.exists():
            with partial.open(encoding='utf8') as stream:
                for i,row in enumerate(final['rows']):
                    assert R.json.loads(next(stream))==row,(h,i)
                assert not stream.readline(),'Unexpected extra partial result rows'
            duplicates.append(partial)
        progress=R.OUT/f'intervention-{h}-progress.json'
        if progress.exists():
            info=R.read(progress)
            assert info['completed']==info['total']==final['cases'] and info['checks']==final['checks']
            duplicates.append(progress)
    # This exact directory was created exclusively for this study. Never follow links.
    target=R.TEMP.resolve(strict=True);root=R.ROOT.resolve(strict=True)
    assert target.is_relative_to(root/'tests/glm5_temp') and target.name=='recursive_branch14_20261113_v6'
    assert target.drive.lower()=='d:'
    children=[target]+list(target.rglob('*'))
    assert all(not x.is_symlink() and not os.path.isjunction(x) for x in children)
    identities={}
    for child in children:
        if not child.is_file():continue
        stat=child.stat();key=(stat.st_dev,stat.st_ino)
        item=identities.setdefault(key,dict(bytes=stat.st_size,nlink=stat.st_nlink,paths=[]))
        item['paths'].append(str(child))
    multiple_links=[item for item in identities.values() if item['nlink']>1]
    releasable=sum(item['bytes'] for item in identities.values() if item['nlink']<=len(item['paths']))
    for script in target.glob('protocol_*.py'):
        assert any(item['path']==str(script.relative_to(R.ROOT)) for item in archive['files'])
    size=sum(x.stat().st_size for x in children if x.is_file())
    removed.append(dict(path=str(target),bytes=size,unique_releasable_file_bytes=releasable,multiple_hardlinks=multiple_links,
        reason='Exclusive browser profile, rebuildable local caches, temporary regression wrappers already archived, completed task scratch'))
    shutil.rmtree(target);assert not target.exists()
    for path in duplicates:
        resolved=path.resolve(strict=True)
        assert resolved.parent==R.OUT.resolve(strict=True) and resolved.drive.lower()=='d:'
        assert not path.is_symlink() and not os.path.isjunction(path)
        removed.append(dict(path=str(resolved),bytes=path.stat().st_size,reason='Exact row/check duplicate of retained immutable intervention JSON',sha256=R.file_sha(path)))
        path.unlink();assert not path.exists()
    R.save('final-cleanup.json',dict(removed=removed,total_bytes=sum(x['bytes'] for x in removed),
        free_before=before,free_after=shutil.disk_usage(R.ROOT).free,
        retained='All native fields; immutable final intervention rows; global progress and per-stage logs; v5 development failures; frozen/current source archives; all summary and visual evidence',
        source_sha256=R.file_sha(__file__)))


def manifest():
    R.configure();R.admission()
    required=['audit.json','report.html','report-build.json','browser-check.json','feature-cards.json',
              'control-comparisons.json','full-field-maps.json','full-field-map-audit.json','native-manifest.json',
              'response-report.json','intervention-report.json','pipeline-complete.json','delivery-source.zip',
              'final-cleanup.json','memo-append.json']
    assert all((R.OUT/n).exists() for n in required)
    assert R.read(R.OUT/'audit.json')['passed'] and R.read(R.OUT/'browser-check.json')['passed']
    paths=[p for p in R.OUT.iterdir() if p.is_file() and p.suffix in ['.json','.html','.png','.npz','.zip','.log','.jsonl'] and p.name!='delivery.json']
    entries=[dict(path=p.name,bytes=p.stat().st_size,sha256=R.file_sha(p)) for p in sorted(paths)]
    R.save('delivery.json',dict(local_completed=True,formal_service_registered=False,public_deployed=False,
        report='report.html',files=entries,native_manifest='native-manifest.json',
        remaining='Four historical 4B candidates not mapped to 14B; training-time origin and full end-to-end mechanism atlas not established.',
        source_sha256=R.file_sha(__file__),created_local=time.strftime('%Y-%m-%d %H:%M')))
    if R.TEMP.exists() and not any(R.TEMP.iterdir()):R.TEMP.rmdir()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['clean-tests','clean-final','archive','manifest']);args=ap.parse_args()
    {'clean-tests':clean_tests,'clean-final':clean_final,'archive':archive,'manifest':manifest}[args.stage]()
