"""Archive methods and remove only finished task-owned scratch."""
import argparse,os,shutil,zipfile
from pathlib import Path
import b36_composition14 as B
import b36_composition14_tail as T


def archive():
    B.configure();T.admit();assert B.read(B.OUT/'audit.json')['passed']
    files=list((B.ROOT/'tests/glm5').glob('b36_composition14*'))+list((B.ROOT/'tests/glm5').glob('test_b36_composition14*'))
    for plan in ['trial-plan.json','execution-freeze.json']:
        files += [(B.ROOT/'tests/glm5'/n).resolve() for n in B.read(B.OUT/plan)['source_hashes']]
    files += [B.ROOT/n for n in ['shared/dist_protocol/mechanism_trace.py','shared/dist_protocol/grammar.py','shared/dist_protocol/natural_register_runner.py',
        'deploy/storage_policy.py','deploy/storage_policy.json','tools/provenance.py','tools/power_analysis.py']]
    pipeline=B.TEMP/'finish_pipeline.py'
    if pipeline.exists():files.append(pipeline)
    monitor=B.TEMP/'resume_monitor.py'
    if monitor.exists():files.append(monitor)
    files=sorted({p.resolve() for p in files if p.is_file()})
    destination=B.OUT/'delivery-source.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            assert p.is_relative_to(B.ROOT.resolve());z.write(p,p.relative_to(B.ROOT).as_posix())
    B.save('delivery-source.json',dict(files=[dict(path=p.relative_to(B.ROOT).as_posix(),sha256=B.file_sha(p),bytes=p.stat().st_size) for p in files],
        archive=destination.name,sha256=B.file_sha(destination),
        historical_versions=['../b36_composition14_20261113_v1/frozen-v1-source.zip','../b36_composition14_20261113_v1/v1-all-sources-before-v2.zip','../b36_composition14_20261113_v2/sources-before-scope-v3.zip'],
        note='Model weights excluded. Frozen hashes retained in plans. Task-private working notes and browser profiles excluded.',source_sha256=B.file_sha(__file__)))


def cleanup():
    B.configure();T.admit()
    assert B.read(B.OUT/'audit.json')['passed'] and B.read(B.OUT/'browser-check.json')['passed']
    assert 'All planned research computation complete' in (B.OUT/'pipeline.log').read_text(encoding='utf8')
    source=B.read(B.OUT/'delivery-source.json');assert B.file_sha(B.OUT/source['archive'])==source['sha256']
    for f in source['files']:assert B.file_sha(B.ROOT/f['path'])==f['sha256'],f['path']
    before=shutil.disk_usage(B.ROOT).free;removed=[]
    base=(B.ROOT/'tests/glm5_temp').resolve(strict=True)
    names=['b36_composition14_20261113_v1','b36_composition14_20261113_v2','b36_composition14_20261113_v3','b36-composition14-browser']
    for name in names:
        p=base/name
        if not p.exists():continue
        p=p.resolve(strict=True)
        assert p.parent==base and p.name==name and p.drive.lower()=='d:'
        contents=[p]+list(p.rglob('*'));assert all(not q.is_symlink() and not os.path.isjunction(q) for q in contents)
        size=sum(q.stat().st_size for q in contents if q.is_file())
        removed.append(dict(path=str(p),bytes=size,reason='Owned finished temporary orchestration, caches and browser/test profile; methods archived and evidence retained.'))
        shutil.rmtree(p);assert not p.exists()
    empty=B.OUT/'pilot/chunk-00000'
    if empty.exists() and not any(empty.iterdir()):
        assert empty.resolve().is_relative_to(B.OUT.resolve());empty.rmdir()
        removed.append(dict(path=str(empty),bytes=0,reason='Empty folder from failed hardlink attempt; pilot uses explicit read-only v2 origin.'))
    earlier=B.read(B.OUT.parent/'b36_composition14_20261113_v2/predecessor-cleanup-receipt.json')
    B.save('cleanup.json',dict(removed=removed,removed_file_bytes=sum(x['bytes'] for x in removed),earlier_receipt='../b36_composition14_20261113_v2/predecessor-cleanup-receipt.json',
        free_before=before,free_after=shutil.disk_usage(B.ROOT).free,
        retained='All v2 pilot fields referenced by v3; unique discovery/confirmation/format native fields and tail arrays, all failed material/scores, logs, models and historical unrelated fields.',
        retention_review='2026-12-13',limitation='Removed file bytes are not an isolated disk-free delta while other tasks run.',source_sha256=B.file_sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['archive','cleanup']);a=p.parse_args();globals()[a.stage]()
