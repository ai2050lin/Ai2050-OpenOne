"""Source archiving, owned duplicate cleanup and final delivery manifest."""
import argparse
import os
from pathlib import Path
import shutil
import time
import zipfile
import branch_reuse14 as B
import branch_reuse14_execution as E


def archive():
    B.configure();E.admit();assert B.read(B.OUT/'audit.json')['passed']
    q=B.read(B.OUT/'qualified-execution.json')
    files=list((B.ROOT/'tests/glm5').glob('branch_reuse14*'))+[B.ROOT/'tests/glm5/test_branch_reuse14_algorithms.py']
    files += [B.ROOT/'tests/glm5'/n for n in B.read(B.OUT/'execution-freeze.json')['sources']]
    files += [B.ROOT/n for n in ['shared/dist_protocol/mechanism_trace.py','shared/dist_protocol/grammar.py',
                               'deploy/storage_policy.py','deploy/storage_policy.json','tools/provenance.py','tests/glm5/recursive_branch14_materials.py']]
    files += list(B.TEMP.glob('protocol_*.py'))
    files=sorted(set(p for p in files if p.is_file()))
    destination=B.OUT/'delivery-source.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.relative_to(B.ROOT).as_posix())
    B.save('delivery-source.json',dict(files=[dict(path=str(p.relative_to(B.ROOT)),sha256=B.file_sha(p)) for p in files],
        archive=destination.name,archive_sha256=B.file_sha(destination),
        note='Original frozen sources separately retained in frozen-confirmation-source.zip and qualified-source-v1.zip. Current archive includes admission/path recovery, audit and report. No model weights or private work-state scratch included.'))


def cleanup():
    B.configure();assert B.read(B.OUT/'audit.json')['passed'] and B.read(B.OUT/'browser-check.json')['passed']
    source=B.read(B.OUT/'delivery-source.json');assert B.file_sha(B.OUT/source['archive'])==source['archive_sha256']
    for item in source['files']:assert B.file_sha(B.ROOT/item['path'])==item['sha256']
    removed=[];dedup=[];before=shutil.disk_usage(B.ROOT).free
    oldbase=(B.ROOT/'tests/glm5/result/branch_reuse14_20261113_v2/pilot').resolve(strict=True)
    newbase=(B.OUT/'pilot').resolve(strict=True)
    assert oldbase.is_relative_to(B.ROOT.resolve()) and newbase.is_relative_to(B.ROOT.resolve())
    assert oldbase.drive.lower()==newbase.drive.lower()=='d:'
    # Exact duplicate calibrated inputs, native arrays and compressed bytes only.
    for chunk in ['chunk-00000','chunk-00064','chunk-00128']:
      older=B.read(oldbase/chunk/'chunk.json');newer=B.read(newbase/chunk/'chunk.json')
      assert older['row_ids']==newer['row_ids']
      for item in older['files']:
        old=(oldbase/chunk/item['path']).resolve(strict=True);new=(newbase/chunk/item['path']).resolve(strict=True)
        assert old.is_relative_to(oldbase) and new.is_relative_to(newbase)
        assert not old.is_symlink() and not new.is_symlink()
        sha=B.file_sha(old);assert sha==B.file_sha(new)==item['sha256']
        if os.path.samefile(old,new):continue
        stat=old.stat();assert stat.st_nlink==1,'Unexpected preexisting shared old artifact'
        temporary=old.with_name(old.name+'.verified-link');assert not temporary.exists()
        os.link(new,temporary);assert os.path.samefile(temporary,new)
        os.replace(temporary,old)
        assert os.path.samefile(old,new) and B.file_sha(old)==sha
        dedup.append(dict(old_path=str(old),canonical_path=str(new),bytes=stat.st_size,sha256=sha,
            meaning='Identical repeated pilot bytes; one physical copy via NTFS hardlinks. Not an independent sample or replication. Both historical paths remain readable.'))
    duplicate=B.OUT/'intervention-B19/uncommitted-first-attempt.npz';canonical=duplicate.with_name('chunk-00000.npz')
    assert duplicate.resolve().is_relative_to(B.OUT.resolve()) and B.file_sha(duplicate)==B.file_sha(canonical)
    removed.append(dict(path=str(duplicate),bytes=duplicate.stat().st_size,sha256=B.file_sha(duplicate),reason='Byte-identical uncommitted attempt; failure log, hash and exact recovery audit retained'))
    duplicate.unlink()
    # Only this run's own scratch plus small older calibration cache trees.
    for target in [B.TEMP,B.ROOT/'tests/glm5_temp/branch_reuse14_20261113_v1',B.ROOT/'tests/glm5_temp/branch_reuse14_20261113_v2']:
      if not target.exists():continue
      target=target.resolve(strict=True);base=(B.ROOT/'tests/glm5_temp').resolve(strict=True)
      assert target.parent==base and target.name in ['branch_reuse14_20261113_v1','branch_reuse14_20261113_v2','branch_reuse14_20261113_v3'] and target.drive.lower()=='d:'
      children=[target]+list(target.rglob('*'));assert all(not p.is_symlink() and not os.path.isjunction(p) for p in children)
      size=sum(p.stat().st_size for p in children if p.is_file())
      removed.append(dict(path=str(target),bytes=size,reason='Task-owned rebuildable caches, completed browser/test scratch, temporary working notes; required test wrappers archived and logs kept'))
      shutil.rmtree(target);assert not target.exists()
    B.save('cleanup.json',dict(removed=removed,deduplicated=dedup,
        removed_file_bytes=sum(r['bytes'] for r in removed),deduplicated_file_bytes=sum(r['bytes'] for r in dedup),
        free_before=before,free_after=shutil.disk_usage(B.ROOT).free,
        limitation='File-byte totals are attributable removed/repeated content, not an exact isolated change in whole-drive free space during concurrent tasks.',
        retained='All unique native arrays, failed calibration material/scores, source archives, audit, final arm outputs and logs; prior recursive_branch14 61GiB field untouched.',source_sha256=B.file_sha(__file__)))


def manifest():
    B.configure();E.admit()
    required=['report.html','audit.json','browser-check.json','cleanup.json','memo-append.json','conclusion.json','delivery-source.zip',
              'qualified-pipeline-state.json','native-manifest.json','intervention-report.json']
    assert all((B.OUT/n).exists() for n in required)
    assert B.read(B.OUT/'qualified-pipeline-state.json')['status']=='completed'
    assert B.read(B.OUT/'audit.json')['passed'] and B.read(B.OUT/'browser-check.json')['passed']
    files=[p for p in B.OUT.iterdir() if p.is_file() and p.name!='delivery.json']
    B.save('delivery.json',dict(local_work_completed=True,original_three_branch_mechanism_confirmed=False,
        report='report.html',files=[dict(path=p.name,bytes=p.stat().st_size,sha256=B.file_sha(p)) for p in sorted(files)],
        native_manifest='native-manifest.json',intervention_archives='intervention-B19/result.json and intervention-B36/result.json bind each compressed chunk',
        service_registered=False,public_deployed=False,source_sha256=B.file_sha(__file__),created_local=time.strftime('%Y-%m-%d %H:%M')))
    if B.TEMP.exists() and not any(B.TEMP.iterdir()):B.TEMP.rmdir()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['archive','cleanup','manifest']);args=p.parse_args();globals()[args.stage]()
