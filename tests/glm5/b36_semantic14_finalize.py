"""Remove only verified current scratch and verify the deliverable manifest."""
import os,shutil
import b36_semantic14 as S

def run():
    S.configure();assert S.read(S.OUT/'audit.json')['status']=='passed'
    assert (S.OUT/'memo-append-receipt.json').exists() and (S.OUT/'delivery-source.zip').exists()
    base=(S.ROOT/'tests/glm5_temp').resolve();target=S.TEMP.resolve()
    assert target.is_relative_to(base) and target==base/'b36_semantic14_20261113_v1' and target.drive.upper()=='D:'
    paths=list(target.rglob('*'))
    assert not target.is_symlink() and not target.is_junction()
    assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(target) for p in paths)
    files=[p for p in paths if p.is_file()];assert all(p.stat().st_nlink==1 for p in files)
    # Save resource receipt before deleting; all raw results and logs are outside this tree.
    total=sum(p.stat().st_size for p in files);before=shutil.disk_usage(S.ROOT).free
    shutil.rmtree(target);assert not target.exists()
    S.save('cleanup-receipt.json',dict(removed_root=str(target.relative_to(S.ROOT)),removed_files=len(files),removed_file_bytes=total,free_before_bytes=before,free_after_bytes=shutil.disk_usage(S.ROOT).free,retained='All frozen plans, actual inputs, successful and failed score chunks, logs, model identities, engineering checks, reports and source package under result directory. No old-phase output modified.',scope='Current-run scratch only; no active task dependency. No symbolic links/junctions or multi-link files. Includes rebuildable plotting/font cache; no native model fields generated.'))
    entries=[dict(path=str(p.relative_to(S.OUT)).replace('\\','/'),bytes=p.stat().st_size,sha256=S.file_sha(p)) for p in sorted(S.OUT.rglob('*')) if p.is_file() and p.name!='manifest.json']
    # Use save_json: S.save would add another provenance file after the manifest inventory.
    inherited=S.read(S.OUT/'cleanup-receipt.json')['provenance']
    S.save_json(S.OUT/'manifest.json',dict(run_id=S.RUN,model_snapshot=S.SNAPSHOT,created_at=S.now(),provenance=inherited,files=entries,total_bytes=sum(r['bytes'] for r in entries),manifest_excludes_self=True))
    for e in entries:
        p=S.OUT/e['path'];assert p.stat().st_size==e['bytes'] and S.file_sha(p)==e['sha256']
    print('Finalized',len(entries),'files;',sum(r['bytes'] for r in entries),'retained bytes;',total,'scratch bytes removed',flush=True)

if __name__=='__main__':run()
