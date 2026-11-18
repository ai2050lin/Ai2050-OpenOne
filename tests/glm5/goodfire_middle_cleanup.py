"""Retain numerical-failure evidence, then clean only this completed campaign's scratch."""
import gc
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import goodfire_middle as M
from deploy.storage_policy import check_write
C=M.C


def main():
    for key in M.SPECS:
        root=M.BASE/key
        assert C.read(root/'audit.json')['passed'] and C.read(root/'api-acceptance.json')['passed']
        assert (M.BASE/f'{key}-b18-b28-native.tar.gz').is_file()
    assert C.read(M.BASE/'desktop-acceptance.json')['passed']
    assert C.read(M.BASE/'browser-acceptance.json')['passed']
    start=time.monotonic();removed=[];retained=[]
    def delete_file(p,root,reason):
        p=p.resolve();root=root.resolve();assert p.parent==root and p.is_file()
        row=dict(path=str(p),bytes=p.stat().st_size,sha256=C.file_sha(p),reason=reason)
        p.unlink();removed.append(row)
    v1=M.ROOT/'tests/glm5/result/goodfire_middle_20261113_v1';old=v1/'qwen4'
    proof=check_write(old/'proof-first4',100*2**20);proof.mkdir(exist_ok=True)
    # The admission failure was measured on the first four discovery documents.
    # Keep their exact BF16 bits plus every original input, diagnosis and source log.
    for p in sorted(old.glob('*.npy')):
        if p.name=='tokens.npy':continue
        if '-W-' in p.name:
            matching=M.BASE/'qwen4'/p.name
            if not matching.is_file() or C.file_sha(p)!=C.file_sha(matching):
                retained.append(dict(path=str(p),reason='No verified byte-identical v2 weight archive'));continue
            retained.append(dict(original=str(p),replacement=str(matching),sha256=C.file_sha(matching)))
        else:
            x=np.load(p,mmap_mode='r')
            if len(x)!=1024:
                retained.append(dict(path=str(p),reason='Unexpected sample axis; not automatically removed'));del x;continue
            subset=proof/p.name;np.save(subset,np.array(x[:4],copy=True));del x;gc.collect()
            retained.append(dict(original=str(p),subset=str(subset),samples=[0,4],sha256=C.file_sha(subset)))
        delete_file(p,old,'Superseded numerical v1; first-four failure proof or identical v2 weight copy retained')
        print('v1 proof/cleanup',p.name,flush=True)
    for key in M.SPECS:
        root=M.BASE/key
        for p in root.glob('*-progress.json'):
            final=root/p.name.replace('-progress.json','-effects.json')
            if not final.exists():continue
            a=C.read(p);b=C.read(final)
            assert a['completed']==b['samples'] and a['metrics']==b['rows'] and a['replay']==b['replay'] and a['controls']==b['controls']
            del a,b;gc.collect();delete_file(p,root,'Exact duplicate of completed intervention rows, controls and replay')
    diagnostic=M.ROOT/'tests/glm5_temp/goodfire_middle_20261113_v1'
    if diagnostic.is_dir():
        with zipfile.ZipFile(v1/'diagnostic-sources.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in diagnostic.rglob('*'):
                if p.is_file():z.write(p,str(p.relative_to(diagnostic)))
    temp_root=(M.ROOT/'tests/glm5_temp').resolve()
    names=['goodfire-middle-browser','goodfire-middle-electron','goodfire-middle-build','goodfire-middle-protocol',
           'native-service-tests','goodfire-lint','goodfire_middle_20261113_v1']
    for name in names:
        p=temp_root/name
        if not p.exists():continue
        assert p.resolve().parent==temp_root and not p.is_symlink()
        size=sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
        shutil.rmtree(p);removed.append(dict(path=str(p),bytes=size,reason='Completed, campaign-exclusive engineering scratch'))
    # Keep the active loopback server's small empty research directory.
    active=M.TEMP.resolve()
    for p in active.iterdir():
        if p.name=='server':continue
        assert p.resolve().parent==active and not p.is_symlink()
        size=sum(x.stat().st_size for x in p.rglob('*') if x.is_file()) if p.is_dir() else p.stat().st_size
        if p.is_dir():shutil.rmtree(p)
        else:p.unlink()
        removed.append(dict(path=str(p),bytes=size,reason='Finished collection/analysis cache; server directory retained'))
    version=(M.ROOT/'VERSION').read_text().strip()
    for obsolete in ['1.0.37','1.0.38']:
        if version==obsolete:continue
        for suffix in ['.tar.gz','.tar.gz.sha256']:
            p=M.ROOT/'release'/('ai2050-center-'+obsolete+suffix)
            if p.exists():delete_file(p,M.ROOT/'release','Unpublished intermediate package generated in this campaign; final package retained')
    checkpoint=temp_root/'goodfire_middle_checkpoint.txt'
    if checkpoint.is_file():delete_file(checkpoint,temp_root,'Completed campaign coordination scratch; scientific and engineering evidence retained separately')
    M.configure('qwen4');M.OUT=M.BASE;M.RUN='NATIVE-MIDDLE-DELIVERY-20261113-v2'
    M.write(M.BASE/'cleanup.json',dict(removed=removed,retained=retained,freed_bytes=sum(r['bytes'] for r in removed),
        seconds=time.monotonic()-start,free_bytes=shutil.disk_usage(M.BASE).free,
        scope='This campaign only; original historical studies and both completed v2 native archives retained. Review native reuse on 2026-12-13.'))
    print('Cleanup complete',sum(r['bytes'] for r in removed),flush=True)


if __name__=='__main__':main()
