"""Build/import immutable CPU-only MLP microscope data. Never touches accounts or model services."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
if __package__:
    from .storage_policy import StorageBlocked, check_write, pack_target, stop_with_notice
else:
    from storage_policy import StorageBlocked, check_write, pack_target, stop_with_notice

RUN='MLP-TAIL-GOODFIRE-20261108-v1'
SNAPSHOT='6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()


def validate_manifest(doc):
    if doc.get('schema')!='mlp_tail_manifest.v1' or doc.get('kind')!='real' or doc.get('run_id')!=RUN or doc.get('state')!='complete' or doc.get('model_snapshot')!=SNAPSHOT or doc.get('blocks')!=[32,33,34,35] or doc.get('width')!=9728:raise ValueError('Wrong atlas manifest identity/state')
    names=set()
    for r in doc['files']:
        name=r['path'];p=PurePosixPath(name)
        if p.is_absolute() or len(p.parts)!=1 or ':' in name or '\\' in name or name in ('.','..','manifest.json') or name in names:raise ValueError('Unsafe/duplicate atlas member')
        if type(r['bytes']) is not int or r['bytes']<0 or r['bytes']>4*2**30:raise ValueError('Invalid member size')
        if len(r['sha256'])!=64 or any(c not in '0123456789abcdef' for c in r['sha256']):raise ValueError('Invalid member digest')
        names.add(name)
    required=[r for r in doc['files'] if r.get('display_required') is True]
    essentials={'report.json','documents.json','tokenizer.json','path-cases.json','path-target-deltas.npy','QWEN-LICENSE.txt','WEIGHT-NOTICE.txt','admissibility-audit.json'}
    essentials.update(f'B{b}-{suffix}' for b in range(32,36) for suffix in ['profile.json','cues.json','readout.json','pole-values.npy','W-gate.npy','W-up.npy','W-down.npy'])
    essentials.update(f'B{b}-B{b+1}-{kind}.npy' for b in range(32,35) for kind in ['rho-discovery','rho-confirm','rho-holdout','weight-gate','weight-up'])
    if not essentials.issubset(r['path'] for r in required):raise ValueError('Missing required atlas objects')
    return required


def build(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('Do not overwrite a released dataset')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'));required=validate_manifest(manifest)
    out=pack_target(out,sum(r['bytes'] for r in required),len(required))
    tmp=check_write(out.with_suffix(out.suffix+'.tmp'),operation='MLP图谱打包暂存')
    if tmp.exists():raise ValueError('Existing MLP package staging file')
    out.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    try:
        # Low compression keeps random FP32 matrices inexpensive to package.
        with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
            tar.add(source/'manifest.json',arcname='manifest.json')
            for i,r in enumerate(required):
                check_write(tmp,operation='继续打包MLP图谱')
                p=source/r['path']
                if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Corrupt/missing source: '+r['path'])
                tar.add(p,arcname=r['path'])
                print(f'pack {i+1}/{len(required)} elapsed={time.monotonic()-started:.1f}s {r["path"]}',flush=True)
        tmp.replace(out)
        out.with_suffix(out.suffix+'.sha256').write_text(sha(out)+'  '+out.name+'\n',encoding='ascii')
    finally:
        if tmp.is_file():tmp.unlink()


def install(archive,data):
    root=Path(data).resolve()/'mlp_tail';dest=root/RUN
    stage=check_write(root/(RUN+'.staging-'+str(os.getpid())),operation='MLP图谱导入暂存')
    if stage.exists():raise ValueError('Staging directory already exists')
    stage.mkdir(parents=True);started=time.monotonic()
    try:
        with tarfile.open(archive,'r:gz') as tar:
            first=tar.next()
            if not first or first.name!='manifest.json' or not first.isfile() or first.size>2**20:raise ValueError('Manifest must be the first regular member')
            raw=tar.extractfile(first).read();man=json.loads(raw);required={r['path']:r for r in validate_manifest(man)};seen=set()
            check_write(stage,sum(r['bytes'] for r in required.values()),operation='解包MLP图谱')
            (stage/'manifest.json').write_bytes(raw)
            for member in tar:
                # TarFile iteration includes its already-read first member.
                if member is first:continue
                r=required.get(member.name)
                if not member.isfile() or r is None or member.name in seen or member.size!=r['bytes']:raise ValueError('Unexpected/duplicate/unsafe archive member: '+member.name)
                p=stage/member.name
                check_write(p,member.size,operation='继续解包MLP图谱')
                with tar.extractfile(member) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,8*2**20)
                if sha(p)!=r['sha256']:raise ValueError('Checksum mismatch: '+member.name)
                seen.add(member.name)
                print(f'import {len(seen)}/{len(required)} elapsed={time.monotonic()-started:.1f}s {member.name}',flush=True)
            if seen!=set(required):raise ValueError('Incomplete display dataset')
        if dest.exists():
            if sha(dest/'manifest.json')!=sha(stage/'manifest.json'):raise ValueError('Different immutable dataset already installed')
            for name,r in required.items():
                if not (dest/name).is_file() or sha(dest/name)!=r['sha256']:raise ValueError('Existing dataset corrupt; repair separately')
            return {'state':'already_installed','files':len(required),'path':str(dest)}
        stage.rename(dest)
        return {'state':'installed','files':len(required),'path':str(dest)}
    finally:
        # Only this PID's verified child of the explicit data root can be removed.
        if stage.exists() and stage.parent==root and stage.name==RUN+'.staging-'+str(os.getpid()):shutil.rmtree(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--out',required=True)
    i=sub.add_parser('import');i.add_argument('--archive',required=True);i.add_argument('--data-dir',required=True)
    a=p.parse_args()
    try:
        if a.action=='build':build(a.source,a.out)
        else:print(json.dumps(install(a.archive,a.data_dir),ensure_ascii=False),flush=True)
    except StorageBlocked as error:
        stop_with_notice(error)
