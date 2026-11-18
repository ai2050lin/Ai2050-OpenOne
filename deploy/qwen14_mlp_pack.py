"""Build, verify and install the immutable Qwen3-14B microscope data separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
if __package__:
    from .storage_policy import StorageBlocked, check_write, pack_target, stop_with_notice
else:
    from storage_policy import StorageBlocked, check_write, pack_target, stop_with_notice

RUN='QWEN14-MLP-B28-B35-20261109-v1'
RUN_V2='QWEN14-GOODFIRE-B28-B39-20261113-v2'
SNAPSHOT='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()


def validate(d):
    version={RUN:('qwen14_mlp_manifest.v1',36),RUN_V2:('qwen14_mlp_manifest.v2',40)}.get(d.get('run_id'))
    if version is None or (d.get('schema'),d.get('state'),d.get('kind'),d.get('model_snapshot'),d.get('blocks'),d.get('width')) != (version[0],'complete','real',SNAPSHOT,list(range(28,version[1])),17408):
        raise ValueError('Wrong model, run, shape or incomplete native atlas')
    files={}
    for r in d['files']:
        n=r['path']
        if not isinstance(n,str) or Path(n).name!=n or n in ('.','..','manifest.json') or ':' in n or '\\' in n or n in files:raise ValueError('Unsafe or repeated filename')
        if type(r['bytes']) is not int or not 0<r['bytes']<=4*2**30:raise ValueError('Invalid file size')
        if len(r['sha256'])!=64 or any(c not in '0123456789abcdef' for c in r['sha256']):raise ValueError('Invalid checksum')
        if 'source_run' in r:
            if d['run_id']!=RUN_V2 or r['source_run']!=RUN:raise ValueError('Unsupported base atlas')
            origin=r.get('source_path',n)
            if not isinstance(origin,str) or Path(origin).name!=origin or origin in ('.','..') or ':' in origin or '\\' in origin:raise ValueError('Unsafe base filename')
        files[n]=r
    required={'report.json','documents.json','tokenizer.json','prefix-extrema.npz','QWEN-LICENSE.txt','WEIGHT-NOTICE.txt'}
    required.update(f'B{b}-{s}.npy' for b in d['blocks'] for s in ['G','V','A','input','W-gate','W-up','W-down'])
    required.update(f'B{b}-{s}.json' for b in d['blocks'] for s in ['profile','cues','prediction','readout'])
    if d['run_id']==RUN_V2:
        required.update(f'B{b}-{s}.npz' for b in d['blocks'] for s in ['regimes','scaled-prediction'])
        required.update({'legacy-prefix-extrema.npz','natural_exact-effects.json','capability_exact-effects.json','legacy-natural_exact-effects.json','legacy-capability_exact-effects.json','capability-inputs.json','legacy-capability-inputs.json'})
    if not required<=files.keys():raise ValueError('Missing necessary microscope files')
    return files


def build(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('Do not overwrite a released native archive')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'));files=validate(manifest)
    out=pack_target(out,sum(r['bytes'] for r in files.values() if 'source_run' not in r),len(files))
    if out.parent!=source:out.parent.mkdir(parents=True,exist_ok=True)
    temp=check_write(out.with_suffix('.partial'),operation='原生包暂存');start=time.monotonic()
    if temp.exists():raise ValueError('Retain interrupted archive for explicit recovery')
    with tarfile.open(temp,'w:gz',compresslevel=1) as tar:
        tar.add(source/'manifest.json',arcname='manifest.json')
        for i,(name,r) in enumerate(files.items()):
            check_write(temp,operation='继续打包14B原场')
            p=(source/name).resolve()
            if p.parent!=source or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Source hash mismatch: '+name)
            if 'source_run' not in r:tar.add(p,arcname=name)
            print(f'pack {i+1}/{len(files)} elapsed={time.monotonic()-start:.1f}s {name}',flush=True)
    temp.replace(out);out.with_suffix(out.suffix+'.sha256').write_text(sha(out)+'  '+out.name+'\n',encoding='ascii')
    return {'archive':str(out),'bytes':out.stat().st_size,'files':len(files)}


def stream_check(archive,write=None):
    with tarfile.open(archive,'r|gz') as tar:
        first=tar.next()
        if not first or first.name!='manifest.json' or not first.isfile() or first.size>4*2**20:raise ValueError('Manifest must be first regular member')
        raw=tar.extractfile(first).read();manifest=json.loads(raw);files=validate(manifest);seen=set()
        if write:
            check_write(write,sum(r['bytes'] for r in files.values() if 'source_run' not in r),operation='解包14B原场')
            (write/'manifest.json').write_bytes(raw)
        for name,r in files.items():
            if 'source_run' not in r:continue
            if write:
                base=write.parent/r['source_run'];origin=r.get('source_path',name)
                source=(base/origin).resolve()
                if source.parent!=base.resolve() or not source.is_file() or source.stat().st_size!=r['bytes'] or sha(source)!=r['sha256']:
                    raise ValueError('Install the verified B28-B35 base atlas first: '+origin)
                # The imported base is immutable. A hard link avoids duplicate native arrays.
                os.link(source,write/name)
            seen.add(name)
        for member in tar:
            if member is first:continue
            r=files.get(member.name)
            if not member.isfile() or r is None or member.name in seen or member.size!=r['bytes']:raise ValueError('Unexpected, repeated or unsafe archive entry')
            if write:check_write(write/member.name,member.size,operation='继续解包14B原场')
            h=hashlib.sha256();src=tar.extractfile(member);dst=(write/member.name).open('xb') if write else None
            try:
                for chunk in iter(lambda:src.read(8*2**20),b''):
                    h.update(chunk)
                    if dst:dst.write(chunk)
            finally:
                if dst:dst.close()
                src.close()
            if h.hexdigest()!=r['sha256']:raise ValueError('Archive checksum mismatch: '+member.name)
            seen.add(member.name)
            if len(seen)%8==0:print(f'verified {len(seen)}/{len(files)}',flush=True)
        if seen!=set(files):raise ValueError('Truncated or incomplete archive')
        return manifest,hashlib.sha256(raw).hexdigest()


def install(archive,data_dir):
    root=Path(data_dir).resolve()/'native_atlases'
    with tarfile.open(archive,'r:gz') as tar:
        first=tar.next()
        if not first or first.name!='manifest.json' or not first.isfile() or first.size>4*2**20:raise ValueError('Manifest must be first regular member')
        header=json.loads(tar.extractfile(first).read());validate(header)
    run=header['run_id'];dest=root/run
    if dest.exists():
        manifest,digest=stream_check(archive)
        if sha(dest/'manifest.json')!=digest:raise ValueError('Different immutable run already installed')
        for n,r in validate(manifest).items():
            if sha(dest/n)!=r['sha256']:raise ValueError('Existing data damaged; explicit repair required')
        return {'state':'already_installed','run_id':run,'files':len(manifest['files'])}
    stage=check_write(root/(run+'.staging-'+str(os.getpid())),operation='14B导入暂存')
    if stage.exists():raise ValueError('Do not overwrite an interrupted staging directory')
    stage.mkdir(parents=True)
    try:
        manifest,_=stream_check(archive,stage)
        stage.rename(dest)
        return {'state':'installed','run_id':run,'files':len(manifest['files'])}
    finally:
        # Only this newly created, checked staging child is disposable.
        if stage.exists() and stage.resolve().parent==root.resolve() and stage.name==run+'.staging-'+str(os.getpid()):shutil.rmtree(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--out',required=True)
    v=sub.add_parser('verify');v.add_argument('--archive',required=True)
    i=sub.add_parser('import');i.add_argument('--archive',required=True);i.add_argument('--data-dir',required=True)
    a=p.parse_args()
    try:
        if a.action=='build':print(build(a.source,a.out))
        elif a.action=='verify':print({'verified_manifest_sha256':stream_check(a.archive)[1]})
        else:print(install(a.archive,a.data_dir))
    except StorageBlocked as error:
        stop_with_notice(error)
