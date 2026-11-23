"""Role- and identity-checked retirement of other completed test intermediates.

Never imports torch, executes a runner, changes original manifests, or recursively
deletes directories. Mixed NPZs retain explicitly identified result/input members.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import checked_dir, check_write

OUT = ROOT/'tests/glm5/result/other_test_storage_cleanup_20261118_v1'
NATIVE = Path('D:/ai2050_native')
FAMILIES = {'ENT-XF', 'GR-Q', 'GR-NEG', 'KN-ISA', 'KN-ATTR', 'GR-ROLE'}
GIB = 2**30


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    from datetime import datetime, timezone, timedelta
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-6))).isoformat(timespec='seconds')


def log(stage, **more):
    line = json.dumps({'time': stamp(), 'stage': stage, **more}, ensure_ascii=False)
    print(line, flush=True)
    with (OUT/'cleanup-progress.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')
        stream.flush()


def save(name, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    path = OUT/name
    check_write(path, len(data), operation='其他测试清理证据')
    with path.open('xb') as stream:
        stream.write(data)


def safe(path):
    p = Path(os.path.abspath(path))
    if not (p.is_relative_to(ROOT/'tests') or p.is_relative_to(ROOT/'research') or p.is_relative_to(NATIVE)):
        raise RuntimeError(f'Outside reviewed roots: {p}')
    if p.is_relative_to(NATIVE):
        first = p.relative_to(NATIVE).parts[0]
        if first not in FAMILIES|{'_cross_model','_mode_atlas','_transcoder','_mainline','_fusion','reusable'}:
            raise RuntimeError(f'Protected native root: {p}')
        boundary = NATIVE
    else:
        boundary = ROOT
    probe = p
    while True:
        st = probe.lstat()
        if probe.is_symlink() or getattr(st, 'st_file_attributes', 0)&0x400:
            raise RuntimeError(f'Link/reparse point rejected: {probe}')
        if probe==boundary:
            break
        probe=probe.parent
    if p.resolve(strict=True)!=p:
        raise RuntimeError(f'Resolved path differs: {p}')
    return p


def metadata(path):
    path = safe(path)
    st = path.stat()  # Windows DirEntry.stat may give zero inode/nlink; never use inventory identities.
    if not st.st_ino or not st.st_nlink:
        raise RuntimeError(f'Unavailable real file identity: {path}')
    return dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                device=st.st_dev, inode=st.st_ino, nlink=st.st_nlink)


def npy_header(stream):
    import numpy as np
    v=np.lib.format.read_magic(stream)
    if v not in {(1,0),(2,0)}:
        raise RuntimeError(f'Unexpected NPY version: {v}')
    shape, order, dtype=(np.lib.format.read_array_header_1_0 if v==(1,0) else np.lib.format.read_array_header_2_0)(stream)
    return dict(shape=list(shape), fortran=bool(order), dtype=str(dtype), first_64_bytes=stream.read(64).hex())


def raw_member(kind, member):
    """Explicit producer-specific member roles, not size or extension heuristics."""
    name=member.removesuffix('.npy')
    if kind=='cross_style': return name in {'Hg','Hs'}
    if kind=='cross_entity_mode': return name.endswith(('_codes','_B','_bare'))
    if kind=='mode': return name.endswith(('_H','_B','_bare'))
    if kind=='cross_final': return name.endswith('_final')
    if kind=='paradigm': return name in {'G','U','A','field','early','target'}
    if kind=='lookup': return bool(re.fullmatch(r'(?:h\d\d|z[f]?2[012]|[gua]f?2[012])', name))
    if kind=='family': return name in {'X','Z','M','G','V','A'}
    raise ValueError(kind)


def exclusive_probe(path):
    """Refuse files currently open by another Windows reader/writer."""
    if os.name!='nt': return
    dll=ctypes.WinDLL('kernel32',use_last_error=True)
    dll.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p]
    dll.CreateFileW.restype=ctypes.c_void_p
    dll.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=dll.CreateFileW(str(path),0x80000000,0,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    dll.CloseHandle(handle)


def collect():
    candidates={}
    proof_paths=set()
    notes=[]
    def add(path, kind, group, expected=None, proof=None, canonical=None):
        p=safe(path)
        if p.is_file():
            candidates[str(p)]={'kind':kind,'group':group,'expected_sha256':expected,
                                'proof':str(proof) if proof else None,
                                'canonical':str(canonical) if canonical else None}
            if proof:proof_paths.add(Path(proof))
            if canonical:proof_paths.add(Path(canonical))
    # Only fields whose exact corpus fingerprint matches the retained frozen bundle.
    for family in sorted(FAMILIES):
        for manifest in sorted((NATIVE/family).rglob('manifest.json')):
            obj=json.loads(manifest.read_text(encoding='utf-8'))
            if obj.get('schema')!='native_field.v1':continue
            bundle=ROOT/'distributed_data/bundles'/str(obj.get('tm_id'))/('v'+str(obj.get('version')))
            corpus=bundle/'corpus.json'
            if not corpus.exists():
                notes.append({'path':str(manifest),'decision':'keep','reason':'Frozen corpus absent'})
                continue
            material=json.loads(corpus.read_text(encoding='utf-8'))
            fp=hashlib.sha256(json.dumps(material,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
            if fp!=obj.get('corpus_fingerprint') or not (bundle/'runner.py').exists():
                notes.append({'path':str(manifest),'decision':'keep','reason':'Frozen corpus fingerprint/runner mismatch'})
                continue
            proof_paths.update(p for p in bundle.iterdir() if p.is_file())
            proof_paths.add(manifest)
            for entry in obj.get('files',[]):
                path=manifest.parent/entry['path']
                if re.fullmatch(r'L\d+\.npz',entry['path']) and path.exists():
                    add(path,'family','family:'+family,entry.get('sha256'),manifest)
            for sub in ['components','mlp']:
                folder=manifest.parent/sub
                if not folder.exists():continue
                # Component collectors use manifest.json with independently hashed G/V or Z/M arrays.
                cms=sorted(folder.glob('*.json'))
                references={}
                for cm in cms:
                    proof_paths.add(cm)
                    co=json.loads(cm.read_text(encoding='utf-8'))
                    files=co.get('files',[])
                    if isinstance(files,dict):files=[dict(path=k,**v) for k,v in files.items()]
                    for ent in files:
                        if isinstance(ent,dict) and ent.get('path'):references[ent['path']]=ent.get('sha256')
                for path in sorted(folder.glob('B*.npz')):
                    if re.fullmatch(r'B\d+\.npz',path.name):add(path,'family','family:'+family,references.get(path.name),manifest)
    # Transcoder training activations are expendable; the learned dictionary and token/window inputs are results/inputs.
    for m in (NATIVE/'_transcoder').rglob('manifest.json'):
        obj=json.loads(m.read_text(encoding='utf-8'))
        if obj.get('schema')!='transcoder_collect.v1':continue
        for name in ['tok_train.npy','tok_eval.npy','eval_windows.npy']:
            if not (m.parent/name).exists():raise RuntimeError(f'Missing preserved transcoder input: {name}')
            proof_paths.add(m.parent/name)
        proof_paths.add(m)
        for name in ['x_train.npy','x_eval.npy']:
            if (m.parent/name).exists():add(m.parent/name,'npy','transcoder-collection',obj['files_sha256'][name],m)
    # Original input/row metadata and manifests remain, only known dense natural-forward arrays retire.
    hm=ROOT/'research/mainline/H2/manifest.json'
    obj=json.loads(hm.read_text(encoding='utf-8'))
    hd=Path(obj['local_dir'])
    proof_paths.update([hm,ROOT/'research/mainline/H2/MA-H2.prereg.json',hd/'meta.json'])
    for name in ['hs.npy','attn.npy','kv.npy','mlp.npy']:
        if (hd/name).exists():add(hd/name,'npy','mainline-H2',obj['files'][name]['sha256'],hm)
    for m in (NATIVE/'reusable/common_bench_b1_v3').glob('*/manifest.json'):
        obj=json.loads(m.read_text(encoding='utf-8'))
        proof_paths.add(m)
        material=ROOT/'tests/glm5/result/common_bench_b1_v3'/f'{m.parent.name}-materials.json'
        if not material.exists():raise RuntimeError('Missing common-benchmark questions')
        proof_paths.add(material)
        for name in ['hidden.npy','postattn.npy']:
            if (m.parent/name).exists():add(m.parent/name,'npy','common-benchmark-cache',obj['files'][name]['sha256'],m)
    for name in ['_cross_model','_mode_atlas']:
        for path in (NATIVE/name).rglob('*.npz'):
            if name=='_cross_model' and path.parent.name=='MF-v1' and path.name=='fields.npz':kind='cross_style'
            elif name=='_cross_model' and path.parent.name=='XM-v1' and path.name=='data.npz':kind='cross_entity_mode'
            elif name=='_cross_model' and path.parent.name=='PS-v1' and path.name=='final_states.npz':kind='cross_final'
            elif name=='_mode_atlas' and path.parent.name in {'MV-v1','MV-v2'} and path.name=='data.npz':kind='mode'
            else:continue
            manifest=path.parent/'manifest.json'
            if not manifest.exists():
                notes.append({'path':str(path),'decision':'keep','reason':'Required source manifest absent; no raw retirement for this file'})
                continue
            add(path,kind,name,proof=manifest)
    for path in (NATIVE/'_fusion').rglob('states*.npy'):
        if path.name in {'states.npy','states_en.npy','states_zh.npy'}:
            manifest=path.parent/'manifest.json'
            if manifest.exists():add(path,'npy','fusion-activation-cache',proof=manifest)
            else:notes.append({'path':str(path),'decision':'keep','reason':'Required source manifest absent'})
    # Two completed November campaigns: preserve output logits/masses/writes and all JSON; retire only states.
    for run,kind,pattern in [('paradigm_routes_20261117_v1','paradigm',r'(?:pilot(?:-conditional)?|discovery|confirm)-\d+\.npz'),
                            ('conditional_lookup_native_v1','lookup',r'(?:pilot-native|calibrate-\d+|discovery-\d+|confirm-\d+)\.npz')]:
        folder=ROOT/'tests/glm5/result'/run
        for path in folder.glob('*.npz'):
            if re.fullmatch(pattern,path.name):add(path,kind,'workspace:'+run)
        proof_paths.update(folder.glob('*.json'))
    # Sandbox object copies are retired only against an extant byte-identical canonical object.
    inventory=OUT/'inventory-files.jsonl'
    opener=lambda:inventory.open(encoding='utf-8')
    if not inventory.exists():
        inventory=OUT/'inventory-files.jsonl.gz'
        opener=lambda:gzip.open(inventory,'rt',encoding='utf-8')
    canonical_root=ROOT/'distributed_data/research_library/objects'
    with opener() as stream:
        for line in stream:
            row=json.loads(line);path=Path(row['path'])
            if not path.is_relative_to(ROOT/'tests'):continue
            if not ('research_library' in path.parts and 'objects' in path.parts):continue
            if not (path.is_relative_to(ROOT/'tests/glm5_temp') or path.is_relative_to(ROOT/'tests/glm5/result/mlp_tail_acceptance')):continue
            if not re.fullmatch(r'[0-9a-f]{64}\.json',path.name):continue
            canonical=canonical_root/path.name[:2]/path.name
            if canonical.is_file() and canonical.stat().st_size==path.stat().st_size:
                add(path,'duplicate-object','sandbox-identical-object',canonical=canonical)
    # Preserve all existing files in affected raw roots, all relevant research reports, and source/protocol scripts.
    protected=set(proof_paths)
    roots={Path(p).parent for p in candidates}
    for parent in roots:
        protected.update(p for p in parent.iterdir() if p.is_file() and str(p) not in candidates)
    for folder in ['research/mainline','research/transcoder','research/mode_atlas','research/cross_model',
                   'research/mode_field','research/path_scan','research/fusion','research/entity_atlas']:
        p=ROOT/folder
        if p.exists():protected.update(x for x in p.rglob('*') if x.is_file() and x.suffix in {'.json','.npz','.npy','.sha256','.md'})
    for folder in [ROOT/'shared/dist_protocol',ROOT/'scripts',ROOT/'tests/glm5']:
        iterator=folder.rglob('*') if folder.name=='dist_protocol' else folder.iterdir()
        protected.update(p for p in iterator if p.is_file() and p.suffix in {'.py','.json','.tsv','.md'} and '__pycache__' not in p.parts)
    protected={p.resolve() for p in protected if p.exists() and p.is_file() and str(p.resolve()) not in candidates}
    return candidates,protected,notes


def plan():
    checked_dir(OUT,512*2**20,operation='其他测试清理计划')
    if (OUT/'plan.json').exists():raise RuntimeError('Immutable plan exists')
    start=time.monotonic();log('plan_start')
    candidates,protected,notes=collect()
    records=[];skipped=[];last=start
    total_bytes=sum(Path(p).stat().st_size for p in candidates)
    done_bytes=0
    for i,(name,detail) in enumerate(sorted(candidates.items()),1):
        p=Path(name)
        exclusive_probe(p)
        r={**metadata(p),**detail,'scope':'native' if p.is_relative_to(NATIVE) else 'workspace'}
        if r['nlink']!=1:
            skipped.append({'path':str(p),'reason':'Multiple hardlinks; outside readers/aliases not retired','nlink':r['nlink']})
            continue
        r['sha256']=sha(p)
        if detail['expected_sha256'] and detail['expected_sha256']!=r['sha256']:
            raise RuntimeError(f'Historical raw digest mismatch, no deletion permitted: {p}')
        if detail['kind']=='duplicate-object':
            if sha(detail['canonical'])!=r['sha256']:raise RuntimeError(f'Canonical duplicate mismatch: {p}')
            r['canonical_sha256']=r['sha256']
        elif p.suffix=='.npz':
            members={}
            with zipfile.ZipFile(p) as z:
                for member in z.namelist():
                    if not member.endswith('.npy'):raise RuntimeError('Unexpected NPZ member')
                    with z.open(member) as stream:header=npy_header(stream)
                    members[member]={**header,'raw':raw_member(detail['kind'],member),
                                     'bytes':z.getinfo(member).file_size,'crc32':z.getinfo(member).CRC}
            if not any(v['raw'] for v in members.values()):
                skipped.append({'path':str(p),'reason':'No eligible raw members'})
                continue
            r['members']=members
            r['retained_members']=[k for k,v in members.items() if not v['raw']]
            r['review_path']=str(OUT/'retained_npz'/(r['sha256'][:20]+'.review.npz')) if r['retained_members'] else None
        elif p.suffix=='.npy':
            with p.open('rb') as stream:r['header']=npy_header(stream)
        records.append(r);done_bytes+=r['bytes']
        if time.monotonic()-last>=25 or i==len(candidates):
            elapsed=time.monotonic()-start
            log('hash_candidates',done=i,total=len(candidates),bytes_done=done_bytes,bytes_total=total_bytes,
                elapsed_seconds=round(elapsed,1),eta_seconds=round(elapsed*(total_bytes-done_bytes)/max(done_bytes,1),1))
            last=time.monotonic()
    kept=[];last=time.monotonic()
    for i,p in enumerate(sorted(protected),1):
        # Protected files may be outside the deletion roots, including frozen production bundles.
        st=p.stat();kept.append({'path':str(p),'bytes':st.st_size,'sha256':sha(p)})
        if time.monotonic()-last>=25:
            log('hash_protected',done=i,total=len(protected),elapsed_seconds=round(time.monotonic()-start,1))
            last=time.monotonic()
    value={'schema':'other_test_raw_retirement.v1','created_at':stamp(),
           'authorization':'User explicitly requested continuing other test intermediate cleanup; retain scripts, questions, results and necessary review; no model runs',
           'script_sha256':sha(Path(__file__)),'root':str(ROOT),'native_root':str(NATIVE),
           'records':records,'protected':kept,'kept_decisions':notes+skipped,
           'candidate_files':len(records),'candidate_logical_bytes':sum(r['bytes'] for r in records),
           'protected_files':len(kept),'protected_bytes':sum(r['bytes'] for r in kept),
           'inventory_identity_note':'Inventory uses directory-entry metadata only. Every deletion record refreshes nonzero Windows file identity and hardlink count using Path.stat().',
           'model_runs':0}
    save('plan.json',value)
    log('plan_ready',files=len(records),GiB=value['candidate_logical_bytes']/GIB,
        protected_files=len(kept),protected_GiB=value['protected_bytes']/GIB,elapsed_seconds=round(time.monotonic()-start,1))


def execute(scope):
    plan_path=OUT/'plan.json';p=json.loads(plan_path.read_text(encoding='utf-8'))
    if p['script_sha256']!=sha(Path(__file__)):raise RuntimeError('Frozen cleanup script changed')
    if (OUT/f'receipt-{scope}.json').exists():raise RuntimeError('Scope already executed')
    rows=[r for r in p['records'] if r['scope']==scope]
    for r in rows:
        now=metadata(r['path'])
        if any(now[k]!=r[k] for k in ['bytes','mtime_ns','device','inode','nlink']):raise RuntimeError(f'Changed candidate: {r["path"]}')
        exclusive_probe(Path(r['path']))
        if r['canonical'] and sha(r['canonical'])!=r['sha256']:raise RuntimeError('Canonical copy changed')
    before=shutil.disk_usage(ROOT).free;start=time.monotonic();deleted=[];last=start
    log('delete_start',scope=scope,done=0,total=len(rows),free_bytes=before)
    for i,r in enumerate(rows,1):
        path=safe(Path(r['path']));exclusive_probe(path)
        current=metadata(path)
        if any(current[k]!=r[k] for k in ['bytes','mtime_ns','inode','nlink']):raise RuntimeError('Concurrent file mutation')
        if r.get('retained_members'):
            review=Path(r['review_path']);checked_dir(review.parent,64*2**20,operation='保留NPZ题目与结果成员')
            if review.exists():raise RuntimeError('Review destination already exists')
            member_sha={}
            with zipfile.ZipFile(path) as original, zipfile.ZipFile(review,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as compact:
                for name in r['retained_members']:
                    with original.open(name) as src, compact.open(name,'w',force_zip64=True) as dst:
                        h=hashlib.sha256()
                        while data:=src.read(4*2**20):dst.write(data);h.update(data)
                    member_sha[name]=h.hexdigest()
            with zipfile.ZipFile(review) as compact:
                assert compact.namelist()==r['retained_members']
                for name,digest in member_sha.items():
                    with compact.open(name) as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                    if actual!=digest:raise RuntimeError('Review member differs; original not deleted')
            r['retained_member_sha256']=member_sha
            r['review_sha256']=sha(review);r['review_bytes']=review.stat().st_size
        path.unlink()
        deleted.append(r)
        if time.monotonic()-last>=20 or i%200==0 or i==len(rows):
            elapsed=time.monotonic()-start
            log('delete_progress',scope=scope,done=i,total=len(rows),group=r['group'],elapsed_seconds=round(elapsed,1),
                eta_seconds=round(elapsed*(len(rows)-i)/i,1),free_bytes=shutil.disk_usage(ROOT).free)
            last=time.monotonic()
    # Individual raw-directory markers; existing historical manifests are left byte-identical.
    parents=collections.defaultdict(list)
    for r in deleted:parents[str(Path(r['path']).parent)].append(r)
    for parent,rr in parents.items():
        marker=Path(parent)/'RAW_STORAGE_STATUS_20261118_CONTINUATION.json'
        check_write(marker,16384,operation='原场退役状态登记')
        obj={'schema':'raw_storage_status.v1','created_at':stamp(),'status':'user_authorized_raw_retirement',
             'files':len(rr),'logical_bytes':sum(r['bytes'] for r in rr),'cleanup_plan':str(plan_path),'plan_sha256':sha(plan_path),
             'review_archives':[r['review_path'] for r in rr if r.get('review_path')],
             'reason':'Keep scripts, questions, results and necessary review; recapturable raw fields retired. No model execution.',
             'recovery':'No full raw backup. Use retained frozen materials/source/model identity to recapture into a new directory. Duplicate sandbox objects can be copied from canonical paths in plan.',
             'historical_manifests':'Unchanged historical execution evidence; no longer a claim that all recorded raw paths exist.'}
        with marker.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    after=shutil.disk_usage(ROOT).free
    receipt={'schema':'other_test_retirement_receipt.v1','scope':scope,'completed_at':stamp(),'plan_sha256':sha(plan_path),
             'deleted':deleted,'deleted_files':len(deleted),'logical_bytes':sum(r['bytes'] for r in deleted),
             'review_bytes':sum(r.get('review_bytes',0) for r in deleted),'free_before_bytes':before,'free_after_bytes':after,
             'observed_free_increase_bytes':after-before,'model_runs':0,'directories_recursively_deleted':0}
    save(f'receipt-{scope}.json',receipt)
    log('delete_complete',scope=scope,deleted=len(deleted),reclaimed_GiB=(after-before)/GIB,free_GiB=after/GIB)


def verify():
    plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'));errors=[];last=time.monotonic();start=last
    receipts=[]
    for scope in ['workspace','native']:
        path=OUT/f'receipt-{scope}.json'
        if path.exists():receipts.append(json.loads(path.read_text(encoding='utf-8')))
    completed={r['scope'] for r in receipts}
    for receipt in receipts:
        if receipt['plan_sha256']!=sha(OUT/'plan.json'):raise RuntimeError('Plan/receipt digest mismatch')
        for r in receipt['deleted']:
            if Path(r['path']).exists():errors.append({'path':r['path'],'reason':'Original still exists'})
            if r.get('review_path'):
                path=Path(r['review_path'])
                if not path.exists() or sha(path)!=r['review_sha256']:errors.append({'path':str(path),'reason':'Review checksum mismatch'})
                else:
                    with zipfile.ZipFile(path) as z:
                        for n,d in r['retained_member_sha256'].items():
                            with z.open(n) as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                            if actual!=d:errors.append({'path':str(path),'member':n,'reason':'Member mismatch'})
    for i,r in enumerate(plan['protected'],1):
        path=Path(r['path'])
        if not path.exists() or sha(path)!=r['sha256']:errors.append({'path':r['path'],'reason':'Protected checksum changed'})
        if time.monotonic()-last>=25:
            log('verify_protected',done=i,total=len(plan['protected']),elapsed_seconds=round(time.monotonic()-start,1),
                eta_seconds=round((time.monotonic()-start)*(len(plan['protected'])-i)/i,1));last=time.monotonic()
    result={'schema':'other_test_retirement_verification.v1','completed_at':stamp(),'passed':not errors,'completed_scopes':sorted(completed),
            'deleted_files':sum(r['deleted_files'] for r in receipts),'protected_files_rehashed':len(plan['protected']),
            'review_archives_checked':sum(bool(r.get('review_path')) for receipt in receipts for r in receipt['deleted']),
            'observed_free_increase_bytes':sum(r['observed_free_increase_bytes'] for r in receipts),
            'free_bytes':shutil.disk_usage(ROOT).free,'model_runs':0,'errors':errors,
            'limitations':['No model execution or recapture; reanalysis of retired fields requires recapture',
                           'Protected clients, dictionaries, formal results, model weights and active service credentials are preserved',
                           'Whole-drive free-space deltas can include other activity; per-file manifests are primary']}
    save('verification.json',result);log('verification_complete',**{k:result[k] for k in ['passed','deleted_files','protected_files_rehashed','review_archives_checked','free_bytes']})
    if errors:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','execute-workspace','execute-native','verify'])
    args=parser.parse_args()
    if args.command=='plan':plan()
    elif args.command=='verify':verify()
    else:execute(args.command.removeprefix('execute-'))
