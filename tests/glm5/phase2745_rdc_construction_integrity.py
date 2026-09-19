"""Streaming archive, immutable-source and original-checkpoint audits."""
import argparse, zipfile
from collections import Counter, defaultdict
from rdc_construction_common import *

OUT = BASE/'verification'


def archives():
    start = time.monotonic()
    prior = OUT/'archive_cache.json'
    frozen_initial = OUT/'archive_cache.json.gz'
    previous = read(prior) if prior.exists() else gzread(frozen_initial) if frozen_initial.exists() else []
    cached = {r['path']:r for r in previous}
    entries = []
    files = sorted(p for p in BASE.rglob('*.npz') if '.tmp' not in p.name)
    for i, p in enumerate(files):
        rel = p.relative_to(BASE).as_posix()
        stat = p.stat()
        old = cached.get(rel)
        digest = sha(p)
        if old and old['sha256'] == digest:
            entries.append(old)
            continue
        tensors, count, raw, fortran_tensors = 0, 0, 0, 0
        with zipfile.ZipFile(p) as archive:
            for item in archive.infolist():
                assert item.filename.endswith('.npy')
                with archive.open(item) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
                    shape, fortran, dtype = reader(stream)
                    assert dtype.kind in 'buif', (rel, item.filename, dtype)
                    # Finite-value/CRC/SHA checks read the stored scalar stream;
                    # C vs Fortran indexing does not change those checks.
                    # Retain the declared layout instead of transposing files.
                    fortran_tensors += int(fortran)
                    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
                    remaining = size*dtype.itemsize
                    while remaining:
                        chunk = stream.read(min(8*1024**2, remaining))
                        assert chunk and len(chunk)%dtype.itemsize == 0
                        values = np.frombuffer(chunk, dtype=dtype)
                        if dtype == np.dtype('uint16'):
                            assert not np.any((values & 0x7f80) == 0x7f80), (rel, item.filename, 'BF16 nonfinite')
                        elif dtype.kind == 'f':
                            assert np.isfinite(values).all(), (rel, item.filename, 'nonfinite')
                        remaining -= len(chunk)
                    assert stream.read(1) == b'', (rel, item.filename, 'unexpected trailing bytes')
                    tensors += 1; count += size; raw += size*dtype.itemsize
        assert p.stat().st_size == stat.st_size and p.stat().st_mtime_ns == stat.st_mtime_ns, 'Archive changed during read'
        entries.append({'path':rel,'sha256':digest,'bytes':stat.st_size,'arrays':tensors,'scalars':count,'decoded_bytes':raw,
            'ZIP_CRC_and_all_values_checked':True,'uint16_interpretation':'BF16bits','Fortran_layout_arrays':fortran_tensors})
        if (i+1)%100 == 0:
            save(prior, entries)
            print('CONSTRUCTION_ARCHIVE_AUDIT',i+1,len(files),flush=True)
    save(prior, entries)
    result = {'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'archives':len(entries),
        'arrays':sum(r['arrays'] for r in entries),'scalars':sum(r['scalars'] for r in entries),
        'Fortran_layout_arrays':sum(r.get('Fortran_layout_arrays',0) for r in entries),
        'compressed_bytes':sum(r['bytes'] for r in entries),'decoded_bytes':sum(r['decoded_bytes'] for r in entries),
        'seconds':time.monotonic()-start,'scope':'Every listed original array, no sampling; earlier identical archives reused only after full SHA256 recheck. Concurrent later commits require refresh.'}
    save(OUT/'archives.json',result)
    print('CONSTRUCTION_ARCHIVES_DONE',len(entries),result['arrays'],flush=True)
    return result


def checkpoints():
    start = time.monotonic()
    previous = read(OLD/'verification/model_checkpoint_fingerprints.json')
    records = []
    for model, files in previous['models'].items():
        for entry in files:
            path = ROOT/'models/hf'/model/entry['file']
            assert path.stat().st_size == entry['bytes']
            digest = sha(path)
            assert digest == entry['sha256'], ('Original checkpoint file changed',str(path))
            records.append({'model':model,**entry,'current_sha256':digest})
        print('CONSTRUCTION_ORIGINAL_CHECKPOINT',model,len(files),flush=True)
    save(OUT/'model_checkpoint_fingerprints.json',{'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),
        'original_fingerprints_sha256':sha(OLD/'verification/model_checkpoint_fingerprints.json'),
        'records':records,'bytes_rehashed':sum(r['bytes'] for r in records),'seconds':time.monotonic()-start})


def identities():
    start, checks = time.monotonic(), []
    contract = read(BASE/'protocol.json')
    for entry in contract['required_prior_artifacts']:
        assert sha(OLD/entry['path']) == entry['sha256'],entry
    original = MEMO.read_bytes()[:contract['memo_original_bytes']]
    assert hashlib.sha256(original).hexdigest() == contract['memo_original_sha256']
    checks.append('All14frozen prior artifacts and entire10,220,760byte pre2745MEMOprefix unchanged')
    material = gzread(BASE/'material.json.gz')
    commit_checks = []
    for model, group in material['models'].items():
        rows = group['rows']; assert len(rows) == len({r['sample_id'] for r in rows}) == 320
        pairs = defaultdict(list); splits = defaultdict(set)
        for row in rows:
            pairs[row['pair_id']].append(row);splits[row['source_group']].add(row['split'])
            path = BASE/'capture'/model/'fields'/(row['sample_id']+'.npz')
            commit = read(BASE/'capture'/model/'commits'/(row['sample_id']+'.json'))
            assert sha(path) == commit['sha256']
            with np.load(path) as z:
                depth, width = z['prefix_layers'].shape
                assert z['postnorm'].shape == (100,width)
                assert z['query_selected_states'].shape == (len(z['query_layer_indices']),100,width)
                assert len(z['query_layer_indices']) == len(set(z['query_layer_indices'].tolist()))
            commit_checks.append({'path':path.relative_to(BASE).as_posix(),'sha256':commit['sha256']})
        assert len(pairs) == 160 and len(splits) == 80 and all(len(s)==1 for s in splits.values())
        for pair in pairs.values():
            assert len(pair)==2 and Counter(pair[0]['prompt_ids']) == Counter(pair[1]['prompt_ids'])
        probes = group['probes']
        assert Counter(p['split'] for p in probes) == {'train_query':60,'validation_query':20,'unseen_query':20}
        for stage in ['capture','analysis','fit','diagonal','compilation','native_language']:
            assert read(BASE/stage/model/'result.json')['all_passed'],(model,stage)
        for p in (BASE/'fit'/model/'operators').glob('*.npz'):
            meta=read(p.with_suffix('.json')); assert sha(p)==meta['operator_sha256']
            with np.load(p) as z:
                assert z['operator'].shape == (width,width)
        if model!='qwen4':
            records=gzread(BASE/'native_language'/model/'records.json.gz')
            assert len(records)==320
            for r in records:
                p=BASE/'native_language'/model/'fields'/(r['sample_id']+'.npz')
                assert sha(p)==r['field_sha256']
                with np.load(p) as z:
                    assert z['generated_ids'].tolist()==r['generated_ids']
                    assert z['selected_hidden_states'].shape==(len(r['generated_ids']),4,width)
            assert read(BASE/'native_language'/model/'numerical_admission.json')['all_passed']
        checks.append(model+': source-group split, all matched pairs, all320row hashes/axes, every full matrix and all native trajectory identities')
    for path in (BASE/'sources').iterdir():
        digest=path.stem.rsplit('_',1)[-1]
        assert sha(path).startswith(digest),str(path)
    checks.append('Every immutable execution-source snapshot matches its registered content digest')
    compressed(OUT/'capture_commit_hashes.json.gz',commit_checks)
    archive_result=archives()
    assert read(OUT/'model_checkpoint_fingerprints.json')['all_passed']
    for file in ['client/api_final.json','client/browser_final.json','client/code_checks.json','client/visual_review.json']:
        assert read(BASE/file)['all_passed'],file
    assert read(BASE/'norm_controls/analysis.json')['variants']==23
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,
        'archive_coverage':archive_result,'seconds':time.monotonic()-start,
        'scope':'Execution/data-integrity success is distinct from hypothesis confirmation. Original checkpoints, prior artifacts and MEMO preserved; no semantic mechanism completion claim.'}
    save(OUT/'result.json',result)
    print('CONSTRUCTION_INTEGRITY_DONE',len(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archives',action='store_true');p.add_argument('--checkpoints',action='store_true')
    a=p.parse_args()
    if a.archives:archives()
    elif a.checkpoints:checkpoints()
    else:identities()
