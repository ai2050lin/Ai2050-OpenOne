"""Full archive/checkpoint preservation audit, aware of lossless XOR word storage."""
import argparse
import zipfile
from rdc_question_common import *
import phase2745_rdc_construction_integrity as previous_integrity

VERIFY=OUT/'verification'


def inspect_archive(path_or_stream,relative_name):
    """Stream every stored scalar and CRC; XOR words are not BF16 numbers."""
    records=[]
    with zipfile.ZipFile(path_or_stream)as archive:
        names=[r.filename for r in archive.infolist()]
        assert len(set(names))==len(names),'Duplicate array member'
        for item in archive.infolist():
            assert item.filename.endswith('.npy') and '/' not in item.filename
            with archive.open(item)as stream:
                version=np.lib.format.read_magic(stream)
                assert version in [(1,0),(2,0)],version
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype=reader(stream)
                assert dtype.kind in 'buif',(relative_name,item.filename,dtype)
                xor=item.filename in ('FP32_XOR_words.npy','BF16_XOR_words.npy')
                if xor:
                    assert relative_name.startswith('phase2748/field_store/training/') and '/checkpoint96/' in relative_name
                    assert dtype==(np.dtype('uint32') if item.filename.startswith('FP32') else np.dtype('uint16'))
                size=int(np.prod(shape,dtype=np.int64)) if shape else 1
                remaining=size*dtype.itemsize
                while remaining:
                    block=stream.read(min(8*1024**2,remaining))
                    assert block and len(block)%dtype.itemsize==0
                    values=np.frombuffer(block,dtype=dtype)
                    if not xor and dtype==np.dtype('uint16'):
                        assert not np.any((values & 0x7f80)==0x7f80),(relative_name,item.filename,'NonfiniteBF16')
                    elif dtype.kind=='f':
                        assert np.isfinite(values).all(),(relative_name,item.filename,'Nonfinitefloat')
                    remaining-=len(block)
                assert stream.read(1)==b'',(relative_name,item.filename,'Trailing bytes')
            records.append({'name':item.filename[:-4],'shape':list(shape),'dtype':str(dtype),'fortran':bool(fortran),
                'scalars':size,'bytes':size*dtype.itemsize,
                'interpretation':'Lossless XOR storage words; actual decoded FP32/BF16 validated separately' if xor else 'Native BF16 words' if dtype==np.dtype('uint16') else 'Declared numeric or Boolean dtype'})
    return records


def inventory():
    # Explicitly include the registered junction target through its workspace
    # alias. Do not assume a recursive scan traverses every Windows junction.
    files={p.relative_to(BASE).as_posix():p for p in BASE.rglob('*.npz') if '.tmp' not in p.name}
    roots=[BASE.resolve(),PHYSICAL.resolve()]
    for phase in ['phase2746','phase2747']:
        previous_store=BASE/phase/'field_store'
        previous_physical=Path('C:/AI2050-RDC-Archive/rdc_query_construction_20260913')/(phase+'_fields')
        registration=read(BASE/phase/'storage.json')
        assert Path(registration['logical_result_entry'])==previous_store
        assert Path(registration['physical_directory'])==previous_physical
        assert previous_store.resolve()==previous_physical.resolve()
        roots.append(previous_physical.resolve())
        files.update({p.relative_to(BASE).as_posix():p for p in previous_store.rglob('*.npz') if '.tmp' not in p.name})
    files.update({p.relative_to(BASE).as_posix():p for p in OUT.rglob('*.npz') if '.tmp' not in p.name})
    for p in files.values():
        resolved=p.resolve()
        assert any(resolved.is_relative_to(root)for root in roots),str(p)
    return dict(sorted(files.items()))


def archives():
    start=time.monotonic();storage_guard()
    unit=read(OUT/'unit/integrity_current.json')
    assert unit['all_passed'] and unit['analysis']['sha256']==sha(__file__)
    priorpath=BASE/'phase2747/verification/archive_cache.json'
    inherited={r['path']:r for r in read(priorpath)}
    cachepath=VERIFY/'archive_cache.json'
    cached=dict(inherited)
    if cachepath.exists():
        cached.update({r['path']:r for r in read(cachepath)})
    files=inventory()
    assert set(inherited).issubset(files),'An inherited archive is no longer accessible'
    assert set(cached).issubset(files),'An already audited archive is no longer accessible'
    entries=[];fresh=0
    for i,(relative,path)in enumerate(files.items()):
        state=path.stat();digest=sha(path);old=cached.get(relative)
        if old:
            assert digest==old['sha256'],('Previously committed archive changed',relative)
            entry=old
        else:
            members=inspect_archive(path,relative)
            entry={'path':relative,'sha256':digest,'bytes':state.st_size,'arrays':len(members),
                'scalars':sum(r['scalars']for r in members),'decoded_bytes':sum(r['bytes']for r in members),
                'Fortran_layout_arrays':sum(r['fortran']for r in members),'members':members,
                'ZIP_CRC_and_all_values_checked':True,
                'XOR_words_checked_as_encoded_storage_not_native_BF16':True}
            fresh+=1
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(state.st_size,state.st_mtime_ns),'Changed during audit'
        entries.append(entry)
        if (i+1)%100==0:
            save(cachepath,entries)
            print('NATURAL_ARCHIVE_AUDIT',i+1,len(files),fresh,round(time.monotonic()-start,1),flush=True)
    assert set(files)==set(inventory()),'Concurrent new archive; repeat final audit after all jobs finish'
    save(cachepath,entries)
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'inherited_cache_sha256':sha(priorpath),'archive_cache_sha256':sha(cachepath),
        'archives':len(entries),'newly_streamed_archives_this_attempt':fresh,'inherited_archives_rehashed':len(inherited),
        'arrays':sum(r['arrays']for r in entries),'scalars':sum(r['scalars']for r in entries),
        'compressed_bytes':sum(r['bytes']for r in entries),'decoded_bytes':sum(r['decoded_bytes']for r in entries),
        'seconds':time.monotonic()-start,
        'scope':'Every archive freshly SHA256-rehashed, prior finite/CRC evidence reused only for exactly equal bytes; every new scalar/CRC streamed. Encoded XOR words are not interpreted as activations. Every decoded learned parameter is independently required by parameter_formation/result.json. No deletion or numerical compression.'}
    save(VERIFY/'archives.json',value)
    print('NATURAL_ARCHIVES_COMPLETE',len(entries),value['compressed_bytes'],flush=True)
    return value


def checkpoints():
    previous_integrity.OUT=VERIFY
    previous_integrity.checkpoints()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['archives','checkpoints'])
    mode=parser.parse_args().mode
    archives() if mode=='archives' else checkpoints()
