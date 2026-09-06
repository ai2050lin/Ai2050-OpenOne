"""Lossless ZIP container change only, verified bitwise arrays; no feature compression."""
import hashlib,os,re,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *

BF=RESULT/'phase2649_output_function_behavior';MAPS=RESULT/'phase2651_output_function_maps';OUT=RESULT/'phase2652_output_function_confirmation'

def array_digest(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    destination=OUT/'analysis/lossless_storage_transition.json'
    if destination.exists():raise RuntimeError('transition already complete; do not repeat')
    raw=read(BF/'analysis/raw_manifest.json');targets=[(Path(r['path']),r['bytes'],'BF16_raw_container') for r in raw]
    targets += [(p,p.stat().st_size,'initial_derived_maps') for p in sorted((MAPS/'maps').glob('*.npz'))]
    assert len(raw)==4096 and len(targets)==4161
    plan=[]
    for p,size,kind in targets:
        p=p.resolve();assert p.is_relative_to(RESULT.resolve()) and p.parent in ((BF/'field').resolve(),(MAPS/'maps').resolve())
        if kind=='BF16_raw_container':assert re.fullmatch(r'case_\d{4}\.npz',p.name)
        assert p.is_file() and p.stat().st_size==size
        plan.append({'path':str(p),'old_bytes':size,'kind':kind})
    save(OUT/'protocol/lossless_storage_plan.json',{'targets':plan,'rule':'same arrays bit-for-bit, onlyZIP container DEFLATE; old phase manifests preserved','before_free_bytes':shutil.disk_usage(OUT).free})
    results=[]
    for i,r in enumerate(plan):
        p=Path(r['path']);oldsha=sha(p)
        with np.load(p,allow_pickle=False) as old:arrays={k:old[k] for k in old.files}
        digests={k:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256_C_bytes':array_digest(a)} for k,a in arrays.items()}
        temp=p.with_name(p.stem+'.lossless-repack.npz');assert not temp.exists()
        np.savez_compressed(temp,**arrays)
        with np.load(temp,allow_pickle=False) as new:
            assert set(new.files)==set(arrays)
            for k,a in arrays.items():
                b=new[k];assert b.shape==a.shape and b.dtype==a.dtype and array_digest(b)==digests[k]['sha256_C_bytes']
        assert p.stat().st_size==r['old_bytes'] and sha(p)==oldsha
        os.replace(temp,p)
        results.append({**r,'old_file_sha256':oldsha,'new_bytes':p.stat().st_size,'new_file_sha256':sha(p),'arrays':digests,'bitwise_identical':True})
        del arrays
        if (i+1)%128==0:save(OUT/'analysis/lossless_progress.json',{'files':i+1,'total':len(plan)});print('lossless file storage',i+1,'/',len(plan),flush=True)
    report={'files':results,'all_arrays_bitwise_identical':all(r['bitwise_identical'] for r in results),'old_bytes':sum(r['old_bytes'] for r in results),'new_bytes':sum(r['new_bytes'] for r in results),
        'after_free_bytes':shutil.disk_usage(OUT).free,'scope':'4096 current BF16 raw packs and65 initial derived NPZs, no old raw manifest mutation, no model or unrelated artifact change',
        'reversible_representation':'np.load returns identical namedarrays, shape,dtype,bit patterns. Can rewrite uncompressed containers; no dimension or low-value deletion.'}
    save(destination,report);print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)

if __name__=='__main__':main()
