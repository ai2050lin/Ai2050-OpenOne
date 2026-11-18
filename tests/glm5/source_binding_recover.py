"""Explicit storage-only v2 recovery, retaining frozen v1 and failure evidence."""
import sys,shutil,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D

def prepare():
    root=D.OUT/'qwen14';folder=root/'discovery-native';failure=root/'discovery-native-v1-failure';assert not failure.exists()
    log=D.OUT/'qwen14-discovery-native-console.log';assert 'RAM/commit resource stop' in log.read_text(encoding='utf-8-sig')
    assert not (root/'discovery-native-manifest.json').exists();failure.mkdir()
    source=D.ROOT/'tests/glm5/source_binding_native.py';text=source.read_text(encoding='utf-8')
    needle='        for chunk in range(0,n,128):\n            rs='
    replacement='''        for chunk in range(0,n,128):
            if chunk:
                for name,array in list(arrays.items()):
                    array.flush();array._mmap.close()
                    arrays[name]=np.lib.format.open_memmap(folder/f'{name}.npy',mode='r+')
                gc.collect()
            rs='''
    assert text.count(needle)==1;text=text.replace(needle,replacement)
    needle='    for a in arrays.values():a.flush()\n    arrays.clear()'
    assert text.count(needle)==1;text=text.replace(needle,'    for a in arrays.values():a.flush();a._mmap.close()\n    arrays.clear()')
    text=text.replace('"""Natural QK/VO and SwiGLU collection after complete behavior admission."""','"""V2: same frozen computation; close file mappings between chunks to release resident RAM."""')
    new=D.ROOT/'tests/glm5/source_binding_native_v2.py';assert not new.exists();new.write_text(text,encoding='utf-8')
    sample={}
    for name,indices in [('hidden',[0,20,40]),('key',[0,20,39]),('gate',[0,20,39])]:
        a=np.load(folder/f'{name}.npy',mmap_mode='r');sample[name]=np.array(a[:8,indices]);a._mmap.close()
    a=np.load(folder/'final_norm.npy',mmap_mode='r');sample['final_norm']=np.array(a[:8]);a._mmap.close();np.savez(failure/'completed-sample.npz',**sample)
    files={};start=time.monotonic()
    for i,p in enumerate(sorted(folder.glob('*.npy'))):
        files[p.name]={'bytes':p.stat().st_size,'sha256':D.sha(p)}
        D.log('v1-failure-preservation',i+1,len(list(folder.glob('*.npy'))),start)
    D.save('qwen14/discovery-native-v1-failure/record.json',{'reason':'hard 2GiB available RAM guard; mapped file residency accumulated across chunks, commit stayed stable; no OOM',
        'v1_code_sha256':D.sha(source),'v2_code_sha256':D.sha(new),'partial_files':files,'scientific_use':False,
        'retained_sample_sha256':D.sha(failure/'completed-sample.npz'),'console_sha256':D.sha(log),
        'change':'close/reopen all flushed memmaps between128-row chunks and close before final hash; no compute, candidate selection, thresholds, materials or model changes'},'qwen14')
    # Deletion is limited to this task's incomplete, unadmitted native stage.
    target=folder.resolve();assert target== (D.ROOT/'tests/glm5/result/source_binding_v1/qwen14/discovery-native').resolve()
    shutil.rmtree(target)
    (root/'discovery-capture-started.json').rename(failure/'capture-started.json')
    D.save('native-v2-code-freeze.json',{'original_internal_freeze_sha256':D.sha(D.OUT/'internal-code-freeze.json'),
        'code_sha256':{str(p.relative_to(D.ROOT)):D.sha(p) for p in [new,Path(__file__)]},
        'reason':'resource-only recovery before candidate selection; original frozen sources retained','freed_partial_bytes':sum(v['bytes'] for v in files.values())})

def run(model,split):
    from source_binding_analysis import verify_code
    verify_code();f=D.read(D.OUT/'native-v2-code-freeze.json')
    for p,h in f['code_sha256'].items():assert D.sha(D.ROOT/p)==h
    proof=D.admission(model,split)
    import source_binding_native_v2
    source_binding_native_v2.collect(model,split,proof)
    if model=='qwen14' and split=='discovery':
        sample=np.load(D.OUT/'qwen14/discovery-native-v1-failure/completed-sample.npz');checks={}
        for name,indices in [('hidden',[0,20,40]),('key',[0,20,39]),('gate',[0,20,39]),('final_norm',None)]:
            a=np.load(D.OUT/model/f'{split}-native/{name}.npy',mmap_mode='r');new=np.asarray(a[:8] if indices is None else a[:8,indices])
            checks[name]=bool(np.array_equal(sample[name],new));assert checks[name]
        D.save('qwen14/native-v2-replay-check.json',{'v1_completed_sample_bitwise_equal':checks},model)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:run(sys.argv[2],sys.argv[3])
