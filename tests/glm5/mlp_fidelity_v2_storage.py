"""Transparent NTFS compression of frozen research files, with full byte-hash checks."""
import sys,subprocess,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import file_sha

def main():
    assert P.read(P.OUT/'controlled-bank/manifest.json')['state']=='complete';assert P.read(P.OUT/'natural-bank/manifest.json')['state']=='complete';assert (P.OUT/'tail-capability-v3/result.json').exists()
    folders=['controlled-bank','natural-bank','trained','development','development-expansion','semantics-v2','semantics','tail-natural','tail-generated-v2','tail-capability-v3'];files=sorted(p.resolve() for f in folders for p in (P.OUT/f).glob('*') if p.suffix in ['.npy','.npz'])
    assert files and all(p.is_relative_to(P.ROOT.resolve()) and p.is_file() for p in files);before=shutil.disk_usage(P.OUT).free;start=time.time();hashes={str(p.relative_to(P.ROOT)):file_sha(p) for p in files}
    with (P.OUT/'storage-compact.log').open('w',encoding='utf-8') as log:
        for i in range(0,len(files),16):
            subprocess.run(['C:/windows/system32/compact.exe','/C','/I','/Q',*[str(p) for p in files[i:i+16]]],stdout=log,stderr=subprocess.STDOUT,check=True)
            print({'compressed_files':min(i+16,len(files)),'total':len(files)},flush=True)
    for p in files:assert file_sha(p)==hashes[str(p.relative_to(P.ROOT))]
    result={'schema':'research_lossless_storage_audit.v1','kind':'real','run_id':P.RUN,'model_snapshot':P.read(P.OUT/'plan.json')['model_snapshot'],'provenance':P.read(P.OUT/'plan.json')['provenance'],'files':hashes,'n_files':len(files),'free_bytes_before':before,'free_bytes_after':shutil.disk_usage(P.OUT).free,'seconds':time.time()-start,'all_logical_file_bytes_unchanged':True,'method':'NTFS transparentlosslesscompression, not modelquantization; originalweights untouched; no researchfiles deleted'};P.save(P.OUT/'storage-audit.json',result);print({'all_hashes_unchanged':True,'free_GiB_after':result['free_bytes_after']/2**30,'seconds':result['seconds']},flush=True)
if __name__=='__main__':main()
