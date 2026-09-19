"""Read-only full checkpoint and tokenizer identity, captured at delivery (not retroactive proof)."""
from rdc_joint_common import *

def main():
    path=BASE/'verification/model_checkpoint_fingerprints.json'
    if path.exists():return
    start=time.monotonic();guard(512*1024);models={}
    for name in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):
        folder=(ROOT/'models/hf'/name).resolve();index=read(folder/'model.safetensors.index.json')
        names=set(index['weight_map'].values())
        names.update(p.name for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt','.model','.tiktoken'))
        records=[]
        for relative in sorted(names):
            target=(folder/relative).resolve();assert target.is_relative_to(folder) and target.is_file()
            before=target.stat();digest=sha(target);after=target.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            records.append({'file':relative,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':digest})
        models[name]=records;print('CHECKPOINT_FINGERPRINTED',name,sum(r['bytes'] for r in records),flush=True)
    save(path,{'timestamp':stamp(),'source':snapshot(Path(__file__)),'models':models,'seconds':time.monotonic()-start,
        'scope':'Full original checkpoint shards and local tokenizer/config/source files, read-only and unchanged within each hash. Captured at delivery; original capture runtime already records tokenizer/model-code/config identities, but this late hash alone is not proof weights were unchanged at every earlier instant.'})
    print('ALL_CHECKPOINT_FINGERPRINTS_COMPLETE',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
