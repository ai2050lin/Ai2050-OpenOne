"""Read-only end-of-campaign checkpoint fingerprints, compared to prior delivery."""
from rdc_operator_common import *


def main():
    path=BASE/'verification/model_checkpoint_fingerprints.json'
    if path.exists():return
    start=time.monotonic();models={};prior=read(PRIOR/'verification/model_checkpoint_fingerprints.json')['models'];comparisons=[]
    for name in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):
        folder=(ROOT/'models/hf'/name).resolve();index=read(folder/'model.safetensors.index.json')
        names=set(index['weight_map'].values())
        names.update(p.name for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt','.model','.tiktoken'))
        records=[];old={r['file']:r for r in prior[name]}
        for relative in sorted(names):
            target=(folder/relative).resolve();assert target.is_relative_to(folder) and target.is_file()
            before=target.stat();digest=sha(target);after=target.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            records.append({'file':relative,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':digest})
            if relative in old:
                same=digest==old[relative]['sha256'];assert same,(name,relative)
                comparisons.append({'model':name,'file':relative,'prior_sha_identical':same})
        models[name]=records;print('OPERATOR_CHECKPOINT_FINGERPRINTED',name,sum(r['bytes'] for r in records),flush=True)
    save(path,{'timestamp':stamp(),'source':snapshot(Path(__file__)),'models':models,'prior_delivery_comparison':comparisons,'seconds':time.monotonic()-start,
        'scope':'Full original checkpoint shards and local tokenizer/config/source files read-only. Same SHA as prior delivery for shared files; no in-flight changes during each hash. End-of-campaign equality is not continuous proof at every intermediate instant.'})
    ledger('read_only_checkpoint_fingerprint',time.monotonic()-start)


if __name__=='__main__':main()
