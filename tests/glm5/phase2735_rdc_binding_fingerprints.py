"""Read every original model shard after all native update processes finish."""
from rdc_binding_common import *

def main():
    assert read(BASE/'format_content/suite_result.json')['all_passed']
    start=time.monotonic();models={};comparisons=[]
    prior=read(LAW/'verification/model_checkpoint_fingerprints.json')['models']
    for name in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):
        folder=(ROOT/'models/hf'/name).resolve()
        names=set(read(folder/'model.safetensors.index.json')['weight_map'].values())
        names.update(p.name for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt','.model','.tiktoken'))
        old={r['file']:r for r in prior[name]};records=[]
        assert set(old)==names,(name,'Changed original checkpoint/config file set')
        for relative in sorted(names):
            target=(folder/relative).resolve();assert target.is_relative_to(folder) and target.is_file()
            before=target.stat();digest=sha(target);after=target.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            assert digest==old[relative]['sha256'],(name,relative)
            records.append({'file':relative,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':digest})
            comparisons.append({'model':name,'file':relative,'prior_sha_identical':True})
        models[name]=records
        print('BINDING_CHECKPOINT_SHA',name,sum(r['bytes'] for r in records),flush=True)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'models':models,
      'prior_delivery_comparison':comparisons,'all_passed':True,'seconds':time.monotonic()-start,
      'scope':'Full original model shards and tokenizer/config/local source files, read only, after the serial GPU suite. Identical to prior delivery; not continuous intermediate-state monitoring.'}
    save(BASE/'verification/model_checkpoint_fingerprints.json',result)
    ledger('binding_original_checkpoint_fingerprints',result['seconds'])

if __name__=='__main__':main()
