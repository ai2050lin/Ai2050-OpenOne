"""Full original checkpoint/config and previous-evidence preservation audit."""
from rdc_query_common import *


def main():
    start=time.monotonic();assert read(BASE/'science_queue/status.json')['all_passed']
    prior=read(PRIOR/'verification/model_checkpoint_fingerprints.json')['models'];models={}
    for name in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):
        folder=(ROOT/'models/hf'/name).resolve();old={r['file']:r for r in prior[name]}
        names=set(read(folder/'model.safetensors.index.json')['weight_map'].values())
        names.update(p.name for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt','.model','.tiktoken'))
        assert names==set(old),(name,'Original file set changed');records=[]
        for relative in sorted(names):
            path=(folder/relative).resolve();assert path.is_relative_to(folder) and path.is_file()
            before=path.stat();digest=sha(path);after=path.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            assert digest==old[relative]['sha256'],(name,relative)
            records.append({'file':relative,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':digest})
        models[name]=records;print('QUERY_ORIGINAL_CHECKPOINT_HASH',name,sum(r['bytes'] for r in records),flush=True)
    evidence=[]
    for r in read(BASE/'contract.json')['prior_required_artifacts_verified']:
        assert sha(PRIOR/r['path'])==r['sha256'];evidence.append(r)
    result={'timestamp':stamp(),'source':snapshot(__file__),'models':models,'all_passed':True,
      'previous_required_artifacts_unchanged':evidence,'prior_manifest_sha256':sha(PRIOR/'verification/model_checkpoint_fingerprints.json'),
      'seconds':time.monotonic()-start,'scope':'All original model shards and registered tokenizer/config files read in full after this campaign native jobs. No original checkpoint write; endpoint verification is not continuous monitoring.'}
    save(BASE/'verification/model_checkpoint_fingerprints.json',result);ledger('query_original_checkpoint_and_previous_evidence_hashes',result['seconds'])


if __name__=='__main__':main()
