"""Read-only complete checkpoint/config comparison after the bounded GPU work."""
from rdc_update_common import *


def main():
    import psutil
    # Do not compete with, or audit a checkpoint during, this campaign's GPU job.
    for process in psutil.process_iter(['pid','cmdline']):
        command=' '.join(process.info['cmdline'] or [])
        if process.pid!=os.getpid() and any(n in command for n in
          ('phase2738_rdc_update_scale.py','phase2739_rdc_update_long_answers.py',
           'phase2739_rdc_update_scale_recovery.py','phase2739_rdc_update_scale_batch.py')):
            raise AssertionError(('Native process still active',process.pid,command))
    start=time.monotonic();prior=read(PRIOR/'verification/model_checkpoint_fingerprints.json')['models'];models={}
    for name in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):
        folder=(ROOT/'models/hf'/name).resolve();old={r['file']:r for r in prior[name]}
        names=set(read(folder/'model.safetensors.index.json')['weight_map'].values())
        names.update(p.name for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt','.model','.tiktoken'))
        assert names==set(old),(name,'Original file set changed')
        records=[]
        for relative in sorted(names):
            path=(folder/relative).resolve();assert path.is_relative_to(folder) and path.is_file()
            before=path.stat();digest=sha(path);after=path.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            assert digest==old[relative]['sha256'],(name,relative)
            records.append({'file':relative,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':digest})
        models[name]=records;print('UPDATE_CHECKPOINT_HASH',name,sum(r['bytes'] for r in records),flush=True)
    result={'timestamp':stamp(),'source':snapshot(__file__),'models':models,'all_passed':True,
      'prior_manifest_sha256':sha(PRIOR/'verification/model_checkpoint_fingerprints.json'),
      'seconds':time.monotonic()-start,
      'scope':'Every original shard and registered tokenizer/config/local-source file read after native runs. No original checkpoint write; this is endpoint verification, not continuous monitoring.'}
    save(BASE/'verification/model_checkpoint_fingerprints.json',result);ledger('update_checkpoint_fingerprints',result['seconds'])


if __name__=='__main__':main()
