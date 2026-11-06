"""Describe the mismatch between two reproduced Head writes and the complete residual."""
import numpy as np

import mlp_functional_common as C
from mlp_functional_trace import contrasts


def main():
    folder=C.OUT/'revision';obs=C.read(folder/'confirm-result.json');native=folder/'observed-native.npz'
    assert C.file_sha(native)==obs['native']['sha256']
    writers=obs['replication']['industry']['writers']
    picked=[j for j,p in enumerate(writers) if p['head'] is not None and p['replicated']]
    if len(picked)!=2:return
    with np.load(native) as raw:
        all_writes=raw['incoming_scores'][:,:,1]
        selected=raw['writer_scores'][:,:,1,:][:,:,picked].sum(-1)
    other=all_writes-selected
    stats={}
    for split,sl in [('confirm',slice(0,160)),('lockbox',slice(160,320))]:
        ds={k:contrasts(v[sl],'industry') for k,v in [('selected_heads',selected),('remainder',other),('full_residual',all_writes)]}
        stats[split]={k:{'paired_accuracy_by_R':(v>0).mean(0).tolist(),'mean_by_R':v.mean(0).tolist(),
                         'sd_by_R':v.std(0,ddof=1).tolist()} for k,v in ds.items()}
        mismatch=np.logical_and(ds['selected_heads']>0,ds['full_residual']<=0)
        start=32+(0 if split=='confirm' else 160)
        stats[split]['head_positive_full_nonpositive']={'count_by_R':mismatch.sum(0).tolist(),
            'background_and_R':[[int(start+b),int(r)] for b,r in np.argwhere(mismatch)],'n_backgrounds':160}
    report={**C.identity(),'schema':'functional_writer_remainder.v1','native_sha256':obs['native']['sha256'],
            'source_sha256':C.file_sha(__file__),'selected_heads':[[writers[j]['block'],writers[j]['head']] for j in picked],
            'summary':stats,'scope':'探索性事后分解。剩余项包含其他写入、嵌入和原生数值舍入。仅做成对排序，不把对比标签当作单条可用信息。'}
    C.save(folder/'remainder.json',report)
    print(stats['lockbox']['selected_heads'],stats['lockbox']['full_residual'],flush=True)


if __name__=='__main__':main()
