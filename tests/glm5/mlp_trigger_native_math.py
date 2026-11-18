"""Postregistered weight-based decomposition, separate from frozen prediction gates."""
import json
import math
import numpy as np
import mlp_trigger_design as D
from shared.dist_protocol.mechanism_trace import unbits

def raw_weight(name):
    index=D.READ(D.MODEL/'model.safetensors.index.json')['weight_map'];path=D.MODEL/index[name]
    with path.open('rb') as f:
        n=int.from_bytes(f.read(8),'little');h=json.loads(f.read(n));e=h[name];assert e['dtype']=='BF16'
        lo,hi=e['data_offsets'];f.seek(8+n+lo);v=unbits(np.frombuffer(f.read(hi-lo),dtype=np.uint16)).reshape(e['shape'])
    return v

def main():
    p=D.checked();gamma=raw_weight('model.layers.33.post_attention_layernorm.weight')
    folder=D.ROOT/'tests/glm5/result/mlp_tail_goodfire_v1'
    wg=unbits(np.load(folder/'B33-W-gate.npy',mmap_mode='r')[1521]);wv=unbits(np.load(folder/'B33-W-up.npy',mmap_mode='r')[1521])
    eps=D.READ(D.MODEL/'config.json')['rms_norm_eps'];g_inf=float(gamma[4]*wg[4]*math.sqrt(2560));v_inf=float(gamma[4]*wv[4]*math.sqrt(2560))
    records=[]
    for f in range(D.N_HOLD):
        z=np.load(D.OUT/f'holdout-{f:04d}.npz');m=D.READ(D.OUT/f'holdout-{f:04d}.json');rr=unbits(z['residual'])[:,1];a=unbits(z['prefix'])
        for k,r in enumerate(rr):
            pos=m['rows'][k]['positions']['rare'];normalized=r.astype(np.float64)/np.sqrt(np.mean(r.astype(np.float64)**2)+eps)*gamma
            g=float(wg@normalized);v=float(wv@normalized);g4=float(wg[4]*normalized[4]);v4=float(wv[4]*normalized[4]);actual=a[k,2,:,pos]
            records.append({'id':m['rows'][k]['id'],'energy4':float(r[4]**2/np.sum(r.astype(np.float64)**2)),
                'actual_gva':actual.tolist(),'full_fp32_gv':[g,v],'coordinate4_gv':[g4,v4],
                'coordinate4_A':g4/(1+math.exp(-g4))*v4,'limit_positive_A':g_inf/(1+math.exp(-g_inf))*v_inf})
    subset=[v for v in records if v['energy4']>=.9]
    assert subset
    doc={'schema':'mlp_trigger_weight_decomposition.v1','status':'postregistered descriptive algebra; no new confirmation gate',
         'formula':'G=sum_j Wg[j]*gamma[j]*r[j]/sqrt(mean(r^2)+eps); V analogous; A=silu(G)*V. If positive r4 dominates energy, G≈sqrt(d)*gamma4*Wg4 and V≈sqrt(d)*gamma4*Wu4.',
         'gamma':gamma.tolist(),'gate_row':wg.tolist(),'up_row':wv.tolist(),'weights':{'gamma4':float(gamma[4]),'gate4':float(wg[4]),'up4':float(wv[4])},
         'positive_dominance_limit':{'G':g_inf,'V':v_inf,'A':g_inf/(1+math.exp(-g_inf))*v_inf},
         'dominant_n':len(subset),'n':len(records),
         'dominant_median_relative_A_error':float(np.median([abs(v['coordinate4_A']-v['actual_gva'][2])/max(abs(v['actual_gva'][2]),1e-12) for v in subset])),
         'dominant_median_energy4':float(np.median([v['energy4'] for v in subset])),
         'full_gv_max_abs_recompute_error':float(max(np.max(abs(np.asarray(v['full_fp32_gv'])-v['actual_gva'][:2])) for v in records)),
         'scope':'Float32/float64 algebraic reconstruction differs from BF16 rounding. Shared coordinate dominance explains response amplitude; formation of the upstream state and its linguistic role remain open.',
         'records':records}
    D.save('native-math.json',doc,p);print({k:v for k,v in doc.items() if k not in ['gamma','gate_row','up_row','records']},flush=True)
if __name__=='__main__':main()
