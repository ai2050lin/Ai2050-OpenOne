"""Descriptive two-coordinate algebra; not a replacement for frozen predictors."""
import argparse
import math
import numpy as np
import mlp_trigger_design as D
from mlp_trigger_native_math import raw_weight
from shared.dist_protocol.mechanism_trace import unbits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--read100',action='store_true');args=parser.parse_args()
    if args.read100:
        import mlp_trigger_read100 as S
        S.configure()
    p=D.checked();folder=D.ROOT/'tests/glm5/result/mlp_tail_goodfire_v1';gamma=raw_weight('model.layers.33.post_attention_layernorm.weight')
    wg=unbits(np.load(folder/'B33-W-gate.npy',mmap_mode='r')[1521]);wv=unbits(np.load(folder/'B33-W-up.npy',mmap_mode='r')[1521])
    discovery=D.ROOT/p['discovery_reused_from'] if p.get('discovery_reused_from') else D.OUT
    rr=np.concatenate([unbits(np.load(discovery/f'discovery-{f:04d}.npz')['residual'])[:,1] for f in range(96)])
    norm=rr/np.sqrt(np.mean(rr**2,1,keepdims=True)+1e-6)*gamma;contributions={}
    for label,w in [('G',wg),('V',wv)]:
        c=norm*w;idx=np.argsort(np.mean(abs(c),0))[-12:][::-1]
        contributions[label]=[{'coordinate':int(j),'mean_signed_contribution':float(c[:,j].mean()),'mean_abs_contribution':float(abs(c[:,j]).mean()),'gamma':float(gamma[j]),'weight':float(w[j])} for j in idx]
    discovery_A=[]
    for f in range(96):
        za=np.load(discovery/f'discovery-{f:04d}.npz');ma=D.READ(discovery/f'discovery-{f:04d}.json');av=unbits(za['prefix'])
        discovery_A.extend(float(av[k,2,2,m['positions']['rare']]) for k,m in enumerate(ma['rows']))
    constant=float(np.mean(discovery_A));records=[]
    for f in range(D.N_HOLD):
        z=np.load(D.OUT/f'holdout-{f:04d}.npz');m=D.READ(D.OUT/f'holdout-{f:04d}.json');a=unbits(z['prefix'])
        for k,r in enumerate(unbits(z['residual'])[:,1]):
            pos=m['rows'][k]['positions']['rare'];den=math.sqrt((float(r[4])**2+float(r[100])**2)/2560+1e-6)
            g=float((wg[[4,100]]*gamma[[4,100]]*r[[4,100]]).sum())/den
            v=float((wv[[4,100]]*gamma[[4,100]]*r[[4,100]]).sum())/den
            pred=g/(1+math.exp(-g))*v;actual=float(a[k,2,2,pos]);energy=float(r[4])**2/np.sum(r.astype(np.float64)**2)
            records.append({'id':m['rows'][k]['id'],'energy4':float(energy),'r4':float(r[4]),'r100':float(r[100]),'actual_A':actual,'two_coordinate_A':pred,
                            'relative_error':abs(pred-actual)/max(abs(actual),1e-12)})
    dominated=[r for r in records if r['energy4']>=.9]
    D.save('native-math-two-coordinate.json',{'schema':'mlp_relative_coordinate_read.v1','status':'descriptive analytic approximation; no new significance gate',
       'formula':'G≈gamma100*Wg100*r100/sqrt((r4^2+r100^2)/d); V analogous; A=silu(G)*V. Full normalized coordinate4 terms are also retained numerically.',
       'interpretation':'Large coordinate4 mainly affects the RMS denominator; coordinate100 provides the dominant direct G/V numerator in this observed state. This is conditional numerical read-in, not a linguistic semantic label.',
       'discovery_contributions':contributions,'weights':{'gamma4':float(gamma[4]),'gamma100':float(gamma[100]),'gate4':float(wg[4]),'gate100':float(wg[100]),'up4':float(wv[4]),'up100':float(wv[100])},
       'n':len(records),'dominated_n':len(dominated),'dominated_median_relative_error':float(np.median([r['relative_error'] for r in dominated])),
       'discovery_constant_A':constant,
       'all_MSE_vs_discovery_constant':float(sum((r['two_coordinate_A']-r['actual_A'])**2 for r in records)/sum((constant-r['actual_A'])**2 for r in records)),
       'dominated_MSE_vs_discovery_constant':float(sum((r['two_coordinate_A']-r['actual_A'])**2 for r in dominated)/sum((constant-r['actual_A'])**2 for r in dominated)),
       'records':records},p)
    print({'n':len(records),'dominated':len(dominated),'median_relative_error':float(np.median([r['relative_error'] for r in dominated]))},flush=True)
if __name__=='__main__':main()
