"""Independent all-coordinate checks of the signed source feature implementation."""
from rdc_binding_common import *

def main():
    from threadpoolctl import threadpool_limits
    from rdc_binding_kernels import source_arrays,apply_roles
    from phase2735_rdc_binding_signed import features,kernels
    threadpool_limits(limits=2);start=time.monotonic();out=BASE/'signed_source'
    rows=gzread(BASE/'natural_discovery.json.gz')[:4]
    h,q,e,_,_,_=source_arrays(rows)
    with np.load(BASE/'prediction/role_probe.npz') as z:coef=z['coefficients'].astype(float)
    pack=features(rows,h,q,e,coef);operational=kernels(pack,pack);roles=apply_roles(h,coef);checks={}
    for key in ('position','role_position','role_shuffled'):
        brute=np.zeros((4,4))
        for i in range(4):
          for j in range(4):
            hi,hj=h[i].astype(float),h[j].astype(float);d=hi.shape[1]
            zi=np.linspace(-1,0,len(hi));zj=np.linspace(-1,0,len(hj))
            pi=np.stack([np.ones_like(zi),zi,zi*zi],1);pj=np.stack([np.ones_like(zj),zj,zj*zj],1)
            ri,rj=roles[i],roles[j]
            if key=='role_shuffled':
                ri=ri[np.random.default_rng(int(ranked('signed-role/'+rows[i]['sample_id'])[:8],16)).permutation(len(ri))]
                rj=rj[np.random.default_rng(int(ranked('signed-role/'+rows[j]['sample_id'])[:8],16)).permutation(len(rj))]
            base=(pack['q'][i]@pack['q'][j]+pack['e'][i]@pack['e'][j])/(2*d)
            dot=(hi@hj.T/d)*(pi@pj.T)
            if key!='position':dot*=1+ri@rj.T
            s=dot.mean();brute[i,j]=1+base+s+base*s
        error=float(np.max(abs(brute-operational['signed_'+key])))
        assert error<1e-10,(key,error);checks[key]=error
    with np.load(BASE/'verification/source_moment_collision/synthetic_normalized_state_pair.npz') as z:
        pair=[z['history_a'],z['history_b']];query=np.stack([z['query']]*2);embedding=np.stack([z['embedding']]*2)
    pairpack=features([rows[0],rows[0]],pair,query,embedding,coef)
    distances={k:float(np.linalg.norm(pairpack[k][0]-pairpack[k][1])) for k in ('position','role_position')}
    assert min(distances.values())>1e-5
    baseline=[];prior=read(BASE/'prediction/frozen.json')['selected']
    for block in (16,35):
        name=prior[str(block)]['kernel']
        with np.load(BASE/'prediction/predictions'/f'b{block}_{name}_128_direct_mlp.npz') as z:old=z['prediction']
        with np.load(out/'discovery_predictions'/f'b{block}_original.npz') as z:new=z['prediction']
        relative=float(np.max(abs(old-new))/np.max(abs(old)))
        assert relative<1e-4,relative
        baseline.append({'block':block,'old_decoder':'direct_mlp','relative_max_prediction_difference':relative,
          'scope':'Independent CPU float64 ridge versus original CUDA float32 fit; same material, kernel and effective df, rounding not identical.'})
    npz(out/'signed_math_values.npz',**operational,**{'collision_'+k:v for k,v in kernels(pairpack,pairpack).items()})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'all_passed':True,'brute_all_source_pair_max_errors':checks,
      'collision_signed_feature_distances':distances,'reproduced_original_direct_decoder':baseline,
      'seconds':time.monotonic()-start,
      'scope':'All source pairs and all2560coordinates in four real prefix inputs; synthetic collision resolution checked separately. Algebra agreement is not language improvement.'}
    save(out/'math.json',result);ledger('signed_source_independent_CPU_math',result['seconds'])
    print('SIGNED_SOURCE_MATH_PASS',json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
