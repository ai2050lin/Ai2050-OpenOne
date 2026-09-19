"""Recompute lexical-cue baselines after the audited four partial-byte corrections."""
from rdc_operator_common import *


def main():
    start=time.monotonic();out=BASE/'identity_audit'
    assert (out/'correction_result.json').exists()
    data={};conservation=[]
    for scope in ('main','confirmation'):
        for path in sorted((BASE/'capture'/scope/'moments').glob('*.npz')):
            fixed=out/'corrected_moments'/path.name;chosen=fixed if fixed.exists() else path
            with np.load(chosen) as z:data[path.stem]={'H':z['H_sums'],'counts':z['counts']}
            if fixed.exists():
                with np.load(path) as z:
                    old=z['H_sums'];original=z['counts']
                new=data[path.stem]['H'];assert np.array_equal(data[path.stem]['counts'][:7],original[:7])
                assert np.array_equal(old[:,:,:7],new[:,:,:7])
                error=float(np.max(np.abs(old[:,:,7:].sum(2)-new[:,:,7:].sum(2))))
                scale=max(float(np.max(np.abs(old[:,:,7:].sum(2)))),1.)
                assert error/scale<1e-12
                conservation.append({'group':path.stem,'piece_moments_bitwise_unchanged':True,'full_cue_partition_sum_relative_error':error/scale})
    design=np.array([[1,*[(mask>>j)&1 for j in range(4)]] for mask in range(16)],dtype=float);reports=[]
    old_reports=read(BASE/'observation/cue_composition.json')['reports'];old_by={(r['language'],r['split'],r['layer'],r['name']):r for r in old_reports}
    for lang in ('en','zh'):
        train=data['train_'+lang];n=train['counts'][7:].astype(float);xtx=design.T@(n[:,None]*design)
        for layer in range(37):
            sy=train['H'][layer,0,7:];mu=sy.sum(0)/n.sum();beta=np.linalg.pinv(xtx,rcond=1e-12)@design.T@sy
            predictions={'global_mean':np.repeat(mu[None],16,axis=0),'additive_four_cues':design@beta,'full16_shrunk_cue_means':(sy+32*mu[None])/(n[:,None]+32)}
            for split in ('validation','test','confirmation'):
                ev=data[split+'_'+lang];ne=ev['counts'][7:].astype(float);sums=ev['H'][layer,0,7:];squares=ev['H'][layer,1,7:]
                for name,p in predictions.items():
                    mse=float(np.sum(squares-2*p*sums+ne[:,None]*p*p)/(ne.sum()*2560))
                    row={'language':lang,'split':split,'layer':layer,'name':name,'all_coordinate_MSE':mse,'tokens':int(ne.sum())}
                    old=old_by.get((lang,split,layer,name))
                    if old:
                        row['original_MSE']=old['all_coordinate_MSE'];row['MSE_change']=mse-old['all_coordinate_MSE']
                        row['relative_MSE_change']=row['MSE_change']/max(abs(old['all_coordinate_MSE']),1e-20)
                    reports.append(row)
    compared=[r for r in reports if 'relative_MSE_change' in r];flips=[]
    for lang in ('en','zh'):
        for split in ('validation','test'):
            for layer in range(37):
                group=[r for r in compared if r['language']==lang and r['split']==split and r['layer']==layer]
                old_best=min(group,key=lambda r:(r['original_MSE'],r['name']))['name'];new_best=min(group,key=lambda r:(r['all_coordinate_MSE'],r['name']))['name']
                if old_best!=new_best:flips.append({'language':lang,'split':split,'layer':layer,'old_best':old_best,'corrected_best':new_best})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'reports':reports,'conservation':conservation,
        'max_absolute_relative_MSE_change':max(abs(r['relative_MSE_change']) for r in compared),'rank_changes':flips,
        'scope':'Same original all-token/all-coordinate cue baseline with four corrected causal byte-piece labels. Initial positions remain included. No native operator bank or selected anchor changes; confirmation is a re-analysis, not a new untouched dataset. All original reports remain unchanged.'}
    save(out/'corrected_cue_composition.json',result);ledger('corrected_all_coordinate_cue_reanalysis',time.monotonic()-start)
    print('CAUSAL_CUE_REANALYSIS_COMPLETE',result['max_absolute_relative_MSE_change'],len(flips),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
