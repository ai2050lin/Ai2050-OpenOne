"""Complete-coordinate dual ridge; validation predictions precede route freezing."""
from collections import defaultdict
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_native_tail import cuda_singleton,CUDA_TASKS
from rdc_history_prediction import weights
from phase2746_rdc_history_prediction_contract import OUT,freeze


def permutation(rows,w):
    buckets=defaultdict(list)
    for index,r in enumerate(rows):buckets[(r['family'],r['language'],r['step'],float(w[index]))].append(index)
    p=np.arange(len(rows))
    for key,ix in sorted(buckets.items()):
        p[ix]=np.random.default_rng(int(rank('history_target_shuffle/'+str(key))[:16],16)).permutation(ix)
    assert np.array_equal(w,w[p])
    return p


def main():
    import torch
    cuda_singleton(CUDA_TASKS|{'phase2746_rdc_history_features.py',Path(__file__).name})
    verify_storage(2*1024**3);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();protocol,rows=freeze();out=OUT/'fit';feature=read(OUT/'features/result.json')
    path=BASE/feature['field_path'];assert sha(path)==feature['field_sha256']
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);validation=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    trainrows=[rows[i] for i in train];w=weights(trainrows);perm=permutation(trainrows,w)
    with np.load(path) as z:
        common=[unbits(z[k]).astype(float) for k in ['H0','H12']]
        targets=np.concatenate([unbits(z['target_H35'][train]).astype(float),unbits(z['target_H36'][train]).astype(float),
            unbits(z['target_Q35'][train]).reshape(len(train),-1).astype(float)],axis=-1)
        candidates={name:z[key].astype(float) for name,key in [('native_history','candidate_native'),('source_value_permuted','candidate_source_value_shuffled')]}
    sw=torch.tensor(np.sqrt(w),device='cuda',dtype=torch.float64)
    Y=torch.tensor(targets,device='cuda',dtype=torch.float64);reports=[]
    for name,candidate in candidates.items():
        receipt=out/(name+'.json')
        if receipt.exists():
            saved=read(receipt);assert sha(BASE/saved['field_path'])==saved['field_sha256'];reports.append(saved);continue
        data=np.concatenate([*common,candidate],axis=1);width=data.shape[1]
        x=torch.tensor(data[train],device='cuda',dtype=torch.float64)
        mu=(x*sw.square()[:,None]).sum(0);scale=((x-mu).square()*sw.square()[:,None]).sum(0).sqrt().clamp_min(1e-8)
        normalized=(x-mu)/scale;weighted=normalized*sw[:,None]
        covariance=weighted@weighted.T/width
        eigenvalues,eigenvectors=torch.linalg.eigh(covariance)
        assert float(eigenvalues.min())>-1e-10
        eigenvalues=eigenvalues.clamp_min(0)
        residual=float(((eigenvectors*eigenvalues[None])@eigenvectors.T-covariance).norm()/covariance.norm())
        assert residual<1e-10
        val=(torch.tensor(data[validation],device='cuda',dtype=torch.float64)-mu)/scale
        cross=val@weighted.T/width;cross_spectral=cross@eigenvectors
        projections=[];means=[];predictions=[]
        for control in ['true_correspondence','target_correspondence_shuffled']:
            target=Y if control=='true_correspondence' else Y[torch.tensor(perm,device='cuda')]
            mean=(target*sw.square()[:,None]).sum(0)
            projected=eigenvectors.T@((target-mean)*sw[:,None])
            predictions.append(np.stack([(cross_spectral@(projected/(eigenvalues+lam)[:,None])+mean).float().cpu().numpy() for lam in protocol['lambdas']]))
            projections.append(projected.cpu().numpy());means.append(mean.cpu().numpy())
        archive=FIELD_STORE/'history_fit'/(name+'.npz')
        arrays={'train_Z':normalized.cpu().numpy(),'sqrt_train_weights':sw.cpu().numpy(),
            'feature_mean':mu.cpu().numpy(),'feature_scale':scale.cpu().numpy(),
            'dual_eigenvalues':eigenvalues.cpu().numpy(),'dual_eigenvectors':eigenvectors.cpu().numpy(),
            'projected_targets':np.stack(projections),'target_means':np.stack(means),
            'validation_predictions':np.stack(predictions),'training_row_indices':train,'validation_row_indices':validation,
            'training_target_permutation':perm,'lambdas':np.array(protocol['lambdas'])}
        verify_storage(sum(a.nbytes for a in arrays.values()));npz(archive,**arrays)
        rec={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'input_variant':name,
            'field_path':archive.relative_to(BASE).as_posix(),'field_sha256':sha(archive),'field_bytes':archive.stat().st_size,
            'feature_source_sha256':feature['field_sha256'],'train_points':len(train),'validation_points':len(validation),
            'feature_dimension':width,'target_dimension':targets.shape[-1],
            'target_slices':{'H35':[0,2560],'H36':[2560,5120],'Q35':[5120,9216]},
            'controls':['true_correspondence','target_correspondence_shuffled'],
            'validation_row_ids':[rows[i]['point_id'] for i in validation],
            'ridge_effective_degrees_of_freedom':[float((eigenvalues/(eigenvalues+lam)).sum()) for lam in protocol['lambdas']],
            'dual_eigensystem_relative_reconstruction_error':residual,
            'permutation_keeps_exact_train_weights':bool(np.array_equal(w,w[perm])),
            'test_targets_used_for_fit_or_selection':False,'test_predictions_computed':False,
            'scope':'Complete dual spectrum used only to solve weighted ridge; no coordinate/component truncation. Shared df does not imply all decoder output dimensions or native constraints have identical parameter counts.'}
        save(receipt,rec);reports.append(rec)
        print('HISTORY_FIT',name,round(time.monotonic()-start,1),flush=True)
        del arrays,data,x,mu,scale,normalized,weighted,covariance,eigenvalues,eigenvectors,val,cross,cross_spectral
        gc.collect();torch.cuda.empty_cache()
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'reports':reports,
        'training_points':len(train),'validation_points':len(validation),'test_points':sum(r['split']=='test' for r in rows),
        'seconds':time.monotonic()-start,'scope':'Fitted external maps; validation compilation/selection, test prediction, independent confirmation and self-fed deployment remain unfinished.'}
    save(out/'result.json',result);ledger('phase2746_history_fit',result['seconds'])
    print('HISTORY_FIT_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    start=time.monotonic()
    try:main()
    except Exception as exc:
        failure(OUT/'fit',start,exc);raise
