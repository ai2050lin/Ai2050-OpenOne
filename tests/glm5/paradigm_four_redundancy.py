"""Test why the old write-ranked predictors lose after controlling for known factor cells."""

import numpy as np

import paradigm_four_discovery as D
import paradigm_routes_report as old
import record_update_study as ridge


def rank(x):
    singular=np.linalg.svd(x,compute_uv=False)**2
    share=singular/np.maximum(singular.sum(),1e-12)
    return float(1/np.maximum(np.sum(share**2),1e-12))


def cell_center(train,val):
    c=np.stack([train[k::8].mean(0) for k in range(8)])
    return train-np.tile(c,(len(train)//8,1)), val-np.tile(c,(len(val)//8,1))


def analyze(train_rows,test_rows,train,test,head):
    tx=np.c_[old.text_features(train_rows),train['head_effect'][:,head]]
    vx=np.c_[old.text_features(test_rows),test['head_effect'][:,head]]
    original=ridge.evaluate(ridge.linear(tx,train['target'],12.),vx)
    baseline_mse=np.mean((original-test['target'])**2,axis=1)
    results={}
    for name,sl in [('selected',slice(0,32)),('random',slice(32,64))]:
        features=train['covariates'][:,sl];future=test['covariates'][:,sl]
        train_res,test_res=cell_center(features,future)
        scale=np.maximum(train_res.std(0),1e-8)
        centroids=features-train_res;target_centroids=future-test_res
        metrics={}
        for variant,u,v in [('original',features,future),('only_cell_centroids',centroids,target_centroids),
                            ('remove_training_cell_centroids',train_res,test_res)]:
            fit=ridge.linear(np.c_[tx,u],train['target'],12.)
            predicted=ridge.evaluate(fit,np.c_[vx,v])
            err=np.mean((predicted-test['target'])**2,axis=1)
            metrics[variant]=dict(mse=D.family_delta(err),gain_over_text_and_head=D.family_delta(baseline_mse-err))
        results[name]=dict(metrics=metrics,residual_effective_rank=rank(train_res/scale),
                           raw_scale_min=float(features.std(0).min()),n_scaling_floor=int((features.std(0)<.01).sum()))
    return dict(results=results,baseline=D.family_delta(baseline_mse),
                method='Retrospective old nonblind confirmation. Remove only each fixed cell mean measured on old discovery. Predictions retain same original text and head covariates. No test labels used to fit. Measures target-state MSE, distinct from full-vocabulary KL.')


def main():
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    D.R.OUT=D.OUTPUT
    frozen=D.load(D.OLD/'candidate-freeze.json')
    tokenizer=AutoTokenizer.from_pretrained(D.R.MODEL,local_files_only=True)
    units=frozen['units']+frozen['random_units']
    train_rows,train=D.previous('discovery',tokenizer,units)
    test_rows,test=D.previous('confirm',tokenizer,units)
    report=analyze(train_rows,test_rows,train,test,frozen['head'][1])
    report['at']=D.C.stamp()
    report['parent']=D.C.A.digest(D.OLD/'plan.json')
    D.R.write('redundancy-diagnostic.json',report)
    print('Retrospective factor-centering complete',flush=True)


if __name__=='__main__':
    main()
