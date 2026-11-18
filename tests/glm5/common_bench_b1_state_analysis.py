"""Retrospective continuous-state / autonomous-rollout diagnostics, CPU only.

Uses exposed discovery and confirm. Does not open or tune on D lockboxes.
All coordinates enter errors; fixed random projection is a baseline input,
never a definition of the native response backbone.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import common_bench_b1_repair as R
import common_bench_b1_stage_c as C


def errors(pred,true,mean,var,sel):
    p=np.asarray(pred[sel],np.float64);y=np.asarray(true[sel],np.float64)
    err=p-y;base=y-mean
    mse=float(np.mean(err**2));den=float(np.mean(base**2))
    scale=np.maximum(var,1e-6)
    std=float(np.mean(err**2/scale));sd=float(np.mean(base**2/scale))
    return {'n':len(y),'MSE':mse,'raw_RMSE':mse**.5,'mean_baseline_MSE':den,'error_ratio_to_mean':mse/den if den>0 else None,
            'standardized_error_ratio':std/sd if sd>0 else None,'raw_explained_vs_mean':1-mse/den if den>0 else None,
            'mean_state_norm':float(np.linalg.norm(y,axis=1).mean()),
            'relative_Frobenius_error':float(np.linalg.norm(err)/np.linalg.norm(y))}


def main():
    R.check_write(R.OUT,512<<20,operation='B1旧场连续状态分析')
    assert not (R.OUT/'state-analysis.json').exists()
    R.save('state-analysis-plan-v2.json',{'code_sha256':R.sha(__file__),
        'amendment':'v1 stopped at hs0: common newline embedding has zero between-item variance; undefined baseline ratios now null. No layer results had been saved.',
        'scope':'retrospective exposed discovery/confirm; no new independent confirmation',
        'starts':[14,27,32],'last_raw_state':35,'methods':['constant','decoded_A2','linear15'],
        'linear15':'fixed seed 15 JL features of standardized current answer state plus intercept; ridge 0.01; 16 output coefficients matches A2 16 columns',
        'inputs':'one initial real answer-position state then predictions only for rollout; teacher-forced comparison uses real source every step',
        'A1_A2':'frozen original estimators, true S/L/colour labels only for descriptive conditional reconstruction, not a deployable text predictor',
        'native_components':'postattn - h[l] and h[l+1] - postattn, blocks0..34; final normalized state excluded from additive residual accounting'},True)
    start=time.monotonic();a=C.labels('discovery');b=C.labels('confirm')
    fit=a['screened']&a['correct']&~a['reserved'];primary=b['screened']&b['correct']
    selections={'all':np.ones(len(b['S']),bool),'primary':primary,'reserved_primary':primary&b['reserved']}
    old=R.read(R.OLD/'stagec-frozen.json');assert old['estimators_sha256']==R.sha(R.OLD/'stagec-estimators.npz')
    z=np.load(R.OLD/'stagec-estimators.npz');coef1=z['coef_A1'];coef2=z['coef_A2'];inc=z['increment_A2'];mi=z['mean_increment']
    X1=C.design(b['S'],b['L'],b['C'],False);X2=C.design(b['S'],b['L'],b['C'],True)
    projection=R.rng_for('linear15','JL').normal(size=(2560,15)).astype(np.float32)/np.sqrt(15)
    np.save(R.OUT/'state-linear15-projection.npy',projection)
    reconstruction=[];components=[];rollouts={};teachers=[];native=[]
    post=np.load(C.store('confirm')/'postattn.npy',mmap_mode='r')
    centres=np.load(R.OLD/'staged-centres.npz')['32__pooled'];directions=centres-centres.mean(0)
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    dirs=directions[b['C']]
    models={};h=C.layer('discovery',0);y=C.layer('confirm',0)
    R.log('state-analysis',0,37,start)
    for l in range(37):
        if l:
            h=C.layer('discovery',l);y=C.layer('confirm',l)
        mean=h[fit].mean(0);var=h[fit].var(0)
        row={'state':l,'A1':{},'A2':{}}
        p1=X1@coef1[l];p2=X2@coef2[l]
        for key,sel in selections.items():
            row['A1'][key]=errors(p1,y,mean,var,sel);row['A2'][key]=errors(p2,y,mean,var,sel)
        reconstruction.append(row)
        native.append(np.mean((y[primary]-p1[primary])**2,axis=0))
        if l>=35: R.log('state-analysis',l+1,37,start);continue
        hn=C.layer('discovery',l+1);yn=C.layer('confirm',l+1)
        scale=np.sqrt(np.maximum(var,1e-6));features=(h[fit]-mean)/scale@projection
        X=np.column_stack([np.ones(fit.sum()),features]);target=hn[fit]-h[fit]
        penalty=np.eye(16)*.01;penalty[0,0]=0
        W=np.linalg.solve(X.T@X+penalty,X.T@target).astype(np.float32)
        models[f'W_{l}']=W;models[f'mean_{l}']=mean;models[f'scale_{l}']=scale
        predlin=lambda v:v+np.column_stack([np.ones(len(v)),(v-mean)/scale@projection])@W
        cellcent=C.CELL_X[True]@coef2[l]
        predcell=lambda v:v+C.CELL_X[True][C.nearest(v,cellcent,np.maximum(z['residual_variance'][l],1e-6))]@inc[l]
        if l in (14,27,32):
            rollouts[l]={name:y.copy() for name in ['constant','decoded_A2','linear15']}
        step={'source_state':l,'methods':{},'rollouts':{}}
        denominator=float(np.mean((yn-(y+mi[l]))**2))
        for name,pred in [('constant',y+mi[l]),('decoded_A2',predcell(y)),('linear15',predlin(y))]:
            step['methods'][name]={'increment_MSE':float(np.mean((yn-pred)**2)),
                                   'increment_error_ratio_to_constant':float(np.mean((yn-pred)**2))/max(denominator,1e-15)}
        for s,states in rollouts.items():
            states['constant']+=mi[l];states['decoded_A2']=predcell(states['decoded_A2']);states['linear15']=predlin(states['linear15'])
            step['rollouts'][str(s)]={name:errors(v,yn,hn[fit].mean(0),hn[fit].var(0),primary) for name,v in states.items()}
        teachers.append(step)
        pa=C.bits_to_f32(np.ascontiguousarray(post[:,l]));att=pa-y;mlp=yn-pa
        comp={'block':l,'addition_max_abs_error':float(np.max(np.abs((att+mlp)-(yn-y)))),'by_source':{}}
        for s in (1,-1):
            sel=primary&(b['S']==s)
            comp['by_source'][str(s)]={name:{'mean_norm':float(np.linalg.norm(v[sel],axis=1).mean()),
                        'mean_colour_projection':float(np.sum(v[sel]*dirs[sel],axis=1).mean()),
                        'coordinate_mean':v[sel].mean(0).tolist()} for name,v in [('attention',att),('MLP',mlp)]}
        components.append(comp)
        if l%4==0:R.log('state-analysis',l+1,37,start)
    np.savez(R.OUT/'state-linear15-models.npz',**models)
    np.save(R.OUT/'state-native-A1-residual-MSE.npy',np.stack(native))
    result={'scope':'retrospective, all 2560 coordinates; exposed confirm; no new blind confirmation',
        'fit_n':int(fit.sum()),'confirm_all_n':len(primary),'confirm_primary_n':int(primary.sum()),
        'reconstruction':reconstruction,'natural_components':components,'step_and_rollout':teachers,
        'manifest_identity':{s:R.sha(R.OLD/f'stagec-native-{s}.json') for s in ['discovery','confirm']},
        'estimators_sha256':R.sha(R.OLD/'stagec-estimators.npz'),'analysis_sha256':R.sha(__file__),
        'elapsed_seconds':time.monotonic()-start,'caveats':['No arbitrary scalar parameter or complete source routing inferred.',
            'One-token answer state cannot generally determine all future-layer updates without the other token states.',
            'Conditional reconstruction sees true colour labels; rollout predictors do not.',
            'Linear15 is capacity matched in output coefficients, not total preprocessing or semantic supervision.']}
    R.save('state-analysis.json',result,True)
    os.environ['MPLCONFIGDIR']=str(R.OUT/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,1,figsize=(14,7),layout='constrained')
    v=np.stack(native)
    im=ax[0].imshow(np.log10(np.maximum(v,1e-8)),aspect='auto',origin='lower',cmap='magma')
    ax[0].set(xlabel='Native coordinate (fixed order)',ylabel='Hidden state boundary',title='A1 residual MSE on exposed confirm: all coordinates, log10 colour scale')
    fig.colorbar(im,ax=ax[0],label='log10 residual MSE')
    for method in ['A1','A2']:ax[1].plot(range(37),[r[method]['primary']['raw_explained_vs_mean'] for r in reconstruction],label=method)
    ax[1].set(xlabel='Hidden state boundary (36 = final norm)',ylabel='1 - MSE / mean baseline MSE',title='Continuous state reconstruction with oracle factor labels')
    ax[1].legend();fig.savefig(R.OUT/'state-continuous.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for aa,s in zip(ax,[14,27,32]):
        entries=[r for r in teachers if str(s) in r['rollouts']]
        for name in ['constant','decoded_A2','linear15']:
            aa.plot([r['source_state']+1 for r in entries],[r['rollouts'][str(s)][name]['relative_Frobenius_error'] for r in entries],label=name)
        aa.set(title=f'Autonomous from hs[{s}]',xlabel='Predicted state',ylabel='Relative whole-state error');aa.legend()
    fig.savefig(R.OUT/'state-rollout.png',dpi=150);plt.close(fig)
    print('state analysis complete',flush=True)

if __name__=='__main__':main()
