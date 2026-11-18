"""Frozen estimators and comparisons. No confirm-derived centers or thresholds."""
import time
import numpy as np
import state_formation_core as C

def states(native,readout=3):
    h=C.unbits(native['hidden'])[:,:,readout];a=C.unbits(native['post_attention'])[:,:,readout]
    value=np.empty((74,8,2560),np.float32);value[0]=h[:,0]
    value[1:73:2]=a.transpose(1,0,2);value[2:73:2]=h[:,1:].transpose(1,0,2)
    value[73]=C.unbits(native['final_norm'])[:,readout];return value
def contrast(h,x):return np.einsum('vf,svd->sfd',x,h,optimize=True)/8
def direction(beta,variance,factor,condition=None):
    center=beta[:,0].copy();d=beta[:,factor].copy()
    if condition:
        other,sign,interaction=condition;center+=sign*beta[:,other];d+=sign*beta[:,interaction]
    w=d/np.maximum(variance,1e-6);den=(w*d).sum(-1)
    w=np.divide(w,den[:,None],out=np.zeros_like(w),where=den[:,None]>1e-12)
    return center,w
def project(h,center,w):return np.einsum('svd,sd->sv',h-center[:,None],w,optimize=True)
def accuracy(scores,labels):return ((scores>=0)==(np.asarray(labels)[None,:]>0)).mean(-1)
def predict(beta,x,columns):return np.einsum('vf,sfd->svd',x[:,columns],beta[:,columns],optimize=True)
def center_predictions(h,beta,variance,norm_coeff,permuted,labels):
    """Returns all eight condition decisions, in two disjoint cross-color folds.

    The decoded color chooses the opposite-color language direction for both
    fixed and corrected models. True color is used only by the oracle comparator.
    """
    base,wcolor=direction(beta,variance,6);cscore=project(h,base,wcolor);predc=np.where(cscore>=0,1.,-1.)
    names=['fixed','oracle_center','decoded_center','pooled','norm','wrong_center','permuted_labels'];result={k:np.zeros(h.shape[:2]) for k in names}
    result['pooled']=project(h,*direction(beta,variance,3))
    magnitude=np.linalg.norm(h,axis=-1);result['norm']=(magnitude-norm_coeff[:,0,None])*norm_coeff[:,3,None]
    pw=permuted/np.maximum(variance,1e-6);result['permuted_labels']=project(h,beta[:,0],pw)
    for source_color in [-1,1]:
        center,w=direction(beta,variance,3,(6,source_color,7));score=project(h,center,w)
        c_to_score=np.sum(beta[:,6]*w,-1)[:,None]
        keep=predc==-source_color
        result['fixed']=np.where(keep,score,result['fixed'])
        result['oracle_center']=np.where(keep,score-(np.asarray(labels)[None,:]-source_color)*c_to_score,result['oracle_center'])
        result['decoded_center']=np.where(keep,score-(predc-source_color)*c_to_score,result['decoded_center'])
        result['wrong_center']=np.where(keep,score+(predc-source_color)*c_to_score,result['wrong_center'])
    return result

def load_family(split,f):
    path=C.OUT/f'{split}-{f:04d}.npz';meta=C.read(path.with_suffix('.json'));assert C.sha(path)==meta['native_sha256']
    return np.load(path)

def select():
    p=C.checked();assert C.read(C.OUT/'discovery-behavior.json')['passed'];assert not list(C.OUT.glob('confirm-*.npz'))
    rows=C.read(C.OUT/'discovery-materials.json')['rows'];x=C.design(rows[:8]);start=time.monotonic();shape=(74,8,2560)
    total=np.zeros(shape);squares=np.zeros((74,2560));norms=np.zeros((74,8));noise=np.zeros((74,2560));train=None
    for f in range(256):
        with load_family('discovery',f) as z:h=states(z)
        assert np.array_equal(C.design(rows[f*8:f*8+8]),x)
        total+=contrast(h,x);squares+=(h.astype(float)**2).mean(1);norms+=np.einsum('vf,sv->sf',x,np.linalg.norm(h,axis=-1))/8
        rng=np.random.default_rng(C.SEED+f);shuffled=rng.permutation(x[:,3]);noise+=np.einsum('v,svd->sd',shuffled,h)/8
        if f==127:train=(total/128,np.maximum(squares/128-(total[:,0]/128)**2,0))
        if (f+1)%32==0:C.log('fit_discovery',f+1,256,start)
    beta=total/256;variance=np.maximum(squares/256-beta[:,0]**2,0);norms/=256;noise/=256
    first,firstvar=train;records=[]
    for factor,field in [('language',3),('color',6)]:
        center,w=direction(first,firstvar,field)
        for module in ['attention','mlp']:
            for block in p['formation'][factor+'_blocks']:
                before=2*block+(module=='mlp');after=before+1
                records.append({'factor':factor,'module':module,'block':block,'before_state':before,'after_state':after,'post_accuracy':[],'pre_accuracy':[],'signed_increment':[]})
    for f in range(128,256):
        with load_family('discovery',f) as z:h=states(z)
        for row in records:
            field=3 if row['factor']=='language' else 6
            post=row['after_state'];pre=row['before_state'];d=first[post,field];w=d/np.maximum(firstvar[post],1e-6);w/=max(float(d@w),1e-12)
            a=(h[post]-first[post,0])@w;b=(h[pre]-first[post,0])@w;labels=x[:,field]
            row['post_accuracy'].append(float(((a>=0)==(labels>0)).mean()));row['pre_accuracy'].append(float(((b>=0)==(labels>0)).mean()))
            row['signed_increment'].append(float(np.mean(labels*(a-b))))
        if (f+1)%32==0:C.log('select_validation',f-127,128,start)
    for row in records:
        for key in ['post_accuracy','pre_accuracy','signed_increment']:row[key]=float(np.mean(row[key]))
        row['accuracy_gain']=row['post_accuracy']-row['pre_accuracy']
    selected=[]
    for factor in ['language','color']:
        for module in ['attention','mlp']:
            best=max([r for r in records if r['factor']==factor and r['module']==module],key=lambda r:(r['accuracy_gain'],r['signed_increment'],-r['block']))
            selected.append({**best,'id':f'{factor}-{module}-B{best["block"]}','selection_support':best['post_accuracy']>=.85 and best['accuracy_gain']>=.05})
    digest=C.save_npz('estimator.npz',beta=beta.astype('<f4'),variance=variance.astype('<f4'),norm_coeff=norms,permuted_direction=noise.astype('<f4'))
    C.save('selection.json',{'schema':'state_formation_selection.v1','plan_sha256':C.sha(C.OUT/'plan.json'),'estimator_sha256':digest,
        'candidates':selected,'all_candidates':records,'confirm_unopened':True,'fit_families':256,'selection_fit_families':128,'selection_validation_families':128})
    print({'selected':selected},flush=True)

def bootstrap(values,frames,alpha=.05/8,seed=C.SEED):
    values=np.asarray(values,float);rng=np.random.default_rng(seed);count=4096
    family=np.array([np.mean(values[rng.integers(len(values),size=len(values))]) for _ in range(count)])
    ids=np.unique(frames);means=np.array([values[np.asarray(frames)==frame].mean() for frame in ids])
    frame=np.array([np.mean(means[rng.integers(len(means),size=len(means))]) for _ in range(count)])
    return {'mean':float(values.mean()),'family_interval':np.quantile(family,[alpha/2,1-alpha/2]).tolist(),
        'frame_interval':np.quantile(frame,[alpha/2,1-alpha/2]).tolist(),'frame_means':means.tolist(),'alpha':alpha,'replicates':count}

if __name__=='__main__':select()
