"""Shared/conditional/equal-capacity readouts plus all-coordinate conditional interactions."""
import gc
from rdc_conditional_common import *
from rdc_conditional_estimators import FullKernel,group_ids,binary_report
from phase2699_rdc_confirmation_analysis import reader
OUT=CAMPAIGN/'i_factorial'


def extract(rows):
    path=OUT/'features/state.npz'
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    n=len(rows);h=np.empty((n,37,2560),np.uint16);uvc=np.empty((n,3,3,2560),np.float32)
    moments={};token_counts={}
    for i,r in enumerate(rows):
        with np.load(OUT/f'fields/{r["sample_id"]}.npz') as z:
            h[i]=z['h_c'];uvc[i]=z['roles'][[12,24,36]]
        key=r['family']+'_'+r['language']
        with np.load(OUT/f'moments/{r["sample_id"]}.npz') as z:
            if key not in moments:moments[key]=[np.zeros((37,2560)),np.zeros((37,2560))];token_counts[key]=0
            moments[key][0]+=z['sum'];moments[key][1]+=z['sumsq'];token_counts[key]+=int(z['tokens'])
        if i%256==0:print('EXTRACT',i,len(rows),flush=True)
    npz(path,h_c=h,uvc=uvc)
    keys=sorted(moments);s=np.stack([moments[k][0] for k in keys]);ss=np.stack([moments[k][1] for k in keys]);count=np.array([token_counts[k] for k in keys])
    mean=s/count[:,None,None];std=np.sqrt(np.maximum(0,ss/count[:,None,None]-mean**2))
    npz(OUT/'features/all_token_moments.npz',sum=s,sumsq=ss,tokens=count,mean=mean,std=std)
    save(OUT/'features/all_token_moments.json',{'groups':keys,'normalization':'Token-weighted raw sums, squares, populationstd. Native indexorder, no sorting.',
      'source_commit_count':len(rows),'sources':{r['sample_id']:sha(OUT/f'moments/{r["sample_id"]}.npz') for r in rows}})
    return {'h_c':h,'uvc':uvc}


def main():
    rows=read(OUT/'material.json');assert len(list((OUT/'commits').glob('*.json')))==4096
    immutable(OUT/'analysis_protocol.json',{'source_sha':sha(Path(__file__)),'estimator_sha':sha(ROOT/'tests/glm5/rdc_conditional_estimators.py'),
      'labels':['positive_support','requested_answer'],'primary_split':'entitygroups0..7 train2048;8..11 validation1024;12..15 test1024',
      'extra_joint_split':'Exclude form1/style1/zh cell from train and validation; test only that cell on unseen entitygroups12..15. All truth/query cases retained. No result-driven cell choice.',
      'models':'Old frozen B linear readers H12/24/36 C/UVC; affine calibration only old scores; new linear C H0..36; H24 shared+familylanguage16, independent16, shared+hash16, shared+layoutquery16, quadratic. Same inputs/no answer or support labels as predictors.',
      'capacity':'Shared+condition and shared+hash/layout have identical feature construction and16group count; independent16 has no shared component, so not identical nominal parametercount. Effective degreesoffreedom reported. Form/style/lang/query are available prompt metadata, not latent native gates.',
      'interaction':'Within each family/entity and otherfactors held fixed, support/query contrast and their2x2 interaction measured on all C coordinates/all37layers. Descriptive contrasts, not donor transport or causal proof.'})
    features=extract(rows);hc=features['h_c'];uvc=features['uvc'];tr,va,te=splits(rows)
    y=np.array([[r['fact_truth'],r['expected_yes']] for r in rows],np.float32);results=[]
    oldroot=RESULT/'rdc_mechanism_campaign_20260909/b_relations/models'
    for j,l in enumerate((12,24,36)):
      for mode in ('C','UVC'):
        x=uvc[:,j,2] if mode=='C' else uvc[:,j].reshape(len(rows),-1);pred=[]
        for target in ('positive_support','requested_answer'):
            suffix='C_only' if mode=='C' else 'A1_linear'
            w,b=reader(oldroot/f'H{l}__{target}__{suffix}.npz');score=x@w+b;pred.append((1+score[:,1]-score[:,0])/2)
        p=np.stack(pred,1);mid=f'old_H{l}_{mode}'
        results.append({'model':mid,'scope':'all4096 prospective',**binary_report(y,p,np.arange(len(rows)),rows)})
        npz(OUT/f'predictions/{mid}.npz',prediction=p,test=np.arange(len(rows)))
        # Four parameters total, separate affine calibration for each old binary score.
        calibrated=np.empty_like(p);params=[]
        for k in range(2):
            a=np.column_stack([np.ones(len(tr)),p[tr,k]]);coef=np.linalg.lstsq(a,y[tr,k],rcond=None)[0]
            calibrated[:,k]=coef[0]+coef[1]*p[:,k];params.append(coef)
        results.append({'model':mid+'_affine','scope':'entity_heldout','parameters':np.asarray(params).tolist(),**binary_report(y,calibrated[te],te,rows)})
    for l in range(37):
        x=unbits(hc[:,l]);f=FullKernel(x,tr,va,te);p,m=f.fit(y,OUT/f'models/H{l}_shared.npz')
        results.append({'model':f'H{l}_shared','scope':'entity_heldout',**m,**binary_report(y,p,te,rows)})
        npz(OUT/f'predictions/H{l}_shared.npz',prediction=p,test=te)
        print('SHARED',l,m['mse'],flush=True);del f,x;gc.collect()
    configs=[('shared','linear',None),('quadratic','quadratic',None),('family_shared','conditional','family_language'),('family_independent','independent','family_language'),('hash_shared','conditional','hash16'),('layout_shared','conditional','layout_query')]
    joint=np.array([r['form']==1 and r['style']==1 and r['language']=='zh' for r in rows])
    for scope,train,val,test in [('entity_heldout',tr,va,te),('joint_condition_entity',tr[~joint[tr]],va[~joint[va]],te[joint[te]])]:
      for label,kind,group in configs:
        if scope=='entity_heldout' and label=='shared':continue
        x=unbits(hc[:,24]);g=group_ids(rows,group) if group else None
        f=FullKernel(x,train,val,test,kind,g);p,m=f.fit(y,OUT/f'models/H24_{scope}_{label}.npz')
        results.append({'model':'H24_'+label,'scope':scope,**m,**binary_report(y,p,test,rows)})
        npz(OUT/f'predictions/H24_{scope}_{label}.npz',prediction=p,test=test)
        print('CONDITIONAL',scope,label,m['mse'],flush=True);del f,x;gc.collect()
    # Rows are rectangular in fixed t,q,form,style,lang order: contrasts across all native C coordinates.
    tensor=unbits(hc).reshape(8,16,2,2,2,2,2,37,2560).astype(np.float32)
    dt=tensor[:,:,1]-tensor[:,:,0];interaction=dt[:,:,1]-dt[:,:,0]
    # Aggregate only after squares to avoid cancellation of opposite patterns.
    npz(OUT/'features/interactions.npz',support_rms=np.sqrt(np.mean(dt.astype(np.float64)**2,axis=(1,2,3,4,5))),
        support_query_interaction_rms=np.sqrt(np.mean(interaction.astype(np.float64)**2,axis=(1,2,3,4))),
        coordinate_order=np.arange(2560),layers=np.arange(37))
    b=[read(OUT/f'behavior/{r["sample_id"]}.json') for r in rows]
    behavior={f:{'n':sum(r['family']==f for r in rows),'first_correct':sum(bb['first_token_correct'] for r,bb in zip(rows,b) if r['family']==f),'pair_correct':sum(bb['pair_correct'] for r,bb in zip(rows,b) if r['family']==f)} for f in sorted({r['family'] for r in rows})}
    save(OUT/'result.json',{'phase':2703,'timestamp':stamp(),'case_count':4096,'results':results,'behavior':behavior,'limits':read(OUT/'protocol.json')['limits']})
    announce('i_factorial',state='analysis_complete',completed=4096,total=4096)


if __name__=='__main__':main()
